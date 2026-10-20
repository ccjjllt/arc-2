#!/usr/bin/env python3
"""019-P1: 27B全量可提交候选端到端打通（对应019计划§4）。

链路：真实challenge输入 → 保留全部train demos + 统计全部test queries(Q)
 → 立即落盘全量[[0]]占位submission.json
 → 阶段1（广覆盖保底）：C8滚动第一遍预算受控直出生成，合法候选入池并持续原子写盘
 → 阶段2（选择性程序+一次反馈补充）：剩余共享总墙钟内，对尚未demo全过或
    双槽相异的题目运行短程序首稿与最多一次demo执行反馈修复
 → 全局截止前收尾：停止新请求、排空在途生成、最终校验与原子写盘、安全关闭服务。

单一总时钟：session_deadline = t0_mono + session_seconds；安装/加载/阶段1/
阶段2/CPU执行/写盘收尾全部共享。T-600停新请求，T-300硬收尾原子写盘。
候选池持续写盘：在chat_wave的on_slot_done回调中每完成一条即更新池并刷新
/kaggle/working/submission.json。绝不使用query真值选槽（见submit_lib）。

固定底座：Qwen/Qwen3.8-27B FP8 + 固定vLLM wheel + TP4 + 服务容量8 + C8滚动。
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import signal
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import run_reasoning_probe as P
import solve_program as SP
import submit_lib as SL

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
SAVE_MARGIN_S = 300
NEW_REQUEST_MARGIN_S = 600

# 阶段1直出（广覆盖保底，预算受控）
GEN_MAX_DIRECT = 4096
THINK_DIRECT = 2048
EFFORT_DIRECT = "low"
SEED_DIRECT = 7101
# 阶段2程序首稿与一次反馈
GEN_MAX_PROG = 12288
THINK_PROG = 8192
EFFORT_PROG = "xhigh"
SEED_P0 = 5101
SEED_F = 5103
EXEC_PER_GRID_S = 2.0
EXEC_PER_PROGRAM_S = 20.0
FEEDBACK_MAX_CHARS = 3000

STATE = {"proc": None, "work": None, "run": None, "closed": False, "phase": "init"}
GPU_SAMPLER_STOP = threading.Event()
POOL_LOCK = threading.Lock()


def log(msg):
    line = f"[019submit {time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    try:
        if STATE.get("work") is not None:
            with open(STATE["work"] / "driver-stdout.log", "a") as h:
                h.write(line + "\n")
    except Exception:
        pass


def sh(cmd, timeout, cwd=None, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)


def sha256_file(path: Path) -> str:
    d = hashlib.sha256()
    with path.open("rb") as h:
        for ch in iter(lambda: h.read(1 << 20), b""):
            d.update(ch)
    return d.hexdigest()


def save(work: Path, run: dict) -> None:
    (work / "run.json").write_text(json.dumps(run, indent=1))


def orderly(work, run, status, code=0):
    run["status"] = status
    started = run.pop("_t0", None)
    run["elapsed_s"] = round(time.time() - (started if started is not None else time.time()), 1)
    import shutil
    shutil.rmtree(work / "site", ignore_errors=True)
    GPU_SAMPLER_STOP.set()
    save(work, run)
    log(f"EVIDENCE-EXIT status={status}")
    return code


def finish(status, code=0, detail=None):
    if STATE["closed"]:
        return code
    STATE["closed"] = True
    run, work = STATE["run"], STATE["work"]
    if detail and run is not None and "first_error" not in run:
        run["first_error"] = str(detail)[:800]
    proc = STATE.get("proc")
    if proc is not None and work is not None:
        P.kill_server(proc, work)
        STATE["proc"] = None
    if run is None or work is None:
        GPU_SAMPLER_STOP.set()
        return code
    return orderly(work, run, status, code)


def on_term(signum, _frame):
    finish(f"signal_{signum}")
    os._exit(0)


def server_alive() -> bool:
    proc = STATE.get("proc")
    return proc is not None and proc.poll() is None


def time_left(session_deadline: float, margin: float = SAVE_MARGIN_S) -> float:
    return session_deadline - time.monotonic() - margin


def compact(grid) -> str:
    return json.dumps(grid, separators=(",", ":"), ensure_ascii=False)


def build_direct_prompt(task: dict) -> list:
    lines = ["Predict the output grid. Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={compact(d['input'])} out={compact(d['output'])}")
    lines.append(f"test in={compact(task['test'][0]['input'])}")
    lines.append("Reason briefly, then end with exactly one closed ```grid block "
                 "containing only space-separated 0-9 rows, one row per line, "
                 "with no text after the block.")
    return [{"role": "system", "content": "You predict ARC grids exactly."},
            {"role": "user", "content": "\n".join(lines)}]


def build_program_prompt(task: dict) -> list:
    lines = ["Solve the grid transformation. Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={compact(d['input'])} out={compact(d['output'])}")
    lines.append(f"test in={compact(task['test'][0]['input'])}")
    lines.append("The FIRST line of your final response must be exactly ```python, followed by one complete "
                 "transform(grid) program closed by ``` on its own line, with no analysis text before or after. "
                 "Do not use ** or pow(); write integers literally. Keep the program compact (<=150 lines).")
    return [{"role": "system", "content": SP.SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(lines)}]


def is_legal_grid(grid) -> bool:
    try:
        SP.checked_grid(grid)
        return True
    except Exception:
        return False


def parse_direct(rec: dict, res: dict):
    if res.get("finish") == "stop" and not rec.get("client_error"):
        grid = P.extract_final_grid(res.get("content") or "", res.get("finish"))
        if grid is not None:
            try:
                return SL.valid_grid(grid), True
            except (ValueError, TypeError):
                return None, False
    return None, False


def parse_program(rec: dict, res: dict) -> dict:
    if res.get("finish") == "stop" and not rec.get("client_error") \
            and not rec.get("cancelled_by_client"):
        parsed = SP.split_program_response(res.get("content") or "", thinking_mode="direct",
                                           generation=None)
        parsed["gen_ok"] = parsed.get("status") == "program_ready"
        return parsed
    reason = rec.get("client_error") or f"finish={res.get('finish')}"
    return {"status": "generation_not_ok", "reason": str(reason)[:200],
            "reasoning_content": res.get("reasoning") or "", "final": res.get("content") or "",
            "code": None, "gen_ok": False}


def exec_program(code, parse_status, demos, test_input, task_deadline):
    demo_results: list = []
    spent = 0.0
    if code is None:
        for _ in demos:
            demo_results.append({"status": parse_status, "error": "no admitted program",
                                 "elapsed_s": 0.0})
        return demo_results, {"status": parse_status, "error": "no admitted program"}, 0.0
    for demo in demos:
        remaining = min(EXEC_PER_PROGRAM_S - spent, task_deadline - time.monotonic(),
                        EXEC_PER_GRID_S)
        if remaining <= 0:
            demo_results.append({"status": "budget_interrupted", "elapsed_s": 0.0})
            continue
        r = SP.execute_grid(code, demo["input"],
                              wall_seconds=min(EXEC_PER_GRID_S, remaining),
                              memory_mb=512)
        spent += r.get("elapsed_s", 0.0)
        demo_results.append(r)
    remaining = min(EXEC_PER_PROGRAM_S - spent, task_deadline - time.monotonic(), EXEC_PER_GRID_S)
    if remaining <= 0:
        query_result = {"status": "budget_interrupted", "elapsed_s": 0.0}
    else:
        query_result = SP.execute_grid(code, test_input, wall_seconds=remaining, memory_mb=512)
        spent += query_result.get("elapsed_s", 0.0)
    return demo_results, query_result, round(spent, 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=43200.0)
    ap.add_argument("--submission", default="/kaggle/working/submission.json")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()
    t0_mono = time.monotonic()
    session_deadline = t0_mono + args.session_seconds
    run = {"config": {"tp": P.TP_SIZE, "max_model_len": P.MAX_MODEL_LEN,
                       "service_max_num_seqs": P.SERVICE_MAX_NUM_SEQS,
                       "gpu_mem_util": P.GPU_MEM_UTIL, "kv_dtype": "default", "mtp": False,
                       "sampling": dict(P.SAMPLING),
                       "stage1": {"effort": EFFORT_DIRECT, "gen_max": GEN_MAX_DIRECT,
                                  "think": THINK_DIRECT, "seed": SEED_DIRECT},
                       "stage2": {"effort": EFFORT_PROG, "gen_max": GEN_MAX_PROG,
                                  "think": THINK_PROG, "seed_p0": SEED_P0, "seed_f": SEED_F},
                       "vllm_wheel": VLLM_WHEEL_NAME, "vllm_wheel_sha256": VLLM_WHEEL_SHA256},
            "budgets": {"session_s": args.session_seconds,
                        "save_margin_s": SAVE_MARGIN_S,
                        "new_request_margin_s": NEW_REQUEST_MARGIN_S},
            "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    try:
        lock_fd = os.open(work / "driver.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        print("[019submit] lock busy, another driver holds it; exiting quietly", flush=True)
        return 0
    log(f"driver start session_s={args.session_seconds}")
    P.record_versions(run)
    log("versions recorded")
    STATE["phase"] = "challenges"

    challenges = json.loads(Path(args.challenges).read_text())
    run["challenges_sha"] = sha256_file(Path(args.challenges))
    submission, query_map = SL.initial_submission(challenges)
    run["total_tasks"] = len(submission)
    run["total_queries"] = SL.count_queries(challenges)
    SL.atomic_json(args.submission, submission)
    run["placeholder_written"] = True
    save(work, run)
    log(f"challenges ok tasks={len(submission)} Q={run['total_queries']}; placeholder written")

    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        return finish(f"env_incompatible_python_{py.major}_{py.minor}")
    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        run["mount_error"] = str(exc)[:200]
        return finish("mount_missing")
    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels = wh / "wheels"
    req = wh / "requirements-qwen38-runtime.txt"
    wheel_path = wheels / VLLM_WHEEL_NAME
    if not wheels.is_dir() or not req.is_file() or not wheel_path.is_file():
        return finish("vllm_wheel_missing")
    STATE["phase"] = "wheelcheck"
    digest = sha256_file(wheel_path)
    run["wheel_hash"] = {"status": "match" if digest == VLLM_WHEEL_SHA256 else "mismatch",
                         "observed": digest[:16]}
    save(work, run)
    if digest != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch")
    log("wheel hash match (sync)")

    pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    if pip_probe.returncode:
        ens = sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
        (work / "ensurepip.log").write_text((ens.stdout + "\n---STDERR---\n" + ens.stderr)[-4000:])
        pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    use_fallback = pip_probe.returncode != 0
    if use_fallback and not args.pip_python:
        return finish("no_pip_for_312_and_no_fallback")

    def _base():
        if use_fallback:
            return [args.pip_python, "-m", "pip", "install", "--python-version", "3.12",
                    "--only-binary", ":all:", "--no-index"]
        return [sys.executable, "-m", "pip", "install", "--no-index"]
    STATE["phase"] = "pip-req"
    try:
        log("phase: requirements install start")
        ins = sh(_base() + ["--find-links", str(wheels), "--requirement", str(req),
                            "--target", str(site)], timeout=INSTALL_BUDGET_S)
        log(f"phase: requirements install rc={ins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    run["pip_requirements_rc"] = ins.returncode
    save(work, run)
    if ins.returncode:
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        log("phase: vllm wheel install start")
        vins = sh(_base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path),
                             "--target", str(site)], timeout=600)
        log(f"phase: vllm wheel install rc={vins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    (work / "pip-vllm.log").write_text(vins.stdout[-4000:] + "\n---STDERR---\n" + vins.stderr[-4000:])
    run["pip_wheel_rc"] = vins.returncode
    save(work, run)
    if vins.returncode:
        return finish("vllm_wheel_install_failed")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    STATE["phase"] = "import"
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__)"], timeout=300,
             env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode, "out": (imp.stdout + imp.stderr)[-500:]}
    save(work, run)
    if imp.returncode:
        return finish("import_vllm_failed")
    run["install_s"] = round(time.time() - t0_wall, 1)

    install_deadline = (session_deadline - args.session_seconds) + INSTALL_BUDGET_S
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    STATE["phase"] = "serve"
    try:
        proc = P.start_server(model_root, work, extra=[], deadline=install_deadline)
    except Exception as exc:
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    run["install_load_s"] = round(time.time() - t0_wall, 1)
    save(work, run)
    STATE["phase"] = "preflight"
    pre = SP.execute_grid("def transform(grid):\n    return [row[:] for row in grid]", [[1]],
                          wall_seconds=2.0, memory_mb=512)
    run["executor_preflight"] = {"status": pre.get("status"), "grid": pre.get("grid")}
    save(work, run)
    if pre.get("status") != "ok" or pre.get("grid") != [[1]]:
        return finish("executor_infra_failed")
    th = threading.Thread(target=_gpu_sampler, args=(work,), daemon=True)
    th.start()

    pool: dict = {}
    order = itertools.count()
    task_deadline = session_deadline - SAVE_MARGIN_S

    def pool_add(key, grid, source, all_pass, matches):
        with POOL_LOCK:
            pool.setdefault(key, []).append(
                {"grid": grid, "source": source, "all_demos_pass": all_pass,
                 "demo_matches": matches, "order": next(order)})
            a, b = SL.merge_candidate_pool(pool[key])
            SL.update_slots(submission, query_map, {key: [a, b]})
            SL.atomic_json(args.submission, submission)

    # ---- 阶段1：广覆盖保底直出 ----
    STATE["phase"] = "stage1"
    keys = sorted(query_map.keys())
    if time_left(session_deadline, NEW_REQUEST_MARGIN_S) <= 0 or not server_alive():
        run["stage1"] = {"status": "skipped_cutoff_or_dead"}
    else:
        limit = min(time.monotonic() + 4 * 3600, session_deadline - SAVE_MARGIN_S)
        specs = []
        for key in keys:
            task_id, index = query_map[key]
            clean = SP.task_inputs_only(
                {"train": challenges[task_id]["train"], "test": [challenges[task_id]["test"][index]]})
            msgs = build_direct_prompt(clean)
            specs.append({"messages": msgs, "effort": EFFORT_DIRECT, "seed": SEED_DIRECT,
                          "max_tokens": GEN_MAX_DIRECT,
                          "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                          "thinking_token_budget": THINK_DIRECT, "enable_thinking": True,
                          "request_id": f"S1-{key}", "query": key, "arm": "direct"})
        def _on_done(i, spec, slot):
            grid, ok = parse_direct(slot["rec"], slot["res"])
            slot["grid_ok"] = ok
            if ok:
                pool_add(spec["query"], grid, "direct_grid", False, 0)
        t1 = time.monotonic()
        w = P.chat_wave(specs, client_concurrency=8, on_slot_done=_on_done)
        for spec, slot in zip(specs, w["calls"]):
            rec = dict(slot["rec"])
            rec["phase"] = "S1"
            rec["query"] = spec["query"]
            rec["arm"] = "direct"
            rec["grid_ok"] = slot.get("grid_ok", False)
            run["calls"].append(rec)
        run["stage1"] = {"n": len(specs), "wall_s": round(time.monotonic() - t1, 2),
                         "wave_wall_s": w["wave_wall_s"],
                         "completion_tokens_sum": w["completion_tokens_sum"],
                         "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"]}
        run["coverage_after_s1"] = SL.coverage_report(submission, query_map)
        save(work, run)
        log(f"stage1 done {run['stage1']} coverage={run['coverage_after_s1']}")

    # ---- 阶段2：选择性程序首稿 + 最多一次反馈 ----
    STATE["phase"] = "stage2"
    if time_left(session_deadline, NEW_REQUEST_MARGIN_S) <= 0 or not server_alive():
        run["stage2"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)
    else:
        needy = []
        with POOL_LOCK:
            for key in keys:
                a, b = SL.merge_candidate_pool(pool.get(key, []))
                if a == [[0]] or b == [[0]] or a == b:
                    needy.append(key)
        run["stage2_needy"] = len(needy)
        save(work, run)
        log(f"stage2 needy={len(needy)}/{len(keys)}")
        if needy:
            limit = min(time.monotonic() + 4 * 3600, session_deadline - SAVE_MARGIN_S)
            specs = []
            p0_store: dict = {}
            for key in needy:
                task_id, index = query_map[key]
                clean = SP.task_inputs_only(
                    {"train": challenges[task_id]["train"],
                     "test": [challenges[task_id]["test"][index]]})
                msgs = build_program_prompt(clean)
                specs.append({"messages": msgs, "effort": EFFORT_PROG, "seed": SEED_P0,
                              "max_tokens": GEN_MAX_PROG,
                              "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                              "thinking_token_budget": THINK_PROG, "enable_thinking": True,
                              "request_id": f"P0-{key}", "query": key, "arm": "P0"})
            t1 = time.monotonic()
            w = P.chat_wave(specs, client_concurrency=8)
            for spec, slot in zip(specs, w["calls"]):
                rec, res = slot["rec"], slot["res"]
                rec["phase"] = "S2"
                rec["query"] = spec["query"]
                rec["arm"] = "P0"
                parsed = parse_program(rec, res)
                task_id, index = query_map[spec["query"]]
                demos = challenges[task_id]["train"]
                test_input = challenges[task_id]["test"][index]["input"]
                demo_results, query_result, spent = exec_program(
                    parsed.get("code") if parsed.get("gen_ok") else None,
                    parsed.get("status"), demos, test_input, task_deadline)
                matches = sum(1 for d, r in zip(demos, demo_results)
                              if r.get("status") == "ok" and r.get("grid") == d["output"])
                all_pass = matches == len(demos) and len(demos) > 0
                query_grid = query_result.get("grid") if is_legal_grid(
                    query_result.get("grid")) else None
                if query_grid is not None:
                    pool_add(spec["query"], query_grid, "program_init", all_pass, matches)
                rec.update({"parse_status": parsed.get("status"), "gen_ok": parsed.get("gen_ok"),
                            "demo_matches": matches, "all_demos_pass": all_pass,
                            "query_legal": query_grid is not None, "exec_seconds": spent})
                rec["grid"] = query_grid
                run["calls"].append(rec)
                p0_store[spec["query"]] = {"parsed": parsed, "messages": spec["messages"],
                                          "demo_results": demo_results, "all_pass": all_pass}
                try:
                    (work / "S2").mkdir(exist_ok=True)
                    if parsed.get("code"):
                        (work / "S2" / f"P0-{spec['query']}.code.py").write_text(parsed["code"])
                    (work / "S2" / f"P0-{spec['query']}.final.txt").write_text(
                        res.get("content") or "")
                except Exception:
                    pass
            run["stage2_p0"] = {"n": len(specs), "wall_s": round(time.monotonic() - t1, 2),
                                "wave_wall_s": w["wave_wall_s"],
                                "completion_tokens_sum": w["completion_tokens_sum"],
                                "usage_missing": w["usage_missing"],
                                "agg_tok_s": w["agg_tok_s"]}
            save(work, run)
            # 一次反馈：仅对P0完整程序
            fixable = [k for k, v in p0_store.items()
                       if v["parsed"].get("gen_ok")
                       and time_left(session_deadline, NEW_REQUEST_MARGIN_S) > 0]
            if fixable and server_alive():
                limit = min(time.monotonic() + 2 * 3600, session_deadline - SAVE_MARGIN_S)
                specs = []
                for key in fixable:
                    task_id, _ = query_map[key]
                    demos = challenges[task_id]["train"]
                    p0 = p0_store[key]
                    try:
                        feedback = SP.demo_feedback(demos, p0["demo_results"],
                                                    FEEDBACK_MAX_CHARS)
                    except Exception as exc:
                        feedback = json.dumps(
                            {"feedback_error": f"{type(exc).__name__}:{exc}"[:200]})
                    f_messages = p0["messages"] + [{
                        "role": "user",
                        "content": ("Execution feedback (JSON, real demo results):\n"
                                    f"{feedback}\nWrite the corrected complete "
                                    "transform(grid) as exactly one closed ```python block.")}]
                    specs.append({"messages": f_messages, "effort": EFFORT_PROG,
                                  "seed": SEED_F, "max_tokens": GEN_MAX_PROG,
                                  "timeout_s": max(1.0, limit - time.monotonic()),
                                  "deadline": limit,
                                  "thinking_token_budget": THINK_PROG, "enable_thinking": True,
                                  "request_id": f"F-{key}", "query": key, "arm": "F"})
                t1 = time.monotonic()
                w = P.chat_wave(specs, client_concurrency=8)
                for spec, slot in zip(specs, w["calls"]):
                    rec, res = slot["rec"], slot["res"]
                    rec["phase"] = "S2"
                    rec["query"] = spec["query"]
                    rec["arm"] = "F"
                    parsed = parse_program(rec, res)
                    task_id, index = query_map[spec["query"]]
                    demos = challenges[task_id]["train"]
                    test_input = challenges[task_id]["test"][index]["input"]
                    demo_results, query_result, spent = exec_program(
                        parsed.get("code") if parsed.get("gen_ok") else None,
                        parsed.get("status"), demos, test_input, task_deadline)
                    matches = sum(1 for d, r in zip(demos, demo_results)
                                  if r.get("status") == "ok" and r.get("grid") == d["output"])
                    all_pass = matches == len(demos) and len(demos) > 0
                    query_grid = query_result.get("grid") if is_legal_grid(
                        query_result.get("grid")) else None
                    if query_grid is not None:
                        pool_add(spec["query"], query_grid, "program_fix", all_pass, matches)
                    rec.update({"parse_status": parsed.get("status"),
                                "gen_ok": parsed.get("gen_ok"),
                                "demo_matches": matches, "all_demos_pass": all_pass,
                                "query_legal": query_grid is not None,
                                "exec_seconds": spent})
                    rec["grid"] = query_grid
                    run["calls"].append(rec)
                    try:
                        if parsed.get("code"):
                            (work / "S2" / f"F-{spec['query']}.code.py").write_text(
                                parsed["code"])
                    except Exception:
                        pass
                run["stage2_fix"] = {"n": len(specs), "wall_s": round(time.monotonic() - t1, 2),
                                     "wave_wall_s": w["wave_wall_s"],
                                     "completion_tokens_sum": w["completion_tokens_sum"],
                                     "usage_missing": w["usage_missing"],
                                     "agg_tok_s": w["agg_tok_s"]}
            else:
                run["stage2_fix"] = {"status": "skipped_no_fixable_or_cutoff"}
            save(work, run)

    # ---- 收尾：最终校验与原子写盘 ----
    STATE["phase"] = "landing"
    with POOL_LOCK:
        SL.atomic_json(args.submission, submission)
    ok, detail = SL.validate_submission(args.submission, submission)
    run["final_validate"] = {"ok": ok, "detail": detail}
    run["coverage_final"] = SL.coverage_report(submission, query_map)
    run["new_output_tokens_total"] = sum(
        ((c.get("usage") or {}).get("completion_tokens") or 0) for c in run.get("calls", []))
    save(work, run)
    log(f"final validate={ok} {detail} coverage={run['coverage_final']}")
    return finish("complete_with_evidence" if ok else "validate_failed")


def _gpu_sampler(work: Path, interval_s: float = 5.0):
    out = work / "gpu-samples.jsonl"
    while not GPU_SAMPLER_STOP.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            line = {"t_wall": round(time.time(), 1), "raw": r.stdout.strip()[:800],
                    "rc": r.returncode}
        except Exception as exc:
            line = {"t_wall": round(time.time(), 1),
                    "error": f"{type(exc).__name__}:{exc}"[:200]}
        try:
            with out.open("a") as h:
                h.write(json.dumps(line) + "\n")
        except Exception:
            pass
        GPU_SAMPLER_STOP.wait(interval_s)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        if STATE.get("run") is not None:
            STATE["run"]["traceback_tail"] = traceback.format_exc()[-1500:]
        raise SystemExit(finish("unexpected_error", 2, f"{type(exc).__name__}:{exc}"))
