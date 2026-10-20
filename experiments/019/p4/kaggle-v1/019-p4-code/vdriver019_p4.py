#!/usr/bin/env python3
"""019-P2: P6其余四题（+4）完整流程验证（对应019计划§5）。

只回答：短而可执行的首稿（P0）＋一次真实执行反馈（F），能否以可接受的
总成本在P6其余四题上重复获得新增正确解。

固定分母=4：4c416de3_0、58c02a16_0、aa4ec2a5_0、fc4aaf52_0
（题面来自p6-challenges.json；评分仅在取回后于本地对照
experiments/010/kaggle-e0-v1/010/scoring.json进行，绝不在GPU上用真值）。
不重跑291dc1e1_0与db118e2a_0。

流程（单次模型加载批量执行）：
- Step 1：批量P0（4次调用，C8并发，seed 5101，12288/8192，final禁散文强契约）。
  按题独立进反馈：P0完整→真实demo执行→demo_feedback（≤3000字符，零人工干预）；
  P0不完整→该题反馈机制记首稿失败，仍保留在分母中。
- Step 2：P0完整题并发 F（题面+P0完整代码+真实反馈，seed 5103）与
  R（仅原始题面，seed 5103）；P0不完整题仅发R。双槽 A_F=[P0,F] / A_R=[P0,R]。
- Step 3：严格保留P0候选保底（F改坏时P0仍在槽中）。

固定底座：Qwen/Qwen3.8-27B FP8 + 固定vLLM wheel + TP4 + 服务容量8 + C8滚动。
"""
from __future__ import annotations

import argparse
import hashlib
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

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
SAVE_MARGIN_S = 300
NEW_REQUEST_CUTOFF_S = 80 * 60

P6_EXTRA4 = ["4c416de3_0", "58c02a16_0", "aa4ec2a5_0", "fc4aaf52_0"]
CHALLENGES_SHA = "03c5c50bd5880e29a3d6b6072168a2e57ea75239ca7b7d951c7e31378501c86c"

GEN_MAX = 12288
THINK_BUDGET = 8192
EFFORT = "xhigh"
SEED_P0 = 5101
SEED_FR = 5103
EXEC_PER_GRID_S = 2.0
EXEC_PER_PROGRAM_S = 20.0
FEEDBACK_MAX_CHARS = 3000

# P0 final通道强契约：首行必须直接是 ```python，代码块前后禁散文。
FINAL_CONTRACT = (
    "You must do all analysis inside the <think> phase. In the final response, "
    "the FIRST line must be exactly ```python, followed by one complete "
    "transform(grid) program, closed by ``` on its own line. Output NO analysis "
    "paragraphs, NO step restatements, NO explanations before or after the code "
    "block. Keep the program compact (<=150 lines) using direct grid-size reads, "
    "connected-component / bounding-box extraction and coordinate mapping."
)

STATE = {"proc": None, "work": None, "run": None, "closed": False}
GPU_SAMPLER_STOP = threading.Event()


def log(msg):
    line = f"[019p4 {time.strftime('%H:%M:%S')}] {msg}"
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
    if "serve_tail" not in run:
        run["serve_tail"] = server_tail(work, 30)
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


def server_tail(work: Path, n=40) -> str:
    p = work / "vllm-server.log"
    if not p.exists():
        return ""
    return "\n".join(p.read_text(errors="replace").splitlines()[-n:])


def count_preempt(log_text: str) -> int:
    t = log_text.lower()
    return t.count("preempt") + t.count("recompute") + t.count("preemption")


def gpu_sampler(work: Path, interval_s: float = 5.0):
    out = work / "gpu-samples.jsonl"
    while not GPU_SAMPLER_STOP.is_set():
        try:
            r = subprocess.run(
                ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10)
            line = {"t_wall": round(time.time(), 1), "t_mono": round(time.monotonic(), 1),
                    "raw": r.stdout.strip()[:800], "rc": r.returncode}
        except Exception as exc:
            line = {"t_wall": round(time.time(), 1), "t_mono": round(time.monotonic(), 1),
                    "error": f"{type(exc).__name__}:{exc}"[:200]}
        try:
            with out.open("a") as h:
                h.write(json.dumps(line) + "\n")
        except Exception:
            pass
        GPU_SAMPLER_STOP.wait(interval_s)


def compact(grid) -> str:
    return json.dumps(grid, separators=(",", ":"), ensure_ascii=False)


def build_p0_prompt(task: dict) -> tuple[str, str, list[dict]]:
    """P0首稿提示：017紧凑写法 + final禁散文强契约。"""
    lines = ["Solve the grid transformation. Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={compact(d['input'])} out={compact(d['output'])}")
    lines.append(f"test in={compact(task['test'][0]['input'])}")
    lines.append("Return exactly one closed ```python block with transform(grid). "
                 "No other text after the block.")
    lines.append(FINAL_CONTRACT)
    system = SP.SYSTEM_PROMPT + " " + FINAL_CONTRACT
    user = "\n".join(lines)
    return system, user, [{"role": "system", "content": system},
                         {"role": "user", "content": user}]


def is_legal_grid(grid) -> bool:
    try:
        SP.checked_grid(grid)
        return True
    except Exception:
        return False


def parse_generation(rec: dict, res: dict) -> dict:
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


def exec_program(code: str | None, parse_status: str, demos: list, test_input,
                 task_deadline: float) -> tuple[list, dict, float]:
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
        r = SP.execute_grid(code, demo["input"], wall_seconds=min(EXEC_PER_GRID_S, remaining),
                            memory_mb=512)
        spent += r.get("elapsed_s", 0.0)
        demo_results.append(r)
    assert len(demo_results) == len(demos), "demo/result length mismatch"
    remaining = min(EXEC_PER_PROGRAM_S - spent, task_deadline - time.monotonic(), EXEC_PER_GRID_S)
    if remaining <= 0:
        query_result = {"status": "budget_interrupted", "elapsed_s": 0.0}
    else:
        query_result = SP.execute_grid(code, test_input, wall_seconds=remaining, memory_mb=512)
        spent += query_result.get("elapsed_s", 0.0)
    return demo_results, query_result, round(spent, 2)


def save_program_reply(directory: Path, stem: str, rec: dict, res: dict, parsed: dict,
                       demo_results: list, query_result: dict, extra: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.reasoning.txt").write_text(res.get("reasoning") or "")
    (directory / f"{stem}.final.txt").write_text(res.get("content") or "")
    if parsed.get("code"):
        (directory / f"{stem}.code.py").write_text(parsed["code"])
    summary = {k: rec.get(k) for k in (
        "phase", "query", "arm", "request", "finish_reason", "usage", "wall_s", "ttft_s",
        "agg_tok_s", "client_error", "cancelled_by_client", "usage_present",
        "content_chars", "reasoning_chars", "status", "events")}
    summary.update({"parse_status": parsed.get("status"), "gen_ok": parsed.get("gen_ok"),
                    "validation_error": parsed.get("validation_error"),
                    "demo_matches": extra.get("demo_matches"),
                    "all_demos_pass": extra.get("all_demos_pass"),
                    "query_legal": extra.get("query_legal"),
                    "exec_seconds": extra.get("exec_seconds"),
                    "feedback_type": extra.get("feedback_type"),
                    "messages_sha256": extra.get("messages_sha256"),
                    "prompt_chars": extra.get("prompt_chars")})
    summary["grid"] = query_result.get("grid") if is_legal_grid(query_result.get("grid")) else None
    (directory / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    rec.update({k: summary[k] for k in (
        "parse_status", "gen_ok", "demo_matches", "all_demos_pass", "query_legal",
        "exec_seconds", "feedback_type")})
    rec["grid"] = summary["grid"]
    return rec


def feedback_tag(parse_status, demo_results, all_pass) -> str:
    if parse_status != "program_ready":
        return "first_draft_failed_no_program"
    if any(r.get("status") != "ok" for r in demo_results):
        return "runtime_error_feedback"
    if not all_pass:
        return "grid_difference_feedback"
    return "all_demo_pass_feedback"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=5400.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()
    t0_mono = time.monotonic()
    session_deadline = t0_mono + args.session_seconds
    run = {"config": {"tp": P.TP_SIZE, "max_model_len": P.MAX_MODEL_LEN,
                       "service_max_num_seqs": P.SERVICE_MAX_NUM_SEQS,
                       "gpu_mem_util": P.GPU_MEM_UTIL, "kv_dtype": "default", "mtp": False,
                       "sampling": dict(P.SAMPLING), "effort": EFFORT,
                       "gen_max": GEN_MAX, "think_budget": THINK_BUDGET,
                       "seeds": {"p0": SEED_P0, "fr": SEED_FR},
                       "vllm_wheel": VLLM_WHEEL_NAME, "vllm_wheel_sha256": VLLM_WHEEL_SHA256,
                       "p6_extra4": P6_EXTRA4,
                       "final_contract": "first_line_fence_no_prose"},
            "budgets": {"session_s": args.session_seconds, "save_margin_s": SAVE_MARGIN_S,
                        "new_cutoff_s": NEW_REQUEST_CUTOFF_S,
                        "plus4_max_new_output": 147456},
            "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    try:
        lock_fd = os.open(work / "driver.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        print("[019p4] lock busy, another driver holds it; exiting quietly", flush=True)
        return 0
    log(f"driver start session_s={args.session_seconds}")
    P.record_versions(run)
    log("versions recorded")

    challenges = json.loads(Path(args.challenges).read_text())
    need = [k.split("_")[0] for k in P6_EXTRA4]
    missing = [k for k in need if k not in challenges]
    if missing:
        return finish("challenges_missing", detail=str(missing))
    run["challenges_sha"] = sha256_file(Path(args.challenges))
    if run["challenges_sha"] != CHALLENGES_SHA:
        run["challenges_sha_warn"] = "输入SHA与017冻结值不一致，继续但标记不可比"
        log(f"WARN challenges sha mismatch {run['challenges_sha']}")
    else:
        log("challenges sha match 017 frozen")

    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        return finish(f"env_incompatible_python_{py.major}_{py.minor}")
    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        run["mount_error"] = str(exc)[:200]
        return finish("mount_missing")
    run["mounts"] = {"model_root": str(model_root), "wheelhouse": str(wh)}
    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels = wh / "wheels"
    req = wh / "requirements-qwen38-runtime.txt"
    wheel_path = wheels / VLLM_WHEEL_NAME
    if not wheels.is_dir() or not req.is_file() or not wheel_path.is_file():
        return finish("vllm_wheel_missing")
    digest = sha256_file(wheel_path)
    run["vllm_wheel_sha256_observed"] = digest
    if digest != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch", detail=digest)
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
    STATE["phase"] = "pip"
    try:
        ins = sh(_base() + ["--find-links", str(wheels), "--requirement", str(req),
                            "--target", str(site)], timeout=INSTALL_BUDGET_S)
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    if ins.returncode:
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        vins = sh(_base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path),
                             "--target", str(site)], timeout=600)
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    if vins.returncode:
        return finish("vllm_wheel_install_failed", detail=(vins.stderr or vins.stdout)[-400:])
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__)"], timeout=300,
             env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode, "out": (imp.stdout.strip() + " " + imp.stderr.strip())[-1500:]}
    if imp.returncode:
        return finish("import_vllm_failed", detail=run["import_check"]["out"][-400:])
    run["install_s"] = round(time.time() - t0_wall, 1)

    tpl = sh([sys.executable, "run_reasoning_probe.py", "--skip-serve", "--work", str(work / "tpl")],
             timeout=300, cwd=str(Path(__file__).resolve().parent), env=dict(os.environ))
    try:
        tpl_run = json.loads((work / "tpl" / "run.json").read_text())
    except Exception as exc:
        return finish("template_check_failed", detail=f"{type(exc).__name__}:{exc} rc={tpl.returncode}")
    run["template_render"] = tpl_run.get("template_render")
    if tpl_run.get("template_render", {}).get("status") != "RENDER_DIFFER":
        return finish("template_efforts_identical")
    log("template xhigh/low differ")

    install_deadline = (session_deadline - args.session_seconds) + INSTALL_BUDGET_S
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    try:
        proc = P.start_server(model_root, work, extra=[], deadline=install_deadline)
    except Exception as exc:
        run["serve_tail"] = server_tail(work)
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    run["install_load_s"] = round(time.time() - t0_wall, 1)
    log(f"server up install_load_s={run['install_load_s']}")
    save(work, run)

    pre = SP.execute_grid("def transform(grid):\n    return [row[:] for row in grid]", [[1]],
                          wall_seconds=2.0, memory_mb=512)
    run["executor_preflight"] = {"status": pre.get("status"), "grid": pre.get("grid")}
    save(work, run)
    if pre.get("status") != "ok" or pre.get("grid") != [[1]]:
        return finish("executor_infra_failed")
    log("executor preflight ok")
    th = threading.Thread(target=gpu_sampler, args=(work, 5.0), daemon=True)
    th.start()

    store: dict = {}
    task_deadline = session_deadline - SAVE_MARGIN_S

    def p0_messages(query: str) -> tuple[list, str]:
        task = challenges[query.split("_")[0]]
        clean = SP.task_inputs_only({"train": task["train"], "test": task["test"]})
        _, _, messages = build_p0_prompt(clean)
        return messages, hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()

    # ---- Step 1：批量P0（4次调用，C8并发，seed 5101） ----
    STATE["phase"] = "P0"
    if (session_deadline - time.monotonic() - SAVE_MARGIN_S) <= 0 or not server_alive():
        return finish("cutoff_before_p0")
    limit = min(time.monotonic() + 1800, session_deadline - SAVE_MARGIN_S)
    specs = []
    for q in P6_EXTRA4:
        messages, msha = p0_messages(q)
        if sum(len(m.get("content", "")) for m in messages) > 100000:
            run.setdefault("prompt_skipped", []).append({"query": q, "reason": "prompt_too_long"})
            continue
        specs.append({"messages": messages, "effort": EFFORT, "seed": SEED_P0,
                      "max_tokens": GEN_MAX, "timeout_s": max(1.0, limit - time.monotonic()),
                      "deadline": limit, "thinking_token_budget": THINK_BUDGET,
                      "enable_thinking": True, "request_id": f"P0-{q}",
                      "query": q, "arm": "P0", "messages_sha": msha})
    t1 = time.monotonic()
    w = P.chat_wave(specs, client_concurrency=8)
    for spec, slot in zip(specs, w["calls"]):
        rec, res = slot["rec"], slot["res"]
        rec["phase"] = "P6plus4"
        rec["query"] = spec["query"]
        rec["arm"] = "P0"
        parsed = parse_generation(rec, res)
        demos = challenges[spec["query"].split("_")[0]]["train"]
        test_input = challenges[spec["query"].split("_")[0]]["test"][0]["input"]
        demo_results, query_result, spent = exec_program(
            parsed.get("code") if parsed.get("gen_ok") else None,
            parsed.get("status"), demos, test_input, task_deadline)
        demo_matches = sum(1 for d, r in zip(demos, demo_results)
                           if r.get("status") == "ok" and r.get("grid") == d["output"])
        all_pass = demo_matches == len(demos) and len(demos) > 0
        extra = {"demo_matches": demo_matches, "all_demos_pass": all_pass,
                 "query_legal": is_legal_grid(query_result.get("grid")),
                 "exec_seconds": spent,
                 "feedback_type": feedback_tag(parsed.get("status"), demo_results, all_pass),
                 "messages_sha256": spec["messages_sha"],
                 "prompt_chars": sum(len(m.get("content", "")) for m in spec["messages"])}
        save_program_reply(work / "P6plus4", spec["request_id"], rec, res, parsed,
                           demo_results, query_result, extra)
        run["calls"].append(rec)
        store[(spec["query"], "P0")] = {
            "rec": rec, "res": res, "parsed": parsed, "demo_results": demo_results,
            "query_result": query_result, "extra": extra, "messages": spec["messages"]}
    run["P0"] = {"n": len(specs), "wall_s": round(time.monotonic() - t1, 2),
                 "wave_wall_s": w["wave_wall_s"],
                 "completion_tokens_sum": w["completion_tokens_sum"],
                 "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"]}
    save(work, run)
    log(f"P0 done {run['P0']}")

    # ---- Step 2：F（一次修复）与 R（独立重启），同预算 seed 5103 ----
    STATE["phase"] = "FR"
    if (session_deadline - time.monotonic() - SAVE_MARGIN_S) <= 0 or not server_alive():
        run["FR"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)
    else:
        limit = min(time.monotonic() + 1800, session_deadline - SAVE_MARGIN_S)
        specs = []
        for idx, q in enumerate(P6_EXTRA4):
            p0 = store.get((q, "P0"))
            if p0 is None:
                continue
            base_messages = p0["messages"]
            p0code = (p0["parsed"] or {}).get("code") or ""
            demos = challenges[q.split("_")[0]]["train"]
            if p0["parsed"].get("gen_ok"):
                try:
                    feedback = SP.demo_feedback(demos, p0["demo_results"], FEEDBACK_MAX_CHARS)
                except Exception as exc:
                    feedback = json.dumps(
                        {"feedback_error": f"{type(exc).__name__}:{exc}"[:200]})
                frep = (f"Execution feedback (JSON, real demo results, "
                        f"<= {FEEDBACK_MAX_CHARS} chars):\n{feedback}\n"
                        f"Write the corrected complete transform(grid) as exactly one closed "
                        f"```python block. {FINAL_CONTRACT}")
                f_messages = base_messages + [
                    {"role": "user", "content": f"Your previous program:\n```python\n{p0code}\n```\n{frep}"}]
                # F与R交错顺序，避免系统性位置偏差
                arms = [("F", f_messages), ("R", base_messages)] if idx % 2 == 0 else [
                    ("R", base_messages), ("F", f_messages)]
            else:
                # P0不完整：仅发同预算独立重启R；A_F在该题无F候选
                arms = [("R", base_messages)]
            for arm, msgs in arms:
                specs.append({"messages": msgs, "effort": EFFORT, "seed": SEED_FR,
                              "max_tokens": GEN_MAX,
                              "timeout_s": max(1.0, limit - time.monotonic()),
                              "deadline": limit, "thinking_token_budget": THINK_BUDGET,
                              "enable_thinking": True, "request_id": f"{arm}-{q}",
                              "query": q, "arm": arm})
        t1 = time.monotonic()
        w = P.chat_wave(specs, client_concurrency=8)
        for spec, slot in zip(specs, w["calls"]):
            rec, res = slot["rec"], slot["res"]
            rec["phase"] = "P6plus4"
            rec["query"] = spec["query"]
            rec["arm"] = spec["arm"]
            parsed = parse_generation(rec, res)
            demos = challenges[spec["query"].split("_")[0]]["train"]
            test_input = challenges[spec["query"].split("_")[0]]["test"][0]["input"]
            demo_results, query_result, spent = exec_program(
                parsed.get("code") if parsed.get("gen_ok") else None,
                parsed.get("status"), demos, test_input, task_deadline)
            demo_matches = sum(1 for d, r in zip(demos, demo_results)
                               if r.get("status") == "ok" and r.get("grid") == d["output"])
            all_pass = demo_matches == len(demos) and len(demos) > 0
            extra = {"demo_matches": demo_matches, "all_demos_pass": all_pass,
                     "query_legal": is_legal_grid(query_result.get("grid")),
                     "exec_seconds": spent,
                     "feedback_type": feedback_tag(parsed.get("status"), demo_results, all_pass),
                     "messages_sha256": hashlib.sha256(
                         json.dumps(spec["messages"], ensure_ascii=False).encode()).hexdigest(),
                     "prompt_chars": sum(len(m.get("content", "")) for m in spec["messages"])}
            save_program_reply(work / "P6plus4", spec["request_id"], rec, res, parsed,
                               demo_results, query_result, extra)
            run["calls"].append(rec)
            store[(spec["query"], spec["arm"])] = {
                "rec": rec, "res": res, "parsed": parsed, "demo_results": demo_results,
                "query_result": query_result, "extra": extra, "messages": spec["messages"]}
        run["FR"] = {"n": len(specs), "wall_s": round(time.monotonic() - t1, 2),
                     "wave_wall_s": w["wave_wall_s"],
                     "completion_tokens_sum": w["completion_tokens_sum"],
                     "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"]}
        save(work, run)
        log(f"FR done {run['FR']}")

    # ---- Step 3：保留起始候选与双槽评价（A_F=[P0,F]，A_R=[P0,R]） ----
    slots: dict = {}
    for q in P6_EXTRA4:
        got = {}
        for arm in ("P0", "F", "R"):
            s = store.get((q, arm))
            got[arm] = s["query_result"].get("grid") if (
                s and is_legal_grid(s["query_result"].get("grid"))) else None
        for name, pair in (("A_F", ("P0", "F")), ("A_R", ("P0", "R"))):
            a, b = got[pair[0]], got[pair[1]]
            if a is None and b is None:
                slots[f"{name}-{q}"] = {"slot_1": None, "slot_2": None, "empty": True}
            elif a is None:
                slots[f"{name}-{q}"] = {"slot_1": b, "slot_2": b, "empty": False}
            elif b is None:
                slots[f"{name}-{q}"] = {"slot_1": a, "slot_2": a, "empty": False}
            else:
                slots[f"{name}-{q}"] = {"slot_1": a, "slot_2": b, "empty": False}
    run["slots_note"] = ("A_F=[P0,F], A_R=[P0,R]; P0-incomplete queries have no F; "
                         "correctness scored locally, never on GPU.")
    (work / "slots.json").write_text(json.dumps(
        {k: {"empty": v["empty"],
             "slot_1_shape": [len(v["slot_1"]), len(v["slot_1"][0])] if v["slot_1"] else None,
             "slot_2_shape": [len(v["slot_2"]), len(v["slot_2"][0])] if v["slot_2"] else None}
         for k, v in slots.items()}, indent=1))
    # 本地可直接报告的漏斗（query exact需取回后对照010 scoring.json，驱动内不判分）
    run["funnel_local"] = {
        q: {arm: {"gen_ok": bool(store.get((q, arm), {}).get("parsed", {}).get("gen_ok")),
                  "finish": (store.get((q, arm), {}).get("res") or {}).get("finish"),
                  "demo_matches": (store.get((q, arm), {}).get("extra") or {}).get("demo_matches"),
                  "all_demos_pass": (store.get((q, arm), {}).get("extra") or {}).get(
                      "all_demos_pass")}
            for arm in ("P0", "F", "R")}
        for q in P6_EXTRA4}
    save(work, run)

    full = (work / "vllm-server.log").read_text(errors="replace") if (work / "vllm-server.log").exists() else ""
    run["preempt_total"] = count_preempt(full)
    run["new_output_tokens_main"] = sum(
        ((c.get("usage") or {}).get("completion_tokens") or 0) for c in run.get("calls", []))
    save(work, run)
    return finish("complete_with_evidence")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:
        if STATE.get("run") is not None:
            STATE["run"]["traceback_tail"] = traceback.format_exc()[-1500:]
        raise SystemExit(finish("unexpected_error", 2, f"{type(exc).__name__}:{exc}"))
