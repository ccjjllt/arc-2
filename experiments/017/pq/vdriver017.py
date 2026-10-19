#!/usr/bin/env python3
"""017 driver: P2 gate -> +4 program comparison (shared P0 + one real demo feedback vs same-budget restart).

Frozen: xhigh, seed P0=5101, F/R=5102, each call 12288 max / thinking<=8192.
SYSTEM_PROMPT arc011-program-v1 unchanged; prompts built from P6 challenges subset.
Parse: driver checks stop+no-error first, then split_program_response direct mode.
Executor: vendored-file solve_program helpers only (no old runner, no torch).
Slots: A_F=[P0,F], A_R=[P0,R]; correctness scored locally, never on GPU.
"""
from __future__ import annotations

import argparse
import glob
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
HARD_STOP_S = 85 * 60

P2_KEYS = ["291dc1e1_0", "db118e2a_0"]
P6_KEYS = ["291dc1e1_0", "4c416de3_0", "58c02a16_0", "aa4ec2a5_0", "db118e2a_0", "fc4aaf52_0"]
P6_EXTRA4 = ["4c416de3_0", "58c02a16_0", "aa4ec2a5_0", "fc4aaf52_0"]
CHALLENGES_SHA = "03c5c50bd5880e29a3d6b6072168a2e57ea75239ca7b7d951c7e31378501c86c"

GEN_MAX = 12288
THINK_BUDGET = 8192
EFFORT = "xhigh"
SEED_P0 = 5101
SEED_FR = 5102
EXEC_PER_GRID_S = 2.0
EXEC_PER_PROGRAM_S = 20.0
FEEDBACK_MAX_CHARS = 3000

STATE = {"proc": None, "work": None, "run": None, "closed": False}
GPU_SAMPLER_STOP = threading.Event()


def log(msg):
    print(f"[017 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def kv_estimate(log_text: str) -> str:
    outs = []
    for ln in log_text.splitlines():
        ll = ln.lower()
        if "kv cache" in ll or "kv_cache" in ll or "gpu kv" in ll or "available kv" in ll:
            outs.append(ln.strip()[:300])
    return "\n".join(outs[-8:])


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


def time_left(session_deadline: float, margin: float = SAVE_MARGIN_S) -> float:
    return session_deadline - time.monotonic() - margin


def past_cutoff(t0_mono: float) -> bool:
    return (time.monotonic() - t0_mono) >= NEW_REQUEST_CUTOFF_S


def compact(grid) -> str:
    return json.dumps(grid, separators=(",", ":"), ensure_ascii=False)


def build_prompt(task: dict) -> tuple[str, str, list[dict]]:
    """SYSTEM_PROMPT unchanged + compact-JSON demos/test. Returns (system, user, messages)."""
    lines = ["Solve the grid transformation. Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={compact(d['input'])} out={compact(d['output'])}")
    lines.append(f"test in={compact(task['test'][0]['input'])}")
    lines.append("Return exactly one closed ```python block with transform(grid). No other text after the block.")
    user = "\n".join(lines)
    messages = [{"role": "system", "content": SP.SYSTEM_PROMPT},
                {"role": "user", "content": user}]
    return SP.SYSTEM_PROMPT, user, messages


def is_legal_grid(grid) -> bool:
    try:
        SP.checked_grid(grid)
        return True
    except Exception:
        return False


def parse_generation(rec: dict, res: dict) -> dict:
    """Driver owns the completion gate: stop+no-error first, then direct parse of final."""
    finish = res.get("finish")
    if finish == "stop" and not rec.get("client_error") and not rec.get("cancelled_by_client"):
        parsed = SP.split_program_response(res.get("content") or "", thinking_mode="direct", generation=None)
        parsed["gen_ok"] = parsed.get("status") == "program_ready"
        return parsed
    reason = rec.get("client_error") or f"finish={finish}"
    return {"status": "generation_not_ok", "reason": str(reason)[:200],
            "reasoning_content": res.get("reasoning") or "", "final": res.get("content") or "",
            "code": None, "gen_ok": False}


def exec_program(code: str | None, parse_status: str, demos: list, test_input,
                 task_deadline: float) -> tuple[list, dict, float]:
    """Execute on all demos + test input. Unexecuted cases get explicit statuses (no silent zip gaps)."""
    demo_results: list = []
    spent = 0.0
    if code is None:
        for _ in demos:
            demo_results.append({"status": parse_status, "error": "no admitted program", "elapsed_s": 0.0})
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


def feedback_tag(parse_status: str, demo_results: list, all_pass: bool) -> str:
    if parse_status != "program_ready":
        return "parse_recovery"
    if any(r.get("status") != "ok" for r in demo_results):
        return "runtime_error_feedback"
    if not all_pass:
        return "grid_difference_feedback"
    return "all_demo_pass_feedback"


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
                    "demo_matches": extra.get("demo_matches"), "all_demos_pass": extra.get("all_demos_pass"),
                    "query_legal": extra.get("query_legal"), "exec_seconds": extra.get("exec_seconds"),
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
                       "p2": P2_KEYS, "p6_extra4": P6_EXTRA4},
           "budgets": {"session_s": args.session_seconds, "save_margin_s": SAVE_MARGIN_S,
                       "new_cutoff_s": NEW_REQUEST_CUTOFF_S, "hard_stop_s": HARD_STOP_S,
                       "p2_max_new_output": 73728, "plus4_max_new_output": 147456},
           "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    P.record_versions(run)
    log(f"env: {json.dumps(run.get('host', {}))[:400]}")

    challenges = json.loads(Path(args.challenges).read_text())
    need = [k.split("_")[0] for k in P2_KEYS + P6_EXTRA4]
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
        run["wheelhouse_layout"] = sorted(p.name for p in wh.iterdir())[:30]
        return finish("vllm_wheel_missing" if not wheel_path.is_file() else "wheelhouse_layout_unexpected")
    digest = sha256_file(wheel_path)
    run["vllm_wheel_sha256_observed"] = digest
    if digest != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch", detail=digest)

    pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    if pip_probe.returncode:
        ens = sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
        (work / "ensurepip.log").write_text((ens.stdout + "\n---STDERR---\n" + ens.stderr)[-4000:])
        run["ensurepip_rc"] = ens.returncode
        pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    use_fallback = pip_probe.returncode != 0
    if use_fallback and not args.pip_python:
        return finish("no_pip_for_312_and_no_fallback")
    run["installer"] = "fallback_pip_python312_target" if use_fallback else "py312_pip"

    def _base():
        if use_fallback:
            return [args.pip_python, "-m", "pip", "install", "--python-version", "3.12",
                    "--only-binary", ":all:", "--no-index"]
        return [sys.executable, "-m", "pip", "install", "--no-index"]
    req_cmd = _base() + ["--find-links", str(wheels), "--requirement", str(req), "--target", str(site)]
    vllm_cmd = _base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path), "--target", str(site)]
    run["install_cmd"] = {"requirements": req_cmd, "vllm_no_deps": vllm_cmd}
    try:
        ins = sh(req_cmd, timeout=INSTALL_BUDGET_S)
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    if ins.returncode:
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        vins = sh(vllm_cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    (work / "pip-vllm.log").write_text(vins.stdout[-4000:] + "\n---STDERR---\n" + vins.stderr[-4000:])
    if vins.returncode:
        return finish("vllm_wheel_install_failed", detail=(vins.stderr or vins.stdout)[-400:])
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__); print(vllm.__file__); print(torch.__version__)"],
             timeout=300, env=dict(os.environ))
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
    if tpl_run.get("template_render", {}).get("status") != "d differ":
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
    log(f"server up service_max=8 install_load_s={run['install_load_s']}")
    save(work, run)
    full_log = (work / "vllm-server.log").read_text(errors="replace") if (work / "vllm-server.log").exists() else ""
    run["kv_estimate_at_start"] = kv_estimate(full_log)[-2000:]
    run["preempt_at_start"] = count_preempt(full_log)
    save(work, run)

    # ---- 执行器预检（无模型）：恒等函数必须可执行 ----
    pre = SP.execute_grid("def transform(grid):\n    return [row[:] for row in grid]", [[1]],
                          wall_seconds=2.0, memory_mb=512)
    run["executor_preflight"] = {"status": pre.get("status"), "grid": pre.get("grid"),
                                 "error": pre.get("error")}
    save(work, run)
    if pre.get("status") != "ok" or pre.get("grid") != [[1]]:
        return finish("executor_infra_failed", detail=json.dumps(run["executor_preflight"])[:300])
    log("executor preflight ok")

    th = threading.Thread(target=gpu_sampler, args=(work, 5.0), daemon=True)
    th.start()

    store: dict = {}  # (query, arm) -> {rec,res,parsed,demo_results,query_result,messages_sha}

    def p0_messages(query: str) -> tuple[list, str]:
        task = challenges[query.split("_")[0]]
        clean = SP.task_inputs_only({"train": task["train"], "test": task["test"]})
        _, _, messages = build_prompt(clean)
        return messages, hashlib.sha256(json.dumps(messages, ensure_ascii=False).encode()).hexdigest()

    def run_p0_wave(queries: list, tag: str) -> dict:
        limit = min(time.monotonic() + 1800, session_deadline - SAVE_MARGIN_S)
        specs = []
        for q in queries:
            messages, msha = p0_messages(q)
            if sum(len(m.get("content", "")) for m in messages) > 100000:
                run.setdefault("prompt_skipped", []).append({"query": q, "reason": "prompt_too_long"})
                continue
            specs.append({"messages": messages, "effort": EFFORT, "seed": SEED_P0,
                          "max_tokens": GEN_MAX, "timeout_s": max(1.0, limit - time.monotonic()),
                          "deadline": limit, "thinking_token_budget": THINK_BUDGET,
                          "enable_thinking": True, "request_id": f"P0-{q}",
                          "query": q, "arm": "P0", "messages_sha": msha})
        if not specs:
            return {"n": 0, "wall_s": 0.0, "completion_tokens_sum": 0, "usage_missing": 0,
                    "agg_tok_s": None}
        t1 = time.monotonic()
        # wave只负责并发拿回文本；解析与执行随后逐条串行落盘
        w = P.chat_wave(specs, client_concurrency=8)
        wall = time.monotonic() - t1
        task_deadline = session_deadline - SAVE_MARGIN_S
        for spec, slot in zip(specs, w["calls"]):
            rec, res = slot["rec"], slot["res"]
            rec["phase"] = "P2" if spec["query"] in P2_KEYS else "P6plus4"
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
            save_program_reply(work / rec["phase"], spec["request_id"], rec, res, parsed,
                               demo_results, query_result, extra)
            run["calls"].append(rec)
            store[(spec["query"], "P0")] = {
                "rec": rec, "res": res, "parsed": parsed, "demo_results": demo_results,
                "query_result": query_result, "extra": extra, "messages": spec["messages"]}
        save(work, run)
        return {"n": len(specs), "wall_s": round(wall, 2), "wave_wall_s": w["wave_wall_s"],
                "completion_tokens_sum": w["completion_tokens_sum"],
                "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"]}

    def run_fr_wave(queries: list) -> dict:
        limit = min(time.monotonic() + 1800, session_deadline - SAVE_MARGIN_S)
        task_deadline = session_deadline - SAVE_MARGIN_S
        specs = []
        for idx, q in enumerate(queries):
            p0 = store.get((q, "P0"))
            if p0 is None:
                continue
            base_messages = p0["messages"]
            p0final = p0["res"].get("content") or ""
            demos = challenges[q.split("_")[0]]["train"]
            fb_src = p0["demo_results"]
            try:
                feedback = SP.demo_feedback(demos, fb_src, FEEDBACK_MAX_CHARS)
            except Exception as exc:
                feedback = json.dumps({"feedback_error": f"{type(exc).__name__}:{exc}"[:200]})
            frep = (f"Execution feedback (JSON, ≤{FEEDBACK_MAX_CHARS} chars):\n{feedback}\n"
                    f"Write the corrected complete transform(grid) as exactly one closed "
                    f"```python block.")
            f_messages = base_messages + [
                {"role": "user", "content": f"Your previous program:\n```python\n{p0final}\n```\n{frep}"}]
            arms = [("F", f_messages), ("R", base_messages)] if idx % 2 == 0 else [("R", base_messages), ("F", f_messages)]
            for arm, msgs in arms:
                specs.append({"messages": msgs, "effort": EFFORT, "seed": SEED_FR,
                              "max_tokens": GEN_MAX, "timeout_s": max(1.0, limit - time.monotonic()),
                              "deadline": limit, "thinking_token_budget": THINK_BUDGET,
                              "enable_thinking": True, "request_id": f"{arm}-{q}",
                              "query": q, "arm": arm})
        t1 = time.monotonic()
        w = P.chat_wave(specs, client_concurrency=8)
        wall = time.monotonic() - t1
        for spec, slot in zip(specs, w["calls"]):
            rec, res = slot["rec"], slot["res"]
            rec["phase"] = "P2" if spec["query"] in P2_KEYS else "P6plus4"
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
            save_program_reply(work / rec["phase"], spec["request_id"], rec, res, parsed,
                               demo_results, query_result, extra)
            run["calls"].append(rec)
            store[(spec["query"], spec["arm"])] = {
                "rec": rec, "res": res, "parsed": parsed, "demo_results": demo_results,
                "query_result": query_result, "extra": extra, "messages": spec["messages"]}
        save(work, run)
        return {"n": len(specs), "wall_s": round(wall, 2), "wave_wall_s": w["wave_wall_s"],
                "completion_tokens_sum": w["completion_tokens_sum"],
                "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"]}

    # ---- P2先导 ----
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        run["P2P0"] = run_p0_wave(P2_KEYS, "P2")
    else:
        run["P2P0"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        run["P2FR"] = run_fr_wave(P2_KEYS)
    else:
        run["P2FR"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)

    # ---- 门控：两题各≥1完整P0，且P2至少一个demo合法输出；否则P2后结束 ----
    def _p0_ok(q):
        s = store.get((q, "P0"))
        return bool(s and s["parsed"].get("status") == "program_ready")

    def _demo_legal_any():
        for (q, _arm), s in store.items():
            if q in P2_KEYS:
                for r in s.get("demo_results", []):
                    if r.get("status") == "ok":
                        return True
        return False

    gate_p0 = all(_p0_ok(q) for q in P2_KEYS)
    gate_demo = _demo_legal_any()
    # 预计+4成本：按P2实测放大
    p2_calls = [c for c in run["calls"] if c.get("phase") == "P2" and c.get("arm") in ("P0", "F", "R")]
    p2_tok = sum(((c.get("usage") or {}).get("completion_tokens") or 0) for c in p2_calls)
    p2_wall = (run.get("P2P0") or {}).get("wall_s", 0) + (run.get("P2FR") or {}).get("wall_s", 0)
    need_est = p2_wall * 2 + 900 if p2_wall else 3600
    gate_time = time_left(session_deadline) > need_est
    run["gate"] = {"p0_both_ok": gate_p0, "demo_legal_any": gate_demo,
                   "p2_tok": p2_tok, "p2_wall_s": round(p2_wall, 1),
                   "need_est_s": round(need_est, 1),
                   "remain_s": round(time_left(session_deadline), 1),
                   "time_ok": gate_time,
                   "decision": "open_plus4" if (gate_p0 and gate_demo and gate_time) else "closed"}
    save(work, run)
    log(f"gate {run['gate']['decision']} p0={gate_p0} demo={gate_demo} time={gate_time}")

    if run["gate"]["decision"] == "open_plus4" and time_left(session_deadline) > 1 \
            and not past_cutoff(t0_mono) and server_alive():
        run["P4P0"] = run_p0_wave(P6_EXTRA4, "P6plus4")
        save(work, run)
        if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
            run["P4FR"] = run_fr_wave(P6_EXTRA4)
        else:
            run["P4FR"] = {"status": "skipped_cutoff_or_dead"}
            save(work, run)
    else:
        if run["gate"]["decision"] != "open_plus4":
            run["P6"] = {"status": "gate_closed_no_plus4", "n": 0, "complete": False,
                         "reason": run["gate"]}
            (work / "P6-summary.json").write_text(json.dumps(
                {"status": "gate_closed_no_plus4", "reason": run["gate"]}, indent=1))
        else:
            run["P6"] = {"status": "skipped_cutoff_or_dead", "n": 0, "complete": False}
        save(work, run)

    # ---- 两槽装配（A_F=[P0,F]，A_R=[P0,R]；缺一份重复另一份，两份缺则空） ----
    slots: dict = {}
    for q in P2_KEYS + P6_EXTRA4:
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
    run["slots_note"] = ("A_F=[P0,F], A_R=[P0,R]; correctness scored locally, never on GPU. "
                         "[F,R] oracle not computed here.")
    (work / "slots.json").write_text(json.dumps(
        {k: {"empty": v["empty"],
             "slot_1_shape": [len(v["slot_1"]), len(v["slot_1"][0])] if v["slot_1"] else None,
             "slot_2_shape": [len(v["slot_2"]), len(v["slot_2"][0])] if v["slot_2"] else None}
         for k, v in slots.items()}, indent=1))
    save(work, run)

    # 会话总账
    full = (work / "vllm-server.log").read_text(errors="replace") if (work / "vllm-server.log").exists() else ""
    run["kv_estimate_at_end"] = kv_estimate(full)[-2000:]
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