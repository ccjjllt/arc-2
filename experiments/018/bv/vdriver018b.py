#!/usr/bin/env python3
"""018-B-verify: constrained final (guided_json) -> CPU validate -> search -> execute.

Fail-closed: unknown guided field (server 400) -> structured_unsupported, stop.
enable_in_reasoning=False is sent top-level per fixed-version verification
(user-checked source); if the server rejects it we stop, never silently rerun
unconstrained. Thinking stays 8192 (plan: judge mechanism effect first).
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
import opspec as O

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
SAVE_MARGIN_S = 300
NEW_REQUEST_CUTOFF_S = 40 * 60

TASKS = ["291dc1e1_0", "db118e2a_0"]
GEN_MAX = 4096
THINK_BUDGET = 8192
EFFORT = "xhigh"
SEEDS = {"291dc1e1_0": 5201, "db118e2a_0": 5202}

STATE = {"proc": None, "work": None, "run": None, "closed": False, "phase": "init"}
STOP = threading.Event()


def log(msg):
    line = f"[018b {time.strftime('%H:%M:%S')}] {msg}"
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
    STOP.set()
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
        STOP.set()
        return code
    return orderly(work, run, status, code)


def on_term(signum, _frame):
    finish(f"signal_{signum}")
    os._exit(0)


def template_prompt(task: dict) -> list:
    ops_doc = ("Output a JSON object {\"steps\":[...]} and nothing else. "
               "Allowed ops and params: "
               "extract_objects(same_color:bool, connectivity:4|8); "
               "filter_objects(criterion:area_gt|area_lt|color_eq|touches_border, value:0..900); "
               "sort_objects(by:area|row|col, order:asc|desc); "
               "transform_objects(op:rotate|flip_h|flip_v|shift, quarter_turns:0..3, dr:-29..29, dc:-29..29); "
               "place_objects(background:0..9, canvas:blank|input, overlap:error|overwrite); "
               "output(). 1..12 steps, must include place_objects and output. "
               "A param may be {\"choices\":[...]} with 1..4 options for CPU search. "
               "No other keys, no text outside JSON.")
    lines = [ops_doc, "Demonstrations (JSON input -> output):"]
    for i, d in enumerate(task["train"]):
        lines.append(f"demo{i} in={json.dumps(d['input'], separators=(',', ':'))} "
                     f"out={json.dumps(d['output'], separators=(',', ':'))}")
    lines.append(f"test in={json.dumps(task['test'][0]['input'], separators=(',', ':'))}")
    return [{"role": "system", "content": "You write short constrained JSON plans."},
            {"role": "user", "content": "\n".join(lines)}]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--challenges", required=True)
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=3600.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()
    t0_mono = time.monotonic()
    session_deadline = t0_mono + args.session_seconds
    run = {"config": {"effort": EFFORT, "gen_max": GEN_MAX, "think_budget": THINK_BUDGET,
                       "seeds": SEEDS, "tasks": TASKS,
                       "enable_in_reasoning": False,
                       "vllm_wheel_sha256": VLLM_WHEEL_SHA256},
           "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    try:
        lock_fd = os.open(work / "driver.lock", os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(lock_fd, str(os.getpid()).encode())
    except FileExistsError:
        print("[018b] lock busy, exiting quietly", flush=True)
        return 0
    log(f"driver start session_s={args.session_seconds}")
    P.record_versions(run)
    log("versions recorded")
    STATE["phase"] = "frozen"

    challenges = json.loads(Path(args.challenges).read_text())
    run["challenges_sha"] = sha256_file(Path(args.challenges))
    save(work, run)
    log("challenges ok")

    # install chain (same as 018a, condensed)
    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        return finish("mount_missing", detail=str(exc)[:200])
    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels, req = wh / "wheels", wh / "requirements-qwen38-runtime.txt"
    wheel_path = wheels / VLLM_WHEEL_NAME
    if sha256_file(wheel_path) != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch")
    pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    if pip_probe.returncode:
        sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
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
    run["pip_requirements_rc"] = ins.returncode
    save(work, run)
    if ins.returncode:
        return finish("install_failed")
    try:
        vins = sh(_base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path),
                             "--target", str(site)], timeout=600)
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    run["pip_wheel_rc"] = vins.returncode
    save(work, run)
    if vins.returncode:
        return finish("vllm_wheel_install_failed")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    STATE["phase"] = "import"
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__)"], timeout=300,
             env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode}
    save(work, run)
    if imp.returncode:
        return finish("import_vllm_failed")
    install_deadline = (session_deadline - args.session_seconds) + INSTALL_BUDGET_S
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    STATE["phase"] = "serve"
    try:
        proc = P.start_server(model_root, work, extra=[], deadline=install_deadline)
    except Exception as exc:
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    save(work, run)

    schema = O.guided_schema()
    t0 = time.monotonic()
    limit = min(t0 + 1500, session_deadline - SAVE_MARGIN_S)
    specs = []
    for q in TASKS:
        task = challenges[q.split("_")[0]]
        clean = SP.task_inputs_only({"train": task["train"], "test": task["test"]})
        specs.append({"messages": template_prompt(clean), "effort": EFFORT,
                      "seed": SEEDS[q], "max_tokens": GEN_MAX,
                      "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                      "thinking_token_budget": THINK_BUDGET, "enable_thinking": True,
                      "request_id": f"PLAN-{q}", "query": q,
                      "extra_body": {"guided_json": schema},
                      "enable_in_reasoning": False})
    STATE["phase"] = "calls"
    try:
        w = P.chat_wave(specs, client_concurrency=8)
    except ValueError as exc:
        return finish("request_build_failed", detail=str(exc)[:300])
    for spec, slot in zip(specs, w["calls"]):
        rec, res = slot["rec"], slot["res"]
        rec["phase"], rec["query"], rec["arm"] = "BV", spec["query"], "PLAN"
        entry = {"request_id": spec["request_id"], "finish": res.get("finish"),
                 "usage": rec.get("usage")}
        # server-side guided rejection surfaces as client_error/400 or bad finish
        if rec.get("client_error") or res.get("finish") != "stop":
            entry["guided_status"] = "server_rejected_or_incomplete"
            entry["detail"] = str(rec.get("client_error") or res.get("finish"))[:200]
            run["calls"].append(entry)
            save(work, run)
            continue
        try:
            plan = json.loads(res.get("content") or "")
            norm = O.validate_template(plan)
            entry["guided_status"] = "valid_plan"
        except (ValueError, json.JSONDecodeError) as exc:
            entry["guided_status"] = "invalid_plan"
            entry["detail"] = str(exc)[:300]
            run["calls"].append(entry)
            save(work, run)
            continue
        demos = challenges[spec["query"].split("_")[0]]["train"]
        expected = [d["output"] for d in demos]
        sr = O.search_template(norm, [d["input"] for d in demos], expected)
        entry["search"] = {k: sr.get(k) for k in ("status", "tried", "best_demo_matches", "elapsed_s")}
        if sr.get("status") == "all_pass":
            test_in = challenges[spec["query"].split("_")[0]]["test"][0]["input"]
            concrete = sr["config"]
            steps = norm["steps"]
            out = O._run_steps(test_in, steps, concrete)
            entry["query_legal"] = bool(out.get("ok"))
            entry["grid"] = out.get("grid") if out.get("ok") else None
        (work / (spec["request_id"] + ".json")).write_text(json.dumps(entry, indent=1))
        run["calls"].append(entry)
        save(work, run)
    run["elapsed_calls_s"] = round(time.monotonic() - t0, 1)
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
