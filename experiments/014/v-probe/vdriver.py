#!/usr/bin/env python3
"""V single-server driver (014 plan section 5). Import-safe (no work on import).

One vLLM server (TP4/64K/C4/FP8, KV default, no MTP — see run_reasoning_probe),
then staged calls: hello xhigh/low (short protocol), then P2 queries. All
budgets are wall-clock from driver start; known negative outcomes exit 0 with
an explicit status in run.json (evidence, not crash). Unexpected errors exit 2.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import run_reasoning_probe as P

INSTALL_BUDGET_S = 20 * 60
PROTOCOL_BUDGET_S = 5 * 60
HELLO_MAX_TOKENS = 4096
P2_MAX_TOKENS = 8192
CALL_TIMEOUT_S = 600.0


def log(msg):
    print(f"[vdriver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def run(cmd, timeout, cwd=None, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                          cwd=cwd, env=env)


def orderly(work, run, status, code=0):
    run["status"] = status
    (work / "run.json").write_text(json.dumps(run, indent=1))
    log(f"EVIDENCE-EXIT status={status}")
    return code


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--prompts", default=None, help="p2-prompts.json path")
    ap.add_argument("--session-seconds", type=float, default=3600.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    run = {"config": {"tp": P.TP_SIZE, "max_model_len": P.MAX_MODEL_LEN,
                       "max_num_seqs": P.MAX_NUM_SEQS, "gpu_mem_util": P.GPU_MEM_UTIL,
                       "kv_dtype": "default", "mtp": False,
                       "sampling": dict(P.SAMPLING),
                       "hello_max_tokens": HELLO_MAX_TOKENS,
                       "p2_max_tokens": P2_MAX_TOKENS},
           "budgets": {"install_s": INSTALL_BUDGET_S, "protocol_s": PROTOCOL_BUDGET_S,
                       "session_s": args.session_seconds}}
    P.record_versions(run)
    log(f"env: {json.dumps(run.get('host', {}))[:400]}")

    # 1) Python gate: cp312 wheels need 3.12.
    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        return orderly(work, run, f"env_incompatible_python_{py.major}_{py.minor}")

    # 2) Mounts.
    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        run["mount_error"] = str(exc)[:200]
        return orderly(work, run, "mount_missing")
    run["mounts"] = {"model_root": str(model_root), "wheelhouse": str(wh)}
    log(f"model: {model_root} wheelhouse: {wh}")

    # 3) Offline install into isolated prefix (never touch system torch).
    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels = wh / "wheels"
    req = wh / "requirements-qwen38-runtime.txt"
    if not wheels.is_dir() or not req.is_file():
        run["wheelhouse_layout"] = sorted(p.name for p in wh.iterdir())[:20]
        return orderly(work, run, "wheelhouse_layout_unexpected")
    log(f"pip install ({len(list(wheels.glob('*.whl')))} wheels) ...")
    try:
        ins = run(["python", "-m", "pip", "install", "--no-index", "--find-links", str(wheels),
                   "--requirement", str(req), "--target", str(site)], timeout=INSTALL_BUDGET_S)
    except subprocess.TimeoutExpired:
        return orderly(work, run, "install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-6000:] + "\n---STDERR---\n" + ins.stderr[-6000:])
    if ins.returncode:
        run["pip_rc"] = ins.returncode
        return orderly(work, run, "install_failed")
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    imp = run(["python", "-c", "import vllm, torch; print(vllm.__version__, torch.__version__)",
               ], timeout=300, env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode,
                           "out": (imp.stdout.strip() + " " + imp.stderr.strip())[-300:]}
    if imp.returncode:
        return orderly(work, run, "import_vllm_failed")
    log("import vllm OK: " + run["import_check"]["out"])
    run["install_s"] = round(time.time() - t0, 1)

    # 4) Template two-effort render check (no server; reuse frozen probe CLI).
    tpl = run(["python", "run_reasoning_probe.py", "--skip-serve", "--work", str(work / "tpl")],
              timeout=300, cwd=str(Path(__file__).resolve().parent),
              env=dict(os.environ))
    try:
        tpl_run = json.loads((work / "tpl" / "run.json").read_text())
    except Exception as exc:
        run["template_error"] = f"{type(exc).__name__}"[:100]
        return orderly(work, run, "template_check_failed")
    run["template_render"] = tpl_run.get("template_render")
    if tpl_run.get("template_render", {}).get("status") != "d differ":
        return orderly(work, run, "template_efforts_identical")
    log("template xhigh/low differ OK")

    # 5) Serve once.
    t_serve = time.time()
    try:
        proc = P.start_server(model_root, work, extra=[])
    except Exception as exc:
        run["serve_error"] = f"{type(exc).__name__}:{exc}"[:300]
        return orderly(work, run, "serve_failed")
    if time.time() - t0 > INSTALL_BUDGET_S:
        P.kill_server(proc, work)
        return orderly(work, run, "install_load_over_20min")
    run["install_load_s"] = round(time.time() - t0, 1)

    run["calls"] = []
    try:
        # 6) Short protocol: hello xhigh + low.
        t_proto = time.time()
        for effort, seed in (("xhigh", P.SEED_XHIGH), ("low", P.SEED_LOW)):
            if time.time() - t_proto > PROTOCOL_BUDGET_S:
                run["calls"].append({"effort": effort, "status": "protocol_budget_exceeded"})
                break
            rec = {}
            res = P.chat_once([{"role": "user", "content": "Return [[1]]."}],
                              effort, seed, HELLO_MAX_TOKENS, 300.0, rec)
            rec["phase"] = "hello"
            (work / f"reply-hello-{effort}.json").write_text(json.dumps(
                {"request": rec.get("request"), "finish_reason": rec.get("finish_reason"),
                 "usage": rec.get("usage"), "wall_s": rec.get("wall_s"),
                 "ttft_s": rec.get("ttft_s"), "agg_tok_s": rec.get("agg_tok_s"),
                 "content": res["content"], "reasoning": res["reasoning"][:2000]}, indent=1))
            run["calls"].append(rec)
            log(f"hello-{effort}: finish={rec.get('finish_reason')} wall={rec.get('wall_s')} "
                f"tok/s={rec.get('agg_tok_s')} err={rec.get('client_error')}")
        run["protocol_s"] = round(time.time() - t_proto, 1)
        ok = [c for c in run["calls"] if c.get("finish_reason") and not c.get("client_error")]
        if not ok:
            run["probes_status"] = "no_real_generation"
            return orderly(work, run, "protocol_no_generation")

        # 7) P2 if budget remains (leave 300s margin for shutdown/export).
        prompts = json.loads(Path(args.prompts).read_text()) if args.prompts else {}
        for qk in sorted(prompts):
            for effort, seed in (("xhigh", P.SEED_XHIGH), ("low", P.SEED_LOW)):
                if time.time() - t0 > args.session_seconds - 300:
                    run["calls"].append({"query": qk, "effort": effort, "status": "session_margin"})
                    break
                rec = {}
                res = P.chat_once([{"role": "user", "content": prompts[qk]}],
                                  effort, seed, P2_MAX_TOKENS, CALL_TIMEOUT_S, rec)
                rec["phase"] = "p2"
                rec["query"] = qk
                rec["grid"] = P.extract_final_grid(res["content"], res["finish"])
                rec["answered"] = rec["grid"] is not None
                (work / f"reply-{qk}-{effort}.json").write_text(json.dumps(
                    {"request": rec.get("request"), "finish_reason": rec.get("finish_reason"),
                     "usage": rec.get("usage"), "wall_s": rec.get("wall_s"),
                     "ttft_s": rec.get("ttft_s"), "agg_tok_s": rec.get("agg_tok_s"),
                     "answered": rec["answered"], "grid": rec["grid"],
                     "content": res["content"][-4000:],
                     "reasoning": res["reasoning"][:2000]}, indent=1))
                run["calls"].append(rec)
                log(f"p2-{qk}-{effort}: finish={rec.get('finish_reason')} wall={rec.get('wall_s')} "
                    f"tok/s={rec.get('agg_tok_s')} answered={rec['answered']} err={rec.get('client_error')}")
    finally:
        P.kill_server(proc, work)
    run["elapsed_s"] = round(time.time() - t0, 1)
    return orderly(work, run, "complete_with_evidence")


if __name__ == "__main__":
    raise SystemExit(main())
