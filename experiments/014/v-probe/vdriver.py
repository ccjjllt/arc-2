#!/usr/bin/env python3
"""015 driver: install the pinned vLLM wheel, recheck the 27B server, then T.

One server (TP4/64K/C4/FP8, KV default, no MTP). Known negative outcomes exit 0
with status in run.json. Unexpected errors exit 2. Full reasoning and final text
are separate files; run.json keeps the summary.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from pathlib import Path

import run_reasoning_probe as P

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
PROTOCOL_BUDGET_S = 15 * 60
HELLO_MAX_TOKENS = 4096
WAVE_MAX_TOKENS = 256
WAVE_N = 4
T_HIGH_MAX = 32768
T_LOW_MAX = 16384
T_HIGH_CAP_S = 75 * 60
T_LOW_CAP_S = 40 * 60
SAVE_MARGIN_S = 180
P2_KEYS = ["291dc1e1_0", "db118e2a_0"]
P6_KEYS = ["291dc1e1_0", "4c416de3_0", "58c02a16_0", "aa4ec2a5_0", "db118e2a_0", "fc4aaf52_0"]
P6_EXTRA = [k for k in P6_KEYS if k not in P2_KEYS]

STATE = {"proc": None, "work": None, "run": None, "closed": False}


def log(msg):
    print(f"[vdriver {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def sh(cmd, timeout, cwd=None, env=None):
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save(work: Path, run: dict) -> None:
    (work / "run.json").write_text(json.dumps(run, indent=1))


def orderly(work, run, status, code=0):
    run["status"] = status
    started = run.pop("_t0", None)
    run["elapsed_s"] = round(time.time() - (started if started is not None else time.time()), 1)
    if "serve_tail" not in run:
        run["serve_tail"] = server_tail(work, 30)
    # Drop the isolated site. It is multi-GB and rebuilds from the wheelhouse.
    import shutil
    shutil.rmtree(work / "site", ignore_errors=True)
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
        return code
    return orderly(work, run, status, code)


def on_term(signum, _frame):
    finish(f"signal_{signum}")
    os._exit(0)


def _pip_base(use_fallback, pip_python):
    if use_fallback:
        return [pip_python, "-m", "pip", "install", "--python-version", "3.12",
                "--only-binary", ":all:", "--no-index"]
    return [sys.executable, "-m", "pip", "install", "--no-index"]


def install_requirements_cmd(use_fallback, pip_python, wheels, req, site):
    return _pip_base(use_fallback, pip_python) + [
        "--find-links", str(wheels), "--requirement", str(req), "--target", str(site)]


def install_vllm_cmd(use_fallback, pip_python, wheels, wheel_path, site):
    # The wheelhouse requirements file has torch/torchvision and no torchaudio.
    # vLLM 0.27.2 declares torchaudio==2.11.0, so one resolver pass finds no match.
    # Install that wheel with --no-deps after the curated runtime set.
    return _pip_base(use_fallback, pip_python) + [
        "--no-deps", "--find-links", str(wheels), str(wheel_path), "--target", str(site)]


def save_reply(directory: Path, stem: str, rec: dict, res: dict) -> dict:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{stem}.reasoning.txt").write_text(res.get("reasoning") or "")
    (directory / f"{stem}.final.txt").write_text(res.get("content") or "")
    flags = P.classify_generation(rec, res.get("content") or "", res.get("finish"))
    rec.update({k: flags[k] for k in (
        "generation_observed", "protocol_complete", "valid_answer", "usage_present")})
    rec["grid"] = flags["grid"]
    rec["answered"] = flags["valid_answer"]
    summary = {k: rec.get(k) for k in (
        "phase", "query", "arm", "request", "finish_reason", "usage", "wall_s", "ttft_s",
        "agg_tok_s", "client_error", "cancelled_by_client", "generation_observed",
        "protocol_complete", "valid_answer", "usage_present", "answered", "grid",
        "content_chars", "reasoning_chars", "status", "literal_probe")}
    (directory / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    return rec


def server_alive() -> bool:
    proc = STATE.get("proc")
    return proc is not None and proc.poll() is None


def server_tail(work: Path, n=40) -> str:
    path = work / "vllm-server.log"
    if not path.exists():
        return ""
    return "\n".join(path.read_text(errors="replace").splitlines()[-n:])


def run_specs(work, run, phase, arm, specs, directory):
    """Run specs in chunks of MAX_NUM_SEQS. Returns call records."""
    records = []
    if not specs:
        return records
    wave = P.chat_wave(specs)
    run.setdefault("waves", []).append({
        "phase": phase, "arm": arm, "concurrency": wave["concurrency"],
        "wave_wall_s": wave["wave_wall_s"], "completion_tokens_sum": wave["completion_tokens_sum"],
        "usage_missing": wave["usage_missing"], "agg_tok_s": wave["agg_tok_s"], "n": len(specs)})
    for spec, slot in zip(specs, wave["calls"]):
        rec, res = slot["rec"], slot["res"]
        rec["phase"] = phase
        rec["arm"] = arm
        rec["query"] = spec.get("query")
        stem = spec["stem"]
        save_reply(directory, stem, rec, res)
        records.append(rec)
        run["calls"].append(rec)
        log(f"{phase}-{stem}: finish={rec.get('finish_reason')} wall={rec.get('wall_s')} "
            f"tok/s={rec.get('agg_tok_s')} valid={rec.get('valid_answer')} err={rec.get('client_error')}")
    save(work, run)
    return records


def slots_for(records, key):
    high = [r for r in records if r.get("query") == key and r.get("arm") == "T-high"]
    low = [r for r in records if r.get("query") == key and r.get("arm") == "T-low"]
    low = sorted(low, key=lambda r: (r.get("request") or {}).get("seed") or 0)
    high_grid = high[0].get("grid") if high else None
    return {
        "T-high": [high_grid, high_grid],
        "T-high_duplicated_second_slot": True,
        "T-low": [low[0].get("grid") if len(low) > 0 else None,
                  low[1].get("grid") if len(low) > 1 else None],
    }


def cohort_summary(keys, records):
    rows = []
    for key in keys:
        mine = [r for r in records if r.get("query") == key]
        rows.append({"query": key, "n_calls": len(mine), "slots": slots_for(records, key),
                     "calls": [{k: r.get(k) for k in (
                         "arm", "finish_reason", "usage", "wall_s", "ttft_s", "agg_tok_s",
                         "client_error", "generation_observed", "protocol_complete",
                         "valid_answer", "usage_present", "grid")} for r in mine]})
    return rows


def planned_complete(keys, records) -> bool:
    for key in keys:
        high = [r for r in records if r.get("query") == key and r.get("arm") == "T-high"]
        low = [r for r in records if r.get("query") == key and r.get("arm") == "T-low"]
        if len(high) != 1 or len(low) != 2:
            return False
        if any(r.get("status") == "session_margin" for r in high + low):
            return False
    return True


def make_specs(keys, prompts, arm, session_deadline):
    specs = []
    if arm == "T-high":
        effort, seeds, max_tokens, cap = "xhigh", (P.SEED_XHIGH,), T_HIGH_MAX, T_HIGH_CAP_S
    else:
        effort, seeds, max_tokens, cap = "low", (P.SEED_LOW, P.SEED_LOW_B), T_LOW_MAX, T_LOW_CAP_S
    limit = min(time.monotonic() + cap, session_deadline - SAVE_MARGIN_S)
    if limit <= time.monotonic() + 1:
        return []
    timeout_s = max(1.0, limit - time.monotonic())
    for key in keys:
        for seed in seeds:
            specs.append({
                "query": key, "messages": [{"role": "user", "content": prompts[key]}],
                "effort": effort, "seed": seed, "max_tokens": max_tokens,
                "timeout_s": timeout_s, "deadline": limit,
                "stem": f"{key}-{arm}-s{seed}"})
    return specs


def run_t_cohort(work, run, label, keys, prompts, session_deadline):
    directory = work / label
    records = []
    t0 = time.monotonic()
    for arm, width in (("T-high", 2), ("T-low", P.MAX_NUM_SEQS)):
        specs = make_specs(keys, prompts, arm, session_deadline)
        if not specs:
            for key in keys:
                run["calls"].append({"phase": label, "query": key, "arm": arm, "status": "session_margin"})
            break
        # Chunk so a 32K arm stays at 2-way and a 16K arm fills C4.
        for start in range(0, len(specs), width):
            if not server_alive():
                run["server_died_during"] = label
                records.extend(run_specs(work, run, label, arm, [], directory))
                break
            chunk = specs[start:start + width]
            # Refresh the shared deadline so an earlier chunk cannot shrink the next one
            # below its own cap, and a late chunk cannot run past the session margin.
            limit = min(time.monotonic() + (T_HIGH_CAP_S if arm == "T-high" else T_LOW_CAP_S),
                        session_deadline - SAVE_MARGIN_S)
            if limit <= time.monotonic() + 1:
                for spec in chunk:
                    run["calls"].append({"phase": label, "query": spec["query"], "arm": arm,
                                         "status": "session_margin"})
                break
            for spec in chunk:
                spec["deadline"] = limit
                spec["timeout_s"] = max(1.0, limit - time.monotonic())
            records.extend(run_specs(work, run, label, arm, chunk, directory))
        if not server_alive():
            break
    wall = time.monotonic() - t0
    tokens = 0
    missing = 0
    for rec in records:
        usage = rec.get("usage") or {}
        got = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if got is None:
            missing += 1
        else:
            tokens += got
    summary = {"label": label, "keys": keys, "wall_s": round(wall, 1),
               "completion_tokens": tokens, "usage_missing": missing,
               "complete": planned_complete(keys, records) and server_alive(),
               "queries": cohort_summary(keys, records)}
    run[label] = {k: summary[k] for k in ("label", "keys", "wall_s", "completion_tokens",
                                           "usage_missing", "complete")}
    (work / f"{label}-summary.json").write_text(json.dumps(summary, indent=1))
    save(work, run)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--p6-prompts", default=None)
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=20700.0)
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    session_deadline = time.monotonic() + args.session_seconds
    run = {"config": {"tp": P.TP_SIZE, "max_model_len": P.MAX_MODEL_LEN,
                       "max_num_seqs": P.MAX_NUM_SEQS, "gpu_mem_util": P.GPU_MEM_UTIL,
                       "kv_dtype": "default", "mtp": False, "sampling": dict(P.SAMPLING),
                       "hello_max_tokens": HELLO_MAX_TOKENS, "wave_n": WAVE_N,
                       "wave_max_tokens": WAVE_MAX_TOKENS,
                       "t_high": {"effort": "xhigh", "calls": 1, "max_tokens": T_HIGH_MAX,
                                  "seed": P.SEED_XHIGH, "concurrency": 2},
                       "t_low": {"effort": "low", "calls": 2, "max_tokens": T_LOW_MAX,
                                 "seeds": [P.SEED_LOW, P.SEED_LOW_B], "concurrency": P.MAX_NUM_SEQS},
                       "second_slot_rule": "T-high duplicates its only answer; T-low keeps seed order",
                       "vllm_wheel": VLLM_WHEEL_NAME, "vllm_wheel_sha256": VLLM_WHEEL_SHA256},
           "budgets": {"install_load_s": INSTALL_BUDGET_S, "protocol_s": PROTOCOL_BUDGET_S,
                       "session_s": args.session_seconds, "save_margin_s": SAVE_MARGIN_S},
           "denominator": {"P2": P2_KEYS, "P6": P6_KEYS, "P6_extra": P6_EXTRA},
           "calls": [], "_t0": t0}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    P.record_versions(run)
    log(f"env: {json.dumps(run.get('host', {}))[:400]}")

    prompts = json.loads(Path(args.prompts).read_text())
    missing = [k for k in P2_KEYS if k not in prompts]
    if missing:
        return finish("prompts_missing", detail=str(missing))
    p6_prompts = {}
    if args.p6_prompts:
        p6_prompts = json.loads(Path(args.p6_prompts).read_text())
        missing_p6 = [k for k in P6_EXTRA if k not in p6_prompts]
        if missing_p6:
            return finish("p6_prompts_missing", detail=str(missing_p6))

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
        log("no pip for 3.12; trying stdlib ensurepip ...")
        ens = sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
        (work / "ensurepip.log").write_text((ens.stdout + "\n---STDERR---\n" + ens.stderr)[-4000:])
        run["ensurepip_rc"] = ens.returncode
        pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    use_fallback = pip_probe.returncode != 0
    if use_fallback and not args.pip_python:
        return finish("no_pip_for_312_and_no_fallback")
    run["installer"] = "fallback_pip_python312_target" if use_fallback else "py312_pip"
    run["vllm_install"] = "requirements_then_vllm_no_deps"
    req_cmd = install_requirements_cmd(use_fallback, args.pip_python, wheels, req, site)
    vllm_cmd = install_vllm_cmd(use_fallback, args.pip_python, wheels, wheel_path, site)
    run["install_cmd"] = {"requirements": req_cmd, "vllm_no_deps": vllm_cmd}
    log("pip install runtime requirements, then vllm wheel --no-deps")
    try:
        ins = sh(req_cmd, timeout=INSTALL_BUDGET_S)
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    if ins.returncode:
        run["pip_rc"] = ins.returncode
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        vins = sh(vllm_cmd, timeout=600)
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    (work / "pip-vllm.log").write_text(vins.stdout[-4000:] + "\n---STDERR---\n" + vins.stderr[-4000:])
    if vins.returncode:
        run["vllm_pip_rc"] = vins.returncode
        return finish("vllm_wheel_install_failed", detail=(vins.stderr or vins.stdout)[-400:])
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    imp = sh([sys.executable, "-c",
              "import vllm, torch; print(vllm.__version__); print(vllm.__file__); print(torch.__version__)"],
             timeout=300, env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode,
                           "out": (imp.stdout.strip() + " " + imp.stderr.strip())[-1500:]}
    log("import vllm rc=%s out=%s" % (imp.returncode, run["import_check"]["out"][-400:]))
    if imp.returncode:
        return finish("import_vllm_failed", detail=run["import_check"]["out"][-400:])
    run["install_s"] = round(time.time() - t0, 1)

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

    # Absolute install+load deadline: 20 minutes from this process start.
    install_deadline = (session_deadline - args.session_seconds) + INSTALL_BUDGET_S
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    try:
        proc = P.start_server(model_root, work, extra=[], deadline=install_deadline)
    except Exception as exc:
        run["serve_tail"] = server_tail(work)
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    run["install_load_s"] = round(time.time() - t0, 1)
    log(f"server up, install_load_s={run['install_load_s']}")
    save(work, run)

    # Short protocol, serial, so each effort has its own wall clock.
    proto_deadline = min(time.monotonic() + PROTOCOL_BUDGET_S, session_deadline - SAVE_MARGIN_S)
    t_proto = time.time()
    for effort, seed in (("xhigh", P.SEED_XHIGH), ("low", P.SEED_LOW)):
        if time.monotonic() >= proto_deadline or not server_alive():
            run["calls"].append({"phase": "hello", "effort": effort, "status": "protocol_budget_exceeded"})
            continue
        rec = {}
        res = P.chat_once([{"role": "user", "content": "Return [[1]]."}],
                          effort, seed, HELLO_MAX_TOKENS, proto_deadline - time.monotonic(), rec,
                          deadline=proto_deadline)
        rec["phase"] = "hello"
        # The probe asks for the literal [[1]], not an ARC fence. A stopped
        # reply of that text is a complete probe answer. ARC scoring still
        # accepts only a closed 0-9 fence.
        rec["literal_probe"] = (
            rec.get("finish_reason") == "stop" and not rec.get("client_error")
            and (res.get("content") or "").strip() == "[[1]]")
        save_reply(work / "hello", f"hello-{effort}", rec, res)
        run["calls"].append(rec)
        log(f"hello-{effort}: finish={rec.get('finish_reason')} usage={rec.get('usage')} "
            f"valid={rec.get('valid_answer')} err={rec.get('client_error')}")
        save(work, run)
    run["protocol_s"] = round(time.time() - t_proto, 1)
    hellos = [c for c in run["calls"] if c.get("phase") == "hello" and c.get("request")]
    run["protocol_admitted"] = (
        len(hellos) == 2 and all(c.get("protocol_complete") and c.get("valid_answer")
                                 and c.get("usage_present") for c in hellos))
    both_generated = len(hellos) == 2 and all(c.get("generation_observed") for c in hellos)
    usage_ok = len(hellos) == 2 and all(c.get("usage_present") for c in hellos)
    any_valid = any(c.get("valid_answer") or c.get("literal_probe") for c in hellos)
    run["generation_admitted"] = bool(both_generated and usage_ok and any_valid and server_alive())

    # C4 is only a server cap until a real wave runs.
    if server_alive() and any(c.get("generation_observed") for c in hellos):
        limit = min(time.monotonic() + 240, session_deadline - SAVE_MARGIN_S)
        specs = []
        for i in range(WAVE_N):
            specs.append({
                "messages": [{"role": "user", "content": "Return [[1]]."}],
                "effort": "xhigh" if i % 2 == 0 else "low",
                "seed": 3001 + i, "max_tokens": WAVE_MAX_TOKENS,
                "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                "stem": f"wave-{i}"})
        wave_recs = run_specs(work, run, "wave", "probe", specs, work / "wave")
        run["wave_measured"] = run["waves"][-1] if run.get("waves") else None
        if not any(r.get("generation_observed") for r in wave_recs):
            run["wave_note"] = "concurrent wave produced no tokens"
    else:
        run["wave_note"] = "skipped"

    if not run["generation_admitted"]:
        run["serve_tail"] = server_tail(work)
        return finish("recheck_not_admitted", detail=json.dumps({
            "both_generated": both_generated, "usage_ok": usage_ok, "any_valid": any_valid,
            "server_alive": server_alive(), "protocol_admitted": run["protocol_admitted"]}))

    p2 = run_t_cohort(work, run, "P2", P2_KEYS, prompts, session_deadline)
    if (not p2["complete"]) or (not server_alive()) or p2["completion_tokens"] <= 0:
        run["p6_gate"] = {"decision": "not_run", "reason": "p2_incomplete_or_no_throughput",
                          "p2_complete": p2["complete"], "server_alive": server_alive()}
        return finish("complete_with_evidence")

    remain = session_deadline - time.monotonic()
    need = p2["wall_s"] * (len(P6_EXTRA) / len(P2_KEYS)) * 1.2 + 300
    run["p6_gate"] = {"remain_s": round(remain, 1), "need_s": round(need, 1),
                      "p2_wall_s": p2["wall_s"], "p2_completion_tokens": p2["completion_tokens"],
                      "rate_tok_s": round(p2["completion_tokens"] / p2["wall_s"], 3)}
    if remain < need or not p6_prompts:
        run["p6_gate"]["decision"] = "not_run"
        run["p6_gate"]["reason"] = "no_p6_prompts" if not p6_prompts else "measured_p2_cost_exceeds_remaining"
        return finish("complete_with_evidence")
    run["p6_gate"]["decision"] = "run"
    run["p6_gate"]["reason"] = "remaining_covers_estimated_extra_queries"
    run_t_cohort(work, run, "P6_extra", P6_EXTRA, p6_prompts, session_deadline)
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
