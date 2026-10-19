#!/usr/bin/env python3
"""016 driver: C4/C8吞吐对照 -> R恢复15份空final -> B固定xhigh比较8K/16K。

同一服务(SERVICE_MAX_NUM_SEQS=8, TP4/64K/FP8, KV默认, 无MTP)，客户端并发显式传4/8。
整份清单一次交入chat_wave滚动队列，完成即落盘；顺序用request_id还原。
只从完成final解析网格；不把思考矩形当答案。未运行即记未运行，不宣称提分。

冻结输入: P6六题固定顺序；题面SHA见run.json；恢复用末16000字符、seed3101、
enable_thinking=False、每次4096；B用seed4101、B8=6144/8192、B16=14336/16384。
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

VLLM_WHEEL_NAME = "vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl"
VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
INSTALL_BUDGET_S = 20 * 60
SAVE_MARGIN_S = 300
NEW_REQUEST_CUTOFF_S = 110 * 60
HARD_STOP_S = 115 * 60

P6_KEYS = ["291dc1e1_0", "4c416de3_0", "58c02a16_0", "aa4ec2a5_0", "db118e2a_0", "fc4aaf52_0"]
P2_KEYS = ["291dc1e1_0", "db118e2a_0"]
P6_EXTRA = [k for k in P6_KEYS if k not in P2_KEYS]
PROMPT_SHA = {
    "p2": "9f0c23dc53994fc21a937ebfc1ea51d11af4377281ed07fa44bdcb723854dd83",
    "p6extra": "76f3b6324e4ee28bebc12b63f2e0d6834e6c65883beb0090892c1f7940eeb572",
}

C_MAX_TOKENS = 512
C_SEED_BASE = 5100
R_MAX_TOKENS = 4096
R_SEED = 3101
R_TAIL_CHARS = 16000
B_SEED = 4101
B8_MAX = 8192
B8_THINK = 6144
B16_MAX = 16384
B16_THINK = 14336
WARM_MAX = 16
PROBE_MAX = 256
PROBE_BUDGET = 8

STATE = {"proc": None, "work": None, "run": None, "closed": False}
GPU_SAMPLER_STOP = threading.Event()


def log(msg):
    print(f"[016 {time.strftime('%H:%M:%S')}] {msg}", flush=True)


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


def parse_running(log_text: str) -> dict:
    """从engine日志行提取Running/Waiting的最大值与样本数。
    016§82/§99要求确认真实八路同时生成区间；若C8阶段从未Running>=8，
    记C8饱和度不足，不得据此判定八路无收益。CPU-only日志解析，不碰GPU。"""
    import re
    max_run, max_wait, n = 0, 0, 0
    for m in re.finditer(r"Running:\s*(\d+)\s*reqs,\s*Waiting:\s*(\d+)\s*reqs", log_text):
        try:
            r, w = int(m.group(1)), int(m.group(2))
        except Exception:
            continue
        n += 1
        if r > max_run:
            max_run = r
        if w > max_wait:
            max_wait = w
    return {"max_running": max_run, "max_waiting": max_wait, "samples": n}


def kv_estimate(log_text: str) -> str:
    outs = []
    for ln in log_text.splitlines():
        ll = ln.lower()
        if "kv cache" in ll or "kv_cache" in ll or "gpu kv" in ll or "available kv" in ll:
            outs.append(ln.strip()[:300])
    return "\n".join(outs[-8:])


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
        "content_chars", "reasoning_chars", "status", "events")}
    (directory / f"{stem}.json").write_text(json.dumps(summary, indent=1))
    return rec


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


def find_reasoning(recovery_dir: Path, stem: str) -> Path | None:
    for name in (f"{stem}.reasoning.txt", f"{stem}.tail.txt"):
        cand = recovery_dir / name
        if cand.exists():
            return cand
        hits = glob.glob(str(recovery_dir / "**" / name), recursive=True)
        if hits:
            return Path(hits[0])
    return None


def build_c_list(prompts: dict) -> list[str]:
    ordered = sorted(P6_KEYS, key=lambda k: len(prompts[k]), reverse=True)
    seq = (ordered * 3)[:16]
    # 按计划：完整重复两轮，再取前四份
    expect = (ordered + ordered + ordered[:4])[:16]
    assert seq == expect, "C清单构造必须为两轮+前四"
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--prompts", required=True)
    ap.add_argument("--p6-prompts", required=True)
    ap.add_argument("--recovery-dir", default=None,
                    help="含15份*.reasoning.txt的目录（可递归）；Kaggle挂载的恢复资产")
    ap.add_argument("--pip-python", default=None)
    ap.add_argument("--session-seconds", type=float, default=7200.0)
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
                       "effort": "xhigh", "c_max": C_MAX_TOKENS, "r_max": R_MAX_TOKENS,
                       "b8": {"max": B8_MAX, "think": B8_THINK},
                       "b16": {"max": B16_MAX, "think": B16_THINK},
                       "seeds": {"c_base": C_SEED_BASE, "r": R_SEED, "b": B_SEED},
                       "vllm_wheel": VLLM_WHEEL_NAME, "vllm_wheel_sha256": VLLM_WHEEL_SHA256,
                       "p6_order": P6_KEYS},
           "budgets": {"session_s": args.session_seconds, "save_margin_s": SAVE_MARGIN_S,
                       "new_cutoff_s": NEW_REQUEST_CUTOFF_S, "hard_stop_s": HARD_STOP_S,
                       "main_new_output_cap": 225280, "aux_cap": 512},
           "denominator": {"P6": P6_KEYS}, "calls": [], "_t0": t0_wall}
    STATE["work"] = work
    STATE["run"] = run
    signal.signal(signal.SIGTERM, on_term)
    P.record_versions(run)
    log(f"env: {json.dumps(run.get('host', {}))[:400]}")

    prompts = json.loads(Path(args.prompts).read_text())
    p6p = json.loads(Path(args.p6_prompts).read_text())
    merged = dict(prompts)
    merged.update(p6p)
    missing = [k for k in P6_KEYS if k not in merged]
    if missing:
        return finish("prompts_missing", detail=str(missing))
    run["prompt_sha"] = {"p2": sha256_file(Path(args.prompts)), "p6extra": sha256_file(Path(args.p6_prompts))}
    if run["prompt_sha"]["p2"] != PROMPT_SHA["p2"] or run["prompt_sha"]["p6extra"] != PROMPT_SHA["p6extra"]:
        run["prompt_sha_warn"] = "输入SHA与016冻结值不一致，继续但标记不可比"
        log(f"WARN prompt sha mismatch {run['prompt_sha']}")
    else:
        log("prompt sha match 016 frozen")

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

    # 模板两档渲染核对（沿用015成功路径）
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

    # GPU采样线程（约5秒）
    th = threading.Thread(target=gpu_sampler, args=(work, 5.0), daemon=True)
    th.start()

    # ---- 统一预热 + 预算小探针（合计<=512新token，单独计时） ----
    c_order = build_c_list(merged)
    run["c_order"] = c_order
    run["c_order_note"] = "六题按输入字符长度降序，两轮+前四；实际模板后token以usage prompt_tokens为准"
    warm_specs = []
    for i, q in enumerate(c_order):
        warm_specs.append({"messages": [{"role": "user", "content": merged[q]}],
                           "effort": "xhigh", "seed": C_SEED_BASE + i, "max_tokens": WARM_MAX,
                           "timeout_s": 300.0, "request_id": f"W-{i:02d}-{q}",
                           "query": q, "enable_thinking": True})
    # 预算探针：用最长题面、小预算，验证触及预算后转final；不用hello冒充
    longest_q = c_order[0]
    probe_spec = {"messages": [{"role": "user", "content": merged[longest_q]}],
                  "effort": "xhigh", "seed": 9999, "max_tokens": PROBE_MAX,
                  "timeout_s": 300.0, "request_id": "PROBE-budget8",
                  "query": longest_q, "enable_thinking": True,
                  "thinking_token_budget": PROBE_BUDGET}
    t_aux0 = time.monotonic()
    aux_recs = []

    def _save_aux(i, spec, slot):
        rec, res = slot["rec"], slot["res"]
        rec["phase"] = "warmup"
        rec["query"] = spec.get("query")
        rec["arm"] = "warmup"
        stem = spec.get("request_id", f"warm-{i}")
        save_reply(work / "warmup", stem, rec, res)
        run["calls"].append(rec)

    if time_left(session_deadline) <= 1 or past_cutoff(t0_mono):
        run["warmup"] = {"status": "skipped_cutoff"}
    else:
        limit = min(time.monotonic() + 600, session_deadline - SAVE_MARGIN_S)
        for s in warm_specs:
            s["deadline"] = limit
            s["timeout_s"] = max(1.0, limit - time.monotonic())
        w = P.chat_wave(warm_specs, client_concurrency=8, on_slot_done=_save_aux)
        run["warmup"] = {"n": len(warm_specs), "wall_s": w["wave_wall_s"],
                         "completion_tokens_sum": w["completion_tokens_sum"],
                         "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"],
                         "client": 8, "each_max": WARM_MAX}
        aux_recs.extend(w["calls"])
        save(work, run)
    # 预算探针单发
    probe_rec, probe_res = {}, {}
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        limit = min(time.monotonic() + 300, session_deadline - SAVE_MARGIN_S)
        probe_spec["deadline"] = limit
        probe_spec["timeout_s"] = max(1.0, limit - time.monotonic())
        probe_rec = {}
        probe_res = P.chat_once(probe_spec["messages"], probe_spec["effort"], probe_spec["seed"],
                                probe_spec["max_tokens"], probe_spec["timeout_s"], probe_rec,
                                deadline=probe_spec["deadline"],
                                thinking_token_budget=probe_spec["thinking_token_budget"],
                                enable_thinking=True, request_id=probe_spec["request_id"])
        probe_rec["phase"] = "budget_probe"
        probe_rec["query"] = longest_q
        probe_rec["arm"] = "budget_probe"
        save_reply(work / "warmup", "PROBE-budget8", probe_rec, probe_res)
        run["calls"].append(probe_rec)
        # 生效判断：字段被接受且出现final正文（自然hello不算）；不作强结论
        sent = (probe_rec.get("request") or {}).get("thinking_token_budget_sent")
        has_final = bool((probe_res.get("content") or "").strip())
        run["budget_probe"] = {"sent": sent, "finish": probe_rec.get("finish_reason"),
                               "usage": probe_rec.get("usage"),
                               "reasoning_chars": probe_rec.get("reasoning_chars"),
                               "content_chars": probe_rec.get("content_chars"),
                               "has_final": has_final,
                               "effective_hint": bool(sent == PROBE_BUDGET and has_final
                                                      and probe_rec.get("usage_present")),
                               "note": "hello自然结束不能证明生效；此处用最长题面+预算8观察是否转final"}
        save(work, run)
    run["aux_wall_s"] = round(time.monotonic() - t_aux0, 1)
    run["aux_new_tokens"] = sum(((s["rec"].get("usage") or {}).get("completion_tokens") or 0)
                                for s in aux_recs) + ((probe_rec.get("usage") or {}).get("completion_tokens") or 0)
    save(work, run)
    budget_ok = run.get("budget_probe", {}).get("effective_hint", False)
    if not budget_ok:
        run["budget_probe_warn"] = "小预算探针未给出明确转final证据；B阶段标记待查，不静默去掉预算烧满清单"
        log("WARN budget probe inconclusive; B will be gated")
        save(work, run)

    # ---- C: 同一清单C4/C8滚动对照 ----
    def run_c(client: int, tag: str):
        specs = []
        limit = min(time.monotonic() + 1800, session_deadline - SAVE_MARGIN_S)
        for i, q in enumerate(c_order):
            specs.append({"messages": [{"role": "user", "content": merged[q]}],
                          "effort": "xhigh", "seed": C_SEED_BASE + i, "max_tokens": C_MAX_TOKENS,
                          "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                          "request_id": f"C-{i:02d}-{q}", "query": q,
                          "enable_thinking": True})
        for s in specs:
            s["timeout_s"] = max(1.0, limit - time.monotonic())
            s["deadline"] = limit
        log_tail_before = server_tail(work, 200)
        log_before = ((work / "vllm-server.log").read_text(errors="replace")
                      if (work / "vllm-server.log").exists() else "")
        pre_before = count_preempt(log_before)
        t1 = time.monotonic()

        def _cb(i, spec, slot):
            rec, res = slot["rec"], slot["res"]
            rec["phase"] = tag
            rec["query"] = spec.get("query")
            rec["arm"] = f"C{client}"
            save_reply(work / tag, spec.get("request_id"), rec, res)
            run["calls"].append(rec)

        w = P.chat_wave(specs, client_concurrency=client, on_slot_done=_cb)
        wall = time.monotonic() - t1
        full = (work / "vllm-server.log").read_text(errors="replace") if (work / "vllm-server.log").exists() else ""
        # 每请求TTFT/延迟/usage/finish；聚合不用每请求tok/s相加
        per = []
        oom_hit = False
        for slot, spec in zip(w["calls"], specs):
            rec = slot["rec"]
            per.append({"request_id": spec["request_id"], "query": spec["query"],
                        "finish": rec.get("finish_reason"), "usage": rec.get("usage"),
                        "wall_s": rec.get("wall_s"), "ttft_s": rec.get("ttft_s"),
                        "agg_tok_s": rec.get("agg_tok_s"), "client_error": rec.get("client_error"),
                        "events": rec.get("events")})
            ce = (rec.get("client_error") or "").lower()
            if "oom" in ce or "out of memory" in ce:
                oom_hit = True
        if not server_alive() or "outofmemory" in server_tail(work, 50).lower().replace(" ", ""):
            oom_hit = True
        pre_after = count_preempt(full)
        phase_delta = full[len(log_before):] if len(full) >= len(log_before) else full
        running = parse_running(phase_delta)
        saturated = (running["max_running"] >= client) if running["samples"] else False
        summary = {"phase": tag, "client": client, "n": len(specs), "each_max": C_MAX_TOKENS,
                   "wall_s": round(wall, 2), "wave_wall_s": w["wave_wall_s"],
                   "completion_tokens_sum": w["completion_tokens_sum"],
                   "usage_missing": w["usage_missing"], "agg_tok_s": w["agg_tok_s"],
                   "per_request": per, "oom_hit": oom_hit,
                   "preempt_delta": pre_after - pre_before,
                   "max_running": running["max_running"], "max_waiting": running["max_waiting"],
                   "running_samples": running["samples"], "saturated": saturated,
                   "log_tail": log_tail_before[-500:],
                   "complete": w["usage_missing"] == 0 and len(w["calls"]) == len(specs)}
        (work / f"{tag}-summary.json").write_text(json.dumps(summary, indent=1))
        run[tag] = {k: summary[k] for k in ("phase", "client", "n", "wall_s", "completion_tokens_sum",
                                            "usage_missing", "agg_tok_s", "oom_hit",
                                            "preempt_delta", "max_running", "max_waiting",
                                            "running_samples", "saturated", "complete")}
        save(work, run)
        return summary

    c4 = c8 = None
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        c4 = run_c(4, "C4")
    else:
        run["C4"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        c8 = run_c(8, "C8")
    else:
        run["C8"] = {"status": "skipped_cutoff_or_dead"}
        save(work, run)

    # C决策：S=T4/T8；>=1.20暂用C8，否则C4；OOM/持续重算停C8
    selected = 4
    decision = "default_C4"
    if c4 and c8 and c4.get("complete") and c8.get("complete"):
        t4 = c4.get("wall_s") or 0
        t8 = c8.get("wall_s") or 0
        s = (t4 / t8) if t8 > 0 else 0
        run["C_compare"] = {"T4": t4, "T8": t8, "S": round(s, 3),
                            "tok4": c4.get("completion_tokens_sum"), "tok8": c8.get("completion_tokens_sum"),
                            "agg4": c4.get("agg_tok_s"), "agg8": c8.get("agg_tok_s"),
                            "oom8": c8.get("oom_hit"), "preempt8": c8.get("preempt_delta"),
                            "max_run4": c4.get("max_running"), "max_run8": c8.get("max_running"),
                            "wait8": c8.get("max_waiting")}
        if not c8.get("saturated"):
            decision = "C8_unsaturated_use_C4"
            selected = 4
        elif c8.get("oom_hit") or (c8.get("preempt_delta", 0) > 0 and (c8.get("agg_tok_s") or 0) < (c4.get("agg_tok_s") or 0) * 0.8):
            decision = "C8_oom_or_recompute_use_C4"
            selected = 4
        elif s >= 1.20 and not c8.get("oom_hit"):
            decision = "S>=1.20_use_C8"
            selected = 8
        elif 1.15 <= s < 1.20:
            decision = "boundary_use_C4_stop_search"
            selected = 4
        else:
            decision = "S<1.15_use_C4_stop_search"
            selected = 4
        run["C_compare"]["decision"] = decision
        run["C_compare"]["selected_client"] = selected
    else:
        run["C_compare"] = {"decision": decision, "selected_client": selected,
                            "note": "某臂不完整/跳过，无法可靠比较，用C4完成解题研究"}
    run["selected_client"] = selected
    log(f"C decision {decision} selected={selected}")
    save(work, run)

    # ---- R: 恢复15份空final ----
    # 固定排序：query(P6序)、原arm(high先)、原seed
    r_items = []
    if args.recovery_dir:
        rdir = Path(args.recovery_dir)
        # 原 stems：高臂5 + 低臂10（58高命中与db低两命中除外）
        high_stems = [f"{q}-T-high-s1001" for q in P6_KEYS
                      if not (q == "58c02a16_0")]
        # db高虽stop但final空，仍在恢复集；上式已含db（58除外），共5：291,4c,aa,db,fc
        low_stems = []
        for q in P6_KEYS:
            if q == "db118e2a_0":
                continue
            low_stems += [f"{q}-T-low-s2001", f"{q}-T-low-s2002"]
        for stem in high_stems + low_stems:
            # query从stem前缀解析
            q = stem.split("-T-")[0]
            arm = "T-high" if "-T-high-" in stem else "T-low"
            try:
                seed = int(stem.rsplit("-s", 1)[1])
            except Exception:
                seed = 0
            r_items.append((q, arm, seed, stem))
        r_items.sort(key=lambda t: (P6_KEYS.index(t[0]), 0 if t[1] == "T-high" else 1, t[2]))
        run["R_plan"] = {"n_planned": 15, "order": [s for _, _, _, s in r_items],
                         "seeds": R_SEED, "each_max": R_MAX_TOKENS, "tail_chars": R_TAIL_CHARS}
    else:
        run["R_plan"] = {"status": "no_recovery_dir_no_run"}
    save(work, run)

    if r_items and time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        r_specs = []
        limit = min(time.monotonic() + 2400, session_deadline - SAVE_MARGIN_S)
        skipped = []
        for idx, (q, arm, seed, stem) in enumerate(r_items):
            rp = find_reasoning(Path(args.recovery_dir), stem)
            if rp is None:
                skipped.append(stem)
                continue
            tail = rp.read_text(errors="replace")[-R_TAIL_CHARS:]
            prompt = (merged[q] + "\n\n以下可能是有误的工作笔记，仅供参考，不要直接照抄：\n"
                      + tail + "\n\n请只输出最终网格（单个闭合代码围栏，空格分隔数字，每行一行）。")
            r_specs.append({"messages": [{"role": "user", "content": prompt}],
                            "effort": None, "seed": R_SEED, "max_tokens": R_MAX_TOKENS,
                            "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                            "request_id": f"R-{idx:02d}-{stem}", "query": q,
                            "orig_arm": arm, "orig_seed": seed, "orig_stem": stem,
                            "enable_thinking": False, "thinking_token_budget": None})
        run["R_plan"]["missing_files"] = skipped
        run["R_plan"]["n_specs"] = len(r_specs)

        def _rcb(i, spec, slot):
            rec, res = slot["rec"], slot["res"]
            rec["phase"] = "R"
            rec["query"] = spec.get("query")
            rec["arm"] = "R"
            rec["orig_arm"] = spec.get("orig_arm")
            rec["orig_seed"] = spec.get("orig_seed")
            rec["orig_stem"] = spec.get("orig_stem")
            save_reply(work / "R", spec.get("request_id"), rec, res)
            run["calls"].append(rec)

        # 同一滚动队列一次交入，不按答案筛选、不追加第二次
        if not r_specs:
            run["R"] = {"status": "no_specs_all_missing", "n": 0, "client": selected,
                        "complete": False}
            (work / "R-summary.json").write_text(json.dumps(
                {"status": "no_specs_all_missing", "n": 0,
                 "missing": skipped}, indent=1))
            save(work, run)
        else:
            rw = P.chat_wave(r_specs, client_concurrency=selected, on_slot_done=_rcb)
            run["R"] = {"n": len(r_specs), "wall_s": rw["wave_wall_s"],
                        "completion_tokens_sum": rw["completion_tokens_sum"],
                        "usage_missing": rw["usage_missing"], "agg_tok_s": rw["agg_tok_s"],
                        "client": selected,
                        "complete": len(r_specs) > 0 and rw["usage_missing"] == 0}
            (work / "R-summary.json").write_text(json.dumps(
                {"n": len(r_specs), "wall_s": rw["wave_wall_s"],
                 "completion_tokens_sum": rw["completion_tokens_sum"],
                 "usage_missing": rw["usage_missing"], "agg_tok_s": rw["agg_tok_s"],
                 "client": selected,
                 "items": [{"request_id": s["request_id"], "query": s["query"],
                            "orig_stem": s["orig_stem"]} for s in r_specs]}, indent=1))
            save(work, run)
    else:
        if r_items:
            run["R"] = {"status": "skipped_cutoff_or_dead_or_no_time"}
            save(work, run)
            log("R skipped (cutoff/dead)")

    # ---- B: 固定xhigh，P6比较8K/16K；12条混合同一队列 ----
    b_specs = []
    if time_left(session_deadline) > 1 and not past_cutoff(t0_mono) and server_alive():
        if not budget_ok:
            run["B_gate"] = {"decision": "blocked", "reason": "budget_probe_inconclusive_no_silent_unbudgeted_run"}
            save(work, run)
            log("B blocked: budget probe inconclusive")
        else:
            limit = min(time.monotonic() + 5400, session_deadline - SAVE_MARGIN_S)
            for idx, q in enumerate(P6_KEYS):
                pair = [("B8", B8_MAX, B8_THINK), ("B16", B16_MAX, B16_THINK)]
                if idx % 2 == 1:
                    pair = list(reversed(pair))
                for arm, mx, th in pair:
                    b_specs.append({"messages": [{"role": "user", "content": merged[q]}],
                                    "effort": "xhigh", "seed": B_SEED, "max_tokens": mx,
                                    "timeout_s": max(1.0, limit - time.monotonic()), "deadline": limit,
                                    "request_id": f"B-{arm}-{q}", "query": q, "arm": arm,
                                    "enable_thinking": True, "thinking_token_budget": th})
            run["B_plan"] = {"n": len(b_specs), "order": [s["request_id"] for s in b_specs],
                             "client": selected, "seed": B_SEED,
                             "note": "奇题B8先、偶题B16先；两臂共享队列，尾部降并发时间另记"}

            def _bcb(i, spec, slot):
                rec, res = slot["rec"], slot["res"]
                rec["phase"] = "B"
                rec["query"] = spec.get("query")
                rec["arm"] = spec.get("arm")
                save_reply(work / "B", spec.get("request_id"), rec, res)
                run["calls"].append(rec)

            bw = P.chat_wave(b_specs, client_concurrency=selected, on_slot_done=_bcb)
            # 每臂EM@1与固定双槽EM@2需本地评分（答案在本地）；此处只记完成/合法
            b_valid = {"B8": 0, "B16": 0}
            b_done = {"B8": 0, "B16": 0}
            for slot, spec in zip(bw["calls"], b_specs):
                rec = slot["rec"]
                if rec.get("usage_present"):
                    b_done[spec["arm"]] += 1
                if rec.get("valid_answer"):
                    b_valid[spec["arm"]] += 1
            run["B"] = {"n": len(b_specs), "wall_s": bw["wave_wall_s"],
                        "completion_tokens_sum": bw["completion_tokens_sum"],
                        "usage_missing": bw["usage_missing"], "agg_tok_s": bw["agg_tok_s"],
                        "client": selected, "done_per_arm": b_done, "valid_per_arm": b_valid,
                        "note": "正确性EM需本地评分；此处valid仅为严格围栏合法率"}
            (work / "B-summary.json").write_text(json.dumps(
                {"n": len(b_specs), "wall_s": bw["wave_wall_s"],
                 "completion_tokens_sum": bw["completion_tokens_sum"],
                 "usage_missing": bw["usage_missing"], "agg_tok_s": bw["agg_tok_s"],
                 "client": selected, "done_per_arm": b_done, "valid_per_arm": b_valid,
                 "order": [s["request_id"] for s in b_specs]}, indent=1))
            save(work, run)
    else:
        run["B"] = {"status": "skipped_cutoff_or_dead_or_no_time"}
        save(work, run)

    # 会话总账
    full = (work / "vllm-server.log").read_text(errors="replace") if (work / "vllm-server.log").exists() else ""
    run["kv_estimate_at_end"] = kv_estimate(full)[-2000:]
    run["preempt_total"] = count_preempt(full)
    run["new_output_tokens_main"] = sum([
        (run.get("C4") or {}).get("completion_tokens_sum", 0),
        (run.get("C8") or {}).get("completion_tokens_sum", 0),
        (run.get("R") or {}).get("completion_tokens_sum", 0),
        (run.get("B") or {}).get("completion_tokens_sum", 0)])
    save(work, run)
    # 有限收尾：不再发起新工作
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
