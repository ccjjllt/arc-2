"""K GDN inference probe (012 plan section 6, unexecuted/unrun here).

Basis (read-only, CPU authoring only):
- docs/plan/012-kernel-program-completion.md section 6 (6.1-6.4).
- experiments/011/gdn-followup-evidence.md (pinned wheels/SHA/static interfaces).

This file is a CPU-authored implementation only. It was NOT executed on GPU,
no model was loaded, and no inference was launched in this session.

Fixed identities (do not rescan combinations):
- Base: Qwen/Qwen3.8-27B, BF16, device_map="balanced", no CPU/disk offload.
- Task: db118e2a/query0, original F1 full prompt, nonthinking, base-only.
- Wheels: causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  SHA256 8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7;
  fla_core-0.5.2-py3-none-any.whl SHA256
  5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761;
  flash_linear_attention-0.5.2-py3-none-any.whl SHA256
  dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400.
- Static interfaces: causal_conv1d_fn/causal_conv1d_update (Dao-AILab v1.7.0),
  FLA chunk.py / fused_recurrent.py gated_delta_rule (v0.5.2),
  TF 5.8 modeling_qwen3_5.py chunk (multi-token) vs fused recurrent (1-token
  cached decode), torch_* fallback fns, norm fixed at construction time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# ---- Fixed constants (012 section 6.3 / 6.4 + 011 evidence) ----
TASK_ID = "db118e2a"
QUERY_ID = "query0"
N_CONTINUATION = 64
N_MIN_DIAGNOSTIC = 16
GDN_EXPECTED_LAYERS = 48

REFERENCE_RMS_TOL = 1e-4
DRIFT_FLOOR = 0.01
DRIFT_REPEAT_MULT = 5.0
GAIN_THRESHOLD = 0.15
RMS_EPS = 1e-12

VERDICT_UNSTABLE_REFERENCE = "unstable_reference"
VERDICT_NUMERICAL_DRIFT = "numerical_drift"
VERDICT_WORTH_BRIDGE = "worth_full_output_bridge"
VERDICT_INSUFFICIENT_GAIN = "insufficient_gain"

PHASE_BUDGETS = {
    "install_env": 120,
    "load": 600,
    "check_jit_warmup_reference": 600,
    "measure": 360,
    "save": 120,
}
TOTAL_BUDGET_S = 1800
TRIAL_ORDER = ("A1", "B1", "B2", "A2")

WHEEL_MANIFEST = (
    {"file": "causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
     "sha256": "8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7"},
    {"file": "fla_core-0.5.2-py3-none-any.whl",
     "sha256": "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761"},
    {"file": "flash_linear_attention-0.5.2-py3-none-any.whl",
     "sha256": "dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400"},
)

EXPECTED_STACK = {"torch": "2.10.0+cu128", "transformers": "5.8.0",
                  "peft": "0.19.1", "accelerate": "1.13.0", "triton": "3.6.0"}

SIX_FUNCTIONS = ("check_environment", "load_probe_session", "set_backend",
                 "make_reference_tokens", "run_trial", "compare_trials")


def emit(event, **fields):
    print("GRID_PROBE_EVENT " + json.dumps({"event": event,
          "monotonic": time.monotonic(), **fields}, ensure_ascii=False, default=str),
          flush=True)


def check_environment(*, strict_gpu=True):
    """Section 6.3 pre-checks. Lazy imports so CPU dry-run never touches CUDA pkgs."""
    import platform
    info = {"python": platform.python_version(), "wheels": list(WHEEL_MANIFEST)}
    import torch
    import transformers
    import accelerate
    import triton  # noqa: F401
    info.update({"torch": torch.__version__, "transformers": transformers.__version__,
                 "accelerate": accelerate.__version__,
                 "cuda_available": torch.cuda.is_available(),
                 "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                 "cxx11_abi": torch._C._GLIBCXX_USE_CXX11_ABI,
                 "cuda_runtime": torch.version.cuda})
    try:
        import einops  # noqa: F401
        info["einops"] = einops.__version__
    except Exception as exc:
        info["einops"] = "missing:%r" % exc
    # Official fast entry points must import; failure is fatal, no fallback reload.
    import causal_conv1d  # noqa: F401
    import fla.ops.gated_delta_rule.chunk  # noqa: F401
    import fla.ops.gated_delta_rule.fused_recurrent  # noqa: F401
    if strict_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
            raise RuntimeError("K probe requires 4 CUDA GPUs, got %s" % info)
        if torch._C._GLIBCXX_USE_CXX11_ABI is not True:
            raise RuntimeError("CXX11 ABI must be True for pinned cu12 cxx11abiTRUE wheel")
    emit("env_checked", **{k: str(v)[:160] for k, v in info.items()})
    return info


def load_probe_session(model_dir, *, deadline, journal_path, runtime_config=None):
    """Reuse grid_runtime.load_session; pin Qwen3_5RMSNormGated via try/finally."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid_runtime
    modeling = __import__("transformers.models.qwen3_5.modeling_qwen3_5", fromlist=["*"])
    saved = getattr(modeling, "FusedRMSNormGated", None)
    modeling.FusedRMSNormGated = None
    try:
        session = grid_runtime.load_session(model_dir, deadline=deadline,
                                            journal_path=journal_path,
                                            runtime_config=runtime_config or {})
    finally:
        modeling.FusedRMSNormGated = saved
    rows = grid_runtime.inspect_gdn_runtime(session.model)
    if len(rows) != GDN_EXPECTED_LAYERS:
        raise RuntimeError("expected 48 GDN layers, got %d" % len(rows))
    if any(r.get("norm") != "Qwen3_5RMSNormGated" for r in rows):
        raise RuntimeError("norm pin failed: %s" % [r.get("norm") for r in rows][:4])
    device_map = dict(getattr(session.model, "hf_device_map", {}))
    if any(str(v) in ("cpu", "disk") for v in device_map.values()):
        raise RuntimeError("unexpected offload: %s" % device_map)
    # Candidate cache must not affect timing.
    session.allow_cached = False
    emit("probe_loaded", gdn_layers=len(rows), device_map=device_map)
    return session


def set_backend(session, mode, *, refs):
    """Switch 48 live GDN attrs in place. No rebuild/copy/.to(); norm ptr unchanged.

    refs: {"torch_update","torch_chunk","torch_recurrent","fast_fn","fast_update",
           "fast_chunk","fast_recurrent"} captured once after official imports.
    mode "A": None + TF torch impls. mode "B": official fast fns.
    """
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid_runtime
    if mode not in ("A", "B"):
        raise ValueError("mode must be A or B")
    layers = [(n, m) for n, m in session.model.named_modules()
              if hasattr(m, "chunk_gated_delta_rule")]
    if len(layers) != GDN_EXPECTED_LAYERS:
        raise RuntimeError("expected 48 GDN layers, got %d" % len(layers))
    before = {}
    for name, mod in layers:
        norm = getattr(mod, "norm", None)
        w = getattr(norm, "weight", None)
        before[name] = w.data_ptr() if w is not None and hasattr(w, "data_ptr") else None
    for _name, mod in layers:
        if mode == "A":
            mod.causal_conv1d_fn = None
            mod.causal_conv1d_update = refs["torch_update"]
            mod.chunk_gated_delta_rule = refs["torch_chunk"]
            mod.recurrent_gated_delta_rule = refs["torch_recurrent"]
        else:
            mod.causal_conv1d_fn = refs["fast_fn"]
            mod.causal_conv1d_update = refs["fast_update"]
            mod.chunk_gated_delta_rule = refs["fast_chunk"]
            mod.recurrent_gated_delta_rule = refs["fast_recurrent"]
    for name, mod in layers:
        norm = getattr(mod, "norm", None)
        w = getattr(norm, "weight", None)
        ptr = w.data_ptr() if w is not None and hasattr(w, "data_ptr") else None
        if ptr != before[name]:
            raise RuntimeError("norm weight pointer moved on %s" % name)
    rows = grid_runtime.inspect_gdn_runtime(session.model)
    emit("backend_set", mode=mode, layers=len(rows),
         sample_fn=str(rows[0].get("causal_conv1d_fn")) if rows else None)
    return rows


def make_reference_tokens(session, prompt_text):
    """Greedy 64 continuation IDs on arm A. Never reads gold/scoring files.

    Returns dict(ids, sha256, n_actual, eos_early, diagnostic_only).
    N<16 -> diagnostic stop, no throughput claim.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solve_grid as solver
    tok = session.tokenizer
    enc = tok(prompt_text, return_tensors="pt")
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    device = session.in_dev if isinstance(session.in_dev, torch.device) else torch.device("cuda:0")
    input_ids = input_ids.to(device)
    if attn is not None:
        attn = attn.to(device)
    eos = session.eos_id
    eos_id = eos[0] if isinstance(eos, (list, tuple)) else eos
    ids = []
    solver._drop_cached_decode_state(session.model)
    past = None
    with torch.inference_mode():
        out = session.model(input_ids, attention_mask=attn, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        nxt = int(out.logits[0, -1].argmax(-1))
        for _ in range(N_CONTINUATION):
            if nxt == eos_id:
                break
            ids.append(nxt)
            step = torch.tensor([[nxt]], device=device)
            out = session.model(step, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            nxt = int(out.logits[0, -1].argmax(-1))
    # Release probe-held cache/output refs; model-mounted _cache cleared too.
    del past
    try:
        del out
    except NameError:
        pass
    solver._drop_cached_decode_state(session.model)
    n_actual = len(ids)
    sha = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
    emit("reference_tokens", n_requested=N_CONTINUATION, n_actual=n_actual,
         sha256=sha, eos_early=n_actual < N_CONTINUATION,
         diagnostic_only=n_actual < N_MIN_DIAGNOSTIC)
    return {"ids": ids, "sha256": sha, "n_actual": n_actual,
            "eos_early": n_actual < N_CONTINUATION,
            "diagnostic_only": n_actual < N_MIN_DIAGNOSTIC}


def run_trial(session, prompt_text, token_ids):
    """One timed trial: prefill (use_cache, logits_to_keep=1) + N 1-token forwards.

    N+1 tail logits stay on GPU during timing, moved to CPU after. Per-trial
    reset_peak_memory_stats on all 4 cards; sync all cards at timing edges; no
    per-token sync, no empty_cache inside timing; inference_mode; never feed 64
    tokens at once (that would take the chunk path, not recurrent decode).
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solve_grid as solver
    n = len(token_ids)
    if n < N_MIN_DIAGNOSTIC:
        raise RuntimeError("reference N=%d < 16: stop at diagnosis" % n)
    tok = session.tokenizer
    enc = tok(prompt_text, return_tensors="pt")
    device = session.in_dev if isinstance(session.in_dev, torch.device) else torch.device("cuda:0")
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    mem_start = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    solver._drop_cached_decode_state(session.model)
    gpu_logits = []
    with torch.inference_mode():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        out = session.model(input_ids, attention_mask=attn, use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        gpu_logits.append(out.logits[0, -1].detach())
        cache_len_l = int(input_ids.shape[1])
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t_prefill = time.perf_counter()
        # Fixed single-token cached decode; past_key_values=None only at trial start.
        for tid in token_ids:
            step = torch.tensor([[tid]], device=device)
            out = session.model(step, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            gpu_logits.append(out.logits[0, -1].detach())
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t1 = time.perf_counter()
    if len(gpu_logits) != n + 1:
        raise RuntimeError("expected N+1=%d tail logits, got %d" % (n + 1, len(gpu_logits)))
    cache_len_end = cache_len_l + n
    peak_alloc = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    peak_res = [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]
    cpu_logits = torch.stack(gpu_logits, dim=0).float().cpu()
    del gpu_logits, past
    try:
        del out
    except NameError:
        pass
    solver._drop_cached_decode_state(session.model)
    result = {"n": n, "prefill_s": t_prefill - t0, "decode_s": t1 - t_prefill,
              "total_s": t1 - t0, "cache_len_start": cache_len_l,
              "cache_len_end": cache_len_end, "peak_alloc": peak_alloc,
              "peak_reserved": peak_res, "mem_start": mem_start, "logits_cpu": cpu_logits}
    emit("trial_done", n=n, total_s=round(result["total_s"], 4))
    return result


def _rel_rms(b, a):
    import torch
    b = b.float()
    a = a.float()
    denom = max(float((a * a).mean().sqrt()), RMS_EPS)
    return float((((b - a) * (b - a)).mean().sqrt()) / denom)


def compare_trials(a1, a2, b1, b2):
    """Thresholds only record verdicts; never auto-switch P backend.

    - A1/A2 top1 mismatch anywhere or rel-RMS > 1e-4 -> unstable_reference.
    - B vs A: any top1 mismatch or rel-RMS > max(0.01, 5*A-repeat-err)
      -> numerical_drift.
    - Else gain1=1-total_B1/total_A1, gain2 likewise; both >= 15% ->
      worth_full_output_bridge else insufficient_gain.
    FP32: RMS(B-A)/max(RMS(A),1e-12); aggregate + per-position max + top1 counts.
    """
    import torch
    out = {}
    for t in (a1, a2, b1, b2):
        if "logits_cpu" not in t or t["logits_cpu"].shape[0] != t["n"] + 1:
            raise ValueError("trial logits must hold exactly N+1 rows")
    la1, la2 = a1["logits_cpu"], a2["logits_cpu"]
    top1_a = int((la1.argmax(-1) != la2.argmax(-1)).sum())
    rep_err = _rel_rms(la2, la1)
    out.update({"a_top1_mismatch": top1_a, "a_repeat_rel_rms": rep_err})
    if top1_a > 0 or rep_err > REFERENCE_RMS_TOL:
        out["verdict"] = VERDICT_UNSTABLE_REFERENCE
        return out
    drift_tol = max(DRIFT_FLOOR, DRIFT_REPEAT_MULT * rep_err)
    out["drift_tol"] = drift_tol
    worst_rms, worst_top1 = 0.0, 0
    for bl, al, tag in ((b1["logits_cpu"], la1, "b1_a1"), (b2["logits_cpu"], a2["logits_cpu"], "b2_a2")):
        if bl.shape != al.shape:
            raise ValueError("B vs A shape mismatch on %s" % tag)
        rms = _rel_rms(bl, al)
        per = (bl - al).float().pow(2).mean(-1).sqrt() / max(
            float(al.float().pow(2).mean(-1).sqrt().mean()), RMS_EPS)
        top1 = int((bl.argmax(-1) != al.argmax(-1)).sum())
        out[tag + "_rel_rms"] = rms
        out[tag + "_per_pos_max_rms"] = float(per.max())
        out[tag + "_top1_mismatch"] = top1
        worst_rms, worst_top1 = max(worst_rms, rms), max(worst_top1, top1)
    out.update({"worst_rel_rms": worst_rms, "worst_top1_mismatch": worst_top1})
    if worst_top1 > 0 or worst_rms > drift_tol:
        out["verdict"] = VERDICT_NUMERICAL_DRIFT
        return out
    g1 = 1.0 - b1["total_s"] / a1["total_s"]
    g2 = 1.0 - b2["total_s"] / a2["total_s"]
    out.update({"gain_b1_vs_a1": g1, "gain_b2_vs_a2": g2})
    out["verdict"] = VERDICT_WORTH_BRIDGE if (g1 >= GAIN_THRESHOLD and g2 >= GAIN_THRESHOLD) \
        else VERDICT_INSUFFICIENT_GAIN
    return out


# ---- Parent/child hard envelope (6.4) ----

def _phase_deadlines(notebook_start):
    bounds, t = {}, notebook_start
    for name in ("install_env", "load", "check_jit_warmup_reference", "measure", "save"):
        t += PHASE_BUDGETS[name]
        bounds[name] = t
    return bounds


def _parent_main(args):
    notebook_start = args.notebook_start_monotonic or time.monotonic()
    bounds = _phase_deadlines(notebook_start)
    print("PROBE_ENVELOPE " + json.dumps({"total_s": TOTAL_BUDGET_S,
          "phases": PHASE_BUDGETS, "order": list(TRIAL_ORDER),
          "thresholds": {"reference_rms": REFERENCE_RMS_TOL, "drift_floor": DRIFT_FLOOR,
                         "drift_repeat_mult": DRIFT_REPEAT_MULT, "gain": GAIN_THRESHOLD}},
          ensure_ascii=False), flush=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--model-dir", args.model_dir, "--output-dir", args.output_dir,
           "--prompt-file", args.prompt_file, "--journal", args.journal]
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "probe-parent.log")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    current_phase, phase_start = "install_env", notebook_start
    hard_end = notebook_start + TOTAL_BUDGET_S
    with open(log_path, "w", encoding="utf-8") as log:
        while True:
            if time.monotonic() > hard_end:
                proc.kill()
                log.write("PARENT_TIMEOUT total 1800s exceeded; worker killed, flushed log kept\n")
                break
            line = proc.stdout.readline() if proc.stdout else ""
            if line == "" and proc.poll() is not None:
                break
            if not line:
                continue
            log.write(line)
            log.flush()
            os.fsync(log.fileno())
            if line.startswith("GRID_PROBE_EVENT "):
                try:
                    ev = json.loads(line[len("GRID_PROBE_EVENT "):])
                except ValueError:
                    continue
                if ev.get("event") == "phase_start":
                    current_phase = ev.get("phase", current_phase)
                    phase_start = time.monotonic()
                loc = bounds.get(current_phase, hard_end)
                if time.monotonic() > loc:
                    proc.kill()
                    log.write("PARENT_TIMEOUT phase %s local deadline exceeded; killed\n" % current_phase)
                    break
    return proc.wait()


def _worker_main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    emit("phase_start", phase="install_env")
    check_environment(strict_gpu=True)
    emit("phase_start", phase="load")
    import grid_runtime
    deadline = time.monotonic() + PHASE_BUDGETS["load"]
    session = load_probe_session(args.model_dir, deadline=deadline,
                                 journal_path=args.journal, runtime_config={})
    emit("phase_start", phase="check_jit_warmup_reference")
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    import causal_conv1d as cc
    import fla.ops.gated_delta_rule.chunk as fchunk
    import fla.ops.gated_delta_rule.fused_recurrent as fuser
    refs = {"torch_update": modeling.torch_causal_conv1d_update,
            "torch_chunk": modeling.torch_chunk_gated_delta_rule,
            "torch_recurrent": modeling.torch_recurrent_gated_delta_rule,
            "fast_fn": cc.causal_conv1d_fn, "fast_update": cc.causal_conv1d_update,
            "fast_chunk": fchunk.chunk_gated_delta_rule,
            "fast_recurrent": fuser.fused_recurrent_gated_delta_rule}
    prompt_text = Path(args.prompt_file).read_text(encoding="utf-8")
    set_backend(session, "A", refs=refs)
    ref = make_reference_tokens(session, prompt_text)
    if ref["diagnostic_only"]:
        raise SystemExit("reference N<16: diagnostic stop, no adopted conclusion")
    emit("phase_start", phase="measure")
    trials = {}
    for tag, mode in (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A")):
        set_backend(session, mode, refs=refs)
        trials[tag] = run_trial(session, prompt_text, ref["ids"])
    verdict = compare_trials(trials["A1"], trials["A2"], trials["B1"], trials["B2"])
    emit("phase_start", phase="save")
    set_backend(session, "A", refs=refs)  #收尾恢复A
    import solve_grid as solver
    solver._drop_cached_decode_state(session.model)
    out = {"reference": {k: v for k, v in ref.items() if k != "ids"},
           "reference_ids": ref["ids"], "verdict": verdict,
           "timing": {k: {kk: vv for kk, vv in t.items() if kk != "logits_cpu"}
                      for k, t in trials.items()},
           "gdn": grid_runtime.inspect_gdn_runtime(session.model)}
    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "probe-result.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, default=str)
        f.flush()
        os.fsync(f.fileno())
    # CUDA/launch/OOM/JIT failures propagate and exit; no fallback reload here.
    return 0


def _dry_run_cpu():
    assert abs(sum(PHASE_BUDGETS.values()) - TOTAL_BUDGET_S) < 1e-9, "phase sum must be 1800"
    assert (REFERENCE_RMS_TOL, DRIFT_FLOOR, DRIFT_REPEAT_MULT, GAIN_THRESHOLD) == (1e-4, 0.01, 5.0, 0.15)
    assert (N_CONTINUATION, N_MIN_DIAGNOSTIC, GDN_EXPECTED_LAYERS) == (64, 16, 48)
    assert tuple(TRIAL_ORDER) == ("A1", "B1", "B2", "A2")
    for mod in ("torch", "transformers", "causal_conv1d", "fla", "grid_runtime"):
        assert mod not in sys.modules, "dry-run must not import %s" % mod
    print("PROBE_ENVELOPE " + json.dumps({"total_s": TOTAL_BUDGET_S, "phases": PHASE_BUDGETS,
          "order": list(TRIAL_ORDER)}, ensure_ascii=False), flush=True)
    print("PROBE_THRESHOLDS " + json.dumps(
        {"reference_rms": REFERENCE_RMS_TOL, "drift_floor": DRIFT_FLOOR,
         "drift_repeat_mult": DRIFT_REPEAT_MULT, "gain": GAIN_THRESHOLD,
         "rms_eps": RMS_EPS, "n_continuation": N_CONTINUATION,
         "n_min": N_MIN_DIAGNOSTIC, "gdn_layers": GDN_EXPECTED_LAYERS,
         "verdicts": [VERDICT_UNSTABLE_REFERENCE, VERDICT_NUMERICAL_DRIFT,
                      VERDICT_WORTH_BRIDGE, VERDICT_INSUFFICIENT_GAIN]},
        ensure_ascii=False), flush=True)
    print("PROBE_FUNCTIONS " + json.dumps(list(SIX_FUNCTIONS), ensure_ascii=False), flush=True)
    print("PROBE_WHEELS " + json.dumps(list(WHEEL_MANIFEST), ensure_ascii=False), flush=True)
    print("DRY_RUN_CPU self-check ok; no model load, no CUDA imports", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="K GDN inference probe (012 s6). Parent enforces 1800s hard envelope.")
    p.add_argument("--model-dir", default="/mnt/models/qwen3-8-27b")
    p.add_argument("--output-dir", default="experiments/012/kernel-probe/output")
    p.add_argument("--prompt-file", default="experiments/012/inputs/db118e2a-query0-f1-prompt.txt")
    p.add_argument("--journal", default="experiments/012/kernel-probe/journal.jsonl")
    p.add_argument("--notebook-start-monotonic", type=float, default=None)
    p.add_argument("--worker", action="store_true", help="child worker (spawned by parent)")
    p.add_argument("--dry-run-cpu", action="store_true",
                   help="CPU-only: parse args, print envelope, self-check thresholds; no model/CUDA imports")
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run_cpu:
        return _dry_run_cpu()
    if args.worker:
        return _worker_main(args)
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
