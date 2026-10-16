"""K GDN inference probe (012 plan section 6, unexecuted/unrun here).

Basis (read-only, CPU authoring only):
- docs/plan/012-kernel-program-completion.md section 6 (6.1-6.4).
- experiments/011/gdn-followup-evidence.md (pinned wheels/SHA/static interfaces).

This file is a CPU-authored implementation only. It was NOT executed on GPU,
no model was loaded, and no inference was launched in this session.

Fixed identities (do not rescan combinations):
- Base: Qwen/Qwen3.8-27B, BF16, device_map="balanced", no CPU/disk offload.
- Model source (012 fixed): ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1.
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
import codecs
import hashlib
import json
import math
import os
import select
import signal
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

# 012 fixed model identity. Worker must pass this as runtime base_source;
# runtime_config={} is rejected in load_probe_session.
MODEL_SOURCE = "ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1"
MODEL_DTYPE = "bfloat16"
MODEL_DEVICE_MAP = "balanced"

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


def write_f1_prompt_with_tokenizer(tokenizer, prompt_text, prompt_file):
    """Packer-side interface (NOT worker-side).

    Must be called during Notebook packing on CPU with the ACTUAL tokenizer
    loaded from MODEL_SOURCE (local_files_only). It verifies the rendered
    original F1 full prompt for db118e2a/query0 and writes prompt_file.
    The worker only reads prompt_file and never synthesizes a prompt.
    Returns dict(path, sha256, n_tokens).
    """
    ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    if len(ids) >= 8192:
        raise ValueError("F1 prompt %d tokens hits 8192 input limit" % len(ids))
    p = Path(prompt_file)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(prompt_text, encoding="utf-8")
    sha = hashlib.sha256(prompt_text.encode()).hexdigest()
    return {"path": str(p), "sha256": sha, "n_tokens": len(ids)}


def assert_stack_versions(info):
    """Compare actual software versions against EXPECTED_STACK item by item."""
    for key in ("torch", "transformers", "peft", "accelerate", "triton"):
        actual = str(info.get(key, "missing"))
        expected = EXPECTED_STACK[key]
        if actual != expected:
            raise RuntimeError(
                "software stack mismatch on %s: expected %s, got %s"
                % (key, expected, actual))


def check_environment(*, strict_gpu=True):
    """Section 6.3 pre-checks. Lazy imports so CPU dry-run never touches CUDA pkgs."""
    import platform
    info = {"python": platform.python_version(), "wheels": list(WHEEL_MANIFEST)}
    import torch
    import transformers
    import accelerate
    import triton  # noqa: F401
    try:
        import peft  # noqa: F401
        peft_ver = peft.__version__
    except Exception as exc:
        peft_ver = "missing:%r" % exc
    info.update({"torch": torch.__version__, "transformers": transformers.__version__,
                 "accelerate": accelerate.__version__, "peft": peft_ver,
                 "triton": getattr(__import__("triton"), "__version__", "unknown"),
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
    from importlib.metadata import version
    info["kernel_versions"] = {name: version(name) for name in
                               ("causal-conv1d", "fla-core", "flash-linear-attention")}
    # Official binary wheel filename has a local build tag; its METADATA is 1.7.0.
    expected_kernels = {"causal-conv1d": "1.7.0",
                        "fla-core": "0.5.2", "flash-linear-attention": "0.5.2"}
    if {k: v.lower() for k, v in info["kernel_versions"].items()} != expected_kernels:
        raise RuntimeError("kernel package version mismatch: %s" % info["kernel_versions"])
    _kernel_refs()  # Explicit function imports; do not rely on package module attributes.
    # 012 progress §2: actual versions must match EXPECTED_STACK item by item.
    assert_stack_versions(info)
    if strict_gpu:
        if not torch.cuda.is_available() or torch.cuda.device_count() != 4:
            raise RuntimeError("K probe requires 4 CUDA GPUs, got %s" % info)
        if torch._C._GLIBCXX_USE_CXX11_ABI is not True:
            raise RuntimeError("CXX11 ABI must be True for pinned cu12 cxx11abiTRUE wheel")
        info["devices"] = [{"name": torch.cuda.get_device_name(i),
                            "capability": list(torch.cuda.get_device_capability(i))}
                           for i in range(4)]
        if any("L4" not in d["name"] or d["capability"] != [8, 9] for d in info["devices"]):
            raise RuntimeError("expected four L4 GPUs: %s" % info["devices"])
        if platform.python_version_tuple()[:2] != ("3", "12") or torch.version.cuda != "12.8":
            raise RuntimeError("expected Python 3.12 and Torch CUDA 12.8")
    emit("env_checked", **{k: str(v)[:160] for k, v in info.items()})
    return info


def _kernel_refs():
    import transformers.models.qwen3_5.modeling_qwen3_5 as modeling
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule, fused_recurrent_gated_delta_rule
    refs = {"torch_update": modeling.torch_causal_conv1d_update,
            "torch_chunk": modeling.torch_chunk_gated_delta_rule,
            "torch_recurrent": modeling.torch_recurrent_gated_delta_rule,
            "fast_fn": causal_conv1d_fn, "fast_update": causal_conv1d_update,
            "fast_chunk": chunk_gated_delta_rule, "fast_recurrent": fused_recurrent_gated_delta_rule}
    if not all(callable(fn) for fn in refs.values()):
        raise RuntimeError("one or more required kernel entry points are not callable")
    return refs


def _norm_identity(model):
    rows = {}
    for name, mod in model.named_modules():
        if hasattr(mod, "chunk_gated_delta_rule"):
            norm = mod.norm
            if type(norm).__name__ != "Qwen3_5RMSNormGated":
                raise RuntimeError("torch norm changed on " + name)
            rows[name] = (id(norm), id(norm.weight), norm.weight.data_ptr(),
                          str(norm.weight.device), str(norm.weight.dtype))
    return rows


def load_probe_session(model_dir, *, deadline, journal_path, runtime_config=None):
    """Reuse grid_runtime.load_session; pin Qwen3_5RMSNormGated via try/finally.

    Requires fixed runtime_config: base_source==MODEL_SOURCE,
    deactivate_async_load is True. runtime_config={} is rejected.
    """
    cfg = dict(runtime_config or {})
    if cfg.get("base_source") != MODEL_SOURCE:
        raise ValueError("runtime_config base_source must be %r, got %r"
                         % (MODEL_SOURCE, cfg.get("base_source")))
    if cfg.get("deactivate_async_load") is not True:
        raise ValueError("runtime_config deactivate_async_load must be True")
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import grid_runtime
    modeling = __import__("transformers.models.qwen3_5.modeling_qwen3_5", fromlist=["*"])
    saved = getattr(modeling, "FusedRMSNormGated", None)
    modeling.FusedRMSNormGated = None
    try:
        session = grid_runtime.load_session(model_dir, deadline=deadline,
                                            journal_path=journal_path,
                                            runtime_config=cfg)
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
    if not session.base_only or session.active_identity.get("kind") != "base":
        raise RuntimeError("probe must use the unadapted base")
    session.probe_norm_identity = _norm_identity(session.model)
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
    if _norm_identity(session.model) != session.probe_norm_identity:
        raise RuntimeError("norm identity changed since model load")
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
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    input_ids = enc["input_ids"]
    attn = enc.get("attention_mask")
    device = session.in_dev
    input_ids = input_ids.to(device)
    if attn is not None:
        attn = attn.to(device)
    eos = session.eos_id
    eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
    ids = []
    solver._drop_cached_decode_state(session.model)
    past = None
    with torch.inference_mode():
        out = session.model(input_ids, attention_mask=attn, past_key_values=None,
                            use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        _actual_cache_len(past, int(input_ids.shape[1]))
        _assert_trial_logits_finite(out.logits[0, -1].unsqueeze(0), "reference-prefill")
        nxt = int(out.logits[0, -1].argmax(-1))
        for _ in range(N_CONTINUATION):
            if nxt in eos_ids:
                break
            ids.append(nxt)
            step = torch.tensor([[nxt]], device=device)
            out = session.model(step, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            _actual_cache_len(past, int(input_ids.shape[1]) + len(ids))
            _assert_trial_logits_finite(out.logits[0, -1].unsqueeze(0), "reference-decode")
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


def _actual_cache_len(past, fallback_expected):
    """Read the real model cache length; never trust L+N arithmetic alone."""
    try:
        # TF 5.8 chooses the first full-attention layer for a hybrid cache.
        length = int(past.get_seq_length())
    except (AttributeError, TypeError, ValueError) as exc:
        raise RuntimeError("cannot read real cache length (expected %d)" % fallback_expected) from exc
    if length != fallback_expected:
        raise RuntimeError("cache length mismatch: real %d vs expected %d" % (length, fallback_expected))
    return length


def _assert_trial_logits_finite(logits_cpu, tag):
    import torch
    if logits_cpu.dim() != 2:
        raise RuntimeError("%s logits must be 2D [N+1, V], got %s" % (tag, tuple(logits_cpu.shape)))
    if not bool(torch.isfinite(logits_cpu).all()):
        raise RuntimeError("%s logits contain NaN/Inf; stop immediately" % tag)


def warmup_backends(session, prompt_text, token_ids, *, refs, phase_deadline):
    """A/B warmup at real shapes incl. first JIT. Must run inside the 600s prep phase.

    Runs one untimed full-shape pass per backend (same L prompt + N 1-token
    decodes) so B1 never contains the first JIT. Timing is discarded; any
    NaN/Inf or cache error stops here, before formal A1/B1/B2/A2.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solve_grid as solver
    tok = session.tokenizer
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    dev_ids = enc["input_ids"]
    n = len(token_ids)
    for mode in ("A", "B"):
        if time.monotonic() >= phase_deadline:
            raise TimeoutError("prep 600s exhausted during %s warmup; stop" % mode)
        set_backend(session, mode, refs=refs)
        device = session.in_dev
        input_ids = dev_ids.to(device)
        attn = enc.get("attention_mask")
        if attn is not None:
            attn = attn.to(device)
        solver._drop_cached_decode_state(session.model)
        with torch.inference_mode():
            out = session.model(input_ids, attention_mask=attn, past_key_values=None,
                                use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            _actual_cache_len(past, int(input_ids.shape[1]))
            probe = out.logits[0, -1].detach().float().cpu()
            _assert_trial_logits_finite(probe.unsqueeze(0), "warmup-%s-prefill" % mode)
            for step_index, tid in enumerate(token_ids, 1):
                if time.monotonic() >= phase_deadline:
                    raise TimeoutError("prep deadline reached during %s warmup" % mode)
                step = torch.tensor([[tid]], device=device)
                out = session.model(step, past_key_values=past, use_cache=True, logits_to_keep=1)
                past = out.past_key_values
                _actual_cache_len(past, int(input_ids.shape[1]) + step_index)
                probe = out.logits[0, -1].detach().float().cpu()
                if not bool(__import__("torch").isfinite(probe).all()):
                    raise RuntimeError("warmup-%s NaN/Inf; stop" % mode)
            real_len = _actual_cache_len(past, int(input_ids.shape[1]) + n)
            if real_len != int(input_ids.shape[1]) + n:
                raise RuntimeError("warmup-%s cache mismatch: real %d vs L+N %d"
                                   % (mode, real_len, int(input_ids.shape[1]) + n))
        del past
        try:
            del out
        except NameError:
            pass
        solver._drop_cached_decode_state(session.model)
        emit("warmup_done", mode=mode, n=n)
    # Leave model on A; measure phase sets each arm explicitly anyway.
    set_backend(session, "A", refs=refs)


def run_trial(session, prompt_text, token_ids):
    """One timed trial: prefill (use_cache, logits_to_keep=1) + N 1-token forwards.

    N+1 tail logits stay on GPU during timing, moved to CPU after. Per-trial
    reset_peak_memory_stats on all 4 cards; sync all cards at timing edges; no
    per-token sync, no empty_cache inside timing; inference_mode; never feed 64
    tokens at once (that would take the chunk path, not recurrent decode).
    Real cache length is read and asserted == L+N; logits shape + finite
    (NaN/Inf) are checked per trial and stop immediately on violation.
    """
    import torch
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import solve_grid as solver
    n = len(token_ids)
    if n < N_MIN_DIAGNOSTIC:
        raise RuntimeError("reference N=%d < 16: stop at diagnosis" % n)
    tok = session.tokenizer
    enc = tok(prompt_text, return_tensors="pt", add_special_tokens=False)
    device = session.in_dev
    input_ids = enc["input_ids"].to(device)
    attn = enc.get("attention_mask")
    if attn is not None:
        attn = attn.to(device)
    solver._drop_cached_decode_state(session.model)
    # Preload fixed token inputs: no host-to-device tensor construction per step.
    continuation = torch.tensor([token_ids], dtype=input_ids.dtype, device=device)
    for i in range(torch.cuda.device_count()):
        torch.cuda.synchronize(i)
        torch.cuda.reset_peak_memory_stats(i)
    mem_start = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    gpu_logits = []
    with torch.inference_mode():
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t0 = time.perf_counter()
        out = session.model(input_ids, attention_mask=attn, past_key_values=None,
                            use_cache=True, logits_to_keep=1)
        past = out.past_key_values
        gpu_logits.append(out.logits[0, -1].detach())
        cache_len_l = _actual_cache_len(past, int(input_ids.shape[1]))
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t_prefill = time.perf_counter()
        # Fixed single-token cached decode; past_key_values=None only at trial start.
        for step_index in range(n):
            step = continuation[:, step_index:step_index + 1]
            out = session.model(step, past_key_values=past, use_cache=True, logits_to_keep=1)
            past = out.past_key_values
            gpu_logits.append(out.logits[0, -1].detach())
        for i in range(torch.cuda.device_count()):
            torch.cuda.synchronize(i)
        t1 = time.perf_counter()
        # Read real cache length BEFORE releasing refs; assert == L+N.
        real_cache_end = _actual_cache_len(past, cache_len_l + n)
    if len(gpu_logits) != n + 1:
        raise RuntimeError("expected N+1=%d tail logits, got %d" % (n + 1, len(gpu_logits)))
    if real_cache_end != cache_len_l + n:
        raise RuntimeError("cache length mismatch: real %d vs L+N %d; stop"
                           % (real_cache_end, cache_len_l + n))
    cache_len_end = real_cache_end
    peak_alloc = [torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())]
    peak_res = [torch.cuda.max_memory_reserved(i) for i in range(torch.cuda.device_count())]
    cpu_logits = torch.stack(gpu_logits, dim=0).float().cpu()
    if cpu_logits.shape[0] != n + 1:
        raise RuntimeError("logits shape mismatch: rows %d vs N+1 %d"
                           % (cpu_logits.shape[0], n + 1))
    _assert_trial_logits_finite(cpu_logits, "trial")
    del gpu_logits, past
    try:
        del out
    except NameError:
        pass
    solver._drop_cached_decode_state(session.model)
    result = {"n": n, "prefill_s": t_prefill - t0, "decode_s": t1 - t_prefill,
              "total_s": t1 - t0, "cache_len_start": cache_len_l,
              "cache_len_end": cache_len_end, "cache_len_expected": cache_len_l + n,
              "peak_alloc": peak_alloc,
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
    NaN/Inf in any trial logits stops immediately (no threshold comparison).
    """
    import torch
    out = {}
    for t in (a1, a2, b1, b2):
        if "logits_cpu" not in t or t["logits_cpu"].shape[0] != t["n"] + 1:
            raise ValueError("trial logits must hold exactly N+1 rows")
        if not bool(torch.isfinite(t["logits_cpu"]).all()):
            raise RuntimeError("NaN/Inf in trial logits; stop, no verdict")
        if not math.isfinite(t["total_s"]) or t["total_s"] <= 0:
            raise ValueError("trial total_s must be finite and positive")
        if t["logits_cpu"].shape != a1["logits_cpu"].shape or t["n"] != a1["n"]:
            raise ValueError("trials must use identical token counts and vocabulary shapes")
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
        per = (bl.float() - al.float()).pow(2).mean(-1).sqrt() / \
            al.float().pow(2).mean(-1).sqrt().clamp_min(RMS_EPS)
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


def _local_deadline(phase_start, phase, hard_end):
    return min(phase_start + PHASE_BUDGETS[phase], hard_end)


def run_cmd_with_remaining(cmd, phase_deadline):
    """Run one child command with install/phase remaining time. Timeout raises."""
    remaining = phase_deadline - time.monotonic()
    if remaining <= 0:
        raise TimeoutError("phase budget exhausted before %s; exit, no further install" % cmd[:3])
    proc = subprocess.run(cmd, timeout=remaining)
    return proc


def pip_install_pinned_wheels(wheel_dir, phase_deadline):
    """Verify SHA then pip install each pinned wheel with phase-remaining timeout.

    On timeout or error: exit immediately, do NOT attempt further installs.
    """
    wheel_dir = Path(wheel_dir)
    for entry in WHEEL_MANIFEST:
        if time.monotonic() >= phase_deadline:
            raise TimeoutError("install 120s exhausted; exit, no further install")
        path = wheel_dir / entry["file"]
        if not path.is_file():
            raise FileNotFoundError("missing pinned wheel: %s" % path)
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        if sha != entry["sha256"]:
            raise RuntimeError("wheel SHA mismatch on %s" % path.name)
        remaining = phase_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("install 120s exhausted before pip %s" % path.name)
        subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", "--no-deps",
                        str(path)], timeout=remaining, check=True)
        emit("wheel_installed", file=entry["file"])


def _drain_and_kill(proc, log):
    try:
        # Only kill a group that this child leads; never signal the caller's group.
        if os.getpgid(proc.pid) == proc.pid:
            os.killpg(proc.pid, signal.SIGKILL)
        else:
            proc.kill()
    except ProcessLookupError:
        pass
    try:
        # Best-effort drain so flushed log keeps the tail; bounded wait.
        proc.wait(timeout=5)
    except Exception:
        pass
    try:
        log.flush()
        os.fsync(log.fileno())
    except Exception:
        pass


def _supervise_stream(proc, *, notebook_start, hard_end, bounds, log, poll_s=0.5):
    """Shared non-blocking supervisor: select-poll stdout, enforce local+global deadlines.

    Returns exit reason string. Kills worker on timeout and keeps flushed log.
    Checks timeouts even when the child produces no output.
    """
    current_phase, phase_start = "install_env", notebook_start
    decoder = codecs.getincrementaldecoder("utf-8")("replace")
    pending = ""
    seen_phases = {"install_env"}

    def timeout(reason):
        log.write("\nPARENT_TIMEOUT %s; worker group killed, flushed log kept\n" % reason)
        _drain_and_kill(proc, log)
        return reason

    while True:
        now = time.monotonic()
        loc = _local_deadline(phase_start, current_phase, hard_end)
        nominal = bounds.get(current_phase, hard_end)
        eff = min(loc, nominal, hard_end)
        if now >= hard_end:
            return timeout("total_timeout")
        if now >= eff:
            return timeout("phase_timeout:%s" % current_phase)
        if proc.stdout is None:
            if proc.poll() is not None:
                return "exited:%s" % proc.returncode
            time.sleep(min(poll_s, max(0.0, eff - now)))
            continue
        # A readable pipe can contain a partial line. os.read avoids readline()
        # blocking forever for a newline during a CUDA hang or progress output.
        rlist, _, _ = select.select([proc.stdout], [], [], min(poll_s, max(0.0, eff - now)))
        if not rlist:
            if proc.poll() is not None:
                return "exited:%s" % proc.returncode
            continue
        chunk = os.read(proc.stdout.fileno(), 65536)
        if not chunk:
            if proc.poll() is not None:
                return "exited:%s" % proc.returncode
            time.sleep(min(poll_s, max(0.0, eff - time.monotonic())))
            continue
        text = decoder.decode(chunk)
        log.write(text)
        log.flush()
        try:
            os.fsync(log.fileno())
        except Exception:
            pass
        pending += text
        lines = pending.split("\n")
        pending = lines.pop()
        for line in lines:
            if not line.startswith("GRID_PROBE_EVENT "):
                continue
            try:
                ev = json.loads(line[len("GRID_PROBE_EVENT "):])
            except ValueError:
                continue
            if ev.get("event") == "phase_start":
                phase = ev.get("phase")
                stamp = ev.get("monotonic")
                if phase not in PHASE_BUDGETS or not isinstance(stamp, (int, float)) or not math.isfinite(stamp):
                    return timeout("invalid_phase_event")
                if not notebook_start <= stamp <= time.monotonic() + 1.0:
                    return timeout("invalid_phase_clock")
                event_deadline = min(_local_deadline(phase_start, current_phase, hard_end),
                                     bounds.get(current_phase, hard_end), hard_end)
                if stamp > event_deadline:
                    return timeout("phase_timeout:%s" % current_phase)
                if phase == "install_env" and current_phase == phase:
                    continue  # Install budget starts at notebook cell 1, not worker import.
                if phase in seen_phases:
                    return timeout("repeated_phase:%s" % phase)
                seen_phases.add(phase)
                current_phase, phase_start = phase, stamp


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _parent_main(args):
    # Parent supervision starts BEFORE any install work in the worker.
    # Notebook ordering requirement: spawn this parent in the first cell,
    # then do pip installs only inside the worker's install phase (or via
    # --wheel-dir), each bounded by the install-phase remaining time.
    now = time.monotonic()
    notebook_start = now if args.notebook_start_monotonic is None else args.notebook_start_monotonic
    if not math.isfinite(notebook_start) or notebook_start < 0 or notebook_start > now + 1.0:
        raise ValueError("notebook start must be a valid timestamp from the same monotonic clock")
    if now >= notebook_start + PHASE_BUDGETS["install_env"]:
        raise TimeoutError("install/environment budget already exhausted before worker start")
    bounds = _phase_deadlines(notebook_start)
    print("PROBE_ENVELOPE " + json.dumps({"total_s": TOTAL_BUDGET_S,
          "phases": PHASE_BUDGETS, "order": list(TRIAL_ORDER),
          "thresholds": {"reference_rms": REFERENCE_RMS_TOL, "drift_floor": DRIFT_FLOOR,
                         "drift_repeat_mult": DRIFT_REPEAT_MULT, "gain": GAIN_THRESHOLD}},
          ensure_ascii=False), flush=True)
    cmd = [sys.executable, os.path.abspath(__file__), "--worker",
           "--model-dir", args.model_dir, "--output-dir", args.output_dir,
           "--prompt-file", args.prompt_file, "--journal", args.journal,
           "--notebook-start-monotonic", str(notebook_start)]
    if args.wheel_dir:
        cmd += ["--wheel-dir", args.wheel_dir]
    if args.setup_script:
        cmd += ["--setup-script", args.setup_script]
    os.makedirs(args.output_dir, exist_ok=True)
    log_path = os.path.join(args.output_dir, "probe-parent.log")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            start_new_session=True)
    hard_end = notebook_start + TOTAL_BUDGET_S
    with open(log_path, "w", encoding="utf-8") as log:
        log.write("PARENT_SUPERVISION_STARTED before worker install; hard_end=%s\n" % hard_end)
        log.flush()
        try:
            reason = _supervise_stream(proc, notebook_start=notebook_start, hard_end=hard_end,
                                       bounds=bounds, log=log)
        except BaseException:
            _drain_and_kill(proc, log)
            raise
    rc = proc.wait(timeout=5)
    _write_json(Path(args.output_dir) / "probe-supervisor.json",
                {"reason": reason, "returncode": rc, "elapsed_s": time.monotonic() - notebook_start,
                 "notebook_start_monotonic": notebook_start, "hard_end": hard_end})
    return rc if reason.startswith("exited:") else 124


def _simulate_hang_supervision(args):
    """CPU-only hang simulation: sleep child + same select supervisor proves kill path.

    Never kills anything important: child is `sleep <secs>` started here.
    Uses a tiny local budget to prove non-blocking timeout works.
    """
    out_dir = args.output_dir
    os.makedirs(out_dir, exist_ok=True)
    log_path = os.path.join(out_dir, "probe-hang-simulation.log")
    notebook_start = time.monotonic()
    hard_end = notebook_start + args.hang_budget_s + 5.0
    bounds = {"install_env": notebook_start + args.hang_budget_s}
    proc = subprocess.Popen(["sleep", str(args.hang_child_secs)],
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, start_new_session=True)
    with open(log_path, "w", encoding="utf-8") as log:
        reason = _supervise_stream(proc, notebook_start=notebook_start, hard_end=hard_end,
                                   bounds=bounds, log=log, poll_s=0.2)
    rc = proc.wait()
    # Supervisor must have killed the sleeper on local deadline.
    print("HANG_SIMULATION reason=%s returncode=%s log=%s" % (reason, rc, log_path), flush=True)
    return 0 if reason.startswith("phase_timeout") else 2


def _worker_main(args):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    notebook_start = args.notebook_start_monotonic or time.monotonic()
    hard_end = notebook_start + TOTAL_BUDGET_S
    bounds = _phase_deadlines(notebook_start)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    emit("phase_start", phase="install_env")
    install_deadline = min(bounds["install_env"], hard_end)
    if args.setup_script:
        setup_path = Path(args.setup_script).resolve()
        namespace = {"CODE": setup_path.parent, "NOTEBOOK_START": notebook_start,
                     "__file__": str(setup_path), "__name__": "__probe_setup__"}
        exec(compile(setup_path.read_bytes(), str(setup_path), "exec"), namespace)
        if namespace.get("model_dir"):
            args.model_dir = str(namespace["model_dir"])
    # Setup renders the prompt with the actual tokenizer; validate before load.
    if not args.prompt_file:
        raise FileNotFoundError("prompt-file is required; generate it packing-side "
                                "with write_f1_prompt_with_tokenizer() and pass --prompt-file")
    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_file():
        raise FileNotFoundError("prompt file missing: %s; generate it packing-side with "
                                "the actual tokenizer, do not synthesize in worker" % prompt_path)
    if args.wheel_dir:
        pip_install_pinned_wheels(args.wheel_dir, install_deadline)
    if time.monotonic() >= install_deadline:
        raise TimeoutError("install 120s exhausted; exit, no further install")
    env = check_environment(strict_gpu=True)
    _write_json(output / "env.json", env)
    if time.monotonic() >= install_deadline:
        raise TimeoutError("install/environment phase exceeded 120s")
    emit("phase_start", phase="load")
    import grid_runtime
    # Correct session cutoff: absolute hard end (1800s) with 120s save window
    # carried in runtime_config; load phase bound is enforced by the parent.
    session_deadline = hard_end
    runtime_config = {"base_source": MODEL_SOURCE, "deactivate_async_load": True,
                      "save_window_s": PHASE_BUDGETS["save"]}
    session = load_probe_session(args.model_dir, deadline=session_deadline,
                                 journal_path=args.journal, runtime_config=runtime_config)
    emit("phase_start", phase="check_jit_warmup_reference")
    prep_deadline = min(time.monotonic() + PHASE_BUDGETS["check_jit_warmup_reference"],
                        bounds["check_jit_warmup_reference"], hard_end)
    refs = _kernel_refs()
    prompt_text = prompt_path.read_text(encoding="utf-8")
    prompt_ids = session.tokenizer.encode(prompt_text, add_special_tokens=False)
    if not prompt_ids or len(prompt_ids) >= 8192:
        raise ValueError("invalid prompt length: %d" % len(prompt_ids))
    identity = {"task_id": TASK_ID, "query_index": 0, "formatter": "f1_compact",
                "enable_thinking": False, "base": session.base_identity,
                "adapter": session.active_identity, "runtime": session.runtime_identity,
                "prompt_sha256": hashlib.sha256(prompt_text.encode()).hexdigest(),
                "prompt_tokens": len(prompt_ids), "template_sha256": grid_runtime.canonical_hash(session.tokenizer.chat_template),
                "probe_sha256": grid_runtime.file_sha256(__file__)}
    _write_json(output / "probe-identity.json", identity)
    set_backend(session, "A", refs=refs)
    ref = make_reference_tokens(session, prompt_text)
    _write_json(output / "reference-tokens.json", ref)
    if ref["diagnostic_only"]:
        _write_json(output / "probe-result.json", {"status": "diagnostic_only", "reference": ref,
                                                   "reason": "reference N<16", "identity": identity})
        return 0
    # A/B warmup + first JIT at real shapes, still inside the 600s prep phase.
    warmup_backends(session, prompt_text, ref["ids"], refs=refs, phase_deadline=prep_deadline)
    if time.monotonic() >= prep_deadline:
        raise TimeoutError("prep 600s exhausted before formal timing; stop")
    emit("phase_start", phase="measure")
    measure_deadline = min(time.monotonic() + PHASE_BUDGETS["measure"], bounds["measure"], hard_end)
    trials = {}
    for tag, mode in (("A1", "A"), ("B1", "B"), ("B2", "B"), ("A2", "A")):
        if time.monotonic() >= measure_deadline:
            raise TimeoutError("measurement budget exhausted before " + tag)
        provenance = set_backend(session, mode, refs=refs)
        trials[tag] = run_trial(session, prompt_text, ref["ids"])
        trials[tag]["backend"] = mode
        trials[tag]["gdn"] = provenance
        # Persist every completed trial before attempting the next arm.
        session.torch.save(trials[tag]["logits_cpu"], output / (tag + "-logits.pt"))
        _write_json(output / (tag + ".json"), {k: v for k, v in trials[tag].items() if k != "logits_cpu"})
    verdict = compare_trials(trials["A1"], trials["A2"], trials["B1"], trials["B2"])
    emit("phase_start", phase="save")
    set_backend(session, "A", refs=refs)  #收尾恢复A
    import solve_grid as solver
    solver._drop_cached_decode_state(session.model)
    out = {"status": "complete", "identity": identity,
           "reference": {k: v for k, v in ref.items() if k != "ids"},
           "reference_ids": ref["ids"], "verdict": verdict,
           "timing": {k: {kk: vv for kk, vv in t.items() if kk != "logits_cpu"}
                      for k, t in trials.items()},
           "gdn": grid_runtime.inspect_gdn_runtime(session.model)}
    _write_json(output / "probe-result.json", out)
    # CUDA/launch/OOM/JIT failures propagate and exit; no fallback reload here.
    return 0


def _dry_run_cpu():
    assert abs(sum(PHASE_BUDGETS.values()) - TOTAL_BUDGET_S) < 1e-9, "phase sum must be 1800"
    assert (REFERENCE_RMS_TOL, DRIFT_FLOOR, DRIFT_REPEAT_MULT, GAIN_THRESHOLD) == (1e-4, 0.01, 5.0, 0.15)
    assert (N_CONTINUATION, N_MIN_DIAGNOSTIC, GDN_EXPECTED_LAYERS) == (64, 16, 48)
    assert tuple(TRIAL_ORDER) == ("A1", "B1", "B2", "A2")
    for mod in ("torch", "transformers", "causal_conv1d", "fla", "grid_runtime"):
        assert mod not in sys.modules, "dry-run must not import %s" % mod
    # New self-checks for the 5 fixed gaps (params/envelope/thresholds only).
    p = build_parser()
    prompt_action = next(a for a in p._actions if "--prompt-file" in a.option_strings)
    assert prompt_action.required is True, "--prompt-file must be required"
    assert MODEL_SOURCE == "ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1"
    import inspect as _inspect
    assert "select.select" in _inspect.getsource(_supervise_stream), "supervisor must select-poll"
    assert "_local_deadline" in _inspect.getsource(_supervise_stream)
    assert "deactivate_async_load" in _inspect.getsource(load_probe_session)
    assert "_actual_cache_len" in _inspect.getsource(run_trial)
    assert "_assert_trial_logits_finite" in _inspect.getsource(run_trial)
    assert "isfinite" in _inspect.getsource(_assert_trial_logits_finite)
    assert "isfinite" in _inspect.getsource(compare_trials)
    assert "assert_stack_versions" in _inspect.getsource(check_environment)
    assert "warmup_backends" in _inspect.getsource(_worker_main)
    assert "run_cmd_with_remaining" in _inspect.getsource(pip_install_pinned_wheels) or \
        "timeout" in _inspect.getsource(pip_install_pinned_wheels)
    assert tuple(EXPECTED_STACK.keys()) == ("torch", "transformers", "peft", "accelerate", "triton")
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
    print("PROBE_MODEL_SOURCE " + json.dumps({"base_source": MODEL_SOURCE, "dtype": MODEL_DTYPE,
          "device_map": MODEL_DEVICE_MAP}, ensure_ascii=False), flush=True)
    print("DRY_RUN_CPU self-check ok; no model load, no CUDA imports", flush=True)
    return 0


def build_parser():
    p = argparse.ArgumentParser(description="K GDN inference probe (012 s6). Parent enforces 1800s hard envelope.")
    p.add_argument("--model-dir", default="/mnt/models/qwen3-8-27b")
    p.add_argument("--output-dir", default="experiments/012/kernel-probe/output")
    p.add_argument("--prompt-file", required=True,
                   help="REQUIRED: F1 prompt file generated packing-side with the actual "
                        "tokenizer via write_f1_prompt_with_tokenizer(); worker never synthesizes it")
    p.add_argument("--journal", default="experiments/012/kernel-probe/journal.jsonl")
    p.add_argument("--wheel-dir", default=None,
                   help="optional dir with 3 pinned wheels; installs use phase-remaining timeout")
    p.add_argument("--setup-script", default=None,
                   help="optional local setup script run inside supervised install_env; renders prompt before load")
    p.add_argument("--notebook-start-monotonic", type=float, default=None)
    p.add_argument("--worker", action="store_true", help="child worker (spawned by parent)")
    p.add_argument("--dry-run-cpu", action="store_true",
                   help="CPU-only: parse args, print envelope, self-check thresholds; no model/CUDA imports")
    p.add_argument("--simulate-hang", action="store_true",
                   help="CPU-only: sleep child + select supervisor kill simulation; kills only that sleep")
    p.add_argument("--hang-child-secs", type=float, default=30.0)
    p.add_argument("--hang-budget-s", type=float, default=2.0)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.dry_run_cpu:
        return _dry_run_cpu()
    if args.simulate_hang:
        return _simulate_hang_supervision(args)
    if args.worker:
        try:
            return _worker_main(args)
        except BaseException as exc:
            _write_json(Path(args.output_dir) / "probe-error.json",
                        {"status": "error", "error_type": type(exc).__name__, "error": str(exc)[:2000]})
            raise
    return _parent_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
