"""011 single-resident-model runtime. Imports are CPU-safe until load_session.

Deadlines are absolute time.monotonic() values. The caller must also supervise
the process: a token stopping criterion cannot interrupt a hung CUDA operation
or from_pretrained. No gold answers are accepted by generation functions.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import solve_grid as solver

TERMINAL_STATUSES = {"complete", "invalid_output", "ineligible_input_too_long"}
RUNTIME_VERSION = "011-resident-v1"


def canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def code_fingerprint():
    root = Path(__file__).resolve().parent
    names = ("grid_runtime.py", "solve_grid.py", "arc_episode_data.py",
             "run_budgeted_grid_experiment.py", "ttt_per_task.py", "solve_program.py")
    return {name: file_sha256(root / name) for name in names if (root / name).is_file()}


def event(event_name, **fields):
    print("GRID_RUNTIME_EVENT " + json.dumps({"event": event_name, "monotonic": time.monotonic(),
          **fields}, ensure_ascii=False, default=str), flush=True)


class CandidateJournal:
    """Append-only candidate records; only complete model outcomes are reused."""
    def __init__(self, path):
        self.path = os.path.abspath(path)
        self.done = {}
        self.malformed_lines = 0
        if os.path.isfile(self.path):
            with open(self.path, encoding="utf-8") as handle:
                for line in handle:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        self.malformed_lines += 1
                        continue
                    if isinstance(row, dict) and row.get("status") in TERMINAL_STATUSES and row.get("identity_key"):
                        self.done[row["identity_key"]] = row

    def get(self, identity):
        row = self.done.get(canonical_hash(identity))
        if row is None:
            return None
        return {**row, "cache_hit": True, "physical_elapsed_s": 0.0}

    def append(self, identity, row):
        row = {**row, "identity": identity, "identity_key": canonical_hash(identity),
               "recorded_unix_s": time.time(), "cache_hit": False}
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        # Repair only the separator after an interrupted final line. Old bytes
        # remain intact, and the next valid record is independently readable.
        prefix = b""
        if os.path.exists(self.path) and os.path.getsize(self.path):
            with open(self.path, "rb") as probe:
                probe.seek(-1, os.SEEK_END)
                if probe.read(1) != b"\n":
                    prefix = b"\n"
        with open(self.path, "ab") as handle:
            handle.write(prefix + (json.dumps(row, ensure_ascii=False, default=str) + "\n").encode())
            handle.flush()
            os.fsync(handle.fileno())
        if row.get("status") in TERMINAL_STATUSES:
            self.done[row["identity_key"]] = row
        return row


@dataclass
class Session:
    model: object
    tokenizer: object
    torch: object
    in_dev: object
    eos_id: object
    pad_id: object
    deadline: float
    model_dir: str
    base_identity: dict
    journal: CandidateJournal
    runtime_identity: dict = field(default_factory=dict)
    active_identity: dict = field(default_factory=dict)
    current_adapter: str | None = None
    base_only: bool = True
    adapters: dict = field(default_factory=dict)
    started: float = field(default_factory=time.monotonic)
    save_window_s: float = 300.0
    allow_cached: bool = True


def base_fingerprint(model_dir, runtime_config):
    """Pin immutable model source + config/index, not a misleading path hash.

    Full base shard hashes can be supplied in base_weight_manifest. We do not
    reread ~55GB each session solely to hash weights. An immutable source ID is
    mandatory without a precomputed content manifest.
    """
    root = Path(model_dir)
    source = runtime_config.get("base_source")
    weights = runtime_config.get("base_weight_manifest")
    if not source and not weights:
        raise ValueError("runtime.base_source (immutable model/version) or base_weight_manifest is required")
    index = root / "model.safetensors.index.json"
    shard_sizes = {}
    if index.is_file():
        with index.open() as handle:
            index_data = json.load(handle)
        for name in sorted(set(index_data.get("weight_map", {}).values())):
            target = root / name
            if not target.is_file():
                raise ValueError("missing base shard: " + name)
            shard_sizes[name] = target.stat().st_size
    else:
        for target in sorted(root.glob("*.safetensors")):
            shard_sizes[target.name] = target.stat().st_size
    return {"source": source, "weight_manifest": weights,
            "config_sha256": file_sha256(root / "config.json"),
            "index_sha256": file_sha256(index) if index.is_file() else None,
            "shard_sizes": shard_sizes, "full_base_weight_sha_computed_here": False}


def inspect_gdn_runtime(model):
    rows = []
    for name, module in model.named_modules():
        if hasattr(module, "chunk_gated_delta_rule"):
            row = {"layer": name, "class": type(module).__name__,
                   "norm": type(getattr(module, "norm", None)).__name__}
            for key in ("causal_conv1d_fn", "causal_conv1d_update",
                        "chunk_gated_delta_rule", "recurrent_gated_delta_rule"):
                fn = getattr(module, key, None)
                row[key] = None if fn is None else "%s.%s" % (
                    getattr(fn, "__module__", ""), getattr(fn, "__name__", type(fn).__name__))
            rows.append(row)
    return rows


def load_session(model_dir, *, deadline, journal_path, runtime_config=None):
    cfg = dict(runtime_config or {})
    if time.monotonic() >= deadline:
        raise TimeoutError("deadline expired before loading")
    base_id = base_fingerprint(model_dir, cfg)
    if "deactivate_async_load" in cfg:
        os.environ["HF_DEACTIVATE_ASYNC_LOAD"] = "1" if cfg["deactivate_async_load"] else "0"
    event("load_start", model_dir=str(model_dir), deadline=deadline)
    start = time.monotonic()
    import torch
    import transformers
    import accelerate
    from transformers import AutoModelForCausalLM, AutoTokenizer
    if not torch.cuda.is_available() or torch.cuda.device_count() < 4:
        raise RuntimeError("011 requires at least 4 CUDA GPUs")
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.bfloat16,
                                               device_map="balanced", low_cpu_mem_usage=True,
                                               local_files_only=True, trust_remote_code=False)
    solver._sync_cuda(torch)
    device_map = dict(getattr(model, "hf_device_map", {}))
    if any(str(value) in ("cpu", "disk") for value in device_map.values()):
        raise RuntimeError("unexpected CPU/disk offload in fixed 4L4 runtime: %s" % device_map)
    gen_cfg = getattr(model, "generation_config", None)
    eos = getattr(gen_cfg, "eos_token_id", None) or tokenizer.eos_token_id
    pad = tokenizer.pad_token_id
    if pad is None:
        pad = eos[0] if isinstance(eos, (list, tuple)) else eos
    identity = {"runtime": RUNTIME_VERSION, "code_sha256": code_fingerprint(),
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__,
                             "accelerate": accelerate.__version__},
                "gdn": inspect_gdn_runtime(model), "device_map": device_map,
                "dtype": "bfloat16", "async_load_deactivated": os.environ.get("HF_DEACTIVATE_ASYNC_LOAD")}
    session = Session(model, tokenizer, torch, solver._embedding_device(model), eos, pad,
                      float(deadline), str(model_dir), base_id, CandidateJournal(journal_path),
                      runtime_identity=identity,
                      active_identity={"kind": "base", "base": canonical_hash(base_id)},
                      save_window_s=float(cfg.get("save_window_s", 300)))
    prepare_eval(session)
    event("load_done", elapsed_s=round(time.monotonic() - start, 3),
          devices=device_map, base_identity=base_id, runtime_identity=identity)
    return session


def prepare_eval(session):
    import train_grid_sft as trainer
    trainer.disable_checkpointing(session.model)
    trainer.set_use_cache(session.model, True)
    session.model.eval()
    solver._drop_cached_decode_state(session.model)


def adapter_fingerprint(path):
    root = Path(path)
    weights = sorted(root.glob("adapter_model*.safetensors"))
    if not weights:
        weights = sorted(root.glob("adapter_model*.bin"))
    if not weights or not (root / "adapter_config.json").is_file():
        raise ValueError("adapter requires config and weight files: %s" % root)
    return {"kind": "adapter", "weights_sha256": {p.name: file_sha256(p) for p in weights},
            "config_sha256": file_sha256(root / "adapter_config.json"),
            "formatter_sha256": file_sha256(root / "formatter.json") if (root / "formatter.json").is_file() else None}


def activate_adapter(session, adapter_ref_or_none, *, name=None, trainable=False):
    solver._drop_cached_decode_state(session.model)
    if adapter_ref_or_none is None:
        if trainable:
            raise ValueError("base-only activation cannot train the full base; create a fresh LoRA explicitly")
        session.base_only = True
        session.current_adapter = None
        session.active_identity = {"kind": "base", "base": canonical_hash(session.base_identity)}
        prepare_eval(session)
        return session.active_identity
    path = os.path.abspath(str(adapter_ref_or_none))
    # Rehash the small adapter at activation. A same-path overwrite must not
    # silently reuse either cached outputs or an already-loaded old tensor set.
    fresh_identity = adapter_fingerprint(path)
    if path in session.adapters:
        entry = session.adapters[path]
        identity, adapter_name = entry["identity"], entry["name"]
        if identity != fresh_identity:
            raise ValueError("adapter snapshot changed during resident session: %s; use a new immutable path" % path)
    else:
        from peft import PeftModel
        identity = fresh_identity
        adapter_name = name or "a_" + canonical_hash(identity)[:16]
        if not isinstance(session.model, PeftModel):
            session.model = PeftModel.from_pretrained(session.model, path, adapter_name=adapter_name,
                                                     is_trainable=trainable)
        else:
            session.model.load_adapter(path, adapter_name=adapter_name, is_trainable=trainable)
        session.adapters[path] = {"identity": identity, "name": adapter_name}
        session.runtime_identity["versions"]["peft"] = __import__("peft").__version__
    session.model.set_adapter(adapter_name)
    for param_name, param in session.model.named_parameters():
        param.requires_grad_(bool(trainable and "lora_" in param_name and ("." + adapter_name + ".") in param_name))
    session.current_adapter = adapter_name
    session.base_only = False
    session.active_identity = identity
    session.in_dev = solver._embedding_device(session.model)
    if not trainable:
        prepare_eval(session)
    event("adapter_active", adapter=adapter_name, identity=identity, trainable=trainable)
    return identity


def mark_adapter_state(session, identity):
    """TTT callers must update identity after changing weights before generating."""
    session.active_identity = dict(identity)
    session.base_only = False
    active = getattr(session.model, "active_adapter", None)
    if isinstance(active, str):
        session.current_adapter = active
    session.in_dev = solver._embedding_device(session.model)
    solver._drop_cached_decode_state(session.model)


@contextlib.contextmanager
def adapter_context(session):
    if session.base_only:
        with solver._base_view(session.model):
            yield
    else:
        yield


@contextlib.contextmanager
def generation_deadline(session, deadline):
    """Inject token-level stopping while retaining the old solver implementation."""
    from transformers import StoppingCriteria, StoppingCriteriaList
    class DeadlineStop(StoppingCriteria):
        hit = False
        def __call__(self, input_ids, scores, **kwargs):
            self.hit = time.monotonic() >= deadline
            return self.hit
    stop = DeadlineStop()
    model = session.model
    had_local = "generate" in model.__dict__
    local_value = model.__dict__.get("generate")
    original = model.generate
    def wrapped(*args, **kwargs):
        existing = list(kwargs.pop("stopping_criteria", None) or [])
        kwargs["stopping_criteria"] = StoppingCriteriaList(existing + [stop])
        return original(*args, **kwargs)
    model.generate = wrapped
    try:
        yield stop
    finally:
        if had_local:
            model.generate = local_value
        else:
            delattr(model, "generate")


def _identity(session, prompt, config, job):
    return {"base": session.base_identity, "adapter": session.active_identity,
            "runtime": session.runtime_identity,
            "template_sha256": canonical_hash(session.tokenizer.chat_template),
            "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
            "generation": config, "job": job}


def _deadline(session, deadline, max_time_s=None):
    values = [float(session.deadline) - session.save_window_s]
    if deadline is not None:
        values.append(float(deadline))
    if max_time_s is not None:
        values.append(time.monotonic() + float(max_time_s))
    return min(values)


def _too_late(session, identity, job, deadline, estimate_s=0):
    if time.monotonic() + float(estimate_s) < deadline:
        return None
    return session.journal.append(identity, {**job, "status": "budget_interrupted",
        "finish_reason": "admission_deadline", "raw_text": "", "text": "", "grid": None,
        "gen_tokens": 0, "elapsed_s": 0.0, "physical_elapsed_s": 0.0})


def generate_grid(session, task_id, query_index, demos, query_input, *, formatter="f0_space",
                  view="original", sample_index=0, do_sample=False, max_new_tokens=2048,
                  seed=20260930, deadline=None, run_tag="", estimate_s=0,
                  max_prompt_tokens=None, max_time_s=None):
    from arc_episode_data import build_messages, text_to_grid
    formatter = solver._normalize_formatter(formatter)
    if view not in ("original", "transpose"):
        raise ValueError("only original/transpose grid views are implemented")
    if view == "transpose":
        demos = [{"input": solver.transpose_grid(p["input"]), "output": solver.transpose_grid(p["output"])} for p in demos]
        query_input = solver.transpose_grid(query_input)
    messages = build_messages(demos, query_input, formatter)
    prompt = session.tokenizer.apply_chat_template(messages, tokenize=False,
              add_generation_prompt=True, enable_thinking=False)
    config = {"formatter": formatter, "enable_thinking": False, "max_new_tokens": int(max_new_tokens),
              "do_sample": bool(do_sample), "temperature": .7 if do_sample else None,
              "top_p": .9 if do_sample else None, "top_k": 0, "seed": int(seed) if do_sample else None,
              "max_prompt_tokens": max_prompt_tokens, "max_time_s": max_time_s}
    job = {"task_id": str(task_id), "query_index": int(query_index), "view": view,
           "sample_index": int(sample_index), "kind": "sample" if do_sample else "greedy", "run_tag": run_tag}
    identity = _identity(session, prompt, config, job)
    cached = session.journal.get(identity) if session.allow_cached else None
    if cached is not None:
        return cached
    limit = _deadline(session, deadline, max_time_s)
    stopped = _too_late(session, identity, job, limit, estimate_s)
    if stopped:
        return stopped
    prompt_tokens = len(session.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if max_prompt_tokens is not None and prompt_tokens > max_prompt_tokens:
        return session.journal.append(identity, {**job, "status": "ineligible_input_too_long",
            "prompt_tokens": prompt_tokens, "limit": max_prompt_tokens, "gen_tokens": 0,
            "raw_text": "", "grid": None, "elapsed_s": 0.0, "physical_elapsed_s": 0.0})
    prepare_eval(session)
    event("candidate_start", **job, remaining_s=round(limit-time.monotonic(), 2))
    start = time.monotonic()
    with adapter_context(session), generation_deadline(session, limit) as stopper:
        gen_cfg = getattr(session.model, "generation_config", None)
        record, plen = solver._run_one_candidate(session.model, session.tokenizer, session.torch,
            prompt, job["kind"], do_sample, max_new_tokens, seed, None, text_to_grid,
            session.in_dev, session.pad_id, session.eos_id, solver._generation_config_has("presence_penalty"),
            getattr(gen_cfg, "cache_implementation", None) in solver._STATIC_CACHE_IMPLEMENTATIONS,
            formatter=formatter)
    elapsed = time.monotonic() - start
    if record.get("error"):
        status, reason = "infrastructure_error", "generation_error"
    elif stopper.hit:
        status, reason = "budget_interrupted", "deadline"
    elif record.get("truncated"):
        status, reason = "invalid_output", "max_new_tokens"
    elif record.get("parse") != "OK":
        status, reason = "invalid_output", "parse_error"
    else:
        status, reason = "complete", "eos"
    grid = record.get("grid")
    if status not in ("budget_interrupted", "infrastructure_error") and grid is not None and (len(grid) > 30 or len(grid[0]) > 30):
        status, reason = "invalid_output", "grid_outside_1_to_30"
    if view == "transpose" and grid is not None:
        grid = solver.transpose_grid(grid)
    record.update(job)
    record.update(status=status, finish_reason=reason, grid=grid,
                  raw_text=record.get("text", ""), prompt_tokens=plen,
                  elapsed_s=round(elapsed, 6), physical_elapsed_s=round(elapsed, 6))
    result = session.journal.append(identity, record)
    event("candidate_done", **job, status=status, gen_tokens=record.get("gen_tokens"), elapsed_s=round(elapsed, 3))
    return result


def generate_text(session, messages, *, max_new_tokens=1536, do_sample=True,
                  temperature=1.0, top_p=.95, top_k=20, seed=20261011,
                  enable_thinking=True, reasoning_effort="low", preserve_thinking=True,
                  deadline=None, max_time_s=360, max_prompt_tokens=8192, identity=None):
    """Bounded raw text generation; keep special tokens for think/final parsing."""
    render = {"tokenize": False, "add_generation_prompt": True,
              "enable_thinking": enable_thinking, "reasoning_effort": reasoning_effort,
              "preserve_thinking": preserve_thinking}
    prompt = session.tokenizer.apply_chat_template(messages, **render)
    cfg = {**render, "max_new_tokens": max_new_tokens, "do_sample": do_sample,
           "temperature": temperature, "top_p": top_p, "top_k": top_k,
           "seed": seed, "max_time_s": max_time_s, "max_prompt_tokens": max_prompt_tokens}
    job = dict(identity or {})
    job.setdefault("kind", "program")
    full_identity = _identity(session, prompt, cfg, job)
    cached = session.journal.get(full_identity) if session.allow_cached else None
    if cached is not None:
        return cached
    limit = _deadline(session, deadline, max_time_s)
    stopped = _too_late(session, full_identity, job, limit)
    if stopped:
        return stopped
    plen = len(session.tokenizer(prompt, add_special_tokens=False)["input_ids"])
    if plen > max_prompt_tokens:
        return session.journal.append(full_identity, {**job, "status": "ineligible_input_too_long",
            "finish_reason": "input_limit", "prompt_tokens": plen, "limit": max_prompt_tokens,
            "raw_text": "", "text": "", "gen_tokens": 0, "elapsed_s": 0.0, "physical_elapsed_s": 0.0})
    prepare_eval(session)
    torch = session.torch
    start = time.monotonic()
    raw, ntokens, ended = "", 0, False
    status, reason, error = "complete", "eos", None
    original_multinomial = None
    event("text_start", job=job, prompt_tokens=plen, remaining_s=round(limit-time.monotonic(), 2))
    try:
        encoded = session.tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
        inputs = {k: v.to(session.in_dev) for k, v in encoded.items() if k in ("input_ids", "attention_mask")}
        kwargs = {**inputs, "max_new_tokens": int(max_new_tokens), "do_sample": bool(do_sample),
                  "repetition_penalty": 1.0, "no_repeat_ngram_size": 0, "num_beams": 1,
                  "eos_token_id": session.eos_id, "pad_token_id": session.pad_id}
        if do_sample:
            kwargs.update(temperature=float(temperature), top_p=float(top_p), top_k=int(top_k))
            random.seed(seed)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            original_multinomial = solver._install_sample_generator(torch, seed)
        gen_cfg = getattr(session.model, "generation_config", None)
        if getattr(gen_cfg, "cache_implementation", None) in solver._STATIC_CACHE_IMPLEMENTATIONS:
            kwargs["cache_implementation"] = "dynamic"
        if solver._generation_config_has("presence_penalty"):
            kwargs["presence_penalty"] = 0
        solver._sync_cuda(torch)
        with adapter_context(session), generation_deadline(session, limit) as stopper, torch.inference_mode():
            sequences = session.model.generate(**kwargs)
        solver._sync_cuda(torch)
        seq = sequences.sequences if hasattr(sequences, "sequences") else sequences
        new_ids = seq[0, plen:].tolist()
        ntokens = len(new_ids)
        raw = session.tokenizer.decode(new_ids, skip_special_tokens=False)
        eos_ids = session.eos_id if isinstance(session.eos_id, (tuple, list)) else [session.eos_id]
        ended = bool(new_ids and new_ids[-1] in eos_ids)
        if stopper.hit:
            status, reason = "budget_interrupted", "deadline"
        elif not ended:
            status, reason = "invalid_output", "max_new_tokens" if ntokens >= max_new_tokens else "no_eos"
        del sequences, seq, encoded, inputs, kwargs
    except Exception as exc:
        status, reason, error = "infrastructure_error", "generation_error", "%s: %s" % (type(exc).__name__, str(exc)[:500])
    finally:
        if original_multinomial is not None:
            torch.multinomial = original_multinomial
        solver._drop_cached_decode_state(session.model)
    elapsed = time.monotonic() - start
    row = {**job, "status": status, "finish_reason": reason, "error": error,
           "raw_text": raw, "text": raw, "prompt_tokens": plen, "gen_tokens": ntokens,
           "ended_eos": ended, "elapsed_s": round(elapsed, 6), "physical_elapsed_s": round(elapsed, 6)}
    result = session.journal.append(full_identity, row)
    event("text_done", job=job, status=status, gen_tokens=ntokens, elapsed_s=round(elapsed, 3))
    return result
