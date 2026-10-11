"""011 finite per-task LoRA adaptation for an already loaded shared model.

Call ensure_ttt_adapter ONCE for the selected initialization, then adapt_task
for each task. This module never loads the 27B base or reads query solutions.
Its CLI only prints a CPU plan; GPU work is called by the shared-session runner.
Imports remain torch-free until an adaptation API is called. Generation and
post-hoc scoring belong to the caller, after updating its weight/cache identity.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import random
import sys
import time

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Torch-free reuse (both modules import no torch at top level).
from solve_grid import grids_equal as _solve_grids_equal  # noqa: E402
from solve_grid import task_view_seed as _task_view_seed  # noqa: E402
from train_grid_sft import build_sequence as _train_build_sequence  # noqa: E402
from train_grid_sft import lr_multiplier as _train_lr_multiplier  # noqa: E402
import train_grid_sft as _trainer  # noqa: E402
import arc_episode_data as _episodes  # noqa: E402

DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(_SCRIPT_DIR), "experiments", "010", "evaluation-manifest.json"
)
# Fallback when invoked from another cwd: path relative to this file.
if not os.path.isfile(DEFAULT_MANIFEST):
    DEFAULT_MANIFEST = "experiments/010/evaluation-manifest.json"
DEFAULT_MANIFEST_KEY = "TTT_P6"
DEFAULT_STEPS = "0,4"
ALLOWED_STEPS = (0, 4)
DEFAULT_GRAD_ACCUM = 4
DEFAULT_SEED = 20261011
PROMPT_LABEL = -100


def parse_ttt_steps(text):
    """The first 011 run implements only A0/A4, not an unverified A8 resume."""
    parts = [p.strip() for p in str(text or "").split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("--ttt-steps is empty")
    steps = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise ValueError("--ttt-steps must be integers from {0,4}, got %r" % part)
        if value not in ALLOWED_STEPS:
            raise ValueError("--ttt-steps allows only 0/4, got %d" % value)
        if value not in steps:
            steps.append(value)
    return sorted(steps)


def load_ttt_p6(manifest_path, key=DEFAULT_MANIFEST_KEY):
    """Load the frozen TTT P6 slice: 6 tasks / 6 queries (H_new 3/3 + E24 3/3).

    Reads input-side metadata only (task_id, n_demo, n_test_queries, tier).
    Never touches query answers; solutions stay in the scorer, not here.
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if key not in manifest:
        raise ValueError("manifest key %r not found in %s" % (key, manifest_path))
    block = manifest[key]
    entries = []
    for group in ("H_new", "E24"):
        for item in block.get(group, {}).get("entries", []):
            entries.append(
                {
                    "task_id": str(item["task_id"]),
                    "group": group,
                    "tier": str(item.get("tier", "")),
                    "n_demo": int(item.get("n_demo", 0)),
                    "n_test_queries": int(item.get("n_test_queries", 0)),
                }
            )
    total_tasks = int(block.get("total_tasks", len(entries)))
    total_queries = int(block.get("total_queries", 0))
    if len(entries) != total_tasks:
        raise ValueError(
            "TTT slice task count diverged: %d entries vs total_tasks=%d"
            % (len(entries), total_tasks)
        )
    return {
        "manifest": os.path.abspath(manifest_path),
        "key": key,
        "entries": entries,
        "n_tasks": len(entries),
        "n_queries": sum(e["n_test_queries"] for e in entries),
        "manifest_total_queries": total_queries,
    }


def per_task_seed(base_seed, task_id):
    """Fixed per-task seed derived from the base seed + task_id (stable hash)."""
    digest = hashlib.sha256(
        ("%d|%s" % (int(base_seed), str(task_id))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def fresh_optimizer_spec(seed):
    """Fresh-optimizer marker: rebuilt from scratch per task, never carried over.

    Returns the config dict the GPU loop must use when constructing a new
    optimizer/scheduler/cursor on top of the same offline checkpoint.
    """
    return {
        "reset": True,
        "carryover_updates": 0,
        "seed": int(seed),
        "note": "same offline checkpoint, fresh optimizer/scheduler/cursor per task",
    }


def plan_runs(entries, steps, grad_accum, base_seed):
    """Per-task adaptation plan. Multi-query tasks share one adaptation.

    Cost is counted per task (one reset + one adaptation), not per query.
    microsteps = updates * grad_accum (microbatch is always 1).
    """
    plans = []
    for entry in entries:
        task_seed = per_task_seed(base_seed, entry["task_id"])
        per_step = {}
        for step in steps:
            per_step[str(step)] = {
                "optimizer_updates": int(step),
                "grad_accum": int(grad_accum),
                "microsteps": int(step) * int(grad_accum),
                "optimizer": fresh_optimizer_spec(task_seed),
                "view_seed": int(_task_view_seed(base_seed, entry["task_id"], 0, "orig", 0)),
            }
        plans.append(
            {
                "task_id": entry["task_id"],
                "group": entry["group"],
                "tier": entry["tier"],
                "n_demo": entry["n_demo"],
                "n_test_queries": entry["n_test_queries"],
                "task_seed": task_seed,
                "cost_unit": "task",
                "steps": per_step,
            }
        )
    return plans


def build_loo_train_contexts(demos, task_id=""):
    """LOO demonstration contexts for TTT. Query outputs never enter here.

    Each held-out demo becomes the supervised target once; the remaining demos
    form the context. Unlike the offline min_loo_demos=2 filter, a 2-demo task
    leaving a single context demo is still usable: it is kept and flagged with
    used_single_context=True instead of skipping the whole task. Tasks with
    fewer than 2 demos yield no contexts (nothing to leave out).
    """
    demos = list(demos or [])
    if len(demos) < 2:
        return []
    contexts = []
    for held in range(len(demos)):
        context = [demos[i] for i in range(len(demos)) if i != held]
        contexts.append(
            {
                "task_id": str(task_id),
                "held_out_index": int(held),
                "n_context": len(context),
                "used_single_context": len(context) == 1,
                "target_demo_index": int(held),
            }
        )
    return contexts


def mask_query_targets(labels, prompt_len):
    """Training collator rule: hidden query outputs must stay masked.

    Query spans keep label -100 so no query gold ever trains. Pure function so
    the CPU dry-run can exercise it without torch. Reuses the same -100
    prompt convention as train_grid_sft.build_sequence.
    """
    masked = [PROMPT_LABEL if i < int(prompt_len) else int(v) for i, v in enumerate(labels)]
    # Everything at/after prompt_len is a query-output position here -> mask it.
    for i in range(int(prompt_len), len(masked)):
        masked[i] = PROMPT_LABEL
    return masked


def collate_ttt_example(prompt_ids, target_ids, eos_id, is_query_without_target=False):
    """Build one TTT training row, or skip (None) when the target is missing.

    Demonstrations/LOO go through build_sequence (answer-only supervision).
    A query row without a target (hidden gold unavailable at adapt time) is
    skipped without error -> returns None. A query row *with* a target is only
    used for post-hoc scoring; its labels are fully masked for training.
    """
    if is_query_without_target or target_ids is None:
        return None
    input_ids, labels, attention = _train_build_sequence(prompt_ids, target_ids, eos_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention,
    }


def empty_trial_record(step):
    """Shared per-step trial schema (also used by the no-TTT second-view arm).

    Timing is split into train_s / gen_s / rescore_s so the equal-wall-clock
    comparison in 008 6.2 can trade TTT updates against one extra view.
    final_train_loss is recorded only; low loss never routes or filters
    (low_loss_routed stays False).
    """
    return {
        "optimizer_updates": int(step),
        "exact_match": None,
        "train_s": None,
        "gen_s": None,
        "rescore_s": None,
        "final_train_loss": None,
        "low_loss_routed": False,
    }


def summarize_task(task_plan, used_single_context):
    """Per-task result row sharing the second-view comparison schema."""
    trials = {name: empty_trial_record(name) for name in task_plan["steps"]}
    return {
        "task_id": task_plan["task_id"],
        "group": task_plan["group"],
        "tier": task_plan["tier"],
        "n_demo": task_plan["n_demo"],
        "n_test_queries": task_plan["n_test_queries"],
        "task_seed": task_plan["task_seed"],
        "used_single_context": bool(used_single_context),
        "trials": trials,
        "net_new": None,
        "lost": None,
    }


def build_output_payload(slice_info, plans, steps, grad_accum, base_seed, out_path=""):
    """Output schema shared with the no-TTT second-view comparison arm."""
    per_task = []
    for plan, entry in zip(plans, slice_info["entries"]):
        loo = build_loo_train_contexts([{"demo": i} for i in range(entry["n_demo"])],
                                       task_id=entry["task_id"])
        single = any(c["used_single_context"] for c in loo)
        per_task.append(summarize_task(plan, single))
    total_micro = {}
    for step in steps:
        total_micro[str(step)] = sum(
            p["steps"][str(step)]["microsteps"] for p in plans
        )
    return {
        "experiment": "ttt_per_task",
        "spec": "011: 0/4 updates, accum %d, per-task reset" % int(grad_accum),
        "manifest": slice_info["manifest"],
        "manifest_key": slice_info["key"],
        "n_tasks": slice_info["n_tasks"],
        "n_queries": slice_info["n_queries"],
        "ttt_steps": [int(s) for s in steps],
        "grad_accum": int(grad_accum),
        "seed": int(base_seed),
        "cost_unit": "task",
        "conditional_microsteps": total_micro,
        "per_task": per_task,
        "aggregates": {
            "net_new": None,
            "lost": None,
            "note": "net_new/lost vs 0-step on identical queries; scored post-hoc only",
        },
        "scoring_note": "query targets used for post-hoc scoring only, never in adaptation",
        "out": out_path,
    }


def _torch(torch_module):
    if torch_module is None:
        import torch
        return torch
    return torch_module


def _adapter_params(model, adapter_name):
    markers = (".lora_A.%s." % adapter_name, ".lora_B.%s." % adapter_name)
    result = {name: param for name, param in model.named_parameters()
              if any(marker in "." + name for marker in markers)}
    if not result:
        raise RuntimeError("no linear LoRA A/B parameters for adapter %r" % adapter_name)
    return result


def _validate_adapter_config(model, adapter_name):
    config = getattr(model, "peft_config", {}).get(adapter_name)
    if config is None:
        raise ValueError("adapter %r is not loaded" % adapter_name)
    expected = set(_trainer.DEFAULT_TARGETS)
    if (int(config.r) != 8 or int(config.lora_alpha) != 32
            or float(config.lora_dropout) != 0.0 or config.bias != "none"
            or set(config.target_modules) != expected
            or getattr(config, "rank_pattern", {}) or getattr(config, "alpha_pattern", {})
            or getattr(config, "use_dora", False) or getattr(config, "use_rslora", False)
            or getattr(config, "modules_to_save", None)):
        raise ValueError("011 TTT requires the existing r8/alpha32/dropout0 standard targets; "
                         "refusing to silently change the adapter recipe")


def _activate(model, adapter_name):
    model.set_adapter(adapter_name)
    tuner = getattr(model, "base_model", None)
    if tuner is not None and hasattr(tuner, "enable_adapter_layers"):
        tuner.enable_adapter_layers()


def _state_sha256(weights, torch_module):
    digest = hashlib.sha256()
    for name in sorted(weights):
        value = weights[name].detach().to("cpu").contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str((tuple(value.shape), str(value.dtype))).encode("ascii"))
        digest.update(value.view(torch_module.uint8).numpy().tobytes())
    return digest.hexdigest()


def capture_initial_adapter_state(model, adapter_name=None, torch_module=None):
    """Small CPU LoRA snapshot; capture once, before any task-specific updates.

    The returned dict contains tensors and is intentionally NOT a JSON record.
    No base-model tensors are copied. Keep it unchanged for all task resets.
    """
    torch_module = _torch(torch_module)
    if adapter_name is None:
        active = getattr(model, "active_adapters", None)
        if isinstance(active, str):
            active = [active]
        if not active or len(active) != 1:
            raise ValueError("capture requires one explicit active adapter")
        adapter_name = active[0]
    _validate_adapter_config(model, adapter_name)
    weights = {name: param.detach().to("cpu").clone()
               for name, param in _adapter_params(model, adapter_name).items()}
    return {"adapter_name": adapter_name, "weights": weights,
            "adapter_sha256": _state_sha256(weights, torch_module),
            "trainable_params": sum(value.numel() for value in weights.values())}


def restore_initial_adapter_state(model, initial_state, torch_module=None):
    """Restore all and only the selected LoRA tensors, including nonzero SFT B."""
    torch_module = _torch(torch_module)
    name = initial_state["adapter_name"]
    _validate_adapter_config(model, name)
    params = _adapter_params(model, name)
    weights = initial_state["weights"]
    if set(params) != set(weights):
        raise RuntimeError("adapter tensor names changed since the initial snapshot")
    for key, param in params.items():
        if tuple(param.shape) != tuple(weights[key].shape):
            raise RuntimeError("adapter shape changed: %s" % key)
    _activate(model, name)
    with torch_module.no_grad():
        for key, param in params.items():
            param.copy_(weights[key].to(device=param.device, dtype=param.dtype))
    model.zero_grad(set_to_none=True)
    return initial_state["adapter_sha256"]


def ensure_ttt_adapter(model, adapter_name="ttt_base", source_adapter=None,
                       seed=DEFAULT_SEED, torch_module=None):
    """Return (possibly PEFT-wrapped model, initial_snapshot), once per session.

    For a base arm, source_adapter=None creates a new adapter exactly once with
    PEFT's normal random A / zero B initialization. Never clear both A and B.
    For an SFT arm, source_adapter names an already loaded adapter; its actual
    values are copied into a separate working adapter. The runtime's original
    file-backed adapter stays unchanged, including its cached identity. Existing
    working names are rejected; retain and reuse the first snapshot.
    """
    torch_module = _torch(torch_module)
    if adapter_name in getattr(model, "peft_config", {}):
        raise ValueError("fresh TTT adapter already exists; reuse the original snapshot")
    if source_adapter is not None:
        source = capture_initial_adapter_state(model, source_adapter, torch_module)
        config = copy.deepcopy(model.peft_config[source_adapter])
        config.inference_mode = False
        devices = list(range(torch_module.cuda.device_count())) if torch_module.cuda.is_available() else []
        with torch_module.random.fork_rng(devices=devices):
            model.add_adapter(adapter_name, config)
        weights = {}
        for key, value in source["weights"].items():
            for leaf in ("lora_A", "lora_B"):
                key = key.replace(".%s.%s." % (leaf, source_adapter),
                                  ".%s.%s." % (leaf, adapter_name))
            weights[key] = value
        initial = {"adapter_name": adapter_name, "weights": weights,
                   "adapter_sha256": _state_sha256(weights, torch_module),
                   "trainable_params": source["trainable_params"],
                   "source_adapter_name": source_adapter,
                   "source_adapter_sha256": source["adapter_sha256"]}
        restore_initial_adapter_state(model, initial, torch_module)
        prepare_ttt_eval(model)
        return model, initial
    from peft import LoraConfig, get_peft_model
    config = LoraConfig(r=8, lora_alpha=32, lora_dropout=0.0, bias="none",
                        task_type="CAUSAL_LM", target_modules=list(_trainer.DEFAULT_TARGETS),
                        inference_mode=False, init_lora_weights=True)
    devices = list(range(torch_module.cuda.device_count())) if torch_module.cuda.is_available() else []
    with torch_module.random.fork_rng(devices=devices):
        torch_module.manual_seed(int(seed))
        if getattr(model, "peft_config", None):
            model.add_adapter(adapter_name, config)
        else:
            model = get_peft_model(model, config, adapter_name=adapter_name)
    _activate(model, adapter_name)
    initial = capture_initial_adapter_state(model, adapter_name, torch_module)
    a_tensors = [v for k, v in initial["weights"].items() if ".lora_A." in k]
    b_tensors = [v for k, v in initial["weights"].items() if ".lora_B." in k]
    if not a_tensors or not b_tensors or not any(bool(v.count_nonzero()) for v in a_tensors):
        raise RuntimeError("fresh LoRA A initialization is unexpectedly zero or missing")
    if any(bool(v.count_nonzero()) for v in b_tensors):
        raise RuntimeError("fresh LoRA B initialization is not zero")
    prepare_ttt_eval(model)
    return model, initial


def _copy_grid(grid):
    if not isinstance(grid, (list, tuple)) or not grid:
        raise ValueError("a demonstration grid must be a nonempty rectangle")
    copied = [list(row) for row in grid]
    width = len(copied[0])
    if not width or any(len(row) != width for row in copied):
        raise ValueError("a demonstration grid is not rectangular")
    if any(not isinstance(cell, int) or isinstance(cell, bool) or not 0 <= cell <= 9
           for row in copied for cell in row):
        raise ValueError("demonstration cells must be integer colors 0..9")
    return copied


def build_ttt_episodes(tokenizer, demos, task_id, formatter="f1_compact",
                       eos_id=None, max_len=4096, seed=DEFAULT_SEED):
    """Torch-free, answer-only LOO preparation using only an explicit demo list.

    Any overlength view makes the task ineligible. No demo/target truncation,
    no per-row skipping, and no query/gold file argument are supported.
    """
    if not isinstance(demos, (list, tuple)):
        raise TypeError("pass task['train'] only, not an entire task or solutions map")
    formatter = _trainer._normalize_train_formatter(formatter)
    eos_id = getattr(tokenizer, "eos_token_id", None) if eos_id is None else eos_id
    if eos_id is None or not isinstance(eos_id, int):
        raise ValueError("one integer EOS token id is required")
    pairs = [{"input": _copy_grid(pair["input"]), "output": _copy_grid(pair["output"])}
             for pair in demos]
    report = {"task_id": str(task_id), "formatter": formatter, "n_demo": len(pairs),
              "used_single_context": len(pairs) == 2, "max_len": int(max_len),
              "episodes": [], "query_gold_used": False, "status": "complete"}
    if len(pairs) < 2:
        report.update(status="ineligible", reason="fewer_than_two_demonstrations")
        return [], report
    order = list(range(len(pairs)))
    random.Random(per_task_seed(seed, task_id)).shuffle(order)
    pool = []
    for view in ("identity", "transpose"):
        transformed = []
        for pair in pairs:
            transformed.append({key: ([list(row) for row in zip(*grid)] if view == "transpose"
                                      else [list(row) for row in grid])
                                for key, grid in pair.items()})
        for held in order:
            context = [pair for index, pair in enumerate(transformed) if index != held]
            target = transformed[held]
            row = {"episode_id": "%s|ttt_loo|%d|%s" % (task_id, held, view),
                   "task_id": str(task_id), "family_id": str(task_id),
                   "source": "task_demonstrations_only", "revision": "011-loo-v1",
                   "formatter": formatter,
                   "messages": _episodes.build_messages(context, target["input"], formatter),
                   "target_grid_text": _trainer._train_target_text(target["output"], formatter, _episodes)}
            episode, flag = _trainer.encode_episode(tokenizer, row, eos_id, len(pool) + 1)
            if episode is None:
                raise ValueError("TTT encode failed %s: %s" % (row["episode_id"], flag))
            _trainer.validate_suffix_contract(episode.labels, episode.prompt_len, episode.target_len, 1)
            report["episodes"].append({"episode_id": episode.episode_id, "held_demo": held,
                                       "view": view, "n_context": len(context),
                                       "prompt_tokens": episode.prompt_len,
                                       "target_tokens": episode.target_len, "full_tokens": episode.full_len})
            pool.append(episode)
    long_ids = [episode.episode_id for episode in pool if episode.full_len > int(max_len)]
    if long_ids:
        report.update(status="ineligible", reason="complete_loo_exceeds_max_len",
                      overlength_episode_ids=long_ids)
    return pool, report


def _core_model(model):
    return model.get_base_model() if hasattr(model, "get_base_model") else model


def prepare_ttt_eval(model):
    """End in inference mode; caller must also invalidate its external caches."""
    core = _core_model(model)
    _trainer.disable_checkpointing(model)
    if hasattr(core, "_require_grads_hook") and hasattr(core, "disable_input_require_grads"):
        core.disable_input_require_grads()
    _trainer.set_use_cache(model, True)
    for holder in (model, getattr(model, "base_model", None), core):
        if holder is not None and "_cache" in vars(holder):
            holder._cache = None
    model.zero_grad(set_to_none=True)
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()


def adapt_task(model, tokenizer, task_id, demos, initial_state, formatter="f1_compact",
               updates=4, grad_accum=DEFAULT_GRAD_ACCUM, deadline=None, eos_id=None,
               max_len=4096, seed=DEFAULT_SEED, on_update=None, torch_module=None,
               boundary_reserve_s=30.0, first_update_estimate_s=120.0):
    """Apply A0/A4 to an existing model and return a JSON-serializable record.

    deadline is absolute time.monotonic(). Checks occur at complete update
    boundaries, with a reserve/admission estimate; a CUDA call itself cannot
    be preempted here. Ineligible/interrupted/failed attempts restore A0, never
    silently score a partial adapter as A4. Runtime failures are recorded and
    rolled back; invalid API/configuration arguments raise before training.
    on_update(record), if supplied, can append each complete update immediately.
    """
    if int(updates) not in ALLOWED_STEPS or int(grad_accum) != DEFAULT_GRAD_ACCUM:
        raise ValueError("011 first session supports 0/4 updates with grad_accum=4 only")
    if int(max_len) < 1 or boundary_reserve_s < 0 or first_update_estimate_s < 0:
        raise ValueError("invalid sequence/deadline limits")
    torch_module = _torch(torch_module)
    if torch_module.is_inference_mode_enabled():
        raise RuntimeError("adapt_task must be called outside torch.inference_mode()")
    started = time.monotonic()
    adapter_name = initial_state["adapter_name"]
    result = {"task_id": str(task_id), "adapter_name": adapter_name,
              "initial_adapter_sha256": initial_state["adapter_sha256"],
              "adapter_sha256": initial_state["adapter_sha256"],
              "requested_updates": int(updates), "completed_updates": 0, "applied_updates": 0,
              "grad_accum": int(grad_accum), "microbatch": 1, "lr": 2e-5,
              "weight_decay": 0.01, "clip_norm": 1.0, "logits_mode": "suffix",
              "status": "complete", "fallback_to_a0": False, "update_log": [],
              "train_s": 0.0, "final_train_loss": None, "low_loss_routed": False,
              "query_gold_used": False, "seed": per_task_seed(seed, task_id)}
    optimizer = None
    output = scaled = input_ids = labels = attention = None
    core = _core_model(model)
    restore_initial_adapter_state(model, initial_state, torch_module)
    result["reset_s"] = time.monotonic() - started
    try:
        if not updates:
            return result
        prep_tick = time.monotonic()
        pool, preparation = build_ttt_episodes(tokenizer, demos, task_id, formatter, eos_id, max_len, seed)
        result["preparation"] = preparation
        result["prepare_s"] = time.monotonic() - prep_tick
        if preparation["status"] == "ineligible":
            result.update(status="ineligible", fallback_to_a0=True, reason=preparation["reason"])
            return result
        params_by_name = _adapter_params(model, adapter_name)
        active_ids = {id(param) for param in params_by_name.values()}
        for param in model.parameters():
            param.requires_grad_(id(param) in active_ids)
        in_dev, _unused_lm_dev, _device_map = _trainer.embedding_and_lm_devices(model, torch_module)
        lm_dev = model.get_output_embeddings().weight.device
        if in_dev.type != "cuda" or lm_dev.type != "cuda":
            raise RuntimeError("real TTT requires CUDA input embedding and lm_head")
        prepare_ttt_eval(model)
        for param in params_by_name.values():
            param.requires_grad_(True)
        _activate(model, adapter_name)
        _trainer.set_use_cache(model, False)
        core.gradient_checkpointing_enable()
        core.enable_input_require_grads()
        model.train()
        optimizer = torch_module.optim.AdamW(list(params_by_name.values()), lr=2e-5,
                                            weight_decay=0.01)
        devices = list(range(torch_module.cuda.device_count()))
        cursor = 0
        with torch_module.random.fork_rng(devices=devices), torch_module.enable_grad():
            torch_module.manual_seed(result["seed"])
            for update in range(1, int(updates) + 1):
                estimate = (max(row["seconds"] for row in result["update_log"]) * 1.2
                            if result["update_log"] else float(first_update_estimate_s))
                if deadline is not None and time.monotonic() + estimate + boundary_reserve_s >= deadline:
                    result.update(status="budget_interrupted", fallback_to_a0=True,
                                  reason="deadline_before_complete_update")
                    break
                optimizer.zero_grad(set_to_none=True)
                tick = time.monotonic()
                losses, episode_ids = [], []
                for _micro in range(int(grad_accum)):
                    episode = pool[cursor % len(pool)]
                    cursor += 1
                    input_ids = _trainer.ids_to_tensor(torch_module, episode.input_ids, in_dev)
                    labels = _trainer.ids_to_tensor(torch_module, episode.labels, lm_dev)
                    attention = torch_module.ones_like(input_ids)
                    output = _trainer.suffix_forward(model, input_ids, attention, labels,
                                                     episode.prompt_len, episode.target_len)
                    if output.loss is None:
                        raise RuntimeError("TTT forward returned no loss")
                    loss_value = float(output.loss.detach().float().item())
                    if not math.isfinite(loss_value):
                        raise RuntimeError("nonfinite TTT loss")
                    scaled = output.loss / float(grad_accum)
                    scaled.backward()
                    losses.append(loss_value)
                    episode_ids.append(episode.episode_id)
                    output = scaled = input_ids = labels = attention = None
                grad_params = [param for param in params_by_name.values() if param.grad is not None]
                if not grad_params:
                    raise RuntimeError("no active-adapter gradients")
                grad_norm = torch_module.nn.utils.clip_grad_norm_(grad_params, 1.0)
                if not bool(torch_module.isfinite(grad_norm)):
                    raise RuntimeError("nonfinite TTT gradient norm")
                optimizer.step()
                _trainer._sync_all_gpus(torch_module)
                record = {"task_id": str(task_id), "update": update, "microsteps": int(grad_accum),
                          "episode_ids": episode_ids, "mean_loss": sum(losses) / len(losses),
                          "grad_norm": float(grad_norm.detach().float().item()),
                          "lr": 2e-5, "seconds": time.monotonic() - tick, "cursor": cursor}
                result["update_log"].append(record)
                result["completed_updates"] = update
                result["train_s"] += record["seconds"]
                result["final_train_loss"] = record["mean_loss"]
                print("TTT_UPDATE " + json.dumps(record, ensure_ascii=False), flush=True)
                if on_update is not None:
                    on_update(dict(record))
            if result["completed_updates"] == int(updates):
                result["applied_updates"] = int(updates)
                result["adapter_sha256"] = _state_sha256(params_by_name, torch_module)
    except Exception as exc:
        result.update(status="infrastructure_error", fallback_to_a0=True,
                      reason="out_of_memory" if "out of memory" in str(exc).lower() else "adaptation_failed",
                      error_type=type(exc).__name__, error=str(exc)[:1000])
    finally:
        output = scaled = input_ids = labels = attention = None
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        optimizer = None
        if result["fallback_to_a0"]:
            restore_initial_adapter_state(model, initial_state, torch_module)
            result["applied_updates"] = 0
            result["adapter_sha256"] = initial_state["adapter_sha256"]
        prepare_ttt_eval(model)
        if torch_module.cuda.is_available():
            torch_module.cuda.empty_cache()
        result["elapsed_s"] = time.monotonic() - started
    return result


def ttt_adapt_and_generate(*args, generate_fn=None, **kwargs):
    """Compatibility wrapper; caller supplies generation and invalidates caches.

    generate_fn receives the adaptation record so the shared runtime can mark
    its adapter identity before emitting candidates, including A0 fallback.
    """
    result = adapt_task(*args, **kwargs)
    if generate_fn is not None:
        result["generation"] = generate_fn(result)
    return result


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "011 shared-session TTT. CLI prints the frozen CPU A0/A4 plan; "
            "call ensure_ttt_adapter/adapt_task from the GPU runner."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="evaluation-manifest.json path.")
    parser.add_argument("--manifest-key", default=DEFAULT_MANIFEST_KEY,
                        help="Manifest slice key (default TTT_P6).")
    parser.add_argument("--ttt-steps", default=DEFAULT_STEPS,
                        help="Comma-separated updates from {0,4}. Default '0,4'.")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM,
                        help="011 gradient accumulation is fixed at 4; microbatch is 1.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base seed; per-task seeds derive via sha256(seed|task_id).")
    parser.add_argument("--checkpoint", default="",
                        help="Legacy plan metadata only; shared-session caller loads the checkpoint.")
    parser.add_argument("--challenges", default="",
                        help="Legacy plan metadata only; caller passes an explicit demos list.")
    parser.add_argument("--out", default="",
                        help="Output JSON path for the shared-schema result skeleton.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pure-CPU plan: parse the P6 slice, print the 0/4 plan, "
                             "no model load, no training.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    steps = parse_ttt_steps(args.ttt_steps)
    if int(args.grad_accum) != DEFAULT_GRAD_ACCUM:
        raise SystemExit("011 --grad-accum must be 4")
    if not os.path.isfile(args.manifest):
        raise SystemExit("manifest not found: %s" % args.manifest)
    slice_info = load_ttt_p6(args.manifest, args.manifest_key)
    plans = plan_runs(slice_info["entries"], steps, int(args.grad_accum), int(args.seed))
    if args.dry_run:
        lines = [
            "TTT dry-run: %s tasks=%d queries=%d steps=%s accum=%d seed=%d"
            % (slice_info["key"], slice_info["n_tasks"], slice_info["n_queries"],
               ",".join(str(s) for s in steps), int(args.grad_accum), int(args.seed))
        ]
        for plan in plans:
            micro = ",".join(
                "%s:%d" % (s, plan["steps"][s]["microsteps"]) for s in plan["steps"]
            )
            loo_n = max(0, plan["n_demo"] - 1)
            single = " used_single_context" if plan["n_demo"] == 2 else ""
            lines.append(
                "  %s [%s/%s] demos=%d queries=%d seed=%d {%s} loo_ctx=%d%s"
                % (plan["task_id"], plan["group"], plan["tier"], plan["n_demo"],
                   plan["n_test_queries"], plan["task_seed"], micro, loo_n, single)
            )
        total4 = sum(p["steps"].get("4", {}).get("microsteps", 0) for p in plans)
        lines.append("conditional_microsteps_if_4_updates_accum%d=%d" % (int(args.grad_accum), total4))
        lines.append("no model loaded; no training; query targets never used in adaptation")
        print("\n".join(lines), flush=True)
        if args.out:
            payload = build_output_payload(
                slice_info, plans, steps, int(args.grad_accum), int(args.seed), args.out)
            parent = os.path.dirname(os.path.abspath(args.out))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print("wrote skeleton: %s" % args.out, flush=True)
        return 0
    raise SystemExit(
        "Use the shared-session GPU runner with ensure_ttt_adapter/adapt_task; "
        "this CLI deliberately does not load another base model. Use --dry-run for a CPU plan."
    )


if __name__ == "__main__":
    raise SystemExit(main())
