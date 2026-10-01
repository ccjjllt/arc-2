"""G1 trainer: Qwen3.8-27B BF16 LoRA that supervises the answer grid only.

Manual Kaggle L4x4 entry point. Importing this file does not import torch.
There is no FSDP, DDP, Unsloth, 4-bit, or tensor-parallel path.

  python3 scripts/train_grid_sft.py --help
"""

from __future__ import annotations

import argparse
import array
import glob
import inspect
import json
import os
import random
import sys
import time
import traceback

FORMATTER_NAME = "space_digits_v1"
DEFAULT_TARGETS = (
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
)
# Not part of the G1 default. Only added when --extra-targets names them.
GDN_TARGET_NAMES = ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a", "in_proj_b")
DEFAULT_ZERO_STEP_IDS = (
    "833966f4,6150a2bd,e9afcf9a,5582e5ca,d037b0a7,d631b094,66e6c45b,00576224"
)
ZERO_STEP_TIME_GUARD_S = 1200
HOLDOUT_SPLITS = {
    "holdout",
    "eval",
    "evaluation",
    "test",
    "val",
    "valid",
    "validation",
    "dev",
}
LR_FORMULA = (
    "1-based update u: if u<=warmup then u/warmup; "
    "else (total-u)/(total-warmup). Update `total` uses learning rate 0."
)
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


class SkipRun(RuntimeError):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


class ResumeMismatch(RuntimeError):
    def __init__(self, message):
        super().__init__(message)
        self.status = "FAILED_RESUME_TARGETS"


def lr_multiplier(update, warmup, total):
    """Learning-rate scale for a 1-based optimizer update.

    Warmup covers updates 1..warmup and does not start at 0.
    The decay anchor is update ``total``, whose scale is 0.
    """
    update = int(update)
    warmup = int(warmup)
    total = int(total)
    if update <= 0 or total <= 0:
        return 0.0
    if warmup <= 0:
        if total <= 1:
            return 1.0
        return max(0.0, float(total - update) / float(total - 1))
    if update <= warmup:
        return float(update) / float(warmup)
    if warmup >= total:
        return 1.0 if update <= total else 0.0
    return max(0.0, float(total - update) / float(total - warmup))


def build_sequence(prompt_ids, target_ids, eos_id):
    """One unshifted sequence. Prompt labels are -100. Target and eos are supervised.

    Hugging Face causal LM shifts labels internally. Do not shift again.
    Pad positions are not inserted here. Do not treat eos as a pad label.
    """
    if eos_id is None:
        raise ValueError("tokenizer eos_token_id is None; refusing to train without an eos id")
    eos_id = int(eos_id)
    prompt_ids = [int(x) for x in prompt_ids]
    target_ids = [int(x) for x in target_ids]
    input_ids = prompt_ids + target_ids + [eos_id]
    labels = ([-100] * len(prompt_ids)) + target_ids + [eos_id]
    attention = [1] * len(input_ids)
    if not (len(input_ids) == len(labels) == len(attention)):
        raise RuntimeError("sequence, labels, and attention_mask lengths diverged")
    return input_ids, labels, attention


def parse_strict_grid(text):
    """Whole stripped generation must be one rectangle of single digits 0-9.

    Rows are newline-separated. Digits in a row are separated by one space.
    Returns the grid, or None. Does not search for a rectangle inside other text.
    """
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if not stripped or "\r" in stripped or "\t" in stripped:
        return None
    rows = stripped.split("\n")
    grid = []
    width = None
    for row in rows:
        parts = row.split(" ")
        if not parts or any(len(part) != 1 or part < "0" or part > "9" for part in parts):
            return None
        if width is None:
            width = len(parts)
        elif len(parts) != width or width == 0:
            return None
        grid.append([ord(part) - 48 for part in parts])
    if not grid or width is None or width == 0:
        return None
    return grid


def select_pool(short, overflow, n_longer, needed):
    """Keep every episode at or under max-len. Top up from overflow only if short.

    Overflow fill is shortest-first, then episode_id. Grids are never truncated.
    ``needed`` is optimizer_updates * grad_accum and is not multiplied by GPU count.
    """
    if len(short) >= int(needed):
        added = []
    else:
        ordered = sorted(overflow, key=lambda ep: (int(ep.full_len), str(ep.episode_id)))
        added = ordered[: int(needed) - len(short)]
    pool = list(short) + list(added)
    families = {str(ep.family_id) for ep in pool}
    stats = {
        "n_kept_le_max_len": len(short),
        "n_added_le_overflow": len(added),
        "n_longer_than_overflow": int(n_longer),
        "n_families_consumed": len(families),
        "pool_size": len(pool),
        "needed_examples": int(needed),
        "overflow_fill": "shortest_then_episode_id" if added else "not_needed",
        "kept_all_episodes_le_max_len": True,
    }
    return pool, stats


def grids_equal(left, right):
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right):
        return False
    for row_a, row_b in zip(left, right):
        if not isinstance(row_a, list) or not isinstance(row_b, list) or len(row_a) != len(row_b):
            return False
        for cell_a, cell_b in zip(row_a, row_b):
            if int(cell_a) != int(cell_b):
                return False
    return True


def parse_id_list(text):
    found = []
    for part in (text or "").split(","):
        item = part.strip()
        if item and item not in found:
            found.append(item)
    return found


def parse_extra_targets(text):
    extras = []
    for part in (text or "").split(","):
        name = part.strip()
        if not name or name in DEFAULT_TARGETS or name in extras:
            continue
        extras.append(name)
    return extras


def dump_json(path, obj, one_line=False):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if one_line:
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"), default=str) + "\n"
    else:
        text = json.dumps(obj, ensure_ascii=False, indent=2, default=str) + "\n"
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def adapter_files_present(adapter_dir):
    if not adapter_dir or not os.path.isdir(adapter_dir):
        return False
    names = set(os.listdir(adapter_dir))
    has_config = "adapter_config.json" in names
    has_weights = any(name.endswith(".safetensors") or name.endswith(".bin") for name in names)
    return has_config and has_weights


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "G1 grid SFT for Qwen3.8-27B. Frozen BF16 base, LoRA, answer-token loss, "
            "single-process device_map=balanced. Does not launch without 4 CUDA GPUs."
        )
    )
    parser.add_argument("--model-dir", default=None, help="Model directory. Default: search /kaggle/input.")
    parser.add_argument("--episodes", required=True, help="Training episodes JSONL.")
    parser.add_argument("--out-dir", required=True, help="Checkpoint and log directory.")
    parser.add_argument("--optimizer-updates", type=int, default=200)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--microbatch", type=int, default=1)
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--overflow-len", type=int, default=2048)
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-updates", type=int, default=10)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--zero-step-ids", default=DEFAULT_ZERO_STEP_IDS)
    parser.add_argument("--zero-step-max-new", type=int, default=960)
    parser.add_argument(
        "--resume",
        default=None,
        help=(
            "Previous out-dir with adapter/ and trainer_state.pt. Exact resume only: pool order, "
            "cursor, optimizer, scheduler, and RNG continue the update counter. "
            "Never combined with --init-adapter."
        ),
    )
    parser.add_argument(
        "--init-adapter",
        default=None,
        help=(
            "Weights-only inheritance for a new stage (008 G2a): an adapter dir, or a previous "
            "out-dir holding adapter/ (+ optional trainer_state.pt/run_config.json for stage "
            "accounting). Pool, cursor, optimizer, and scheduler always restart from zero; old "
            "optimizer/scheduler/cursor are never reused. Same-structure adapters load directly; "
            "changed target modules build a fresh LoRA and copy only intersecting shape-matched "
            "LoRA tensors, leaving new modules at zero-init. Mutually exclusive with --resume "
            "(exact resume keeps old pool/cursor/optimizer/scheduler/RNG). Stage switch records "
            "old vs new training volume in run_config.json. Periodic saves also write immutable "
            "adapter_step_{N} snapshots and never overwrite an existing snapshot "
            "(adapter/ stays the latest pointer for --resume). G1 defaults unchanged: r8/alpha32, "
            "q/k/v/o+MLP targets, BF16, answer-only mask, thinking=False, microbatch=1, accum=4."
        ),
    )
    parser.add_argument(
        "--extra-targets",
        default="",
        help="Comma-separated extra LoRA module names. Default empty. G1 does not add GDN targets.",
    )
    parser.add_argument("--challenges", default=None, help="Optional ARC challenges JSON for zero-step.")
    parser.add_argument("--solutions", default=None, help="Optional ARC solutions JSON for zero-step exact match.")
    return parser


def validate_args(args):
    if args.microbatch != 1:
        raise RuntimeError(
            "microbatch must be 1. Each microstep is one episode. "
            "effective batch = microbatch * grad_accum and is not multiplied by GPU count."
        )
    if args.grad_accum < 1:
        raise RuntimeError("grad-accum must be >= 1")
    if args.optimizer_updates < 1:
        raise RuntimeError("optimizer-updates must be >= 1")
    if args.max_len < 1 or args.overflow_len < args.max_len:
        raise RuntimeError("require 1 <= max-len <= overflow-len")
    if args.save_every < 1:
        raise RuntimeError("save-every must be >= 1")
    if args.warmup_updates < 0:
        raise RuntimeError("warmup-updates must be >= 0")
    if not (args.lr > 0):
        raise RuntimeError("lr must be > 0")
    if not (args.clip_norm > 0):
        raise RuntimeError("clip-norm must be > 0")
    if args.zero_step_max_new < 1:
        raise RuntimeError("zero-step-max-new must be >= 1")
    if not os.path.isfile(args.episodes):
        raise RuntimeError("episodes file not found: %s" % args.episodes)
    if args.challenges and not os.path.isfile(args.challenges):
        raise RuntimeError("challenges file not found: %s" % args.challenges)
    if args.solutions and not os.path.isfile(args.solutions):
        raise RuntimeError("solutions file not found: %s" % args.solutions)
    if args.resume:
        resolve_resume_paths(args.resume)
    if getattr(args, "init_adapter", None) and args.resume:
        raise RuntimeError(
            "--init-adapter and --resume are mutually exclusive: weights-only new stage "
            "vs exact resume of pool/cursor/optimizer/scheduler/RNG"
        )
    if getattr(args, "init_adapter", None):
        resolve_init_adapter_dir(args.init_adapter)


def resolve_resume_paths(resume_dir):
    if not os.path.isdir(resume_dir):
        raise RuntimeError("resume dir not found: %s" % resume_dir)
    adapter = os.path.join(resume_dir, "adapter")
    state = os.path.join(resume_dir, "trainer_state.pt")
    if not os.path.isfile(os.path.join(adapter, "adapter_config.json")):
        raise RuntimeError("resume adapter/adapter_config.json missing under %s" % resume_dir)
    if not os.path.isfile(state):
        raise RuntimeError("resume trainer_state.pt missing under %s" % resume_dir)
    return adapter, state


def resolve_init_adapter_dir(path):
    """Accept an adapter dir or a previous out-dir holding adapter/."""
    if not path:
        raise RuntimeError("init-adapter path is empty")
    direct = os.path.join(path, "adapter_config.json")
    if os.path.isfile(direct):
        return os.path.abspath(path)
    nested = os.path.join(path, "adapter", "adapter_config.json")
    if os.path.isfile(nested):
        return os.path.abspath(os.path.join(path, "adapter"))
    raise RuntimeError(
        "init-adapter has no adapter_config.json as %s nor %s" % (direct, nested)
    )


def read_adapter_targets(adapter_dir):
    try:
        with open(os.path.join(adapter_dir, "adapter_config.json"), "r", encoding="utf-8") as handle:
            config = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError("cannot read init-adapter config: %s" % exc)
    targets = config.get("target_modules")
    if isinstance(targets, dict):
        targets = list(targets.keys())
    return [str(name) for name in (targets or [])]


def load_init_adapter_weights(model, adapter_dir, torch_module):
    """Copy intersecting shape-matched LoRA tensors; never touches base weights or optimizer."""
    names = os.listdir(adapter_dir)
    weight_file = None
    for candidate in ("adapter_model.safetensors", "adapter_model.safetensor"):
        if candidate in names:
            weight_file = os.path.join(adapter_dir, candidate)
            break
    if weight_file is None:
        for name in sorted(names):
            if name.endswith(".safetensors"):
                weight_file = os.path.join(adapter_dir, name)
                break
    use_torch_bin = False
    if weight_file is None:
        for candidate in ("adapter_model.bin", "adapter_model.pytorch_bin"):
            if candidate in names:
                weight_file = os.path.join(adapter_dir, candidate)
                use_torch_bin = True
                break
    if weight_file is None:
        raise RuntimeError("init-adapter has no .safetensors/.bin weights under %s" % adapter_dir)
    if use_torch_bin:
        old_sd = torch_module.load(weight_file, map_location="cpu", weights_only=True)
    else:
        try:
            from safetensors.torch import load_file as _load_file
        except ImportError as exc:
            raise RuntimeError("need safetensors to read init-adapter %s (%s)" % (weight_file, exc))
        old_sd = _load_file(weight_file, device="cpu")
    current = model.state_dict()
    copied = 0
    shape_mismatch = []
    not_in_new = []
    with torch_module.no_grad():
        for key, value in old_sd.items():
            if "lora_" not in key:
                continue
            if key not in current:
                not_in_new.append(key)
                continue
            if tuple(value.shape) != tuple(current[key].shape):
                shape_mismatch.append(key)
                continue
            current[key].copy_(value)
            copied += 1
    try:
        model.load_state_dict(current, strict=False)
    except TypeError:
        pass
    return {
        "weight_file": os.path.basename(weight_file),
        "copied_lora_tensors": copied,
        "shape_mismatch": shape_mismatch[:16],
        "n_shape_mismatch": len(shape_mismatch),
        "not_in_new": not_in_new[:16],
        "n_not_in_new": len(not_in_new),
    }


def old_stage_amount(adapter_dir):
    """Best-effort old training volume from JSON sidecars (torch-free)."""
    parent = os.path.dirname(os.path.abspath(adapter_dir))
    info = {"run_config_path": None, "completed_updates": None, "cursor": None}
    for name in ("run_config.json", "SUMMARY.json"):
        path = os.path.join(parent, name)
        if not os.path.isfile(path):
            continue
        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if info["run_config_path"] is None:
            info["run_config_path"] = path
        for key in ("completed_updates", "cursor"):
            if info[key] is None and isinstance(data, dict) and data.get(key) is not None:
                try:
                    info[key] = int(data[key])
                except (TypeError, ValueError):
                    pass
        inner = data.get("run_config") if isinstance(data, dict) else None
        if isinstance(inner, dict):
            for key in ("completed_updates", "cursor"):
                if info[key] is None and inner.get(key) is not None:
                    try:
                        info[key] = int(inner[key])
                    except (TypeError, ValueError):
                        pass
    return info


def save_adapter_snapshot(model, out_dir, completed):
    """Immutable per-step adapter dir. Never overwrites an existing snapshot."""
    snapshot = os.path.join(out_dir, "adapter_step_%d" % int(completed))
    if adapter_files_present(snapshot):
        return snapshot
    os.makedirs(snapshot, exist_ok=True)
    model.save_pretrained(snapshot)
    if not adapter_files_present(snapshot):
        raise RuntimeError("snapshot save did not write adapter files: %s" % snapshot)
    return snapshot


def find_model_dir(explicit):
    if explicit:
        config_path = os.path.join(explicit, "config.json")
        if not os.path.isfile(config_path):
            raise RuntimeError("model config.json not found under %s" % explicit)
        return explicit
    found = None
    for config_path in sorted(glob.glob("/kaggle/input/**/config.json", recursive=True)):
        try:
            with open(config_path, "r", encoding="utf-8") as handle:
                config = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if config.get("model_type") in ("qwen3_5", "qwen3"):
            found = os.path.dirname(config_path)
            break
    if not found:
        raise RuntimeError("FAILED_NO_MODEL: no --model-dir and no qwen3_5/qwen3 config under /kaggle/input")
    return found


def load_episode_module():
    if _SCRIPT_DIR not in sys.path:
        sys.path.insert(0, _SCRIPT_DIR)
    try:
        import arc_episode_data as module
    except Exception as exc:
        return None, "%s: %s" % (type(exc).__name__, exc)
    return module, None


def read_json(path):
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def load_task_map(path):
    data = read_json(path)
    if isinstance(data, dict):
        for key in ("challenges", "tasks", "data", "solutions"):
            inner = data.get(key)
            if isinstance(inner, dict) and inner:
                return inner
        return data
    if isinstance(data, list):
        mapped = {}
        for index, item in enumerate(data):
            if not isinstance(item, dict):
                continue
            task_id = str(item.get("task_id") or item.get("id") or index)
            mapped[task_id] = item
        return mapped
    raise RuntimeError("unrecognized challenges/solutions JSON: %s" % path)


def solution_grid(solutions, task_id):
    if not isinstance(solutions, dict) or task_id not in solutions:
        return None
    item = solutions[task_id]
    if not isinstance(item, list) or not item:
        return None
    grid = item[0]
    if isinstance(grid, dict) and "output" in grid:
        grid = grid["output"]
    if not isinstance(grid, list):
        return None
    return grid


def coerce_messages(result, system_prompt):
    if isinstance(result, list) and result and isinstance(result[0], dict) and "role" in result[0]:
        return result
    if isinstance(result, dict) and isinstance(result.get("messages"), list):
        return result["messages"]
    if isinstance(result, str):
        if not system_prompt:
            raise RuntimeError("build_messages returned text but SYSTEM_PROMPT is missing")
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": result},
        ]
    raise RuntimeError("build_messages returned %s; expected chat messages" % type(result).__name__)


def messages_for_test_query(module, tokenizer, task_id, task, solutions):
    """Official test query 0. Train pairs are demos. The answer is not in the prompt.

    ``tokenizer`` and ``solutions`` are unused here. Exact match reads solutions later.
    """
    del tokenizer, solutions
    builder = getattr(module, "build_messages", None)
    if not callable(builder):
        raise RuntimeError("arc_episode_data message builder is missing")
    system_prompt = getattr(module, "SYSTEM_PROMPT", None)
    if not isinstance(task, dict):
        raise RuntimeError("task %s is not an object" % task_id)
    train_pairs = task.get("train")
    tests = task.get("test")
    if not isinstance(train_pairs, list) or not isinstance(tests, list) or not tests:
        raise RuntimeError("task %s has no train list or test query" % task_id)
    query = tests[0]
    if not isinstance(query, dict) or "input" not in query:
        raise RuntimeError("task %s test[0] has no input" % task_id)
    demos = []
    for pair in train_pairs:
        if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
            raise RuntimeError("task %s has a train pair without input and output" % task_id)
        demos.append({"input": pair["input"], "output": pair["output"]})
    return coerce_messages(builder(demos, query["input"]), system_prompt)


def render_prompt(tokenizer, messages):
    # Do not inspect the signature for enable_thinking. 006 skipped the flag because
    # the real signature is **kwargs, and the template then left thinking on.
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(prompt, str):
        raise RuntimeError(
            "apply_chat_template(tokenize=False, enable_thinking=False) returned %s, expected str. "
            "Not retrying without enable_thinking." % type(prompt).__name__
        )
    return prompt


def tokenize_text(tokenizer, text):
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded["input_ids"]
    if hasattr(token_ids, "tolist"):
        token_ids = token_ids.tolist()
    if token_ids and isinstance(token_ids[0], (list, tuple)):
        if len(token_ids) != 1:
            raise RuntimeError("tokenizer returned a batch for one string")
        token_ids = list(token_ids[0])
    return [int(token_id) for token_id in token_ids]


class Episode(object):
    __slots__ = (
        "episode_id",
        "task_id",
        "family_id",
        "input_ids",
        "labels",
        "prompt_len",
        "target_len",
        "target_grid_text",
        "full_len",
        "source",
        "revision",
        "prompt_text",
    )

    def __init__(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


def encode_episode(tokenizer, row, eos_id, line_no):
    messages = row.get("messages")
    target = row.get("target_grid_text")
    if not isinstance(messages, list) or not messages:
        return None, "bad_messages"
    if not isinstance(target, str) or target == "":
        return None, "bad_target"
    prompt = render_prompt(tokenizer, messages)
    prompt_ids = tokenize_text(tokenizer, prompt)
    target_ids = tokenize_text(tokenizer, target)
    if not prompt_ids:
        return None, "empty_prompt"
    if not target_ids:
        return None, "empty_target_tokens"
    input_ids, labels, _attention = build_sequence(prompt_ids, target_ids, eos_id)
    if len(input_ids) == 0:
        return None, "empty"
    task_id = str(row.get("task_id") or "")
    family = row.get("family_id")
    family_filled = family is None or family == ""
    if family_filled:
        family = task_id
    episode_id = row.get("episode_id")
    if episode_id is None or episode_id == "":
        episode_id = "%s:%s:%s:%s" % (
            task_id,
            row.get("query_kind"),
            row.get("query_index"),
            line_no,
        )
    episode = Episode(
        episode_id=str(episode_id),
        task_id=task_id,
        family_id=str(family),
        input_ids=array.array("i", input_ids),
        labels=array.array("i", labels),
        prompt_len=len(prompt_ids),
        target_len=len(target_ids) + 1,
        target_grid_text=target,
        full_len=len(input_ids),
        source=row.get("source"),
        revision=row.get("revision"),
        prompt_text=prompt,
    )
    return episode, ("family_filled" if family_filled else None)


def load_jsonl_rows(path):
    rows = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, 1):
            text = line.strip()
            if not text:
                continue
            try:
                row = json.loads(text)
            except json.JSONDecodeError as exc:
                raise RuntimeError("episodes JSONL line %d is not JSON: %s" % (line_no, exc))
            if not isinstance(row, dict):
                raise RuntimeError("episodes JSONL line %d is not an object" % line_no)
            row["_line_no"] = line_no
            rows.append(row)
    return rows


def row_is_trainable(row, diagnostic_ids):
    split = row.get("split")
    if isinstance(split, str) and split.strip().lower() in HOLDOUT_SPLITS:
        return False, "split"
    task_id = str(row.get("task_id") or "")
    if task_id and task_id in diagnostic_ids:
        return False, "diagnostic"
    return True, None


def encode_training_rows(tokenizer, rows, eos_id, args, diagnostic_ids):
    counts = {
        "n_rows": len(rows),
        "n_excluded_split": 0,
        "n_excluded_diagnostic": 0,
        "n_bad_messages": 0,
        "n_bad_target": 0,
        "n_empty_prompt": 0,
        "n_empty_target_tokens": 0,
        "n_empty": 0,
        "n_family_filled_from_task": 0,
        "n_encode_error": 0,
    }
    short = []
    overflow = []
    n_longer = 0
    seen_ids = {}
    for row in rows:
        keep, reason = row_is_trainable(row, diagnostic_ids)
        if not keep:
            if reason == "split":
                counts["n_excluded_split"] += 1
            else:
                counts["n_excluded_diagnostic"] += 1
            continue
        try:
            episode, flag = encode_episode(tokenizer, row, eos_id, row.get("_line_no"))
        except Exception as exc:
            message = str(exc)
            if "enable_thinking" in message or "Not retrying" in message:
                raise
            counts["n_encode_error"] += 1
            counts["last_encode_error"] = message[:300]
            if counts["n_encode_error"] <= 3:
                print("encode skip line %s: %s" % (row.get("_line_no"), message[:200]), flush=True)
            continue
        if episode is None:
            key = {
                "bad_messages": "n_bad_messages",
                "bad_target": "n_bad_target",
                "empty_prompt": "n_empty_prompt",
                "empty_target_tokens": "n_empty_target_tokens",
                "empty": "n_empty",
            }.get(flag)
            if key:
                counts[key] += 1
            continue
        if flag == "family_filled":
            counts["n_family_filled_from_task"] += 1
        if episode.episode_id in seen_ids:
            raise RuntimeError("duplicate episode_id %s" % episode.episode_id)
        seen_ids[episode.episode_id] = True
        if episode.full_len <= args.max_len:
            short.append(episode)
        elif episode.full_len <= args.overflow_len:
            overflow.append(episode)
        else:
            n_longer += 1
    needed = int(args.optimizer_updates) * int(args.grad_accum)
    pool, stats = select_pool(short, overflow, n_longer, needed)
    stats.update(counts)
    stats["max_len"] = int(args.max_len)
    stats["overflow_len"] = int(args.overflow_len)
    stats["effective_batch"] = int(args.microbatch) * int(args.grad_accum)
    stats["effective_batch_note"] = "microbatch * grad_accum; not multiplied by GPU count"
    return pool, stats


def package_versions():
    import numpy
    import torch
    import transformers
    import peft

    versions = {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "peft": peft.__version__,
        "numpy": numpy.__version__,
    }
    try:
        import accelerate

        versions["accelerate"] = accelerate.__version__
    except Exception:
        versions["accelerate"] = None
    return versions


def seed_everything(seed, torch_module, numpy_module):
    random.seed(int(seed))
    numpy_module.random.seed(int(seed))
    torch_module.manual_seed(int(seed))
    if torch_module.cuda.is_available():
        torch_module.cuda.manual_seed_all(int(seed))


def capture_rng(torch_module, numpy_module):
    state = {
        "py_rng": random.getstate(),
        "numpy_rng": numpy_module.random.get_state(),
        "torch_rng": torch_module.get_rng_state(),
        "cuda_rng": None,
    }
    if torch_module.cuda.is_available():
        state["cuda_rng"] = torch_module.cuda.get_rng_state_all()
    return state


def restore_rng(state, torch_module, numpy_module):
    random.setstate(state["py_rng"])
    numpy_module.random.set_state(state["numpy_rng"])
    torch_module.set_rng_state(state["torch_rng"])
    if state.get("cuda_rng") is not None:
        if not torch_module.cuda.is_available():
            raise RuntimeError("trainer_state has CUDA RNG but CUDA is not available")
        torch_module.cuda.set_rng_state_all(state["cuda_rng"])


def iter_runtime_configs(model):
    holders = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        holders.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            holders.append(inner)
    seen = set()
    for holder in holders:
        if holder is None:
            continue
        for attr in ("config", "generation_config"):
            config = getattr(holder, attr, None)
            if config is None or id(config) in seen:
                continue
            seen.add(id(config))
            yield config


def set_use_cache(model, enabled):
    for config in iter_runtime_configs(model):
        try:
            config.use_cache = bool(enabled)
        except Exception:
            pass


def enable_checkpointing(model):
    errors = []
    enabled = False
    holders = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        holders.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            holders.append(inner)
    for holder in holders:
        if holder is None:
            continue
        if hasattr(holder, "gradient_checkpointing_enable"):
            try:
                holder.gradient_checkpointing_enable()
                enabled = True
            except Exception as exc:
                errors.append(str(exc)[:200])
        if hasattr(holder, "enable_input_require_grads"):
            try:
                holder.enable_input_require_grads()
            except Exception as exc:
                errors.append("input_grads:" + str(exc)[:160])
    return enabled, errors


def disable_checkpointing(model):
    holders = [model]
    base = getattr(model, "base_model", None)
    if base is not None:
        holders.append(base)
        inner = getattr(base, "model", None)
        if inner is not None:
            holders.append(inner)
    for holder in holders:
        if holder is not None and hasattr(holder, "gradient_checkpointing_disable"):
            try:
                holder.gradient_checkpointing_disable()
            except Exception:
                pass


def freeze_non_lora(model):
    n_lora = 0
    n_frozen = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad_(True)
            n_lora += 1
        else:
            param.requires_grad_(False)
            n_frozen += 1
    if n_lora == 0:
        raise RuntimeError("no LoRA parameters found; refusing to train the base model")
    return n_lora, n_frozen


def trainable_named_params(model):
    names = []
    params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            names.append(name)
            params.append(param)
    return names, params


def module_leaf_names(model):
    leaves = set()
    for name, _module in model.named_modules():
        if name:
            leaves.add(name.split(".")[-1])
    return leaves


def as_device(torch_module, value, fallback):
    if value is None:
        return fallback
    if isinstance(value, torch_module.device):
        return value
    if isinstance(value, int):
        return torch_module.device("cuda", value)
    text = str(value)
    if text.isdigit():
        return torch_module.device("cuda", int(text))
    return torch_module.device(text)


def embedding_and_lm_devices(model, torch_module):
    try:
        in_dev = model.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(model.parameters()).device
    device_map = getattr(model, "hf_device_map", None)
    if device_map is None:
        try:
            device_map = model.base_model.model.hf_device_map
        except Exception:
            device_map = {}
    if not isinstance(device_map, dict):
        device_map = {}
    lm_value = None
    if "lm_head" in device_map:
        lm_value = device_map["lm_head"]
    else:
        for key, value in device_map.items():
            if key == "lm_head" or str(key).endswith(".lm_head"):
                lm_value = value
                break
    lm_dev = as_device(torch_module, lm_value, in_dev) if lm_value is not None else in_dev
    return in_dev, lm_dev, device_map


def vocab_report(model, tokenizer, config):
    embedding = model.get_input_embeddings()
    head = model.get_output_embeddings()
    if head is None:
        lm_out = None
    elif getattr(head, "weight", None) is not None:
        lm_out = int(head.weight.shape[0])
    else:
        lm_out = int(getattr(head, "out_features"))
    added = tokenizer.get_added_vocab() if hasattr(tokenizer, "get_added_vocab") else {}
    revision = None
    for key in ("revision", "_commit_hash"):
        if config.get(key):
            revision = config.get(key)
            break
    if revision is None:
        revision = getattr(model.config, "_commit_hash", None) or getattr(model.config, "revision", None)
    tie = config.get("tie_word_embeddings")
    if tie is None:
        tie = getattr(model.config, "tie_word_embeddings", None)
    names = sorted(added)
    return {
        "model_type": config.get("model_type") or getattr(model.config, "model_type", None),
        "config_revision": revision,
        "tie_word_embeddings": tie,
        "config_vocab_size": config.get("vocab_size", getattr(model.config, "vocab_size", None)),
        "tokenizer_len": len(tokenizer),
        "tokenizer_vocab_size": getattr(tokenizer, "vocab_size", None),
        "added_tokens": len(added),
        "added_token_names": names if len(names) <= 100 else names[:100],
        "embedding_rows": int(embedding.weight.shape[0]),
        "lm_head_out_features": lm_out,
        "vocab_sizes_may_differ": True,
    }


def ids_to_tensor(torch_module, values, device):
    if isinstance(values, array.array):
        values = values.tolist()
    return torch_module.tensor([list(values)], dtype=torch_module.long).to(device)


def move_optimizer_state(optimizer, torch_module):
    for group in optimizer.param_groups:
        for param in group["params"]:
            slot = optimizer.state.get(param)
            if not slot:
                continue
            for key, value in list(slot.items()):
                if torch_module.is_tensor(value):
                    slot[key] = value.to(device=param.device)


def memory_peaks(torch_module):
    allocated = {}
    reserved = {}
    for index in range(torch_module.cuda.device_count()):
        allocated["gpu%d" % index] = round(torch_module.cuda.max_memory_allocated(index) / 1e9, 4)
        reserved["gpu%d" % index] = round(torch_module.cuda.max_memory_reserved(index) / 1e9, 4)
    return allocated, reserved


def read_chat_template(tokenizer, model_dir):
    template = getattr(tokenizer, "chat_template", None) or ""
    if template:
        return template
    path = os.path.join(model_dir, "chat_template.jinja")
    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as handle:
            return handle.read()
    return ""


def think_probe_facts(prompt_text, chat_template):
    template = chat_template or ""
    has_close = "</think>" in prompt_text
    mentions = ("enable_thinking" in template) and ("</think>" in template)
    return {
        "has_think_close": has_close,
        "has_empty_think_block": ("<think>\n\n</think>" in prompt_text) or ("<think>\n</think>" in prompt_text),
        "template_mentions_enable_thinking": "enable_thinking" in template,
        "template_mentions_think_close": "</think>" in template,
        "assert_applied": mentions,
        "assert_ok": (has_close if mentions else None),
        "tail_400": prompt_text[-400:],
    }


def write_template_probe(path, prompt_text, facts, episode):
    lines = [
        "formatter=%s" % FORMATTER_NAME,
        "episode_id=%s" % episode.episode_id,
        "task_id=%s" % episode.task_id,
        "assert_applied=%s" % facts["assert_applied"],
        "assert_ok=%s" % facts["assert_ok"],
        "has_think_close=%s" % facts["has_think_close"],
        "has_empty_think_block=%s" % facts["has_empty_think_block"],
        "template_mentions_enable_thinking=%s" % facts["template_mentions_enable_thinking"],
        "--- tail 400 ---",
        facts["tail_400"],
        "--- full prompt ---",
        prompt_text,
        "",
    ]
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def generate_supported(model, name):
    try:
        signature = inspect.signature(model.generate)
    except (TypeError, ValueError):
        return False
    return name in signature.parameters


def run_zero_step(model, tokenizer, args, module, in_dev, eos_id, torch_module):
    """Greedy base-behavior generations. Diagnostic ids are not taken from the train JSONL."""
    started = time.perf_counter()
    report = {
        "skipped": False,
        "reason": None,
        "time_guard": False,
        "time_guard_s": ZERO_STEP_TIME_GUARD_S,
        "max_new_tokens": int(args.zero_step_max_new),
        "do_sample": False,
        "repetition_penalty": 1.0,
        "top_k": 0,
        "no_repeat_ngram_size": 0,
        "presence_penalty": None,
        "tasks": [],
    }
    task_ids = parse_id_list(args.zero_step_ids)
    if not args.challenges:
        report["skipped"] = True
        report["reason"] = (
            "challenges path not provided. Diagnostic task ids are not in the train pool, "
            "so zero-step generation was not run."
        )
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        return report
    if module is None or not callable(getattr(module, "build_messages", None)):
        report["skipped"] = True
        report["reason"] = (
            "arc_episode_data.build_messages is unavailable, so zero-step prompts were not "
            "invented from a second format. template_probe still comes from the first training episode."
        )
        report["elapsed_s"] = round(time.perf_counter() - started, 3)
        return report

    challenges = load_task_map(args.challenges)
    solutions = load_task_map(args.solutions) if args.solutions else None
    text_to_grid = getattr(module, "text_to_grid", None)
    use_presence = generate_supported(model, "presence_penalty")
    report["presence_penalty"] = 0 if use_presence else "omitted"
    disable_checkpointing(model)
    set_use_cache(model, True)
    model.eval()
    builder_error = None

    for task_id in task_ids:
        elapsed = time.perf_counter() - started
        if elapsed >= ZERO_STEP_TIME_GUARD_S:
            report["time_guard"] = True
            report["tasks"].append({
                "task_id": task_id,
                "status": "skipped_time_guard",
                "decoded_text": None,
                "parse": None,
                "grid": None,
                "exact_match": None,
            })
            continue
        if builder_error:
            report["tasks"].append({
                "task_id": task_id,
                "status": "message_fail",
                "error": builder_error,
                "decoded_text": None,
                "parse": None,
                "grid": None,
                "exact_match": None,
            })
            continue
        record = {"task_id": task_id, "status": "pending"}
        wall0 = time.perf_counter()
        task = challenges.get(task_id) if isinstance(challenges, dict) else None
        if not isinstance(task, dict):
            record.update({
                "status": "missing_challenge",
                "decoded_text": None,
                "parse": None,
                "grid": None,
                "exact_match": None,
                "wall_s": round(time.perf_counter() - wall0, 3),
            })
            report["tasks"].append(record)
            continue
        try:
            messages = messages_for_test_query(module, tokenizer, task_id, task, solutions)
            prompt_text = render_prompt(tokenizer, messages)
            torch_module.cuda.synchronize()
            prep0 = time.perf_counter()
            prompt_ids = tokenize_text(tokenizer, prompt_text)
            input_ids = torch_module.tensor([prompt_ids], dtype=torch_module.long).to(in_dev)
            attention = torch_module.ones((1, len(prompt_ids)), dtype=torch_module.long).to(in_dev)
            torch_module.cuda.synchronize()
            prepare_s = time.perf_counter() - prep0
            kwargs = {
                "input_ids": input_ids,
                "attention_mask": attention,
                "max_new_tokens": int(args.zero_step_max_new),
                "do_sample": False,
                "repetition_penalty": 1.0,
                # Override the published chat top_k=20. top_k=0 disables the warper.
                "top_k": 0,
                "no_repeat_ngram_size": 0,
                "pad_token_id": eos_id,
                "eos_token_id": eos_id,
                "use_cache": True,
            }
            if use_presence:
                kwargs["presence_penalty"] = 0
            torch_module.cuda.synchronize()
            gen0 = time.perf_counter()
            with torch_module.inference_mode():
                try:
                    generated = model.generate(**kwargs)
                except TypeError as exc:
                    message = str(exc)
                    reduced = dict(kwargs)
                    if "presence_penalty" in reduced and "presence_penalty" in message:
                        reduced.pop("presence_penalty")
                        record["presence_penalty"] = "omitted"
                        generated = model.generate(**reduced)
                    elif "use_cache" in reduced and "use_cache" in message:
                        reduced.pop("use_cache")
                        generated = model.generate(**reduced)
                    else:
                        raise
            torch_module.cuda.synchronize()
            generate_s = time.perf_counter() - gen0
            sequence = generated[0].detach().cpu()
            prompt_len = int(input_ids.shape[1])
            if int(sequence.shape[0]) >= prompt_len:
                new_ids = sequence[prompt_len:].tolist()
            else:
                new_ids = sequence.tolist()
                record["generate_omitted_prompt"] = True
            decoded = tokenizer.decode(new_ids, skip_special_tokens=False)
            parse_ids = list(new_ids)
            dropped_eos = False
            if parse_ids and int(parse_ids[-1]) == int(eos_id):
                parse_ids = parse_ids[:-1]
                dropped_eos = True
            parse_text = tokenizer.decode(parse_ids, skip_special_tokens=False)
            grid = parse_strict_grid(parse_text)
            truth = solution_grid(solutions, task_id) if solutions is not None else None
            exact = None
            if truth is not None and grid is not None:
                exact = bool(grids_equal(grid, truth))
            agrees = None
            if grid is not None and callable(text_to_grid):
                try:
                    other = text_to_grid(parse_text.strip())
                    agrees = bool(other == grid)
                except Exception as exc:
                    agrees = "error:%s" % str(exc)[:120]
            target_text = None
            if truth is not None:
                target_text = "\n".join(" ".join(str(int(cell)) for cell in row) for row in truth)
            record.update({
                "status": "generated",
                "decoded_text": decoded,
                "parse_text": parse_text,
                "parse": "ok" if grid is not None else "parse_failure",
                "grid": grid,
                "exact_match": exact,
                "prepare_s": round(prepare_s, 4),
                "generate_s": round(generate_s, 4),
                "wall_s": round(time.perf_counter() - wall0, 4),
                "prompt_tokens": prompt_len,
                "generated_tokens": len(new_ids),
                "truncated": bool(len(new_ids) >= int(args.zero_step_max_new)),
                "dropped_trailing_eos": dropped_eos,
                "text_to_grid_agrees": agrees,
                "target_text_in_prompt": (target_text in prompt_text) if target_text else None,
            })
            del generated, sequence, input_ids, attention
        except Exception as exc:
            message = str(exc)[:500]
            if "build_messages" in message:
                builder_error = message
                record["status"] = "message_fail"
            record.update({
                "status": record.get("status") if record.get("status") == "message_fail" else "gen_fail",
                "error": message,
                "decoded_text": None,
                "parse": None,
                "grid": None,
                "exact_match": None,
                "wall_s": round(time.perf_counter() - wall0, 4),
            })
            print("zero-step %s failed: %s" % (task_id, message[:300]), flush=True)
        report["tasks"].append(record)
        print(
            "zero-step %s status=%s parse=%s" % (task_id, record.get("status"), record.get("parse")),
            flush=True,
        )
        torch_module.cuda.empty_cache()

    report["elapsed_s"] = round(time.perf_counter() - started, 3)
    if report["time_guard"]:
        report["reason"] = "time_guard"
    set_use_cache(model, False)
    torch_module.cuda.empty_cache()
    return report


def write_mask_probe(path, tokenizer, episode, eos_id):
    labels = episode.labels.tolist()
    supervised = [token for token in labels if token != -100]
    supervised_text = tokenizer.decode(supervised, skip_special_tokens=False)
    eos_piece = tokenizer.decode([int(eos_id)], skip_special_tokens=False)
    target_ids = tokenize_text(tokenizer, episode.target_grid_text)
    expected_ids = target_ids + [int(eos_id)]
    probe = {
        "episode_id": episode.episode_id,
        "prompt_len": episode.prompt_len,
        "target_len": episode.target_len,
        "supervised_text": supervised_text,
        "target_grid_text": episode.target_grid_text,
        "eos_piece": eos_piece,
        "target_plus_eos_piece": episode.target_grid_text + eos_piece,
        "supervised_text_equals_target_plus_eos": supervised_text == (episode.target_grid_text + eos_piece),
        "supervised_ids_equal_target_plus_eos": supervised == expected_ids,
        "prompt_labels_all_ignored": labels[: episode.prompt_len] == ([-100] * episode.prompt_len),
        "n_supervised": len(supervised),
        "n_ignored": len(labels) - len(supervised),
        "target_in_prompt_text": episode.target_grid_text in episode.prompt_text,
        "label_shift": "none_hf_shifts_once",
    }
    dump_json(path, probe, one_line=False)
    if not probe["prompt_labels_all_ignored"] or not probe["supervised_ids_equal_target_plus_eos"]:
        raise RuntimeError("mask probe failed; see %s" % path)
    return probe


def append_jsonl(path, record):
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


def save_checkpoint(model, optimizer, schedule, bundle, out_dir, torch_module):
    adapter_dir = os.path.join(out_dir, "adapter")
    os.makedirs(out_dir, exist_ok=True)
    model.save_pretrained(adapter_dir)
    payload = {
        "formatter": FORMATTER_NAME,
        "completed_updates": int(bundle["completed_updates"]),
        "cursor": int(bundle["cursor"]),
        "pool_episode_ids": list(bundle["pool_episode_ids"]),
        "optimizer": optimizer.state_dict(),
        "scheduler": schedule.state_dict(),
        "trainable_param_names": list(bundle["trainable_param_names"]),
        "argv": list(sys.argv),
        "versions": bundle["versions"],
        "seed": bundle["seed"],
        "loss_divided_by_grad_accum": True,
        "run_config": bundle["run_config"],
    }
    payload.update(bundle["rng"])
    temporary = os.path.join(out_dir, "trainer_state.pt.tmp")
    torch_module.save(payload, temporary)
    os.replace(temporary, os.path.join(out_dir, "trainer_state.pt"))
    dump_json(os.path.join(out_dir, "run_config.json"), bundle["run_config"], one_line=False)
    if not adapter_files_present(adapter_dir):
        raise RuntimeError("save_pretrained did not write adapter config and weights")
    completed = int(bundle.get("completed_updates", 0))
    if completed > 0:
        # Immutable per-step snapshot; an existing snapshot is never overwritten, so a
        # resumed re-save of the same update keeps the original directory intact.
        snapshot = os.path.join(out_dir, "adapter_step_%d" % completed)
        if not adapter_files_present(snapshot):
            model.save_pretrained(snapshot)
            if not adapter_files_present(snapshot):
                raise RuntimeError("snapshot save did not write adapter files: %s" % snapshot)
    return adapter_dir


class UpdateSchedule(object):
    def __init__(self, optimizer, base_lr, warmup, total):
        self.optimizer = optimizer
        self.base_lr = float(base_lr)
        self.warmup = int(warmup)
        self.total = int(total)
        self.completed = 0
        self.last_lr = None

    def apply_update_lr(self, update):
        lr = self.base_lr * lr_multiplier(update, self.warmup, self.total)
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        self.last_lr = lr
        return lr

    def step(self):
        # Once per optimizer update, never per microstep.
        self.completed += 1

    def state_dict(self):
        return {
            "completed": self.completed,
            "base_lr": self.base_lr,
            "warmup": self.warmup,
            "total": self.total,
            "last_lr": self.last_lr,
            "formula": LR_FORMULA,
        }

    def load_state_dict(self, state):
        self.completed = int(state.get("completed", 0))
        self.last_lr = state.get("last_lr")


def branch_note(extras, resumed):
    base = (
        "G1 default LoRA targets only (%s). GDN targets are not included."
        % ", ".join(DEFAULT_TARGETS)
    )
    if not extras:
        return base
    gdn = [name for name in extras if name in GDN_TARGET_NAMES]
    note = (
        "New LoRA target branch. extra_targets=%s. Fresh optimizer; do not resume an older "
        "optimizer state whose parameter names differ." % ",".join(extras)
    )
    if gdn:
        note += " Caller passed GDN-related targets (%s); this is not the G1 default." % ",".join(gdn)
    if resumed:
        note += " Resume was requested and is allowed only when trainable names match."
    return note


def build_run_config(args, model_dir, versions, vocab, length_stats, targets, extras, devices, trainable, resumed_from):
    return {
        "formatter": FORMATTER_NAME,
        "argv": list(sys.argv),
        "seed": int(args.seed),
        "model_dir": model_dir,
        "episodes": os.path.abspath(args.episodes),
        "out_dir": os.path.abspath(args.out_dir),
        "resumed_from": resumed_from,
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "warmup_updates": int(args.warmup_updates),
        "optimizer_updates": int(args.optimizer_updates),
        "clip_norm": float(args.clip_norm),
        "grad_accum": int(args.grad_accum),
        "microbatch": int(args.microbatch),
        "effective_batch": int(args.microbatch) * int(args.grad_accum),
        "effective_batch_note": "microbatch * grad_accum; not multiplied by GPU count",
        "max_len": int(args.max_len),
        "overflow_len": int(args.overflow_len),
        "save_every": int(args.save_every),
        "lora": {
            "r": 8,
            "alpha": 32,
            "dropout": 0.0,
            "bias": "none",
            "task_type": "CAUSAL_LM",
            "target_modules": list(targets),
            "extra_targets": list(extras),
        },
        "branch_note": branch_note(extras, bool(args.resume)),
        "loss": {
            "prompt_labels": -100,
            "pad_labels": -100,
            "pad_positions": "only inserted padding; eos is supervised, not treated as pad",
            "supervised": "target_grid_text tokens plus one eos id",
            "shift": "huggingface causal LM shifts once; this script does not",
            "backward_divisor": "grad_accum",
            "logged_loss": "mean of unscaled microstep losses",
            "target_token_count": "supervised tokens, including the eos id",
        },
        "lr_formula": LR_FORMULA,
        "versions": versions,
        "vocab": vocab,
        "length_policy": length_stats,
        "devices": devices,
        "trainable_params": int(trainable),
        "parallel": "single process device_map=balanced; no FSDP, DDP, TP, or 4-bit",
        "zero_step_ids": parse_id_list(args.zero_step_ids),
        "zero_step_max_new": int(args.zero_step_max_new),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def train_loop(
    model,
    optimizer,
    schedule,
    pool,
    args,
    tokenizer,
    eos_id,
    in_dev,
    lm_dev,
    torch_module,
    bundle,
    summary,
    inside_step,
):
    target = int(args.optimizer_updates)
    start = int(schedule.completed)
    cursor = int(bundle["cursor"])
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    mask_path = os.path.join(args.out_dir, "mask_probe.json")
    wrote_mask = os.path.isfile(mask_path) and start > 0
    seen_families = set()
    micro_total = 0
    pool_exhausted = False
    if start >= target:
        summary["note"] = "resume already reached optimizer-updates; no further updates"
        summary["n_families_seen"] = 0
        summary["completed_microsteps"] = 0
        return cursor, False

    for update in range(start + 1, target + 1):
        if cursor >= len(pool):
            pool_exhausted = True
            break
        # Floor for this update. An OOM episode stays consumed; do not rewind onto it.
        cursor_at_update = cursor
        lr = schedule.apply_update_lr(update)
        optimizer.zero_grad(set_to_none=True)
        losses = []
        episode_ids = []
        family_ids = []
        prompt_counts = []
        target_counts = []
        torch_module.cuda.synchronize()
        tick = time.perf_counter()
        oom_this_update = False
        oom_episode_id = None
        for _micro in range(int(args.grad_accum)):
            if cursor >= len(pool):
                break
            episode = pool[cursor]
            cursor += 1
            try:
                input_ids = ids_to_tensor(torch_module, episode.input_ids, in_dev)
                label_list = episode.labels.tolist()
                labels = torch_module.tensor([label_list], dtype=torch_module.long).to(lm_dev)
                attention = torch_module.ones_like(input_ids)
                output = model(input_ids=input_ids, attention_mask=attention, labels=labels)
                if output.loss is None:
                    raise RuntimeError("model returned loss=None at update %d" % update)
                loss_value = float(output.loss.detach().float().item())
                if loss_value != loss_value or loss_value in (float("inf"), float("-inf")):
                    raise RuntimeError("non-finite loss at update %d: %s" % (update, loss_value))
                if not wrote_mask:
                    write_mask_probe(mask_path, tokenizer, episode, eos_id)
                    wrote_mask = True
                    print("mask_probe wrote %s" % mask_path, flush=True)
                # Divide by grad_accum so four backwards match the mean gradient. Log the raw loss.
                scaled = output.loss / float(args.grad_accum)
                scaled.backward()
            except torch_module.cuda.OutOfMemoryError:
                oom_this_update = True
            except RuntimeError as exc:
                if "out of memory" not in str(exc).lower():
                    raise
                oom_this_update = True
            if oom_this_update:
                oom_episode_id = getattr(episode, "episode_id", None)
                input_ids = labels = attention = output = scaled = None
                break
            losses.append(loss_value)
            episode_ids.append(episode.episode_id)
            family_ids.append(episode.family_id)
            seen_families.add(episode.family_id)
            prompt_counts.append(int(episode.prompt_len))
            target_counts.append(int(episode.target_len))
            del scaled, output, input_ids, labels, attention
        if oom_this_update:
            # Partial accumulation must not step or advance the schedule.
            if cursor < cursor_at_update:
                cursor = cursor_at_update
            input_ids = labels = attention = output = scaled = None
            optimizer.zero_grad(set_to_none=True)
            torch_module.cuda.empty_cache()
            oom_count = int(summary.get("oom_count") or 0) + 1
            summary["oom_count"] = oom_count
            oom_record = {"status": "oom", "update": update}
            if oom_episode_id is not None:
                oom_record["episode_id"] = oom_episode_id
            append_jsonl(log_path, oom_record)
            print(
                "update %d oom episode=%s count=%d" % (update, oom_episode_id, oom_count),
                flush=True,
            )
            if oom_count >= 8:
                summary["oom_stop"] = True
                summary["oom_count"] = 8
                break
            continue
        if not losses:
            pool_exhausted = True
            break
        trainable = [
            param for param in model.parameters() if param.requires_grad and param.grad is not None
        ]
        if not trainable:
            raise RuntimeError("no gradients at update %d" % update)
        grad_norm = torch_module.nn.utils.clip_grad_norm_(trainable, float(args.clip_norm))
        if not torch_module.isfinite(grad_norm):
            raise RuntimeError("non-finite grad norm at update %d" % update)
        inside_step[0] = True
        optimizer.step()
        inside_step[0] = False
        schedule.step()
        bundle["completed_updates"] = int(schedule.completed)
        bundle["cursor"] = int(cursor)
        bundle["run_config"]["completed_updates"] = int(schedule.completed)
        bundle["run_config"]["cursor"] = int(cursor)
        torch_module.cuda.synchronize()
        seconds = time.perf_counter() - tick
        mean_loss = sum(losses) / float(len(losses))
        micro_total += len(losses)
        record = {
            "update": update,
            "lr": lr,
            "mean_loss": mean_loss,
            "microsteps": len(losses),
            "episode_ids": episode_ids,
            "family_ids": family_ids,
            "prompt_token_count": prompt_counts,
            "target_token_count": target_counts,
            "seconds": round(seconds, 4),
            "loss_divided_by_grad_accum": True,
        }
        append_jsonl(log_path, record)
        bundle["rng"] = capture_rng(torch_module, bundle["numpy_module"])
        print(
            "update %d/%d lr=%g mean_loss=%.6f microsteps=%d %.2fs"
            % (update, target, lr, mean_loss, len(losses), seconds),
            flush=True,
        )
        summary["last_mean_loss"] = mean_loss
        summary["completed_updates"] = int(schedule.completed)
        if update % int(args.save_every) == 0:
            save_checkpoint(model, optimizer, schedule, bundle, args.out_dir, torch_module)
            summary["last_saved_update"] = int(schedule.completed)
            summary["saved_this_run"] = True
            summary["checkpoint_saved"] = True
            print("saved update %d" % update, flush=True)
    summary["n_families_seen"] = len(seen_families)
    summary["completed_microsteps"] = micro_total
    expected_cursor = target * int(args.grad_accum)
    if int(schedule.completed) < target or cursor < expected_cursor:
        if summary.get("oom_stop"):
            shortfall_reason = "oom_stop"
        elif pool_exhausted or cursor >= len(pool):
            shortfall_reason = "filtered pool exhausted"
        else:
            shortfall_reason = "microstep shortfall"
        summary["shortfall"] = {
            "reason": shortfall_reason,
            "expected_updates": target,
            "completed_updates": int(schedule.completed),
            "expected_microsteps": expected_cursor,
            "completed_microsteps_this_process": micro_total,
            "cursor": int(cursor),
            "pool_size": len(pool),
        }
        return cursor, True
    return cursor, False


def execute(args, summary):
    import numpy
    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    validate_args(args)
    if not torch.cuda.is_available():
        raise SkipRun("SKIP_NO_GPU", "SKIP_NO_GPU: cuda unavailable")
    gpu_count = int(torch.cuda.device_count())
    print("GPUs:", gpu_count, flush=True)
    if gpu_count < 4:
        raise SkipRun("SKIP_GPU_COUNT", "SKIP_GPU_COUNT: need 4 GPUs, found %d" % gpu_count)
    summary["gpu_count"] = gpu_count

    os.makedirs(args.out_dir, exist_ok=True)
    diagnostic_ids = set(parse_id_list(args.zero_step_ids))
    extras = parse_extra_targets(args.extra_targets)
    targets = list(DEFAULT_TARGETS) + extras
    versions = package_versions()
    model_dir = find_model_dir(args.model_dir)
    print("MODEL_DIR:", model_dir, flush=True)
    with open(os.path.join(model_dir, "config.json"), "r", encoding="utf-8") as handle:
        raw_config = json.load(handle)

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("tokenizer eos_token_id is None; refusing to train without an eos id")
    eos_id = int(eos_id)

    print("encoding episodes", flush=True)
    rows = load_jsonl_rows(args.episodes)
    by_id_rows = {}
    for row in rows:
        if row.get("episode_id") is not None:
            by_id_rows.setdefault(str(row["episode_id"]), row)

    resume_state = None
    if args.resume:
        _adapter_path, state_path = resolve_resume_paths(args.resume)
        # Our checkpoint stores RNG tuples and optimizer tensors, not a weights-only blob.
        resume_state = torch.load(state_path, map_location="cpu", weights_only=False)
        saved_ids = [str(item) for item in resume_state.get("pool_episode_ids") or []]
        if not saved_ids:
            raise RuntimeError("trainer_state.pt is missing pool_episode_ids")
        pool = []
        missing = []
        for episode_id in saved_ids:
            row = by_id_rows.get(episode_id)
            if row is None:
                missing.append(episode_id)
                continue
            episode, flag = encode_episode(tokenizer, row, eos_id, row.get("_line_no"))
            if episode is None:
                raise RuntimeError("resume episode %s failed to encode (%s)" % (episode_id, flag))
            pool.append(episode)
        if missing:
            raise RuntimeError("resume pool episodes missing from JSONL: %s" % missing[:8])
        previous = {}
        saved_config = resume_state.get("run_config")
        if isinstance(saved_config, dict) and isinstance(saved_config.get("length_policy"), dict):
            previous = dict(saved_config["length_policy"])
        length_stats = previous
        length_stats.update({
            "resumed_pool": True,
            "pool_size": len(pool),
            "n_families_consumed": len({ep.family_id for ep in pool}),
            "needed_examples": int(args.optimizer_updates) * int(args.grad_accum),
            "max_len": int(args.max_len),
            "overflow_len": int(args.overflow_len),
            "note": "resume keeps the stored cursor order; length buckets were not rebuilt",
        })
        summary["length_policy"] = length_stats
    else:
        pool, length_stats = encode_training_rows(tokenizer, rows, eos_id, args, diagnostic_ids)
        summary["length_policy"] = length_stats
        if not pool:
            detail = length_stats.get("last_encode_error") or ""
            raise RuntimeError(
                "FAILED_NO_DATA: no trainable episodes at or under overflow-len %d "
                "(kept_le_max=%s longer=%s encode_errors=%s). %s"
                % (
                    args.overflow_len,
                    length_stats.get("n_kept_le_max_len"),
                    length_stats.get("n_longer_than_overflow"),
                    length_stats.get("n_encode_error"),
                    detail,
                )
            )
        order = list(pool)
        random.Random(int(args.seed)).shuffle(order)
        pool = order
        print(
            "length_policy kept_le_%d=%d added_le_%d=%d longer_than_%d=%d families=%d pool=%d needed=%d"
            % (
                args.max_len,
                length_stats["n_kept_le_max_len"],
                args.overflow_len,
                length_stats["n_added_le_overflow"],
                args.overflow_len,
                length_stats["n_longer_than_overflow"],
                length_stats["n_families_consumed"],
                length_stats["pool_size"],
                length_stats["needed_examples"],
            ),
            flush=True,
        )

    if length_stats.get("n_excluded_diagnostic"):
        print(
            "WARNING excluded %d diagnostic-id rows from the train pool"
            % length_stats["n_excluded_diagnostic"],
            flush=True,
        )

    seed_everything(args.seed, torch, numpy)
    print("loading model", flush=True)
    base = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
    )
    ckpt_ok, ckpt_errors = enable_checkpointing(base)
    set_use_cache(base, False)
    summary["grad_ckpt"] = True if ckpt_ok and not ckpt_errors else {"enabled": ckpt_ok, "errors": ckpt_errors}

    init_info = None
    if args.resume:
        adapter_path, _state_path = resolve_resume_paths(args.resume)
        try:
            model = PeftModel.from_pretrained(base, adapter_path, is_trainable=True)
        except TypeError:
            model = PeftModel.from_pretrained(base, adapter_path)
        init_info = None
    elif getattr(args, "init_adapter", None):
        # Weights-only new stage: fresh LoRA, then copy intersecting old LoRA tensors.
        # Optimizer/scheduler below are rebuilt; old ones are never loaded.
        init_adapter_dir = resolve_init_adapter_dir(args.init_adapter)
        init_old_targets = read_adapter_targets(init_adapter_dir)
        missing_targets = [name for name in targets if name not in module_leaf_names(base)]
        if missing_targets:
            raise RuntimeError("LoRA target modules not found: %s" % missing_targets)
        lora = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            inference_mode=False,
        )
        model = get_peft_model(base, lora)
        init_stats = load_init_adapter_weights(model, init_adapter_dir, torch)
        init_old_amount = old_stage_amount(init_adapter_dir)
        try:
            _parent = os.path.dirname(os.path.abspath(init_adapter_dir))
            _old_state_path = os.path.join(_parent, "trainer_state.pt")
            if os.path.isfile(_old_state_path):
                _old_state = torch.load(_old_state_path, map_location="cpu", weights_only=False)
                if init_old_amount.get("completed_updates") is None:
                    init_old_amount["completed_updates"] = int(
                        _old_state.get("completed_updates", 0)
                    )
                if init_old_amount.get("cursor") is None:
                    init_old_amount["cursor"] = int(_old_state.get("cursor", 0))
        except Exception as exc:
            init_old_amount["state_read_error"] = str(exc)[:200]
        init_info = {
            "source_adapter": os.path.abspath(init_adapter_dir),
            "old_targets": init_old_targets,
            "new_targets": list(targets),
            "targets_changed": set(init_old_targets) != set(targets),
            "old_completed_updates": init_old_amount.get("completed_updates"),
            "old_cursor": init_old_amount.get("cursor"),
            "new_optimizer_updates": int(args.optimizer_updates),
            "copy": init_stats,
        }
        print(
            "init-adapter copied=%d mismatch=%d not_in_new=%d targets_changed=%s" % (
                init_stats["copied_lora_tensors"],
                init_stats["n_shape_mismatch"],
                init_stats["n_not_in_new"],
                set(init_old_targets) != set(targets),
            ),
            flush=True,
        )
    else:
        missing_targets = [name for name in targets if name not in module_leaf_names(base)]
        if missing_targets:
            raise RuntimeError("LoRA target modules not found: %s" % missing_targets)
        lora = LoraConfig(
            r=8,
            lora_alpha=32,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=targets,
            inference_mode=False,
        )
        model = get_peft_model(base, lora)

    enable_checkpointing(model)
    set_use_cache(model, False)
    n_lora, n_frozen = freeze_non_lora(model)
    try:
        model.print_trainable_parameters()
    except Exception as exc:
        print("print_trainable_parameters skipped:", str(exc)[:200], flush=True)
    names, params = trainable_named_params(model)
    trainable = sum(param.numel() for param in params)
    print("trainable_params:", trainable, "lora_tensors:", n_lora, "frozen_tensors:", n_frozen, flush=True)
    if any("lora_" not in name for name in names):
        raise RuntimeError("a non-LoRA parameter requires grad; refusing to train the base")

    in_dev, lm_dev, device_map = embedding_and_lm_devices(model, torch)
    print("input_device:", in_dev, "lm_head_device:", lm_dev, flush=True)
    print("device_map:", device_map, flush=True)
    if getattr(in_dev, "type", None) != "cuda" or getattr(lm_dev, "type", None) != "cuda":
        raise RuntimeError("input embedding or lm_head is not on CUDA: %s %s" % (in_dev, lm_dev))

    optimizer = torch.optim.AdamW(params, lr=float(args.lr), weight_decay=float(args.weight_decay))
    schedule = UpdateSchedule(optimizer, float(args.lr), int(args.warmup_updates), int(args.optimizer_updates))
    vocab = vocab_report(model, tokenizer, raw_config)
    print("vocab:", json.dumps(vocab, ensure_ascii=False, default=str), flush=True)
    devices = {"input": str(in_dev), "lm_head": str(lm_dev), "gpu_count": gpu_count}
    run_config = build_run_config(
        args,
        model_dir,
        versions,
        vocab,
        length_stats,
        targets,
        extras,
        devices,
        trainable,
        os.path.abspath(args.resume) if args.resume else None,
    )
    if int(args.warmup_updates) >= int(args.optimizer_updates):
        run_config["schedule_warning"] = "warmup_updates >= optimizer_updates; decay interval is empty"
        print("WARNING", run_config["schedule_warning"], flush=True)
    if init_info is not None:
        # New-stage accounting: old volume is reference only; the new run restarts at update 0
        # with a fresh pool, cursor, optimizer, and scheduler.
        run_config["init_adapter"] = init_info
        run_config["stage"] = {
            "mode": "init_adapter",
            "old_completed_updates": init_info["old_completed_updates"],
            "old_cursor": init_info["old_cursor"],
            "new_optimizer_updates": int(args.optimizer_updates),
            "note": "weights-only inheritance; not a continuation of the old loss curve",
        }
        summary["init_adapter"] = {
            "source_adapter": init_info["source_adapter"],
            "targets_changed": init_info["targets_changed"],
            "copied_lora_tensors": init_info["copy"]["copied_lora_tensors"],
            "old_completed_updates": init_info["old_completed_updates"],
            "new_optimizer_updates": int(args.optimizer_updates),
        }

    cursor = 0
    if resume_state is not None:
        saved_names = list(resume_state.get("trainable_param_names") or [])
        saved_adapter = read_json(os.path.join(resolve_resume_paths(args.resume)[0], "adapter_config.json"))
        saved_targets = saved_adapter.get("target_modules") or []
        if isinstance(saved_targets, dict):
            saved_targets = list(saved_targets.keys())
        saved_targets = [str(name) for name in saved_targets]
        names_differ = saved_names != names
        targets_differ = set(saved_targets) != set(targets)
        if extras and (names_differ or targets_differ):
            raise ResumeMismatch(
                "extra targets refuse resume: requested %s, adapter targets %s, "
                "optimizer parameter names differ=%s. Start a fresh run for this branch."
                % (targets, saved_targets, names_differ)
            )
        if names_differ or targets_differ:
            raise ResumeMismatch(
                "refusing resume because trainable names or LoRA targets differ "
                "(names_differ=%s targets_differ=%s)." % (names_differ, targets_differ)
            )
        optimizer.load_state_dict(resume_state["optimizer"])
        move_optimizer_state(optimizer, torch)
        if resume_state.get("scheduler"):
            saved_sched = resume_state["scheduler"]
            schedule.completed = int(saved_sched.get("completed", resume_state.get("completed_updates", 0)))
            if int(saved_sched.get("total", args.optimizer_updates)) != int(args.optimizer_updates) or float(
                saved_sched.get("base_lr", args.lr)
            ) != float(args.lr):
                run_config["schedule_warning"] = (
                    "CLI schedule differs from trainer_state; continuing the update counter "
                    "on the CLI lr/warmup/total curve."
                )
                print("WARNING", run_config["schedule_warning"], flush=True)
        else:
            schedule.completed = int(resume_state.get("completed_updates", 0))
        cursor = int(resume_state.get("cursor", 0))
        if cursor > len(pool):
            raise RuntimeError("saved cursor %d is past pool size %d" % (cursor, len(pool)))
        try:
            restore_rng(resume_state, torch, numpy)
        except Exception as exc:
            raise RuntimeError("failed to restore RNG from trainer_state: %s" % exc)
        run_config["resumed_completed_updates"] = int(schedule.completed)
        run_config["resumed_cursor"] = int(cursor)

    same_out = bool(args.resume) and os.path.abspath(args.resume) == os.path.abspath(args.out_dir)
    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    if not same_out:
        with open(log_path, "w", encoding="utf-8") as handle:
            handle.write("")

    probe_episode = None
    if pool:
        probe_index = cursor if cursor < len(pool) else 0
        probe_episode = pool[probe_index]
    if probe_episode is None:
        raise RuntimeError("FAILED_NO_DATA: training pool is empty")
    facts = think_probe_facts(probe_episode.prompt_text, read_chat_template(tokenizer, model_dir))
    probe_path = os.path.join(args.out_dir, "template_probe.txt")
    write_template_probe(probe_path, probe_episode.prompt_text, facts, probe_episode)
    print(facts["tail_400"], flush=True)
    print(
        "template_probe assert_applied=%s assert_ok=%s has_empty_think_block=%s"
        % (facts["assert_applied"], facts["assert_ok"], facts["has_empty_think_block"]),
        flush=True,
    )
    if facts["assert_applied"] and not facts["has_think_close"]:
        raise RuntimeError(
            "rendered prompt has no </think> close, but the chat template contains "
            "enable_thinking and </think>. Refusing to train with thinking left on. "
            "See template_probe.txt. Not retrying without enable_thinking."
        )

    module, module_error = load_episode_module()
    if module_error:
        print("arc_episode_data import:", module_error, flush=True)
        run_config["episode_api_error"] = module_error
    else:
        run_config["episode_api"] = getattr(module, "__file__", "arc_episode_data")
        run_config["build_messages"] = bool(callable(getattr(module, "build_messages", None)))

    sources = sorted({str(ep.source) for ep in pool if ep.source is not None})
    revisions = sorted({str(ep.revision) for ep in pool if ep.revision is not None})
    run_config["data_sources"] = sources
    run_config["data_revisions"] = revisions
    bundle = {
        "completed_updates": int(schedule.completed),
        "cursor": int(cursor),
        "pool_episode_ids": [ep.episode_id for ep in pool],
        "trainable_param_names": names,
        "versions": versions,
        "seed": int(args.seed),
        "run_config": run_config,
        "rng": capture_rng(torch, numpy),
        "numpy_module": numpy,
    }
    save_checkpoint(model, optimizer, schedule, bundle, args.out_dir, torch)
    summary["saved_this_run"] = True
    summary["checkpoint_saved"] = True
    summary["last_saved_update"] = int(schedule.completed)
    summary["step0_saved"] = schedule.completed == 0 and not args.resume
    print("saved initial adapter at update %d" % schedule.completed, flush=True)

    if args.resume:
        zero = {
            "skipped": True,
            "reason": "resume: adapter is not step-0 base behavior, so zero-step was not run",
            "time_guard": False,
            "tasks": [],
        }
    else:
        try:
            zero = run_zero_step(model, tokenizer, args, module, in_dev, eos_id, torch)
        except Exception as exc:
            zero = {
                "skipped": False,
                "reason": "zero-step failed; training still starts",
                "error": str(exc)[:500],
                "time_guard": False,
                "tasks": [],
            }
            print("zero-step error, continuing to training: %s" % str(exc)[:300], flush=True)
    dump_json(os.path.join(args.out_dir, "zero_step.json"), zero, one_line=False)
    summary["zero_step_skipped"] = bool(zero.get("skipped"))
    summary["zero_step_reason"] = zero.get("reason")
    summary["zero_step_time_guard"] = bool(zero.get("time_guard"))
    summary["zero_step_parse_ok"] = sum(1 for task in zero.get("tasks") or [] if task.get("parse") == "ok")
    summary["zero_step_exact_match"] = sum(1 for task in zero.get("tasks") or [] if task.get("exact_match") is True)
    print("zero_step skipped=%s reason=%s" % (zero.get("skipped"), zero.get("reason")), flush=True)

    enable_checkpointing(model)
    set_use_cache(model, False)
    model.train()
    torch.set_grad_enabled(True)
    torch.cuda.empty_cache()
    for index in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(index)
    summary["memory_peak_scope"] = "training_loop"

    # Refresh RNG after zero-step so a resume from this update matches the training process.
    bundle["rng"] = capture_rng(torch, numpy)
    bundle["run_config"]["zero_step_skipped"] = summary["zero_step_skipped"]
    bundle["run_config"]["zero_step_reason"] = summary["zero_step_reason"]
    save_checkpoint(model, optimizer, schedule, bundle, args.out_dir, torch)
    summary["saved_this_run"] = True

    inside_step = [False]
    summary["_inside_step"] = inside_step
    try:
        _cursor, shortfall = train_loop(
            model,
            optimizer,
            schedule,
            pool,
            args,
            tokenizer,
            eos_id,
            in_dev,
            lm_dev,
            torch,
            bundle,
            summary,
            inside_step,
        )
    except BaseException:
        if not inside_step[0]:
            try:
                bundle["rng"] = capture_rng(torch, numpy)
                save_checkpoint(model, optimizer, schedule, bundle, args.out_dir, torch)
                summary["saved_this_run"] = True
                summary["saved_on_failure"] = True
                summary["last_saved_update"] = int(bundle.get("completed_updates", 0))
            except Exception as save_exc:
                summary["save_on_exit_error"] = str(save_exc)[:300]
        raise
    bundle["cursor"] = int(_cursor)
    bundle["completed_updates"] = int(schedule.completed)
    bundle["rng"] = capture_rng(torch, numpy)
    bundle["run_config"]["completed_updates"] = int(schedule.completed)
    bundle["run_config"]["cursor"] = int(_cursor)
    save_checkpoint(model, optimizer, schedule, bundle, args.out_dir, torch)
    summary["saved_this_run"] = True
    summary["checkpoint_saved"] = True
    summary["last_saved_update"] = int(schedule.completed)
    summary["completed_updates"] = int(schedule.completed)
    summary["cursor"] = int(_cursor)
    summary["pool_size"] = len(pool)
    allocated, reserved = memory_peaks(torch)
    summary["max_memory_allocated_gb"] = allocated
    summary["max_memory_reserved_gb"] = reserved
    summary["memory_unit"] = "1e9_bytes"
    if summary.get("oom_stop"):
        summary["final"] = "FAILED_OOM"
    elif shortfall:
        summary["final"] = "SHORTFALL"
    else:
        summary["final"] = "OK"
    summary["inside_optimizer_step"] = False
    return summary


def emit(summary, code):
    out_dir = summary.get("out_dir")
    if out_dir:
        try:
            files_ok = adapter_files_present(os.path.join(out_dir, "adapter"))
            summary["checkpoint_saved"] = bool(summary.get("saved_this_run") and files_ok)
            if (
                summary.get("checkpoint_saved")
                and summary.get("last_saved_update", 0) == 0
                and summary.get("final") != "OK"
            ):
                summary["checkpoint_note"] = (
                    "adapter on disk is the step-0 initialization, not a finished training run"
                )
            dump_json(os.path.join(out_dir, "SUMMARY.json"), summary, one_line=True)
        except Exception as exc:
            summary["summary_write_error"] = str(exc)[:300]
    line = json.dumps(summary, ensure_ascii=False, separators=(",", ":"), default=str)
    print("G1_RESULT=%s" % summary.get("final"), flush=True)
    print("G1_SUMMARY=%s" % line, flush=True)
    print("G1_DONE", flush=True)
    return code


def run(args):
    summary = {
        "final": "PENDING",
        "checkpoint_saved": False,
        "saved_this_run": False,
        "out_dir": os.path.abspath(args.out_dir) if args.out_dir else None,
        "formatter": FORMATTER_NAME,
        "loss_divided_by_grad_accum": True,
        "effective_batch": int(args.microbatch) * int(args.grad_accum),
    }
    code = 1
    try:
        try:
            import torch  # noqa: F401
            import transformers  # noqa: F401
            import peft  # noqa: F401
            import numpy  # noqa: F401
        except ImportError as exc:
            summary["final"] = "FAILED_MISSING_DEPS"
            summary["error"] = (
                "torch, transformers, peft, and numpy are required to train (%s). "
                "--help does not need them." % exc
            )
            return emit(summary, 1)
        execute(args, summary)
        if summary.get("final") == "SHORTFALL":
            code = 0
        elif summary.get("final") == "OK":
            code = 0
        elif str(summary.get("final", "")).startswith("SKIP"):
            code = 2
        else:
            code = 1
    except SkipRun as exc:
        summary["final"] = exc.status
        summary["error"] = str(exc)
        code = 2
    except ResumeMismatch as exc:
        summary["final"] = exc.status
        summary["error"] = str(exc)
        summary["traceback_tail"] = traceback.format_exc()[-2000:]
        code = 1
    except BaseException as exc:
        status = getattr(exc, "status", None)
        text = str(exc) if str(exc) else type(exc).__name__
        if status:
            summary["final"] = status
        elif isinstance(exc, KeyboardInterrupt):
            summary["final"] = "FAILED_INTERRUPTED"
        elif text.startswith("SKIP_"):
            summary["final"] = text.split(":", 1)[0]
        elif text.startswith("FAILED_"):
            summary["final"] = text.split(":", 1)[0]
        else:
            summary["final"] = "FAILED"
        summary["error"] = text[:500]
        summary["traceback_tail"] = traceback.format_exc()[-2000:]
        code = 130 if isinstance(exc, KeyboardInterrupt) else (2 if str(summary["final"]).startswith("SKIP") else 1)
    if summary.get("out_dir") and summary.get("memory_peak_scope") == "training_loop":
        try:
            import torch

            if torch.cuda.is_available() and "max_memory_allocated_gb" not in summary:
                allocated, reserved = memory_peaks(torch)
                summary["max_memory_allocated_gb"] = allocated
                summary["max_memory_reserved_gb"] = reserved
                summary["memory_unit"] = "1e9_bytes"
        except Exception:
            pass
    summary.pop("_inside_step", None)
    summary.pop("inside_optimizer_step", None)
    return emit(summary, code)


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
