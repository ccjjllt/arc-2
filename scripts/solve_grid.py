"""G1 grid inference on a saved adapter, or on the base model when no adapter is given.

Kaggle L4x4 only: fewer than 4 GPUs exits SKIP_GPU_COUNT. Prompts come from
arc_episode_data. The full generation is parsed as one rectangle; this does not
search a chain-of-thought for a later grid, and it does not keep only a text prefix.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import random
import sys
import time


TIME_BUDGET_S = 3600
HOLDOUT_EVAL_LIMIT = 24
PROMPT_MODE = "enable_thinking_false"
PROMPT_TAIL_CHARS = 200

# Inference output formatters. F0 is the original space-separated grid text;
# F1 is the compact 009 §3.2 form (continuous single digits per row + newline,
# full vocab, output cap such as 960 is a format upper bound, RLE not mainline).
# F1 prompt-side compact serialization is owned by the data layer
# (arc_episode_data.py); this file only parses outputs + reads the formatter.
# Data-layer gap is tracked in experiments/008/code_gaps_followup.json,
# not edited here.
FORMATTER_F0 = "f0_space"
FORMATTER_F1 = "f1_compact"
FORMATTER_CHOICES = ("auto", FORMATTER_F0, FORMATTER_F1)
FORMATTER_CONFIG_FILES = ("adapter_config.json", "run_config.json", "config.json")


class FormatterMismatchError(ValueError):
    """Formatter disagrees with the actual text shape. Loud fail, no fallback."""

# Static / hybrid caches live on the module instance and are reset across generate()
# calls. A fresh dynamic cache is built inside each generate(), which is what we want.
_STATIC_CACHE_IMPLEMENTATIONS = frozenset(
    (
        "static",
        "offloaded_static",
        "sliding_window",
        "hybrid",
        "hybrid_chunked",
        "offloaded_hybrid",
        "offloaded_hybrid_chunked",
    )
)


class UsageError(Exception):
    pass


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=(
            "Score a G1 adapter (or the base model) on ARC tasks. "
            "Needs 4 GPUs; otherwise prints SKIP_GPU_COUNT and exits."
        )
    )
    parser.add_argument("--model-dir", required=True, help="Base model directory.")
    parser.add_argument(
        "--adapter",
        default="",
        help="PEFT adapter directory. Empty means the base model, not merged.",
    )
    parser.add_argument("--challenges", required=True, help="ARC challenges JSON.")
    parser.add_argument(
        "--solutions",
        default="",
        help="ARC solutions JSON. Empty: still generate, exact-match is null.",
    )
    parser.add_argument(
        "--task-ids",
        default="",
        help="Comma-separated task ids. If omitted, use --split-file.",
    )
    parser.add_argument(
        "--split-file",
        default="",
        help="JSON with diag and holdout id lists.",
    )
    parser.add_argument("--out", required=True, help="Per-query JSON output path.")
    parser.add_argument("--max-new-tokens", type=int, default=960)
    parser.add_argument(
        "--time-budget",
        type=int,
        default=TIME_BUDGET_S,
        help="Stop before the next query once this many seconds have passed since startup, including model load. Default 3600.",
    )
    parser.add_argument(
        "--n-sample",
        type=int,
        default=1,
        help="Sampled candidates after greedy. Default 1.",
    )
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument(
        "--compare-mode",
        default="none",
        choices=("none", "base_vs_adapter"),
        help=(
            "none: single-arm run (base when --adapter empty, else adapter). "
            "base_vs_adapter: one model load, per query first base greedy (LoRA disabled) then "
            "adapter greedy; no per-question reload. Switching arms drops cached decode state and "
            "confirms the LoRA is disabled (recorded per candidate). Requires --adapter plus "
            "--max-new-tokens 2048 and --n-sample 0 (greedy only, full prompt, BF16, original "
            "template, thinking=False)."
        ),
    )
    parser.add_argument(
        "--task-view-seed",
        action="store_true",
        help=(
            "Derive sample seeds from stable sha256(task_id,query_index,view,sample_index), not "
            "from process hash or run order. No effect on greedy candidates."
        ),
    )
    parser.add_argument(
        "--second-view",
        default="none",
        choices=("none", "transpose"),
        help=(
            "none: original view only. transpose: extra greedy on the transposed task (all demos "
            "and query transposed), inverse-transposed before scoring, with dedup+stats vs the "
            "original view. Rotation/fixed-recolor views are deferred, not implemented here."
        ),
    )
    parser.add_argument(
        "--formatter",
        default="auto",
        choices=FORMATTER_CHOICES,
        help=(
            "auto: read formatter from the adapter dir (adapter_config.json, run_config.json, "
            "config.json formatter field); no field falls back to f0_space and records "
            "formatter_fallback=true. f0_space/f1_compact: force that output parser."
        ),
    )
    return parser.parse_args(argv)


def _dedupe(ids):
    seen = set()
    out = []
    for item in ids:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out


def _norm_split_label(value):
    label = str(value).strip().lower().replace("-", "_")
    if label in ("diag", "diagnostic", "diagnostics"):
        return "diag"
    if label in ("holdout", "heldout", "held_out"):
        return "holdout"
    return ""


def _id_from_item(item):
    if isinstance(item, str):
        text = item.strip()
        return text or None
    if isinstance(item, dict):
        for key in ("task_id", "id", "tid"):
            if item.get(key):
                return str(item[key]).strip()
    return None


def _ids_in(value):
    if value is None:
        return []
    if isinstance(value, dict):
        for key in ("ids", "task_ids", "tasks"):
            if key in value:
                return _ids_in(value[key])
        return []
    if isinstance(value, list):
        ids = []
        for item in value:
            tid = _id_from_item(item)
            if tid:
                ids.append(tid)
        return ids
    if isinstance(value, str) and value.strip():
        return [value.strip()]
    return []


def _section(data, names):
    if not isinstance(data, dict):
        return None
    for name in names:
        if name in data:
            return data[name]
    for nest_name in ("splits", "split", "groups"):
        nested = data.get(nest_name)
        if isinstance(nested, dict):
            for name in names:
                if name in nested:
                    return nested[name]
    return None


def _labeled_rows(data):
    rows = None
    if isinstance(data, list):
        rows = data
    elif isinstance(data, dict):
        for key in ("tasks", "assignments", "records"):
            if isinstance(data.get(key), list):
                rows = data[key]
                break
        if rows is None and data and all(isinstance(v, str) for v in data.values()):
            rows = [{"task_id": k, "split": v} for k, v in data.items()]
    if not rows:
        return []
    labeled = []
    for item in rows:
        if not isinstance(item, dict):
            continue
        tid = _id_from_item(item)
        label = _norm_split_label(
            item.get("split") or item.get("role") or item.get("group") or ""
        )
        if tid and label:
            labeled.append((tid, label))
    return labeled


def load_split(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)
    diag_raw = _section(data, ("diag", "diagnostic", "diagnostics"))
    hold_raw = _section(data, ("holdout", "heldout", "held_out"))
    diag = _dedupe(_ids_in(diag_raw))
    holdout = _dedupe(_ids_in(hold_raw))
    if diag or holdout:
        return diag, holdout
    diag = []
    holdout = []
    for tid, label in _labeled_rows(data):
        if label == "diag":
            diag.append(tid)
        elif label == "holdout":
            holdout.append(tid)
    diag = _dedupe(diag)
    holdout = _dedupe(holdout)
    if not diag and not holdout:
        raise UsageError(
            "split file has no diag/holdout ids: %s" % path
        )
    return diag, holdout


def select_tasks(challenges, task_ids_arg, split_path):
    diag = []
    holdout = []
    if split_path:
        diag, holdout = load_split(split_path)
    diag_set = set(diag)
    holdout_set = set(holdout)
    if task_ids_arg.strip():
        requested = _dedupe(
            [part.strip() for part in task_ids_arg.split(",") if part.strip()]
        )
        chosen = requested
    elif split_path:
        lex_holdout = sorted(holdout)[:HOLDOUT_EVAL_LIMIT]
        chosen = _dedupe(list(diag) + [tid for tid in lex_holdout if tid not in diag_set])
    else:
        raise UsageError("pass --task-ids or --split-file")
    if not chosen:
        raise UsageError("no tasks selected")
    missing = [tid for tid in chosen if tid not in challenges]
    if missing:
        raise UsageError("task ids not in challenges: %s" % ",".join(missing))
    labeled = []
    for tid in chosen:
        if tid in diag_set:
            split_name = "diag"
        elif tid in holdout_set:
            split_name = "holdout"
        else:
            split_name = "other"
        labeled.append((tid, split_name))
    return labeled


def as_challenge_map(obj):
    if isinstance(obj, dict):
        if "train" in obj and "test" in obj and "task_id" not in obj:
            raise UsageError("challenges JSON is one task; expected a task map")
        return obj
    if isinstance(obj, list):
        mapped = {}
        for item in obj:
            if not isinstance(item, dict):
                raise UsageError("challenges list contains a non-object")
            tid = _id_from_item(item)
            if not tid:
                raise UsageError("challenges list item missing task_id")
            mapped[tid] = item
        return mapped
    raise UsageError("challenges JSON must be an object or a list")


def load_solutions(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as handle:
        obj = json.load(handle)
    if not isinstance(obj, dict):
        raise UsageError("solutions JSON must be an object of task_id -> grids")
    return obj


def gold_for(solutions, task_id, query_index):
    if not solutions or task_id not in solutions:
        return None
    grids = solutions[task_id]
    if not isinstance(grids, list) or query_index >= len(grids) or query_index < 0:
        return None
    grid = grids[query_index]
    if not isinstance(grid, list):
        return None
    return grid


def gold_hw(grid):
    if grid is None:
        return None
    height = len(grid)
    width = len(grid[0]) if height and isinstance(grid[0], list) else 0
    return [height, width]


def _norm_grid(grid):
    return tuple(tuple(int(value) for value in row) for row in grid)


def grids_equal(pred, gold):
    try:
        return _norm_grid(pred) == _norm_grid(gold)
    except (TypeError, ValueError):
        return pred == gold


def score_text(text, gold, text_to_grid):
    """Parse the full new-token string. No search for a later rectangle."""
    try:
        pred = text_to_grid(text)
    except Exception:
        pred = None
    if pred is None:
        parse = "FAIL"
        em = False if gold is not None else None
        return parse, em, None
    parse = "OK"
    em = grids_equal(pred, gold) if gold is not None else None
    stored = pred
    try:
        json.dumps(pred)
    except TypeError:
        stored = None
    return parse, em, stored


def parse_grid_f0_space(text, base_parser=None):
    """F0 space parser. With base_parser, literally uses the original parser.

    Raises FormatterMismatchError when the text is not exact space form.
    Never falls back to compact parsing.
    """
    if base_parser is not None:
        pred = base_parser(text)
        if pred is None:
            raise FormatterMismatchError(
                "f0_space mismatch: text is not exact space-separated digits"
            )
        return pred
    if not isinstance(text, str):
        raise FormatterMismatchError("f0_space mismatch: not a string")
    stripped = text.strip()
    if stripped == "":
        raise FormatterMismatchError("f0_space mismatch: empty text")
    rows = []
    width = None
    for line in stripped.split("\n"):
        if line == "" or line != line.strip():
            raise FormatterMismatchError("f0_space mismatch: bad line padding")
        cells = line.split(" ")
        parsed = []
        for cell in cells:
            if len(cell) != 1 or cell < "0" or cell > "9":
                raise FormatterMismatchError(
                    "f0_space mismatch: cell %r is not one digit" % (cell,)
                )
            parsed.append(ord(cell) - 48)
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise FormatterMismatchError("f0_space mismatch: ragged width")
        rows.append(parsed)
    if width is None or width < 1:
        raise FormatterMismatchError("f0_space mismatch: no columns")
    return rows


def parse_grid_f1_compact(text):
    """F1 compact parser: each row is continuous single digits, newline-split.

    Any space in any row is a loud FormatterMismatchError, never silently
    accepted. Returns the same list-of-lists int grid shape as F0, so the
    outer submission JSON schema (task/query/answer grids) is unchanged.
    """
    if not isinstance(text, str):
        raise FormatterMismatchError("f1_compact mismatch: not a string")
    stripped = text.strip()
    if stripped == "":
        raise FormatterMismatchError("f1_compact mismatch: empty text")
    rows = []
    width = None
    for line in stripped.split("\n"):
        if line == "" or line != line.strip():
            raise FormatterMismatchError("f1_compact mismatch: bad line padding")
        if " " in line:
            raise FormatterMismatchError(
                "f1_compact mismatch: row contains a space"
            )
        parsed = []
        for char in line:
            if char < "0" or char > "9":
                raise FormatterMismatchError(
                    "f1_compact mismatch: char %r is not one digit" % (char,)
                )
            parsed.append(ord(char) - 48)
        if not parsed:
            raise FormatterMismatchError("f1_compact mismatch: empty row")
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            raise FormatterMismatchError("f1_compact mismatch: ragged width")
        rows.append(parsed)
    if width is None or width < 1:
        raise FormatterMismatchError("f1_compact mismatch: no columns")
    return rows


def parse_grid_for_formatter(text, formatter, f0_base_parser=None):
    """Dispatch to exactly one parser. Never mixes, guesses, or falls back."""
    if formatter == FORMATTER_F0:
        return parse_grid_f0_space(text, base_parser=f0_base_parser)
    if formatter == FORMATTER_F1:
        return parse_grid_f1_compact(text)
    raise FormatterMismatchError("unknown formatter: %r" % (formatter,))


def score_text_with_formatter(text, gold, formatter, f0_base_parser=None):
    """Formatter-aware parse. Returns (parse, em, grid, parse_error).

    Mismatch raises inside the parser and lands here as parse FAIL plus an
    explicit parse_error string. No cross-format retry.
    """
    try:
        pred = parse_grid_for_formatter(text, formatter, f0_base_parser)
    except Exception as exc:
        em = False if gold is not None else None
        return "FAIL", em, None, "%s: %s" % (type(exc).__name__, str(exc)[:300])
    parse, em, stored = score_text_pred(pred, gold)
    return parse, em, stored, None


def score_text_pred(pred, gold):
    parse = "OK"
    em = grids_equal(pred, gold) if gold is not None else None
    stored = pred
    try:
        json.dumps(pred)
    except TypeError:
        stored = None
    return parse, em, stored


def _normalize_formatter(value):
    text = str(value).strip().lower().replace("-", "_")
    if text in ("f0_space", "f0", "space"):
        return FORMATTER_F0
    if text in ("f1_compact", "f1", "compact"):
        return FORMATTER_F1
    raise UsageError("unknown formatter value: %r" % (value,))


def _read_formatter_field(obj):
    if isinstance(obj, dict):
        direct = obj.get("formatter")
        if isinstance(direct, str) and direct.strip():
            return direct
        for nested_key in ("data", "episode", "format", "training", "run"):
            nested = obj.get(nested_key)
            if isinstance(nested, dict):
                inner = nested.get("formatter")
                if isinstance(inner, str) and inner.strip():
                    return inner
    return None


def _load_formatter_from_file(path):
    try:
        with open(path, "r", encoding="utf-8") as handle:
            obj = json.load(handle)
    except (OSError, ValueError):
        return None
    found = _read_formatter_field(obj)
    if found is None:
        return None
    return _normalize_formatter(found)


def resolve_formatter(cli_value, adapter_dir):
    """Resolve the output formatter. auto reads the checkpoint, else CLI wins.

    Returns (formatter, source, fell_back). auto with no formatter field
    falls back to f0_space and marks fell_back=True so the caller records
    the fallback explicitly instead of silently assuming F0.
    """
    cli = str(cli_value or "auto").strip().lower().replace("-", "_")
    if cli in ("f0_space", "f0", "space"):
        return FORMATTER_F0, "cli", False
    if cli in ("f1_compact", "f1", "compact"):
        return FORMATTER_F1, "cli", False
    if cli != "auto":
        raise UsageError("--formatter must be auto, f0_space, or f1_compact")
    adapter = (adapter_dir or "").strip()
    if adapter:
        for filename in FORMATTER_CONFIG_FILES:
            candidate = os.path.join(adapter, filename)
            if os.path.isfile(candidate):
                found = _load_formatter_from_file(candidate)
                if found is not None:
                    return found, "adapter:" + filename, False
    return FORMATTER_F0, "fallback_no_field", True


def sample_seed(base, task_index, query_index, sample_index):
    # hash() is process-salted. The key is the tuple the run must reproduce.
    if sample_index == 0:
        key = "%d,%d,%d" % (int(base), int(task_index), int(query_index))
    else:
        key = "%d,%d,%d,%d" % (int(base), int(task_index), int(query_index), int(sample_index))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def text_sha256(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def task_view_seed(base, task_id, query_index, view, sample_index):
    """Stable sha256 seed from task/query/view. Never uses process hash()."""
    key = "%s|%s|%d|%s|%d" % (int(base), str(task_id), int(query_index), str(view), int(sample_index))
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def transpose_grid(grid):
    if not isinstance(grid, list) or not grid or not all(isinstance(row, list) for row in grid):
        raise UsageError("transpose needs a rectangular list-of-lists grid")
    width = len(grid[0])
    if any(len(row) != width for row in grid):
        raise UsageError("transpose needs a rectangular grid")
    return [list(row) for row in zip(*grid)]


def pair_category(base_em, adapter_em):
    if base_em is None or adapter_em is None:
        return None
    if base_em and adapter_em:
        return "both"
    if base_em:
        return "base_only"
    if adapter_em:
        return "adapter_only"
    return "neither"


def size_ok(pred_grid, gold):
    if pred_grid is None or gold is None:
        return None
    try:
        return [len(pred_grid), len(pred_grid[0])] == [len(gold), len(gold[0])]
    except (TypeError, IndexError):
        return False


def pair_aggregate(records):
    out = {
        "n_queries": 0,
        "both": 0,
        "base_only": 0,
        "adapter_only": 0,
        "neither": 0,
        "n_unscored": 0,
        "base_em": 0,
        "adapter_em": 0,
        "base_parse_ok": 0,
        "adapter_parse_ok": 0,
        "base_truncated": 0,
        "adapter_truncated": 0,
    }
    for row in records:
        pair = row.get("pair") or {}
        category = pair.get("category")
        out["n_queries"] += 1
        if category in ("both", "base_only", "adapter_only", "neither"):
            out[category] += 1
        else:
            out["n_unscored"] += 1
        if pair.get("base_em") is True:
            out["base_em"] += 1
        if pair.get("adapter_em") is True:
            out["adapter_em"] += 1
        if pair.get("base_parse") == "OK":
            out["base_parse_ok"] += 1
        if pair.get("adapter_parse") == "OK":
            out["adapter_parse_ok"] += 1
        if pair.get("base_truncated"):
            out["base_truncated"] += 1
        if pair.get("adapter_truncated"):
            out["adapter_truncated"] += 1
    return out


@contextlib.contextmanager
def _base_view(model):
    """Disable the LoRA for one base-arm generate. Yields True when confirmed disabled."""
    disabler = getattr(model, "disable_adapter", None)
    if callable(disabler):
        with disabler():
            yield True
        return
    yield False


def empty_bucket():
    return {
        "n_queries": 0,
        "n_parse_ok": 0,
        "n_truncated_candidates": 0,
        "exact_match_at_1": 0,
        "exact_match_at_2": 0,
        "n_tasks": 0,
        "n_scored": 0,
    }


def aggregate(records):
    """Counts, not rates. exact_match_at_1 is greedy. at_2 is any candidate.

    n_parse_ok counts queries with at least one parse OK. Null em (no gold)
    does not increment exact-match. n_scored is queries whose em is boolean.
    """
    grouped = {"diag": [], "holdout": [], "other": []}
    for record in records:
        grouped[record["split"]].append(record)
    out = {}
    for name, rows in grouped.items():
        bucket = empty_bucket()
        tasks = set()
        for row in rows:
            tasks.add(row["task_id"])
            candidates = row["candidates"]
            if any(item.get("parse") == "OK" for item in candidates):
                bucket["n_parse_ok"] += 1
            bucket["n_truncated_candidates"] += sum(
                1 for item in candidates if item.get("truncated")
            )
            greedy = candidates[0] if candidates else {}
            if greedy.get("em") is True:
                bucket["exact_match_at_1"] += 1
            if any(item.get("em") is True for item in candidates):
                bucket["exact_match_at_2"] += 1
            if any(item.get("em") is not None for item in candidates):
                bucket["n_scored"] += 1
        bucket["n_queries"] = len(rows)
        bucket["n_tasks"] = len(tasks)
        out[name] = bucket
    return out


def write_json(path, payload):
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    temporary = path + ".tmp"
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    os.replace(temporary, path)


def import_episode_api():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    from arc_episode_data import (  # noqa: WPS433
        SYSTEM_PROMPT,
        build_messages,
        grid_to_text,
        text_to_grid,
    )

    if not isinstance(SYSTEM_PROMPT, str) or not SYSTEM_PROMPT.strip():
        raise UsageError("SYSTEM_PROMPT must be a non-empty str")
    for fn in (build_messages, grid_to_text, text_to_grid):
        if not callable(fn):
            raise UsageError("arc_episode_data formatter is not callable")
    return SYSTEM_PROMPT, build_messages, grid_to_text, text_to_grid


def _require_file(path, label):
    if not path or not os.path.isfile(path):
        raise UsageError("%s not found: %s" % (label, path))


def _sync_cuda(torch_mod):
    if not torch_mod.cuda.is_available():
        return
    for index in range(torch_mod.cuda.device_count()):
        torch_mod.cuda.synchronize(index)


def _embedding_device(model):
    try:
        return model.get_input_embeddings().weight.device
    except Exception:
        return next(model.parameters()).device


def _generation_config_has(name):
    from transformers import GenerationConfig

    return hasattr(GenerationConfig(), name)


def _drop_cached_decode_state(model):
    # Static caches hang off the module. Drop them so the next generate() cannot
    # reset or crop the previous candidate's state.
    objects = [model]
    base = getattr(model, "base_model", None)
    if base is not None and not isinstance(base, (str, bytes)):
        objects.append(base)
        inner = getattr(base, "model", None)
        if inner is not None and not isinstance(inner, (str, bytes)):
            objects.append(inner)
    getter = getattr(model, "get_base_model", None)
    if callable(getter):
        try:
            got = getter()
        except Exception:
            got = None
        if got is not None:
            objects.append(got)
    for obj in objects:
        if hasattr(obj, "_cache"):
            try:
                delattr(obj, "_cache")
            except Exception:
                pass


def _install_sample_generator(torch_mod, seed):
    """Point torch.multinomial at one Generator for this generate() call.

    transformers 5.8 samples with torch.multinomial(probs, num_samples=1) and
    does not accept a generator argument on generate(). Passing one fails
    model-kwarg validation. The Generator is created on the probs device.
    """
    original = torch_mod.multinomial
    state = {"gen": None, "off": False}

    def _wrapped(*args, **kwargs):
        if not args or state["off"]:
            return original(*args, **kwargs)
        generator = kwargs.get("generator", None)
        positional_generator = len(args) >= 4
        if generator is None and not positional_generator:
            if state["gen"] is None:
                gen = torch_mod.Generator(device=args[0].device)
                gen.manual_seed(int(seed))
                state["gen"] = gen
            kwargs = dict(kwargs)
            kwargs["generator"] = state["gen"]
        try:
            return original(*args, **kwargs)
        except TypeError as exc:
            if "generator" in kwargs and "generator" in str(exc):
                state["off"] = True
                cleaned = dict(kwargs)
                cleaned.pop("generator", None)
                return original(*args, **cleaned)
            raise

    torch_mod.multinomial = _wrapped
    return original


def _decode_new_text(tokenizer, sequences, prompt_len):
    if hasattr(sequences, "sequences"):
        sequences = sequences.sequences
    row = sequences[0]
    if int(row.shape[0]) < prompt_len:
        new_ids = row[:0]
    else:
        new_ids = row[prompt_len:]
    gen_tokens = int(new_ids.shape[0])
    text = tokenizer.decode(new_ids, skip_special_tokens=True)
    return gen_tokens, text


def _candidate_record(kind, prepare_s, generate_s, gen_tokens, truncated, parse, em, text, grid, error, sample_seed_value, parse_error=None):
    record = {
        "kind": kind,
        "prepare_s": prepare_s,
        "generate_s": generate_s,
        "gen_tokens": gen_tokens,
        "truncated": truncated,
        "parse": parse,
        "em": em,
        "text": text,
        "text_sha256": text_sha256(text),
        "grid": grid,
    }
    if sample_seed_value is not None:
        record["sample_seed"] = sample_seed_value
    if error:
        record["error"] = error
    if parse_error:
        record["parse_error"] = parse_error
    return record


def _run_one_candidate(
    model,
    tokenizer,
    torch_mod,
    prompt_text,
    kind,
    do_sample,
    max_new_tokens,
    seed_value,
    gold,
    text_to_grid,
    in_dev,
    pad_id,
    eos_id,
    pass_presence_penalty,
    force_dynamic_cache,
    formatter=FORMATTER_F0,
):
    prepare_s = None
    generate_s = None
    error = None
    gen_tokens = 0
    text = ""
    prompt_len = None
    t_gen = None
    try:
        _sync_cuda(torch_mod)
        t_prepare = time.perf_counter()
        # Chat text already contains the special tokens. Match training, which
        # tokenizes with add_special_tokens=False.
        encoded = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False)
        input_ids = encoded["input_ids"].to(in_dev)
        attention = encoded["attention_mask"].to(in_dev) if "attention_mask" in encoded else None
        _sync_cuda(torch_mod)
        prepare_s = round(time.perf_counter() - t_prepare, 4)
        prompt_len = int(input_ids.shape[1])
        gen_kwargs = {
            "input_ids": input_ids,
            "max_new_tokens": max_new_tokens,
            "do_sample": bool(do_sample),
            "repetition_penalty": 1.0,
            "num_beams": 1,
            # Override the published chat top_k=20. top_k=0 disables the warper.
            "top_k": 0,
            "no_repeat_ngram_size": 0,
        }
        if attention is not None:
            gen_kwargs["attention_mask"] = attention
        if pad_id is not None:
            gen_kwargs["pad_token_id"] = pad_id
        if eos_id is not None:
            gen_kwargs["eos_token_id"] = eos_id
        if pass_presence_penalty:
            gen_kwargs["presence_penalty"] = 0
        if force_dynamic_cache:
            gen_kwargs["cache_implementation"] = "dynamic"
        if do_sample:
            gen_kwargs["temperature"] = 0.7
            gen_kwargs["top_p"] = 0.9
            random.seed(seed_value)
            torch_mod.manual_seed(seed_value)
            if torch_mod.cuda.is_available():
                torch_mod.cuda.manual_seed_all(seed_value)
        _sync_cuda(torch_mod)
        t_gen = time.perf_counter()
        original_multinomial = None
        try:
            if do_sample:
                original_multinomial = _install_sample_generator(torch_mod, seed_value)
            with torch_mod.inference_mode():
                sequences = model.generate(**gen_kwargs)
            _sync_cuda(torch_mod)
        finally:
            if original_multinomial is not None:
                torch_mod.multinomial = original_multinomial
        generate_s = round(time.perf_counter() - t_gen, 4)
        gen_tokens, text = _decode_new_text(tokenizer, sequences, prompt_len)
        del sequences
        del encoded
        del input_ids
        if attention is not None:
            del attention
    except Exception as exc:
        error = "%s: %s" % (type(exc).__name__, str(exc)[:300])
        print("  %s GEN_FAIL %s" % (kind, error), flush=True)
        if generate_s is None and t_gen is not None:
            generate_s = round(time.perf_counter() - t_gen, 4)
    finally:
        _drop_cached_decode_state(model)
    truncated = bool(gen_tokens >= max_new_tokens)
    parse, em, grid, parse_error = score_text_with_formatter(
        text, gold, formatter, f0_base_parser=text_to_grid
    )
    if error and gold is None:
        em = None
    record = _candidate_record(
        kind,
        prepare_s,
        generate_s,
        gen_tokens,
        truncated,
        parse,
        em,
        text,
        grid,
        error,
        seed_value if do_sample else None,
        parse_error,
    )
    record["formatter"] = formatter
    return record, prompt_len


def _query_jobs(tests):
    if not isinstance(tests, list):
        return []
    jobs = []
    for index, item in enumerate(tests):
        if not isinstance(item, dict) or "input" not in item:
            jobs.append((index, None))
        else:
            jobs.append((index, item["input"]))
    return jobs


def build_payload(
    args,
    records,
    prompt_tail,
    prompt_tail_from,
    time_guard,
    elapsed_s,
    system_prompt,
    adapter_path,
    env,
    sampling_note,
    task_ids,
    formatter=FORMATTER_F0,
    formatter_source="fallback_no_field",
    formatter_fallback=True,
):
    payload = {
        "prompt_mode": PROMPT_MODE,
        "prompt_tail": prompt_tail,
        "prompt_tail_from": prompt_tail_from,
        "system_prompt": system_prompt,
        "adapter": adapter_path,
        "base_merged": False,
        "model_dir": args.model_dir,
        "max_new_tokens": args.max_new_tokens,
        "n_sample": args.n_sample,
        "seed": args.seed,
        "compare_mode": str(getattr(args, "compare_mode", "none")),
        "task_view_seed": bool(getattr(args, "task_view_seed", False)),
        "second_view": str(getattr(args, "second_view", "none")),
        "formatter": formatter,
        "formatter_source": formatter_source,
        "formatter_fallback": bool(formatter_fallback),
        "time_guard": bool(time_guard),
        "elapsed_s": elapsed_s,
        "task_ids": task_ids,
        "sampling": sampling_note,
        "env": env,
        "aggregates": aggregate(records),
        "queries": records,
    }
    if any(isinstance(row, dict) and "pair" in row for row in records):
        payload["pair_aggregates"] = pair_aggregate(records)
    return payload


def summary_from(records, time_guard):
    buckets = aggregate(records)
    summary = {
        "time_guard": bool(time_guard),
        "diag": buckets["diag"],
        "holdout": buckets["holdout"],
        "other": buckets["other"],
    }
    if any(isinstance(row, dict) and "pair" in row for row in records):
        summary["pair"] = pair_aggregate(records)
    return summary


def main(argv=None):
    args = parse_args(argv)
    t_start = time.perf_counter()
    try:
        if args.max_new_tokens <= 0:
            raise UsageError("--max-new-tokens must be > 0")
        if args.time_budget <= 0:
            raise UsageError("--time-budget must be > 0")
        if args.n_sample < 0:
            raise UsageError("--n-sample must be >= 0")
        if not os.path.isdir(args.model_dir):
            raise UsageError("model dir not found: %s" % args.model_dir)
        if not os.path.isfile(os.path.join(args.model_dir, "config.json")):
            raise UsageError("model dir has no config.json: %s" % args.model_dir)
        _require_file(args.challenges, "challenges")
        adapter = args.adapter.strip()
        if adapter and not os.path.isdir(adapter):
            raise UsageError("adapter dir not found: %s" % adapter)
        solutions_path = args.solutions.strip()
        if solutions_path:
            _require_file(solutions_path, "solutions")
        split_path = args.split_file.strip()
        if split_path:
            _require_file(split_path, "split file")
        if not args.task_ids.strip() and not split_path:
            raise UsageError("pass --task-ids or --split-file")
        compare = str(args.compare_mode)
        if compare not in ("none", "base_vs_adapter"):
            raise UsageError("--compare-mode must be none or base_vs_adapter")
        second_view = str(args.second_view)
        if second_view not in ("none", "transpose"):
            raise UsageError("--second-view must be none or transpose")
        formatter_cli = str(getattr(args, "formatter", "auto"))
        if formatter_cli not in FORMATTER_CHOICES:
            raise UsageError("--formatter must be auto, f0_space, or f1_compact")
        if compare == "base_vs_adapter":
            if not adapter:
                raise UsageError("--compare-mode base_vs_adapter requires --adapter")
            if int(args.max_new_tokens) != 2048:
                raise UsageError(
                    "--compare-mode base_vs_adapter requires --max-new-tokens 2048 "
                    "(unified cap, full prompt)"
                )
            if int(args.n_sample) != 0:
                raise UsageError(
                    "--compare-mode base_vs_adapter requires --n-sample 0 (greedy pair only)"
                )

        try:
            system_prompt, build_messages, grid_to_text, text_to_grid = import_episode_api()
        except ImportError as exc:
            raise UsageError(
                "cannot import arc_episode_data (%s). Prompts are not formatted here." % exc
            )
        if not callable(grid_to_text):
            raise UsageError("grid_to_text is not callable")
        formatter, formatter_source, formatter_fallback = resolve_formatter(
            formatter_cli, adapter
        )
        print(
            "formatter=%s source=%s fallback=%s"
            % (formatter, formatter_source, bool(formatter_fallback)),
            flush=True,
        )
        with open(args.challenges, "r", encoding="utf-8") as handle:
            challenges = as_challenge_map(json.load(handle))
        solutions = load_solutions(solutions_path)
        selected = select_tasks(challenges, args.task_ids, split_path)
        n_queries = 0
        for task_id, _split_name in selected:
            tests = challenges[task_id].get("test") or []
            n_queries += len(tests) if isinstance(tests, list) else 0
        print(
            "tasks=%d queries=%d adapter=%s"
            % (len(selected), n_queries, adapter or "none"),
            flush=True,
        )

        try:
            import torch
        except ImportError:
            print("SKIP_GPU_COUNT", flush=True)
            print("SKIP_GPU_COUNT need 4 GPUs, found 0 (torch is not installed)", flush=True)
            return 3

        if (not torch.cuda.is_available()) or torch.cuda.device_count() < 4:
            found = torch.cuda.device_count() if torch.cuda.is_available() else 0
            print("SKIP_GPU_COUNT", flush=True)
            print("SKIP_GPU_COUNT need 4 GPUs, found %d" % found, flush=True)
            return 3

        from transformers import AutoModelForCausalLM, AutoTokenizer

        peft_model_cls = None
        if adapter:
            from peft import PeftModel

            peft_model_cls = PeftModel
        pass_presence_penalty = _generation_config_has("presence_penalty")
        tokenizer = AutoTokenizer.from_pretrained(args.model_dir, trust_remote_code=True)
        t_load = time.perf_counter()
        model = AutoModelForCausalLM.from_pretrained(
            args.model_dir,
            dtype=torch.bfloat16,
            device_map="balanced",
            low_cpu_mem_usage=True,
            trust_remote_code=False,
        )
        adapter_path = None
        if adapter:
            model = peft_model_cls.from_pretrained(model, adapter)
            adapter_path = os.path.abspath(adapter)
        model.eval()
        _sync_cuda(torch)
        load_s = round(time.perf_counter() - t_load, 3)
        print("model loaded in %.1fs base_merged=false" % load_s, flush=True)
        in_dev = _embedding_device(model)
        print("input_device: %s" % in_dev, flush=True)
        gen_cfg = getattr(model, "generation_config", None)
        eos_id = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
        if eos_id is None:
            eos_id = tokenizer.eos_token_id
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            if isinstance(eos_id, (list, tuple)) and eos_id:
                pad_id = eos_id[0]
            else:
                pad_id = eos_id
        inherited_top_k = getattr(gen_cfg, "top_k", None) if gen_cfg is not None else None
        inherited_ngram = getattr(gen_cfg, "no_repeat_ngram_size", None) if gen_cfg is not None else None
        cache_impl = getattr(gen_cfg, "cache_implementation", None) if gen_cfg is not None else None
        force_dynamic_cache = cache_impl in _STATIC_CACHE_IMPLEMENTATIONS
        sampling_note = {
            "greedy": {
                "do_sample": False,
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "top_k": 0,
                "no_repeat_ngram_size": 0,
            },
            "sample": {
                "do_sample": True,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.0,
                "num_beams": 1,
                "top_k": 0,
                "no_repeat_ngram_size": 0,
                "n": args.n_sample,
            },
            "top_k": 0,
            "no_repeat_ngram_size": 0,
            "presence_penalty_passed": bool(pass_presence_penalty),
            "inherited_top_k": inherited_top_k,
            "inherited_no_repeat_ngram_size": inherited_ngram,
            "seed_key": "{seed},{task_index},{query_index}",
            "task_view_seed": bool(args.task_view_seed),
            "task_view_seed_key": "{seed}|{task_id}|{query_index}|{view}|{sample_index} sha256"
            if args.task_view_seed
            else None,
            "compare_mode": compare,
            "second_view": second_view,
            "formatter": formatter,
            "formatter_source": formatter_source,
            "formatter_fallback": bool(formatter_fallback),
            "cache": "new DynamicCache per generate; not cropped across candidates",
        }
        try:
            import transformers

            tf_version = transformers.__version__
        except Exception:
            tf_version = None
        env = {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "transformers": tf_version,
            "gpu_count": torch.cuda.device_count(),
            "load_s": load_s,
        }
        if adapter_path:
            try:
                import peft

                env["peft"] = peft.__version__
            except Exception:
                env["peft"] = None

        records = []
        prompt_tail = None
        prompt_tail_from = None
        time_guard = False
        stopped = False
        for task_index, (task_id, split_name) in enumerate(selected):
            if stopped:
                break
            task = challenges[task_id]
            demos = task.get("train") or []
            jobs = _query_jobs(task.get("test") or [])
            elapsed = time.perf_counter() - t_start
            print(
                "-- task %d/%d %s split=%s queries=%d elapsed=%.1fs"
                % (task_index + 1, len(selected), task_id, split_name, len(jobs), elapsed),
                flush=True,
            )
            for query_index, query_input in jobs:
                if time.perf_counter() - t_start >= args.time_budget:
                    time_guard = True
                    stopped = True
                    print(
                        "TIME_GUARD elapsed %.1fs >= %ds; not starting %s query %d"
                        % (time.perf_counter() - t_start, args.time_budget, task_id, query_index),
                        flush=True,
                    )
                    break
                gold = gold_for(solutions, task_id, query_index)
                if query_input is None:
                    candidates = [
                        _candidate_record(
                            "greedy",
                            None,
                            None,
                            0,
                            False,
                            "FAIL",
                            False if gold is not None else None,
                            "",
                            None,
                            "missing query input",
                            None,
                        )
                    ]
                    records.append(
                        {
                            "task_id": task_id,
                            "query_index": query_index,
                            "split": split_name,
                            "prompt_tokens": None,
                            "gold_hw": gold_hw(gold),
                            "candidates": candidates,
                        }
                    )
                    continue
                try:
                    messages = build_messages(demos, query_input)
                    prompt_text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                except Exception as exc:
                    prompt_text = None
                    prompt_error = "%s: %s" % (type(exc).__name__, str(exc)[:300])
                else:
                    prompt_error = None
                if not isinstance(prompt_text, str):
                    if prompt_error is None:
                        prompt_error = "apply_chat_template did not return a string"
                    print("  q%d PROMPT_FAIL %s" % (query_index, prompt_error), flush=True)
                    records.append(
                        {
                            "task_id": task_id,
                            "query_index": query_index,
                            "split": split_name,
                            "prompt_tokens": None,
                            "gold_hw": gold_hw(gold),
                            "candidates": [
                                _candidate_record(
                                    "greedy",
                                    None,
                                    None,
                                    0,
                                    False,
                                    "FAIL",
                                    False if gold is not None else None,
                                    "",
                                    None,
                                    prompt_error,
                                    None,
                                )
                            ],
                        }
                    )
                    continue
                if prompt_tail is None:
                    prompt_tail = prompt_text[-PROMPT_TAIL_CHARS:]
                    prompt_tail_from = {"task_id": task_id, "query_index": query_index}
                candidates = []
                prompt_tokens = None
                prompt_text_t = None
                transpose_error = None
                if second_view == "transpose":
                    # Same template/thinking=False/full prompt; every demo and the query share
                    # one transposed view. Rotation/recolor views are deferred.
                    try:
                        t_demos = [
                            {
                                "input": transpose_grid(demo.get("input")),
                                "output": transpose_grid(demo.get("output")),
                            }
                            for demo in demos
                        ]
                        t_messages = build_messages(t_demos, transpose_grid(query_input))
                        prompt_text_t = tokenizer.apply_chat_template(
                            t_messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False,
                        )
                    except Exception as exc:
                        transpose_error = "%s: %s" % (type(exc).__name__, str(exc)[:300])

                def _gen(prompt, kind, do_sample, seed_value, gold_ref, use_base):
                    if use_base:
                        _drop_cached_decode_state(model)
                        with _base_view(model) as confirmed:
                            made, plen = _run_one_candidate(
                                model,
                                tokenizer,
                                torch,
                                prompt,
                                kind,
                                do_sample,
                                args.max_new_tokens,
                                seed_value,
                                gold_ref,
                                text_to_grid,
                                in_dev,
                                pad_id,
                                eos_id,
                                pass_presence_penalty,
                                force_dynamic_cache,
                                formatter,
                            )
                            made["adapter_disabled"] = True
                            made["adapter_disabled_confirmed"] = bool(confirmed)
                            return made, plen
                    _drop_cached_decode_state(model)
                    made, plen = _run_one_candidate(
                        model,
                        tokenizer,
                        torch,
                        prompt,
                        kind,
                        do_sample,
                        args.max_new_tokens,
                        seed_value,
                        gold_ref,
                        text_to_grid,
                        in_dev,
                        pad_id,
                        eos_id,
                        pass_presence_penalty,
                        force_dynamic_cache,
                        formatter,
                    )
                    return made, plen

                def _log(kind, record):
                    print(
                        "  q%d %s gen_tok=%s trunc=%s parse=%s em=%s prepare=%s generate=%s"
                        % (
                            query_index,
                            kind,
                            record["gen_tokens"],
                            record["truncated"],
                            record["parse"],
                            record["em"],
                            record["prepare_s"],
                            record["generate_s"],
                        ),
                        flush=True,
                    )

                def _transpose_candidate(orig_grid, prompt_t, kind, use_base):
                    if prompt_t is None:
                        return _candidate_record(
                            kind,
                            None,
                            None,
                            0,
                            False,
                            "FAIL",
                            False if gold is not None else None,
                            "",
                            None,
                            transpose_error or "transpose prompt failed",
                            None,
                        ), None
                    made, plen = _gen(prompt_t, kind, False, None, None, use_base)
                    inv = None
                    try:
                        if made.get("grid") is not None:
                            inv = transpose_grid(made["grid"])
                    except UsageError:
                        inv = None
                    if inv is None and made.get("parse") == "OK":
                        made["parse"] = "FAIL"
                    made["grid"] = inv
                    made["em"] = grids_equal(inv, gold) if (gold is not None and inv is not None) else (
                        False if gold is not None else None
                    )
                    made["view"] = "transpose"
                    made["inverse_applied"] = True
                    try:
                        made["dedupe_equal_to_orig"] = (
                            _norm_grid(inv) == _norm_grid(orig_grid)
                            if (inv is not None and orig_grid is not None)
                            else None
                        )
                    except (TypeError, ValueError):
                        made["dedupe_equal_to_orig"] = (inv == orig_grid)
                    return made, plen

                row = {
                    "task_id": task_id,
                    "query_index": query_index,
                    "split": split_name,
                    "prompt_tokens": None,
                    "gold_hw": gold_hw(gold),
                    "compare_mode": compare,
                    "second_view": second_view,
                    "formatter": formatter,
                    "formatter_source": formatter_source,
                    "formatter_fallback": bool(formatter_fallback),
                    "candidates": candidates,
                }
                if compare == "base_vs_adapter":
                    # Budget is checked before the query. Both arms still run.
                    for kind, use_base in (("base_greedy", True), ("adapter_greedy", False)):
                        record, prompt_len = _gen(
                            prompt_text, kind, False, None, gold, use_base
                        )
                        record["view"] = "original"
                        if prompt_tokens is None and prompt_len is not None:
                            prompt_tokens = prompt_len
                        candidates.append(record)
                        _log(kind, record)
                    by_kind = {item.get("kind"): item for item in candidates}
                    if second_view == "transpose":
                        for kind, use_base in (
                            ("base_greedy_transpose", True),
                            ("adapter_greedy_transpose", False),
                        ):
                            orig_grid = (by_kind.get(kind.replace("_transpose", "")) or {}).get("grid")
                            record, _plen = _transpose_candidate(
                                orig_grid, prompt_text_t, kind, use_base
                            )
                            candidates.append(record)
                            _log(kind, record)
                    base_rec = by_kind.get("base_greedy", {})
                    adapter_rec = by_kind.get("adapter_greedy", {})
                    row["prompt_tokens"] = prompt_tokens
                    row["pair"] = {
                        "category": pair_category(base_rec.get("em"), adapter_rec.get("em")),
                        "base_em": base_rec.get("em"),
                        "adapter_em": adapter_rec.get("em"),
                        "base_parse": base_rec.get("parse"),
                        "adapter_parse": adapter_rec.get("parse"),
                        "base_size_ok": size_ok(base_rec.get("grid"), gold),
                        "adapter_size_ok": size_ok(adapter_rec.get("grid"), gold),
                        "base_truncated": base_rec.get("truncated"),
                        "adapter_truncated": adapter_rec.get("truncated"),
                        "base_gen_tokens": base_rec.get("gen_tokens"),
                        "adapter_gen_tokens": adapter_rec.get("gen_tokens"),
                        "base_prepare_s": base_rec.get("prepare_s"),
                        "base_generate_s": base_rec.get("generate_s"),
                        "adapter_prepare_s": adapter_rec.get("prepare_s"),
                        "adapter_generate_s": adapter_rec.get("generate_s"),
                    }
                else:
                    plan = [("greedy", False, None, "original")]
                    for sample_index in range(args.n_sample):
                        plan.append(("sample", True, sample_index, "original"))
                    # Budget is checked before the query. All candidates still run.
                    for kind, do_sample, sample_index, _view in plan:
                        seed_value = None
                        if do_sample:
                            if args.task_view_seed:
                                seed_value = task_view_seed(
                                    args.seed, task_id, query_index, "original", sample_index
                                )
                            else:
                                seed_value = sample_seed(
                                    args.seed, task_index, query_index, sample_index
                                )
                        record, prompt_len = _gen(
                            prompt_text, kind, do_sample, seed_value, gold, False
                        )
                        record["view"] = "original"
                        if prompt_tokens is None and prompt_len is not None:
                            prompt_tokens = prompt_len
                        candidates.append(record)
                        _log(kind, record)
                    if second_view == "transpose":
                        record, _plen = _transpose_candidate(
                            (candidates[0] if candidates else {}).get("grid"),
                            prompt_text_t,
                            "greedy_transpose",
                            False,
                        )
                        candidates.append(record)
                        _log("greedy_transpose", record)
                    row["prompt_tokens"] = prompt_tokens
                if second_view == "transpose":
                    seen = set()
                    for item in candidates:
                        grid = item.get("grid")
                        if grid is None:
                            continue
                        try:
                            seen.add(_norm_grid(grid))
                        except (TypeError, ValueError):
                            seen.add(repr(grid))
                    row["view_stats"] = {
                        "n_candidates": len(candidates),
                        "n_unique_grids": len(seen),
                        "oracle_em": any(item.get("em") is True for item in candidates),
                        "transpose_dedupe_equal": [
                            item.get("dedupe_equal_to_orig")
                            for item in candidates
                            if item.get("view") == "transpose"
                        ],
                    }
                records.append(row)
            payload = build_payload(
                args,
                records,
                prompt_tail,
                prompt_tail_from,
                time_guard,
                round(time.perf_counter() - t_start, 3),
                system_prompt,
                adapter_path,
                env,
                sampling_note,
                [tid for tid, _name in selected],
                formatter,
                formatter_source,
                formatter_fallback,
            )
            write_json(args.out, payload)
        summary = summary_from(records, time_guard)
        print(
            "SOLVE_SUMMARY=" + json.dumps(summary, ensure_ascii=False, separators=(",", ":")),
            flush=True,
        )
        print("SOLVE_DONE", flush=True)
        return 0
    except UsageError as exc:
        print(str(exc), flush=True)
        return 2


if __name__ == "__main__":
    sys.exit(main())
