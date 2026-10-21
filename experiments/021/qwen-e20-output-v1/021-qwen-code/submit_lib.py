#!/usr/bin/env python3
"""020 submission helpers: demo-only evidence ranking, deterministic deduplication.

The 017/019 JSON and submission helpers are retained. Candidate source names
never contribute to rank. Query gold is neither accepted nor inspected.
"""
from __future__ import annotations

import json
import math
from collections import Counter
import numbers
import os
import tempfile
from pathlib import Path

VENDORED_FROM = "experiments/013/submission-src/build_submission.py"
VENDORED_NOTE = ("atomic_json, initial_submission, valid_grid, "
                 "existing_submission_matches: verbatim stdlib-only copies; "
                 "no kgmon/NLL/beam/pickle logic vendored.")
INHERITED_FROM = "experiments/017/submit_entry/submit_lib.py"
INHERIT_NOTE = ("017 scheduling/update/validate helpers copied verbatim; "
                "020 ranks candidates by known demo evidence; source labels have no priority.")

PLACEHOLDER = [[0]]



def atomic_json(path, value):
    """Readers see either the previous complete JSON or the new complete JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def initial_submission(challenges):
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("Challenges must be a nonempty task object")
    submission = {}
    query_map = {}
    for task_id, task in challenges.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("Invalid challenge task")
        queries = task.get("test")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Task {task_id!r} has no test queries")
        submission[task_id] = []
        for index, query in enumerate(queries):
            if not isinstance(query, dict) or "input" not in query:
                raise ValueError(f"Invalid test query {task_id}_{index}")
            submission[task_id].append({"attempt_1": [[0]], "attempt_2": [[0]]})
            query_map[f"{task_id}_{index}"] = (task_id, index)
    return submission, query_map


def valid_grid(value):
    """Normalize integer arrays while rejecting ragged/floating/bool grids."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 30:
        raise ValueError("Grid height must be 1..30")
    width = None
    normalized = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 30:
            raise ValueError("Grid width must be 1..30")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("Grid must be rectangular")
        converted = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, numbers.Integral):
                raise ValueError("Grid cells must be integers")
            if not 0 <= cell <= 9:
                raise ValueError("Grid cells must be 0..9")
            converted.append(int(cell))
        normalized.append(converted)
    return normalized


def existing_submission_matches(path, reference):
    """Keep an already valid snapshot while a later candidate scan is running."""
    try:
        with open(path, encoding="utf-8") as stream:
            existing = json.load(stream)
        if not isinstance(existing, dict) or set(existing) != set(reference):
            return False
        for task_id, expected_queries in reference.items():
            queries = existing[task_id]
            if not isinstance(queries, list) or len(queries) != len(expected_queries):
                return False
            for query in queries:
                if not isinstance(query, dict) or set(query) != {"attempt_1", "attempt_2"}:
                    return False
                for grid in query.values():
                    valid_grid(grid)
        return True
    except (OSError, ValueError, TypeError):
        return False


# ---- 017新增：调度、更新、校验（CPU） ----

def count_queries(challenges):
    """Q = 所有test长度之和；240是task数，不保证仅240个query。"""
    total = 0
    for task in challenges.values():
        total += len(task.get("test", []))
    return total


def estimate_and_guard(n_queries, avg_out_tokens, agg_tok_s, wall_left_s,
                       init_s=700.0, prefill_s=2000.0, landing_s=600.0):
    """估算是否装得下。返回(go, report)。超预算返回缺口秒数，不抛异常。"""
    gen_s = (n_queries * avg_out_tokens) / agg_tok_s if agg_tok_s > 0 else float("inf")
    need_s = init_s + prefill_s + gen_s + landing_s
    gap_s = need_s - wall_left_s
    return (gap_s <= 0, {"need_s": round(need_s, 1), "gen_s": round(gen_s, 1),
                         "wall_left_s": round(wall_left_s, 1),
                         "gap_s": round(max(0.0, gap_s), 1)})


def update_slots(submission, query_map, predictions):
    """predictions: {(task_id, index): [grid_a, grid_b]}。单候选重复填两槽；
    坏候选记错，不破坏已有文件（调用方决定是否落盘）。

    返回 (updated_count, errors)。errors为{key: reason}，不抛异常。
    """
    updated, errors = 0, {}
    for key, grids in predictions.items():
        if key not in query_map:
            errors[str(key)] = "unknown query key"
            continue
        task_id, index = query_map[key]
        try:
            ok = [valid_grid(g) for g in grids if g is not None]
        except (ValueError, TypeError) as exc:
            errors[str(key)] = f"invalid candidate: {exc}"[:200]
            continue
        if not ok:
            errors[str(key)] = "no legal candidate"
            continue
        if len(ok) == 1:
            ok = [ok[0], ok[0]]
        submission[task_id][index] = {"attempt_1": ok[0], "attempt_2": ok[1]}
        updated += 1
    return updated, errors


def validate_submission(path, reference):
    """格式校验：键集合/query数/双attempt键/网格合法性。返回(ok, detail)。"""
    try:
        with open(path, encoding="utf-8") as stream:
            existing = json.load(stream)
    except (OSError, ValueError) as exc:
        return False, {"error": f"unreadable: {exc}"[:200]}
    if not isinstance(existing, dict) or set(existing) != set(reference):
        return False, {"error": "task key mismatch"}
    n_q = 0
    for task_id, expected in reference.items():
        queries = existing.get(task_id)
        if not isinstance(queries, list) or len(queries) != len(expected):
            return False, {"error": f"query count mismatch: {task_id}"}
        for q in queries:
            if not isinstance(q, dict) or set(q) != {"attempt_1", "attempt_2"}:
                return False, {"error": f"attempt keys mismatch: {task_id}"}
            for grid in q.values():
                try:
                    valid_grid(grid)
                except (ValueError, TypeError) as exc:
                    return False, {"error": f"bad grid {task_id}: {exc}"[:200]}
            n_q += 1
    return True, {"tasks": len(reference), "queries": n_q}


# ---- 019新增：候选池去重选槽与覆盖率统计（对应019计划§3.2） ----

def _is_placeholder(grid) -> bool:
    return grid == PLACEHOLDER


def demo_metrics(demos, results):
    """Score only known demos; missing/invalid executions remain in denominator.

    Foreground mask is the union of expected/predicted non-background cells,
    where background is the most frequent INPUT color (ties: smaller color).
    Changed mask is the union of expected/predicted differences from input;
    for resizing tasks all output positions count. Shape failures score zero
    on all accuracy components. Means are over demos, not background cells.
    """
    total = len(demos)
    exact = shapes = valid = 0
    fg_sum = changed_sum = pixel_sum = 0.0
    details = []
    for index, demo in enumerate(demos):
        result = results[index] if index < len(results) else {}
        item = {"demo_index": index, "status": result.get("status", "missing"),
                "exact": False, "shape_match": False,
                "foreground_accuracy": 0.0, "changed_accuracy": 0.0,
                "pixel_accuracy": 0.0}
        expected = valid_grid(demo["output"])
        original = valid_grid(demo["input"])
        try:
            if result.get("status") != "ok":
                raise ValueError("not a successful execution")
            predicted = valid_grid(result.get("grid"))
            valid += 1
        except (ValueError, TypeError):
            details.append(item)
            continue
        same_shape = len(predicted) == len(expected) and len(predicted[0]) == len(expected[0])
        if not same_shape:
            details.append(item)
            continue
        shapes += 1
        is_exact = predicted == expected
        exact += int(is_exact)
        counts = Counter(cell for row in original for cell in row)
        background = min(counts, key=lambda color: (-counts[color], color))
        input_same_shape = len(original) == len(expected) and len(original[0]) == len(expected[0])
        fg_n = fg_ok = change_n = change_ok = pixel_ok = 0
        cells = len(expected) * len(expected[0])
        for r, row in enumerate(expected):
            for c, target in enumerate(row):
                value = predicted[r][c]
                correct = value == target
                pixel_ok += int(correct)
                if value != background or target != background:
                    fg_n += 1
                    fg_ok += int(correct)
                if not input_same_shape or target != original[r][c] or value != original[r][c]:
                    change_n += 1
                    change_ok += int(correct)
        foreground = fg_ok / fg_n if fg_n else 1.0
        changed = change_ok / change_n if change_n else 1.0
        pixel = pixel_ok / cells
        fg_sum += foreground
        changed_sum += changed
        pixel_sum += pixel
        item.update(exact=is_exact, shape_match=True, background=background,
                    foreground_accuracy=foreground, foreground_cells=fg_n,
                    changed_accuracy=changed, changed_cells=change_n,
                    pixel_accuracy=pixel)
        details.append(item)
    denominator = max(total, 1)
    return {"demo_total": total, "demo_matches": exact, "matches": exact,
            "all_demos_pass": total > 0 and exact == total,
            "demo_exact_rate": exact / denominator,
            "shape_matches": shapes, "shape_rate": shapes / denominator,
            "foreground_accuracy": fg_sum / denominator,
            "changed_accuracy": changed_sum / denominator,
            "pixel_accuracy": pixel_sum / denominator,
            "valid_demo_outputs": valid, "execution_failures": total - valid,
            "demo_evidence": total > 0, "per_demo": details}


def _metric_record(candidate):
    nested = candidate.get("metrics", candidate.get("demo_metrics"))
    return nested if isinstance(nested, dict) else candidate


def _bounded_metric(metrics, name, default=0.0):
    value = metrics.get(name, default)
    if type(value) not in (int, float) or not math.isfinite(value):
        return default
    return max(0.0, min(1.0, float(value)))


def candidate_sort_key(candidate):
    """Smaller key wins. Source labels and query gold cannot change this key.

    Rank full-demo passes, exact fraction, shape fraction, foreground/change
    accuracy, then pixel accuracy. Ties preserve generation order, then code
    identity. Unknown demo evidence gets zero scores, never fabricated passes.
    """
    metrics = _metric_record(candidate)
    total = metrics.get("demo_total", 0)
    matches = metrics.get("demo_matches", metrics.get("matches", 0))
    known = type(total) is int and total > 0 and type(matches) is int and 0 <= matches <= total
    all_pass = known and matches == total and metrics.get("all_demos_pass") is True
    exact = matches / total if known else 0.0
    scores = [_bounded_metric(metrics, field) if known else 0.0
              for field in ("shape_rate", "foreground_accuracy", "changed_accuracy", "pixel_accuracy")]
    order = candidate.get("order", 0)
    if type(order) not in (int, float) or not math.isfinite(order):
        order = 0
    return (not all_pass, -exact, *(-score for score in scores),
            order, str(candidate.get("code_sha256", candidate.get("label", ""))))


def select_candidates(candidates, limit=2):
    """Return normalized, distinct grid records; accepts flat or nested metrics."""
    if not isinstance(candidates, list):
        return []
    rows = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        try:
            row = dict(candidate)
            row["grid"] = valid_grid(row.get("grid"))
            rows.append(row)
        except (ValueError, TypeError):
            continue
    rows.sort(key=candidate_sort_key)
    selected, seen = [], set()
    for row in rows:
        key = tuple(tuple(cells) for cells in row["grid"])
        if key in seen:
            continue
        seen.add(key)
        selected.append(row)
        if len(selected) >= max(0, min(int(limit), 2)):
            break
    return selected if limit > 0 else []


def merge_candidate_pool(candidates):
    """Compatibility API: rank/deduplicate evidence, return exactly two grids."""
    selected = select_candidates(candidates, limit=2)
    if not selected:
        return ([[0]], [[0]])
    first = selected[0]["grid"]
    return (first, selected[1]["grid"] if len(selected) > 1 else first)


def coverage_report(submission, query_map, updated_keys=None):
    """分离格式校验与真实预测覆盖统计。绝不使用query真值。

    updated_keys 显式提供时，它是当前完整候选池中有真实候选的query键集合，
    不是单次更新增量。按该provenance区分真实 [[0]] 与初始化占位；不判断对错。
    未提供时保留旧的网格值推断，无法区分真实 [[0]]。
      - tasks: 总task数；queries: 总query数Q；
      - format_ok: 双槽键齐全且网格合法的query数；
      - single_covered: 有真实候选的query数；
      - double_distinct: 有真实候选且两槽不同的query数；
      - placeholder: 尚无真实候选的query数。

    两槽必须由完整候选池选出后调用；传pool.keys()前确认池值非空。
    返回 dict，不抛异常（坏输入记入 format_ok=False 的缺口）。
    """
    report = {"tasks": 0, "queries": 0, "format_ok": 0,
              "single_covered": 0, "double_distinct": 0,
              "placeholder": 0}
    try:
        task_ids = set(query_map.values())
    except (AttributeError, TypeError):
        task_ids = set()
    report["tasks"] = len({t for t, _ in task_ids}) if task_ids else (
        len(submission) if isinstance(submission, dict) else 0)
    try:
        keys = list(query_map.keys()) if isinstance(query_map, dict) else []
    except (AttributeError, TypeError):
        keys = []
    report["queries"] = len(keys)
    provenance_keys = None
    if updated_keys is not None:
        try:
            provenance_keys = set(updated_keys) & set(keys)
        except TypeError:
            provenance_keys = set()
        report["updated_keys"] = len(provenance_keys)
    for key in keys:
        try:
            task_id, index = query_map[key]
            entry = submission[task_id][index]
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                continue
            a = valid_grid(entry["attempt_1"])
            b = valid_grid(entry["attempt_2"])
            report["format_ok"] += 1
            if provenance_keys is not None:
                if key in provenance_keys:
                    report["single_covered"] += 1
                    report["double_distinct"] += int(a != b)
                else:
                    report["placeholder"] += 1
                continue
            a_real, b_real = not _is_placeholder(a), not _is_placeholder(b)
            if a_real or b_real:
                report["single_covered"] += 1
            if a_real and b_real and a != b:
                report["double_distinct"] += 1
            if not a_real and not b_real:
                report["placeholder"] += 1
        except (KeyError, IndexError, TypeError, ValueError):
            continue
    return report
