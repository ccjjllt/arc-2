#!/usr/bin/env python3
"""017 B: 最小全量提交连接（本地CPU）。只借用013四个纯函数，其余为本轮新增。

借用来源: experiments/013/submission-src/build_submission.py（逐字复制，
不含kgmon/NLL/beam排序/pickle候选路径）。不导入013模块，避免其重量依赖。
本文件只处理整数网格JSON；围栏解析属于模型文本侧，不在这里重复。
"""
from __future__ import annotations

import json
import numbers
import os
import tempfile
from pathlib import Path

VENDORED_FROM = "experiments/013/submission-src/build_submission.py"
VENDORED_NOTE = ("atomic_json, initial_submission, valid_grid, "
                 "existing_submission_matches: verbatim stdlib-only copies; "
                 "no kgmon/NLL/beam/pickle logic vendored.")


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
