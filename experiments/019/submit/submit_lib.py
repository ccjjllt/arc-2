#!/usr/bin/env python3
"""019 submit_lib: 继承017最小全量提交连接，新增候选池去重选槽与覆盖率统计。

继承（逐字复制，不改017语义）:
- 源文件: experiments/017/submit_entry/submit_lib.py
- atomic_json, initial_submission, valid_grid, existing_submission_matches,
  count_queries, estimate_and_guard, update_slots, validate_submission
  均与017逐字一致；update_slots 的“单候选重复填两槽”行为保留，
  但019驱动不再逐候选直接调用它，改为先经候选池选出双槽后再调用一次。

019新增（对应019计划§3.2）:
- merge_candidate_pool: 每题候选池去重并选出双槽（绝不使用query真值）。
- coverage_report: 分开统计格式合法 / 单槽真实 / 双槽相异 / 占位。
- PLACEHOLDER: 初始占位网格 [[0]] 的唯一标记。

stdlib only。
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
INHERITED_FROM = "experiments/017/submit_entry/submit_lib.py"
INHERIT_NOTE = ("017 scheduling/update/validate helpers copied verbatim; "
                "019 adds pool selection + coverage only.")

PLACEHOLDER = [[0]]

# 来源优先级（数字越小越优先）：program_fix > program_init > direct_grid
SOURCE_PRIORITY = {"program_fix": 0, "program_init": 1, "direct_grid": 2}


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


def merge_candidate_pool(candidates):
    """按固定规则去重并选出双槽。绝不使用query真值。

    candidates: list[dict]，每条记录：
      - grid: 通过 valid_grid 校验的归一化整数网格
      - source: 'program_fix' | 'program_init' | 'direct_grid'
      - all_demos_pass: bool（程序候选在全部train示例上是否逐格全过；
        直出网格记 False）
      - demo_matches: int（命中的train示例数）
      - order: 生成先后序号（用于稳定同分排序）

    规则：
      1. 排序：all_demos_pass==True 优先 → demo_matches 降序 →
         来源优先级（program_fix > program_init > direct_grid）→ order 升序；
      2. 按网格内容逐格去重（保留首次出现的最高优先级候选）；
      3. 去重后 >= 2 个不同合法网格取前2个；仅1个则重复填两槽；
         0个返回 ([[0]], [[0]]) 占位。

    返回 (attempt_1, attempt_2)。非法输入一律视为0候选，不抛异常。
    """
    if not isinstance(candidates, list):
        return ([[0]], [[0]])
    rows = []
    for c in candidates:
        try:
            if not isinstance(c, dict):
                continue
            grid = valid_grid(c.get("grid"))
            source = c.get("source")
            if source not in SOURCE_PRIORITY:
                continue
            all_pass = c.get("all_demos_pass")
            matches = c.get("demo_matches")
            order = c.get("order")
            if type(all_pass) is not bool:
                continue
            if type(matches) is not int or matches < 0:
                continue
            if type(order) not in (int, float):
                continue
            rows.append({"grid": grid, "source": source,
                         "all_demos_pass": all_pass,
                         "demo_matches": matches, "order": order})
        except (ValueError, TypeError, AttributeError):
            continue
    rows.sort(key=lambda r: (not r["all_demos_pass"], -r["demo_matches"],
                             SOURCE_PRIORITY[r["source"]], r["order"]))
    uniq = []
    seen = set()
    for r in rows:
        key = tuple(tuple(row) for row in r["grid"])
        if key in seen:
            continue
        seen.add(key)
        uniq.append(r["grid"])
    if len(uniq) >= 2:
        return (uniq[0], uniq[1])
    if len(uniq) == 1:
        return (uniq[0], uniq[0])
    return ([[0]], [[0]])


def coverage_report(submission, query_map, updated_keys=None):
    """分离格式校验与真实预测覆盖统计。绝不使用query真值。

    只看网格是否为初始 [[0]] 占位，不判断对错：
      - tasks: 总task数；queries: 总query数Q；
      - format_ok: 双槽键齐全且网格合法的query数；
      - single_covered: 至少1槽为真实非占位预测的query数；
      - double_distinct: 两槽均为真实预测且互不相同的query数；
      - placeholder: 仍停留在初始 [[0]] 占位的query数（双槽皆占位）。

    updated_keys 可选，用于标注本次更新涉及的key集合（仅计数用）。
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
    if updated_keys is not None:
        try:
            report["updated_keys"] = len(set(updated_keys) & set(keys))
        except TypeError:
            report["updated_keys"] = 0
    for key in keys:
        try:
            task_id, index = query_map[key]
            entry = submission[task_id][index]
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"}:
                continue
            a = valid_grid(entry["attempt_1"])
            b = valid_grid(entry["attempt_2"])
            report["format_ok"] += 1
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
