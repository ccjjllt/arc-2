#!/usr/bin/env python3
"""018-B1/B2: constrained op-list programs + reliable object primitives (CPU only).

Object model keeps true member cells [(r,c,v)]; bbox is derived, never content.
Semantics aligned with michaelhodel/arc-dsl README (consulted 2026-10-19):
objects(grid, univalued, diagonal, without_bg) -> cell sets keeping
positions+colors; paint/fill/cover operate on those sets. Small subset ported;
limits (background = most frequent color; first-version coverage) in plan 018.
"""
from __future__ import annotations

import itertools
import time

MAX_STEPS = 12
MAX_CHOICES_PER_PARAM = 4
MAX_CANDIDATES = 32
SEARCH_TIME_BUDGET_S = 10.0

OP_NAMES = ("extract_objects", "filter_objects", "sort_objects",
            "transform_objects", "place_objects", "output")

PARAM_SPECS = {
    "extract_objects": {
        "same_color": ("bool", None),
        "connectivity": ("enum", (4, 8)),
    },
    "filter_objects": {
        "criterion": ("enum", ("area_gt", "area_lt", "color_eq", "touches_border")),
        "value": ("int", (0, 900)),
    },
    "sort_objects": {
        "by": ("enum", ("area", "row", "col")),
        "order": ("enum", ("asc", "desc")),
    },
    "transform_objects": {
        "op": ("enum", ("rotate", "flip_h", "flip_v", "shift")),
        "quarter_turns": ("int", (0, 3)),
        "dr": ("int", (-29, 29)),
        "dc": ("int", (-29, 29)),
    },
    "place_objects": {
        "background": ("color", None),
        "canvas": ("enum", ("blank", "input")),
        "overlap": ("enum", ("error", "overwrite")),
    },
    "output": {},
}

REQUIRED_PARAMS = {
    "extract_objects": ("same_color", "connectivity"),
    "transform_objects": ("op",),
    "place_objects": ("background",),
}

DEFAULT_PARAMS = {
    "place_objects": {"canvas": "blank", "overlap": "error"},
    "transform_objects": {"quarter_turns": 0, "dr": 0, "dc": 0},
}


def _check_scalar(kind, constraint, value, where):
    if kind == "bool":
        if type(value) is not bool:
            raise ValueError(f"{where}: expected bool")
    elif kind == "int":
        if type(value) is not int:
            raise ValueError(f"{where}: expected int")
        lo, hi = constraint
        if not (lo <= value <= hi):
            raise ValueError(f"{where}: out of range")
    elif kind == "enum":
        if value not in constraint:
            raise ValueError(f"{where}: not in enum")
    elif kind == "color":
        if type(value) is not int or not (0 <= value <= 9):
            raise ValueError(f"{where}: color must be 0..9")
    else:
        raise ValueError(f"{where}: unknown spec kind")


def _norm_value(kind, constraint, value, where):
    if isinstance(value, dict):
        if set(value.keys()) != {"choices"}:
            raise ValueError(f"{where}: only {{choices}} mapping allowed")
        ch = value["choices"]
        if not isinstance(ch, list) or not 1 <= len(ch) <= MAX_CHOICES_PER_PARAM:
            raise ValueError(f"{where}: choices need 1..{MAX_CHOICES_PER_PARAM} options")
        seen = set()
        for c in ch:
            _check_scalar(kind, constraint, c, where)
            key = (type(c).__name__, c)
            if key in seen:
                raise ValueError(f"{where}: duplicate choice")
            seen.add(key)
        return {"choices": list(ch)}
    _check_scalar(kind, constraint, value, where)
    return value


def validate_template(plan) -> dict:
    """Strict structural check. Returns normalized plan or raises ValueError."""
    if not isinstance(plan, dict) or set(plan.keys()) != {"steps"}:
        raise ValueError("plan must be exactly {'steps': [...]}")
    steps = plan["steps"]
    if not isinstance(steps, list) or not 1 <= len(steps) <= MAX_STEPS:
        raise ValueError("steps must be a list of 1..12")
    norm = []
    for i, step in enumerate(steps):
        where = f"steps[{i}]"
        if not isinstance(step, dict) or set(step.keys()) != {"op", "params"}:
            raise ValueError(f"{where}: must be exactly {{op, params}}")
        op, params = step["op"], step["params"]
        if op not in PARAM_SPECS:
            raise ValueError(f"{where}: unknown op {op!r}")
        if not isinstance(params, dict):
            raise ValueError(f"{where}: params must be an object")
        spec = PARAM_SPECS[op]
        unknown = set(params.keys()) - set(spec.keys())
        if unknown:
            raise ValueError(f"{where}: unknown params {sorted(unknown)}")
        for req in REQUIRED_PARAMS.get(op, ()):
            if req not in params:
                raise ValueError(f"{where}: missing required param {req}")
        np = {}
        for k, v in params.items():
            kind, constraint = spec[k]
            np[k] = _norm_value(kind, constraint, v, f"{where}.{k}")
        for k, v in DEFAULT_PARAMS.get(op, {}).items():
            np.setdefault(k, v)
        norm.append({"op": op, "params": np})
    if not any(s["op"] == "place_objects" for s in norm):
        raise ValueError("plan must contain place_objects")
    if not any(s["op"] == "output" for s in norm):
        raise ValueError("plan must contain output")
    return {"steps": norm}


def guided_schema() -> dict:
    """JSON-schema for vLLM guided_json, generated from the same constants.
    Server enforcement is best-effort; validate_template() is authoritative."""
    def p(name, kind, constraint):
        if kind == "bool":
            return {"type": "boolean"}
        if kind == "int":
            lo, hi = constraint
            return {"type": "integer", "minimum": lo, "maximum": hi}
        if kind == "enum":
            return {"enum": list(constraint)}
        if kind == "color":
            return {"type": "integer", "minimum": 0, "maximum": 9}
        raise ValueError(name)
    steps = []
    for op in OP_NAMES:
        props = {k: p(k, *v) for k, v in PARAM_SPECS[op].items()}
        steps.append({"type": "object",
                      "properties": {"op": {"const": op}, "params": {
                          "type": "object", "properties": props,
                          "additionalProperties": False,
                          **({"required": list(REQUIRED_PARAMS[op])}
                             if op in REQUIRED_PARAMS else {})}},
                      "required": ["op", "params"],
                      "additionalProperties": False})
    return {"type": "object",
            "properties": {"steps": {"type": "array", "items": {"anyOf": steps},
                                     "minItems": 1, "maxItems": MAX_STEPS}},
            "required": ["steps"], "additionalProperties": False}


def _neighbors(r, c, connectivity):
    if connectivity == 4:
        return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1),
            (r - 1, c - 1), (r - 1, c + 1), (r + 1, c - 1), (r + 1, c + 1))


def _extract(grid, same_color, connectivity, bg):
    H, W = len(grid), len(grid[0])
    seen = [[False] * W for _ in range(H)]
    objects = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc, grid[cr][cc]))
                for nr, nc in _neighbors(cr, cc, connectivity):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] \
                            and grid[nr][nc] != bg \
                            and (not same_color or grid[nr][nc] == grid[cr][cc]):
                        seen[nr][nc] = True
                        stack.append((nr, nc))
            objects.append(_make_object(cells))
    return objects


def _make_object(cells):
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return {"cells": sorted(cells),
            "colors": sorted({v for _, _, v in cells}),
            "bbox": [min(rs), min(cs), max(rs), max(cs)],
            "area": len(cells)}


def _touches_border(obj, H, W):
    r0, c0, r1, c1 = obj["bbox"]
    return r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1


def _key(obj, by):
    if by == "area":
        return obj["area"]
    if by == "row":
        return obj["bbox"][0]
    return obj["bbox"][1]


def _transform_cells(cells, op, qt, dr, dc):
    if op == "rotate":
        for _ in range(qt % 4):
            cells = [(-c, r, v) for r, c, v in cells]
    elif op == "flip_h":
        cells = [(-r, c, v) for r, c, v in cells]
    elif op == "flip_v":
        cells = [(r, -c, v) for r, c, v in cells]
    cells = [(r + dr, c + dc, v) for r, c, v in cells]
    return cells


def _run_steps(grid, steps, concrete):
    H, W = len(grid), len(grid[0])
    state = {"grid": [row[:] for row in grid], "objects": [],
             "selected": [], "canvas": None}
    for step_no, step in enumerate(steps):
        op = step["op"]
        raw = concrete[step_no]
        if op == "extract_objects":
            bg = max(range(10), key=lambda v: sum(cell == v for row in grid for cell in row))
            state["objects"] = _extract(grid, raw["same_color"],
                                        raw["connectivity"], bg)
            state["selected"] = list(range(len(state["objects"])))
        elif op == "filter_objects":
            crit, val = raw["criterion"], raw["value"]
            keep = []
            for i in state["selected"]:
                o = state["objects"][i]
                if crit == "area_gt" and o["area"] > val:
                    keep.append(i)
                elif crit == "area_lt" and o["area"] < val:
                    keep.append(i)
                elif crit == "color_eq" and o["colors"] == [val]:
                    keep.append(i)
                elif crit == "touches_border" and _touches_border(o, H, W):
                    keep.append(i)
            state["selected"] = keep
        elif op == "sort_objects":
            by, order = raw["by"], raw["order"]
            state["selected"].sort(key=lambda i: _key(state["objects"][i], by),
                                   reverse=(order == "desc"))
        elif op == "transform_objects":
            for i in state["selected"]:
                o = state["objects"][i]
                # rotate/flip about the object's own bbox origin, then shift
                r0, c0, _, _ = o["bbox"]
                local = [(r - r0, c - c0, v) for r, c, v in o["cells"]]
                moved = _transform_cells(local, raw["op"], raw.get("quarter_turns", 0),
                                         raw.get("dr", 0), raw.get("dc", 0))
                cells = [(r + r0, c + c0, v) for r, c, v in moved]
                state["objects"][i] = _make_object(cells)
        elif op == "place_objects":
            bg = raw["background"]
            if raw.get("canvas", "blank") == "blank":
                canvas = [[bg] * W for _ in range(H)]
            else:
                canvas = [row[:] for row in state["grid"]]
            for i in state["selected"]:
                for r, c, v in state["objects"][i]["cells"]:
                    if not (0 <= r < H and 0 <= c < W):
                        return {"ok": False, "error": "place out of bounds"}
                    if canvas[r][c] != bg and canvas[r][c] != v \
                            and raw.get("overlap", "error") == "error":
                        return {"ok": False, "error": "place overlap"}
                    canvas[r][c] = v
            state["canvas"] = canvas
        elif op == "output":
            if state["canvas"] is None:
                return {"ok": False, "error": "no canvas to output"}
        else:
            return {"ok": False, "error": f"bad op {op}"}
    if state["canvas"] is None:
        return {"ok": False, "error": "no canvas produced"}
    return {"ok": True, "grid": state["canvas"]}


def _concrete_options(norm_steps):
    per_step = []
    for step in norm_steps:
        opts = {}
        for k, v in step["params"].items():
            if isinstance(v, dict) and "choices" in v:
                opts[k] = v["choices"]
            else:
                opts[k] = [v]
        keys = sorted(opts.keys())
        per_step.append((keys, list(itertools.product(*(opts[k] for k in keys)))))
    return per_step


def search_template(plan, demos, expected_list, time_budget_s=SEARCH_TIME_BUDGET_S,
                    max_candidates=MAX_CANDIDATES):
    """Enumerate configs; a config must pass ALL demos identically."""
    t0 = time.monotonic()
    norm = validate_template(plan)["steps"]
    per_step = _concrete_options(norm)
    total = 1
    for _, combos in per_step:
        total *= len(combos)
    if total > max_candidates or total < 1:
        return {"status": "invalid_plan",
                "error": f"{total} candidates, limit {max_candidates}"}
    tried = 0
    best = 0
    for combo in itertools.product(*[c for _, c in per_step]):
        tried += 1
        if time.monotonic() - t0 > time_budget_s:
            return {"status": "timeout", "tried": tried, "best_demo_matches": best}
        concrete = [dict(zip(keys, vals)) for (keys, _), vals in zip(per_step, combo)]
        # fill defaults already in norm; missing optional keys impossible
        hits = 0
        ok_all = True
        for demo, expected in zip(demos, expected_list):
            if time.monotonic() - t0 > time_budget_s:
                return {"status": "timeout", "tried": tried, "best_demo_matches": best}
            r = _run_steps(demo, norm, concrete)
            if r.get("ok") and r["grid"] == expected:
                hits += 1
            else:
                ok_all = False
                break
        best = max(best, hits)
        if ok_all and hits == len(demos) and demos:
            out_grids = []
            for demo in demos:
                out_grids.append(_run_steps(demo, norm, concrete)["grid"])
            return {"status": "all_pass", "tried": tried, "config": concrete,
                    "demo_matches": hits, "elapsed_s": round(time.monotonic() - t0, 3),
                    "outputs": out_grids}
    return {"status": "no_pass", "tried": tried, "best_demo_matches": best,
            "elapsed_s": round(time.monotonic() - t0, 3)}

