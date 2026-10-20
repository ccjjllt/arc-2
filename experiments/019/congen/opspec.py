#!/usr/bin/env python3
"""019-congen opspec: 受约束操作模板 + 可靠对象原语（CPU），演进自018-B。

继承 018 (`experiments/018/bv/opspec.py`, SHA 753ceb0c…) 的对象模型：
真实成员格点 [(r,c,v)]，bbox 只派生不存内容；validate_template() 逐字拒绝
未知op/多余键/自由文本/超长/超候选；guided_schema() 同源生成（服务端尽力、
CPU 为准）；search_template 上限32候选/10秒CPU、同一配置必须全过所有demo。

019变更（对应019计划§6.2）：
1. guided_schema() 每个参数包装为 anyOf [标量, {"choices": [...]}]，
   与 validate_template()/search_template 支持的 {"choices": [...]} 协议对齐，
   模型在解码约束下可合法输出 choices 供 CPU 搜索。
2. 变尺寸输出、裁剪与对象排列/拼接原语：
   - crop_region：框内子区域提取为归零单对象；
   - normalize_objects：height>width 时逆时针旋转90°并局部坐标归零；
   - pack_objects：指定目标画布尺寸或按选中对象序列纵向/横向拼接
    （layout stack_v|stack_h|canvas，对齐align，背景background，间距spacing）；
   - place_objects：blank画布支持显式 height/width（0=auto沿用输入尺寸）。
3. 输出门控放宽为 place_objects 或 pack_objects 任一 + output（两者都产画布）。

针对 291dc1e1（多demo变尺寸：提取框内多色矩形块→按长边排序→竖向对象
逆时针旋转90°后居中纵向拼接）与 db118e2a（固定15x15铺放）设计。
本地真实题面干跑门控（新模板4/4通过291全部train示例）通过前不进GPU。
"""
from __future__ import annotations

import itertools
import time

MAX_STEPS = 12
MAX_CHOICES_PER_PARAM = 4
MAX_CANDIDATES = 32
SEARCH_TIME_BUDGET_S = 10.0

OP_NAMES = ("extract_objects", "crop_region", "filter_objects", "sort_objects",
            "transform_objects", "normalize_objects", "place_objects",
            "pack_objects", "output")

PARAM_SPECS = {
    "extract_objects": {
        "same_color": ("bool", None),
        "connectivity": ("enum", (4, 8)),
    },
    "crop_region": {
        "top": ("int", (0, 29)),
        "left": ("int", (0, 29)),
        "height": ("int", (1, 30)),
        "width": ("int", (1, 30)),
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
    "normalize_objects": {
        "rotate_tall_ccw": ("bool", None),
    },
    "place_objects": {
        "background": ("color", None),
        "canvas": ("enum", ("blank", "input")),
        "overlap": ("enum", ("error", "overwrite")),
        "height": ("int", (0, 30)),
        "width": ("int", (0, 30)),
    },
    "pack_objects": {
        "layout": ("enum", ("stack_v", "stack_h", "canvas")),
        "height": ("int", (0, 30)),
        "width": ("int", (0, 30)),
        "align": ("enum", ("center", "left", "right")),
        "background": ("color", None),
        "spacing": ("int", (0, 29)),
    },
    "output": {},
}

REQUIRED_PARAMS = {
    "extract_objects": ("same_color", "connectivity"),
    "crop_region": ("top", "left", "height", "width"),
    "transform_objects": ("op",),
    "normalize_objects": ("rotate_tall_ccw",),
    "place_objects": ("background",),
    "pack_objects": ("layout", "background"),
}

DEFAULT_PARAMS = {
    "place_objects": {"canvas": "blank", "overlap": "error", "height": 0, "width": 0},
    "transform_objects": {"quarter_turns": 0, "dr": 0, "dc": 0},
    "pack_objects": {"height": 0, "width": 0, "align": "center", "spacing": 0},
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
    if not any(s["op"] in ("place_objects", "pack_objects") for s in norm):
        raise ValueError("plan must contain place_objects or pack_objects")
    if not any(s["op"] == "output" for s in norm):
        raise ValueError("plan must contain output")
    return {"steps": norm}


def guided_schema() -> dict:
    """JSON-schema for vLLM structured_outputs, generated from the same constants.
    Each param accepts either the scalar or {"choices": [...]} (anyOf),
    aligned with validate_template()/search_template(). Server enforcement is
    best-effort; validate_template() is authoritative."""
    def scalar(name, kind, constraint):
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
    def p(name, kind, constraint):
        scalar_schema = scalar(name, kind, constraint)
        return {
            "anyOf": [
                scalar_schema,
                {
                    "type": "object",
                    "properties": {
                        "choices": {
                            "type": "array",
                            "items": scalar_schema,
                            "minItems": 1,
                            "maxItems": MAX_CHOICES_PER_PARAM,
                        }
                    },
                    "required": ["choices"],
                    "additionalProperties": False,
                },
            ]
        }
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


def _to_origin(cells):
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    r0, c0 = min(rs), min(cs)
    return [(r - r0, c - c0, v) for r, c, v in cells]


def _dims(cells):
    rs = [r for r, _, _ in cells]
    cs = [c for _, c, _ in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


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
        elif op == "crop_region":
            top, left, hh, ww = raw["top"], raw["left"], raw["height"], raw["width"]
            if not (0 <= top < H and 0 <= left < W and top + hh <= H and left + ww <= W):
                return {"ok": False, "error": "crop out of bounds"}
            cells = [(r - top, c - left, grid[r][c])
                     for r in range(top, top + hh) for c in range(left, left + ww)]
            state["objects"].append(_make_object(cells))
            state["selected"] = [len(state["objects"]) - 1]
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
        elif op == "normalize_objects":
            for i in state["selected"]:
                o = state["objects"][i]
                local = _to_origin(o["cells"])
                if raw["rotate_tall_ccw"]:
                    hh, ww = _dims(local)
                    if hh > ww:
                        # CCW 90° about origin: (r,c) -> (w-1-c, r); new dims (w,h)
                        local = [(ww - 1 - c, r, v) for r, c, v in local]
                        local = _to_origin(local)
                state["objects"][i] = _make_object(local)
        elif op == "place_objects":
            bg = raw["background"]
            if raw.get("canvas", "blank") == "blank":
                ch = raw.get("height", 0) or H
                cw = raw.get("width", 0) or W
                if not (1 <= ch <= 30 and 1 <= cw <= 30):
                    return {"ok": False, "error": "place canvas size invalid"}
                canvas = [[bg] * cw for _ in range(ch)]
                oH, oW = ch, cw
            else:
                if raw.get("height", 0) or raw.get("width", 0):
                    return {"ok": False, "error": "explicit size needs blank canvas"}
                canvas = [row[:] for row in state["grid"]]
                oH, oW = H, W
            for i in state["selected"]:
                for r, c, v in state["objects"][i]["cells"]:
                    if not (0 <= r < oH and 0 <= c < oW):
                        return {"ok": False, "error": "place out of bounds"}
                    if canvas[r][c] != bg and canvas[r][c] != v \
                            and raw.get("overlap", "error") == "error":
                        return {"ok": False, "error": "place overlap"}
                    canvas[r][c] = v
            state["canvas"] = canvas
        elif op == "pack_objects":
            bg = raw["background"]
            layout = raw["layout"]
            spacing = raw.get("spacing", 0)
            align = raw.get("align", "center")
            locals_list = [_to_origin(state["objects"][i]["cells"])
                           for i in state["selected"]]
            if not locals_list:
                return {"ok": False, "error": "pack with no selected objects"}
            if layout == "canvas":
                ch = raw.get("height", 0) or H
                cw = raw.get("width", 0) or W
                if not (1 <= ch <= 30 and 1 <= cw <= 30):
                    return {"ok": False, "error": "pack canvas size invalid"}
                canvas = [[bg] * cw for _ in range(ch)]
                for cells in locals_list:
                    for r, c, v in cells:
                        if not (0 <= r < ch and 0 <= c < cw):
                            return {"ok": False, "error": "pack out of bounds"}
                        canvas[r][c] = v
                state["canvas"] = canvas
            elif layout in ("stack_v", "stack_h"):
                dims = [_dims(cells) for cells in locals_list]
                if layout == "stack_v":
                    cw = raw.get("width", 0) or max(w for _, w in dims)
                    ch = (sum(h for h, _ in dims) + spacing * (len(dims) - 1))
                    ch = raw.get("height", 0) or ch
                    if not (1 <= ch <= 30 and 1 <= cw <= 30):
                        return {"ok": False, "error": "pack stack size invalid"}
                    canvas = [[bg] * cw for _ in range(ch)]
                    y = 0
                    for cells, (hh, ww) in zip(locals_list, dims):
                        if align == "center":
                            x0 = (cw - ww) // 2
                        elif align == "right":
                            x0 = cw - ww
                        else:
                            x0 = 0
                        for r, c, v in cells:
                            rr, cc = y + r, x0 + c
                            if not (0 <= rr < ch and 0 <= cc < cw):
                                return {"ok": False, "error": "pack out of bounds"}
                            canvas[rr][cc] = v
                        y += hh + spacing
                else:
                    ch = raw.get("height", 0) or max(h for h, _ in dims)
                    cw = (sum(w for _, w in dims) + spacing * (len(dims) - 1))
                    cw = raw.get("width", 0) or cw
                    if not (1 <= ch <= 30 and 1 <= cw <= 30):
                        return {"ok": False, "error": "pack stack size invalid"}
                    canvas = [[bg] * cw for _ in range(ch)]
                    x = 0
                    for cells, (hh, ww) in zip(locals_list, dims):
                        if align == "center":
                            y0 = (ch - hh) // 2
                        elif align == "right":
                            y0 = ch - hh
                        else:
                            y0 = 0
                        for r, c, v in cells:
                            rr, cc = y0 + r, x + c
                            if not (0 <= rr < ch and 0 <= cc < cw):
                                return {"ok": False, "error": "pack out of bounds"}
                            canvas[rr][cc] = v
                        x += ww + spacing
                state["canvas"] = canvas
            else:
                return {"ok": False, "error": f"bad layout {layout}"}
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
