def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = min(counts, key=lambda x: (counts[x], x))

    seen = [[False] * W for _ in range(H)]
    boxes = []
    keys = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and grid[ny][nx] != bg:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(cells) == 1:
                keys.append((r, c, grid[r][c]))
                continue
            ys = [p[0] for p in cells]
            xs = [p[1] for p in cells]
            r0, r1 = min(ys), max(ys)
            c0, c1 = min(xs), max(xs)
            interior = []
            for y in range(r0 + 1, r1):
                for x in range(c0 + 1, c1):
                    v = grid[y][x]
                    if v != bg:
                        interior.append((y, x, v))
            if interior:
                col_counts = {}
                for _, _, v in interior:
                    col_counts[v] = col_counts.get(v, 0) + 1
                item = max(col_counts, key=lambda v: (col_counts[v], v))
                cells2 = [(y, x) for y, x, v in interior if v == item]
                boxes.append({"color": item, "cells": cells2,
                              "r0": r0, "r1": r1, "c0": c0, "c1": c1})

    if keys:
        kr = min(k[0] for k in keys)
        key_vals = [v for r, c, v in sorted(keys, key=lambda p: (p[0], p[1])) if r == kr]
    else:
        key_vals = []

    out = [row[:] for row in grid]

    def box_for_color(color, boxes, prev_box=None, next_box=None):
        cands = [b for b in boxes if b["color"] == color]
        if len(cands) == 1:
            return cands[0]
        best, best_dist = None, None
        for b in cands:
            dist = 0
            if prev_box is not None:
                if b["r0"] == prev_box["r1"] and not (b["c1"] < prev_box["c0"] or b["c0"] > prev_box["c1"]):
                    pass
                elif b["c0"] == prev_box["c1"] and not (b["r1"] < prev_box["r0"] or b["r0"] > prev_box["r1"]):
                    pass
                else:
                    dr = max(prev_box["r0"] - b["r1"], b["r0"] - prev_box["r1"], 0)
                    dc = max(prev_box["c0"] - b["c1"], b["c0"] - prev_box["c1"], 0)
                    if dr == 0:
                        dist += dc
                    elif dc == 0:
                        dist += dr
                    else:
                        dist += dr + dc + 1000
            if next_box is not None:
                if b["r1"] == next_box["r0"] and not (b["c1"] < next_box["c0"] or b["c0"] > next_box["c1"]):
                    pass
                elif b["c1"] == next_box["c0"] and not (b["r1"] < next_box["r0"] or b["r0"] > next_box["r1"]):
                    pass
                else:
                    dr = max(next_box["r0"] - b["r1"], b["r0"] - next_box["r1"], 0)
                    dc = max(next_box["c0"] - b["c1"], b["c0"] - next_box["c1"], 0)
                    if dr == 0:
                        dist += dc
                    elif dc == 0:
                        dist += dr
                    else:
                        dist += dr + dc + 1000
            if best_dist is None or dist < best_dist:
                best, best_dist = b, dist
        return best

    for i in range(len(key_vals) - 1):
        a, b = key_vals[i], key_vals[i + 1]
        src = box_for_color(a, boxes)
        tgt = box_for_color(b, boxes, src)
        if src is None or tgt is None:
            continue
        src_cells = src["cells"]
        tgt_cells = tgt["cells"]
        sy = sum(p[0] for p in src_cells) // len(src_cells)
        sx = sum(p[1] for p in src_cells) // len(src_cells)
        ty = sum(p[0] for p in tgt_cells) // len(tgt_cells)
        tx = sum(p[1] for p in tgt_cells) // len(tgt_cells)
        if sy == ty:
            lo, hi = sorted((sx, tx))
            for x in range(lo + 1, hi):
                if out[sy][x] == bg:
                    out[sy][x] = src["color"]
        elif sx == tx:
            lo, hi = sorted((sy, ty))
            for y in range(lo + 1, hi):
                if out[y][sx] == bg:
                    out[y][sx] = src["color"]
        else:
            if src["r0"] < tgt["r0"]:
                if src["c0"] == tgt["c1"] and not (src["r1"] < tgt["r0"] or src["r0"] > tgt["r1"]):
                    x = src["c1"]
                    lo, hi = sorted((src["r0"], tgt["r1"]))
                    for y in range(lo + 1, hi):
                        if out[y][x] == bg:
                            out[y][x] = src["color"]
            elif tgt["r0"] < src["r0"]:
                if tgt["c0"] == src["c1"] and not (tgt["r1"] < src["r0"] or tgt["r0"] > src["r1"]):
                    x = tgt["c1"]
                    lo, hi = sorted((tgt["r0"], src["r1"]))
                    for y in range(lo + 1, hi):
                        if out[y][x] == bg:
                            out[y][x] = src["color"]
            elif src["c0"] < tgt["c0"]:
                if src["r0"] == tgt["c1"] and not (src["c1"] < tgt["c0"] or src["c0"] > tgt["c1"]):
                    y = src["r1"]
                    lo, hi = sorted((src["c0"], tgt["c1"]))
                    for x in range(lo + 1, hi):
                        if out[y][x] == bg:
                            out[y][x] = src["color"]
            elif tgt["c0"] < src["c0"]:
                if tgt["r0"] == src["c1"] and not (tgt["c1"] < src["c0"] or tgt["c0"] > src["c1"]):
                    y = tgt["r1"]
                    lo, hi = sorted((tgt["c0"], src["c1"]))
                    for x in range(lo + 1, hi):
                        if out[y][x] == bg:
                            out[y][x] = src["color"]
    return out