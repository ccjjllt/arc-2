def transform(grid):
    if not grid or not grid[0]:
        return []
    H, W = len(grid), len(grid[0])

    total = {}
    border = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            total[v] = total.get(v, 0) + 1
            if r == 0 or r == H - 1 or c == 0 or c == W - 1:
                border[v] = border.get(v, 0) + 1

    bg = max(sorted(total), key=lambda v: (total[v], border.get(v, 0), -v))

    # 8-neighbor components of non-background cells.
    seen = [[False] * W for _ in range(H)]
    comps = []
    for sr in range(H):
        for sc in range(W):
            if seen[sr][sc] or grid[sr][sc] == bg:
                continue
            q = [(sr, sc)]
            seen[sr][sc] = True
            cells = []
            colors = set()
            while q:
                r, c = q.pop()
                cells.append((r, c))
                colors.add(grid[r][c])
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and grid[nr][nc] != bg:
                            seen[nr][nc] = True
                            q.append((nr, nc))
            comps.append({"cells": cells, "colors": colors})

    def bbox_of(cells):
        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        return min(rs), min(cs), max(rs), max(cs)

    # Merge components that share a non-background color, and nearby components.
    n = len(comps)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    color_comps = {}
    for i, comp in enumerate(comps):
        for v in comp["colors"]:
            color_comps.setdefault(v, []).append(i)
    for lst in color_comps.values():
        for i in range(1, len(lst)):
            union(lst[0], lst[i])

    bboxes = [bbox_of(comp["cells"]) for comp in comps]
    for i in range(n):
        for j in range(i + 1, n):
            a, b = bboxes[i], bboxes[j]
            gap_r = max(0, max(a[0], b[0]) - min(a[2], b[2]) - 1)
            gap_c = max(0, max(a[1], b[1]) - min(a[3], b[3]) - 1)
            if gap_r <= 1 and gap_c <= 1:
                union(i, j)

    clusters = {}
    for i, comp in enumerate(comps):
        clusters.setdefault(find(i), []).append(i)

    objects = []
    for members in clusters.values():
        cells = []
        colors = set()
        for i in members:
            cells.extend(comps[i]["cells"])
            colors.update(comps[i]["colors"])
        b = bbox_of(cells)
        area = (b[2] - b[0] + 1) * (b[3] - b[1] + 1)
        fill = len(cells) / area if area else 0
        objects.append({"cells": cells, "colors": colors, "bbox": b, "area": area, "fill": fill})

    if not objects:
        return [[bg] * W for _ in range(H)]

    if len(objects) == 1:
        # Fallback: treat the single object as both pattern and a full mask.
        b = objects[0]["bbox"]
        pat_h = b[2] - b[0] + 1
        pat_w = b[3] - b[1] + 1
        pattern = [grid[r][c] for r in range(b[0], b[2] + 1) for c in [0] * 0]  # placeholder not used
        pattern = [list(grid[r][b[1]:b[3] + 1]) for r in range(b[0], b[2] + 1)]
        out = []
        for _r in range(pat_h):
            row = []
            for _c in range(pat_w):
                row.extend(pattern[_r])
            out.append(row)
        return out

    # Choose mask: prefer single-color object, then lower fill, then smaller area.
    def is_mask_key(o):
        return (0 if len(o["colors"]) == 1 else 1, -o["fill"], -o["area"], len(o["cells"]))

    objects.sort(key=is_mask_key)
    mask = objects[0]

    # Pattern: merge all non-mask objects, unless there is exactly one obvious multi-color object.
    if len(objects) == 2 and len(objects[1]["colors"]) > 1:
        pattern_obj = objects[1]
    else:
        pattern_cells = []
        pattern_colors = set()
        for o in objects:
            if o is mask:
                continue
            pattern_cells.extend(o["cells"])
            pattern_colors.update(o["colors"])
        if not pattern_cells:
            pattern_obj = mask
        else:
            b = bbox_of(pattern_cells)
            pattern_obj = {"cells": pattern_cells, "colors": pattern_colors,
                           "bbox": b, "area": (b[2]-b[0]+1)*(b[3]-b[1]+1),
                           "fill": len(pattern_cells) / ((b[2]-b[0]+1)*(b[3]-b[1]+1))}

    pb = pattern_obj["bbox"]
    mb = mask["bbox"]
    pat_h = pb[2] - pb[0] + 1
    pat_w = pb[3] - pb[1] + 1
    mask_h = mb[2] - mb[0] + 1
    mask_w = mb[3] - mb[1] + 1

    pattern = [list(grid[r][pb[1]:pb[3] + 1]) for r in range(pb[0], pb[2] + 1)]
    out = []
    for mr in range(mb[0], mb[2] + 1):
        for pr in range(pat_h):
            row = []
            for mc in range(mb[1], mb[3] + 1):
                if grid[mr][mc] != bg:
                    row.extend(pattern[pr])
                else:
                    row.extend([bg] * pat_w)
            out.append(row)

    return out