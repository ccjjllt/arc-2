def transform(grid):
    if not grid or not grid[0]:
        return []

    h, w = len(grid), len(grid[0])

    def count_color(g, color):
        return sum(1 for r in range(len(g)) for c in range(len(g[0])) if g[r][c] == color)

    # Background is the most common color; break ties by border color.
    border = []
    for r in range(h):
        border.append(grid[r][0])
        border.append(grid[r][w - 1])
    for c in range(w):
        border.append(grid[0][c])
        border.append(grid[h - 1][c])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(sorted(counts), key=lambda x: counts[x])
    if count_color(grid, bg) <= max(counts.values()) // 2:
        border_counts = {}
        for v in border:
            border_counts[v] = border_counts.get(v, 0) + 1
        if border_counts:
            bg = max(sorted(border_counts), key=lambda x: border_counts[x])

    nonbg = [(r, c, grid[r][c]) for r in range(h) for c in range(w) if grid[r][c] != bg]
    if not nonbg:
        return [row[:] for row in grid]

    colors = sorted({v for _, _, v in nonbg})

    def bbox_of_cells(cells):
        min_r = min(r for r, _, _ in cells)
        max_r = max(r for r, _, _ in cells)
        min_c = min(c for _, c, _ in cells)
        max_c = max(c for _, c, _ in cells)
        return min_r, max_r, min_c, max_c

    def bbox_intersects(a, b):
        return not (a[1] < b[0] or b[1] < a[0] or a[3] < b[2] or b[3] < a[2])

    by_color = {}
    for r, c, v in nonbg:
        by_color.setdefault(v, []).append((r, c, v))

    # Prefer the unique monochrome color whose bounding box contains no other color.
    bboxes = {c: bbox_of_cells(by_color[c]) for c in colors}
    candidates = []
    for c in colors:
        r1, r2, c1, c2 = bboxes[c]
        inner = set()
        for r in range(r1, r2 + 1):
            for cc in range(c1, c2 + 1):
                v = grid[r][cc]
                if v != bg:
                    inner.add(v)
        if len(inner) == 1:
            candidates.append(c)

    mask_color = None
    if len(candidates) == 1:
        mask_color = candidates[0]
    elif len(candidates) > 1:
        # If several single-color boxes exist, choose the candidate whose box is
        # separated from all other non-background cells (not touching a multi-color object).
        for c in candidates:
            r1, r2, c1, c2 = bboxes[c]
            separated = True
            for other in by_color:
                if other == c:
                    continue
                for r, cc, _ in by_color[other]:
                    if r1 <= r <= r2 and c1 <= cc <= c2:
                        separated = False
                        break
                    if r1 - 1 <= r <= r2 + 1 and c1 - 1 <= cc <= c2 + 1:
                        # Diagonal contact with another color is treated as same object.
                        if (r1 <= r <= r2 and c1 <= cc <= c2) or (r1 - 1 <= r <= r2 + 1 and c1 - 1 <= cc <= c2 + 1):
                            pass
                        separated = False
                        break
                if not separated:
                    break
            if separated:
                mask_color = c
                break
        if mask_color is None:
            # Fallback: the color with the smallest compactness? Use smaller occupied area.
            mask_color = min(candidates, key=lambda c: (bboxes[c][1] - bboxes[c][0] + 1) * (bboxes[c][3] - bboxes[c][2] + 1))

    if mask_color is None:
        # Fallback: use 8-connected components; pick the monochrome component as mask.
        seen = set()
        comps = []
        for start in nonbg:
            r0, c0, _ = start
            if (r0, c0) in seen:
                continue
            stack = [(r0, c0)]
            seen.add((r0, c0))
            comp = []
            while stack:
                r, c = stack.pop()
                comp.append((r, c, grid[r][c]))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in seen and grid[nr][nc] != bg:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
            comps.append(comp)
        if len(comps) == 2:
            mono = []
            for comp in comps:
                if len({v for _, _, v in comp}) == 1:
                    mono.append(comp[0][2])
            mask_color = mono[0] if mono else comps[0][0][2]
        else:
            mask_color = colors[0]

    mask_bbox = bboxes[mask_color]
    mask_h = mask_bbox[1] - mask_bbox[0] + 1
    mask_w = mask_bbox[3] - mask_bbox[2] + 1

    tile_cells = [t for t in nonbg if t[2] != mask_color]
    if not tile_cells:
        # Degenerate: if no second object, repeat the mask cell color.
        out = [[grid[mask_bbox[0] + r][mask_bbox[2] + c] for c in range(mask_w)] * 1
               for r in range(mask_h)]
        return [[grid[r][c] for c in range(w)] for r in range(h)]

    # The tile is the object made of all non-background cells except the mask color.
    # To avoid accidentally including another disconnected monochrome object, keep only
    # the largest cluster by bounding-box union if multiple remain.
    def group_by_bbox(cells):
        groups = []
        for r, c, v in cells:
            box = (r, r, c, c)
            placed = False
            for i, grp in enumerate(groups):
                if any(bbox_intersects(box, b) for b in grp):
                    r1 = min(gr[0] for gr in grp)
                    r2 = max(gr[1] for gr in grp)
                    c1 = min(gr[2] for gr in grp)
                    c2 = max(gr[3] for gr in grp)
                    groups[i].append(box)
                    placed = True
                    break
            if not placed:
                groups.append([box])
        return groups

    groups = group_by_bbox(tile_cells)
    if len(groups) > 1:
        # Choose the group containing the most cells (the true tile).
        best = None
        best_size = -1
        for grp in groups:
            r1 = min(b[0] for b in grp)
            r2 = max(b[1] for b in grp)
            c1 = min(b[2] for b in grp)
            c2 = max(b[3] for b in grp)
            size = 0
            for r, c, v in tile_cells:
                if r1 <= r <= r2 and c1 <= c <= c2:
                    size += 1
            if size > best_size:
                best_size = size
                best = (r1, r2, c1, c2)
        tile_bbox = best
    else:
        tile_bbox = bbox_of_cells(tile_cells)

    tile_h = tile_bbox[1] - tile_bbox[0] + 1
    tile_w = tile_bbox[3] - tile_bbox[2] + 1

    def get_tile_value(r, c):
        r = tile_bbox[0] + r
        c = tile_bbox[2] + c
        if 0 <= r < h and 0 <= c < w and grid[r][c] != bg:
            return grid[r][c]
        return bg

    out_h = mask_h * tile_h
    out_w = mask_w * tile_w
    out = [[bg] * out_w for _ in range(out_h)]

    for mr in range(mask_h):
        for mc in range(mask_w):
            if grid[mask_bbox[0] + mr][mask_bbox[2] + mc] == mask_color:
                base_r = mr * tile_h
                base_c = mc * tile_w
                for tr in range(tile_h):
                    for tc in range(tile_w):
                        out[base_r + tr][base_c + tc] = get_tile_value(tr, tc)
            else:
                base_r = mr * tile_h
                base_c = mc * tile_w
                for tr in range(tile_h):
                    for tc in range(tile_w):
                        out[base_r + tr][base_c + tc] = bg

    return out