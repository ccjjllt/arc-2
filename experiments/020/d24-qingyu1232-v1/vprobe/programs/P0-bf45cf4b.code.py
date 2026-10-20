def transform(grid):
    if not grid or not grid[0]:
        return []
    H = len(grid)
    W = len(grid[0])

    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = grid[0][0]
    best = -1
    for v, c in counts.items():
        if c > best:
            best = c
            bg = v

    color_cells = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                color_cells.setdefault(v, []).append((r, c))

    def bbox(cells):
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        return min_r, min_c, max_r, max_c

    boxes = {v: bbox(cells) for v, cells in color_cells.items()}

    def intersects(b1, b2):
        return not (b1[2] < b2[0] or b2[2] < b1[0] or
                    b1[3] < b2[1] or b2[3] < b1[1])

    colors = list(boxes.keys())
    parent = {v: v for v in colors}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(colors)):
        for j in range(i + 1, len(colors)):
            if intersects(boxes[colors[i]], boxes[colors[j]]):
                union(colors[i], colors[j])

    clusters = {}
    for v in colors:
        clusters.setdefault(find(v), []).append(v)

    cluster_list = list(clusters.values())
    if len(cluster_list) == 1:
        mask_colors = cluster_list[0]
        pattern_colors = cluster_list[0]
    else:
        def cluster_rect(cols):
            r0 = min(boxes[v][0] for v in cols)
            c0 = min(boxes[v][1] for v in cols)
            r1 = max(boxes[v][2] for v in cols)
            c1 = max(boxes[v][3] for v in cols)
            return r0, c0, r1, c1

        def cluster_score(cols):
            rect = cluster_rect(cols)
            rh = rect[2] - rect[0] + 1
            rw = rect[3] - rect[1] + 1
            area = rh * rw
            cell_count = sum(len(color_cells[v]) for v in cols)
            return len(cols), cell_count, area

        cluster_list.sort(key=cluster_score, reverse=True)
        if len(cluster_list) == 2:
            if len(cluster_list[0]) > len(cluster_list[1]):
                pattern_colors, mask_colors = cluster_list[0], cluster_list[1]
            elif len(cluster_list[0]) < len(cluster_list[1]):
                pattern_colors, mask_colors = cluster_list[1], cluster_list[0]
            else:
                p0, m0, p1, m1 = cluster_rect(cluster_list[0]), cluster_rect(cluster_list[1]), cluster_rect(cluster_list[0]), cluster_rect(cluster_list[1])
                if (p0[2] - p0[0] + 1) * (p0[3] - p0[1] + 1) >= (m0[2] - m0[0] + 1) * (m0[3] - m0[1] + 1):
                    pattern_colors, mask_colors = cluster_list[0], cluster_list[1]
                else:
                    pattern_colors, mask_colors = cluster_list[1], cluster_list[0]
        else:
            pattern_colors, mask_colors = cluster_list[0], cluster_list[1]

    def rect_for(cols):
        r0 = min(boxes[v][0] for v in cols)
        c0 = min(boxes[v][1] for v in cols)
        r1 = max(boxes[v][2] for v in cols)
        c1 = max(boxes[v][3] for v in cols)
        return r0, c0, r1, c1

    mask_rect = rect_for(mask_colors)
    pattern_rect = rect_for(pattern_colors)

    mask_set = set(mask_colors)
    pr0, pc0, pr1, pc1 = pattern_rect
    ph = pr1 - pr0 + 1
    pw = pc1 - pc0 + 1
    pattern = []
    for r in range(pr0, pr1 + 1):
        pattern.append([grid[r][c] for c in range(pc0, pc1 + 1)])

    mr0, mc0, mr1, mc1 = mask_rect
    out = []
    for r in range(mr0, mr1 + 1):
        for c in range(mc0, mc1 + 1):
            if grid[r][c] in mask_set:
                for prow in pattern:
                    out.append(list(prow))
            else:
                for _ in range(ph):
                    out.append([bg] * pw)

    return out