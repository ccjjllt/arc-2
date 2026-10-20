def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=lambda x: (counts[x], x))

    nonbg = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                nonbg.add((r, c))

    def inside(r, c):
        return 0 <= r < H and 0 <= c < W

    dirs = [(0, 1), (1, 0), (1, 1), (1, -1)]
    segments = []
    used = set()

    for d in dirs:
        dr, dc = d
        for r in range(H):
            for c in range(W):
                if (r, c) in used or (r, c) not in nonbg:
                    continue
                if not inside(r - dr, c - dc) or (r - dr, c - dc) in used or grid[r - dr][c - dc] == bg:
                    color = grid[r][c]
                    cells = []
                    rr, cc = r, c
                    while inside(rr, cc) and (rr, cc) not in used and grid[rr][cc] == color:
                        cells.append((rr, cc))
                        used.add((rr, cc))
                        rr += dr
                        cc += dc
                    if len(cells) >= 2:
                        segments.append((d, color, cells))
    for (r, c) in nonbg:
        if (r, c) not in used:
            segments.append(((0, 0), grid[r][c], [(r, c)]))

    by_dir = {}
    for seg in segments:
        d, color, cells = seg
        by_dir.setdefault(d, []).append((color, cells))

    for d, segs in by_dir.items():
        if len(segs) < 2:
            continue
        if d == (0, 1):
            def offset(cells):
                return cells[0][0]
            def start(cells):
                return cells[0]
            def point_at(start, length, i):
                s0, s1 = start
                return (s0, s1 + i)
            def sort_key(cells):
                return (cells[0][0], cells[0][1])
        elif d == (1, 0):
            def offset(cells):
                return cells[0][1]
            def start(cells):
                return cells[0]
            def point_at(start, length, i):
                s0, s1 = start
                return (s0 + i, s1)
            def sort_key(cells):
                return (cells[0][1], cells[0][0])
        elif d == (1, 1):
            def offset(cells):
                return cells[0][1] - cells[0][0]
            def start(cells):
                return cells[0]
            def point_at(start, length, i):
                s0, s1 = start
                return (s0 + i, s1 + i)
            def sort_key(cells):
                return offset(cells)
        else:
            def offset(cells):
                return cells[0][0] + cells[0][1]
            def start(cells):
                return cells[-1]
            def point_at(start, length, i):
                s0, s1 = start
                return (s0 - i, s1 + i)
            def sort_key(cells):
                return offset(cells)

        infos = []
        for color, cells in segs:
            st = start(cells)
            infos.append((sort_key(cells), st, len(cells)))
        infos.sort()
        if not infos:
            continue
        med = infos[len(infos) // 2]
        target_len = med[2]

        new_cells = []
        for color, cells in segs:
            st = start(cells)
            for i in range(target_len):
                p = point_at(st, target_len, i)
                if inside(p[0], p[1]):
                    new_cells.append((p, color))

        for (color, cells) in segs:
            for (r, c) in cells:
                if grid[r][c] != bg:
                    grid[r][c] = bg
        for (r, c), color in new_cells:
            if 0 <= r < H and 0 <= c < W:
                grid[r][c] = color

    return grid