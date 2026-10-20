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
    used = set()
    by_dir = {}
    for dr, dc in dirs:
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
                        rr += dr
                        cc += dc
                    if len(cells) >= 2:
                        used.update(cells)
                        by_dir.setdefault((dr, dc), []).append((color, cells))
    for r, c in nonbg:
        if (r, c) not in used:
            by_dir.setdefault((0, 0), []).append((grid[r][c], [(r, c)]))

    for d, segs in by_dir.items():
        if len(segs) < 2:
            continue
        if d == (0, 1):
            def offset(cells): return cells[0][0]
            def end(cells): return cells[0]
            def point_at(end, length, i):
                e0, e1 = end
                return (e0, e1 + i)
            def key(cells): return (cells[0][0], cells[0][1])
        elif d == (1, 0):
            def offset(cells): return cells[0][1]
            def end(cells): return cells[0]
            def point_at(end, length, i):
                e0, e1 = end
                return (e0 + i, e1)
            def key(cells): return (cells[0][1], cells[0][0])
        elif d == (1, 1):
            def offset(cells): return cells[0][1] - cells[0][0]
            def end(cells): return cells[0]
            def point_at(end, length, i):
                e0, e1 = end
                return (e0 + i, e1 + i)
            def key(cells): return offset(cells)
        else:
            def offset(cells): return cells[0][0] + cells[0][1]
            def end(cells): return cells[-1]
            def point_at(end, length, i):
                e0, e1 = end
                return (e0 - i, e1 + i)
            def key(cells): return offset(cells)

        infos = []
        for color, cells in segs:
            infos.append((key(cells), offset(cells), end(cells), len(cells)))
        infos.sort()
        lens = [x[3] for x in infos]
        target = sorted(lens)[len(lens) // 2]
        new_cells = []
        for _, _, end, _ in infos:
            for i in range(target):
                p = point_at(end, target, i)
                if inside(p[0], p[1]):
                    new_cells.append(p)
        for color, cells in segs:
            for r, c in cells:
                if grid[r][c] != bg:
                    grid[r][c] = bg
        for r, c in new_cells:
            grid[r][c] = grid[infos[0][0]]
    return grid