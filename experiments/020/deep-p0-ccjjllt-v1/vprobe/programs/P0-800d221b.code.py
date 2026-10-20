def transform(grid):
    h = len(grid)
    w = len(grid[0])
    colors = {}
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            colors[v] = colors.get(v, 0) + 1
    bg = max(colors, key=lambda x: (colors[x], -x))
    rest = [c for c in colors if c != bg]
    rest.sort(key=lambda x: (colors[x], -x))
    wall = rest[-1]
    sources = [c for c in rest if c != wall]
    if len(sources) > 2:
        sources = sorted(sources, key=lambda x: (colors[x], -x), reverse=True)[:2]

    out = [row[:] for row in grid]
    keep = [[False] * w for _ in range(h)]
    for r in range(h - 2):
        for c in range(w - 2):
            if all(grid[r + dr][c + dc] == wall for dr in range(3) for dc in range(3)):
                for dr in range(3):
                    for dc in range(3):
                        if dr == 1 and dc == 1:
                            continue
                        keep[r + dr][c + dc] = True

    def sources_near(r, c):
        res = []
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] in sources:
                    res.append(grid[rr][cc])
        return sorted(set(res), reverse=True)

    best = {}
    frontier = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] == wall and not keep[r][c]:
                ss = sources_near(r, c)
                if ss:
                    best[(r, c)] = ss[0]
                    frontier[(r, c)] = 0

    d = 0
    while frontier:
        d += 1
        new = {}
        for (r, c) in frontier:
            val = best[(r, c)]
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] == wall and not keep[rr][cc]:
                    cur = best.get((rr, cc))
                    if cur is None or (val, d) > (cur, d):
                        best[(rr, cc)] = val
                    elif cur == val:
                        pass
                    if (rr, cc) not in best:
                        best[(rr, cc)] = val
                    new[(rr, cc)] = True
        frontier = new

    for r in range(h):
        for c in range(w):
            if grid[r][c] == wall and (r, c) in best:
                out[r][c] = best[(r, c)]
    return out