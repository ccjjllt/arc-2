def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=lambda c: (counts[c], c))
    filler = max(counts, key=lambda c: (counts[c], -c)) if counts.get(bg, 0) > 0 else bg

    groups = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v in (0, bg, filler):
                continue
            if v not in groups:
                groups[v] = [[(r, c)]]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == v:
                            groups[v].append((rr, cc))
            else:
                found = False
                for i, (rr, cc) in enumerate(groups[v][-1]):
                    if abs(rr - r) <= 1 and abs(cc - c) <= 1:
                        groups[v][-1].append((r, c))
                        found = True
                        break
                if not found:
                    groups[v].append([(r, c)])

    specials, patterns = [], []
    for v, comps in groups.items():
        for comp in comps:
            rs = [p[0] for p in comp]
            cs = [p[1] for p in comp]
            if len(comp) >= 4 and (max(rs) - min(rs) + 1) * (max(cs) - min(cs) + 1) == len(comp) and (max(rs) - min(rs) <= 2 and max(cs) - min(cs) <= 2):
                specials.append((min(rs), min(cs), v))
            else:
                patterns.append((v, sorted(comp, key=lambda p: (p[0], p[1]))))

    patterns.sort(key=lambda x: (x[1][0][0], x[1][0][1], x[0]))
    seq = [v for v, _ in patterns]
    if not seq:
        seq = [1]
    if not specials:
        specials.append((H - 1, W - 1, seq[0]))

    out = [row[:] for row in grid]
    for r, c in range(H):
        pass
    for r in range(H):
        for c in range(W):
            out[r][c] = bg

    def can(r, c):
        return 0 <= r < H and 0 <= c < W and out[r][c] == bg

    def paint(r, c, v):
        if can(r, c):
            out[r][c] = v

    for i, (sr, sc, sv) in enumerate(specials):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = sr + dr, sc + dc
                if 0 <= rr < H and 0 <= cc < W and out[rr][cc] != bg:
                    out[rr][cc] = bg

    if len(specials) >= 2:
        a, b = specials[0], specials[1]
        order = [0, 1, 2, 3]
        def perimeter(pts):
            p = 0
            for i in range(len(pts)):
                x1, y1 = pts[i]
                x2, y2 = pts[(i + 1) % len(pts)]
                p += abs(x1 - x2) + abs(y1 - y2)
            return p
        best = order
        for o in ([0,1,3,2],[0,2,1,3],[0,3,2,1],[1,0,3,2],[1,3,0,2],[2,0,1,3],[2,1,0,3],[3,0,2,1],[3,2,0,1]):
            if len(specials) == 4 and perimeter([specials[o[0]], specials[o[1]], specials[o[2]], specials[o[3]]]) < perimeter([specials[best[0]], specials[best[1]], specials[best[2]], specials[best[3]]]):
                best = o
        route = [specials[best[0]]]
        if len(specials) == 4:
            route += [specials[best[1]], specials[best[2]], specials[best[3]]]
        else:
            route += [specials[best[1]]]
    else:
        route = specials[:]

    if len(route) >= 2:
        cells = []
        for i in range(len(route) - 1):
            r1, c1, _ = route[i]
            r2, c2, _ = route[i + 1]
            steps = max(abs(r2 - r1), abs(c2 - c1))
            dr = 0 if r2 == r1 else (1 if r2 > r1 else -1)
            dc = 0 if c2 == c1 else (1 if c2 > c1 else -1)
            for s in range(steps + 1):
                cell = (r1 + dr * s, c1 + dc * s)
                if not cells or cells[-1] != cell:
                    cells.append(cell)
        if cells:
            for idx, (r, c) in enumerate(cells):
                paint(r, c, seq[idx % len(seq)])
            for sr, sc, sv in route:
                paint(sr, sc, sv)
    else:
        sr, sc, sv = route[0]
        paint(sr, sc, sv)

    return out