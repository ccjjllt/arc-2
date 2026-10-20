def transform(grid):
    if not grid or not grid[0]:
        return grid
    h = len(grid)
    w = len(grid[0])

    def runs(row, target):
        r = []
        c = 0
        while c < w:
            if row[c] == target:
                d = c
                while d < w and row[d] == target:
                    d += 1
                r.append((c, d - 1))
                c = d
            else:
                c += 1
        return r

    bg = grid[0][0]
    legend = None
    for row in grid:
        nonbg = [x for x in row if x != bg]
        if not nonbg:
            continue
        ok = True
        for a, b in zip(row, row[1:]):
            if a != bg and b != bg:
                ok = False
                break
        if ok and len(nonbg) >= 2:
            legend = nonbg
    if legend is None:
        return [r[:] for r in grid]

    candidates = set()
    for x in grid:
        for v in x:
            if v != bg:
                candidates.add(v)
    for v in legend:
        candidates.discard(v)
    if not candidates:
        candidates = set(c for row in grid for c in row if c != bg)
    containers = []
    for v in candidates:
        rs = []
        for r in range(h):
            if any(grid[r][c] == v for c in range(w)):
                rs.append(r)
        if not rs:
            continue
        rmin, rmax = rs[0], rs[-1]
        if rmax - rmin + 1 <= 2:
            continue
        col_runs = [runs(row, v) for row in grid[rmin:rmax + 1]]
        if any(not cr for cr in col_runs):
            continue
        cmin = min(cr[0][0] for cr in col_runs)
        cmax = max(cr[0][1] for cr in col_runs)
        if cmax - cmin + 1 <= 2:
            continue
        if any(grid[r][c] == v for r in range(rmin, rmax + 1) for c in range(cmin, cmax + 1)):
            containers.append((rmin, rmax, cmin, cmax, v))

    patches = []
    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == bg or v in containers:
                continue
            if v in [cv for cv in containers]:
                continue
            if any(grid[r][c2] == v for c2 in (c - 1, c + 1) if 0 <= c2 < w):
                continue
            if any(grid[r2][c] == v for r2 in (r - 1, r + 1) if 0 <= r2 < h):
                continue
            patches.append((r, c, v))

    out = [row[:] for row in grid]

    def in_patch(r, c):
        for pr, pc, pv in patches:
            if pr == r and pc == c:
                return pv
        return None

    def paint(a, b):
        r1, r2, c1, c2 = a[:4]
        s1, s2, t1, t2 = b[:4]
        pa, pb = a[5], b[5]
        if c2 < t1:
            lo, hi = max(1, c2 + 1), min(t1 - 1, w - 2)
            if lo <= hi:
                for rr in range(s1, s2 + 1):
                    for cc in range(lo, hi + 1):
                        if bg in (out[rr][cc], grid[rr][cc]) and in_patch(rr, cc) is None:
                            out[rr][cc] = pa
        elif t2 < c1:
            lo, hi = max(1, t2 + 1), min(c1 - 1, w - 2)
            if lo <= hi:
                for rr in range(s1, s2 + 1):
                    for cc in range(lo, hi + 1):
                        if bg in (out[rr][cc], grid[rr][cc]) and in_patch(rr, cc) is None:
                            out[rr][cc] = pa
        elif r2 < t1:
            lo, hi = max(1, r2 + 1), min(t1 - 1, h - 2)
            if lo <= hi:
                for rr in range(lo, hi + 1):
                    for cc in range(t1, t2 + 1):
                        if bg in (out[rr][cc], grid[rr][cc]) and in_patch(rr, cc) is None:
                            out[rr][cc] = pa
        elif t2 < r1:
            lo, hi = max(1, t2 + 1), min(r1 - 1, h - 2)
            if lo <= hi:
                for rr in range(lo, hi + 1):
                    for cc in range(t1, t2 + 1):
                        if bg in (out[rr][cc], grid[rr][cc]) and in_patch(rr, cc) is None:
                            out[rr][cc] = pa

    used = [False] * len(patches)
    for i in range(len(legend) - 1):
        va, vb = legend[i], legend[i + 1]
        best = None
        for a in range(len(patches)):
            if used[a] or patches[a][2] != va:
                continue
            for b in range(len(patches)):
                if used[b] or b == a or patches[b][2] != vb:
                    continue
                if (patches[a][3] >= patches[b][1] and patches[b][3] >= patches[a][1] and
                    patches[a][1] >= patches[b][3] and patches[b][1] >= patches[a][3]):
                    continue
                if not (patches[a][1] == patches[b][1] or patches[a][2] == patches[b][2] or
                        patches[a][3] == patches[b][3] or patches[a][4] == patches[b][4]):
                    continue
                gap = 0
                if patches[a][3] < patches[b][1]:
                    gap = patches[b][1] - patches[a][3] - 1
                elif patches[b][3] < patches[a][1]:
                    gap = patches[a][1] - patches[b][3] - 1
                elif patches[a][4] < patches[b][3]:
                    gap = patches[b][3] - patches[a][4] - 1
                elif patches[b][4] < patches[a][1]:
                    gap = patches[a][1] - patches[b][4] - 1
                if gap < 1:
                    continue
                inter = (max(patches[a][0], patches[b][0]), min(patches[a][1], patches[b][1]),
                         max(patches[a][2], patches[b][2]), min(patches[a][3], patches[b][3]))
                if inter[0] <= inter[1] and inter[2] <= inter[3]:
                    continue
                score = (gap, abs(patches[a][0] - patches[b][0]) + abs(patches[a][2] - patches[b][2]), i)
                if best is None or score < best[0]:
                    best = (score, a, b)
        if best:
            paint(patches[best[1]], patches[best[2]])
            used[best[1]] = True
            used[best[2]] = True
    return out