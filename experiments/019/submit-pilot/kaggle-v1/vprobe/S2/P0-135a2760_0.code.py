def transform(grid):
    H = len(grid)
    W = len(grid[0])
    g = [row[:] for row in grid]

    def is_uniform(v):
        first = v[0][0]
        return all(x == first for row in v for x in row)

    def is_rect(v, frame):
        top = all(x == frame for x in v[0])
        bottom = all(x == frame for x in v[-1])
        left = all(v[r][0] == frame for r in range(len(v)))
        right = all(v[r][-1] == frame for r in range(len(v)))
        return top and bottom and left and right

    best = None
    for h in range(2, H + 1):
        for w in range(2, W + 1):
            for r0 in range(H - h + 1):
                for c0 in range(W - w + 1):
                    sub = g[r0:r0 + h][c0:c0 + w]
                    if not is_rect(sub, g[r0][c0]):
                        continue
                    inner = [row[1:-1] for row in sub[1:-1]]
                    if not inner or not inner[0]:
                        continue
                    vals = sorted({x for row in inner for x in row})
                    if len(vals) != 2:
                        continue
                    if not all(not is_uniform(row) for row in inner):
                        continue
                    key = (h, w)
                    if best is None or key < best[0]:
                        best = (key, r0, c0, sub[0][0])

    if best is None:
        return g

    _, r0, c0, frame = best
    h, w = best[0]
    inner = [row[1:-1] for row in g[r0:r0 + h][c0:c0 + w][1:-1]]
    ih, iw = len(inner), len(inner[0])
    vals = sorted({x for row in inner for x in row})
    a, b = vals

    def fill(pattern, start):
        for i in range(ih):
            for j in range(iw):
                val = pattern[(start + i) % ih][(start + j) % iw]
                g[r0 + 1 + i][c0 + 1 + j] = val

    for sh in range(ih):
        for sw in range(iw):
            pat = [row[sh:sh + ih] for row in inner]
            for sa in range(iw):
                for sb in range(iw):
                    p2 = [[row[sa:sa + iw] for row in pat]] * ih
                    ok = True
                    for i in range(ih):
                        for j in range(iw):
                            if inner[i][j] not in (p2[i][j], p2[(i + sh) % ih][(j + sw) % iw]):
                                ok = False
                                break
                        if not ok:
                            break
                    if ok:
                        fill(p2, (0, 0))
                        return g

    # Fallback for larger non-pure cases: replace cells inconsistent with majority local 2x3 pattern.
    for i in range(ih):
        for j in range(iw):
            vals = []
            for di in (-1, 0, 1):
                ii = (i + di) % ih
                for dj in (-1, 0, 1):
                    jj = (j + dj) % iw
                    vals.append(inner[ii][jj])
            cnt = {a: 0, b: 0}
            for x in vals:
                cnt[x] += 1
            g[r0 + 1 + i][c0 + 1 + j] = a if cnt[a] >= cnt[b] else b
    return g