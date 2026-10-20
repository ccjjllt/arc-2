def transform(grid):
    H = len(grid)
    W = len(grid[0])
    sep = 0
    for x in range(W):
        if all(grid[r][x] == 4 for r in range(H)):
            sep = x
            break
    left_w = sep
    out = [[0] * left_w for _ in range(H)]

    key = None
    for r in range(H):
        for c in range(sep + 1, W):
            if grid[r][c] == 5:
                key = (r, c - sep - 1)
                break
        if key is not None:
            break

    def slot_at(r0, c0):
        vals = set()
        pat = {}
        for dr in range(3):
            for dc in range(3):
                v = grid[r0 + dr][c0 + dc]
                if v != 0:
                    vals.add(v)
                    pat[(dr, dc)] = v
        if len(vals) == 1:
            return vals.pop(), pat
        return None, None

    def orient(p, t):
        keys = sorted(p.keys())
        if t == 1:
            m = {(-c, r): v for (r, c), v in p.items()}
        elif t == 2:
            m = {(2 - r, 2 - c): v for (r, c), v in p.items()}
        elif t == 3:
            m = {(c, 2 - r): v for (r, c), v in p.items()}
        else:
            m = dict(p)
        if not all(m.get(k, 0) == p[k] for k in keys):
            return None
        return m

    def find_anchor(m, p, kr, kc):
        best = None
        for (r, c) in p.keys():
            nr, nc = m.get((r, c), (-10, -10))
            if nr == kr and nc == kc:
                cand = (r0, c0) if (r0, c0) is not None else None
        return None

    anchors = []
    if key is not None:
        kr, kc = key
        for r0 in range(0, H - 2, 4):
            for c0 in [0, 3]:
                color, pat = slot_at(r0, c0)
                if color is None:
                    continue
                for t in range(4):
                    m = orient(pat, t)
                    if m is None:
                        continue
                    for (r, c) in pat.keys():
                        nr, nc = m[(r, c)]
                        if kr - r0 == nr and kc - c0 == nc:
                            anchors.append((color, r0, c0, t, m))
                            break

    for r, c in anchors:
        color, r0, c0, t, m = r
        for (dr, dc), v in m.items():
            rr, cc = r0 + dr, c0 + dc
            if 0 <= rr < H and 0 <= cc < left_w:
                out[rr][cc] = v

    if key is not None:
        out[key[0]][key[1]] = 5

    return out