def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    solids = {}
    pat = {}
    used = [[False] * W for _ in range(H)]
    max_h = max_w = 0

    for r in range(H):
        for c in range(W):
            if used[r][c] or grid[r][c] == bg:
                continue
            col = grid[r][c]
            r0, c0 = r, c
            while r0 > 0 and grid[r0 - 1][c] == col:
                r0 -= 1
            while c0 > 0 and grid[r0][c0 - 1] == col:
                c0 -= 1
            r1, c1 = r0, c0
            while r1 + 1 < H and grid[r1 + 1][c0] == col:
                r1 += 1
            while c1 + 1 < W and grid[r1][c1 + 1] == col:
                c1 += 1
            solid = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if grid[rr][cc] != col:
                        solid = False
                    used[rr][cc] = True
            if solid:
                key = (col, r0, c0)
                solids[key] = (r0, r1, c0, c1)
                max_h = max(max_h, r1 - r0 + 1)
                max_w = max(max_w, c1 - c0 + 1)

    for r in range(H):
        for c in range(W):
            if used[r][c]:
                continue
            col = grid[r][c]
            r0, c0 = r, c
            while r0 > 0 and grid[r0 - 1][c] == col:
                r0 -= 1
            while c0 > 0 and grid[r0][c0 - 1] == col:
                c0 -= 1
            r1, c1 = r0, c0
            while r1 + 1 < H and grid[r1 + 1][c0] == col:
                r1 += 1
            while c1 + 1 < W and grid[r1][c1 + 1] == col:
                c1 += 1
            if (r1 - r0 + 1, c1 - c0 + 1) != (max_h, max_w):
                p = []
                for rr in range(r0, r1 + 1):
                    p.append([1 if grid[rr][cc] == col else 0 for cc in range(c0, c1 + 1)])
                pat[col] = p

    rows = [x[0] for x in solids.values()]
    cols = [x[1] for x in solids.values()]
    out_r0, out_r1 = min(rows) - 1, max(x[1] for x in solids.values()) + 1
    out_c0, out_c1 = min(cols) - 1, max(x[3] for x in solids.values()) + 1
    out = [[bg] * (out_c1 - out_c0 + 1) for _ in range(out_r1 - out_r0 + 1)]
    for col, r0, r1, c0, c1 in solids.values():
        p = pat.get(col)
        if p is None:
            continue
        ph, pw = len(p), len(p[0])
        for i in range(max_h):
            for j in range(max_w):
                out[r0 - out_r0 + i][c0 - out_c0 + j] = (
                    bg if (1 + i < 1 + ph and 1 + j < 1 + pw and p[1 + i][1 + j]) else col
                )
    return out