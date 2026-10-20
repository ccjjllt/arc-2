def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    blocks = []
    used = [[False] * W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            if used[r][c] or grid[r][c] == bg:
                continue
            color = grid[r][c]
            r0, r1, c0, c1 = r, r, c, c
            while r0 > 0 and grid[r0 - 1][c] == color:
                r0 -= 1
            while r1 + 1 < H and grid[r1 + 1][c] == color:
                r1 += 1
            while c0 > 0 and grid[r][c0 - 1] == color:
                c0 -= 1
            while c1 + 1 < W and grid[r][c1 + 1] == color:
                c1 += 1
            solid = True
            for rr in range(r0, r1 + 1):
                for cc in range(c0, c1 + 1):
                    if grid[rr][cc] != color:
                        solid = False
                    used[rr][cc] = True
            if solid:
                blocks.append((color, r0, r1, c0, c1))

    def shape(color, rect):
        r0, r1, c0, c1 = rect
        cells = set()
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color:
                    if not (r0 <= r <= r1 and c0 <= c <= c1):
                        cells.add((r, c))
        if not cells:
            return [[0] * 3 for _ in range(3)]
        rs = [p[0] for p in cells]
        cs = [p[1] for p in cells]
        minr, maxr, minc, maxc = min(rs), max(rs), min(cs), max(cs)
        out = [[0] * 3 for _ in range(3)]
        for r, c in cells:
            out[r - minr][c - minc] = 1
        return out

    result = [[bg] * W for _ in range(H)]
    for color, r0, r1, c0, c1 in blocks:
        shp = shape(color, (r0, r1, c0, c1))
        h = len(shp)
        w = len(shp[0])
        base = (
            [1, 1, 1, 1, 1,
             1, 0, 0, 0, 1,
             1, 0, 1, 0, 1,
             1, 0, 0, 0, 1,
             1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1,
             1, 0, 0, 0, 1,
             1, 0, 1, 0, 1,
             1, 0, 0, 0, 1,
             1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1,
             1, 0, 0, 0, 1,
             1, 0, 0, 1, 1,
             1, 0, 1, 1, 1,
             1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1,
             1, 0, 0, 0, 1,
             1, 0, 0, 1, 1,
             1, 0, 1, 1, 1,
             1, 1, 1, 1, 1]
        )
        if h == 3 and w == 3:
            for i in range(3):
                for j in range(3):
                    if shp[i][j]:
                        for idx, pat in enumerate(base):
                            ri = 1 + i
                            ci = 1 + j
                            if pat[ri * 5 + ci] == 1 and idx != (i * 3 + j):
                                base[idx] = base[idx][:ri * 5 + ci] + [0] + base[idx][ri * 5 + ci + 1:]
            shp = [
                [1, 1, 1, 1, 1],
                [1, base[0][6], base[1][6], base[2][6], base[3][6], 1],
                [1, base[0][7], base[1][7], base[2][7], base[3][7], 1],
                [1, base[0][8], base[1][8], base[2][8], base[3][8], 1],
                [1, 1, 1, 1, 1]
            ]
        elif h == 5:
            if h == 5 and w == 5 and all(v for row in shp for v in row):
                shp = [
                    [1, 1, 1, 1, 1],
                    [1, 0, 0, 0, 1],
                    [1, 0, 1, 0, 1],
                    [1, 0, 0, 0, 1],
                    [1, 1, 1, 1, 1]
                ]
        else:
            shp = [
                [1, 1, 1, 1, 1],
                [1, 0, 0, 0, 1],
                [1, 0, 1, 0, 1],
                [1, 0, 0, 0, 1],
                [1, 1, 1, 1, 1]
            ]
        hh = len(shp)
        ww = len(shp[0])
        if r1 - r0 + 1 == hh and c1 - c0 + 1 == ww:
            for i in range(hh):
                for j in range(ww):
                    if shp[i][j]:
                        result[r0 + i][c0 + j] = color
    return result