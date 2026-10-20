def transform(grid):
    h = len(grid)
    w = len(grid[0])
    colors = set()
    for row in grid:
        for v in row:
            if v != 0:
                colors.add(v)
    if len(colors) != 2:
        return [row[:] for row in grid]
    a, b = sorted(colors)
    other = {a: b, b: a}
    out = [row[:] for row in grid]
    mask = [[False] * w for _ in range(h)]

    for r in range(h):
        for c in range(w):
            v = grid[r][c]
            if v == 0 or v == 8:
                continue
            col = v
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        nv = grid[rr][cc]
                        if nv != col and nv != 0:
                            ok = False
                            break
                if not ok:
                    break
            if ok:
                mask[r][c] = True

    for r in range(h):
        for c in range(w):
            if mask[r][c]:
                col = grid[r][c]
                rep = other[col]
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr = r + dr
                        cc = c + dc
                        if 0 <= rr < h and 0 <= cc < w:
                            if out[rr][cc] != 8:
                                out[rr][cc] = rep
                out[r][c] = 8

    for r in range(h):
        for c in range(w):
            if out[r][c] == 8:
                continue
            vals = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < h and 0 <= cc < w:
                        if out[rr][cc] != 8:
                            vals.add(out[rr][cc])
            if len(vals) == 1:
                col = vals.pop()
                if col in other and other[col] != col:
                    out[r][c] = other[col]

    return out