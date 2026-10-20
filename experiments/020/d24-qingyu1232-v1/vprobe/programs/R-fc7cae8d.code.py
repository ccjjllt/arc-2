def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def row_has(r, val):
        return all(grid[r][c] == val for c in range(W) if grid[r][c] != 0)

    def col_has(c, val):
        return all(grid[r][c] == val for r in range(H) if grid[r][c] != 0)

    frame = None
    for v in range(1, 10):
        if row_has(0, v) or row_has(H - 1, v):
            frame = v
            break
    if frame is None:
        for v in range(1, 10):
            if col_has(0, v) or col_has(W - 1, v):
                frame = v
                break
    if frame is None:
        return grid

    best = None
    for v in range(1, 10):
        if v == frame:
            continue
        cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == v]
        if not cells:
            continue
        r0 = min(r for r, c in cells)
        r1 = max(r for r, c in cells)
        c0 = min(c for r, c in cells)
        c1 = max(c for r, c in cells)
        if r1 - r0 >= 2 and c1 - c0 >= 2:
            score = (r1 - r0) * (c1 - c0)
            if best is None or score > best[0]:
                best = (score, r0, r1, c0, c1)

    r0, r1, c0, c1 = best[1:]
    width = c1 - c0 + 1
    height = r1 - r0 + 1

    region = []
    for r in range(r0, r1 + 1):
        row = []
        for c in range(c0, c1 + 1):
            val = grid[r][c]
            row.append(0 if val == frame else val)
        region.append(row)

    out = [[0] * height for _ in range(width)]
    for r in range(height):
        for c in range(width):
            out[c][height - 1 - r] = region[r][c]
    return out