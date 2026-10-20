def transform(grid):
    H = len(grid)
    W = len(grid[0])

    best = (9, 9, 9, 9)
    for c in range(W):
        vals = set(grid[r][c] for r in range(H))
        if len(vals) == 1:
            best = min(best, (c, W - 1 - c, -1, -1))
    for r in range(H):
        vals = set(grid[r][c] for c in range(W))
        if len(vals) == 1:
            best = min(best, (-1, -1, r, H - 1 - r))

    if best[0] >= 0:
        c = best[0]
        col = [grid[r][c] for r in range(H)]
        out = [[col[x] if (y - x) % 2 == 0 else 1 for y in range(W)] for x in range(H)]
    elif best[2] >= 0:
        r = best[2]
        row = grid[r][:]
        out = [[row[x] if (y - x) % 2 == 0 else 1 for y in range(W)] for x in range(H)]
    else:
        out = [[0] * W for _ in range(H)]

    if len(out) != W or any(len(r) != H for r in out):
        out = [[0] * H for _ in range(W)]
    return out