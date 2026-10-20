def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [[0] * W for _ in range(H)]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    for r in range(H):
        for c in range(W):
            x = grid[r][c]
            if x != 8 and x != 0:
                if inb(r, c - 1) and grid[r][c - 1] == x:
                    for cc in range(c, W):
                        out[r][cc] = x
                        if cc > c and grid[r][cc] != 0:
                            break
                if inb(r + 1, c) and grid[r + 1][c] == x:
                    for rr in range(r, H):
                        out[rr][c] = x
                        if rr > r and grid[rr][c] != 0:
                            break

    for r in range(H):
        for c in range(W):
            x = grid[r][c]
            if x == 8:
                if inb(r - 1, c) and grid[r - 1][c] == 8:
                    rr = r - 1
                    while inb(rr - 1, c) and grid[rr - 1][c] == 8:
                        rr -= 1
                    if inb(rr + 1, c + 1) and grid[rr + 1][c + 1] == 0:
                        for dr, dc in ((0, 0), (0, 1), (1, 1), (1, 0)):
                            nr, nc = rr + dr, c + dc
                            if inb(nr, nc) and out[nr][nc] == 0:
                                out[nr][nc] = 2
    return out