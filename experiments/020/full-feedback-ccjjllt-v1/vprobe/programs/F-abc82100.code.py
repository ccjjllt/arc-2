def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [[0] * W for _ in range(H)]

    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0:
                continue
            dr = 0
            dc = 0
            if r + 1 < H and grid[r + 1][c] == grid[r][c]:
                dr = 1
            if c + 1 < W and grid[r][c + 1] == grid[r][c]:
                dc = 1
            if dr == dc:
                continue
            if dr and not dc:
                if r > 0 and grid[r - 1][c] == 0:
                    nr, nc = r - 1, c + 1
                    while 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                        out[nr][nc] = 2
                        nr -= 1
                        nc += 1
                if r + 1 < H and grid[r + 1][c] == 0:
                    nr, nc = r + 1, c - 1
                    while 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                        out[nr][nc] = 2
                        nr += 1
                        nc -= 1
            else:
                if c > 0 and grid[r][c - 1] == 0:
                    nr, nc = r - 1, c - 1
                    while 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                        out[nr][nc] = 2
                        nr -= 1
                        nc -= 1
                if c + 1 < W and grid[r][c + 1] == 0:
                    nr, nc = r + 1, c + 1
                    while 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0:
                        out[nr][nc] = 2
                        nr += 1
                        nc += 1

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8:
                continue
            if r == 0 or r + 1 == H or c == 0 or c + 1 == W:
                continue
            left = grid[r][c - 1]
            right = grid[r][c + 1]
            up = grid[r - 1][c]
            down = grid[r + 1][c]
            if left != 0 and right != 0 and up == 0 and down == 0:
                v = left
                for rr in range(r - 1, -1, -1):
                    if grid[rr][c] != 0:
                        break
                    out[rr][c] = v
                for rr in range(r + 1, H):
                    if grid[rr][c] != 0:
                        break
                    out[rr][c] = v
            elif up != 0 and down != 0 and left == 0 and right == 0:
                v = up
                for cc in range(c - 1, -1, -1):
                    if grid[r][cc] != 0:
                        break
                    out[r][cc] = v
                for cc in range(c + 1, W):
                    if grid[r][cc] != 0:
                        break
                    out[r][cc] = v

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8:
                continue
            for dr in (-1, 1):
                for dc in (-1, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 0 and out[nr][nc] == 0:
                        if (r + c) % 2 == 0:
                            out[nr][nc] = 7

    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 0 or v == 8 or out[r][c]:
                continue
            if c + 1 < W and grid[r][c + 1] == v:
                while c < W and grid[r][c] == v:
                    out[r][c] = v
                    c += 1
            if r + 1 < H and grid[r + 1][c] == v:
                rr = r
                while rr < H and grid[rr][c] == v:
                    out[rr][c] = v
                    rr += 1

    return out