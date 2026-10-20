def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    border_rows = 0
    border_cols = 0
    while border_rows < H and all(v == 5 for v in grid[border_rows]):
        border_rows += 1
    while border_cols < W and all(v == 5 for v in grid[i][border_cols] for i in range(H)):
        border_cols += 1

    if border_rows == 0 or border_cols == 0:
        return out

    top, bottom = border_rows, H - 1 - border_rows
    left, right = border_cols, W - 1 - border_cols

    for r in range(top + 1, bottom):
        for c in range(left + 1, right):
            if out[r][c] == 5:
                out[r][c] = 0

    def value_at(r, c):
        for dr in (-1, 1, 0):
            for dc in (-1, 1, 0):
                rr, cc = r + dr, c + dc
                if top <= rr < bottom and left <= cc < right:
                    v = out[rr][cc]
                    if v != 0 and v != 5:
                        return v
        return None

    for c in range(left + 1, right, 2):
        for r in range(top + 1, bottom):
            if out[r][c] != 5:
                continue
            below = True
            for rr in range(r + 1, bottom):
                if out[rr][c] not in (0, 5):
                    below = False
                    break
            if below:
                v = value_at(r, c)
                if v is not None:
                    out[r][c] = v
                    for rr in range(r + 1, bottom):
                        if out[rr][c] != 5:
                            out[rr][c] = 5

    for r in range(top + 1, bottom):
        for c in range(left + 1, right, 2):
            if out[r][c] != 5:
                continue
            below = True
            for rr in range(r + 1, bottom):
                if out[rr][c] not in (0, 5):
                    below = False
                    break
            if below:
                v = value_at(r, c)
                if v is not None:
                    out[r][c] = v
                    for rr in range(r + 1, bottom):
                        if out[rr][c] != 5:
                            out[rr][c] = 5

    return out