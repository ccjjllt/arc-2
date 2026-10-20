def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    def val(r, c):
        if 0 <= r < H and 0 <= c < W:
            return grid[r][c]
        return None

    def put(r, c, v):
        if 0 <= r < H and 0 <= c < W and grid[r][c] == 0:
            out[r][c] = v

    def horiz(r, lo, hi):
        for c in range(lo + 1, hi):
            if val(r, c) not in (0, 2):
                return False
        return True

    def diag(r, c, dr, dc, v):
        nr, nc = r + dr, c + dc
        while True:
            cell = val(nr, nc)
            if cell not in (v, 0):
                return
            if cell == v:
                put(nr, nc, v)
            nr += dr
            nc += dc

    def stamp(r, c, v, sr, sc):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                put(r + sr * dr, c + sc * dc, v)

    for r in range(H - 2):
        for c in range(W - 2):
            if not (val(r, c) == 0 and val(r + 2, c) == 0 and
                    val(r, c) == val(r + 1, c) == val(r + 2, c) == 0 and
                    val(r, c) == val(r, c + 1) == val(r, c + 2) == 0 and
                    val(r + 2, c) == val(r + 2, c + 1) == val(r + 2, c + 2) == 0):
                continue
            if not horiz(r + 1, c, c + 2):
                continue
            if not (val(r + 1, c - 1) == 2 and val(r + 1, c + 3) == 2):
                continue
            if val(r, c + 1) != 0 or val(r + 2, c + 1) != 0:
                continue
            for (pr, pc, sr, sc, ddr, ddc) in (
                (r + 1, c + 1, -1, -1, -1, -1),
                (r + 1, c + 1, -1, 1, -1, 1),
                (r + 1, c + 1, 1, -1, 1, -1),
                (r + 1, c + 1, 1, 1, 1, 1),
            ):
                v = val(pr, pc)
                if v not in (0, 2):
                    stamp(pr, pc, v, sr, sc)
                    diag(pr, pc, ddr, ddc, v)
    return out