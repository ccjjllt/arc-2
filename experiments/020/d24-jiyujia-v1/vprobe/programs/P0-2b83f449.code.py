def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    # First pass: turn every horizontal 777 run into 868 and mark its center.
    centers = set()
    for r in range(H):
        c = 0
        while c < W:
            if grid[r][c] == 7:
                if c + 2 < W and grid[r][c + 1] == 7 and grid[r][c + 2] == 7:
                    out[r][c] = 8
                    out[r][c + 1] = 6
                    out[r][c + 2] = 8
                    centers.add((r, c + 1))
                    c += 3
                    continue
            c += 1

    # Second pass: propagate 6 upward and downward into adjacent non-zero cells.
    for (r, c) in centers:
        if r > 0 and out[r - 1][c] != 0:
            out[r - 1][c] = 6
        if r + 1 < H and out[r + 1][c] != 0:
            out[r + 1][c] = 6

    # Determine the first and last 8-cell in each original row.
    # These act as active wall endpoints only in rows of the main alternating pattern.
    first8 = [W] * H
    last8 = [-1] * H
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 8:
                if first8[r] == W:
                    first8[r] = c
                last8[r] = c

    def is_wall_row(r):
        row = grid[r]
        if first8[r] == W:
            return False
        if row[0] != 3 or row[-1] != 3:
            return False
        # The last row of the puzzle is not a normal 3-bordered wall row.
        if r + 1 < H:
            if all(x in (0, 7) for x in grid[r + 1]):
                return False
        return True

    # Third pass: move the two endpoint markers (color 3) inward by one cell
    # when their corresponding wall row is a normal 3-bordered wall row.
    for r in range(H):
        if is_wall_row(r):
            f = first8[r]
            if f < W:
                if out[r][f] == 3:
                    out[r][f] = 8
                if f + 1 < W and out[r][f + 1] == 8:
                    out[r][f + 1] = 3
            l = last8[r]
            if l >= 0:
                if out[r][l] == 3:
                    out[r][l] = 8
                if l - 1 >= 0 and out[r][l - 1] == 8:
                    out[r][l - 1] = 3

    return out