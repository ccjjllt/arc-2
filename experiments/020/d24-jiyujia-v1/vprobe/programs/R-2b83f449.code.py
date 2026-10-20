def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Locate horizontal 7-triplets.
    centers = []
    for r in range(H):
        c = 0
        while c < W:
            if grid[r][c] == 7 and c + 2 < W and grid[r][c + 1] == 7 and grid[r][c + 2] == 7:
                centers.append((r, c + 1))
                c += 3
            else:
                c += 1

    # Base output: 7-triplets become 8,6,8.
    out = [row[:] for row in grid]
    for r, cm in centers:
        out[r][cm - 1] = 8
        out[r][cm] = 6
        out[r][cm + 1] = 8

    # Vertical propagation of 6 to adjacent rows, where the source/target
    # are non-zero cells (usually 8 or 3).
    for r, cm in centers:
        if r > 0 and out[r][cm] == 6 and out[r - 1][cm] != 0:
            out[r - 1][cm] = 6
        if r + 1 < H and out[r][cm] == 6 and out[r + 1][cm] != 0:
            out[r + 1][cm] = 6

    # Add 3s on full-width structural rows (rows that are all 8/3 in input,
    # no holes) at the two cells immediately left/right of each 6, if those
    # cells are originally non-zero 8s.
    for r in range(H):
        if any(grid[r][c] == 0 for c in range(W)):
            continue
        for c in range(W):
            if out[r][c] == 6:
                if c - 1 >= 0 and grid[r][c - 1] == 8:
                    out[r][c - 1] = 3
                if c + 1 < W and grid[r][c + 1] == 8:
                    out[r][c + 1] = 3

    return out