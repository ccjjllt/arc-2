def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Find the two colors that are not 0 and 8.
    colors = set()
    for row in grid:
        for v in row:
            if v not in (0, 8):
                colors.add(v)
    if len(colors) != 2:
        return [row[:] for row in grid]
    a, b = tuple(sorted(colors))
    other = {a: b, b: a}

    # Mark 0 cells that are local "cracks": all nonzero 8-neighbors
    # (if any) have the same color.  Border cells are allowed if this holds.
    centers = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 0:
                continue
            seen = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr = r + dr
                    cc = c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        v = grid[rr][cc]
                        if v not in (0, 8):
                            seen.add(v)
            if len(seen) == 1:
                centers.append((r, c))

    out = [row[:] for row in grid]

    # Paint each selected 3x3 neighborhood with the opposite color.
    for r, c in centers:
        for dr in (-1, 0, 1):
            rr = r + dr
            if not (0 <= rr < H):
                continue
            for dc in (-1, 0, 1):
                cc = c + dc
                if not (0 <= cc < W):
                    continue
                # The neighborhood should be filled with the color opposite
                # to the one surrounding the center.  Use the majority nonzero
                # neighbor as the surrounding color.
                if out[r][c] == 0:
                    # Determine fill color from original grid neighbors.
                    neigh = set()
                    for nr in (r - 1, r, r + 1):
                        for nc in (c - 1, c, c + 1):
                            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] not in (0, 8):
                                neigh.add(grid[nr][nc])
                    fill = other.get(neigh.pop(), 0)
                    if fill:
                        out[rr][cc] = fill

    # Place 8 at the centers, after overwriting.
    for r, c in centers:
        if 0 <= r < H and 0 <= c < W:
            out[r][c] = 8

    return out