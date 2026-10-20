def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    colors = set()
    for row in grid:
        for v in row:
            if v:
                colors.add(v)
    if len(colors) != 2:
        return out
    c1, c2 = sorted(colors)
    swap = {c1: c2, c2: c1}

    # A zero is a seed if it is not on the outer border and its 3x3 neighborhood
    # contains no other zeros and has both non-zero colors around it.
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if grid[r][c] != 0:
                continue
            vals = set()
            ok = True
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    v = grid[r + dr][c + dc]
                    if v == 0:
                        ok = False
                        break
                    if v != 8:
                        vals.add(v)
                if not ok:
                    break
            if ok and len(vals) == 2:
                out[r][c] = 8
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        rr, cc = r + dr, c + dc
                        if out[rr][cc] != 8:
                            out[rr][cc] = swap[grid[rr][cc]]

    # After all seeds, any zero whose remaining neighbors are all one color
    # becomes 8 and its neighbors flip to the other color.
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if out[r][c] != 0:
                continue
            vals = set()
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    v = out[r + dr][c + dc]
                    if v != 0 and v != 8:
                        vals.add(v)
            if len(vals) == 1:
                col = vals.pop()
                if col in swap:
                    out[r][c] = 8
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            if dr == 0 and dc == 0:
                                continue
                            rr, cc = r + dr, c + dc
                            if out[rr][cc] != 8:
                                out[rr][cc] = swap[grid[rr][cc]]
    return out