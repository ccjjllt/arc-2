def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    # 1. Replace horizontal 7-runs (always length 3 in demos) with 868.
    centers = []
    for r in range(h):
        c = 0
        while c < w:
            if grid[r][c] == 7:
                if c + 2 < w and grid[r][c + 1] == 7 and grid[r][c + 2] == 7:
                    out[r][c] = 8
                    out[r][c + 1] = 6
                    out[r][c + 2] = 8
                    centers.append((r, c + 1))
                    c += 3
                    continue
            c += 1

    # 2. Put 6 in the vertical neighbors of every new center when they are wall cells.
    for r, c in centers:
        if r > 0 and out[r - 1][c] == 8:
            out[r - 1][c] = 6
        if r + 1 < h and out[r + 1][c] == 8:
            out[r + 1][c] = 6

    # 3. Mark active ends of 8-runs with 3.
    #    Only rows that are internal 3-bordered walls participate.
    active = []
    for r in range(h):
        row = grid[r]
        if row[0] != 3 or row[-1] != 3:
            active.append(False)
            continue
        if r + 1 < h and all(x in (0, 7) for x in grid[r + 1]):
            active.append(False)
        else:
            active.append(True)

    for r in range(h):
        if not active[r]:
            continue
        c = 0
        while c < w:
            if out[r][c] == 8:
                start = c
                while c < w and out[r][c] == 8:
                    c += 1
                end = c - 1
                # Left boundary of this 8-run.
                if start == 0:
                    out[r][start] = 3
                elif out[r][start - 1] == 6:
                    out[r][start] = 3
                elif out[r][start - 1] == 0:
                    # For bottom/final wall rows, a gap is followed by a start marker.
                    if r + 1 >= h:
                        out[r][start] = 3
                # Right boundary of this 8-run.
                if end == w - 1:
                    out[r][end] = 3
                elif out[r][end + 1] == 6:
                    out[r][end] = 3
                elif out[r][end + 1] == 0:
                    # Only rightmost segment of a bottom wall keeps its right end.
                    # This matches the demo outputs.
                    if r + 1 >= h:
                        nxt = end + 1
                        if nxt < w:
                            while nxt < w and out[r][nxt] == 0:
                                nxt += 1
                            if nxt == w - 1 and out[r][nxt] == 8:
                                out[r][end] = 3
            else:
                c += 1

    return out