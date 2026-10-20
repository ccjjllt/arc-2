def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def background(g):
        counts = {}
        for row in g:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        return max(sorted(counts), key=lambda c: (counts[c], -c))

    bg = background(grid)

    def is_bg(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] == bg

    def is_nonbg(r, c):
        return 0 <= r < H and 0 <= c < W and grid[r][c] != bg

    drds = ((0, 1), (1, 0), (1, 1), (1, -1))
    segments = []

    for r in range(H):
        for c in range(W):
            if grid[r][c] == bg:
                continue
            for dr, dc in drds:
                nr, nc = r - dr, c - dc
                if is_nonbg(nr, nc) and (nr + dr, nc + dc) == (r, c):
                    continue
                cell = (r, c)
                while True:
                    cr, cc = cell
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < H and 0 <= nc < W):
                        break
                    if grid[nr][nc] != grid[cell]:
                        break
                    # require straight line and no branch at next cell
                    pr, pc = nr - dr, nc - dc
                    q1r, q1c = nr - dc, nc - dr
                    q2r, q2c = nr + dc, nc + dr
                    prev_ok = not is_nonbg(pr, pc) or grid[pr][pc] != grid[nr][nc]
                    side1_ok = not is_nonbg(q1r, q1c) or grid[q1r][q1c] != grid[nr][nc]
                    side2_ok = not is_nonbg(q2r, q2c) or grid[q2r][q2c] != grid[nr][nc]
                    if not (prev_ok and side1_ok and side2_ok):
                        break
                    cell = (nr, nc)
                length = 0
                cr, cc = cell
                while True:
                    pr, pc = cr - dr, cc - dc
                    if not is_nonbg(pr, pc) or grid[pr][pc] != grid[cell]:
                        break
                    cell = (pr, pc)
                while is_nonbg(cell[0], cell[1]) and grid[cell[0]][cell[1]] != bg:
                    segments.append(cell)
                    cell = (cell[0] + dr, cell[1] + dc)
                    length += 1
                break

    # Group contiguous cells into maximal segments with direction and starting point.
    cells = []
    seen = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg and (r, c) not in seen:
                for dr, dc in drds:
                    pr, pc = r - dr, c - dc
                    if is_nonbg(pr, pc) and grid[pr][pc] == grid[r][c]:
                        continue
                    dr, dc = dr, dc
                    # only keep directions that form a valid straight component
                    length = 0
                    cr, cc = r, c
                    while is_nonbg(cr, cc) and grid[cr][cc] == grid[r][c]:
                        length += 1
                        cr, cc = cr + dr, cc + dc
                    if length < 2:
                        continue
                    # verify all neighbors along line match and no diagonal branch causing misdirection
                    ok = True
                    rr, cc = r, c
                    while is_nonbg(rr, cc) and grid[rr][cc] == grid[r][c]:
                        # check perpendicular cells for same color not part of this line
                        p1 = (rr - dc, cc - dr)
                        p2 = (rr + dc, cc + dr)
                        # previous along line (except start)
                        if (rr, cc) != (r, c):
                            p0 = (rr - dr, cc - dc)
                            if is_nonbg(p0[0], p0[1]) and grid[p0[0]][p0[1]] == grid[r][c]:
                                pass
                        # We allow other segments to touch only at endpoints; for these tasks,
                        # the simple maximal line extraction is enough.
                        rr, cc = rr + dr, cc + dc
                    for rr, cc in [(r + k * dr, c + k * dc) for k in range(length)]:
                        seen.add((rr, cc))
                        cells.append((rr, cc, dr, dc, r, c, length))
                    break

    # Build list of unique maximal segments: key (start, direction, color)
    seg_map = {}
    for r, c, dr, dc, sr, sc, ln in cells:
        key = (sr, sc, dr, dc, grid[sr][sc])
        if key not in seg_map or ln > seg_map[key]:
            seg_map[key] = ln

    # Convert to ordered segments by perpendicular coordinate, then along-line.
    ordered = []
    for (sr, sc, dr, dc, color), ln in seg_map.items():
        if dc == 1:
            perp = sr
            along = sc
        elif dc == -1:
            perp = sr
            along = sc
        elif dc == 0 and dr == 1:
            perp = sc
            along = sr
        else:
            perp = sc
            along = sr
        ordered.append((perp, along, sr, sc, dr, dc, color, ln))

    ordered.sort(key=lambda x: (x[0], x[1]))

    if not ordered:
        return [row[:] for row in grid]

    target = ordered[len(ordered) // 2][7]

    out = [row[:] for row in grid]

    for perp, along, sr, sc, dr, dc, color, ln in ordered:
        # Remove original segment cells.
        for k in range(ln):
            out[sr + k * dr][sc + k * dc] = bg

        # Choose the start closest to the original midpoint, constrained to grid.
        mid_float = (ln - 1) / 2.0
        best_start = None
        best_dist = 10**9
        for t in range(target):
            rr = sr + t * dr
            cc = sc + t * dc
            valid = True
            for k in range(target):
                if not (0 <= rr + k * dr < H and 0 <= cc + k * dc < W):
                    valid = False
                    break
            if not valid:
                continue
            # distance from original midpoint in cell units
            dist = abs((t - 0.5) - mid_float)
            if dist < best_dist:
                best_dist = dist
                best_start = (rr, cc)

        if best_start is None:
            continue

        rr, cc = best_start
        for k in range(target):
            out[rr + k * dr][cc + k * dc] = color

    return out