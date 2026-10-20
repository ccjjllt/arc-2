def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    out = [row[:] for row in grid]

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    seed = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 4:
                seed = (r, c)
                break
        if seed:
            break
    if not seed:
        return out

    sr, sc = seed
    beam_dir = None
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = sr + dr, sc + dc
        if inb(nr, nc) and grid[nr][nc] == 2:
            beam_dir = (dr, dc)
            break

    active = set()
    lines = []

    if beam_dir:
        r, c = sr + beam_dir[0], sr + beam_dir[1] if False else sc + beam_dir[1]
        r = sr + beam_dir[0]
        c = sc + beam_dir[1]
        segs = set()
        while inb(r, c) and grid[r][c] == 2:
            segs.add((r, c))
            r += beam_dir[0]
            c += beam_dir[1]

        r, c = sr
        while inb(r, c):
            if (r, c) not in segs:
                out[r][c] = 2
                active.add((r, c))
            else:
                break
            r += beam_dir[0]
            c += beam_dir[1]

        r, c = sr - beam_dir[0], sc - beam_dir[1]
        stop = False
        while inb(r, c):
            if grid[r][c] in (1, 3):
                stop = True
                break
            out[r][c] = 2
            active.add((r, c))
            r += -beam_dir[0]
            c += -beam_dir[1]
        if stop:
            lines.append(((r, c), (-beam_dir[0], -beam_dir[1]), True))

    def paint_line(start, d, stop_at_shape):
        r, c = start
        while inb(r, c):
            if grid[r][c] in (1, 3):
                if stop_at_shape:
                    break
            out[r][c] = 2
            active.add((r, c))
            if grid[r][c] in (1, 3):
                stop_at_shape = True
            r += d[0]
            c += d[1]

    def activate_shape(r, c):
        if (r, c) in active or grid[r][c] not in (1, 3):
            return
        comp = []
        stack = [(r, c)]
        seen = {(r, c)}
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = cr + dr, cc + dc
                if inb(nr, nc) and grid[nr][nc] in (1, 3) and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        for rr, cc in comp:
            out[rr][cc] = 2
            active.add((rr, cc))

        threes = [(rr, cc) for rr, cc in comp if grid[rr][cc] == 3]
        if not threes:
            return

        ones = [(rr, cc) for rr, cc in comp if grid[rr][cc] == 1]
        if not ones:
            return

        avg_r = sum(rr for rr, cc in ones) / len(ones)
        avg_c = sum(cc for rr, cc in ones) / len(ones)
        tr = sum(rr for rr, cc in threes) / len(threes)
        tc = sum(cc for rr, cc in threes) / len(threes)

        vr = tr - avg_r
        vc = tc - avg_c
        if abs(vr) >= abs(vc) and vr != 0:
            d = (1 if vr > 0 else -1, 0)
        elif vc != 0:
            d = (0, 1 if vc > 0 else -1)
        else:
            return

        start = None
        if d[0] != 0:
            cols = sorted(set(cc for _, cc in threes))
            for rr, cc in threes:
                nr, nc = rr + d[0], cc
                if inb(nr, nc):
                    if start is None or nr < start[0] or (nr == start[0] and cc < start[1]):
                        start = (nr, nc)
        else:
            rows = sorted(set(rr for rr, _ in threes))
            for rr, cc in threes:
                nr, nc = rr, cc + d[1]
                if inb(nr, nc):
                    if start is None or nc < start[1] or (nc == start[1] and rr < start[0]):
                        start = (nr, nc)
        if start:
            lines.append((start, d, True))

    def flood_active():
        changed = True
        while changed:
            changed = False
            for r, c in list(active):
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if inb(nr, nc) and (nr, nc) not in active and grid[nr][nc] in (1, 3):
                        activate_shape(nr, nc)
                        changed = True
                        break

    def apply_line(start, d, stop_at_shape):
        r, c = start
        while inb(r, c):
            if grid[r][c] in (1, 3):
                if stop_at_shape:
                    break
                activate_shape(r, c)
                flood_active()
                stop_at_shape = True
            out[r][c] = 2
            active.add((r, c))
            r += d[0]
            c += d[1]

    # process initial seed hit first, then emitted lines in stable order
    for line in lines:
        apply_line(line[0], line[1], line[2])

    for r in range(H):
        for c in range(W):
            if out[r][c] not in (2, bg):
                out[r][c] = bg
    return out