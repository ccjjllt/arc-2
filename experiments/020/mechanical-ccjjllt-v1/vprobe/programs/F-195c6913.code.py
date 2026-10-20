def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = None
    bcount = -1
    seen = {}
    for row in grid:
        for v in row:
            seen[v] = seen.get(v, 0) + 1
    if 0 in seen:
        bg, bcount = 0, seen[0]
    else:
        for v in seen:
            if seen[v] > bcount:
                bg, bcount = v, seen[v]

    specials = []
    for v, cnt in seen.items():
        if v == bg:
            continue
        if cnt == 4:
            pts = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == v]
            if (max(p[0] for p in pts) - min(p[0] for p in pts) == 1 and
                max(p[1] for p in pts) - min(p[1] for p in pts) == 1):
                specials.append((min(p[0] for p in pts), min(p[1] for p in pts), v))
    if len(specials) < 2:
        return grid

    rmin = min(p[0] for p in specials)
    cmin = min(p[1] for p in specials)
    rmax = max(p[0] for p in specials)
    cmax = max(p[1] for p in specials)
    left = [s for s in specials if s[1] == cmin]
    right = [s for s in specials if s[1] == cmax]
    left.sort(key=lambda s: s[0])
    right.sort(key=lambda s: s[0])
    if len(left) > 1 and len(right) > 1:
        route = [left[0], right[0], right[-1], left[-1]]
    elif len(right) > 1:
        route = [left[0], right[0], right[-1]]
    elif len(left) > 1:
        route = [left[0], left[-1], right[0]]
    else:
        top = [s for s in specials if s[0] == rmin]
        bot = [s for s in specials if s[0] == rmax]
        top.sort(key=lambda s: s[1])
        bot.sort(key=lambda s: s[1])
        route = [top[0], top[-1], bot[-1], bot[0]]

    out = [row[:] for row in grid]
    prev = route[0]
    seq_idx = 1
    for i in range(1, len(route)):
        cur = route[i]
        r = prev[0]
        c = prev[1]
        sr = 1 if cur[0] > r else -1
        sc = 1 if cur[1] > c else -1
        while (r, c) != (cur[0], cur[1]):
            if sc != 0:
                c += sc
                out[r][c] = seq_idx % 10
                seq_idx += 1
            elif sr != 0:
                r += sr
                out[r][c] = seq_idx % 10
                seq_idx += 1
        r = cur[0]
        c = cur[1]
        out[r][c] = cur[2]
        prev = cur
    return out