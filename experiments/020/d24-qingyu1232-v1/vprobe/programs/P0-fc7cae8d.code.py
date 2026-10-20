def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1

    def rect(r0, c0, r1, c1):
        return (r0, c0, r1, c1)

    best = None
    best_key = (-1, -1, -1, -1)

    for color in sorted(counts, key=lambda x: (-counts[x], x)):
        if counts[color] < 4:
            continue
        rows = [r for r in range(H) if sum(1 for c in range(W) if grid[r][c] == color) >= 2]
        cols = [c for c in range(W) if sum(1 for r in range(H) if grid[r][c] == color) >= 2]
        if len(rows) < 2 or len(cols) < 2:
            continue
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 1 or c1 - c0 < 1:
            continue
        interior_h = r1 - r0 - 1
        interior_w = c1 - c0 - 1
        if interior_h < 2 or interior_w < 2:
            continue
        inner = []
        for r in range(r0 + 1, r1):
            inner.append(grid[r][c0 + 1:c1])
        base_counts = {}
        for row in inner:
            for v in row:
                if v != 0:
                    base_counts[v] = base_counts.get(v, 0) + 1
        if not base_counts:
            continue
        base = max(base_counts, key=lambda x: base_counts[x])
        specials = sorted(set(v for row in inner for v in row if v != 0 and v != base),
                          key=lambda x: (-sum(1 for rr in inner for vv in rr if vv == x), x))
        if len(specials) != 4:
            continue
        s1, s2, s3, s4 = specials
        score = counts[color]
        score += 100 * (2 if color not in base_counts else 0)
        score += 50 * min(interior_h, interior_w)
        score -= sum(base_counts[v] for v in (s1, s2, s3, s4) if v in base_counts)
        key = (score, interior_h * interior_w, counts[color])
        if key > best_key:
            best_key = key
            best = (rect(r0, c0, r1, c1), base, (s1, s2, s3, s4))

    if best is None:
        return [[0]]

    r0, c0, r1, c1 = best[0]
    base, (s1, s2, s3, s4) = best[1], best[2]
    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1

    def get(r, c):
        return grid[r0 + 1 + r][c0 + 1 + c]

    def put(out, r, c, v):
        if 0 <= r < out_h and 0 <= c < out_w:
            out[r][c] = v

    def add_shape(out, kind):
        if kind == "H":
            for r in range(out_h):
                for c in range(out_w):
                    put(out, r, c, s1)
            for r in range(out_h):
                for c in range(1, min(3, out_w)):
                    put(out, r, c, s2)
        elif kind == "V":
            for r in range(out_h):
                for c in range(out_w):
                    put(out, r, c, s1)
            for r in range(1, min(3, out_h)):
                for c in range(out_w):
                    put(out, r, c, s2)
        else:
            for r in range(out_h):
                for c in range(out_w):
                    put(out, r, c, s1)
            for r in range(1, min(3, out_h)):
                for c in range(1, min(3, out_w)):
                    put(out, r, c, s2)

    out = [[base for _ in range(out_w)] for _ in range(out_h)]
    kind = None

    for r in range(out_h):
        for c in range(out_w):
            if get(r, c) == s3:
                put(out, r, c, s3)
                kind = kind or ("H" if (r + 1 < out_h and get(r + 1, c) == s3) else
                                ("V" if (c + 1 < out_w and get(r, c + 1) == s3) else "B"))

    if kind is None:
        has3 = [[False] * out_w for _ in range(out_h)]
        for r in range(out_h):
            for c in range(out_w):
                if get(r, c) == s3:
                    has3[r][c] = True
        vertical = sum(1 for c in range(out_w) if any(has3[r][c] for r in range(out_h)))
        horizontal = sum(1 for r in range(out_h) if any(has3[r][c] for c in range(out_w)))
        if vertical >= horizontal:
            kind = "V"
        elif horizontal >= vertical:
            kind = "H"
        else:
            kind = "B"
    else:
        if kind == "H":
            if out_h >= out_w:
                kind = "H"
            else:
                kind = "V"
        elif kind == "V":
            if out_w >= out_h:
                kind = "V"
            else:
                kind = "H"

    add_shape(out, kind)

    if s4 == 0:
        if kind == "H":
            for c in range(out_w - 2, out_w):
                put(out, 0, c, 0)
        elif kind == "V":
            for r in range(out_h - 2, out_h):
                put(out, r, 0, 0)
        else:
            for r in range(out_h - 1, out_h):
                for c in range(out_w - 1, out_w):
                    put(out, r, c, 0)
    else:
        for r in range(out_h):
            for c in range(out_w):
                if get(r, c) == s4:
                    put(out, r, c, s4)

    return out