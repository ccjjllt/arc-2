def transform(grid):
    n = len(grid)
    h = n // 2
    w = n // 2

    def crop(rect):
        vals = set()
        for row in rect:
            vals.update(row)
        vals.discard(0)
        if not vals:
            return []
        rows = [i for i, row in enumerate(rect) if any(v != 0 for v in row)]
        cols = [j for j in range(len(rect[0])) if any(rect[i][j] != 0 for i in range(len(rect)))]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return [row[c0:c1 + 1] for row in rect[r0:r1 + 1]]

    tl = [row[:w] for row in grid[:h]]
    tr = [row[w:] for row in grid[:h]]
    bl = [row[:w] for row in grid[h:]]
    br = [row[w:] for row in grid[h:]]
    rects = [crop(tl), crop(tr), crop(bl), crop(br)]

    mask = None
    fill = None
    for m in range(4):
        for f in range(4):
            if m == f or not rects[m] or not rects[f]:
                continue
            m0 = min(min(row) for row in rects[m])
            fvals = set(v for row in rects[f] for v in row)
            if m0 in fvals and len(fvals) <= len(set(v for row in rects[m] for v in row)):
                mask, fill = rects[m], rects[f]
                break
        if mask is not None:
            break

    if mask is None:
        return grid

    mh, mw = len(mask), len(mask[0])
    fh, fw = len(fill), len(fill[0])

    def blend(m, f, r, c):
        return m if m != 0 else f[(r * fh) // mh][(c * fw) // mw]

    out = [[blend(mask[r][c], fill, r, c) for c in range(mw)] for r in range(mh)]

    repeat_v = 1
    repeat_h = 1
    for idx, rect in enumerate(rects):
        if rect and idx != mask and idx != fill:
            rh = len(rect)
            rw = len(rect[0])
            if rh >= repeat_v:
                repeat_v = rh
            if rw >= repeat_h:
                repeat_h = rw

    if repeat_v * mh > 30:
        repeat_v = 1
    if repeat_h * mw > 30:
        repeat_h = 1

    result = []
    for _ in range(repeat_v):
        result.extend(out)
    if repeat_h > 1:
        result = [row * repeat_h for row in result]

    return result