def transform(grid):
    n = len(grid)
    s = n // 2
    q = [[row[:s] for row in grid[:s]], [row[s:] for row in grid[:s]],
         [row[:s] for row in grid[s:]], [row[s:] for row in grid[s:]]]

    def crop(rect):
        if not rect:
            return None
        rows = [i for i, row in enumerate(rect) if any(v != 0 for v in row)]
        cols = [j for j in range(len(rect[0])) if any(rect[i][j] != 0 for i in range(len(rect)))]
        if not rows or not cols:
            return None
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return [row[c0:c1 + 1] for row in rect[r0:r1 + 1]]

    def solid_rows(rect):
        if not rect:
            return False
        vals = set()
        for row in rect:
            nz = set(v for v in row if v != 0)
            if not nz:
                continue
            vals.update(nz)
            if len(nz) > 1:
                return False
        return 1 <= len(vals) <= 4

    def hole_count(rect):
        if not rect:
            return 0
        return sum(1 for row in rect for v in row if v == 0)

    def nz_count(rect):
        if not rect:
            return 0
        return sum(1 for row in rect for v in row if v != 0)

    def nonz_color(rect):
        if not rect:
            return 0
        vals = set(v for row in rect for v in row if v != 0)
        return sorted(vals)[0]

    def base_from(mask, fill):
        mh, mw = len(mask), len(mask[0])
        fh, fw = len(fill), len(fill[0])
        return [[mask[r][c] if mask[r][c] != 0 else fill[(r * fh) // mh][(c * fw) // mw]
                 for c in range(mw)] for r in range(mh)]

    def size_for(mask, fill, others):
        mh, mw = len(mask), len(mask[0])
        if not fill or len(fill) < len(mask):
            return mh, mw
        if not others:
            return mh, mw
        ov = set(tuple(v for v in row if v != 0) for row in others)
        if len(ov) > 2:
            return mh, mw
        if all(tuple(row) == tuple(fill[-1]) for row in fill):
            return mh, mw
        if solid_rows(fill):
            if all(tuple(row) == tuple(fill[0]) for row in fill):
                return mh, mw
        if any(all(v == 0 for v in row) for row in mask):
            if all(tuple(row) == tuple(fill[-1]) for row in fill):
                return mh, mw
            return min(2 * mh, 30), min(2 * mw, 30)
        return min(2 * mh, 30), min(2 * mw, 30)

    mask_idx = None
    for idx in (3, 2, 0, 1):
        rect = crop(q[idx])
        if rect is None:
            continue
        if hole_count(rect) > 0 or nz_count(rect) >= 4:
            mask_idx = idx
            break
    if mask_idx is None:
        best, best_score = None, -1
        for idx in (3, 2, 0, 1):
            rect = crop(q[idx])
            if rect is None:
                continue
            score = (len(rect), len(rect[0]), idx in (3, 2))
            if score > best_score:
                best, best_score = rect, score
        if best is None:
            return [[0 for _ in range(len(grid[0]))] for _ in range(len(grid))]
        mask = best
        return mask

    mask = crop(q[mask_idx])
    fill_idx = None
    for idx in (2, 0, 1, 3):
        if idx == mask_idx:
            continue
        rect = crop(q[idx])
        if rect is None:
            continue
        if solid_rows(rect) or len(set(v for row in rect for v in row if v != 0)) <= 2:
            fill_idx = idx
            break
    fill = crop(q[fill_idx]) if fill_idx is not None else None
    if fill is None:
        fill = mask

    others = []
    for idx in range(4):
        if idx == mask_idx or idx == fill_idx:
            continue
        rect = crop(q[idx])
        if rect:
            others.append(rect)

    base = base_from(mask, fill)
    h, w = size_for(mask, fill, others)
    rh = (h + len(base) - 1) // len(base)
    rw = (w + len(base[0]) - 1) // len(base[0])
    out = [row[:w] for row in (base * rh)[:h]]
    if rw > 1:
        out = [row * rw for row in out]
    return out