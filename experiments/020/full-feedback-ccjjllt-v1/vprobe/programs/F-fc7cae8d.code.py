def transform(grid):
    h = len(grid)
    w = len(grid[0])
    colors = {}
    for row in grid:
        for v in row:
            colors[v] = colors.get(v, 0) + 1

    def full_rect(v):
        rows = []
        for r in range(h):
            if sum(1 for c in range(w) if grid[r][c] == v) >= 2:
                rows.append(r)
        cols = []
        for c in range(w):
            if sum(1 for r in range(h) if grid[r][c] == v) >= 2:
                cols.append(c)
        if len(rows) < 2 or len(cols) < 2:
            return None
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        if r1 - r0 < 1 or c1 - c0 < 1:
            return None
        ok = all(grid[r][c] == v for r in (r0, r1) for c in range(c0, c1 + 1))
        ok = ok and all(grid[r][c] == v for r in range(r0, r1 + 1) for c in (c0, c1))
        return (r0, c0, r1, c1) if ok else None

    rects = {}
    for v in range(1, 10):
        if full_rect(v):
            rects[v] = full_rect(v)

    chosen = None
    best = -1
    for v, (r0, c0, r1, c1) in rects.items():
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        if inner_h < 1 or inner_w < 1:
            continue
        base_count = 0
        special = 0
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if grid[r][c] == v:
                    continue
                if grid[r][c] == 0:
                    continue
                if base_count == 0:
                    base_count = 1
                else:
                    special += 1
        if base_count == 0:
            continue
        score = inner_h * inner_w * 1000 + special * 10 + (20 if v != 1 else 0)
        if score > best:
            best = score
            chosen = (v, r0, c0, r1, c1)

    if chosen is None:
        return [[0]]

    border, r0, c0, r1, c1 = chosen
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1

    def rot90(mat):
        m = len(mat)
        n = len(mat[0])
        return [list(reversed(row)) for row in zip(*mat)]

    def rot270(mat):
        m = len(mat)
        n = len(mat[0])
        return [list(row) for row in zip(*[list(reversed(row)) for row in mat])]

    def rot180(mat):
        return [list(reversed(row)) for row in reversed(mat)]

    if ih == iw:
        cands = [grid[r0 + 1:r1][c0 + 1:c1]]
    else:
        cands = [rot90(grid[r0 + 1:r1][c0 + 1:c1])]
        cands.append(rot270(grid[r0 + 1:r1][c0 + 1:c1]))

    def fit_score(mat):
        out_h = len(mat)
        out_w = len(mat[0])
        total = 0
        for r in range(out_h):
            row_sum = 0
            for c in range(out_w):
                if mat[r][c] != 0:
                    row_sum += 1
            total += row_sum * row_sum
        return total

    best_mat = None
    best_s = -1
    for mat in cands:
        if len(mat) == 0 or len(mat[0]) == 0:
            continue
        s = fit_score(mat)
        if s > best_s:
            best_s = s
            best_mat = mat

    return best_mat