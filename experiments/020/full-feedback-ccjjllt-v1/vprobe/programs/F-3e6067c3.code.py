def transform(grid):
    h = len(grid)
    w = len(grid[0])

    cnt = {}
    for row in grid:
        for v in row:
            cnt[v] = cnt.get(v, 0) + 1
    bg = min(cnt, key=lambda x: (cnt[x], x))

    best_row, best_n = -1, 0
    for r in range(h):
        n = 0
        for c in range(w):
            if grid[r][c] != bg:
                left_ok = c == 0 or grid[r][c - 1] == bg
                right_ok = c == w - 1 or grid[r][c + 1] == bg
                if left_ok and right_ok:
                    n += 1
        if n > best_n:
            best_n = n
            best_row = r

    seq = []
    if best_row >= 0:
        for c in range(w):
            if grid[best_row][c] != bg:
                left_ok = c == 0 or grid[best_row][c - 1] == bg
                right_ok = c == w - 1 or grid[best_row][c + 1] == bg
                if left_ok and right_ok:
                    seq.append(grid[best_row][c])

    seen = [[False] * w for _ in range(h)]
    boxes = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == bg or seen[r][c]:
                continue
            stack = [(r, c)]
            seen[r][c] = True
            comp = []
            while stack:
                y, x = stack.pop()
                comp.append((y, x, grid[y][x]))
                for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < h and 0 <= nx < w and not seen[ny][nx] and grid[ny][nx] != bg:
                        seen[ny][nx] = True
                        stack.append((ny, nx))
            if len(comp) <= 1:
                continue
            vc = {}
            for _, _, v in comp:
                vc[v] = vc.get(v, 0) + 1
            label = min(vc, key=lambda x: (vc[x], x))
            rs = [y for y, _, _ in comp]
            cs = [x for _, x, _ in comp]
            boxes.append({"label": label, "r0": min(rs), "r1": max(rs), "c0": min(cs), "c1": max(cs)})

    out = [row[:] for row in grid]

    def same_row(a, b):
        return not (a["r1"] < b["r0"] - 1 or b["r1"] < a["r0"] - 1)

    def same_col(a, b):
        return not (a["c1"] < b["c0"] - 1 or b["c1"] < a["c0"] - 1)

    def aligned(a, b):
        return same_row(a, b) or same_col(a, b)

    def match_score(a, b):
        if not aligned(a, b):
            return -10**9
        score = 0
        if same_row(a, b):
            score += 1000
            if abs(a["c1"] - b["c0"]) <= 6 or abs(b["c1"] - a["c0"]) <= 6:
                score += 100
        if same_col(a, b):
            score += 1000
            if abs(a["r1"] - b["r0"]) <= 6 or abs(b["r1"] - a["r0"]) <= 6:
                score += 100
        return score

    for i in range(len(seq) - 1):
        a_label = seq[i]
        b_label = seq[i + 1]
        cands = [b for b in boxes if b["label"] in (a_label, b_label)]
        best = None
        best_score = -10**9
        for bi, a in enumerate(cands):
            if a["label"] != a_label:
                continue
            for bj, b in enumerate(cands):
                if a is b or b["label"] != b_label:
                    continue
                score = match_score(a, b)
                if score > best_score:
                    best_score = score
                    best = (a, b)
        if best is None or best_score < 0:
            continue
        a, b = best
        col_a = (a["c0"] + a["c1"]) // 2
        row_a = (a["r0"] + a["r1"]) // 2
        col_b = (b["c0"] + b["c1"]) // 2
        row_b = (b["r0"] + b["r1"]) // 2
        if same_row(a, b):
            if a["c0"] < b["c0"]:
                c_start = a["c1"] + 1
                c_end = b["c0"]
            else:
                c_start = b["c1"] + 1
                c_end = a["c0"]
            if c_start <= c_end:
                if same_row(a, b) and not same_col(a, b):
                    r = row_a
                elif a["r0"] <= b["r1"] and b["r0"] <= a["r1"]:
                    r = max(a["r0"], b["r0"])
                else:
                    r = row_a
                for c in range(c_start, c_end + 1):
                    if out[r][c] == bg:
                        out[r][c] = a["label"]
        elif same_col(a, b):
            if a["r0"] < b["r0"]:
                r_start = a["r1"] + 1
                r_end = b["r0"]
            else:
                r_start = b["r1"] + 1
                r_end = a["r0"]
            if r_start <= r_end:
                if same_col(a, b) and not same_row(a, b):
                    c = col_a
                elif a["c0"] <= b["c1"] and b["c0"] <= a["c1"]:
                    c = max(a["c0"], b["c0"])
                else:
                    c = col_a
                for r in range(r_start, r_end + 1):
                    if out[r][c] == bg:
                        out[r][c] = a["label"]

    return out