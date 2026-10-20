def transform(grid):
    h = len(grid)
    w = len(grid[0])

    seen = [[False] * w for _ in range(h)]
    cells = []

    for r in range(h):
        for c in range(w):
            if seen[r][c]:
                continue
            color = grid[r][c]
            if color == 1:
                continue

            top, bottom, left, right = r, r, c, c
            row = r
            while row < h and min(grid[row][c], color) == color and all(x == color for x in grid[row]):
                row += 1
            bottom = row - 1
            row = r
            while row >= 0 and all(grid[row][c] == color for _ in [1]):
                if any(grid[row][cc] != color for cc in range(c, right + 1) if cc < w):
                    break
                row -= 1
            top = row + 1

            col = c
            while col < w:
                ok = True
                for rr in range(top, bottom + 1):
                    if grid[rr][col] != color:
                        ok = False
                        break
                if not ok:
                    break
                col += 1
            right = col - 1

            col = c
            while col >= 0:
                ok = True
                for rr in range(top, bottom + 1):
                    if grid[rr][col] != color:
                        ok = False
                        break
                if not ok:
                    break
                col -= 1
            left = col + 1

            for rr in range(top, bottom + 1):
                for cc in range(left, right + 1):
                    seen[rr][cc] = True

            ones = []
            special = None
            for rr in range(top, bottom + 1):
                for cc in range(left, right + 1):
                    v = grid[rr][cc]
                    if v == 1:
                        ones.append((rr - top, cc - left))
                    elif v != color:
                        special = v

            cells.append({
                "rect": (top, left, bottom, right),
                "color": color,
                "ones": ones,
                "special": special
            })

    rows = sorted(set(cell["rect"][0] for cell in cells) |
                  set(cell["rect"][2] + 1 for cell in cells))
    cols = sorted(set(cell["rect"][1] for cell in cells) |
                  set(cell["rect"][3] + 1 for cell in cells))

    out_h = len(rows)
    out_w = len(cols)
    out = [[0] * out_w for _ in range(out_h)]

    for cell in cells:
        top, left, bottom, right = cell["rect"]
        ri = rows.index(top)
        ci = cols.index(left)
        rh = rows.index(bottom + 1) - ri
        rw = cols.index(right + 1) - ci

        ones = cell["ones"]
        special = cell["special"]
        dr = dc = 0
        fill = cell["color"]

        if ones:
            rmin = min(p[0] for p in ones)
            rmax = max(p[0] for p in ones)
            cmin = min(p[1] for p in ones)
            cmax = max(p[1] for p in ones)
            h1 = rmax - rmin + 1
            w1 = cmax - cmin + 1

            if (special is not None and h1 >= 3 and w1 >= 3 and
                    (h1, w1) == (3, 3)):
                dr = 1
            elif h1 >= 3 and w1 == 1:
                dr = 1
            elif w1 >= 3 and h1 == 1:
                dc = 1
            elif h1 >= 2 and w1 >= 2:
                if special is not None:
                    sr = special_row = None
                    sc = None
                    for rr, cc in zip(range(top, bottom + 1), range(left, right + 1)):
                        if grid[rr][cc] == special:
                            sr = rr - top
                            sc = cc - left
                            break
                    if sr is not None and sc is not None:
                        if sr >= rmax:
                            dr = 1
                        elif sr <= rmin:
                            dr = -1
                        if sc >= cmax:
                            dc = 1
                        elif sc <= cmin:
                            dc = -1
                        dr = dr or 1
                        dc = dc or 0
                else:
                    arms_r = set()
                    arms_c = set()
                    for rr, cc in ones:
                        if cmin < cc < cmax:
                            if rr == rmin:
                                arms_r.add(-1)
                            elif rr == rmax:
                                arms_r.add(1)
                        if rmin < rr < rmax:
                            if cc == cmin:
                                arms_c.add(-1)
                            elif cc == cmax:
                                arms_c.add(1)
                    if len(arms_r) + len(arms_c) <= 1:
                        if arms_r:
                            dr = arms_r.pop()
                        elif arms_c:
                            dc = arms_c.pop()
                    else:
                        dr = 1
            if special is not None:
                fill = special

        for tr in range(out_h):
            for tc in range(out_w):
                srr = tr - dr
                stc = tc - dc
                if 0 <= srr < out_h and 0 <= stc < out_w:
                    s_top = rows[srr]
                    s_left = cols[stc]
                    for cell2 in cells:
                        if cell2["rect"][0] == s_top and cell2["rect"][1] == s_left:
                            if (tr, tc) in cell2["rect"]:
                                continue
                            out[tr][tc] = cell2["color"]
                            break
                    else:
                        out[tr][tc] = fill

        for tr in range(out_h):
            for tc in range(out_w):
                if out[tr][tc] == 0 and (tr, tc) in cell["rect"]:
                    out[tr][tc] = fill

    return out