def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] != bg]
    if not cells:
        return [row[:] for row in grid]

    vals = sorted({grid[r][c] for r, c in cells})
    a, b = vals[0], vals[1]

    def block(r0, r1, c0, c1):
        out = []
        for r in range(r0, r1 + 1):
            row = []
            for c in range(c0, c1 + 1):
                v = grid[r][c]
                if v == a:
                    row.append(b)
                elif v == b:
                    row.append(a)
                else:
                    row.append(bg)
            out.append(row)
        return out

    def rows_with_content(block):
        return [i for i, row in enumerate(block) if any(v != bg for v in row)]

    def cols_with_content(block):
        Hb = len(block)
        Wb = len(block[0]) if Hb else 0
        return [j for j in range(Wb) if any(row[j] != bg for row in block)]

    def tight(block):
        ridx = rows_with_content(block)
        if not ridx:
            return []
        cidx = cols_with_content(block)
        if not cidx:
            return []
        r0, r1 = min(ridx), max(ridx)
        c0, c1 = min(cidx), max(cidx)
        return [row[c0:c1 + 1] for row in block[r0:r1 + 1]]

    def place(block, r, c):
        for i, row in enumerate(block):
            for j, v in enumerate(row):
                rr = r + i
                cc = c + j
                if 0 <= rr < H and 0 <= cc < W and v != bg:
                    out[rr][cc] = v

    out = [row[:] for row in grid]

    rmin = min(r for r, c in cells)
    rmax = max(r for r, c in cells)
    mid = (rmin + rmax + 1) // 2

    top_in = block(rmin, mid - 1, 0, W - 1)
    bot_in = block(mid, rmax, 0, W - 1)
    top = tight(top_in)
    bot = tight(bot_in)

    tr0 = next(i for i, row in enumerate(top_in) if any(v != bg for v in row))
    br0 = next(i for i, row in enumerate(bot_in) if any(v != bg for v in row))
    tr1 = max(i for i, row in enumerate(top_in) if any(v != bg for v in row))
    br1 = min(i for i, row in enumerate(bot_in) if any(v != bg for v in row))
    tc = min(j for j in range(W) if any(top_in[i][j] != bg for i in range(len(top_in))))
    bc = min(j for j in range(W) if any(bot_in[i][j] != bg for i in range(len(bot_in))))

    target_c = max(tc, bc)

    # Preserve the horizontal displacement already present in the two half silhouettes.
    top_shift = target_c - tc
    bot_shift = target_c - bc

    # If halves overlap horizontally, pull the upper half just enough to make a one-cell gap.
    if top and bot:
        top_cols = set(j for i, row in enumerate(top) for j, v in enumerate(row) if v != bg)
        bot_cols = set(j for i, row in enumerate(bot) for j, v in enumerate(row) if v != bg)
        if top_cols and bot_cols and max(bot_cols) >= min(top_cols):
            gap_shift = max(bot_cols) - min(top_cols) + 1
            if gap_shift > top_shift:
                top_shift = gap_shift

    place(top, rmin + tr0 - tr0, target_c + top_shift)
    place(bot, mid + br0 - br0, target_c + bot_shift)

    return out