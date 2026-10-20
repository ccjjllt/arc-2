def transform(grid):
    H = len(grid)
    W = len(grid[0])
    top = [row[:] for row in grid[:6]]
    out = [row[:] for row in grid[6:]]
    h = len(out)
    if h == 0:
        return out

    # Background is the most common non-8 value in the lower field.
    counts = {}
    for row in out:
        for v in row:
            if v != 8:
                counts[v] = counts.get(v, 0) + 1
    bg = 0
    best = -1
    for v, n in counts.items():
        if n > best:
            best = n
            bg = v

    # Find the 2x2 8 marker.
    mx = my = None
    for r in range(h - 1):
        for c in range(W - 1):
            if out[r][c] == 8 and out[r][c + 1] == 8 and out[r + 1][c] == 8 and out[r + 1][c + 1] == 8:
                mx, my = c, r
                break
        if mx is not None:
            break

    # Parse header boxes. A side is left if the digit is in the left half of the box.
    header = []
    box_cols = [(1, 5), (7, 11), (13, 17), (19, 23)]
    for c0, c1 in box_cols:
        digit = None
        side = 0
        for x in range(c0, c1):
            for y in range(4):
                v = top[y][x]
                if v not in (0, 5):
                    digit = v
                    side = 0 if x < (c0 + c1) // 2 else 1
                    break
            if digit is not None:
                break
        if digit is not None:
            header.append((digit, side))

    # Find rectangular blocks of colors that are neither background nor 8.
    blocks = []
    used = [[False] * W for _ in range(h)]
    for r in range(h):
        for c in range(W):
            if used[r][c] or out[r][c] in (bg, 8):
                continue
            v = out[r][c]
            # Horizontal span in starting row.
            c2 = c
            while c2 + 1 < W and out[r][c2 + 1] == v:
                c2 += 1
            if c2 <= c:
                continue
            # Vertical span as long as full row segment remains v.
            r2 = r
            ok = True
            while r2 + 1 < h:
                for x in range(c, c2 + 1):
                    if out[r2 + 1][x] != v:
                        ok = False
                        break
                if not ok:
                    break
                r2 += 1
            if r2 <= r:
                continue
            for rr in range(r, r2 + 1):
                for cc in range(c, c2 + 1):
                    used[rr][cc] = True
            blocks.append((v, r, c, r2, c2))

    # Match header colors to lower blocks.
    matched = []
    for v, side in header:
        same = [b for b in blocks if b[0] == v]
        if not same:
            continue
        # Prefer blocks vertically closest to the marker; then horizontally closest.
        def key(b):
            vdist = min(abs(b[1] - my), abs(b[3] - (my + 1)))
            hd = (b[2] + b[4]) // 2
            hdist = abs(hd - mx)
            return (vdist, hdist)
        same.sort(key=key)
        b = same[0]
        matched.append((b, side))

    if mx is None or not matched:
        return out

    # Determine the bounding rectangle of the marker and all matched blocks.
    top_row = min([my] + [b[1] for b, _ in matched] + [b[3] for b, _ in matched])
    bot_row = max([my + 1] + [b[1] for b, _ in matched] + [b[3] for b, _ in matched])
    left_col = min([mx] + [b[2] for b, _ in matched] + [b[4] for b, _ in matched])
    right_col = max([mx + 1] + [b[2] for b, _ in matched] + [b[4] for b, _ in matched])

    def paint(r0, r1, c0, c1):
        if r0 < 0:
            r0 = 0
        if c0 < 0:
            c0 = 0
        if r1 > h - 1:
            r1 = h - 1
        if c1 > W - 1:
            c1 = W - 1
        if r0 > r1 or c0 > c1:
            return
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if out[r][c] == bg:
                    out[r][c] = 8

    # Draw the outer frame implied by all matched blocks.
    paint(top_row, top_row, left_col, right_col)
    paint(bot_row, bot_row, left_col, right_col)
    paint(top_row, bot_row, left_col, right_col)

    # Draw the central horizontal and vertical arms through the marker.
    paint(my, my + 1, left_col, right_col)
    paint(top_row, bot_row, mx, mx + 1)

    # For each matched block, add the short side arm shown in the demonstrations.
    for (bv, br0, bc0, br1, bc1), side in matched:
        if side == 0:
            # Block is associated with its left side; connect marker vertically to the block's left edge.
            if br1 < my:
                paint(br1, my - 1, mx, mx + 1)
                paint(br1, br1, min(bc0, mx), max(bc0, mx + 1))
            elif br0 > my + 1:
                paint(my + 2, br0, mx, mx + 1)
                paint(br0, br0, min(bc0, mx), max(bc0, mx + 1))
            else:
                paint(max(br0, my), min(br1, my + 1), min(bc0, mx), max(bc0, mx + 1))
        else:
            # Block is associated with its right side; connect marker vertically to the block's right edge.
            if br1 < my:
                paint(br1, my - 1, mx, mx + 1)
                paint(br1, br1, min(bc1, mx + 1), max(bc1, mx + 1))
            elif br0 > my + 1:
                paint(my + 2, br0, mx, mx + 1)
                paint(br0, br0, min(bc1, mx + 1), max(bc1, mx + 1))
            else:
                paint(max(br0, my), min(br1, my + 1), min(bc1, mx + 1), max(bc1, mx + 1))

    return out