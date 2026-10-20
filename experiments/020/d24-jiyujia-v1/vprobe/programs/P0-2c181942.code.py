def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = 8
    cells = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                cells.setdefault(v, []).append((r, c))

    anchors = [
        (2, 1, -5),
        (3, 0, -5),
        (4, 1, 0),
        (6, 0, -5),
        (7, 1, 5),
        (1, 1, 5),
    ]

    out = [row[:] for row in grid]
    for r in range(H):
        for c in range(W):
            out[r][c] = bg

    def paint(offset, cells):
        if not cells:
            return
        minr = min(r for r, _ in cells)
        minc = min(c for _, c in cells)
        for r, c in cells:
            rr = offset[0] + (r - minr)
            cc = offset[1] + (c - minc)
            if 0 <= rr < H and 0 <= cc < W:
                out[rr][cc] = cells[0][2]

    for color, anchor_row_off, anchor_col_off in anchors:
        if color not in cells:
            continue
        group = cells[color]
        minr = min(r for r, _ in group)
        minc = min(c for _, c in group)
        # anchor at the top-left of the color's current bounding box
        paint((minr + anchor_row_off, minc + anchor_col_off), group)

    return out