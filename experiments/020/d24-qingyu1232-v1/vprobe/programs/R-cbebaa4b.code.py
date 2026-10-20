def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def bbox(color):
        rows = [r for r in range(H) for c in range(W) if grid[r][c] == color]
        cols = [c for r in range(H) for c in range(W) if grid[r][c] == color]
        if not rows:
            return None
        return (min(rows), min(cols), max(rows), max(cols))

    main_colors = []
    for col in range(1, 10):
        if col == 2:
            continue
        b = bbox(col)
        if b:
            main_colors.append((col, b))

    def piece_points(col, b):
        r0, c0, r1, c1 = b
        pts = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if grid[r][c] in (col, 2):
                    val = 2 if grid[r][c] == 2 else col
                    pts.append((r - r0, c - c0, val))
        return pts

    pieces = []
    for col, b in main_colors:
        pts = piece_points(col, b)
        h = max(p[0] for p in pts) + 1
        w = max(p[1] for p in pts) + 1
        pieces.append((col, pts, h, w))

    pieces.sort(key=lambda x: (x[0], -x[2], -x[3]))

    total_w = sum(p[3] for p in pieces)
    total_h = max(p[2] for p in pieces)
    if total_w > W or total_h > H:
        return [row[:] for row in grid]

    start_c = (W - total_w) // 2
    start_r = (H - total_h) // 2
    out = [[0] * W for _ in range(H)]

    cur_c = start_c
    for col, pts, h, w in pieces:
        top = start_r + (total_h - h) // 2
        for dr, dc, val in pts:
            r = top + dr
            c = cur_c + dc
            if 0 <= r < H and 0 <= c < W:
                if out[r][c] == 0:
                    out[r][c] = val
                elif out[r][c] != val:
                    out[r][c] = val
        cur_c += w

    return out