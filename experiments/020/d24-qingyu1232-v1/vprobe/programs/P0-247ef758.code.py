def transform(grid):
    H = len(grid)
    W = len(grid[0])
    g = [row[:] for row in grid]

    # Find the vertical separator: a non-zero column that is uniform.
    sep = -1
    for x in range(W):
        col = [g[r][x] for r in range(H)]
        if all(v == col[0] for v in col) and col[0] != 0:
            sep = x
            break
    if sep < 0:
        return g

    # Identify rectangular frame rows/columns around the interior.
    top = -1
    bottom = -1
    for r in range(H):
        vals = set()
        for x in range(sep + 2, W):
            vals.add(g[r][x])
        if len(vals) > 1 and 0 not in vals:
            top = r
            break
    for r in range(H - 1, -1, -1):
        vals = set()
        for x in range(sep + 2, W):
            vals.add(g[r][x])
        if len(vals) > 1 and 0 not in vals:
            bottom = r
            break
    if top < 0 or bottom < 0:
        return g

    left = -1
    right = -1
    for x in range(sep + 2, W):
        vals = [g[r][x] for r in range(top, bottom + 1)]
        if all(v == vals[0] for v in vals) and vals[0] != 0:
            left = x
            break
    for x in range(W - 1, sep + 1, -1):
        vals = [g[r][x] for r in range(top, bottom + 1)]
        if all(v == vals[0] for v in vals) and vals[0] != 0:
            right = x
            break
    if left < 0 or right < 0:
        return g

    bg = g[top + 1][left + 1]

    def is_rect(cells, color):
        if not cells:
            return True
        rmin = min(r for r, c in cells)
        rmax = max(r for r, c in cells)
        cmin = min(c for r, c in cells)
        cmax = max(c for r, c in cells)
        return len(cells) == (rmax - rmin + 1) * (cmax - cmin + 1)

    # Extract templates from the left margin.
    templates = {}
    visited = [[False] * (sep + 1) for _ in range(H)]

    def cells_for_color(r, c, color):
        seen = {(r, c)}
        stack = [(r, c)]
        while stack:
            y, x = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx <= sep:
                    if not visited[ny][nx] and g[ny][nx] == color and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        visited[ny][nx] = True
                        stack.append((ny, nx))
        return seen

    for y in range(H):
        for x in range(sep + 1):
            if not visited[y][x] and g[y][x] != 0:
                color = g[y][x]
                cells = cells_for_color(y, x, color)
                rmin = min(r for r, _ in cells)
                cmin = min(c for _, c in cells)
                rel = frozenset((r - rmin, c - cmin) for r, c in cells)
                templates.setdefault(color, rel)

    def stamp(rel, center_r, center_c, color):
        for dr, dc in rel:
            r = center_r + dr
            c = center_c + dc
            if top < r < bottom and left < c < right:
                if g[r][c] == 0 or g[r][c] == bg:
                    g[r][c] = color

    # Apply templates at intersections of matching frame markers.
    top_marks = [c for c in range(left + 1, right) if g[top][c] != 0 and g[top][c] != bg]
    left_marks = [r for r in range(top + 1, bottom) if g[r][left] != 0 and g[r][left] != bg]
    right_marks = [r for r in range(top + 1, bottom) if g[r][right] != 0 and g[r][right] != bg]
    bottom_marks = [c for c in range(left + 1, right) if g[bottom][c] != 0 and g[bottom][c] != bg]

    for color, rel in templates.items():
        ts = [c for c in top_marks if g[top][c] == color]
        bs = [c for c in bottom_marks if g[bottom][c] == color]
        ls = [r for r in left_marks if g[r][left] == color]
        rs = [r for r in right_marks if g[r][right] == color]
        for r in ls:
            for c in ts:
                stamp(rel, r, c, color)
        for r in rs:
            for c in bs:
                stamp(rel, r, c, color)

    # Clear non-rectangular templates from the margin.
    for y in range(H):
        for x in range(sep + 1):
            if g[y][x] == 0:
                continue
            color = g[y][x]
            cells = cells_for_color(y, x, color)
            if not is_rect(cells, color):
                for r, c in cells:
                    g[r][c] = 0

    return g