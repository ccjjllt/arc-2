def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Find the vertical separator line of 5s.
    sep = None
    for c in range(W):
        if all(grid[r][c] == 5 for r in range(H)):
            sep = c
            break

    if sep is None:
        return [row[:] for row in grid]

    # Find horizontal separator lines of 5s (if present).
    bands = []
    start = 0
    for r in range(H):
        if all(grid[r][c] == 5 for c in range(W)):
            if start < r:
                bands.append((start, r - 1))
            start = r + 1
    if start < H:
        bands.append((start, H - 1))

    # Collect connected components of 0s inside each horizontal band,
    # on the left of the vertical separator.
    used = [[False] * W for _ in range(H)]
    components = []
    for r0, r1 in bands:
        for c in range(sep):
            for r in range(r0, r1 + 1):
                if grid[r][c] == 0 and not used[r][c]:
                    stack = [(r, c)]
                    used[r][c] = True
                    cells = []
                    while stack:
                        x, y = stack.pop()
                        cells.append((x, y))
                        for nx, ny in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                            if 0 <= nx <= r1 and 0 <= ny < sep:
                                if not used[nx][ny] and grid[nx][ny] == 0:
                                    used[nx][ny] = True
                                    stack.append((nx, ny))
                    components.append(cells)

    # Work left-to-right over components.
    result = [row[:] for row in grid]

    # Remove all original 0 cells from result; they will be redrawn shifted.
    for cells in components:
        for r, c in cells:
            result[r][c] = 6

    prev_max = 0
    for cells in sorted(components, key=lambda cs: min(c for _, c in cs)):
        min_r = min(r for r, _ in cells)
        max_r = max(r for r, _ in cells)
        min_c = min(c for _, c in cells)
        max_c = max(c for _, c in cells)
        rows = range(min_r, max_r + 1)

        # Choose the smallest right shift that does not overlap
        # the previous component's shifted rightmost column.
        shift = max(0, prev_max - (max_c + 1) + 1)
        # Also avoid crossing the separator.
        shift = min(shift, sep - 1 - max_c)
        if shift < 0:
            shift = 0

        # Track which rows in this component have a zero at its left edge.
        left_edge_rows = set()
        for r, c in cells:
            nc = c + shift
            result[r][nc] = 0
            if c == min_c:
                left_edge_rows.add(r)

        prev_max = max_c + shift

        # If the component's left-edge rows do not cover every row of the
        # component, paint the right side of the separator with 2s on the
        # missing rows.
        if len(left_edge_rows) < (max_r - min_r + 1):
            for r in rows:
                if r not in left_edge_rows:
                    for c in range(sep + 1, W):
                        if result[r][c] == 6:
                            result[r][c] = 2

    return result