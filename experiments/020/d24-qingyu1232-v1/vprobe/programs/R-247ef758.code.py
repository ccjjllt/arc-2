def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Find the vertical separator: a column (not first/last) whose every
    # cell has the same non-zero value.
    sep = None
    for c in range(1, W - 1):
        v = grid[0][c]
        if v != 0 and all(grid[r][c] == v for r in range(H)):
            sep = c
            break
    if sep is None:
        return [row[:] for row in grid]

    m = min(3, sep)
    inner_top = 0
    inner_bot = H
    left_cols = range(m)
    top_row = 0
    bot_row = H - 1

    # The marker/background column just right of the separator.
    marker_col = sep + 1
    if marker_col >= W:
        marker_col = W - 1
    bg = grid[inner_top + 1][marker_col] if inner_top + 1 < inner_bot else 0

    # Top row special positions: non-background non-zero entries in the
    # top border between the marker column and the right border.
    top_cols = {}
    for c in range(marker_col, min(W, sep + 2 * m + 1)):
        v = grid[top_row][c]
        if v not in (0, bg):
            top_cols[v] = c

    # Extract connected components from the left strip.
    visited = set()
    objects = []
    for sr in range(H):
        for sc in left_cols:
            if (sr, sc) in visited:
                continue
            v = grid[sr][sc]
            if v == 0:
                visited.add((sr, sc))
                continue
            comp = []
            stack = [(sr, sc)]
            visited.add((sr, sc))
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if (
                        nr < H
                        and nc >= 0
                        and nc < sep
                        and grid[nr][nc] == v
                        and (nr, nc) not in visited
                    ):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
            rows = [r for r, _ in comp]
            cols = [c for _, c in comp]
            rmin, rmax = min(rows), max(rows)
            cmin, cmax = min(cols), max(cols)
            # The canonical center: the middle row of the bounding box,
            # and for that row the leftmost cell of the object.
            center_row = (rmin + rmax) // 2
            center_col = cmin
            for r, c in comp:
                if r == center_row and c == center_col:
                    break
            objects.append((v, center_row, center_col, comp))

    # Markers in the marker column: non-background non-zero values,
    # paired with the same value appearing in the top border.
    markers = []
    for r in range(inner_top + 1, inner_bot):
        v = grid[r][marker_col]
        if v not in (0, bg) and v in top_cols:
            markers.append((r, v))

    out = [row[:] for row in grid]

    # Clear objects that are actually stamped somewhere.
    used = set()
    for r, v in markers:
        for v2, cr, cc, comp in objects:
            if v2 == v:
                used.add((v, cr, cc))

    for v, cr, cc, comp in objects:
        if (v, cr, cc) in used:
            for r, c in comp:
                out[r][c] = 0

    # Stamp objects for every marker and every matching top position,
    # from lower markers upward so upper markers can overwrite overlaps.
    for r, v in sorted(markers, key=lambda x: -x[0]):
        ctop = top_cols[v]
        for v2, cr, cc, comp in objects:
            if v2 != v:
                continue
            for rr, cc_off in ((r - cr, ctop - cc) for _ in (0,)):
                for sr, sc in comp:
                    nr, nc = rr + sr, cc_off + sc
                    if inner_top <= nr < inner_bot and 0 <= nc < W:
                        out[nr][nc] = v2

    return out