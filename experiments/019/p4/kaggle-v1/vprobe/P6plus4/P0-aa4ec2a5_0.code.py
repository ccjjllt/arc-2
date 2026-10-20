def transform(grid):
    H = len(grid)
    W = len(grid[0])
    g = [row[:] for row in grid]

    def is_fg(r, c):
        return 0 <= r < H and 0 <= c < W and g[r][c] == 1

    def is_bg(r, c):
        return 0 <= r < H and 0 <= c < W and g[r][c] == 4

    comps = []
    for r in range(H):
        for c in range(W):
            if g[r][c] == 1:
                stack = [(r, c)]
                g[r][c] = 1
                cells = []
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if is_fg(nx, ny):
                            g[nx][ny] = 0
                            stack.append((nx, ny))
                comps.append(cells)

    out = [row[:] for row in grid]

    for cells in comps:
        if not cells:
            continue
        comp_set = set(cells)
        hole_cells = []
        for r, c in comp_set:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 4 and (nr, nc) not in hole_cells:
                    start = (nr, nc)
                    stack = [start]
                    vis = {start}
                    hole = [start]
                    enclosed = True
                    while stack:
                        x, y = stack.pop()
                        for nx, ny in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                            if nx < 0 or ny < 0 or nx >= H or ny >= W:
                                enclosed = False
                                continue
                            if (nx, ny) in vis:
                                continue
                            if g[nx][ny] == 4:
                                vis.add((nx, ny))
                                hole.append((nx, ny))
                                stack.append((nx, ny))
                    if enclosed:
                        hole_cells.extend(hole)

        for r, c in cells:
            out[r][c] = 8 if hole_cells else 1
        for r, c in hole_cells:
            out[r][c] = 6

        border = set()
        for r, c in cells:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 4:
                    border.add((nr, nc))
        for r, c in border:
            if out[r][c] == 4:
                out[r][c] = 2

    return out