def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    # Connected components of 1 (4-neighbor).
    seen = [[False] * w for _ in range(h)]
    comps = []
    for r in range(h):
        for c in range(w):
            if grid[r][c] == 1 and not seen[r][c]:
                stack = [(r, c)]
                seen[r][c] = True
                cells = []
                minr = maxr = r
                minc = maxc = c
                while stack:
                    x, y = stack.pop()
                    cells.append((x, y))
                    if x < minr:
                        minr = x
                    if x > maxr:
                        maxr = x
                    if y < minc:
                        minc = y
                    if y > maxc:
                        maxc = y
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if 0 <= nx < h and 0 <= ny < w and not seen[nx][ny] and grid[nx][ny] == 1:
                            seen[nx][ny] = True
                            stack.append((nx, ny))
                comps.append((cells, minr, maxr, minc, maxc))

    cset = set()
    for cells, *_ in comps:
        cset.update(cells)

    for cells, minr, maxr, minc, maxc in comps:
        filled = set(cells)
        holes = set()
        for r in range(minr, maxr + 1):
            for c in range(minc, maxc + 1):
                if (r, c) in filled:
                    continue
                # flood outside of bounding box from this cell
                start = (r, c)
                vis = {start}
                st = [start]
                escaped = False
                while st and not escaped:
                    x, y = st.pop()
                    if x == minr or x == maxr or y == minc or y == maxc:
                        escaped = True
                        break
                    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nx, ny = x + dx, y + dy
                        if (
                            minr <= nx <= maxr
                            and minc <= ny <= maxc
                            and (nx, ny) not in filled
                            and (nx, ny) not in vis
                        ):
                            vis.add((nx, ny))
                            st.append((nx, ny))
                if not escaped:
                    holes.update(vis)

        if holes:
            fill = 8
            hole = 6
        else:
            fill = 1
            hole = 1

        for r, c in cells:
            out[r][c] = fill
        for r, c in holes:
            out[r][c] = hole

        for (r, c), cell_set in zip(cells, cells):
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nr, nc = r + dx, c + dy
                    if 0 <= nr < h and 0 <= nc < w and out[nr][nc] == 4:
                        out[nr][nc] = 2

    return out