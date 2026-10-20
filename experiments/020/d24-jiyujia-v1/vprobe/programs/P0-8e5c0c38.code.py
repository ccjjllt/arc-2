def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    seen = [[False] * W for _ in range(H)]
    out = [row[:] for row in grid]
    dirs = ((1, 0), (-1, 0), (0, 1), (0, -1))

    for r in range(H):
        for c in range(W):
            if seen[r][c] or grid[r][c] == bg:
                continue
            color = grid[r][c]
            stack = [(r, c)]
            seen[r][c] = True
            cells = []
            min_r = max_r = r
            min_c = max_c = c
            while stack:
                y, x = stack.pop()
                cells.append((y, x))
                if y < min_r:
                    min_r = y
                if y > max_r:
                    max_r = y
                if x < min_c:
                    min_c = x
                if x > max_c:
                    max_c = x
                for dy, dx in dirs:
                    ny, nx = y + dy, x + dx
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny][nx] and grid[ny][nx] == color:
                        seen[ny][nx] = True
                        stack.append((ny, nx))

            if min_c == max_c:
                continue

            keep = set()
            for y, x in cells:
                mx = min_c + max_c - x
                if min_c <= mx <= max_c and grid[y][mx] == color:
                    keep.add((y, x))
                    keep.add((y, mx))

            for y, x in cells:
                if (y, x) not in keep:
                    out[y][x] = bg
    return out