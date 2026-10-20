def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def background(g):
        counts = {}
        for row in g:
            for v in row:
                counts[v] = counts.get(v, 0) + 1
        return max(sorted(counts), key=lambda c: counts[c])

    def clean(color, cells, bg):
        cells = list(cells)
        if not cells:
            return []
        s = 1
        while True:
            changed = False
            for r, c in cells:
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == bg:
                            # Count 4-neighbours of same color within the candidate shape.
                            n = 0
                            for a, b in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                                if (r + a, c + b) in cells:
                                    n += 1
                            if n == 0:
                                cells.remove((r, c))
                                changed = True
                                break
                    if changed:
                        break
                if changed:
                    break
            s += 1
            if not changed or s > 60:
                break
        return cells

    bg = background(grid)
    out = [row[:] for row in grid]

    for color in set(v for row in grid for v in row):
        if color == bg:
            continue
        cells = set((r, c) for r in range(H) for c in range(W) if grid[r][c] == color)
        for r, c in cells:
            out[r][c] = bg
        for r, c in clean(color, cells, bg):
            out[r][c] = color
    return out