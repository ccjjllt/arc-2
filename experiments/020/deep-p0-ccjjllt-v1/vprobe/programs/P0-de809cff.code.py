def transform(grid):
    if not grid or not grid[0]:
        return []
    h = len(grid)
    w = len(grid[0])

    def same_nonzero_neighbors(r, c):
        counts = {}
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    v = grid[rr][cc]
                    if v != 0:
                        counts[v] = counts.get(v, 0) + 1
        if not counts:
            return 0, 0
        best = max(counts, key=lambda k: counts[k])
        return counts[best], best

    centers = []
    out = [row[:] for row in grid]

    for r in range(h):
        for c in range(w):
            if grid[r][c] != 0:
                continue
            cnt, col = same_nonzero_neighbors(r, c)
            if cnt >= 5:
                centers.append((r, c, col))
                out[r][c] = 8

    for r, c, col in centers:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    if out[rr][cc] != 8:
                        out[rr][cc] = 2 if col == 1 else 1 if col == 2 else 1 if col == 4 else 4 if col == 3 else 0

    return out