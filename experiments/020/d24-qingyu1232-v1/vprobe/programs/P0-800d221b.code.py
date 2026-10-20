def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1

    # The wall is the non-background color whose removal leaves exactly two
    # significant non-wall colors that appear as visible clusters.
    wall = None
    pair = None
    for w, cw in counts.items():
        others = [c for c, n in counts.items() if c != w and n >= 4]
        if len(others) == 2:
            wall = w
            pair = tuple(sorted(others))
            break

    if wall is None:
        # Fallback: least frequent color with at least 4 occurrences.
        candidates = [c for c, n in counts.items() if n >= 4]
        wall = min(candidates, key=lambda c: counts[c])
        pair = tuple(sorted(c for c in counts if c != wall and counts[c] >= 2))

    a, b = pair
    out = [row[:] for row in grid]

    # BFS over wall pixels from those adjacent to the two target colors.
    best = [[None] * W for _ in range(H)]
    q = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != wall:
                continue
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        if grid[rr][cc] == a:
                            if best[r][c] is None:
                                best[r][c] = a
                                q.append((r, c))
                        elif grid[rr][cc] == b:
                            if best[r][c] is None:
                                best[r][c] = b
                                q.append((r, c))

    head = 0
    while head < len(q):
        r, c = q[head]
        head += 1
        cur = best[r][c]
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr, cc = r + dr, c + dc
            if 0 <= rr < H and 0 <= cc < W and grid[rr][cc] == wall:
                if best[rr][cc] is None:
                    best[rr][cc] = cur
                    q.append((rr, cc))

    for r in range(H):
        for c in range(W):
            if grid[r][c] == wall and best[r][c] is not None:
                out[r][c] = best[r][c]

    return out