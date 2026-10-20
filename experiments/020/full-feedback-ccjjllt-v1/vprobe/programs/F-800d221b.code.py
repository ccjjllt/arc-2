def transform(grid):
    h = len(grid)
    w = len(grid[0])

    # Identify the two non-wall colors: they are colors that appear outside
    # the connected wall structure and are adjacent to the wall.
    colors = set()
    for row in grid:
        for v in row:
            colors.add(v)

    wall = None
    seeds = None
    for cand in sorted(colors):
        # Try cand as wall. Find non-wall colors adjacent to it.
        adj = set()
        for r in range(h):
            for c in range(w):
                if grid[r][c] == cand:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != cand:
                                adj.add(grid[rr][cc])
        if len(adj) == 2:
            wall = cand
            seeds = tuple(sorted(adj))
            break

    if wall is None:
        # Fallback: wall is the most frequent non-background color.
        cnt = {}
        for row in grid:
            for v in row:
                cnt[v] = cnt.get(v, 0) + 1
        wall = max(cnt, key=lambda x: cnt[x])
        adj = set()
        for r in range(h):
            for c in range(w):
                if grid[r][c] == wall:
                    for dr in (-1, 0, 1):
                        for dc in (-1, 0, 1):
                            rr, cc = r + dr, c + dc
                            if 0 <= rr < h and 0 <= cc < w and grid[rr][cc] != wall:
                                adj.add(grid[rr][cc])
        seeds = tuple(sorted(adj))[:2]

    if len(seeds) != 2:
        return [row[:] for row in grid]

    a, b = seeds
    out = [row[:] for row in grid]

    # For each wall cell, decide whether it should be painted with a or b.
    # A wall cell is painted if it is not part of a solid 3x3 wall block.
    # The color is chosen by 8-connected reachability to that seed through wall
    # cells after removing the centers of solid 3x3 blocks.
    solid_center = [[False] * w for _ in range(h)]
    for r in range(h - 2):
        for c in range(w - 2):
            ok = True
            for dr in range(3):
                for dc in range(3):
                    if grid[r + dr][c + dc] != wall:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                solid_center[r + 1][c + 1] = True

    # Paint every wall cell that is not a solid 3x3 center.
    # Use BFS from each seed over cells that are either seed or paintable wall.
    for target, seed in ((a, a), (b, b)):
        seen = [[False] * w for _ in range(h)]
        q = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == seed:
                    seen[r][c] = True
                    q.append((r, c))
        head = 0
        while head < len(q):
            r, c = q[head]
            head += 1
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < h and 0 <= cc < w and not seen[rr][cc]:
                        if grid[rr][cc] == seed:
                            seen[rr][cc] = True
                            q.append((rr, cc))
                        elif grid[rr][cc] == wall and not solid_center[rr][cc]:
                            seen[rr][cc] = True
                            out[rr][cc] = target
                            q.append((rr, cc))
    return out