def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def find_bg():
        cnt = {}
        for row in grid:
            for v in row:
                cnt[v] = cnt.get(v, 0) + 1
        return max(sorted(cnt.keys()), key=lambda c: (cnt[c], -c))

    bg = find_bg()

    def find_wall():
        cands = []
        for c in range(10):
            if c == bg:
                continue
            for r in range(H - 2):
                for cc in range(W - 2):
                    if all(grid[r + dr][cc + dc] == c for dr in range(3) for dc in range(3)):
                        cands.append(c)
                        break
                else:
                    continue
                break
        return cands[0] if cands else None

    wall = find_wall()
    if wall is None:
        return [row[:] for row in grid]

    others = [c for c in set(v for row in grid for v in row) if c != bg and c != wall]
    if len(others) >= 2:
        target1, target2 = sorted(others)[:2]
    elif len(others) == 1:
        target1 = target2 = others[0]
    else:
        target1 = target2 = 0

    def comp_of(start):
        sr, sc = start
        stack = [(sr, sc)]
        seen = {(sr, sc)}
        while stack:
            r, c = stack.pop()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in seen and grid[nr][nc] == wall:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        return seen

    comps = []
    seen_cells = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == wall and (r, c) not in seen_cells:
                comp = comp_of((r, c))
                seen_cells.update(comp)
                comps.append(comp)

    center = None
    for r in range(H - 2):
        for c in range(W - 2):
            if all(grid[r + dr][c + dc] == wall for dr in range(3) for dc in range(3)):
                center = (r + 1, c + 1)
                break
        if center is not None:
            break

    center_comp = None
    for comp in comps:
        if center in comp:
            center_comp = comp
            break

    def neighbor_colors(comp):
        s = set()
        for r, c in comp:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    v = grid[nr][nc]
                    if v in others:
                        s.add(v)
        return s

    target_for = {}
    if center_comp is not None:
        nc = neighbor_colors(center_comp)
        if len(nc) >= 2:
            a, b = sorted(nc)[:2]
            target_for[target1] = a
            target_for[target2] = b
        elif len(nc) == 1:
            target_for[target1] = target_for[target2] = list(nc)[0]

    for t in (target1, target2):
        if t not in target_for:
            target_for[t] = t

    out = [row[:] for row in grid]

    for comp in comps:
        if center_comp is not None and comp is center_comp:
            for r, c in comp:
                if abs(r - center[0]) <= 1 and abs(c - center[1]) <= 1:
                    continue
                out[r][c] = target_for[target1] if out[r][c] == wall and (r, c) in comp else out[r][c]
            out[center[0]][center[1]] = target_for[target2]
            continue

        adj = neighbor_colors(comp)
        if not adj:
            continue
        if len(adj) >= 2:
            a, b = sorted(adj)[:2]
            vals = {a: 0, b: 0}
            for r, c in comp:
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] in vals:
                        vals[grid[nr][nc]] += 1
            if vals[a] == vals[b]:
                left = sum(1 for r, c in comp if c < center[1]) if center else 0
                right = sum(1 for r, c in comp if c >= center[1]) if center else 1
                pick = a if left > right else b
            else:
                pick = a if vals[a] > vals[b] else b
        else:
            pick = list(adj)[0]
        pick = target_for.get(pick, pick)
        for r, c in comp:
            out[r][c] = pick

    if center is not None:
        for dr in range(-1, 2):
            for dc in range(-1, 2):
                if dr == 0 and dc == 0:
                    continue
                r, c = center[0] + dr, center[1] + dc
                if 0 <= r < H and 0 <= c < W and grid[r][c] == wall:
                    out[r][c] = wall

    return out