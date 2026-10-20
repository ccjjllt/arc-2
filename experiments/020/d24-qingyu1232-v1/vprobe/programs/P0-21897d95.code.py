def transform(grid):
    H = len(grid)
    W = len(grid[0])
    N = H * W
    # connected components of non-1 cells, 4-connected
    comp = [[-1] * W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] != 1 and comp[r][c] == -1:
                color = grid[r][c]
                mask = [[False] * W for _ in range(H)]
                q = [(r, c)]
                comp[r][c] = len(comps)
                mask[r][c] = True
                s = 1
                head = 0
                while head < len(q):
                    cr, cc = q[head]
                    head += 1
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < H and 0 <= nc < W:
                            if grid[nr][nc] == color and not mask[nr][nc]:
                                mask[nr][nc] = True
                                comp[nr][nc] = len(comps)
                                q.append((nr, nc))
                                s += 1
                comps.append((color, s, mask))
    # components of 1-cells; use only components that touch at least two other components
    onecomps = []
    seen = [[False] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 1 and not seen[r][c]:
                q = [(r, c)]
                seen[r][c] = True
                head = 0
                neigh = set()
                while head < len(q):
                    cr, cc = q[head]
                    head += 1
                    for nr, nc in ((cr + 1, cc), (cr - 1, cc), (cr, cc + 1), (cr, cc - 1)):
                        if 0 <= nr < H and 0 <= nc < W:
                            if grid[nr][nc] == 1:
                                if not seen[nr][nc]:
                                    seen[nr][nc] = True
                                    q.append((nr, nc))
                            elif comp[nr][nc] != -1:
                                neigh.add(comp[nr][nc])
                if len(neigh) >= 2:
                    onecomps.append(neigh)

    edges = {}
    for neigh in onecomps:
        lst = sorted(neigh)
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                edges[(lst[i], lst[j])] = edges.get((lst[i], lst[j]), 0) + 1

    # build adjacency
    adj = {i: set() for i in range(len(comps))}
    for (a, b), _ in edges.items():
        if a in adj and b in adj:
            adj[a].add(b)
            adj[b].add(a)

    # root component: minimum degree then smallest area
    root = 0
    best = (10**9, 10**9)
    for i in range(len(comps)):
        deg = len(adj[i])
        area = comps[i][1]
        if (deg, area) < best:
            best = (deg, area)
            root = i

    order = [root]
    seen = {root}
    while len(order) < len(comps):
        nxt = None
        for i in order:
            for nb in sorted(adj[i]):
                if nb not in seen:
                    if nxt is None or (len(adj[nb]), comps[nb][1]) > (len(adj[i]), comps[i][1]):
                        nxt = nb
            if nxt is not None:
                break
        if nxt is None:
            # fallback to any unvisited
            for i in range(len(comps)):
                if i not in seen:
                    nxt = i
                    break
        order.append(nxt)
        seen.add(nxt)

    out = [[0] * H for _ in range(W)]
    for out_r, comp_i in enumerate(order):
        color = comps[comp_i][0]
        for out_c in range(H):
            mask = comps[comp_i][2]
            val = 1 if mask[out_c][out_r] else 0
            out[out_r][out_c] = color if val else 1
    # If any cell remains 1, fill with background/root color (robustness)
    bg = comps[root][0]
    for r in range(W):
        for c in range(H):
            if out[r][c] == 1:
                out[r][c] = bg
    return out