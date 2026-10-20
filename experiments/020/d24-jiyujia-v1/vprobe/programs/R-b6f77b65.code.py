def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def dr(c):
        return (c - 1) % 9

    def same(a, b):
        return a == b or dr(a) == b or dr(b) == a

    key = grid[0][0]
    out = [row[:] for row in grid]

    # collect non-key cells that may be affected
    candidates = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != 0 and not (r == 0 and c == 0):
                candidates[(r, c)] = v

    # find key cells that are part of a long vertical run (not isolated top-left)
    removable = set()
    for (r, c), v in list(candidates.items()):
        if v != key:
            continue
        # vertical run length at least 2
        rr = r
        while rr > 0 and grid[rr - 1][c] == key:
            rr -= 1
        re = r
        while re < H - 1 and grid[re + 1][c] == key:
            re += 1
        if re - rr + 1 >= 2:
            removable.add((r, c))
            removable.add((c, r))  # marker for column? not used

    # determine which columns are active based on horizontal key run above a long non-key run? 
    # We will compute for each column if top block falls.
    def can_drop_block(cells, d):
        # cells: set of (r,c), value same or same family
        if d == 0:
            return True
        for (r, c) in cells:
            nr = r + d
            if nr >= H:
                return False
            # destination empty in original unless occupied by a cell moving with this object?
            # For simplicity: destination must be 0 in original or part of this object shifted.
            if grid[nr][c] != 0:
                # allow if original dest is same value and part of source above? we handle by checking source cells
                pass
        return True

    # We'll process connected components of non-key cells, but treat components separately.
    # Build components by 4-neighbor with same color or compatible (dr).
    visited = set()
    comps = []
    for (r, c), v in candidates.items():
        if (r, c) in visited:
            continue
        # BFS
        stack = [(r, c)]
        comp = []
        compv = v
        visited.add((r, c))
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for (nr, nc) in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                    nv = grid[nr][nc]
                    if nv != 0 and nv not in removable and same(nv, compv):
                        visited.add((nr, nc))
                        stack.append((nr, nc))
        comps.append(comp)

    # For each component, try shifts downward that match demo behavior:
    # if all destination cells in original are 0 or in removable set, and no collision with other moving components.
    # Choose max d such that shape remains within grid and empty/removable dest.
    # But demos show only one specific d, often 2 or 3. We can infer from nearest obstacle below.
    out = [row[:] for row in grid]
    # Remove all key vertical-run cells (long runs) from output
    for (r, c) in removable:
        if r < H and c < W:
            out[r][c] = 0

    # For each component, compute max drop by scanning columns: for each cell, distance to first non-zero below that is not removable and not in same component.
    # Use original grid, but treat removable as empty.
    def drop_distance(comp):
        comp_set = set(comp)
        min_d = 10**9
        for (r, c) in comp:
            # scan down
            rr = r + 1
            while rr < H:
                if grid[rr][c] != 0 and (rr, c) not in comp_set and not (grid[rr][c] == key and (rr, c) in removable):
                    # obstacle
                    min_d = min(min_d, rr - r - 1)
                    break
                rr += 1
            else:
                min_d = min(min_d, H - 1 - r)
        return min_d

    moved = []
    for comp in comps:
        d = drop_distance(comp)
        # Demos: components move only if d >= 1? They sometimes move 2/3. Use d.
        if d > 0:
            for (r, c) in comp:
                out[r + d][c] = grid[r][c]
            for (r, c) in comp:
                out[r][c] = 0
            moved.append(comp)

    # restore top-left key
    out[0][0] = key

    return out