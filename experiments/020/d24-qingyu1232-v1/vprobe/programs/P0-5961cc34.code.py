def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = grid[0][0]
    out = [row[:] for row in grid]

    seed_cells = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 4 or grid[r][c] == 2:
                seed_cells.append((r, c))
    if not seed_cells:
        return out

    comp_seed = []
    for r, c in seed_cells:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 2:
                comp_seed.append((nr, nc))

    if comp_seed:
        # Primary direction is the direction from the 4 to the 2-segment if visible.
        dir_set = {}
        for r, c in comp_seed:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 4:
                    dir_set[(dr, dc)] = dir_set.get((dr, dc), 0) + 1
        if dir_set:
            primary = max(sorted(dir_set.items()), key=lambda kv: kv[1])[0]
        else:
            rs = [r for r, _ in comp_seed]
            cs = [c for _, c in comp_seed]
            if max(rs) - min(rs) >= max(cs) - min(cs):
                primary = (1, 0) if comp_seed[-1][0] >= comp_seed[0][0] else (-1, 0)
            else:
                primary = (0, 1) if comp_seed[-1][1] >= comp_seed[0][1] else (0, -1)

        for r, c in comp_seed:
            d = primary
            while True:
                nr, nc = r + d[0], c + d[1]
                if not (0 <= nr < H and 0 <= nc < W):
                    break
                val = grid[nr][nc]
                if val == 1 or val == 3:
                    r, c = nr, nc
                    continue
                out[nr][nc] = 2
                r, c = nr, nc
    else:
        # No 2 adjacent to 4: fall back to the 4 alone and guess vertical if possible.
        for r, c in seed_cells:
            if grid[r][c] == 4:
                for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + d[0], c + d[1]
                    if 0 <= nr < H and 0 <= nc < W:
                        while 0 <= nr < H and 0 <= nc < W:
                            val = grid[nr][nc]
                            if val == 1 or val == 3:
                                nr, nc = nr + d[0], nc + d[1]
                                continue
                            out[nr][nc] = 2
                            nr, nc = nr + d[0], nc + d[1]
                break

    # Flood-fill through active 1/3 cells, expanding in the direction implied by each 3 cluster.
    active = set()
    for r, c in seed_cells:
        active.add((r, c))
    for r in range(H):
        for c in range(W):
            if grid[r][c] in (1, 3) and out[r][c] == 2:
                active.add((r, c))

    def add_active(r, c):
        if 0 <= r < H and 0 <= c < W and grid[r][c] in (1, 3):
            if (r, c) not in active:
                active.add((r, c))
                out[r][c] = 2
                return True
        return False

    # Initial closure: from all active cells, grow through orthogonally adjacent 1/3 cells.
    changed = True
    while changed:
        changed = False
        for r, c in list(active):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if add_active(r + dr, c + dc):
                    changed = True

    # Determine expansion directions for 3 clusters.
    threes = [(r, c) for r in range(H) for c in range(W)
              if (r, c) in active and grid[r][c] == 3]
    dirs = set()
    for r, c in threes:
        neigh = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 1:
                neigh.append((nr, nc))
        if neigh:
            ar = sum(x[0] for x in neigh) / len(neigh)
            ac = sum(x[1] for x in neigh) / len(neigh)
            dr = ar - r
            dc = ac - c
            if abs(dr) >= abs(dc) and dr != 0:
                dirs.add((1 if dr > 0 else -1, 0))
            elif dc != 0:
                dirs.add((0, 1 if dc > 0 else -1))

    # For each active 3, expand along the local direction toward adjacent 1s.
    for r, c in threes:
        neigh = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and grid[nr][nc] == 1:
                neigh.append((nr, nc))
        if not neigh:
            continue
        ar = sum(x[0] for x in neigh) / len(neigh)
        ac = sum(x[1] for x in neigh) / len(neigh)
        dr = ar - r
        dc = ac - c
        if abs(dr) >= abs(dc) and dr != 0:
            d = (1 if dr > 0 else -1, 0)
        elif dc != 0:
            d = (0, 1 if dc > 0 else -1)
        else:
            continue

        # Expand all contiguous cells of the same 3 line/edge in this direction.
        line = [(r, c)]
        # Gather other 3s in same row/col near this 3 that point same way.
        if d[0] == 0:
            for c2 in range(c):
                if grid[r][c2] == 3:
                    line.append((r, c2))
                else:
                    break
            for c2 in range(c + 1, W):
                if grid[r][c2] == 3:
                    line.append((r, c2))
                else:
                    break
        else:
            for r2 in range(r):
                if grid[r2][c] == 3:
                    line.append((r2, c))
                else:
                    break
            for r2 in range(r + 1, H):
                if grid[r2][c] == 3:
                    line.append((r2, c))
                else:
                    break

        for sr, sc in line:
            # The expansion starts just beyond the 3 in the direction.
            rr, cc = sr + d[0], sc + d[1]
            while 0 <= rr < H and 0 <= cc < W:
                val = grid[rr][cc]
                if val == 1 or val == 3:
                    add_active(rr, cc)
                    rr, cc = rr + d[0], cc + d[1]
                    continue
                out[rr][cc] = 2
                if (rr, cc) in active:
                    break
                rr, cc = rr + d[0], cc + d[1]

    # Final pass: convert every connected 1/3 component that touches any active 2 to 2.
    changed = True
    while changed:
        changed = False
        for r, c in list(active):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if add_active(r + dr, c + dc):
                    changed = True

    # Clean: keep only 8 and 2; active/seed and drawn lines are already 2.
    for r in range(H):
        for c in range(W):
            if out[r][c] not in (2, bg):
                out[r][c] = bg
    return out