def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def out_of(r, c):
        return r < 0 or r >= H or c < 0 or c >= W

    # Find the 4 marker and the adjacent 2 cells.
    fr = fc = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 4:
                fr, fc = r, c
                break
        if fr is not None:
            break

    # Beam direction is away from the attached 2 segment.
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    direction = None
    for dr, dc in dirs:
        nr, nc = fr + dr, fc + dc
        if not out_of(nr, nc) and grid[nr][nc] == 2:
            direction = (dr, dc)
            break
    if direction is None:
        direction = (-1, 0)

    # Label connected components of 1 or 3.
    comp = [[-1] * W for _ in range(H)]
    comps = []
    for r in range(H):
        for c in range(W):
            if grid[r][c] in (1, 3) and comp[r][c] == -1:
                cid = len(comps)
                cells = []
                stack = [(r, c)]
                comp[r][c] = cid
                while stack:
                    y, x = stack.pop()
                    cells.append((y, x))
                    for dy, dx in dirs:
                        ny, nx = y + dy, x + dx
                        if (not out_of(ny, nx) and comp[ny][nx] == -1
                                and grid[ny][nx] in (1, 3)):
                            comp[ny][nx] = cid
                            stack.append((ny, nx))
                comp_id = [cid] * len(cells)
                has1 = any(grid[y][x] == 1 for y, x in cells)
                has3 = any(grid[y][x] == 3 for y, x in cells)
                ys = [y for y, _ in cells]
                xs = [x for _, x in cells]
                comps.append({
                    "cells": cells,
                    "has1": has1,
                    "has3": has3,
                    "minr": min(ys),
                    "maxr": max(ys),
                    "minc": min(xs),
                    "maxc": max(xs),
                })

    # Find the side where a 3 sits on the component boundary.
    def side_for(comp_id):
        cells = comps[comp_id]["cells"]
        s3 = set(cells)
        for y, x in cells:
            if grid[y][x] != 3:
                continue
            if y == comps[comp_id]["minr"] and (y - 1, x) not in s3:
                return (0, 1)
            if y == comps[comp_id]["maxr"] and (y + 1, x) not in s3:
                return (0, -1)
            if x == comps[comp_id]["minc"] and (y, x - 1) not in s3:
                return (1, 1)
            if x == comps[comp_id]["maxc"] and (y, x + 1) not in s3:
                return (1, -1)
        return None

    side_dir = [None] * len(comps)
    for cid in range(len(comps)):
        side_dir[cid] = side_for(cid)

    answer = [[8] * W for _ in range(H)]

    def paint_line(r, c, dr, dc, stop_comp=-1):
        nr, nc = r + dr, c + dc
        while not out_of(nr, nc):
            cid = comp[nr][nc]
            if cid != -1 and cid != stop_comp:
                return
            answer[nr][nc] = 2
            nr += dr
            nc += dc

    def process(cid, visited):
        if cid in visited:
            return
        visited.add(cid)
        info = comps[cid]
        if info["has1"]:
            for y, x in info["cells"]:
                answer[y][x] = 2

        # Primary beam passes through this component without stopping.
        for y, x in info["cells"]:
            paint_line(y, x, direction[0], direction[1], cid)

        # Secondary beam exits from the 3 side.
        sd = side_dir[cid]
        if sd is None:
            return
        dr, dc = sd
        for y, x in info["cells"]:
            if grid[y][x] != 3:
                continue
            nr, nc = y + dr, x + dc
            while not out_of(nr, nc):
                ncid = comp[nr][nc]
                if ncid != -1 and ncid != cid:
                    process(ncid, visited)
                    return
                answer[nr][nc] = 2
                nr += dr
                nc += dc

    # Start the beam at the 4 marker and go away from the 2 tail.
    r, c = fr, fc
    while True:
        r += direction[0]
        c += direction[1]
        if out_of(r, c):
            break
        cid = comp[r][c]
        if cid != -1:
            process(cid, set())
            break
        answer[r][c] = 2

    return answer