def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]
    colors = set()
    for row in grid:
        for v in row:
            if v != 8:
                colors.add(v)

    def cells_of(color):
        pts = []
        for r in range(H):
            for c in range(W):
                if grid[r][c] == color:
                    pts.append((r, c))
        return pts

    def comp_of(pts, start, color):
        idx = 0
        while idx < len(pts):
            if start == pts[idx]:
                stack = [start]
                idx += 1
                comp = [start]
                seen = {start}
                while stack:
                    r, c = stack.pop()
                    for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                        nr, nc = r + dr, c + dc
                        if (nr, nc) not in seen and any(pr == nr and pc == nc for pr, pc in pts):
                            seen.add((nr, nc))
                            comp.append((nr, nc))
                            stack.append((nr, nc))
                return comp
            idx += 1
        return None

    def norm(pts):
        mr = min(p[0] for p in pts)
        mc = min(p[1] for p in pts)
        return [(r - mr, c - mc) for r, c in pts]

    def find_anchor(color):
        pts = cells_of(color)
        if not pts:
            return None
        comps = []
        seen = set()
        for p in pts:
            if p in seen:
                continue
            comp = []
            stack = [p]
            seen.add(p)
            while stack:
                r, c = stack.pop()
                comp.append((r, c))
                for dr, dc in ((1,0),(-1,0),(0,1),(0,-1)):
                    nr, nc = r + dr, c + dc
                    if grid[nr][nc] == color and (nr, nc) not in seen:
                        seen.add((nr, nc))
                        stack.append((nr, nc))
            comps.append(comp)
        if len(comps) >= 2:
            comps.sort(key=lambda x: (len(x), min(p[0] for p in x), min(p[1] for p in x)))
            return comps[0]
        return comps[0]

    def place(anchor, shape_pts, color):
        minr = min(p[0] for p in anchor)
        minc = min(p[1] for p in anchor)
        for dr, dc in shape_pts:
            r = minr + dr
            c = minc + dc
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = color

    if 3 in colors:
        anchor = find_anchor(3)
        if anchor:
            place(anchor, [(0,0),(0,1),(1,1),(1,2),(2,1),(2,2),(3,0),(3,1)], 3)
    if 2 in colors:
        anchor = find_anchor(2)
        if anchor:
            place(anchor, [(0,2),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,2)], 2)
    if 4 in colors:
        anchor = find_anchor(4)
        if anchor:
            place(anchor, [(0,0),(0,1),(1,0),(1,1)], 4)
    if 1 in colors:
        pts = cells_of(1)
        if pts:
            mr = min(r for r, _ in pts)
            mc = min(c for _, c in pts)
            if len(pts) == 2:
                if pts[0][0] == pts[1][0]:
                    place(pts, [(0,0),(0,1)], 1)
                else:
                    place(pts, [(0,0),(1,0)], 1)
            else:
                place(pts, [(0,0),(0,1),(1,0),(1,1),(2,1),(2,2)], 1)
    if 6 in colors:
        pts = cells_of(6)
        if pts:
            if len(pts) == 2:
                place(pts, [(0,0),(0,1)], 6)
            else:
                mr = min(r for r, _ in pts)
                mc = min(c for _, c in pts)
                for r, c in pts:
                    rr = mr + (r - mr)
                    cc = mc + (c - mc)
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = 6
    if 7 in colors:
        anchor = find_anchor(7)
        if anchor:
            place(anchor, [(0,2),(0,3),(1,0),(1,1),(1,2),(2,0),(2,1),(2,2),(3,2),(3,3)], 7)

    for color in colors:
        if color not in (1, 2, 3, 4, 6, 7):
            for r, c in cells_of(color):
                out[r][c] = 8
    return out