def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def inb(r, c):
        return 0 <= r < H and 0 <= c < W

    def same_rect(cells):
        if not cells:
            return None
        rs = [r for r, c in cells]
        cs = [c for r, c in cells]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        if len(cells) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return None
        return (r0, c0, r1, c1)

    counts = {}
    for r in range(H):
        for c in range(W):
            counts[grid[r][c]] = counts.get(grid[r][c], 0) + 1
    bg = max(sorted(counts, key=lambda x: (-counts[x], x)), key=lambda x: counts[x])

    box_color = None
    best = 0
    for val in counts:
        if val == bg:
            continue
        cells = [(r, c) for r in range(H) for c in range(W) if grid[r][c] == val]
        rect = same_rect(cells)
        if rect is None:
            continue
        r0, c0, r1, c1 = rect
        inside = 0
        for rr in range(r0, r1 + 1):
            for cc in range(c0, c1 + 1):
                if grid[rr][cc] != val and grid[rr][cc] != bg:
                    inside += 1
        if inside > best:
            best = inside
            box_color = val

    if box_color is None:
        return [row[:] for row in grid]

    boxes = []
    seen = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] == box_color and (r, c) not in seen:
                stack = [(r, c)]
                comp = []
                seen.add((r, c))
                while stack:
                    rr, cc = stack.pop()
                    comp.append((rr, cc))
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if inb(nr, nc) and grid[nr][nc] == box_color and (nr, nc) not in seen:
                            seen.add((nr, nc))
                            stack.append((nr, nc))
                rect = same_rect(comp)
                if rect is None:
                    continue
                r0, c0, r1, c1 = rect
                markers = []
                for rr in range(r0, r1 + 1):
                    for cc in range(c0, c1 + 1):
                        v = grid[rr][cc]
                        if v != bg and v != box_color:
                            markers.append((rr, cc, v))
                if markers:
                    vr0 = min(m[0] for m in markers)
                    vr1 = max(m[0] for m in markers)
                    vc0 = min(m[1] for m in markers)
                    vc1 = max(m[1] for m in markers)
                    val = max(set(m[2] for m in markers), key=lambda x: sum(1 for m in markers if m[2] == x))
                    boxes.append((r0, c0, r1, c1, vr0, vc0, vr1, vc1, val))

    legend_row = None
    legend_seq = []
    for r in range(H - 1, -1, -1):
        vals = [grid[r][c] for c in range(W) if grid[r][c] != bg]
        if not vals:
            continue
        if box_color is not None and box_color in vals:
            continue
        isolated = 0
        for c, v in enumerate(vals):
            left = grid[r][c - 1] == bg if c > 0 else True
            right = grid[r][c + 1] == bg if c < W - 1 else True
            if left and right:
                isolated += 1
        if isolated == len(vals) and len(vals) >= 2:
            legend_row = r
            legend_seq = vals
            break

    out = [row[:] for row in grid]

    def bg_at(r, c):
        return grid[r][c] == bg

    def first_color(a, b):
        ia = ib = None
        for i, v in enumerate(legend_seq):
            if ia is None and v == a:
                ia = i
            if ib is None and v == b:
                ib = i
        if ia is None or ib is None:
            return None
        if abs(ia - ib) != 1:
            return None
        return legend_seq[ia] if ia < ib else legend_seq[ib]

    n = len(boxes)
    for i in range(n):
        for j in range(i + 1, n):
            A = boxes[i]
            B = boxes[j]
            ar0, ac0, ar1, ac1, arv0, acv0, arv1, acv1, av = A
            br0, bc0, br1, bc1, brv0, bcv0, brv1, bcv1, bv = B
            if av == bv:
                continue
            col = first_color(av, bv)
            if col is None:
                continue

            if ar0 <= br1 and br0 <= ar1 and ac0 == bc0 and ac1 == bc1:
                c0, c1 = acv0, acv1
                if ac1 + 1 >= br0:
                    continue
                top, bottom = (A, B) if ar1 < br0 else (B, A)
                gap0 = top[2] + 1
                gap1 = bottom[0] - 1
                if gap0 > gap1:
                    continue
                if not all(bg_at(r, c) for r in range(gap0, gap1 + 1) for c in range(c0, c1 + 1)):
                    continue
                for r in range(gap0, gap1 + 1):
                    for c in range(c0, c1 + 1):
                        out[r][c] = col

            if ac0 <= bc1 and bc0 <= ac1 and ar0 == br0 and ar1 == br1:
                r0, r1 = arv0, arv1
                if ac1 + 1 >= bc0:
                    continue
                left, right = (A, B) if ac1 < bc0 else (B, A)
                gap0 = left[3] + 1
                gap1 = right[1] - 1
                if gap0 > gap1:
                    continue
                if not all(bg_at(r, c) for r in range(r0, r1 + 1) for c in range(gap0, gap1 + 1)):
                    continue
                for r in range(r0, r1 + 1):
                    for c in range(gap0, gap1 + 1):
                        out[r][c] = col

    return out