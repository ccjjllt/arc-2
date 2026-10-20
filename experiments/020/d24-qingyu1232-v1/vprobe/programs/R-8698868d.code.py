def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def bbox(c):
        rows = [r for r in range(H) for ch in range(W) if grid[r][ch] == c]
        cols = [ch for r in range(H) for ch in range(W) if grid[r][ch] == c]
        if not rows or not cols:
            return None
        return min(rows), min(cols), max(rows), max(cols)

    def get_rect(r0, c0, r1, c1):
        vals = [grid[r][c] for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)]
        counts = {}
        for v in vals:
            counts[v] = counts.get(v, 0) + 1
        main, hole = None, None
        best = -1
        for v, n in counts.items():
            if n > best:
                best = n
                main, hole = v, v
        for v, n in counts.items():
            if v != main and (hole is None or n < counts[hole] if counts.get(hole, 0) != counts[main] else False):
                pass
        # choose main as most common, hole as most common non-main
        non = {v: n for v, n in counts.items() if v != main}
        if non:
            hv = max(non, key=lambda x: non[x])
            return main, hole if hole is not None else main, hv
        return main, main, main
    # Simpler robust rect extraction:
    def rect_info(r0, c0, r1, c1):
        counts = {}
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                v = grid[r][c]
                counts[v] = counts.get(v, 0) + 1
        main = max(counts, key=lambda v: counts[v])
        non = {v: n for v, n in counts.items() if v != main}
        hole = max(non, key=lambda v: non[v]) if non else main
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        pat = [[0] * w for _ in range(h)]
        for i, r in enumerate(range(r0, r1 + 1)):
            for j, c in enumerate(range(c0, c1 + 1)):
                if grid[r][c] == main:
                    pat[i][j] = main
                elif grid[r][c] == hole:
                    pat[i][j] = 0
                else:
                    pat[i][j] = 0
        return main, h, w, pat

    candidates = []
    for c in range(10):
        b = bbox(c)
        if b is None:
            continue
        r0, c0, r1, c1 = b
        main, h, w, pat = rect_info(r0, c0, r1, c1)
        if main != c:
            continue
        if h < 3 or w < 3:
            continue
        # must be nearly a solid rectangle: at least 70% main and some interior hole possible
        area = h * w
        main_count = sum(row.count(main) for row in pat)
        if main_count / area < 0.5:
            continue
        candidates.append((r0, c0, r1, c1, c, h, w, pat))

    # A frame must contain at least one candidate of a different color strictly inside it.
    frames = []
    tiles = []
    for i, (r0, c0, r1, c1, c, h, w, pat) in enumerate(candidates):
        contains = False
        for j, (a, b, d, e, tc, th, tw, tpat) in enumerate(candidates):
            if i == j or tc == c:
                continue
            if a >= r0 and b >= c0 and d <= r1 and e <= c1:
                contains = True
                break
        if contains:
            frames.append((r0, c0, r1, c1, c, h, w))
        else:
            tiles.append((r0, c0, r1, c1, c, h, w, pat))

    # If no frame contains tiles by strict containment, use largest frames.
    if not frames:
        candidates.sort(key=lambda x: (x[5], x[6]), reverse=True)
        k = max(1, len(candidates) // 2)
        frames = [(r0, c0, r1, c1, c, h, w) for r0, c0, r1, c1, c, h, w, _ in candidates[:k]]
        tiles = [cand for cand in candidates if (cand[0], cand[1], cand[2], cand[3]) not in
                 [(f[0], f[1], f[2], f[3]) for f in frames]]

    # Crop frames bounding box
    R0 = min(f[0] for f in frames)
    C0 = min(f[1] for f in frames)
    R1 = max(f[2] for f in frames)
    C1 = max(f[3] for f in frames)
    Hout = R1 - R0 + 1
    Wout = C1 - C0 + 1
    out = [[0] * Wout for _ in range(Hout)]

    # Assign tiles to frames.
    # 1) By nearest matching-size tile if possible, otherwise sort both by position.
    frame_order = sorted(range(len(frames)), key=lambda i: (frames[i][0], frames[i][1]))
    used = [False] * len(tiles)
    assign = {}
    for fi in frame_order:
        fr = frames[fi]
        fh, fw = fr[5], fr[6]
        best_j = -1
        best_score = None
        for j, (a, b, d, e, tc, th, tw, tpat) in enumerate(tiles):
            if used[j]:
                continue
            if th >= fh or tw >= fw:
                continue
            # prefer exact centered or proportional placement, then small total size
            score = (abs(th * 4 - (fh - 2)), abs(tw * 4 - (fw - 2)), abs(a - fr[0]) + abs(b - fr[1]))
            if best_score is None or score < best_score:
                best_score = score
                best_j = j
        if best_j >= 0:
            used[best_j] = True
            assign[fi] = best_j

    # Ensure every frame gets a tile if possible; fallback by order.
    if len(assign) < len(frames):
        order_frames = frame_order
        order_tiles = sorted(range(len(tiles)), key=lambda j: (tiles[j][0], tiles[j][1]))
        idx = 0
        for fi in order_frames:
            if fi not in assign:
                while idx < len(order_tiles) and used[order_tiles[idx]]:
                    idx += 1
                if idx < len(order_tiles):
                    used[order_tiles[idx]] = True
                    assign[fi] = order_tiles[idx]
                    idx += 1

    for fi, (fr0, fc0, fr1, fc1, color, fh, fw) in enumerate(frames):
        # border
        for r in range(fr0, fr1 + 1):
            out[r - R0][fc0 - C0] = color
            out[r - R0][fc1 - C0] = color
        for c in range(fc0, fc1 + 1):
            out[fr0 - R0][c - C0] = color
            out[fr1 - R0][c - C0] = color
        if fi not in assign:
            continue
        a, b, d, e, tc, th, tw, tpat = tiles[assign[fi]]
        off_r = fr0 + (fh - th) // 2
        off_c = fc0 + (fw - tw) // 2
        for i in range(th):
            for j in range(tw):
                v = tpat[i][j]
                out[off_r - R0 + i][off_c - C0 + j] = color if v == 0 else v

    return out