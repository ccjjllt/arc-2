def transform(grid):
    H = len(grid)
    W = len(grid[0])

    # Locate the 6-row legend (top block bounded by 5).
    legend_start = 0
    for i in range(H):
        if all(v == 5 for v in grid[i]):
            legend_start = i
            break
    legend_end = legend_start + 5
    lower = [row[legend_end + 1:] for row in grid[legend_end + 1:]]
    LH = len(lower)
    LW = len(lower[0])

    # Find all maximal monochromatic rectangles in the lower grid.
    used = [[False] * LW for _ in range(LH)]
    rects = []
    for r in range(LH):
        for c in range(LW):
            if used[r][c]:
                continue
            v = lower[r][c]
            r2 = r
            while r2 + 1 < LH and lower[r2 + 1][c] == v:
                r2 += 1
            c2 = c
            while c2 + 1 < LW and lower[r][c2 + 1] == v:
                c2 += 1
            valid = True
            for rr in range(r, r2 + 1):
                for cc in range(c, c2 + 1):
                    if lower[rr][cc] != v or used[rr][cc]:
                        valid = False
                        break
                if not valid:
                    break
            if not valid:
                continue
            for rr in range(r, r2 + 1):
                for cc in range(c, c2 + 1):
                    used[rr][cc] = True
            rects.append((r, r2, c, c2, v))

    # Parse the legend into four horizontal anchors.
    anchors = []
    for r in range(legend_start + 1, legend_end + 1):
        row = grid[r]
        c = 0
        while c < W:
            if row[c] != 5:
                val = row[c]
                if val != 0:
                    left = c
                    while c + 1 < W and row[c + 1] != 5:
                        c += 1
                    right = c
                    anchors.append((left, right, val))
                    break
                c += 1
            else:
                c += 1
        if anchors:
            break
    anchors.sort(key=lambda x: x[0])

    # Existing 2x2 8 rectangle.
    marker = None
    for (r1, r2, c1, c2, v) in rects:
        if v == 8 and r2 - r1 == 1 and c2 - c1 == 1:
            marker = (r1, r2, c1, c2)
            break

    # Background is the color of the cell at the top-left of the lower grid.
    bg = lower[0][0]

    # For each legend label, find the lower rectangle of that color.
    targets = []
    for _, _, lab in anchors:
        for (r1, r2, c1, c2, v) in rects:
            if v == lab:
                targets.append((r1, r2, c1, c2))
                break

    def add_h(row, c1, c2, out):
        for cc in range(c1, c2 + 1):
            if out[row][cc] == bg:
                out[row][cc] = 8

    def add_v(col, r1, r2, out):
        for rr in range(r1, r2 + 1):
            if out[rr][col] == bg:
                out[rr][col] = 8

    # Build horizontal and vertical spans to connect marker with targets.
    h_spans = []
    v_spans = []

    if marker is not None:
        mr1, mr2, mc1, mc2 = marker
        mc_mid = (mc1 + mc2) // 2
        mr_mid = (mr1 + mr2) // 2

        # The marker itself is kept.
        for rr in range(mr1, mr2 + 1):
            for cc in range(mc1, mc2 + 1):
                if out[rr][cc] == bg:
                    out[rr][cc] = 8

        for (t1, t2, t3, t4) in targets:
            # Connect horizontally to the target's left/right edge as needed.
            if t4 < mc1:
                # Target is to the left.
                h1 = min(mc1, t3)
                h2 = max(mc1, t4)
                h_spans.append((mr_mid, h1, h2))
                h_spans.append((mr_mid + 1, h1, h2))
                v_spans.append((t4, min(mr_mid, t2), t4))
                v_spans.append((t4, min(mr_mid, t2), t4))
            elif t3 > mc2:
                # Target is to the right.
                h1 = min(mc2, t3)
                h2 = max(mc2, t4)
                h_spans.append((mr_mid, h1, h2))
                h_spans.append((mr_mid + 1, h1, h2))
                v_spans.append((t3, min(mr_mid, t2), t3))
                v_spans.append((t3, min(mr_mid, t2), t3))
            else:
                # Target horizontally overlaps marker.
                if t2 < mr1:
                    v_spans.append((mc_mid, t2, mr1))
                    v_spans.append((mc_mid + 1, t2, mr1))
                    h_spans.append((t2, min(mc_mid, t3), max(mc_mid, t4)))
                elif t1 > mr2:
                    v_spans.append((mc_mid, mr2, t1))
                    v_spans.append((mc_mid + 1, mr2, t1))
                    h_spans.append((t1, min(mc_mid, t3), max(mc_mid, t4)))

        # Always connect vertically to the top and bottom of the lower area
        # through the marker columns, preserving background only.
        v_spans.append((mc1, 0, mr1))
        v_spans.append((mc2, 0, mr1))
        v_spans.append((mc1, mr2, LH - 1))
        v_spans.append((mc2, mr2, LH - 1))

    for (row, c1, c2) in h_spans:
        add_h(row, c1, c2, out)
    for (col, r1, r2) in v_spans:
        add_v(col, r1, r2, out)

    return out