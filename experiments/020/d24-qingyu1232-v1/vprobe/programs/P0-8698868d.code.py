def transform(grid):
    H = len(grid)
    W = len(grid[0])
    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=lambda c: (counts[c], c))

    def bbox(v):
        xs = []
        ys = []
        for y, row in enumerate(grid):
            for x, c in enumerate(row):
                if c == v:
                    xs.append(x)
                    ys.append(y)
        return min(ys), min(xs), max(ys), max(xs)

    # Identify background regions (4-connectivity) of the background color.
    # A region is a "stage" if it contains a connected foreground component
    # whose interior holes are all background cells (a pattern).
    # Otherwise, if it contains a solid-ish colored blob, it is a container.
    visited = [[False] * W for _ in range(H)]
    stages = []
    containers = {}

    for sy in range(H):
        for sx in range(W):
            if visited[sy][sx] or grid[sy][sx] != bg:
                continue
            # BFS background region
            region = []
            stack = [(sy, sx)]
            visited[sy][sx] = True
            while stack:
                cy, cx = stack.pop()
                region.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and not visited[ny][nx] and grid[ny][nx] == bg:
                        visited[ny][nx] = True
                        stack.append((ny, nx))
            region_set = set(region)
            miny = min(y for y, x in region)
            maxy = max(y for y, x in region)
            minx = min(x for y, x in region)
            maxx = max(x for y, x in region)
            w = maxx - minx + 1
            h = maxy - miny + 1

            # Find foreground components fully inside this region.
            seen_fg = [[False] * W for _ in range(H)]
            fg_comps = []
            for cy in range(miny, maxy + 1):
                for cx in range(minx, maxx + 1):
                    if (cy, cx) in region_set and not seen_fg[cy][cx] and grid[cy][cx] != bg:
                        cval = grid[cy][cx]
                        comp = []
                        st = [(cy, cx)]
                        seen_fg[cy][cx] = True
                        while st:
                            y, x = st.pop()
                            comp.append((y, x))
                            for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                                if 0 <= ny < H and 0 <= nx < W and not seen_fg[ny][nx] and grid[ny][nx] == cval:
                                    seen_fg[ny][nx] = True
                                    st.append((ny, nx))
                        fg_comps.append((cval, comp))

            pattern = None
            cont = None
            for cval, comp in fg_comps:
                comp_set = set(comp)
                ys = [y for y, x in comp]
                xs = [x for y, x in comp]
                ay0, ay1 = min(ys), max(ys)
                ax0, ax1 = min(xs), max(xs)
                aw = ax1 - ax0 + 1
                ah = ay1 - ay0 + 1
                area = len(comp)
                # A pattern has at least one hole (background pixel) strictly inside its bbox.
                has_hole = False
                for yy in range(ay0, ay1 + 1):
                    for xx in range(ax0, ax1 + 1):
                        if grid[yy][xx] == bg and (yy, xx) in region_set:
                            has_hole = True
                # If the component is mostly solid, treat it as a container.
                if not has_hole and area >= 0.75 * (aw * ah):
                    if cont is None or aw * ah > cont[1]:
                        cont = (cval, aw * ah, (ay0, ax0, ay1, ax1))
                elif has_hole:
                    # Prefer the largest holed component in this background stage.
                    if pattern is None or aw * ah > pattern[1]:
                        pattern = (cval, aw * ah, (ay0, ax0, ay1, ax1))

            if pattern:
                stages.append(pattern)
            elif cont:
                # Store container by color (in demos, colors are unique containers).
                if cont[0] not in containers or cont[1] > containers[cont[0]][1]:
                    containers[cont[0]] = cont

    # Fallback: if the region logic missed something, use simple color bbox.
    # (Kept minimal and general.)
    def solid_bbox(v):
        y0, x0, y1, x1 = bbox(v)
        return (v, (y1 - y0 + 1) * (x1 - x0 + 1), (y0, x0, y1, x1))

    # Pair stages to containers by spatial proximity in their original rows/cols.
    # Use centers and choose nearest unused container to each pattern.
    pairs = []
    used_cont = set()
    # Patterns sorted from top-left to bottom-right.
    pats = sorted(stages, key=lambda t: (t[2][0], t[2][1], t[2][2], t[2][3]))
    conts = list(containers.values())
    conts_sorted = sorted(conts, key=lambda t: (t[2][0], t[2][1], t[2][2], t[2][3]))

    # If counts match, pair in order (works for the rectangular demo layouts).
    if len(pats) == len(conts):
        for p, c in zip(pats, conts_sorted):
            pairs.append((c, p))
    else:
        # Nearest center fallback.
        for p in pats:
            best = None
            best_d = None
            pc = ((p[2][0] + p[2][2]) / 2, (p[2][1] + p[2][3]) / 2)
            for c in conts:
                cc = ((c[2][0] + c[2][2]) / 2, (c[2][1] + c[2][3]) / 2)
                d = abs(pc[0] - cc[0]) + abs(pc[1] - cc[1])
                if best is None or d < best_d:
                    best = c
                    best_d = d
            pairs.append((best, p))

    # Build output: concatenate containers in original sorted order, replacing
    # each container's interior with its paired pattern.
    # Determine a grid layout for output: sort containers top-to-bottom, left-to-right.
    out_conts = sorted(conts, key=lambda t: (t[2][0], t[2][1]))
    # Map container color to pattern
    cmap = {}
    for c, p in pairs:
        cmap[c[0]] = p

    # Output blocks in container order; arrange in rows if multiple rows.
    # We need infer number of columns. Use distinct y coordinates of containers.
    ys = sorted(set(c[2][0] for c in out_conts))
    n_rows = len(ys)
    if n_rows == 0:
        return []
    blocks_per_row = len(out_conts) // n_rows
    if blocks_per_row == 0:
        blocks_per_row = len(out_conts)
        n_rows = 1

    # For each block, compute container size from bbox, and pattern size.
    # If pattern does not fit, resize by cropping/padding to interior.
    out_rows = []
    for r in range(n_rows):
        row_blocks = out_conts[r * blocks_per_row:(r + 1) * blocks_per_row]
        # Max height/width for this output row
        max_h = max((c[2][2] - c[2][0] + 1) for c in row_blocks)
        row_out = []
        max_w = max((c[2][3] - c[2][1] + 1) for c in row_blocks)
        # Build each block as a rectangle of container bbox size.
        block_rects = []
        for c in row_blocks:
            cy0, cx0, cy1, cx1 = c[2]
            ch = cy1 - cy0 + 1
            cw = cx1 - cx0 + 1
            cval = c[0]
            rect = [[cval] * cw for _ in range(ch)]
            p = cmap.get(cval)
            if p:
                py0, px0, py1, px1 = p[2]
                ph = py1 - py0 + 1
                pw = px1 - px0 + 1
                # Target interior size is container interior (ch-2, cw-2).
                ih = ch - 2
                iw = cw - 2
                if ih > 0 and iw > 0:
                    # Extract pattern pixels from input; bg inside becomes container color.
                    for i in range(ih):
                        for j in range(iw):
                            # Map interior coordinate to pattern by center alignment.
                            # If pattern is smaller, center it; if larger, crop center.
                            py = py0 + (i * ph) // ih if ih != 0 else py0
                            px = px0 + (j * pw) // iw if iw != 0 else px0
                            # Better: center-crop/scale by direct nearest neighbor:
                            py = py0 + (i * ph) // ih
                            px = px0 + (j * pw) // iw
                            v = grid[py][px]
                            if v == bg:
                                rect[1 + i][1 + j] = cval
                            else:
                                rect[1 + i][1 + j] = v
            block_rects.append(rect)

        # Pad each block to max_h x max_w with container color (or 0) for row alignment.
        padded = []
        for rect in block_rects:
            ch = len(rect)
            cw = len(rect[0]) if ch else 0
            # Determine pad color: use container color if available; fallback 0.
            # We lost color here; reconstruct from original conts order.
            # Instead, find matching rect by shape? Simpler: rebuild with color.
            pass

        # Rebuild padded with color info.
        padded = []
        for c, rect in zip(row_blocks, block_rects):
            ch = len(rect)
            cw = len(rect[0]) if ch else 0
            cval = c[0]
            pr = []
            for i in range(max_h):
                line = []
                for j in range(max_w):
                    if 0 <= i < ch and 0 <= j < cw:
                        line.append(rect[i][j])
                    else:
                        line.append(cval)
                pr.append(line)
            padded.append(pr)

        # Concatenate horizontally.
        for i in range(max_h):
            out_rows.append(sum((padded[b][i] for b in range(len(padded))), []))
    return out_rows