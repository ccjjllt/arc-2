def transform(grid):
    h = len(grid)
    w = len(grid[0])

    counts = {}
    for row in grid:
        for v in row:
            counts[v] = counts.get(v, 0) + 1
    bg = max(counts, key=lambda v: (counts[v], v))

    def is_bg(y, x):
        return 0 <= y < h and 0 <= x < w and grid[y][x] == bg

    def component(y, x):
        c = grid[y][x]
        seen = {(y, x)}
        stack = [(y, x)]
        while stack:
            cy, cx = stack.pop()
            for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                if 0 <= ny < h and 0 <= nx < w and grid[ny][nx] == c and (ny, nx) not in seen:
                    seen.add((ny, nx))
                    stack.append((ny, nx))
        return c, seen

    boxes = []
    seen = set()
    for y in range(h):
        for x in range(w):
            if (y, x) not in seen and grid[y][x] != bg:
                c, pix = component(y, x)
                seen |= pix
                y0 = min(p[0] for p in pix)
                y1 = max(p[0] for p in pix)
                x0 = min(p[1] for p in pix)
                x1 = max(p[1] for p in pix)
                bh = y1 - y0 + 1
                bw = x1 - x0 + 1
                boxes.append((c, y0, x0, y1, x1, bh, bw, pix))

    if not boxes:
        return [[bg]]

    def is_frame(box):
        c, y0, x0, y1, x1, bh, bw, pix = box
        for y in range(y0, y1 + 1):
            for x in range(x0, x1 + 1):
                if (y, x) in pix:
                    continue
                if y == y0 or y == y1 or x == x0 or x == x1:
                    return False
        return True

    frames = [b for b in boxes if is_frame(b)]
    if not frames:
        frames = boxes

    def score_container(box):
        c, y0, x0, y1, x1, bh, bw, pix = box
        area = bh * bw
        inside = 0
        for y in range(y0 + 1, y1):
            for x in range(x0 + 1, x1):
                if (y, x) in pix:
                    inside += 1
        return -area - inside

    def score_pattern(box):
        c, y0, x0, y1, x1, bh, bw, pix = box
        holes = 0
        for y in range(y0 + 1, y1):
            for x in range(x0 + 1, x1):
                if (y, x) not in pix:
                    holes += 1
        if holes == 0:
            holes -= 1
        return holes

    # Containers are the largest solid-ish rectangular boxes; patterns are smaller holed frames.
    # In the demos, all container boxes have equal size and are larger than their patterns.
    containers = sorted(frames, key=lambda b: (-b[5], -b[6], b[1], b[2]))
    patterns = [b for b in frames if b[5] <= max(c[5] for c in containers) or True]

    # Prefer a strict separation: containers are the boxes with maximal bbox height/width.
    max_h = max(b[5] for b in frames)
    max_w = max(b[6] for b in frames)
    main = [b for b in frames if b[5] == max_h or b[6] == max_w]
    if len(main) >= 2:
        containers = sorted(main, key=lambda b: (b[1], b[2]))
    else:
        containers = sorted(frames, key=lambda b: (-b[5], -b[6], b[1], b[2]))

    # Assign each container the nearest pattern (by center distance) that is not the same box.
    def center(box):
        return ((box[1] + box[3]) / 2.0, (box[2] + box[4]) / 2.0)

    pair = {}
    used = set()
    # Sort containers in reading order to make assignment stable.
    for cont in containers:
        cc = center(cont)
        best = None
        bestd = None
        for pat in frames:
            if pat is cont or id(pat) in used:
                continue
            pc = center(pat)
            d = abs(cc[0] - pc[0]) + abs(cc[1] - pc[1])
            # Prefer smaller patterns if distances are close.
            d2 = (d, pat[5] * pat[6])
            if bestd is None or d2 < bestd:
                bestd = d2
                best = pat
        if best is not None:
            pair[id(cont)] = best
            used.add(id(best))

    # Output contains only containers, cropped to their collective bbox and packed in rows.
    if not containers:
        return [[bg]]

    # Group containers into visual rows by overlapping y-ranges.
    ordered = sorted(containers, key=lambda b: (b[1], b[2]))
    rows = []
    cur = []
    last_y1 = -1
    for b in ordered:
        if cur and b[1] > last_y1:
            rows.append(cur)
            cur = []
        cur.append(b)
        last_y1 = max(last_y1, b[3])
    if cur:
        rows.append(cur)

    def resize_pattern(pat, th, tw, fill):
        _, py0, px0, py1, px1, ph, pw, _ = pat
        if th <= 0 or tw <= 0:
            return []
        out = []
        for i in range(th):
            row = []
            sy = py0 + round(i * (ph - 1) / (th - 1)) if th > 1 else py0
            for j in range(tw):
                sx = px0 + round(j * (pw - 1) / (tw - 1)) if tw > 1 else px0
                row.append(fill if grid[sy][sx] == bg else grid[sy][sx])
            out.append(row)
        return out

    # Build each container rectangle, then pack row-wise.
    built_rows = []
    for row in rows:
        row = sorted(row, key=lambda b: b[2])
        max_h = max(b[5] for b in row)
        rects = []
        for b in row:
            c, y0, x0, y1, x1, bh, bw, _ = b
            rect = [[c] * bw for _ in range(bh)]
            pat = pair.get(id(b))
            if pat is not None:
                resized = resize_pattern(pat, bh - 2, bw - 2, c)
                for i in range(bh - 2):
                    for j in range(bw - 2):
                        if i < len(resized) and j < len(resized[i]):
                            rect[1 + i][1 + j] = resized[i][j]
            rects.append(rect)

        max_w = max(len(r[0]) for r in rects)
        for i in range(max_h):
            out_row = []
            for rect in rects:
                if i < len(rect):
                    out_row.extend(rect[i])
                    out_row.extend([rect[0][0]] * (max_w - len(rect[i])))
                else:
                    out_row.extend([row[0][0]] * max_w)
            built_rows.append(out_row)

    # Normalize width across rows by padding with first container color.
    if not built_rows:
        return [[bg]]
    width = max(len(r) for r in built_rows)
    padc = built_rows[0][0]
    return [r + [padc] * (width - len(r)) for r in built_rows]