def transform(grid):
    if not grid or not grid[0]:
        return [[0]]

    H = len(grid)
    W = len(grid[0])

    # Locate the 0 corner.
    zero = None
    for r in range(H):
        for c in range(W):
            if grid[r][c] == 0:
                zero = (r, c)
                break
        if zero is not None:
            break

    # Find all non-8 cells and keep only those connected to the 0 corner,
    # which selects the frame and excludes isolated objects if any.
    seen = [[False] * W for _ in range(H)]
    stack = [zero]
    seen[zero[0]][zero[1]] = True
    frame_cells = []
    while stack:
        r, c = stack.pop()
        frame_cells.append((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < H and 0 <= nc < W and not seen[nr][nc] and grid[nr][nc] != 8:
                seen[nr][nc] = True
                stack.append((nr, nc))

    rmin = min(r for r, c in frame_cells)
    rmax = max(r for r, c in frame_cells)
    cmin = min(c for r, c in frame_cells)
    cmax = max(c for r, c in frame_cells)

    # Frame is either top+left or top+right.  Inner region is just inside it.
    if zero[1] == 0:
        # top-left: top frame row rmin, left frame column cmin
        ir0 = rmin + 1
        ir1 = rmax + 1
        ic0 = cmin + 1
        ic1 = cmax + 1
        col_dir = 1
    else:
        # top-right: top frame row rmin, right frame column cmax
        ir0 = rmin + 1
        ir1 = rmax + 1
        ic0 = cmin
        ic1 = cmax
        col_dir = -1

    h = ir1 - ir0
    w = ic1 - ic0

    # Detect connected non-8 rectangles inside the inner region.
    inner_seen = [[False] * w for _ in range(h)]
    rects = []
    for sr in range(h):
        for sc in range(w):
            if inner_seen[sr][sc] or grid[ir0 + sr][ic0 + sc] == 8:
                continue
            color = grid[ir0 + sr][ic0 + sc]
            stack = [(sr, sc)]
            inner_seen[sr][sc] = True
            cells = []
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = r + dr
                    nc = c + dc
                    if (0 <= nr < h and 0 <= nc < w and not inner_seen[nr][nc]
                            and grid[ir0 + nr][ic0 + nc] == color):
                        inner_seen[nr][nc] = True
                        stack.append((nr, nc))
            # For each monochrome connected cell group, take its bounding box.
            r1 = min(r for r, c in cells)
            r2 = max(r for r, c in cells) + 1
            c1 = min(c for r, c in cells)
            c2 = max(c for r, c in cells) + 1
            rects.append((r1, c1, r2, c2, color))

    # The visible objects are exactly the maximal all-non-8 rectangles.
    # Group monochrome components by their bounding box.
    boxes = {}
    for r1, c1, r2, c2, color in rects:
        boxes[(r1, c1, r2, c2)] = color

    # For each bounding box, verify it is full of non-8 and record its cells.
    objects = []
    for (r1, c1, r2, c2), color in boxes.items():
        ok = True
        for rr in range(r1, r2):
            for cc in range(c1, c2):
                if grid[ir0 + rr][ic0 + cc] == 8:
                    ok = False
                    break
            if not ok:
                break
        if ok:
            obj = []
            for rr in range(r1, r2):
                obj.append([grid[ir0 + rr][ic0 + cc] for cc in range(c1, c2)])
            objects.append((r1, c1, r2 - r1, c2 - c1, obj))

    # Decide reading order: use the longer frame side as primary direction.
    # Top edge length is w + 1, side edge length is h + 1.
    if h > w:
        # column-major
        cols = sorted(set(c1 for _, c1, _, _, _ in objects), key=lambda c: -c if col_dir < 0 else c)
        ordered = []
        for c in cols:
            objs = [o for o in objects if o[1] == c]
            objs.sort(key=lambda o: o[0])
            ordered.extend(objs)
    else:
        # row-major
        rows = sorted(set(r1 for r1, _, _, _, _ in objects), key=lambda r: r)
        ordered = []
        for r in rows:
            objs = [o for o in objects if o[0] == r]
            if col_dir < 0:
                objs.sort(key=lambda o: -o[1])
            else:
                objs.sort(key=lambda o: o[1])
            ordered.extend(objs)

    # Rotate vertical rectangles 90 degrees counter-clockwise so their long
    # side becomes horizontal.  Squares and horizontal rectangles stay as is.
    pieces = []
    for _, _, hh, ww, mat in ordered:
        if hh > ww:
            # Counter-clockwise: new row j = old column (ww-1-j), old row i
            new_h = ww
            new_w = hh
            rotated = [[0] * new_w for _ in range(new_h)]
            for i in range(hh):
                for j in range(ww):
                    rotated[j][i] = mat[i][ww - 1 - j]
            pieces.append(rotated)
        else:
            pieces.append(mat)

    out_w = max(max(len(row) for row in p) for p in pieces)
    out = []
    for p in pieces:
        ph = len(p)
        pw = len(p[0])
        left = (out_w - pw) // 2
        for row in p:
            out.append([8] * left + row + [8] * (out_w - pw - left))

    return out