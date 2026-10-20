def transform(grid):
    H = len(grid)
    W = len(grid[0])

    def is_bg(v):
        return (v == min(
            [min(r) for r in grid]
        ) or v == max(
            [max(r) for r in grid]
        ) or v == sorted({c for r in grid for c in r})[len({c for r in grid for c in r}) // 2])

    # More direct background: the color that forms the large boundary frame.
    candidates = set()
    for r in range(H):
        candidates.add(grid[r][0])
        candidates.add(grid[r][W - 1])
    for c in range(W):
        candidates.add(grid[0][c])
        candidates.add(grid[H - 1][c])
    bg = max(candidates, key=lambda v: sum(1 for r in range(H) for c in range(W) if grid[r][c] == v))

    cells = set()
    for r in range(H):
        for c in range(W):
            if grid[r][c] != bg:
                cells.add((r, c))

    # Connected components (4-neighbor) by color.
    seen = set()
    comps = []
    for r in range(H):
        for c in range(W):
            if (r, c) in cells and (r, c) not in seen:
                color = grid[r][c]
                stack = [(r, c)]
                comp = set()
                seen.add((r, c))
                while stack:
                    y, x = stack.pop()
                    comp.add((y, x))
                    for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < H and 0 <= nx < W and (ny, nx) not in seen:
                            if grid[ny][nx] == color:
                                seen.add((ny, nx))
                                stack.append((ny, nx))
                comps.append((color, comp))

    def bbox(comp):
        rs = [p[0] for p in comp]
        cs = [p[1] for p in comp]
        return min(rs), max(rs), min(cs), max(cs)

    # Full 5x5 solid squares.
    fulls = []
    for color, comp in comps:
        r0, r1, c0, c1 = bbox(comp)
        if r1 - r0 == 4 and c1 - c0 == 4 and len(comp) == 25:
            fulls.append((color, r0, c0, comp))

    # Small pattern components (not the full square used as a base).
    patterns = {}
    used_base = set()
    for color, r0, c0, comp in fulls:
        used_base.add(frozenset(comp))

    for color, comp in comps:
        if frozenset(comp) in used_base:
            continue
        r0, r1, c0, c1 = bbox(comp)
        if (r1 - r0 + 1) * (c1 - c0 + 1) <= 9:
            shape = []
            for r in range(r0, r1 + 1):
                row = []
                for c in range(c0, c1 + 1):
                    row.append(1 if (r, c) in comp else 0)
                shape.append(row)
            key = (color, tuple(tuple(row) for row in shape))
            # Keep the first/most central; all should be the intended one.
            if key not in patterns:
                patterns[key] = shape

    def center_shape(base_color, shape):
        if not shape:
            return []
        sh = len(shape)
        sw = max(len(row) for row in shape) if shape else 0
        off_r = (5 - sh) // 2
        off_c = (5 - sw) // 2
        holes = []
        for i in range(5):
            row_h = [False] * 5
            for dr, srow in enumerate(shape):
                r = off_r + dr
                if 0 <= r < 5:
                    for dc, bit in enumerate(srow):
                        c = off_c + dc
                        if 0 <= c < 5 and bit:
                            row_h[c] = True
            holes.append(row_h)
        return holes

    # Choose a line (row or column) of full squares.
    best_line = None
    best_score = -1
    for line_type in (0, 1):
        groups = {}
        for color, r0, c0, comp in fulls:
            key = r0 if line_type == 0 else c0
            groups.setdefault(key, []).append((color, r0, c0))
        for key, items in groups.items():
            if len(items) < 2:
                continue
            score = 0
            ok = True
            for color, r0, c0 in items:
                found = any(color == pc and len(s) <= 3 and len(s[0]) <= 3 for pc, s in patterns.items())
                if found:
                    score += 1
                else:
                    ok = False
            if score > best_score:
                best_score = score
                best_line = (line_type, key, items)

    if best_line is None:
        # Fallback: pick the line with most full squares.
        for line_type in (0, 1):
            groups = {}
            for color, r0, c0, comp in fulls:
                key = r0 if line_type == 0 else c0
                groups.setdefault(key, []).append((color, r0, c0))
            for key, items in groups.items():
                if len(items) > best_score:
                    best_score = len(items)
                    best_line = (line_type, key, items)

    if best_line is None:
        return [[bg for _ in range(1)] for _ in range(1)]

    line_type, key, items = best_line
    items.sort(key=lambda x: x[2] if line_type == 0 else x[1])

    out = []
    margin = 1
    cell = 5
    gap = 1

    if line_type == 0:
        # Horizontal line: output height 7, width 3*7 + gaps? In demos: 7 x 19 for three cells.
        rows = 2 * margin + cell
        cols = 2 * margin + len(items) * (cell + gap)
        # Actually demo1: 7 x 19 with 3 blocks: margin 1 + 3*5 + 2 gaps*1 = 1+15+2=18, plus final margin? 1+15+2+1=19.
        cols = margin + len(items) * cell + (len(items) - 1) * gap + margin
        out = [[bg] * cols for _ in range(rows)]
        for idx, (color, r0, c0) in enumerate(items):
            # find a pattern for this color
            pat = None
            for pc, shape in patterns.items():
                if pc == color:
                    pat = shape
                    break
            holes = center_shape(color, pat) if pat else []
            base_r = margin
            base_c = margin + idx * (cell + gap)
            for dr in range(cell):
                for dc in range(cell):
                    out[base_r + dr][base_c + dc] = color
                    if holes and holes[dr][dc]:
                        out[base_r + dr][base_c + dc] = bg
    else:
        # Vertical line: output width 7, height blocks.
        cols = 2 * margin + cell
        rows = margin + len(items) * cell + (len(items) - 1) * gap + margin
        out = [[bg] * cols for _ in range(rows)]
        for idx, (color, r0, c0) in enumerate(items):
            pat = None
            for pc, shape in patterns.items():
                if pc == color:
                    pat = shape
                    break
            holes = center_shape(color, pat) if pat else []
            base_r = margin + idx * (cell + gap)
            base_c = margin
            for dr in range(cell):
                for dc in range(cell):
                    out[base_r + dr][base_c + dc] = color
                    if holes and holes[dr][dc]:
                        out[base_r + dr][base_c + dc] = bg

    return out