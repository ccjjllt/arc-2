def transform(grid):
    h = len(grid)
    w = len(grid[0])
    mid_r = h // 2
    mid_c = w // 2

    def crop_block(block):
        rows = [r for r in block if any(c != 0 for c in r)]
        if not rows:
            return block
        first = 0
        while first < len(rows[0]) and all(row[first] == 0 for row in rows):
            first += 1
        last = len(rows[0]) - 1
        while last >= 0 and all(row[last] == 0 for row in rows):
            last -= 1
        if first > last:
            return block
        return [row[first:last + 1] for row in rows]

    def crop_rect(block):
        rows = [i for i, row in enumerate(block) if any(c != 0 for c in row)]
        cols = [j for j in range(len(block[0])) if any(block[i][j] != 0 for i in range(len(block)))]
        if not rows or not cols:
            return block
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        return [row[c0:c1 + 1] for row in block[r0:r1 + 1]]

    def block_value(block):
        for row in block:
            for val in row:
                if val != 0:
                    return val
        return 0

    def is_solid(block):
        vals = set()
        for row in block:
            for val in row:
                vals.add(val)
        vals.discard(0)
        if len(vals) != 1:
            return False
        val = list(vals)[0]
        return all(cell == val for row in block for cell in row)

    tl = [row[:mid_c] for row in grid[:mid_r]]
    tr = [row[mid_c:] for row in grid[:mid_r]]
    bl = [row[:mid_c] for row in grid[mid_r:]]
    br = [row[mid_c:] for row in grid[mid_r:]]

    content = [row[:] for row in bl]
    if any(c != 0 for row in br for c in row):
        for i in range(len(br)):
            for j in range(len(br[0])):
                if br[i][j] != 0:
                    content[i][j] = br[i][j]
    content = crop_block(content)
    bh, bw = len(content), len(content[0])

    left = crop_block(bl)
    right = crop_block(br)
    left_solid = is_solid(left)
    right_solid = is_solid(right)

    top_l = crop_block(tl)
    top_r = crop_block(tr)

    h_f = 1
    w_f = 1

    if len(left) == bh and len(right) == bh:
        if left_solid and not right_solid:
            if len(top_l) <= len(top_r):
                h_f = max(1, min(len(top_l), len(top_r)))
                w_f = 1
            else:
                h_f = 1
                w_f = max(1, min(len(top_l), len(top_r)))
        elif right_solid and not left_solid:
            if len(top_l) <= len(top_r):
                h_f = 1
                w_f = max(1, min(len(top_l), len(top_r)))
            else:
                h_f = max(1, min(len(top_l), len(top_r)))
                w_f = 1
        else:
            h_f = max(1, min(len(top_l), len(top_r)))
            w_f = max(1, min(len(top_l), len(top_r)))
    else:
        h_f = max(1, min(len(top_l), len(top_r)))
        w_f = max(1, min(len(top_l), len(top_r)))

    out_h = max(1, bh * h_f)
    out_w = max(1, bw * w_f)
    out = [[0 for _ in range(out_w)] for _ in range(out_h)]
    for i in range(out_h):
        src_i = i % bh
        for j in range(out_w):
            out[i][j] = content[src_i][j % bw]
    return out