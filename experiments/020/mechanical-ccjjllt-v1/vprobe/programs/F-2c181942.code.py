def transform(grid):
    H = len(grid)
    W = len(grid[0])
    bg = 8
    out = [[bg] * W for _ in range(H)]
    positions = {}
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v != bg:
                positions.setdefault(v, []).append((r, c))

    def norm(cells):
        mr = min(r for r, _ in cells)
        mc = min(c for _, c in cells)
        return [(r - mr, c - mc) for r, c in cells]

    def area(cells):
        return len(cells)

    def draw(cells, top, left, color):
        for dr, dc in cells:
            r = top + dr
            c = left + dc
            if 0 <= r < H and 0 <= c < W:
                out[r][c] = color

    def is_open(cells):
        s = set(cells)
        return not all(p in s for p in ((0, 1), (1, 0), (1, 1), (2, 1)))

    shapes = {color: norm(cells) for color, cells in positions.items()}

    for color in shapes:
        if len(shapes[color]) == 4:
            if is_open(shapes[color]):
                shapes[color] = [(0, 0), (0, 1), (1, 1), (1, 2)]
            else:
                shapes[color] = [(0, 0), (0, 1), (1, 1), (1, 2), (2, 1)]

    color2 = shapes.get(2, [])
    color3 = shapes.get(3, [])
    color4 = shapes.get(4, [])
    color6 = shapes.get(6, [])

    def block_left(cells, height):
        for dr in range(height):
            found = False
            for dc in range(W):
                if (dr, dc) in cells:
                    if found:
                        return dc
                    found = True
        return 0

    def block_top(cells, width):
        for dc in range(width):
            found = False
            for dr in range(H):
                if (dr, dc) in cells:
                    if found:
                        return dr
                    found = True
        return 0

    if color4:
        if (0, 0) in color4 and (0, 1) in color4 and (1, 1) in color4 and (2, 1) in color4:
            a = 4
            b = 3
            c = 0
            d = 1
        elif (0, 0) in color4 and (0, 1) in color4 and (1, 0) in color4 and (1, 1) in color4:
            a = 5
            b = 4
            c = 0
            d = 0
        elif len(color4) == 4:
            if (0, 1) in color4 and (1, 0) in color4 and (1, 2) in color4 and (2, 1) in color4:
                a = 4
                b = 3
                c = 0
                d = 1
            else:
                a = 4
                b = 3
                c = 0
                d = 1
        else:
            a = 4
            b = 3
            c = 0
            d = 1

        if color6:
            h6 = max(dr for dr, _ in color6) + 1
            w6 = max(dc for _, dc in color6) + 1
        else:
            h6 = 1
            w6 = 1

        if a <= 5:
            base = a - 1
        else:
            base = a + 1
            if len(color6) == 2:
                if h6 <= 2 and w6 >= 2:
                    base -= 1
        if b >= 4:
            base += 1

        c6r = base + d
        c6c = 1
        if c6r + h6 > H:
            c6r = max(0, H - h6)
        if c6c + w6 > W:
            c6c = max(0, W - w6)
        draw(color6, c6r, c6c, 6)

        if color3:
            if (0, 1) in color3 and (1, 0) in color3 and (1, 2) in color3 and (2, 1) in color3:
                t3r = 2
                t3c = 0
            else:
                t3r = 3
                t3c = 0
            t3c = block_left(color3, 4)
            t3c += 3
            t3r = block_top(color3, 4)
            t3r += 5
            if t3r + 4 > H:
                t3r = max(0, H - 4)
            if t3c + 4 > W:
                t3c = max(0, W - 4)
            draw(color3, t3r, t3c, 3)

        if color2:
            if (0, 0) in color2 and (0, 1) in color2 and (0, 2) in color2 and (0, 3) in color2:
                h2 = 1
            else:
                h2 = max(dr for dr, _ in color2) + 1
            t2r = 3
            if color2:
                t2r = block_top(color2, 4)
            t2r += 5
            t2c = block_left(color2, 4)
            t2c += 8
            if t2r + h2 > H:
                t2r = max(0, H - h2)
            if t2c + 4 > W:
                t2c = max(0, W - 4)
            draw(color2, t2r, t2c, 2)

    if color1:
        if (0, 1) in color1 and (1, 0) in color1 and (1, 1) in color1 and (1, 2) in color1:
            t1r = 1
            t1c = 0
        elif len(color1) >= 4 and all((0, c) in color1 for c in range(4)):
            t1r = 3
            t1c = 0
        else:
            t1r = 1
            t1c = 0
        h1 = max(dr for dr, _ in color1) + 1
        w1 = max(dc for _, dc in color1) + 1
        if color4:
            if (0, 0) in color4 and (0, 1) in color4 and (1, 1) in color4 and (2, 1) in color4:
                h4 = 3
                w4 = 2
            else:
                h4 = 2
                w4 = 2
            t1r = max(t1r, 5)
            t1c = min(W - w1, 7)
            t1r = min(t1r, max(0, H - h1))
            t1c = max(0, t1c)
        else:
            t1r = min(t1r, max(0, H - h1))
            t1c = min(t1c, max(0, W - w1))
        draw(color1, t1r, t1c, 1)

    if color7:
        if (0, 1) in color7 and (1, 0) in color7 and (1, 1) in color7 and (1, 2) in color7:
            t7r = 0
            t7c = 0
        else:
            t7r = 0
            t7c = 0
        h7 = max(dr for dr, _ in color7) + 1
        w7 = max(dc for _, dc in color7) + 1
        if color4:
            t7c = max(0, W - w7 - 1)
        else:
            t7c = max(0, W - w7)
        t7r = min(t7r, max(0, H - h7))
        draw(color7, t7r, t7c, 7)

    return out