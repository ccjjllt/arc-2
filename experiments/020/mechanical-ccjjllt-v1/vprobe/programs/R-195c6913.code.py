def transform(grid):
    h = len(grid)
    w = len(grid[0])
    colors = set()
    for row in grid:
        for v in row:
            if v != 0:
                colors.add(v)
    colors.discard(0)
    if not colors:
        return [row[:] for row in grid]

    # Choose the dominant border/frame color: the color touching the most edges.
    bg = max(colors, key=lambda c: (
        sum(1 for j in range(w) if grid[0][j] == c) +
        sum(1 for j in range(w) if grid[h - 1][j] == c) +
        sum(1 for i in range(h) if grid[i][0] == c) +
        sum(1 for i in range(h) if grid[i][w - 1] == c),
        c
    ))

    pattern_colors = [c for c in sorted(colors) if c != bg]
    if not pattern_colors:
        return [row[:] for row in grid]

    # Find 2x2 solid blocks of a non-background color.
    blocks = []
    for c in pattern_colors:
        for i in range(h - 1):
            row = grid[i]
            nxt = grid[i + 1]
            for j in range(w - 1):
                if row[j] == c and row[j + 1] == c and nxt[j] == c and nxt[j + 1] == c:
                    blocks.append((c, i, j))

    if not blocks:
        return [row[:] for row in grid]

    # Reconstruct the repeating 3x4 pattern from all observed blocks.
    pat = {}
    for c, i, j in blocks:
        for r in range(2):
            for col in range(2):
                pat[(i + r, j + col)] = c

    pcol = {}
    for (r, c), v in pat.items():
        pcol.setdefault(r, {})[c] = v

    def val(r, c):
        return pcol.get(r, {}).get(c, 0)

    def put(g, r, c, v):
        if r >= 0 and r < h and 0 <= c < w:
            g[r][c] = v

    out = [row[:] for row in grid]

    # Place patterns whose 2x2 anchor fits entirely in background.
    for r in range(h - 1):
        for c in range(w - 1):
            if grid[r][c] == 0 and grid[r][c + 1] == 0 and grid[r + 1][c] == 0 and grid[r + 1][c + 1] == 0:
                for rr in range(3):
                    for cc in range(4):
                        v = val(rr, cc)
                        if v:
                            put(out, r + rr, c + cc, v)

    # If nothing could be placed, fall back to every 2x2 background window.
    if all(out[i][j] == grid[i][j] for i in range(h) for j in range(w)):
        out = [row[:] for row in grid]
        for r in range(h - 1):
            for c in range(w - 1):
                if grid[r][c] == 0 and grid[r][c + 1] == 0 and grid[r + 1][c] == 0 and grid[r + 1][c + 1] == 0:
                    for rr in range(2):
                        for cc in range(2):
                            v = val(rr, cc)
                            if v:
                                put(out, r + rr, c + cc, v)

    # Fill any remaining isolated background cells using the local 2x2 pattern.
    for r in range(h):
        for c in range(w):
            if out[r][c] != 0:
                continue
            best = 0
            for dr in (-2, -1, 0, 1, 2):
                for dc in (-3, -2, -1, 0, 1, 2, 3):
                    r0, c0 = r - dr, c - dc
                    if 0 <= r0 < h - 1 and 0 <= c0 < w - 1:
                        if grid[r0][c0] == 0 and grid[r0][c0 + 1] == 0 and \
                           grid[r0 + 1][c0] == 0 and grid[r0 + 1][c0 + 1] == 0:
                            v = val(dr, dc)
                            if v:
                                best = v
                                break
                if best:
                    break
            if best:
                out[r][c] = best

    return out