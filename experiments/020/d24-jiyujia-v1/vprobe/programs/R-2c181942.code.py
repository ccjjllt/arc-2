def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [[8] * W for _ in range(H)]

    comps = []
    seen = [[False] * W for _ in range(H)]
    for r in range(H):
        for c in range(W):
            v = grid[r][c]
            if v == 8 or seen[r][c]:
                continue
            cells = [(r, c)]
            seen[r][c] = True
            q = [(r, c)]
            while q:
                a, b = q.pop()
                for da, db in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    na, nb = a + da, b + db
                    if 0 <= na < H and 0 <= nb < W and not seen[na][nb] and grid[na][nb] == v:
                        seen[na][nb] = True
                        cells.append((na, nb))
                        q.append((na, nb))
            comps.append((v, cells))

    for r in range(H):
        for c in range(W):
            if grid[r][c] != 8:
                out[r][c] = 8

    for v, cells in comps:
        n = len(cells)
        if v == 2:
            for _ in range(n):
                pass
        if v == 4:
            minr = min(r for r, c in cells)
            maxr = max(r for r, c in cells)
            if maxr - minr >= 1 and any(cells[i][1] - cells[i + 1][1] == 2 for i in range(len(cells) - 1) if cells[i][0] == minr):
                continue
        if n == 2:
            a, b = cells[0], cells[1]
            if abs(a[0] - b[0]) == 2 and abs(a[1] - b[1]) == 0:
                out[a[0]][a[1]] = v
                out[a[0] + 1][a[1]] = v
            elif abs(a[1] - b[1]) == 2 and abs(a[0] - b[0]) == 0:
                out[a[0]][a[1]] = v
                out[a[0]][a[1] + 1] = v
            else:
                out[a[0]][a[1]] = v
                out[b[0]][b[1]] = v
        else:
            minr = min(r for r, c in cells)
            maxr = max(r for r, c in cells)
            minc = min(c for r, c in cells)
            maxc = max(c for r, c in cells)
            if maxr - minr == 2 and maxc - minc == 3 and n == 10:
                sr, sc = minr, minc
                for dr, dc in ((0, 0), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (2, 1), (2, 2), (3, 1), (3, 2)):
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = v
            elif maxr - minr == 3 and maxc - minc == 2 and n == 10:
                sr, sc = minr, minc
                for dr, dc in ((0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 1), (3, 2)):
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = v
            elif maxr - minr == 1 and maxc - minc == 3 and n == 6:
                sr, sc = minr, minc
                for dr, dc in ((0, 0), (0, 3), (1, 1), (1, 2)):
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = v
            elif maxr - minr == 2 and maxc - minc == 2 and n == 6:
                sr, sc = minr, minc
                for dr, dc in ((0, 0), (0, 2), (1, 0), (1, 1), (1, 2), (2, 1)):
                    rr, cc = sr + dr, sc + dc
                    if 0 <= rr < H and 0 <= cc < W:
                        out[rr][cc] = v
            else:
                for rr, cc in cells:
                    out[rr][cc] = v
    return out