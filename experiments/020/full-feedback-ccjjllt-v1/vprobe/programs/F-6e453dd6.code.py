def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    sep = -1
    for c in range(w):
        if all(grid[r][c] == 5 for r in range(h)):
            sep = c
            break
    if sep <= 0 or sep >= w - 1:
        return out

    left = sep
    right_start = sep + 1

    zero_set = set()
    for r in range(h):
        for c in range(left):
            if grid[r][c] == 0:
                zero_set.add((r, c))

    seen = set()
    components = []
    for start in sorted(zero_set):
        if start in seen:
            continue
        stack = [start]
        comp = []
        seen.add(start)
        while stack:
            r, c = stack.pop()
            comp.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if (nr, nc) in zero_set and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        components.append(comp)

    for comp in components:
        min_r = min(r for r, c in comp)
        max_r = max(r for r, c in comp)
        min_c = min(c for r, c in comp)
        max_c = max(c for r, c in comp)
        shift = (sep - 1) - max_c
        if shift < 0:
            shift = 0

        comp_set = set((r + shift, c + shift) for r, c in comp)

        for r, c in comp:
            out[r][c] = 6
        for r, c in comp_set:
            if r >= 0 and r < h and c < w:
                out[r][c] = 0

        if max_c == sep - 1:
            right_col = sep - 1
            rows_to_mark = set()
            for r in range(min_r, max_r + 1):
                for c in range(max_c + 1, sep):
                    if grid[r][c] == 6 and grid[r - 1][c] == 0 and grid[r + 1][c] == 0 and grid[r][c - 1] == 0 and grid[r][c + 1] == 0:
                        if (r, sep - 1) in comp_set:
                            rows_to_mark.add(r)
            for r in rows_to_mark:
                for c in range(right_start, w):
                    out[r][c] = 2

    return out