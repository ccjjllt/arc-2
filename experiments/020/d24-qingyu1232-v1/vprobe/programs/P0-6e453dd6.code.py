def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    # Find the separator column containing only 5s.
    sep = -1
    for c in range(W):
        if all(grid[r][c] == 5 for r in range(H)):
            sep = c
            break
    if sep == -1:
        return out

    left_w = sep
    right_w = W - sep - 1
    if left_w <= 0 or right_w <= 0:
        return out

    # Collect zero pixels on the left side.
    zeros = []
    for r in range(H):
        for c in range(left_w):
            if grid[r][c] == 0:
                zeros.append((r, c))

    # Group zero pixels into vertical components separated by all-background rows.
    comps = []
    r = 0
    while r < H:
        if any(grid[r][c] == 0 for c in range(left_w)):
            start = r
            while r < H and any(grid[r][c] == 0 for c in range(left_w)):
                r += 1
            comp = [grid[i][:] for i in range(start, r)]
            comps.append((start, comp))
        else:
            r += 1

    for start, comp in comps:
        h = len(comp)
        if h == 0:
            continue

        # Bounding box of the component.
        min_r = 0
        max_r = h - 1
        min_c = left_w
        max_c = -1
        for rr in range(h):
            for cc in range(left_w):
                if comp[rr][cc] == 0:
                    if cc < min_c:
                        min_c = cc
                    if cc > max_c:
                        max_c = cc

        width = max_c - min_c + 1

        # Maximum vertical length of any enclosed background hole.
        hole_rows = set()
        for rr in range(h):
            for cc in range(min_c + 1, max_c):
                if comp[rr][cc] == 6:
                    # Check four-neighbor enclosure inside the bounding box.
                    if comp[rr - 1][cc] == 0 and comp[rr + 1][cc] == 0 and \
                       comp[rr][cc - 1] == 0 and comp[rr][cc + 1] == 0:
                        hole_rows.add(rr)

        # Shift right until the component's right edge touches the separator.
        shift = sep - 1 - max_c
        if shift < 0:
            shift = 0

        # Move the component: clear old zeros, set shifted zeros.
        for dr in range(h):
            r2 = start + dr
            for c in range(left_w):
                if comp[dr][c] == 0:
                    out[r2][c] = 6
                    nc = c + shift
                    out[r2][nc] = 0

        # Mark right-side rows for holes that have vertical length at least 2
        # or for single-cell holes when the component height is at most 4.
        mark_rows = []
        for rr in hole_rows:
            length = 1
            rr2 = rr - 1
            while rr2 >= 0 and rr2 in hole_rows and comp[rr2][0 + min_c] == 6 or (
                rr2 >= 0 and rr2 in hole_rows
            ):
                # Simpler: count consecutive hole rows in the set.
                length += 1
                rr2 -= 1
            if length >= 2 or h <= 4:
                mark_rows.append(rr)

        # The above loop is not robust for multi-cell hole rows; compute properly.
        mark_rows = []
        if hole_rows:
            rows_sorted = sorted(hole_rows)
            i = 0
            while i < len(rows_sorted):
                j = i
                while j + 1 < len(rows_sorted) and rows_sorted[j + 1] == rows_sorted[j] + 1:
                    j += 1
                length = j - i + 1
                if length >= 2 or h <= 4:
                    mark_rows.extend(rows_sorted[i:j + 1])
                i = j + 1

        for rr in mark_rows:
            r2 = start + rr
            for c in range(sep + 1, W):
                out[r2][c] = 2

    return out