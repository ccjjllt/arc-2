def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    colors = set()
    for row in grid:
        for v in row:
            if v != grid[0][0]:
                colors.add(v)

    bg = grid[0][0]
    for color in colors:
        cells = []
        for r in range(h):
            for c in range(w):
                if grid[r][c] == color:
                    cells.append((r, c))
        if not cells:
            continue

        rmin = min(r for r, _ in cells)
        rmax = max(r for r, _ in cells)
        cmin = min(c for _, c in cells)
        cmax = max(c for _, c in cells)
        width = cmax - cmin + 1

        # If object touches the grid border, leave it untouched.
        if rmin == 0 or rmax == h - 1 or cmin == 0 or cmax == w - 1:
            continue

        # Find a horizontal mirror axis that fixes the object as a set.
        # Prefer an axis that leaves the row lengths unchanged.
        best_axis = None
        best_score = -10**9

        for shift in range(0, 2 * width + 1):
            score = 0
            for r, c in cells:
                mc = cmin + cmax + shift - c
                if rmin <= r <= rmax and cmin <= mc <= cmax and grid[r][mc] == color:
                    score += 1
                else:
                    score -= 10
            # Prefer integer/fractional axes that produce no row-length change.
            if score > best_score:
                best_score = score
                best_axis = shift
                best_row_len_ok = True
                for rr in range(rmin, rmax + 1):
                    orig = sum(1 for cc in range(cmin, cmax + 1) if grid[rr][cc] == color)
                    new = 0
                    for cc in range(cmin, cmax + 1):
                        mc = cmin + cmax + best_axis - cc
                        if grid[rr][cc] == color or (cmin <= mc <= cmax and grid[rr][mc] == color):
                            new += 1
                    if new != orig:
                        best_row_len_ok = False
                        break
                if best_row_len_ok:
                    break

        # If the axis is not exact, try a smaller local symmetry around the object center.
        # This handles small protrusions that should be trimmed.
        center = (cmin + cmax) / 2.0
        # Use the most stable mirror: keep a cell only if its mirror exists, unless the
        # object is already stable under a full mirror.
        mirror_sum = 2 * center
        if not any(abs(mirror_sum - round(mirror_sum)) < 0.1 for _ in [0]):
            pass

        # Determine final mask by taking the largest mirror-closed subset.
        cell_set = set(cells)
        best_mask = None
        best_keep = -1
        for shift in range(0, 2 * width + 1):
            mirror_ok = True
            mask = set()
            for r, c in cells:
                mc = cmin + cmax + shift - c
                if not (cmin <= mc <= cmax) or (r, mc) not in cell_set:
                    mirror_ok = False
                    break
            if not mirror_ok:
                continue
            # For cells that are mirror-symmetric as a set, keep all cells.
            # Otherwise, keep only cells whose mirror exists.
            if best_keep < len(cells):
                best_keep = len(cells)
                best_mask = set(cells)
            break

        if best_mask is None:
            # Fallback: keep only cells whose mirror exists about the center.
            best_mask = set()
            for r, c in cells:
                mc = int(round(2 * center - c))
                if cmin <= mc <= cmax and (r, mc) in cell_set:
                    best_mask.add((r, c))
                    best_mask.add((r, mc))

        # If the full component is already mirror-symmetric, do not change it.
        # Otherwise, apply the largest stable mirror-closed subset if found; else fallback.
        # For the demos, stable objects are unchanged.
        changed = best_mask is not None and best_mask != set(cells)
        if changed:
            for r, c in cells:
                if (r, c) not in best_mask:
                    out[r][c] = bg

    return out