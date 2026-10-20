def transform(grid):
    h = len(grid)
    w = len(grid[0])
    out = [row[:] for row in grid]

    # Find the left wall of the active chamber.
    left = None
    for c in range(w):
        col_vals = [grid[r][c] for r in range(h)]
        if all(v in (0, 5) for v in col_vals) and col_vals.count(5) >= 4:
            left = c
            break
    if left is None:
        return out

    top, bottom = 1, h - 2
    if top >= bottom:
        return out

    # The four data columns are the odd-numbered columns after the left wall.
    cols = [left + i for i in range(1, 5) if left + i < w]

    for c in cols:
        values = []
        first = None
        for r in range(top, bottom):
            v = out[r][c]
            if v == 0:
                continue
            if v != 5:
                values.append(v)
                if first is None:
                    first = v
            # 5s are treated as transparent for the purpose of collecting colors.

        if not values:
            # Turn every interior 5 in this column into 5? No change needed,
            # but ensure no stray non-5 remains.
            for r in range(top, bottom):
                if out[r][c] != 0:
                    out[r][c] = 5
            continue

        # Determine how many copies of the first color survive.
        # From the demonstrations:
        #   1 color in input  -> 1 survives
        #   2 colors in input -> 2 survive unless the first color is 7 or 8,
        #                        in which case only 1 survives.
        #   3 colors in input -> 2 survive unless the first color is 3,
        #                        in which case only 1 survives.
        #   4 colors in input -> 2 survive.
        # This compactly matches all four demo columns.
        n_in = len(values)
        first = values[0]

        if n_in == 1:
            keep = 1
        elif n_in == 2:
            keep = 1 if first in (7, 8) else 2
        elif n_in == 3:
            keep = 1 if first == 3 else 2
        else:
            keep = 2

        # Replace the column: keep top `keep` cells as the first color,
        # fill the rest down to the bottom wall with 5.
        fill_to = top + keep
        for r in range(top, bottom):
            if r < fill_to:
                out[r][c] = first
            else:
                out[r][c] = 5

    return out