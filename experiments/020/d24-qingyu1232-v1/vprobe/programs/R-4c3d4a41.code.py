def transform(grid):
    H = len(grid)
    W = len(grid[0])
    out = [row[:] for row in grid]

    # Identify the two active column groups separated by the main vertical wall.
    cols = sorted(set(i for j, row in enumerate(grid) for i, v in enumerate(row) if v != 0))
    if not cols:
        return out

    def col_group(c):
        # Active columns are the odd positions inside each half; use the vertical wall
        # column (always 5 in the top row) to separate left and right halves.
        wall = None
        for cc in range(W):
            if all(grid[r][cc] == 5 for r in range(H) if grid[r][cc] != 0 or cc == 9):
                pass
        # Simpler robust separator: first column that is 5 across all rows that contain right border.
        # In all tasks the separator is the first column where top row is 5.
        for cc in range(W):
            if grid[0][cc] == 5:
                wall = cc
                break
        if wall is None:
            wall = W // 2
        return 0 if c < wall else 1

    # Find active columns in each group: columns that contain a non-0, non-5 value.
    active_cols = [0, 0]
    for g in range(2):
        cset = set(i for j in range(H) for i in range(W)
                   if grid[j][i] not in (0, 5) and col_group(i) == g)
        # If no digits in a group, infer from 5 pattern or use odd columns near wall.
        if cset:
            active_cols[g] = sorted(cset)
        else:
            active_cols[g] = []

    # In the given tasks there are exactly four active columns in the right group.
    # Use left group pattern to infer right columns if needed.
    if len(active_cols[1]) != 4 and len(active_cols[0]) == 4:
        wall = None
        for cc in range(W):
            if grid[0][cc] == 5:
                wall = cc
                break
        offset = active_cols[1][0] - active_cols[0][0] if active_cols[1] else (wall - active_cols[0][0] if active_cols[0] else 10)
        active_cols[1] = [c + offset for c in active_cols[0]]

    if len(active_cols[1]) != 4:
        # Fallback: choose four odd columns in the right half.
        wall = None
        for cc in range(W):
            if grid[0][cc] == 5:
                wall = cc
                break
        if wall is None:
            wall = W // 2
        right_odd = [i for i in range(wall + 1, W) if i % 2 == 1 and i % 4 == 1]
        if len(right_odd) >= 4:
            active_cols[1] = right_odd[:4]
        else:
            active_cols[1] = list(range(wall + 1, wall + 5))

    # Determine rows of the 5x4 interior: use rows 1 through 5 (first five non-border rows)
    rows = list(range(1, min(6, H)))

    # Extract the right interior 5x4 state (top to bottom, left to right active cols)
    A = active_cols[1]
    right_state = [[grid[r][c] for c in A] for r in rows]

    # Extract the left 5 pattern in the same row range. Left active columns if available.
    if len(active_cols[0]) == 4:
        L = active_cols[0]
        left_pattern = [[1 if grid[r][c] == 5 else 0 for c in L] for r in rows]
    else:
        # Infer left pattern from output-like default if no left digits/walls: use a full triangle
        left_pattern = [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ]

    # The rule:
    # For each active column, keep the contiguous non-zero/non-5 digit block, move it up
    # until its bottom sits at the first row where the left pattern has a 5 in that column,
    # and fill the cells below that bottom with 5s according to the left pattern.
    # The topmost row of the moved block may replace a 5.
    result = [[0 for _ in range(4)] for _ in range(5)]

    # Compute digit blocks per column from input right interior.
    for ci in range(4):
        col = [right_state[ri][ci] for ri in range(5)]
        # Find all non-zero/non-5 cells.
        digit_rows = [ri for ri, v in enumerate(col) if v not in (0, 5)]
        if not digit_rows:
            continue

        # Group contiguous digit rows (allow different colors in same contiguous run? keep as is)
        groups = []
        start = digit_rows[0]
        prev = digit_rows[0]
        for r in digit_rows[1:]:
            if r == prev + 1:
                prev = r
            else:
                groups.append((start, prev))
                start = r
                prev = r
        groups.append((start, prev))

        # For each contiguous group, treat it as a block. In tasks there is at most one group per column,
        # except query0 has two different colors stacked contiguously; move the whole stacked block together.
        # Determine target bottom row from left pattern: first row in [1..4] where pattern is 1.
        target_bottom = None
        for ri in range(1, 5):
            if left_pattern[ri][ci]:
                target_bottom = ri
                break
        if target_bottom is None:
            target_bottom = 4

        # Use the bottommost group's bottom as anchor? Move so that the bottom of the entire digit run
        # reaches target_bottom. Preserve internal order and length of the contiguous run.
        total_digit_cells = len(digit_rows)
        # If there are gaps (not contiguous), compress? In given tasks no gaps in digit runs except 5s separating colors;
        # here we only consider contiguous non-0/non-5. If multiple groups, use the union as a block compressed.
        if len(groups) == 1:
            s, e = groups[0]
            length = e - s + 1
            new_bottom = max(s, min(target_bottom, 4))
            # If target bottom is above current top, place at top (row 0) preserving length.
            if target_bottom < s:
                new_top = 0
            else:
                new_top = max(0, new_bottom - length + 1)
            new_bottom = new_top + length - 1
            for k in range(length):
                if new_top + k >= 0 and new_top + k < 5:
                    result[new_top + k][ci] = col[s + k]
        else:
            # Compress the contiguous colored cells into a single block preserving order.
            seq = [col[r] for r in digit_rows]
            length = len(seq)
            if target_bottom < digit_rows[0]:
                new_top = 0
            else:
                new_top = max(0, target_bottom - length + 1)
            for k in range(length):
                if new_top + k < 5:
                    result[new_top + k][ci] = seq[k]

    # Fill 5s according to left pattern below/beside digits, but do not overwrite existing digits.
    for ri in range(5):
        for ci in range(4):
            if result[ri][ci] == 0 and left_pattern[ri][ci]:
                result[ri][ci] = 5

    # Write back to the grid. Keep all other cells unchanged, but clear the original right interior
    # active cells that are not in result to remove moved digits/walls that should disappear.
    for ri in range(5):
        r = rows[ri]
        for ci in range(4):
            c = A[ci]
            out[r][c] = result[ri][ci]

    return out