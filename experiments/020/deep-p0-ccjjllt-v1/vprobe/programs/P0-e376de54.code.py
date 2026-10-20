def transform(grid):
    if not grid or not grid[0]:
        return grid

    h = len(grid)
    w = len(grid[0])

    def norm(v):
        a = abs(v[0])
        b = abs(v[1])
        g = a if a > b else b
        while g:
            a, b = b, a % b
            g = b
        return (v[0] // g, v[1] // g)

    def add(p, q):
        return (p[0] + q[0], p[1] + q[1])

    def sub(p, q):
        return (p[0] - q[0], p[1] - q[1])

    def cross(a, b):
        return a[0] * b[1] - a[1] * b[0]

    def orient(a, b, c):
        return cross(sub(b, a), sub(c, a))

    def collinear(a, b, c):
        return orient(a, b, c) == 0

    def on_segment(a, b, c):
        # c on segment a-b inclusive
        return (collinear(a, b, c) and
                min(a[0], b[0]) <= c[0] <= max(a[0], b[0]) and
                min(a[1], b[1]) <= c[1] <= max(a[1], b[1]))

    def find_endpoints(pts, direction):
        best_s = None
        best_t = None
        best_len = -1
        s = 0
        while s < len(pts) + 1:
            t = s
            while t < len(pts) and (sub(pts[t], pts[s])) == (direction[0] * (t - s), direction[1] * (t - s)):
                t += 1
            if t - s > best_len:
                best_len = t - s
                best_s = s
                best_t = t
            s += 1
        return pts[best_s], pts[best_t]

    def median(values):
        xs = sorted(values)
        n = len(xs)
        return xs[(n - 1) // 2]

    # collect all non-background pixels
    nonbg = {}
    for r in range(h):
        for c in range(w):
            if grid[r][c] != 7:
                nonbg.setdefault(grid[r][c], []).append((r, c))

    # detect all maximal collinear pixel segments of length >= 2
    seg_info = []
    seen = set()
    for color, pts in nonbg.items():
        # sort to make endpoints stable
        points = sorted(set(pts))
        n = len(points)
        for i in range(n):
            for j in range(i + 1, n):
                p1 = points[i]
                p2 = points[j]
                d = norm(sub(p2, p1))
                # only consider primitive directions that can be pixel steps
                if d[0] == 0 and d[1] == 0:
                    continue
                # find all points on this line
                line_pts = []
                for p in points:
                    if collinear(p1, p2, p):
                        line_pts.append(p)
                if len(line_pts) < 2:
                    continue
                # sort by progression along direction
                # use dot product with direction as key
                def key(p):
                    return (p[0] * d[0] + p[1] * d[1], p[0], p[1])
                line_pts.sort(key=key)
                # split into consecutive runs separated by at most 1 step
                runs = []
                cur = [line_pts[0]]
                for idx in range(1, len(line_pts)):
                    a = line_pts[idx - 1]
                    b = line_pts[idx]
                    diff = sub(b, a)
                    if diff == d:
                        cur.append(b)
                    else:
                        if len(cur) >= 2:
                            runs.append(cur)
                        cur = [b]
                if len(cur) >= 2:
                    runs.append(cur)
                for run in runs:
                    # canonical frozenset of points
                    fp = frozenset(run)
                    if fp in seen:
                        continue
                    seen.add(fp)
                    start, end = run[0], run[-1]
                    # direction from start to end, normalized
                    dnorm = norm(sub(end, start))
                    length = len(run)
                    # endpoint keys for median selection:
                    # use the endpoint with smaller (row, col) as 'start_key'
                    skey = min(start, end)
                    ekey = max(start, end)
                    seg_info.append({
                        "color": color,
                        "start": start,
                        "end": end,
                        "dir": dnorm,
                        "length": length,
                        "skey": skey,
                        "ekey": ekey,
                        "pts": list(run),
                    })

    if not seg_info:
        return [row[:] for row in grid]

    # group segments by direction
    groups = {}
    for seg in seg_info:
        groups.setdefault(seg["dir"], []).append(seg)

    # Choose reference segment per group:
    # Prefer the segment whose start endpoint is the median among all start endpoints
    # (lexicographic median by row, then column). If tie, use length then endpoint.
    chosen_refs = {}
    for direction, segs in groups.items():
        starts = sorted([s["skey"] for s in segs])
        ref_start = starts[len(starts) // 2]
        # If multiple segments have same start, choose by end key then length
        cand = [s for s in segs if s["skey"] == ref_start]
        cand.sort(key=lambda s: (s["ekey"], -s["length"], s["color"]))
        chosen_refs[direction] = cand[0]

    # Determine new pixels to add/remove for each group
    # We resize segments to match the reference segment's endpoint line on the
    # adjustable side, while keeping the fixed side endpoint unchanged.
    # The adjustable side is the side that is not common among the group if one
    # side is exactly common; otherwise it is the 'end' endpoint side (larger
    # lexicographic endpoint) for horizontal/vertical, and the 'start' endpoint
    # side for diagonal anti segments as seen in the demo.
    # To keep the rule general and simple, we infer the adjustable side from the
    # reference and the set of fixed endpoints: use the side where the group has
    # variation; if both vary, use the side that makes fewer cells change.

    out = [row[:] for row in grid]

    for direction, segs in groups.items():
        ref = chosen_refs[direction]

        # Try two possible adjustable sides:
        # side=0 means keep the 'start' endpoint and adjust 'end'
        # side=1 means keep the 'end' endpoint and adjust 'start'
        best_side = None
        best_cost = None
        best_plans = None

        for side in (0, 1):
            plans = []
            cost = 0
            ok = True
            for s in segs:
                if side == 0:
                    fixed = s["start"]
                    # target adjustable point: project reference end line onto s's line
                    # We want the new endpoint so that the vector from fixed to new
                    # is the same length as reference, in the same direction.
                    ref_len = ref["length"]
                    # reference vector from start to end
                    v = sub(ref["end"], ref["start"])
                    new_adj = add(fixed, v)
                else:
                    fixed = s["end"]
                    ref_len = ref["length"]
                    v = sub(ref["start"], ref["end"])
                    new_adj = add(fixed, v)

                # Ensure new_adj is on the same infinite line as fixed and direction.
                # It should be by construction, but clamp to grid.
                new_pts = []
                step = direction
                cur = fixed
                target = new_adj
                # move from fixed to target
                dr = target[0] - cur[0]
                dc = target[1] - cur[1]
                nsteps = abs(dr)
                if dr == 0:
                    nsteps = abs(dc)
                # step sign should equal direction or -direction
                if nsteps == 0:
                    new_pts.append(cur)
                else:
                    # determine step
                    if (dr * step[0] > 0 or dc * step[1] > 0):
                        st = step
                    else:
                        st = (-step[0], -step[1])
                    for k in range(nsteps + 1):
                        p = (fixed[0] + st[0] * k, fixed[1] + st[1] * k)
                        if not (0 <= p[0] < h and 0 <= p[1] < w):
                            ok = False
                            break
                        new_pts.append(p)
                    if not ok:
                        break
                if not ok:
                    break

                # cost: cells removed from old not in new + cells added in new not old
                old_set = set(s["pts"])
                new_set = set(new_pts)
                cost += len(old_set - new_set) + len(new_set - old_set)
                plans.append((s, new_set))

            if ok and (best_cost is None or cost < best_cost):
                best_cost = cost
                best_side = side
                best_plans = plans

        if best_plans is None:
            # fallback: do nothing for this group
            continue

        # Apply changes for this group: first clear old pixels of all segments in group,
        # then set new pixels, preserving background and other colors.
        clear_pts = []
        for s, _ in best_plans:
            clear_pts.extend(s["pts"])
        for p in clear_pts:
            out[p[0]][p[1]] = 7

        for s, new_set in best_plans:
            for p in new_set:
                out[p[0]][p[1]] = s["color"]

    return out