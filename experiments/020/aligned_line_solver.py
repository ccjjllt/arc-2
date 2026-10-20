#!/usr/bin/env python3
"""One fixed CPU rule: resize parallel lines to the spatial middle line.

Only demonstration outputs select whether this solver is applicable to a task.
Query outputs are never read. Unsupported or ambiguous inputs return no grid.
This module does not execute model-written programs or change their executor.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import time
from pathlib import Path


def transform(grid):
    """Pure, dependency-free candidate program; [] means unsupported input."""
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    counts = {}
    for row in grid:
        if len(row) != width:
            return []
        for value in row:
            counts[value] = counts.get(value, 0) + 1
    background = min(counts, key=lambda value: (-counts[value], value))
    pixels = set((r, c) for r in range(height) for c in range(width)
                 if grid[r][c] != background)
    if not pixels:
        return []

    def inside(r, c):
        return 0 <= r < height and 0 <= c < width

    predictions = []
    for dr, dc in ((0, 1), (1, 0), (1, 1), (1, -1)):
        segments = []
        covered = set()
        for r, c in sorted(pixels):
            color = grid[r][c]
            pr, pc = r - dr, c - dc
            if inside(pr, pc) and grid[pr][pc] == color:
                continue
            cells = []
            rr, cc = r, c
            while inside(rr, cc) and grid[rr][cc] == color:
                cells.append((rr, cc))
                rr += dr
                cc += dc
            if len(cells) < 2:
                continue
            covered.update(cells)
            # Projection on the normal gives spatial order across parallel lines.
            normal_offset = -dc * r + dr * c
            segments.append((normal_offset, color, cells))

        # This rule describes the entire foreground, not a hand-picked fragment.
        if covered != pixels or len(segments) < 3 or len(segments) % 2 == 0:
            continue
        if len(set(segment[0] for segment in segments)) != len(segments):
            continue
        segments.sort(key=lambda segment: segment[0])
        target_length = len(segments[len(segments) // 2][2])
        first_projections = [cells[0][0] * dr + cells[0][1] * dc
                             for _, _, cells in segments]
        last_projections = [cells[-1][0] * dr + cells[-1][1] * dc
                            for _, _, cells in segments]
        first_span = max(first_projections) - min(first_projections)
        last_span = max(last_projections) - min(last_projections)
        # Require an actual shared baseline; do not extrapolate loose alignment.
        if min(first_span, last_span) != 0:
            continue
        keep_first = first_span <= last_span
        sign = 1 if keep_first else -1
        output = [[background] * width for _ in range(height)]
        fits = True
        for _, color, cells in segments:
            anchor = cells[0] if keep_first else cells[-1]
            for step in range(target_length):
                rr = anchor[0] + sign * step * dr
                cc = anchor[1] + sign * step * dc
                if not inside(rr, cc):
                    fits = False
                    break
                output[rr][cc] = color
            if not fits:
                break
        if fits and output not in predictions:
            predictions.append(output)
    # Distinct valid orientation interpretations must agree before predicting.
    return predictions[0] if len(predictions) == 1 else []


def candidate_code():
    """Return the same general rule, without embedding task data or answers."""
    return inspect.getsource(transform)


def solve_task(task):
    """Accept a candidate only after all demos pass and a demo is non-identity."""
    demos = task.get("train", [])
    demo_results = []
    for index, demo in enumerate(demos):
        predicted = transform(demo["input"])
        demo_results.append({
            "demo_index": index,
            "supported": bool(predicted),
            "exact": predicted == demo["output"],
            "input_differs_from_output": demo["input"] != demo["output"],
            "prediction": predicted if predicted else None,
        })
    matches = sum(result["exact"] for result in demo_results)
    changed = sum(result["input_differs_from_output"] for result in demo_results)
    all_exact = bool(demos) and matches == len(demos)
    admitted = all_exact and changed > 0
    queries = []
    if admitted:
        for index, query in enumerate(task.get("test", [])):
            prediction = transform(query["input"])
            queries.append({
                "query_index": index,
                "supported": bool(prediction),
                "grid": prediction if prediction else None,
            })
    return {
        "demo_total": len(demos),
        "demo_exact": matches,
        "all_demos_exact": all_exact,
        "changed_demo_count": changed,
        "admitted": admitted,
        "demo_results": demo_results,
        "candidate_code": candidate_code() if admitted else None,
        "query_predictions": queries,
        "query_gold_used": False,
    }


def solve_challenges(challenges):
    started = time.monotonic()
    tasks = {task_id: solve_task(task) for task_id, task in challenges.items()}
    accepted = [task_id for task_id, result in tasks.items() if result["admitted"]]
    return {
        "solver": "parallel_lines_spatial_middle_shared_endpoint_v1",
        "task_count": len(challenges),
        "query_count": sum(len(task.get("test", [])) for task in challenges.values()),
        "admitted_task_count": len(accepted),
        "admitted_task_ids": accepted,
        "supported_query_count": sum(
            query["supported"] for result in tasks.values()
            for query in result["query_predictions"]
        ),
        "elapsed_cpu_wall_seconds": time.monotonic() - started,
        "tasks": tasks,
        "query_score_status": "not_scored",
        "query_gold_used": False,
        "rule_parameters_fitted_to_query_answers": False,
        "limitations": [
            "Assumes a most-frequent background and complete foreground made of "
            "at least three odd-count, parallel, monochromatic straight segments.",
            "Only horizontal, vertical and 45-degree diagonals are represented; "
            "every segment must have at least two cells and its own parallel line.",
            "Requires an exactly aligned endpoint baseline and an in-bounds output; "
            "ambiguous orientation outputs produce no prediction.",
            "All-demo exact is applicability evidence, not a query correctness guarantee.",
            "The rule was developed from e376de54 demos and its query input after a "
            "prior miss was reported. D24 coverage is development evidence, not "
            "an unseen validation result.",
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    path = Path(args.challenges)
    raw = path.read_bytes()
    result = solve_challenges(json.loads(raw))
    result["input_path"] = str(path)
    result["input_sha256"] = hashlib.sha256(raw).hexdigest()
    result["solver_source_sha256"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "task_count", "query_count", "admitted_task_count", "admitted_task_ids",
        "supported_query_count", "elapsed_cpu_wall_seconds", "query_score_status",
    )}))


if __name__ == "__main__":
    main()
