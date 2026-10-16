"""Build a complete ARC submission from this run's trusted worker outputs.

Only base-pass candidates are used. This script never reads solutions and does
not import the model/tokenizer stack. The worker pickle format may require numpy
when actually loading candidates; --help and an empty-output build use stdlib.
Do not point --outputs at downloaded or otherwise untrusted pickle files.
"""

import argparse
import bz2
import json
import math
import numbers
import os
from pathlib import Path
import pickle
import tempfile


SELECTION_POLICY = {
    "source": "base_only",
    "algorithm": "score_kgmon",
    "metric": "support_count - mean(mean(score_aug) per candidate)",
    "unique_grids": True,
    "attempts": 2,
    "tie_breaks": ["support_desc", "beam_best_asc", "aug_mean_asc", "grid_shape_and_cells_asc"],
    "missing_attempt": [[0]],
}


def atomic_json(path, value):
    """Readers see either the previous complete JSON or the new complete JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as stream:
            tmp_path = Path(stream.name)
            json.dump(value, stream, allow_nan=False, separators=(",", ":"))
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path is not None and tmp_path.exists():
            tmp_path.unlink()


def initial_submission(challenges):
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError("Challenges must be a nonempty task object")
    submission = {}
    query_map = {}
    for task_id, task in challenges.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("Invalid challenge task")
        queries = task.get("test")
        if not isinstance(queries, list) or not queries:
            raise ValueError(f"Task {task_id!r} has no test queries")
        submission[task_id] = []
        for index, query in enumerate(queries):
            if not isinstance(query, dict) or "input" not in query:
                raise ValueError(f"Invalid test query {task_id}_{index}")
            submission[task_id].append({"attempt_1": [[0]], "attempt_2": [[0]]})
            query_map[f"{task_id}_{index}"] = (task_id, index)
    return submission, query_map


def valid_grid(value):
    """Normalize integer arrays while rejecting ragged/floating/bool grids."""
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)) or not 1 <= len(value) <= 30:
        raise ValueError("Grid height must be 1..30")
    width = None
    normalized = []
    for row in value:
        if not isinstance(row, (list, tuple)) or not 1 <= len(row) <= 30:
            raise ValueError("Grid width must be 1..30")
        if width is None:
            width = len(row)
        if len(row) != width:
            raise ValueError("Grid must be rectangular")
        converted = []
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, numbers.Integral):
                raise ValueError("Grid cells must be integers")
            if not 0 <= cell <= 9:
                raise ValueError("Grid cells must be 0..9")
            converted.append(int(cell))
        normalized.append(converted)
    return normalized


def existing_submission_matches(path, reference):
    """Keep an already valid snapshot while a later candidate scan is running."""
    try:
        with open(path, encoding="utf-8") as stream:
            existing = json.load(stream)
        if not isinstance(existing, dict) or set(existing) != set(reference):
            return False
        for task_id, expected_queries in reference.items():
            queries = existing[task_id]
            if not isinstance(queries, list) or len(queries) != len(expected_queries):
                return False
            for query in queries:
                if not isinstance(query, dict) or set(query) != {"attempt_1", "attempt_2"}:
                    return False
                for grid in query.values():
                    valid_grid(grid)
        return True
    except (OSError, ValueError, TypeError):
        return False


def finite_score(value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError("Scores must be real numbers")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("Scores must be finite")
    return result


def normalize_candidate(sample):
    if not isinstance(sample, dict):
        raise ValueError("Candidate must be a dictionary")
    grid = valid_grid(sample["solution"])
    scores = sample["score_aug"]
    if hasattr(scores, "tolist"):
        scores = scores.tolist()
    if not isinstance(scores, (list, tuple)) or not scores:
        raise ValueError("score_aug must be a nonempty vector")
    scores = [finite_score(value) for value in scores]
    return {
        "solution": grid,
        "beam_score": finite_score(sample["beam_score"]),
        "score_aug": scores,
    }


def select_kgmon(candidates):
    """Match the existing decoder's kgmon metric and deterministic tie breaks."""
    groups = {}
    for candidate in candidates:
        grid_key = tuple(tuple(row) for row in candidate["solution"])
        groups.setdefault(grid_key, []).append(candidate)
    ranked = []
    for grid_key, values in groups.items():
        support = len(values)
        aug_mean = sum(sum(v["score_aug"]) / len(v["score_aug"]) for v in values) / support
        metric = support - aug_mean
        beam_best = min(v["beam_score"] for v in values)
        signature = ((len(grid_key), len(grid_key[0])), tuple(cell for row in grid_key for cell in row))
        ranked.append(((-metric, -support, beam_best, aug_mean, signature), values[0]["solution"]))
    ranked.sort(key=lambda item: item[0])
    return [grid for _, grid in ranked[:2]]


def empty_report(challenges_path, output_dirs, submission, query_map):
    return {
        "challenges": str(challenges_path),
        "outputs": [str(path) for path in output_dirs],
        "task_count": len(submission),
        "query_count": len(query_map),
        "generated_query_count": 0,
        "generated_attempt_count": 0,
        "fallback_count": len(query_map),
        "fallback_attempt_count": 2 * len(query_map),
        "candidate_files": 0,
        "scanned_candidate_files": 0,
        "corrupt_count": 0,
        "valid_candidate_count": 0,
        "invalid_candidate_count": 0,
        "ignored_file_count": 0,
        "temporary_file_count": 0,
        "missing_output_dirs": [],
        "unreadable_output_dirs": [],
        "corrupt_files": [],
        "selection_policy": SELECTION_POLICY,
        "status": "fallback_initialized",
    }


def populate_submission(submission, query_map, output_dirs, report):
    candidates = {}
    seen_dirs = set()
    for output_dir in output_dirs:
        output_dir = Path(output_dir)
        canonical_dir = output_dir.resolve()
        if canonical_dir in seen_dirs:
            continue
        seen_dirs.add(canonical_dir)
        if not output_dir.is_dir():
            report["missing_output_dirs"].append(str(output_dir))
            continue
        try:
            paths = sorted(output_dir.iterdir(), key=lambda path: path.name)
        except OSError as exc:
            report["unreadable_output_dirs"].append({"path": str(output_dir), "error": type(exc).__name__})
            continue
        for path in paths:
            name = path.name
            if name.startswith(".") or name.endswith(".tmp") or ".tmp." in name:
                report["temporary_file_count"] += 1
                continue
            query_key = name.split(".", 1)[0]
            # The launcher forces ARC_RUN_TAG=base. Rescue/legacy outputs cannot
            # enter this frozen baseline even if accidentally left in the folder.
            if query_key not in query_map or not (".base." in name or name.endswith(".base")) or not path.is_file():
                report["ignored_file_count"] += 1
                continue
            report["scanned_candidate_files"] += 1
            try:
                with bz2.BZ2File(path, "rb") as stream:
                    payload = pickle.load(stream)
                if not isinstance(payload, list):
                    raise ValueError("Candidate file must contain a list")
            except Exception as exc:
                report["corrupt_count"] += 1
                report["corrupt_files"].append({"path": str(path), "error": type(exc).__name__})
                continue
            report["candidate_files"] += 1
            for sample in payload:
                try:
                    normalized = normalize_candidate(sample)
                except Exception:
                    report["invalid_candidate_count"] += 1
                    continue
                candidates.setdefault(query_key, []).append(normalized)
                report["valid_candidate_count"] += 1

    rows = []
    for query_key, (task_id, index) in query_map.items():
        values = candidates.get(query_key, [])
        selected = select_kgmon(values)
        for attempt, grid in enumerate(selected, start=1):
            submission[task_id][index][f"attempt_{attempt}"] = grid
        if selected:
            report["generated_query_count"] += 1
            report["fallback_count"] -= 1
            report["generated_attempt_count"] += len(selected)
            report["fallback_attempt_count"] -= len(selected)
        rows.append({
            "key": query_key,
            "candidate_count": len(values),
            "unique_candidate_count": len({tuple(tuple(row) for row in v["solution"]) for v in values}),
            "generated_attempt_count": len(selected),
            "fallback_only": not bool(selected),
        })
    report["rows"] = rows
    report["status"] = "complete"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", default=os.getenv("ARC_CHALLENGES_PATH"), help="Explicit challenges JSON; never inferred from rerun state")
    parser.add_argument("--outputs", action="append", help="Directory of this run's trusted base worker pickles; repeat for multiple directories")
    parser.add_argument("--submission", default=os.getenv("ARC_SUBMISSION_PATH", "/kaggle/working/submission.json"))
    parser.add_argument("--report", default=os.getenv("ARC_REPORT_PATH", "/kaggle/working/submission-report.json"))
    args = parser.parse_args(argv)
    if not args.challenges:
        parser.error("--challenges or ARC_CHALLENGES_PATH is required")
    output_dirs = args.outputs or [os.getenv("ARC_OUTPUT_DIR", "/kaggle/inference_outputs")]
    paths = [Path(args.challenges).resolve(), Path(args.submission).resolve(), Path(args.report).resolve()]
    if len(set(paths)) != 3:
        parser.error("Challenges, submission and report must be different paths")
    with open(args.challenges, encoding="utf-8") as stream:
        challenges = json.load(stream)
    submission, query_map = initial_submission(challenges)
    report = empty_report(args.challenges, output_dirs, submission, query_map)
    # The supervisor rebuilds repeatedly. Never replace a valid earlier result
    # with fallback before scanning: a timeout or interruption must preserve it.
    if not existing_submission_matches(args.submission, submission):
        atomic_json(args.submission, submission)
        atomic_json(args.report, report)
    populate_submission(submission, query_map, output_dirs, report)
    atomic_json(args.submission, submission)
    atomic_json(args.report, report)
    print(json.dumps({
        "submission": args.submission,
        "report": args.report,
        "tasks": report["task_count"],
        "queries": report["query_count"],
        "generated_query_count": report["generated_query_count"],
        "fallback_count": report["fallback_count"],
        "candidate_files": report["candidate_files"],
        "corrupt_count": report["corrupt_count"],
    }))
    return report


if __name__ == "__main__":
    main()
