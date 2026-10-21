#!/usr/bin/env python3
"""Ordinary Save & Run artifact only: fixed CPU rule, no model inference.

The aligned-line module, when enabled, must be packaged beside this file.
Only known demo outputs select applicability. Query answers are not accepted.
This run does not measure the hidden model policy or predict its score.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import submit_lib as SL


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_challenges(payload):
    if not isinstance(payload, dict) or not payload:
        raise ValueError("Challenges must contain every task in a nonempty object")
    clean = {}
    for task_id, task in payload.items():
        if not isinstance(task_id, str) or not isinstance(task, dict):
            raise ValueError("Invalid challenge task")
        demos, queries = task.get("train"), task.get("test")
        if not isinstance(demos, list) or not isinstance(queries, list) or not queries:
            raise ValueError("Invalid train/test lists: " + task_id)
        if any(not isinstance(query, dict) or "output" in query for query in queries):
            raise ValueError("Query rows must contain inputs without answers: " + task_id)
        clean[task_id] = {
            "train": [{"input": SL.valid_grid(demo["input"]),
                       "output": SL.valid_grid(demo["output"])} for demo in demos],
            "test": [{"input": SL.valid_grid(query["input"])} for query in queries],
        }
    return clean


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--submission", default="/kaggle/working/submission.json")
    parser.add_argument("--report", default="/kaggle/working/ordinary-run.json")
    parser.add_argument("--line-complement", action="store_true")
    parser.add_argument("--session-seconds", type=float, default=300)
    parser.add_argument("--session-start-monotonic", type=float)
    args = parser.parse_args()
    if not 0 < args.session_seconds <= 300:
        raise ValueError("Ordinary CPU session must be at most 300 seconds")
    started = time.monotonic()
    session_start = args.session_start_monotonic or started
    # Finish before the notebook parent's 45-second termination margin.
    solve_deadline = session_start + args.session_seconds - 60
    source = Path(args.challenges)
    raw = source.read_bytes()
    challenges = clean_challenges(json.loads(raw))
    submission, query_map = SL.initial_submission(challenges)
    SL.atomic_json(args.submission, submission)
    run = {
        "status": "running", "phase": "ordinary_cpu_save",
        "model_inference_performed": False, "model_validation_performed": False,
        "GPU_compute_not_invoked": True, "gpu_quota_consumption": "not_measured",
        "query_gold_used": False, "query_score_status": "not_scored",
        "input_path": str(source), "input_sha256": hashlib.sha256(raw).hexdigest(),
        "source_sha256": {path.name: sha_file(path)
                          for path in sorted(Path(__file__).parent.glob("*.py"))},
        "n_tasks": len(challenges), "n_queries": len(query_map),
        "line_complement_enabled": args.line_complement,
        "cpu_applicable_tasks": 0, "cpu_supported_queries": 0,
        "actual_predicted_queries": 0, "placeholder_queries": len(query_map),
        "tasks": {}, "errors": [], "session_seconds": args.session_seconds,
        "scope_note": "CPU save-stage artifact from full public inputs; this is not "
                      "a full-model evaluation. Hidden scoring runs the separate model driver.",
    }
    updated_keys = set()

    def persist():
        run["actual_predicted_queries"] = len(updated_keys)
        run["placeholder_queries"] = len(query_map) - len(updated_keys)
        run["elapsed_s"] = round(time.monotonic() - started, 6)
        run["session_elapsed_s"] = round(time.monotonic() - session_start, 6)
        SL.atomic_json(args.report, run)

    persist()
    line_solver = None
    if args.line_complement:
        import aligned_line_solver as line_solver
        run["source_sha256"]["aligned_line_solver.py"] = sha_file(line_solver.__file__)
    for task_id, task in challenges.items():
        detail = {"n_queries": len(task["test"]), "cpu_admitted": False,
                  "predicted_query_indices": [], "status": "disabled"}
        run["tasks"][task_id] = detail
        if line_solver is None:
            continue
        if time.monotonic() >= solve_deadline:
            detail["status"] = "skipped_deadline"
            continue
        try:
            solved = line_solver.solve_task(task)
            admitted = bool(solved["admitted"] and solved["all_demos_exact"]
                            and solved["changed_demo_count"] > 0)
            detail.update(status="evaluated", cpu_admitted=admitted,
                          demo_total=solved["demo_total"], demo_exact=solved["demo_exact"],
                          changed_demo_count=solved["changed_demo_count"])
            if not admitted:
                continue
            run["cpu_applicable_tasks"] += 1
            for prediction in solved["query_predictions"]:
                if not prediction["supported"]:
                    continue
                index = prediction["query_index"]
                if type(index) is not int or not 0 <= index < len(task["test"]):
                    raise ValueError("Invalid solver query index")
                grid = SL.valid_grid(prediction["grid"])
                key = "%s_%d" % (task_id, index)
                updated, errors = SL.update_slots(submission, query_map, {key: [grid]})
                if updated != 1 or errors:
                    raise ValueError("Could not fill supported CPU prediction: " + str(errors))
                updated_keys.add(key)
                detail["predicted_query_indices"].append(index)
                run["cpu_supported_queries"] += 1
            SL.atomic_json(args.submission, submission)
        except Exception as exc:
            detail["status"] = "solver_error"
            run["errors"].append({"task_id": task_id,
                                  "error": "%s: %s" % (type(exc).__name__, str(exc)[:300])})
        persist()
    SL.atomic_json(args.submission, submission)
    valid, validation = SL.validate_submission(args.submission, submission)
    run["format_validation"] = {"valid": valid, **validation}
    run["submission_sha256"] = sha_file(args.submission)
    run["status"] = ("invalid_submission" if not valid else
                     "complete_with_errors" if run["errors"] else
                     "complete_with_deadline_skips"
                     if any(row["status"] == "skipped_deadline" for row in run["tasks"].values())
                     else "complete")
    persist()
    print(json.dumps({key: run[key] for key in (
        "status", "phase", "n_tasks", "n_queries", "cpu_applicable_tasks",
        "actual_predicted_queries", "placeholder_queries", "elapsed_s",
        "model_inference_performed")}), flush=True)
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
