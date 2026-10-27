#!/usr/bin/env python3
"""Manually freeze030 B4 outputs, candidate pools and runtime evidence without gold.

This reads completed local run artifacts only. It verifies the original B4 from
ordered scored records, retains every planned query, and never calls a model,
loads query solutions, changes a ranking rule, or submits anything remotely.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


HERE = Path(__file__).resolve().parent
REPLAY_PATH = HERE.parent / "025/replay_selection.py"
spec = importlib.util.spec_from_file_location("replay025_for030", REPLAY_PATH)
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
ARMS = ("base", "challenger")


def event_summary(rows, query_keys):
    counts = defaultdict(int)
    completed_batches = {key: 0 for key in query_keys}
    started_batches = {key: 0 for key in query_keys}
    worker_costs = {}
    fingerprints = defaultdict(list)
    pending = {}
    pairs = {"ttt_start": ("ttt_end", "TTT"),
             "search_start": ("search_end", "DFS"),
             "score_start": ("score_end", "rescoring")}
    ending = {end: (start, stage) for start, (end, stage) in pairs.items()}
    for row in rows:
        kind, rank, task = row.get("event"), row.get("rank"), row.get("task_id")
        counts[str(kind)] += 1
        if kind in {"search_start", "search_end"}:
            keys = row.get("query_keys") or list(dict.fromkeys(
                subkey.split(".")[0] for subkey in row.get("subkeys", [])))
            for key in keys:
                if key not in query_keys:
                    raise ValueError("Event query outside frozen cohort: " + str(key))
                (started_batches if kind == "search_start" else completed_batches)[key] += 1
        if "adapter" in str(kind) and "fingerprint" in str(kind):
            value = row.get("adapter_fingerprint", row.get("fingerprint", row.get("sha256")))
            if task is not None and value is not None:
                fingerprints[task].append(value)
        key = (rank, task)
        worker = worker_costs.setdefault(str(rank), {
            "TTT_seconds": 0.0, "DFS_seconds": 0.0, "rescoring_seconds": 0.0,
            "closed_intervals": 0, "unmatched_end_events": 0})
        if kind in pairs:
            token = (key, kind)
            if token in pending:
                raise ValueError("Overlapping same-stage events in one worker")
            pending[token] = row["timestamp"]
        elif kind in ending:
            start, stage = ending[kind]
            began = pending.pop((key, start), None)
            if began is None:
                worker["unmatched_end_events"] += 1
            else:
                elapsed = row["timestamp"] - began
                if elapsed < 0:
                    raise ValueError("Event timestamps decrease within a stage")
                worker[stage + "_seconds"] += elapsed
                worker["closed_intervals"] += 1
    return {
        "event_counts": dict(counts), "started_batches_by_query": started_batches,
        "completed_batches_by_query": completed_batches,
        "queries_never_started": sorted(key for key, count in started_batches.items() if not count),
        "queries_with_four_completed_batches": sum(count == 4 for count in completed_batches.values()),
        "per_worker_stage_costs": worker_costs, "unclosed_stage_intervals": len(pending),
        "adapter_fingerprints_by_task": dict(fingerprints),
        "cost_interpretation": "Host stage interval sums per worker; not synchronized CUDA profiles or wall-clock critical path. Unclosed cutoff intervals are not silently estimated.",
    }


def arm_record(state, arm):
    record = state.get(arm, state.get("arms", {}).get(arm))
    if not isinstance(record, dict):
        raise ValueError("Missing arm terminal record: " + arm)
    if record.get("timed_out") is True and record.get("status") == "budget_cutoff":
        if state["status"] != "completed_budget_cutoffs":
            raise ValueError("Budget cutoff not declared by top-level run status")
        if record.get("return_code") not in {0, -9}:
            raise ValueError("Unexpected return code in declared budget cutoff: " + arm)
    elif record.get("return_code") != 0 or record.get("timed_out") or record.get("status") != "completed":
        raise ValueError("Unexpected arm error; cannot freeze as a successful comparison: " + arm)
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    manifest_path = args.manifest or args.run_root / "input/manifest.json"
    manifest, manifest_id = replay.read_json(manifest_path)
    runtime_manifest, runtime_manifest_id = replay.read_json(args.run_root / "input/manifest.json")
    if manifest_id["sha256"] != runtime_manifest_id["sha256"]:
        raise ValueError("Runtime panel manifest differs from requested frozen manifest")
    challenges, challenge_id = replay.read_json(args.run_root / "input/challenges.json")
    replay.validate_manifest(manifest, challenges, challenge_id)
    query_keys = {key for _, _, key, _ in replay.query_rows(challenges)}
    state, state_id = replay.read_json(args.run_root / "run-state.json")
    if state.get("status") not in {"completed", "completed_budget_cutoffs"} or state.get("return_code") != 0:
        raise ValueError("Run did not reach a declared normal/budget-cutoff terminal state")
    expected_identity = state["input_identity"]
    if (expected_identity["challenges_sha256"] != challenge_id["sha256"]
            or expected_identity["n_tasks"] != manifest["n_tasks"]
            or expected_identity["n_queries"] != len(query_keys)):
        raise ValueError("Runtime input identity/denominator differs")

    copy_files = {"input/manifest.json": manifest_path,
                  "input/challenges.json": args.run_root / "input/challenges.json",
                  "run-state.json": args.run_root / "run-state.json"}
    inputs = [manifest_id, runtime_manifest_id, challenge_id, state_id]
    scored_pools, raw_pools, ledger, all_events, arm_summaries = {}, {}, [], [], {}
    for arm in ARMS:
        terminal = arm_record(state, arm)
        root, policy = args.run_root / arm, arm + "_B4"
        prediction_path = root / "submission-B4.json"
        prediction, pred_id = replay.read_json(prediction_path)
        if pred_id["sha256"] != state["arms"][arm]["prediction_sha256"]:
            raise ValueError("Runtime final prediction SHA differs: " + arm)
        scored, scored_id = replay.read_jsonl(root / "scored-candidates.jsonl")
        pools = replay.aggregate_ordered_records(scored, query_keys)
        replay.validate_pools(pools, query_keys)
        replay.validate_baseline(prediction, challenges, pools)
        raw_paths = [root / ("raw-candidates-rank%d.jsonl" % rank) for rank in range(4)]
        raw, raw_ids, invalid = replay.raw_pools(raw_paths, query_keys)
        if invalid:
            raise ValueError("Invalid raw exported grids: " + arm)
        events = []
        for rank in range(4):
            rows, event_id = replay.read_jsonl(root / ("events-rank%d.jsonl" % rank))
            inputs.append(event_id)
            for row in rows:
                if row.get("rank", rank) != rank:
                    raise ValueError("Event rank differs from filename")
                row = dict(row, rank=rank)
                if row.get("task_id") is not None and row["task_id"] not in challenges:
                    raise ValueError("Event task outside frozen panel")
                events.append(row)
                all_events.append(dict(row, arm=arm))
        arm_summaries[arm] = dict(terminal=terminal,
                                  B4_exactly_reproduced=True,
                                  scored_record_count=len(scored),
                                  runtime_events=event_summary(events, query_keys))
        inputs.extend([pred_id, scored_id, *raw_ids])
        copy_files["submission-" + policy + ".json"] = prediction_path
        scored_pools[policy] = {key: pools.get(key, []) for key in sorted(query_keys)}
        raw_pools[policy] = {key: raw.get(key, []) for key in sorted(query_keys)}
        for key in sorted(query_keys):
            ledger.append(dict(arm=arm, query_key=key,
                               raw_candidate_sha256=sorted(row["grid_sha256"] for row in raw.get(key, [])),
                               scored_candidate_sha256=sorted(row["grid_sha256"] for row in pools.get(key, [])),
                               raw_unique_candidates=len(raw.get(key, [])),
                               scored_unique_candidates=len(pools.get(key, []))))

    base_fp = arm_summaries["base"]["runtime_events"]["adapter_fingerprints_by_task"]
    other_fp = arm_summaries["challenger"]["runtime_events"]["adapter_fingerprints_by_task"]
    fp_comparison = {}
    for task in challenges:
        a, b = base_fp.get(task), other_fp.get(task)
        fp_comparison[task] = "unconfirmed" if not a or not b else ("identical" if a == b else "paired_TTT_not_identical")
    summary = dict(run_status=state["status"], arms=arm_summaries,
                   adapter_comparison_by_task=fp_comparison,
                   noise_control_status="unknown_separate_mechanism_report_required",
                   interpretation="Same seed/full-run environment does not assert identical TTT; no accuracy measured here.")
    args.output.mkdir(parents=True, exist_ok=False)
    for name, source in copy_files.items():
        dest = args.output / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(source.read_bytes())
    replay.write_json(args.output / "candidate-pools-scored.json", scored_pools)
    replay.write_json(args.output / "candidate-pools-raw.json", raw_pools)
    replay.write_json(args.output / "runtime-summary.json", summary)
    for name, rows in [("candidate-ledger.jsonl", ledger), ("worker-events.jsonl", all_events)]:
        with (args.output / name).open("x") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    outputs = {}
    for path in sorted(args.output.rglob("*")):
        if path.is_file():
            raw = path.read_bytes()
            outputs[str(path.relative_to(args.output))] = dict(bytes=len(raw), sha256=replay.sha(raw))
    freeze = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
                  cohort=manifest["cohort"], n_tasks=manifest["n_tasks"], n_queries=len(query_keys),
                  gold_read=False, B4_reproduced_for_both_arms=True, outputs=outputs,
                  inputs=inputs, helper_source=str(REPLAY_PATH.resolve()),
                  helper_source_sha256=replay.sha(REPLAY_PATH.read_bytes()),
                  note="Both original runtime B4 outputs and all diagnostic evidence frozen before gold; no alternate ranking generated.")
    replay.write_json(args.output / "prediction-freeze.json", freeze)
    print(json.dumps(dict(cohort=freeze["cohort"], n_tasks=freeze["n_tasks"],
                          n_queries=freeze["n_queries"], run_status=state["status"],
                          B4_reproduced_for_both_arms=True, output=str(args.output.resolve())), indent=2))


if __name__ == "__main__":
    main()
