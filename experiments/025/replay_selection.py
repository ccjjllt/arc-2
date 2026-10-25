#!/usr/bin/env python3
"""Freeze a paired B4/R2 replay from one NVARC candidate pool without gold.

Use --candidate-pool for an already deduplicated pool in original kgmon order,
or --ordered-candidates for JSONL records exported in ArcDecoder load order.
Never infer that order by sorting downloaded pickle filenames. The reproduced
B4 must exactly match --baseline before this command writes any outputs.

Example:
  python3 experiments/025/replay_selection.py --manifest MANIFEST.json \
    --challenges CHALLENGES.json --baseline submission-B4.json \
    --candidate-pool candidate-pools.json --raw-candidates raw-candidates.jsonl \
    --output-dir NEW_DIRECTORY

After this command, use experiments/020/score_frozen_outputs.py separately.
This command does not load answers, score predictions, or call any model.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import platform
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def read_json(path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "bytes": len(raw),
                             "sha256": sha(raw)}


def read_jsonl(path):
    raw = path.read_bytes()
    rows = [json.loads(line) for line in raw.decode().splitlines() if line.strip()]
    return rows, {"path": str(path.resolve()), "bytes": len(raw),
                  "sha256": sha(raw), "rows": len(rows)}


def write_json(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def valid_grid(grid):
    return (isinstance(grid, list) and 1 <= len(grid) <= 30
            and all(isinstance(row, list) for row in grid)
            and 1 <= len(grid[0]) <= 30
            and all(len(row) == len(grid[0]) for row in grid)
            and all(type(cell) is int and 0 <= cell <= 9
                    for row in grid for cell in row))


def finite_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) \
        and math.isfinite(value)


def query_rows(challenges):
    for task_id, task in challenges.items():
        if re.fullmatch(r"[0-9a-f]{8}", task_id) is None:
            raise ValueError("Invalid task ID: " + task_id)
        if not isinstance(task.get("train"), list) or not isinstance(task.get("test"), list):
            raise ValueError("Invalid challenge structure: " + task_id)
        for idx, query in enumerate(task["test"]):
            if set(query) != {"input"} or not valid_grid(query["input"]):
                raise ValueError("Challenge test queries must contain input only")
            yield task_id, idx, "%s_%d" % (task_id, idx), query["input"]


def validate_manifest(manifest, challenges, challenge_identity):
    entries = manifest["entries"]
    ids = [entry["task_id"] for entry in entries]
    if len(ids) != len(set(ids)) or set(ids) != set(challenges):
        raise ValueError("Manifest and challenges have different task IDs")
    if manifest["n_tasks"] != len(ids):
        raise ValueError("Manifest task denominator differs")
    if manifest["input_artifact"]["sha256"] != challenge_identity["sha256"]:
        raise ValueError("Frozen challenge SHA differs from manifest")
    count = 0
    for entry in entries:
        task = challenges[entry["task_id"]]
        if entry["demo_pairs_sha256"] != fingerprint(task["train"]):
            raise ValueError("Manifest demonstration fingerprint differs")
        if entry["n_test_queries"] != len(task["test"]):
            raise ValueError("Manifest query denominator differs")
        queries = {query["query_index"]: query for query in entry["queries"]}
        if set(queries) != set(range(len(task["test"]))):
            raise ValueError("Manifest query indices differ")
        for idx, query in enumerate(task["test"]):
            expected = "%s_%d" % (entry["task_id"], idx)
            if queries[idx]["query_key"] != expected \
                    or queries[idx]["input_sha256"] != fingerprint(query["input"]):
                raise ValueError("Manifest query identity differs: " + expected)
            count += 1
    if count != manifest["n_queries"]:
        raise ValueError("Manifest total query denominator differs")


def aggregate_ordered_records(records, allowed):
    """Keep first grid occurrence for equal kgmon, as the original decoder does."""
    groups = {}
    for index, record in enumerate(records):
        key = record.get("query_key")
        if key not in allowed:
            raise ValueError("Ordered candidate query is outside cohort: " + str(key))
        grid = record.get("solution", record.get("grid"))
        scores = record.get("score_aug")
        if not valid_grid(grid) or not isinstance(scores, list) or not scores \
                or not all(finite_number(value) for value in scores):
            raise ValueError("Invalid scored candidate record at index %d" % index)
        grid_sha = fingerprint(grid)
        group = groups.setdefault(key, {}).setdefault(grid_sha, {
            "grid": grid, "record_means": [], "source_record_indices": [],
            "beam_scores": [],
        })
        group["record_means"].append(sum(scores) / len(scores))
        group["source_record_indices"].append(index)
        if finite_number(record.get("beam_score")):
            group["beam_scores"].append(record["beam_score"])
    pools = {}
    for key, values in groups.items():
        rows = []
        for group in values.values():
            means = group.pop("record_means")
            group["support"] = len(means)
            group["mean_aug_nll"] = sum(means) / len(means)
            group["kgmon"] = len(means) - group["mean_aug_nll"]
            rows.append(group)
        pools[key] = sorted(rows, key=lambda row: row["kgmon"], reverse=True)
    return pools


def validate_pools(pools, allowed):
    if not isinstance(pools, dict) or set(pools) - allowed:
        raise ValueError("Candidate pool contains unknown query IDs")
    for key, rows in pools.items():
        if not isinstance(rows, list):
            raise ValueError("Query pool must be a list: " + key)
        seen, previous = set(), math.inf
        for row in rows:
            grid = row.get("grid")
            nll, support = row.get("mean_aug_nll"), row.get("support")
            if not valid_grid(grid) or not finite_number(nll) \
                    or type(support) is not int or support < 1:
                raise ValueError("Invalid candidate pool row: " + key)
            score = support - nll
            if "kgmon" in row and (not finite_number(row["kgmon"]) or not math.isclose(
                    score, row["kgmon"], rel_tol=1e-12, abs_tol=1e-12)):
                raise ValueError("kgmon does not equal support - mean_aug_nll")
            if score > previous:
                raise ValueError("Pool is not in original descending kgmon order: " + key)
            previous = score
            grid_sha = fingerprint(grid)
            if grid_sha in seen:
                raise ValueError("Pool has duplicate grids: " + key)
            seen.add(grid_sha)
            row["grid_sha256"] = grid_sha
            row["kgmon"] = score


def validate_baseline(baseline, challenges, pools):
    if not isinstance(baseline, dict) or set(baseline) != set(challenges):
        raise ValueError("Baseline must contain exactly the frozen cohort task IDs")
    for task_id, task in challenges.items():
        if not isinstance(baseline[task_id], list) \
                or len(baseline[task_id]) != len(task["test"]):
            raise ValueError("Baseline query count differs: " + task_id)
        for idx, entry in enumerate(baseline[task_id]):
            if not isinstance(entry, dict) or set(entry) != {"attempt_1", "attempt_2"} \
                    or not all(valid_grid(entry[name]) for name in ("attempt_1", "attempt_2")):
                raise ValueError("Baseline query has invalid two-slot structure")
            key = "%s_%d" % (task_id, idx)
            expected = {"attempt_1": [[0]], "attempt_2": [[0]]}
            for slot, candidate in enumerate(pools.get(key, [])[:2], 1):
                expected["attempt_%d" % slot] = candidate["grid"]
            if entry != expected:
                raise ValueError("Replayed kgmon B4 differs from frozen remote B4: " + key)


def raw_pools(paths, allowed):
    groups, identities, invalid = {}, [], 0
    for path in paths:
        records, identity = read_jsonl(path)
        identities.append(identity)
        for record in records:
            key = record.get("query_key")
            if key not in allowed:
                raise ValueError("Raw candidate query is outside cohort: " + str(key))
            grid = record.get("solution", record.get("grid"))
            if not valid_grid(grid):
                invalid += 1
                continue
            groups.setdefault(key, {}).setdefault(fingerprint(grid), {
                "grid": grid, "grid_sha256": fingerprint(grid),
            })
    return {key: list(rows.values()) for key, rows in groups.items()}, identities, invalid


def main():
    started = time.monotonic()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--candidate-pool", type=Path)
    source.add_argument("--ordered-candidates", type=Path)
    parser.add_argument("--raw-candidates", type=Path, action="append", default=[])
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    manifest, m_identity = read_json(args.manifest)
    challenges, c_identity = read_json(args.challenges)
    validate_manifest(manifest, challenges, c_identity)
    queries = list(query_rows(challenges))
    allowed = {row[2] for row in queries}
    baseline, b_identity = read_json(args.baseline)
    if args.candidate_pool:
        pools, p_identity = read_json(args.candidate_pool)
        source_note = "Pre-deduplicated pool preserves original kgmon order, including ties."
    else:
        records, p_identity = read_jsonl(args.ordered_candidates)
        pools = aggregate_ordered_records(records, allowed)
        source_note = ("Ordered records use supplied original decoder load order; "
                       "Python double means; complete B4 is checked against remote output.")
    validate_pools(pools, allowed)
    validate_baseline(baseline, challenges, pools)
    raw, raw_identities, invalid_raw = raw_pools(args.raw_candidates, allowed)
    challenger, ledger = copy.deepcopy(baseline), []
    for task_id, idx, key, query_input in queries:
        rows = pools.get(key, [])
        if len(rows) >= 2:
            # Stable sorting means equal NLL retains the original kgmon order.
            selected = min(rows[1:], key=lambda row: row["mean_aug_nll"])
            challenger[task_id][idx]["attempt_2"] = selected["grid"]
        b4_grids = baseline[task_id][idx]
        r2_grids = challenger[task_id][idx]
        ledger.append({
            "task_id": task_id, "query_index": idx, "query_key": key,
            "input_sha256": fingerprint(query_input),
            "scored_unique_candidates": len(rows),
            "raw_unique_candidates": len(raw.get(key, [])) if args.raw_candidates else None,
            "scored_candidate_sha256": [row["grid_sha256"] for row in rows],
            "raw_candidate_sha256": [row["grid_sha256"] for row in raw.get(key, [])]
            if args.raw_candidates else None,
            "B4_attempt_sha256": [fingerprint(b4_grids[name]) for name in ("attempt_1", "attempt_2")],
            "R2_attempt_sha256": [fingerprint(r2_grids[name]) for name in ("attempt_1", "attempt_2")],
            "second_slot_changed": b4_grids["attempt_2"] != r2_grids["attempt_2"],
            "coverage": "no_scored_candidate" if not rows else "scored_candidates_available",
        })
    outputs = {
        "submission-B4.json": baseline,
        "submission-R2.json": challenger,
        "candidate-pools-scored.json": {name: pools for name in ("B4", "R2")},
    }
    if args.raw_candidates:
        outputs["candidate-pools-raw.json"] = {name: raw for name in ("B4", "R2")}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    output_identities = {}
    for filename, value in outputs.items():
        path = args.output_dir / filename
        write_json(path, value)
        output_identities[filename] = {"sha256": sha(path.read_bytes()),
                                       "bytes": path.stat().st_size}
    ledger_path = args.output_dir / "query-candidate-ledger.jsonl"
    with ledger_path.open("x", encoding="utf-8") as stream:
        for row in ledger:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    output_identities[ledger_path.name] = {"sha256": sha(ledger_path.read_bytes()),
                                         "bytes": ledger_path.stat().st_size}
    report = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": manifest["cohort"], "n_tasks": len(challenges), "n_queries": len(queries),
        "status": "paired_predictions_frozen_without_gold",
        "source_order": source_note,
        "baseline_exactly_reproduced": True,
        "policies": {"B4": "Original kgmon top2 distinct grids",
                     "R2": "Keep B4 first slot; lowest mean_aug_nll other grid; stable kgmon ties; preserve unfilled slots"},
        "inputs": {"manifest": m_identity, "challenges": c_identity,
                   "remote_B4": b_identity, "scored_candidates": p_identity,
                   "raw_candidates": raw_identities},
        "coverage": {"queries_with_scored_candidates": sum(bool(pools.get(key)) for key in allowed),
                     "queries_without_scored_candidates": sum(not pools.get(key) for key in allowed),
                     "scored_unique_candidates": sum(len(rows) for rows in pools.values()),
                     "raw_unique_candidates": sum(len(rows) for rows in raw.values()) if args.raw_candidates else None,
                     "invalid_raw_records": invalid_raw,
                     "raw_pool_available": bool(args.raw_candidates),
                     "second_slot_changed_queries": [row["query_key"] for row in ledger if row["second_slot_changed"]]},
        "outputs": output_identities,
        "gold_access": False, "new_model_calls": 0,
        "hardware": platform.platform(), "python": platform.python_version(),
        "wall_seconds": time.monotonic() - started,
        "interpretation": "Selection-only paired replay. Correctness and oracle are unscored until separate input-matched gold access.",
    }
    write_json(args.output_dir / "prediction-freeze.json", report)
    print(json.dumps({key: report[key] for key in ("status", "n_tasks", "n_queries", "coverage", "wall_seconds")}, ensure_ascii=False))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        print("Selection replay stopped: %s: %s" % (type(error).__name__, error), file=sys.stderr)
        raise SystemExit(2)
