#!/usr/bin/env python3
"""Manually score frozen policy outputs against input-matched official ARC2 tasks.

No gold is read or downloaded without --allow-gold-read. The full original
cohort manifest is mandatory even when predictions arrive as task shards.
Repeated --prediction M0=PATH arguments merge disjoint task outputs for M0.
Missing predictions remain failures in the matched, predetermined denominator.

Example (after all D24 predictions have been frozen):
  python3 experiments/020/score_frozen_outputs.py \
    --manifest experiments/020/cohorts/development-d24-manifest.json \
    --challenges experiments/020/cohorts/development-d24-challenges.json \
    --prediction M0=PATH_A/submission-M0.json \
    --prediction M0=PATH_B/submission-M0.json \
    --prediction M1=PATH_A/submission-M1.json \
    --prediction M1=PATH_B/submission-M1.json \
    --official-split evaluation --official-revision FULL_40_CHAR_COMMIT \
    --output-dir NEW_SCORING_DIRECTORY --allow-gold-read

Only the manifest's task IDs are requested. For an existing official checkout,
--official-root REPO reads each task with git show REV:data/SPLIT/ID.json;
otherwise the immutable raw.githubusercontent.com URL is requested. Local
checkout files are not trusted in place of the selected Git revision.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fingerprint(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def read_json(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), {"path": str(path.resolve()), "sha256": sha(raw),
                            "bytes": len(raw)}


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def grid_is_valid(grid) -> bool:
    return (
        isinstance(grid, list) and 1 <= len(grid) <= 30
        and all(isinstance(row, list) for row in grid)
        and 1 <= len(grid[0]) <= 30
        and all(len(row) == len(grid[0]) for row in grid)
        and all(type(cell) is int and 0 <= cell <= 9
                for row in grid for cell in row)
    )


def load_inputs(args):
    manifest, m_identity = read_json(args.manifest)
    challenges, c_identity = read_json(args.challenges)
    entries = manifest["entries"]
    ids = [entry["task_id"] for entry in entries]
    if len(set(ids)) != len(ids) or set(ids) != set(challenges):
        raise ValueError("Original manifest and challenges must have identical unique task IDs")
    if manifest["input_artifact"]["sha256"] != c_identity["sha256"]:
        raise ValueError("Challenge SHA differs from frozen original manifest")
    if len(entries) != manifest["n_tasks"]:
        raise ValueError("Original task denominator mismatch")
    query_count = 0
    for entry in entries:
        task_id = entry["task_id"]
        if re.fullmatch(r"[0-9a-f]{8}", task_id) is None:
            raise ValueError("Unexpected task ID syntax")
        task = challenges[task_id]
        if fingerprint(task["train"]) != entry["demo_pairs_sha256"]:
            raise ValueError("Demo fingerprint mismatch: " + task_id)
        if len(task["test"]) != entry["n_test_queries"]:
            raise ValueError("Query count mismatch: " + task_id)
        queries = entry["queries"]
        if {q["query_index"] for q in queries} != set(range(len(task["test"]))):
            raise ValueError("Manifest query indices do not cover original task")
        for query in queries:
            idx = query["query_index"]
            if set(task["test"][idx]) != {"input"}:
                raise ValueError("Solver challenge query contains fields other than input")
            if fingerprint(task["test"][idx]["input"]) != query["input_sha256"]:
                raise ValueError("Query fingerprint mismatch: " + query["query_key"])
        query_count += len(task["test"])
    if query_count != manifest["n_queries"]:
        raise ValueError("Original query denominator mismatch")

    policies, identities = {}, []
    for specification in args.prediction:
        name, separator, filename = specification.partition("=")
        if not separator or not name or not filename:
            raise ValueError("--prediction must be POLICY=PATH")
        data, identity = read_json(Path(filename))
        if not isinstance(data, dict) or set(data) - set(ids):
            raise ValueError("Prediction file contains invalid/unknown task IDs: " + filename)
        pool = policies.setdefault(name, {})
        overlap = set(pool) & set(data)
        if overlap:
            raise ValueError("Overlapping shards for policy " + name + ": " + str(sorted(overlap)))
        pool.update(data)
        identities.append(dict(identity, policy=name, task_ids=sorted(data)))
    if not policies:
        raise ValueError("At least one policy prediction file is required")

    # Candidate pools are optional diagnostics, never a source for final slots.
    candidates, candidate_identities = {}, []
    for filename in args.candidate_pool:
        data, identity = read_json(Path(filename))
        if not isinstance(data, dict):
            raise ValueError("Candidate pool must map policy to query-key candidates")
        for policy, query_pools in data.items():
            if policy not in policies or not isinstance(query_pools, dict):
                raise ValueError("Candidate pool has an unknown policy or invalid structure")
            dest = candidates.setdefault(policy, {})
            for key, rows in query_pools.items():
                if key not in {q["query_key"] for e in entries for q in e["queries"]}:
                    raise ValueError("Candidate pool contains a query outside the original cohort")
                if key in dest or not isinstance(rows, list):
                    raise ValueError("Candidate pool shards overlap or contain a non-list pool")
                dest[key] = rows
        candidate_identities.append(identity)
    return manifest, challenges, policies, candidates, {
        "manifest": m_identity, "challenges": c_identity,
        "predictions": identities, "candidate_pools": candidate_identities,
    }


def official_task(args, task_id):
    relative = "data/%s/%s.json" % (args.official_split, task_id)
    record = {"revision": args.official_revision, "relative_path": relative}
    if args.official_root:
        command = ["git", "-C", str(args.official_root), "show",
                   args.official_revision + ":" + relative]
        process = subprocess.run(command, capture_output=True, timeout=30)
        if process.returncode:
            return None, dict(record, status="official_task_unavailable",
                              error="git show returned " + str(process.returncode))
        raw = process.stdout
        record["source"] = "git show from " + str(args.official_root.resolve())
    else:
        url = "https://raw.githubusercontent.com/arcprize/ARC-AGI-2/%s/%s" % (
            args.official_revision, relative)
        record["source"] = url
        try:
            with urllib.request.urlopen(url, timeout=30) as response:
                raw = response.read(2_000_001)
        except urllib.error.HTTPError as exc:
            return None, dict(record, status="official_task_unavailable",
                              error="HTTP " + str(exc.code))
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            return None, dict(record, status="official_task_unavailable",
                              error=type(exc).__name__)
    if len(raw) > 2_000_000:
        return None, dict(record, status="invalid_official_task", error="file too large")
    record["sha256"] = sha(raw)
    try:
        task = json.loads(raw)
        if not isinstance(task, dict) or not isinstance(task.get("train"), list) \
                or not isinstance(task.get("test"), list):
            raise ValueError("invalid official task structure")
    except (ValueError, TypeError):
        return None, dict(record, status="invalid_official_task")
    return task, dict(record, status="loaded")


def resolve_gold(challenge, official, query_index):
    # The complete ordered demo pairs must agree. Query indices are not used
    # to align different dataset versions.
    if challenge["train"] != official["train"]:
        return None, "demo_mismatch", []
    query_input = challenge["test"][query_index]["input"]
    matching = [(i, query.get("output")) for i, query in enumerate(official["test"])
                if isinstance(query, dict) and query.get("input") == query_input]
    if not matching:
        return None, "query_input_unmatched", []
    if any(not grid_is_valid(output) for _, output in matching):
        return None, "official_gold_missing_or_invalid", [i for i, _ in matching]
    unique_outputs = {fingerprint(output) for _, output in matching}
    if len(unique_outputs) != 1:
        return None, "ambiguous_query_input", [i for i, _ in matching]
    return matching[0][1], "matched", [i for i, _ in matching]


def score_prediction(policy, task_id, idx, gold):
    queries = policy.get(task_id)
    entry = queries[idx] if isinstance(queries, list) and idx < len(queries) else None
    result = {"prediction_present": isinstance(entry, dict),
              "attempt_1_valid": False, "attempt_2_valid": False,
              "both_attempt_keys_exact": False, "distinct_valid_attempts": False}
    if not isinstance(entry, dict):
        result["prediction_status"] = "missing_or_invalid_entry"
        if gold is not None:
            result.update(attempt_1_exact=False, attempt_2_exact=False, exact=False)
        return result
    result["both_attempt_keys_exact"] = set(entry) == {"attempt_1", "attempt_2"}
    grids = [entry.get("attempt_1"), entry.get("attempt_2")]
    valid = [grid_is_valid(grid) for grid in grids]
    result["attempt_1_valid"], result["attempt_2_valid"] = valid
    result["prediction_status"] = "valid" if all(valid) and result["both_attempt_keys_exact"] \
        else "invalid_attempt_structure"
    result["distinct_valid_attempts"] = all(valid) and grids[0] != grids[1]
    result["both_placeholder"] = grids == [[[0]], [[0]]]
    result["attempt_sha256"] = [fingerprint(grid) if ok else None
                               for grid, ok in zip(grids, valid)]
    if gold is not None:
        # Count a malformed two-attempt object as an invalid submission query,
        # even if one accidentally well-formed grid happens to equal the gold.
        accepted = result["both_attempt_keys_exact"] and all(valid)
        exact = [accepted and grid == gold for grid in grids]
        result.update(attempt_1_exact=exact[0], attempt_2_exact=exact[1], exact=any(exact))
    return result


def summarize(rows, policies):
    matched = [row for row in rows if row["gold_match_status"] == "matched"]
    result = {"planned_queries": len(rows),
              "input_matched_queries": len(matched),
              "unresolved_or_mismatched_queries": len(rows) - len(matched),
              "policies": {}}
    for name in policies:
        exact = sum(row["policies"][name]["exact"] for row in matched)
        result["policies"][name] = {
            "exact_queries": exact, "matched_denominator": len(matched),
            "em2": exact / len(matched) if matched else None,
            "prediction_present_queries": sum(row["policies"][name]["prediction_present"]
                                              for row in rows),
            "valid_queries": sum(row["policies"][name]["prediction_status"] == "valid"
                                for row in rows),
            "different_valid_attempts_queries": sum(
                row["policies"][name]["distinct_valid_attempts"] for row in rows),
        }
        task_rows = {}
        for row in matched:
            task_rows.setdefault(row["task_id"], []).append(row)
        planned_by_task = {}
        for row in rows:
            planned_by_task.setdefault(row["task_id"], []).append(row)
        fully_matched = {k: v for k, v in task_rows.items()
                         if len(v) == len(planned_by_task[k])}
        result["policies"][name]["fully_matched_tasks"] = len(fully_matched)
        result["policies"][name]["all_queries_exact_tasks"] = sum(
            all(row["policies"][name]["exact"] for row in values)
            for values in fully_matched.values())
    if "M0" in policies and "M1" in policies:
        gains = [row for row in matched
                 if row["policies"]["M1"]["exact"] and not row["policies"]["M0"]["exact"]]
        losses = [row for row in matched
                  if row["policies"]["M0"]["exact"] and not row["policies"]["M1"]["exact"]]
        result["M1_vs_M0"] = {
            "gained_queries": [row["query_key"] for row in gains],
            "lost_queries": [row["query_key"] for row in losses],
            "net_queries": len(gains) - len(losses),
            "gain_task_ids": sorted({row["task_id"] for row in gains}),
            "loss_task_ids": sorted({row["task_id"] for row in losses}),
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--challenges", required=True, type=Path)
    parser.add_argument("--prediction", action="append", required=True, metavar="POLICY=PATH")
    parser.add_argument("--candidate-pool", action="append", default=[], metavar="PATH")
    parser.add_argument("--official-revision", required=True)
    parser.add_argument("--official-split", choices=["training", "evaluation"], required=True)
    parser.add_argument("--official-root", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--allow-gold-read", action="store_true")
    args = parser.parse_args()
    if not args.allow_gold_read:
        parser.error("Gold access is disabled: explicitly pass --allow-gold-read after freezing outputs")
    if re.fullmatch(r"[0-9a-f]{40}", args.official_revision) is None:
        parser.error("--official-revision must be a full immutable 40-character commit SHA")
    manifest, challenges, policies, candidates, identities = load_inputs(args)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    freeze = {
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": manifest["cohort"], "planned_tasks": manifest["n_tasks"],
        "planned_queries": manifest["n_queries"], "artifacts": identities,
        "official_revision": args.official_revision, "official_split": args.official_split,
        "note": "All supplied policy predictions were loaded and hashed before any gold access.",
    }
    write_json(args.output_dir / "prediction-freeze.json", freeze)
    rows, sources = [], []
    for entry in manifest["entries"]:
        task_id = entry["task_id"]
        official, source = official_task(args, task_id)
        sources.append(source)
        for query in entry["queries"]:
            idx, key = query["query_index"], query["query_key"]
            if official is None:
                gold, status, indices = None, source["status"], []
            else:
                gold, status, indices = resolve_gold(challenges[task_id], official, idx)
            row = {"task_id": task_id, "query_index": idx, "query_key": key,
                   "source_group": entry.get("source_group"), "tier": entry.get("tier"),
                   "input_sha256": query["input_sha256"], "gold_match_status": status,
                   "official_matching_indices": indices,
                   "gold_sha256": fingerprint(gold) if gold is not None else None,
                   "policies": {}}
            for name, prediction in policies.items():
                scored = score_prediction(prediction, task_id, idx, gold)
                if gold is not None and name in candidates:
                    pool = candidates[name].get(key, [])
                    oracle = any(isinstance(candidate, dict)
                                 and grid_is_valid(candidate.get("grid"))
                                 and candidate["grid"] == gold for candidate in pool)
                    scored["candidate_oracle_exact"] = oracle
                    scored["correct_candidate_not_selected"] = oracle and not scored["exact"]
                row["policies"][name] = scored
            rows.append(row)
        # Persist an audit trail during the deliberately serial, small read.
        with (args.output_dir / "query-ledger.jsonl").open("a", encoding="utf-8") as stream:
            for row in rows[-len(entry["queries"]):]:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    report = summarize(rows, policies)
    report.update(
        cohort=manifest["cohort"], planned_tasks=manifest["n_tasks"],
        official_sources=sources,
        by_source_group={str(group): summarize(
            [row for row in rows if row["source_group"] == group], policies)
            for group in sorted({row["source_group"] for row in rows}, key=str)},
        interpretation="Local fixed-two-slot exact-match only; not a Kaggle score or evidence of exceeding 30.14.",
        denominator_note="Every original manifest query is in the ledger. Missing/invalid predictions count as failures when inputs match. Official input/version mismatches and unavailable gold are separate unresolved rows, never silently counted wrong or removed from the planned cohort.",
    )
    write_json(args.output_dir / "score-report.json", report)
    print(json.dumps({k: v for k, v in report.items()
                      if k not in {"official_sources", "by_source_group"}}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired) as exc:
        print("Scoring stopped: %s: %s" % (type(exc).__name__, exc), file=sys.stderr)
        raise SystemExit(2)
