#!/usr/bin/env python3
"""Freeze the quota-limited030 pilot from initial40, using inputs only.

Preserves the initial40 length tiers, crosses them with single/multi-query,
then selects20 by proportional largest-remainder allocation and a fixed hash.
No query solutions, model predictions, remote service, or model is read/run.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


HERE = Path(__file__).resolve().parent
INITIAL = HERE / "panels-v1/initial40"
OUTPUT = HERE / "panels-v1/pilot20"
SEED = "030:pilot20:quota20261027"
INITIAL_MANIFEST_SHA = "a1cd3da40aa0cd0fc218ac121c58ef18a75c37e2f5dd71ab822fb5a9572c815f"
INITIAL_CHALLENGES_SHA = "38df4c7126f160f9dd045b4b0c9fb37ccb4f24dfbb51c8d32aa7b50f132ebeb9"
OFFICIAL_CHALLENGES_SHA = "e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value):
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":"), allow_nan=False).encode())


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))


def write(path, value):
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare():
    original, manifest_id = read(INITIAL / "manifest.json")
    inputs, challenges_id = read(INITIAL / "challenges.json")
    if manifest_id["sha256"] != INITIAL_MANIFEST_SHA or challenges_id["sha256"] != INITIAL_CHALLENGES_SHA:
        raise ValueError("Frozen initial40 source identity changed")
    if original["source"]["sha256"] != OFFICIAL_CHALLENGES_SHA:
        raise ValueError("Initial40 does not reference the expected official Kaggle snapshot")
    if original["input_artifact"]["sha256"] != challenges_id["sha256"]:
        raise ValueError("Initial40 input artifact differs from its manifest")
    if len(inputs) != 40 or sum(len(task["test"]) for task in inputs.values()) != 58:
        raise ValueError("Expected the complete initial40/58 source")
    if any(set(query) != {"input"} for task in inputs.values() for query in task["test"]):
        raise ValueError("Query output fields are forbidden")
    entries = {entry["task_id"]: entry for entry in original["entries"]}
    if set(entries) != set(inputs) or len(original["entries"]) != 40:
        raise ValueError("Source manifest task IDs differ from challenges")

    strata = {}
    for task_id in sorted(inputs):
        task, entry = inputs[task_id], entries[task_id]
        tier = entry["tier"]
        group = "single" if len(task["test"]) == 1 else "multi"
        if tier not in {"S", "M", "L"} or entry["query_group"] != group:
            raise ValueError("Unexpected original length tier/query group")
        if entry["demo_pairs_sha256"] != fingerprint(task["train"]):
            raise ValueError("Source demo fingerprint differs")
        if entry["n_test_queries"] != len(task["test"]):
            raise ValueError("Source query count differs")
        for index, query in enumerate(task["test"]):
            q = entry["queries"][index]
            if (q["query_index"] != index or q["query_key"] != task_id + "_" + str(index)
                    or q["input_sha256"] != fingerprint(query["input"])):
                raise ValueError("Source query fingerprint differs")
        strata.setdefault(tier + "_" + group, []).append(task_id)

    allocation = {name: 20 * len(ids) // 40 for name, ids in strata.items()}
    remaining = 20 - sum(allocation.values())
    remainder_order = sorted(strata, key=lambda name: (-(20 * len(strata[name]) % 40), name))
    for name in remainder_order[:remaining]:
        allocation[name] += 1
    selected, records = [], {}
    for name, ids in sorted(strata.items()):
        ordered = sorted(ids, key=lambda task_id: (sha((SEED + ":" + task_id).encode()), task_id))
        chosen = ordered[:allocation[name]]
        selected.extend(chosen)
        records[name] = dict(eligible_tasks=len(ids), selected_tasks=len(chosen),
            selected_queries=sum(len(inputs[key]["test"]) for key in chosen),
            task_ids=sorted(chosen), hash_ranked_eligible_task_ids=ordered)
    selected.sort()
    if len(selected) != 20 or len(set(selected)) != 20:
        raise ValueError("Pilot selection did not produce exactly20 distinct tasks")
    selected_inputs = {key: inputs[key] for key in selected}
    selected_entries = []
    for key in selected:
        entry = copy.deepcopy(entries[key])
        entry["source_group"] = "public_evaluation_NVARC030_pilot20_quota_limited"
        selected_entries.append(entry)

    if OUTPUT.exists():
        raise FileExistsError(OUTPUT)
    OUTPUT.mkdir(parents=True, exist_ok=False)
    write(OUTPUT / "challenges.json", selected_inputs)
    manifest = dict(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort="030_pilot20_quota_limited_fixed_public_regression",
        status="input_panel_frozen_GPU_not_run", panel="pilot20",
        n_tasks=20, n_queries=sum(len(task["test"]) for task in selected_inputs.values()),
        source_task_count=original["source_task_count"], source_query_count=original["source_query_count"],
        eligible_task_count=40, selection_seed=SEED,
        selection_rule=("Use all40 initial40 task IDs as eligible. Preserve each task's original S/M/L tier; "
            "cross tier with its single/multi-query status. Allocate20 proportionally by integer largest "
            "remainders, ties by stratum name. Within each stratum select ascending sha256(seed+':'+task_id), "
            "ties task_id. Keep every demo and every query; runtime queue order is ascending task_id. "
            "Correctness, predictions and historical pressure annotations are not selection inputs."),
        length_measure=original["length_measure"],
        length_tier_definition="Inherited original initial40/025 tier, not recomputed within the pilot",
        task_order=selected, task_order_sha256=fingerprint(selected), strata=records,
        entries=selected_entries, source=copy.deepcopy(original["source"]),
        sources=[dict(manifest_id, role="Frozen initial40 membership, original tiers and exposure metadata"),
                 dict(challenges_id, role="Input-only initial40 challenge content; all selected queries retained")],
        preparation_script=identity(Path(__file__)), input_artifact=identity(OUTPUT / "challenges.json"),
        query_solutions_read=False, model_predictions_read=False, query_output_fields_in_challenges=0,
        independent_confirmation=False, quota_limited_initial_replacement=True,
        replaces="initial40 runtime panel under currently available zhuating quota; original40 remains unchanged",
        frozen_solver_hypothesis=dict(A_cumulative_path_probability_floor=0.2,
            B_cumulative_path_probability_floor=0.1, python_hash_seed=27030,
            selection="Original B4 kgmon, first two distinct grids",
            initial_promotion="Final EM@2 net>0 and gained queries from at least two distinct tasks; A/A noise and budget review required",
            extension_promotion="Separate fixed extension must also have final EM@2 net>0 before formal review",
            note="Executable package and equal deployment-budget identity must be frozen separately before this pilot runs"),
        historical_pressure_annotations_used_for_selection=False,
        limitations=[
            "Subset of already scored025/current initial40 public regression material; not a new blind confirmation.",
            "Initial40 and upstream-model exposure boundaries are inherited without upgrading their independence status.",
            "All tasks retain all original queries, including failures/missing outputs after future budget cutoffs.",
            "Smaller quota-limited pilot increases uncertainty; its percentage is not a hidden leaderboard prediction.",
            "Official full-source SHA is inherited from initial40 metadata; only initial40 inputs/manifest were reopened here.",
            "This script performs CPU input preparation only; no030 pilot output, score, GPU session or competition submission is produced.",
        ],
    )
    write(OUTPUT / "manifest.json", manifest)
    return dict(panel="pilot20", n_tasks=manifest["n_tasks"], n_queries=manifest["n_queries"],
        manifest=identity(OUTPUT / "manifest.json"), challenges=identity(OUTPUT / "challenges.json"),
        task_order_sha256=manifest["task_order_sha256"], strata=records,
        query_solutions_read=False, model_predictions_read=False, GPU_run=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    print(json.dumps(prepare(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
