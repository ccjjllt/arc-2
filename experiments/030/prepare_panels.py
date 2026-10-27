#!/usr/bin/env python3
"""Freeze input-only development/regression panels for one manual 030 comparison.

The first panel retains all 025 forty tasks and every query. The extension uses
only challenge inputs and fixed historical membership metadata. No query gold,
model predictions, model invocation, or remote execution is read or started.
Existing output directories are never replaced.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SOURCE_SHA = "e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8"
DEFAULT_SOURCE = ROOT / "experiments/008/raw-eval/arc-agi_evaluation_challenges.json"
DEFAULT_OUTPUT = ROOT / "experiments/030/panels-v1"
DEFAULT_SEED = "030:extension40:input-only:20261027"
PRESSURE_NOTES = {
    "2ba387bc_0": "025 known second-slot-only correct; historical pressure annotation, not a blind example",
    "d35bdbdc_1": "025 correct raw candidate first observed in query batch 3; historical pressure annotation",
    "7ed72f31_1": "025 correct raw candidate first observed in query batch 3; historical pressure annotation",
    "7b0280bc_0": "025 correct raw candidate first observed in query batch 3; historical pressure annotation",
}


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def load(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw),
    }, raw


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def identity(path: Path):
    raw = path.read_bytes()
    return {"path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw)}


def input_cells(task) -> int:
    return sum(len(row) for pair in task["train"] + task["test"]
               for row in pair["input"])


def members(manifest):
    return {entry["task_id"] for entry in manifest["entries"]}


def stratified_selection(source, eligible, wanted, seed):
    ranked_inputs = sorted(eligible, key=lambda task_id: (input_cells(source[task_id]), task_id))
    if not 0 < wanted <= len(ranked_inputs):
        raise ValueError("Requested extension count is outside the eligible input pool")
    strata = {}
    tier_for = {}
    for i, tier in enumerate(("S", "M", "L")):
        tier_ids = ranked_inputs[i * len(ranked_inputs) // 3:(i + 1) * len(ranked_inputs) // 3]
        tier_for.update({task_id: tier for task_id in tier_ids})
        for query_group in ("single", "multi"):
            ids = [task_id for task_id in tier_ids
                   if (len(source[task_id]["test"]) == 1) == (query_group == "single")]
            if ids:
                strata[tier + "_" + query_group] = ids
    allocations = {name: wanted * len(ids) // len(ranked_inputs) for name, ids in strata.items()}
    rest = wanted - sum(allocations.values())
    order = sorted(strata, key=lambda name: (
        -(wanted * len(strata[name]) % len(ranked_inputs)), name))
    for name in order[:rest]:
        allocations[name] += 1
    selected = []
    records = {}
    for name, ids in sorted(strata.items()):
        seeded = sorted(ids, key=lambda task_id: (sha((seed + ":" + task_id).encode()), task_id))
        chosen = seeded[:allocations[name]]
        selected.extend(chosen)
        records[name] = {
            "eligible_tasks": len(ids), "selected_tasks": len(chosen),
            "selected_queries": sum(len(source[task_id]["test"]) for task_id in chosen),
            "eligible_input_cells_min": min(input_cells(source[task_id]) for task_id in ids),
            "eligible_input_cells_max": max(input_cells(source[task_id]) for task_id in ids),
            "task_ids": sorted(chosen),
        }
    return sorted(selected), tier_for, records


def prepare(args):
    source, source_identity, _ = load(args.source)
    if source_identity["sha256"] != SOURCE_SHA:
        raise ValueError("Source is not the exact Kaggle challenge snapshot frozen by 027")
    if len(source) != 120 or sum(len(t["test"]) for t in source.values()) != 172:
        raise ValueError("Unexpected official public challenge task/query count")
    if any(set(query) != {"input"} for task in source.values() for query in task["test"]):
        raise ValueError("Query fields must be input-only")

    old_manifest, old_identity, _ = load(ROOT / "experiments/025/cohort-v1/manifest.json")
    old_inputs, old_input_identity, old_input_raw = load(ROOT / "experiments/025/cohort-v1/challenges.json")
    c23, c23_identity, _ = load(ROOT / "experiments/027/confirmation-v1/manifest.json")
    p16, p16_identity, _ = load(ROOT / "experiments/026/panel-v1/manifest.json")
    initial_ids = sorted(members(old_manifest))
    c23_ids, p16_ids = members(c23), members(p16)
    historical_excluded_ids = {entry["task_id"] for entry in old_manifest["exclusions"]
                               if entry.get("in_source")}
    if old_inputs != {task_id: source[task_id] for task_id in initial_ids}:
        raise ValueError("025 initial challenges differ from the exact official snapshot")
    if len(initial_ids) != 40 or sum(len(source[k]["test"]) for k in initial_ids) != 58:
        raise ValueError("025 initial panel must retain 40 tasks and 58 queries")
    if set(initial_ids) & c23_ids or not p16_ids <= set(initial_ids):
        raise ValueError("Unexpected historical panel membership")
    if set(initial_ids) | c23_ids | historical_excluded_ids != set(source):
        raise ValueError("Historical public-source exposure membership no longer matches")

    remaining = set(source) - set(initial_ids)
    extension_ids, extension_tiers, extension_strata = stratified_selection(
        source, remaining, args.extension_tasks, args.seed)
    old_entries = {entry["task_id"]: entry for entry in old_manifest["entries"]}
    initial_tiers = {task_id: old_entries[task_id]["tier"] for task_id in initial_ids}
    now = datetime.now(timezone.utc).isoformat()
    evidence = [dict(old_identity, role="025 initial membership and historical exclusion metadata"),
                dict(old_input_identity, role="025 exact input-only panel retained byte-for-byte"),
                dict(c23_identity, role="027 already consumed confirmation membership only"),
                dict(p16_identity, role="026 development membership only; outcomes are not used")]
    limitations = [
        "Both panels are exposed public development/regression data, not an independent blind confirmation set.",
        "All public tasks were inferred in 019; historical per-query gold exposure is not exhaustively audited.",
        "C23 membership denotes a consumed prior confirmation set; its inclusion here does not restore holdout status.",
        "Upstream NVARC public-eval validation/checkpoint selection and rule-derived data preclude model-unseen claims.",
        "Historical exclusion records include metadata-only selections and reservations; they do not imply all120 query golds were read.",
        "Input-cell strata approximate input size, not token length, semantic difficulty or measured runtime.",
        "Full query denominators remain fixed even when a future budget cutoff leaves outputs missing.",
        "This freeze records inputs/membership only; solver time budgets and executable source identity must be frozen separately before GPU execution.",
    ]
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    outputs = {}
    for name, selected, tiers in [
        ("initial40", initial_ids, initial_tiers),
        ("extension40", extension_ids, extension_tiers),
    ]:
        directory = args.output_dir / name
        directory.mkdir(exist_ok=False)
        challenges = {task_id: source[task_id] for task_id in selected}
        challenge_path = directory / "challenges.json"
        if name == "initial40":
            with challenge_path.open("xb") as stream:
                stream.write(old_input_raw)
        else:
            write_json(challenge_path, challenges)
        entries = []
        for task_id in selected:
            task = source[task_id]
            entries.append({
                "task_id": task_id, "source_group": "public_evaluation_NVARC030_" + name,
                "tier": tiers[task_id],
                "query_group": "single" if len(task["test"]) == 1 else "multi",
                "n_demo": len(task["train"]), "total_input_cells": input_cells(task),
                "demo_pairs_sha256": fingerprint(task["train"]),
                "n_test_queries": len(task["test"]),
                "queries": [{"query_index": i, "query_key": task_id + "_" + str(i),
                             "input_sha256": fingerprint(query["input"])}
                            for i, query in enumerate(task["test"])],
                "exposure": {
                    "in_025_scored_panel": task_id in set(initial_ids),
                    "in_026_training_recipe_development_panel": task_id in p16_ids,
                    "in_027_consumed_C23_confirmation": task_id in c23_ids,
                    "in_025_historical_exclusion_metadata": task_id in historical_excluded_ids,
                    "prior_019_full_public_inference": True,
                    "other_historical_query_gold_exposure": "Not exhaustively audited",
                    "upstream_model_unseen": "Not established",
                },
            })
        if name == "initial40":
            rule = ("Retain all025 forty task IDs, all original demos and all queries; do not select by030 outputs. "
                    "Challenge bytes are copied unchanged from025. Task queue order is ascending task_id, as in original V1.")
            strata = old_manifest["strata"]
        else:
            rule = ("Exclude initial40 only. Sort remaining80 tasks by(input cells,task_id), split equal-rank S/M/L thirds, "
                    "cross with single/multi-query. Allocate40 proportionally by largest integer remainders(ties stratum name). "
                    "Within each stratum select ascending sha256(seed+':'+task_id), ties task_id. Preserve every query. "
                    "Task queue order is ascending task_id. No query answers or model predictions are selection inputs.")
            strata = extension_strata
        manifest = {
            "created_at_utc": now, "cohort": "030_" + name + "_fixed_public_regression",
            "status": "input_panel_frozen_GPU_not_run", "panel": name,
            "n_tasks": len(selected), "n_queries": sum(len(source[k]["test"]) for k in selected),
            "source_task_count": len(source), "source_query_count": 172,
            "eligible_task_count": 40 if name == "initial40" else len(remaining),
            "selection_seed": None if name == "initial40" else args.seed,
            "selection_rule": rule,
            "length_measure": "Sum of demo-input and query-input cells only; no demo-output or query-output cells",
            "task_order": list(selected), "task_order_sha256": fingerprint(selected),
            "strata": strata, "entries": entries,
            "source": dict(source_identity, competition="arc-prize-2026-arc-agi-2",
                           provenance="Official Kaggle challenge snapshot archived by008 and exact-SHA verified by027",
                           matches_027_official_challenge_sha256=True),
            "sources": evidence, "input_artifact": identity(challenge_path),
            "query_solutions_read": False, "model_predictions_read": False,
            "query_output_fields_in_challenges": 0, "independent_confirmation": False,
            "consumed_C23_tasks": sorted(set(selected) & c23_ids),
            "historical_pressure_annotations": {key: note for key, note in PRESSURE_NOTES.items()
                                                if key.split("_")[0] in selected},
            "pressure_annotation_usage": "Historical known behaviors only; never a deployment whitelist, selection weight or independent confirmation claim",
            "frozen_solver_hypothesis": {
                "A_cumulative_path_probability_floor": 0.2,
                "B_cumulative_path_probability_floor": 0.1,
                "selection": "Original B4 kgmon, first two distinct grids",
                "note": "Input freeze does not replace the separate executable package/budget manifest",
            },
            "limitations": limitations,
        }
        manifest_path = directory / "manifest.json"
        write_json(manifest_path, manifest)
        outputs[name] = {"n_tasks": manifest["n_tasks"], "n_queries": manifest["n_queries"],
                         "manifest": identity(manifest_path), "challenges": identity(challenge_path),
                         "task_order_sha256": manifest["task_order_sha256"]}
    root_manifest = {
        "created_at_utc": now, "status": "input_panels_frozen_GPU_not_run",
        "source": source_identity, "selection_seed": args.seed,
        "preparation_script": identity(Path(__file__)), "outputs": outputs,
        "initial_extension_task_overlap": sorted(set(initial_ids) & set(extension_ids)),
        "initial40_exactly_retains_025_challenge_bytes": outputs["initial40"]["challenges"]["sha256"] == old_input_identity["sha256"],
        "query_solutions_read": False, "model_predictions_read": False,
        "notes": "Both are exposed public regression panels. No remote execution, GPU run or score was produced by this preparation.",
    }
    write_json(args.output_dir / "manifest.json", root_manifest)
    return root_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    parser.add_argument("--extension-tasks", type=int, default=40, choices=[40])
    args = parser.parse_args()
    result = prepare(args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
