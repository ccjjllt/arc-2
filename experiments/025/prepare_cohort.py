#!/usr/bin/env python3
"""Manually freeze an input-only cohort for the NVARC B4/R2 comparison.

This reads local challenge and historical selection metadata only. It neither
reads query solutions nor launches a model. The destination must be new.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SOURCE = Path("/private/tmp/arc2-023-public-input/arc-agi_evaluation_challenges.json")
ARCHIVED_SOURCE = ROOT / "experiments/008/raw-eval/arc-agi_evaluation_challenges.json"
DEFAULT_SEED = "025:B4R2:20261025"


def sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def fingerprint(value) -> str:
    return sha(json.dumps(value, ensure_ascii=False, sort_keys=True,
                          separators=(",", ":")).encode())


def load(path: Path):
    raw = path.read_bytes()
    return json.loads(raw), {
        "path": str(path.resolve()), "sha256": sha(raw), "bytes": len(raw),
    }


def input_cells(task) -> int:
    return sum(len(row) for pair in task["train"] + task["test"]
               for row in pair["input"])


def write_json(path: Path, value) -> None:
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")


def prepare(args):
    source, source_identity = load(args.source)
    archive, archive_identity = load(ARCHIVED_SOURCE)
    if not isinstance(source, dict) or not source:
        raise ValueError("Challenge source is not a nonempty task mapping")
    for task_id, task in source.items():
        if not isinstance(task_id, str) or len(task_id) != 8:
            raise ValueError("Unexpected task ID")
        if any(set(query) != {"input"} for query in task["test"]):
            raise ValueError("Source contains query fields other than input")

    exclusions = defaultdict(set)
    evidence = []

    def add(ids, reason):
        for task_id in ids:
            exclusions[task_id].add(reason)

    e20, ident = load(ROOT / "experiments/020/evaluation-e20/manifest.json")
    evidence.append(dict(ident, role="E20 and inherited explicit development exclusions"))
    add(e20["excluded_task_ids"], "020_recorded_development_confirmation_or_scored_exclusion")
    add((entry["task_id"] for entry in e20["entries"]), "020_022_023_E20_development_and_R2_replay")

    ordinary, ident = load(ROOT / "experiments/023/ordinary-v1-input/manifest.json")
    evidence.append(dict(ident, role="Original V1 ordinary four tasks"))
    add((entry["task_id"] for entry in ordinary["entries"]), "023_original_V1_ordinary_four")

    old, ident = load(ROOT / "experiments/010/evaluation-manifest.json")
    evidence.append(dict(ident, role="010 probes and historical confirmation reservations"))
    for group in ("probe_P12", "confirmation_C12", "TTT_P6"):
        for name, description in old[group].items():
            if isinstance(description, dict) and "task_ids" in description:
                add(description["task_ids"], "010_" + group + "_" + name)

    old_e24, ident = load(ROOT / "experiments/008/e0_e24_24.json")
    evidence.append(dict(ident, role="Prior metadata-only E24 selection exposure"))
    previously_selected = {entry["task_id"] for entry in old_e24["selected24"]}

    eligible = sorted(set(source) - set(exclusions),
                      key=lambda task_id: (input_cells(source[task_id]), task_id))
    wanted = min(args.tasks, len(eligible))
    if wanted < 1:
        raise ValueError("No eligible tasks")

    # Cross three input-length rank buckets with one versus multiple queries.
    # Proportional largest-remainder allocation avoids selecting by model result.
    strata = {}
    tier_for = {}
    for i, tier in enumerate(("S", "M", "L")):
        tier_ids = eligible[i * len(eligible) // 3:(i + 1) * len(eligible) // 3]
        for task_id in tier_ids:
            tier_for[task_id] = tier
        for query_group in ("single", "multi"):
            ids = [task_id for task_id in tier_ids
                   if (len(source[task_id]["test"]) == 1) == (query_group == "single")]
            if ids:
                strata[tier + "_" + query_group] = ids
    counts = {name: wanted * len(ids) // len(eligible) for name, ids in strata.items()}
    remaining = wanted - sum(counts.values())
    remainders = sorted(strata,
                        key=lambda name: (-(wanted * len(strata[name]) % len(eligible)), name))
    for name in remainders[:remaining]:
        counts[name] += 1

    chosen = []
    stratum_records = {}
    for name, ids in sorted(strata.items()):
        ranked = sorted(ids, key=lambda task_id: (sha((args.seed + ":" + task_id).encode()), task_id))
        selected = ranked[:counts[name]]
        chosen.extend(selected)
        stratum_records[name] = {
            "eligible_tasks": len(ids), "selected_tasks": len(selected),
            "selected_queries": sum(len(source[task_id]["test"]) for task_id in selected),
            "input_cells_min": min(input_cells(source[task_id]) for task_id in ids),
            "input_cells_max": max(input_cells(source[task_id]) for task_id in ids),
            "task_ids": sorted(selected),
        }
    chosen.sort()
    challenges = {task_id: source[task_id] for task_id in chosen}
    entries = []
    for task_id in chosen:
        task = source[task_id]
        entries.append({
            "task_id": task_id,
            "source_group": "public_evaluation_NVARC025_confirmation",
            "tier": tier_for[task_id],
            "query_group": "single" if len(task["test"]) == 1 else "multi",
            "n_demo": len(task["train"]),
            "total_input_cells": input_cells(task),
            "demo_pairs_sha256": fingerprint(task["train"]),
            "n_test_queries": len(task["test"]),
            "queries": [{"query_index": i, "query_key": task_id + "_" + str(i),
                         "input_sha256": fingerprint(query["input"])}
                        for i, query in enumerate(task["test"])],
            "exposure": {
                "explicit_recorded_tuning_or_probe_exclusion_match": False,
                "prior_008_metadata_selection": task_id in previously_selected,
                "prior_019_full_public_inference": True,
                "prior_local_query_scoring": "Not audited per query; source has historical aggregate scoring exposure",
                "upstream_unseen_status": "Not established; NVARC used public-eval validation and derived rule data",
            },
        })

    args.output_dir.mkdir(parents=True, exist_ok=False)
    challenge_path = args.output_dir / "challenges.json"
    write_json(challenge_path, challenges)
    challenge_identity = load(challenge_path)[1]
    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "cohort": "025_NVARC_V1_B4_R2_confirmation40",
        "status": "input_selection_frozen_not_run",
        "selection_seed": args.seed,
        "selection_rule": (
            "Exclude every recorded exclusion before selection. Sort eligible tasks by "
            "(sum of demo-input and query-input cells, task_id); split by rank into equal S/M/L thirds. "
            "Cross each tier with single/multi-query. Allocate requested tasks proportionally using "
            "integer largest remainders, ties by stratum name. Within each stratum rank "
            "sha256(UTF-8 seed + ':' + task_id), ties by task_id. Preserve every original query. "
            "Write selected tasks in task_id order. No model predictions or query answers enter selection."
        ),
        "length_measure": "Demo and query input grid cell count; no demo-output cells or query outputs",
        "requested_tasks": args.tasks,
        "n_tasks": len(entries),
        "n_queries": sum(entry["n_test_queries"] for entry in entries),
        "source_task_count": len(source),
        "source_query_count": sum(len(task["test"]) for task in source.values()),
        "eligible_task_count": len(eligible),
        "excluded_in_source_count": len(set(source) & set(exclusions)),
        "source": dict(source_identity,
                       provenance="Local official Kaggle evaluation-challenges download retained by 023",
                       archived_008_identity=archive_identity,
                       matches_archived_008_bytes=source_identity["sha256"] == archive_identity["sha256"],
                       matches_archived_008_contents=source == archive),
        "exclusion_sources": evidence,
        "exclusions": [{"task_id": task_id, "in_source": task_id in source,
                        "reasons": sorted(reasons)}
                       for task_id, reasons in sorted(exclusions.items())],
        "strata": stratum_records,
        "input_artifact": challenge_identity,
        "entries": entries,
        "scoring_reference_revision": "f3283f727488ad98fe575ea6a5ac981e4a188e49",
        "scoring_reference_note": (
            "Reference used by earlier project scoring, not a claim every selected input matches it. "
            "After freezing both outputs, match full ordered demos and each query input. "
            "Report mismatches separately; do not choose gold versions by correctness."
        ),
        "frozen_policies": {
            "B4": "Original V1 kgmon first two distinct grids",
            "R2": "Retain original kgmon first grid; choose remaining distinct grid with minimum existing mean_aug_nll, ties in original kgmon order",
        },
        "query_solutions_read": False,
        "query_output_fields_in_challenges": 0,
        "limitations": [
            "Confirmation means unused for this round's R2 tuning, not model-unseen or unbiased hidden-score validation.",
            "019 inferred the entire public source and historical aggregate scores exist; selected-query scoring exposure is not fully audited.",
            "Exclusion metadata reflects recorded development/probe/reservation IDs and is not proof of an exhaustive historical task inspection audit.",
            "Upstream public-eval validation/checkpoint selection and rule-derived training preclude an independent-blind-test claim.",
            "Cell count stratification approximates input size, not tokenizer length, semantic difficulty or measured runtime.",
            "Use a single generation pool and all frozen queries including missing predictions for B4/R2 comparison.",
        ],
    }
    write_json(args.output_dir / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path,
                        default=DEFAULT_SOURCE if DEFAULT_SOURCE.exists() else ARCHIVED_SOURCE)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "experiments/025/cohort-v1")
    parser.add_argument("--tasks", type=int, default=40)
    parser.add_argument("--seed", default=DEFAULT_SEED)
    args = parser.parse_args()
    if args.tasks < 1:
        parser.error("--tasks must be positive")
    manifest = prepare(args)
    print(json.dumps({key: manifest[key] for key in
                      ("cohort", "n_tasks", "n_queries", "eligible_task_count",
                       "excluded_in_source_count", "input_artifact", "strata")}, indent=2))


if __name__ == "__main__":
    main()
