#!/usr/bin/env python3
"""Manually score a no-gold-frozen030 pair using an explicit official Kaggle snapshot.

Requires --allow-gold-read. Snapshot files must already exist and match their
recorded SHA/size. No download, model invocation, new ranking, or submission
is performed. All frozen artifacts are verified before query gold is opened.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace


HERE = Path(__file__).resolve().parent
SCORE_PATH = HERE.parent / "020/score_frozen_outputs.py"
spec = importlib.util.spec_from_file_location("score020_for030", SCORE_PATH)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
POLICIES = ("base_B4", "challenger_B4")


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", required=True, type=Path)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--stage", required=True, choices=["initial", "extension"])
    parser.add_argument("--mechanism-report", type=Path)
    parser.add_argument("--mechanism-manifest", type=Path,
                        default=HERE / "mechanism-v1-package-manifest.json")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--allow-gold-read", action="store_true")
    args = parser.parse_args()
    if not args.allow_gold_read:
        parser.error("Use --allow-gold-read only after both predictions/candidate pools/runtime evidence are frozen")
    frozen, freeze_id = score.read_json(args.frozen / "prediction-freeze.json")
    if frozen.get("gold_read") is not False or not frozen.get("B4_reproduced_for_both_arms"):
        raise ValueError("Missing completed no-gold B4 reproduction freeze")
    for name, expected in frozen["outputs"].items():
        raw = (args.frozen / name).read_bytes()
        if len(raw) != expected["bytes"] or score.sha(raw) != expected["sha256"]:
            raise ValueError("Frozen artifact changed: " + name)
    config = SimpleNamespace(
        manifest=args.frozen / "input/manifest.json",
        challenges=args.frozen / "input/challenges.json",
        prediction=[name + "=" + str(args.frozen / ("submission-" + name + ".json")) for name in POLICIES],
        candidate_pool=[args.frozen / "candidate-pools-scored.json"])
    manifest, challenges, policies, candidates, identities = score.load_inputs(config)
    if frozen["n_tasks"] != manifest["n_tasks"] or frozen["n_queries"] != manifest["n_queries"]:
        raise ValueError("Frozen denominator differs from original input panel")
    ledger = {(row["arm"] + "_B4", row["query_key"]): row
              for row in read_lines(args.frozen / "candidate-ledger.jsonl")}
    runtime, runtime_id = score.read_json(args.frozen / "runtime-summary.json")
    mechanism, mechanism_id, noise_status = None, None, "unconfirmed"
    mechanism_identity_verified = False
    if args.mechanism_report:
        mechanism, mechanism_id = score.read_json(args.mechanism_report)
        expected_mechanism, mechanism_manifest_id = score.read_json(args.mechanism_manifest)
        mechanism_identity_verified = (
            mechanism.get("python_hash_seed") == expected_mechanism["python_hash_seed"]
            and mechanism.get("probability_cutoffs") == expected_mechanism["probability_cutoffs"]
            and mechanism.get("input_identity", {}).get("challenges_sha256")
                == expected_mechanism["source_challenges_sha256"]
            and all(mechanism.get("runtime_source_sha256", {}).get(name) == digest
                    for name, digest in expected_mechanism["runtime_source_sha256"].items())
            and mechanism.get("completed_tasks") == expected_mechanism["planned_tasks"]
            and mechanism.get("completed_calls") == expected_mechanism["planned_calls"])
        if mechanism.get("status") == "completed" and mechanism_identity_verified:
            noise_status = mechanism.get("aa_control_status", "unconfirmed")
        if noise_status not in {"passed", "failed", "unconfirmed"}:
            raise ValueError("Unknown mechanism A/A control status")
    provenance, provenance_id = score.read_json(args.snapshot_manifest)
    if provenance.get("competition") != "arc-prize-2026-arc-agi-2":
        raise ValueError("Snapshot is not labelled as the target Kaggle competition")
    official_names = ("arc-agi_evaluation_challenges.json", "arc-agi_evaluation_solutions.json")
    # Check availability of both files before reading either snapshot, with no fallback download.
    for name in official_names:
        if not Path(provenance["files"][name]["path"]).is_file():
            raise FileNotFoundError("Explicit official snapshot file is unavailable: " + name)
    data = {}
    for name in official_names:
        record = provenance["files"][name]
        raw = Path(record["path"]).read_bytes()
        if len(raw) != record["bytes"] or score.sha(raw) != record["sha256"]:
            raise ValueError("Official snapshot identity changed: " + name)
        data[name] = json.loads(raw)
    official_inputs, solutions = (data[name] for name in official_names)
    rows, sources = [], []
    for entry in manifest["entries"]:
        task_id = entry["task_id"]
        task = official_inputs[task_id]
        if task != challenges[task_id] or len(solutions[task_id]) != len(task["test"]):
            raise ValueError("Full official challenge or query denominator differs: " + task_id)
        official = dict(train=task["train"], test=[dict(input=q["input"], output=g)
                        for q, g in zip(task["test"], solutions[task_id])])
        sources.append(dict(task_id=task_id, source_kind="Exact Kaggle challenge/solution JSON pair",
                            full_task_content_matches=True))
        for query in entry["queries"]:
            idx, key = query["query_index"], query["query_key"]
            gold, status, matched_indices = score.resolve_gold(challenges[task_id], official, idx)
            if status != "matched":
                raise ValueError("Official gold cannot be uniquely input-matched: " + key + " " + status)
            gold_sha = score.fingerprint(gold)
            row = dict(task_id=task_id, query_index=idx, query_key=key,
                       source_group=entry.get("source_group"), tier=entry.get("tier"),
                       gold_match_status=status, official_matching_indices=matched_indices,
                       input_sha256=query["input_sha256"], gold_sha256=gold_sha, policies={})
            for name in POLICIES:
                result = score.score_prediction(policies[name], task_id, idx, gold)
                candidate_row = ledger[(name, key)]
                raw_oracle = gold_sha in candidate_row["raw_candidate_sha256"]
                scored_oracle = gold_sha in candidate_row["scored_candidate_sha256"]
                result.update(raw_candidate_oracle_exact=raw_oracle,
                              candidate_oracle_exact=scored_oracle,
                              correct_candidate_not_selected=scored_oracle and not result["exact"],
                              second_slot_only_exact=result["attempt_2_exact"] and not result["attempt_1_exact"])
                row["policies"][name] = result
            rows.append(row)
    if len(rows) != manifest["n_queries"]:
        raise ValueError("Not all frozen queries were scored")
    report = score.summarize(rows, policies)
    gains = [row for row in rows if row["policies"]["challenger_B4"]["exact"]
             and not row["policies"]["base_B4"]["exact"]]
    losses = [row for row in rows if row["policies"]["base_B4"]["exact"]
              and not row["policies"]["challenger_B4"]["exact"]]
    gain_tasks = sorted({row["task_id"] for row in gains})
    net = len(gains) - len(losses)
    for name in POLICIES:
        result = report["policies"][name]
        result.update(raw_oracle_queries=sum(row["policies"][name]["raw_candidate_oracle_exact"] for row in rows),
                      scored_oracle_queries=sum(row["policies"][name]["candidate_oracle_exact"] for row in rows),
                      first_slot_exact_queries=sum(row["policies"][name]["attempt_1_exact"] for row in rows),
                      second_slot_only_exact_queries=sum(row["policies"][name]["second_slot_only_exact"] for row in rows))
    initial_gain_condition = args.stage != "initial" or len(gain_tasks) >= 2
    if net <= 0 or not initial_gain_condition:
        decision = "stop_fixed_0p1_recipe" if net <= 0 else "hold_insufficient_independent_gain_tasks"
    elif noise_status != "passed":
        decision = "hold_noise_control_unconfirmed" if noise_status == "unconfirmed" else "hold_noise_control_failed"
    else:
        decision = "eligible_for_extension_pending_budget_review" if args.stage == "initial" else "eligible_for_formal_review_pending_budget_and_initial_result"
    report.update(
        scored_at_utc=datetime.now(timezone.utc).isoformat(), cohort=manifest["cohort"], stage=args.stage,
        planned_tasks=manifest["n_tasks"], frozen_input_identity=identities,
        original_prediction_freeze=freeze_id, runtime_summary_identity=runtime_id,
        official_snapshot_manifest=provenance_id, official_sources=sources,
        paired=dict(gained_queries=[row["query_key"] for row in gains], lost_queries=[row["query_key"] for row in losses],
                    net_queries=net, gain_task_ids=gain_tasks, loss_task_ids=sorted({row["task_id"] for row in losses}),
                    lost_base_second_slot_only_queries=[row["query_key"] for row in losses
                                                       if row["policies"]["base_B4"]["second_slot_only_exact"]]),
        runtime_evidence=runtime, mechanism_report_identity=mechanism_id,
        mechanism_source_and_input_identity_verified=mechanism_identity_verified,
        noise_control_status=noise_status,
        promotion_criteria=dict(all_frozen_queries_input_matched=True, net_queries_strictly_positive=net > 0,
                                initial_gains_from_at_least_two_tasks=initial_gain_condition,
                                aa_noise_control_passed=noise_status == "passed",
                                budget_and_tail_coverage_review="required_manual_review",
                                extension_positive_result="required_before_formal" if args.stage == "initial" else net > 0),
        decision=decision, scoring_helper_sha256=score.sha(SCORE_PATH.read_bytes()),
        interpretation="Exposed public Kaggle regression, not hidden leaderboard score or independent blind confirmation. Same-seed TTT fingerprints and A/A noise controls are separate evidence; no automatic submission.")
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output / "query-ledger.jsonl").open("x") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    score.write_json(args.output / "score-report.json", report)
    print(json.dumps(dict(cohort=manifest["cohort"], stage=args.stage, policies=report["policies"],
                          paired=report["paired"], decision=decision, noise_control_status=noise_status),
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
