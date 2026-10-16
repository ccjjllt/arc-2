"""Manual offline summary of a downloaded 012 session. Stdlib only, no pytest/CI.

Usage:
  python3 experiments/012/summarize_results.py SESSION_DIR [--scoring p2-scoring.json] [--out summary.json]
  python3 experiments/012/summarize_results.py --help

Inputs: a directory of downloaded K and/or P artifacts (JSON/JSONL/CSV).
Outputs: JSON file (with --out) or stdout, plus a short terminal digest.
K verdicts follow plan 012 sec 6.3; P counts follow sec 7.4 on the dev P2 only.
Missing files are reported as missing, never as zero scores.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path

GEN_DONE = {"complete", "invalid_output"}
GEN_RESOURCE = {"budget_interrupted", "infrastructure_error", "executor_unavailable",
                "worker_protocol_error", "admission_deadline"}
A_RMS_TOL = 1e-4
B_REL_FLOOR = 0.01
GAIN = 0.15


def read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return None


def read_lines(path: Path):
    rows = []
    try:
        with path.open(encoding="utf-8") as h:
            for line in h:
                try:
                    v = json.loads(line)
                    if isinstance(v, dict):
                        rows.append(v)
                except ValueError:
                    continue
    except OSError:
        pass
    return rows


def find_files(root: Path, *patterns):
    hits = []
    for pat in patterns:
        hits.extend(sorted(root.rglob(pat)))
    return hits


def summarize_kernel(root: Path):
    files = find_files(root, "probe-result.json")
    if len(files) > 1:
        return {"status": "ambiguous", "files_seen": [str(p) for p in files],
                "note": "select one K session directory"}
    diagnostics = [{"file": str(p), "record": read_json(p)} for p in
                   find_files(root, "probe-error.json", "probe-supervisor.json")]
    data = read_json(files[0]) if files else None
    if not isinstance(data, dict):
        partial = {}
        for tag in ("A1", "B1", "B2", "A2"):
            saved = find_files(root, tag + ".json")
            if len(saved) == 1:
                partial[tag] = read_json(saved[0])
        return {"status": "incomplete" if partial or diagnostics else "missing",
                "trials": partial, "diagnostics": diagnostics,
                "note": "no final probe-result.json; partial trials are not a completed comparison"}
    out = {"status": data.get("status", "incomplete"), "files_used": [str(files[0])],
           "trials": data.get("timing", {}), "comparison": data.get("verdict", {}),
           "reference": data.get("reference"), "diagnostics": diagnostics,
           "note": "logit errors are recorded GPU comparisons; timing gains are recomputed here"}
    if data.get("status") != "complete":
        return out
    timing, comp = out["trials"], out["comparison"]
    def finite(v):
        return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
    try:
        times = [timing[k]["total_s"] for k in ("A1", "A2", "B1", "B2")]
        lengths = [timing[k]["n"] for k in ("A1", "A2", "B1", "B2")]
        if not all(finite(t) and t > 0 for t in times):
            raise ValueError("invalid trial times")
        if len(set(lengths)) != 1 or not all(isinstance(n, int) and n >= 16 for n in lengths):
            raise ValueError("unequal or insufficient continuation lengths")
        ar, at = comp["a_repeat_rel_rms"], comp["a_top1_mismatch"]
        if not all(finite(v) and v >= 0 for v in (ar, at)):
            raise ValueError("invalid reference comparison")
        if ar > A_RMS_TOL or at > 0:
            verdict = "unstable_reference"
        else:
            drift = False
            for pair in ("b1_a1", "b2_a2"):
                br, bt = comp[pair + "_rel_rms"], comp[pair + "_top1_mismatch"]
                if not all(finite(v) and v >= 0 for v in (br, bt)):
                    raise ValueError("invalid candidate comparison")
                drift |= br > max(B_REL_FLOOR, 5 * ar) or bt > 0
            if drift:
                verdict = "numerical_drift"
            else:
                gains = [1 - timing[b]["total_s"] / timing[a]["total_s"]
                         for a, b in (("A1", "B1"), ("A2", "B2"))]
                out["recomputed_gains"] = gains
                verdict = "worth_full_output_bridge" if all(g >= GAIN for g in gains) else "insufficient_gain"
        out["status"] = verdict
        if comp.get("verdict") != verdict:
            out.update(status="inconsistent_record", derived_verdict=verdict,
                       reason="stored verdict disagrees with saved comparisons and times")
    except (KeyError, TypeError, ValueError) as exc:
        out.update(status="incomplete", reason=str(exc))
    return out


def arm_of(item: dict) -> str:
    for key in ("mode", "thinking_mode", "arm"):
        value = str(item.get(key, "")).lower()
        if value in ("d", "direct", "nonthinking"):
            return "D"
        if value in ("l", "thinking"):
            return "L"
    if isinstance(item.get("enable_thinking"), bool):
        return "L" if item["enable_thinking"] else "D"
    stage = item.get("stage", item.get("name", ""))
    if isinstance(stage, dict):
        return arm_of(stage)
    for mode in ("D", "L"):
        if str(stage).upper().endswith(("-" + mode, "_" + mode)):
            return mode
    return "?"


PLAN = (("db118e2a", "D"), ("db118e2a", "L"), ("291dc1e1", "L"), ("291dc1e1", "D"))
EXECUTED = {"ok", "execution_error", "execution_timeout", "output_limit"}


def canonical_program_tasks(root):
    # SUMMARY owns completed stages; per-stage files only fill absent stages.
    records, sources, conflicts = {}, {}, []
    summaries = find_files(root, "SUMMARY.json")
    if len(summaries) > 1:
        return records, sources, ["multiple SUMMARY.json files; select one session directory"]
    paths = summaries + [p for p in find_files(root, "*.json") if p not in summaries]
    for path in paths:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        items = data.get("stages", []) if path.name == "SUMMARY.json" else [data]
        for item in items:
            if not isinstance(item, dict) or not isinstance(item.get("tasks"), list):
                continue
            stage = item.get("stage")
            meta = {**(stage if isinstance(stage, dict) else {}), **item}
            if meta.get("policy") != "p0_only" and not meta.get("p0_only"):
                continue
            for task in item["tasks"]:
                if not isinstance(task, dict):
                    continue
                key = (task.get("task_id"), arm_of({**meta, **task}))
                if key not in PLAN:
                    conflicts.append("unexpected task/mode %r in %s" % (key, path))
                    continue
                if key in records:
                    if task != records[key] and Path(sources[key]).name != "SUMMARY.json":
                        conflicts.append("conflicting fallback records for %r" % (key,))
                    continue
                records[key], sources[key] = task, str(path)
    return records, sources, conflicts


def program_row(tid, mode, task, scoring, source):
    row = {"task_id": tid, "arm": mode, "query_index": 0, "source_file": source,
           "outcome": "missing_result", "complete_code": False, "ast_pass": False,
           "thinking_closed": None if mode == "D" else False,
           "executed_demos": 0, "demo_successes": 0, "all_demos_executed_ok": False,
           "all_demos_matched": False, "executed_queries": 0, "valid_query_grid": False,
           "policy_complete": False, "query_em_any": None, "observed_query_em": None,
           "physical_generation_calls": 0, "physical_new_tokens": 0,
           "physical_generation_seconds": 0.0, "execution_seconds": 0.0,
           "task_elapsed_s": None, "repair_invoked_count": 0}
    golds = scoring.get(tid) if isinstance(scoring, dict) else None
    row["gold_available"] = isinstance(golds, list) and bool(golds)
    if task is None:
        return row
    row.update(task_status=task.get("status"), reason=task.get("reason"),
               task_elapsed_s=task.get("elapsed_s"), repair_invoked_count=task.get("repair_invoked_count", 0))
    p0 = (task.get("proposals") or {}).get("p0", {})
    gen = p0.get("generation") or {}
    status, finish = gen.get("status"), gen.get("finish_reason")
    if task.get("status") == "not_run" and not gen:
        outcome = "not_run"
    elif status in ("infrastructure_error", "worker_protocol_error"):
        outcome = "infrastructure_error"
    elif status == "budget_interrupted" or p0.get("status") == "budget_interrupted":
        outcome = "not_admitted" if finish == "admission_deadline" else "time_interrupted"
    elif str(status or "").startswith("ineligible"):
        outcome = status
    elif finish in ("max_new_tokens", "length") or status == "truncated":
        outcome = "token_truncated"
    elif status == "complete" and gen.get("ended_eos") is True:
        outcome = "eos_raw"
    elif status == "invalid_output" or finish == "no_eos":
        outcome = "no_eos"
    elif not gen:
        outcome = "missing_generation"
    else:
        outcome = "other:" + str(status)
    row.update(outcome=outcome, generation_status=status, finish_reason=finish,
               proposal_status=p0.get("status"), gen_tokens=gen.get("gen_tokens"),
               elapsed_s=gen.get("elapsed_s"), cache_hit=bool(gen.get("cache_hit")))
    row["complete_code"] = outcome == "eos_raw" and bool(p0.get("code"))
    row["ast_pass"] = outcome == "eos_raw" and p0.get("status") == "program_ready"
    if mode == "L":
        row["thinking_closed"] = str(gen.get("raw_text", "")).count("</think>") == 1
    demos, queries = p0.get("demo_results") or [], p0.get("query_results") or []
    # Non-ready proposals create placeholders; they never invoke the executor.
    if row["ast_pass"]:
        row["executed_demos"] = sum(r.get("status") in EXECUTED for r in demos)
        row["demo_successes"] = sum(r.get("status") == "ok" for r in demos)
        row["all_demos_executed_ok"] = bool(demos) and all(r.get("status") == "ok" for r in demos)
        row["all_demos_matched"] = row["all_demos_executed_ok"] and p0.get("all_demos_pass") is True
        row["executed_queries"] = sum(r.get("status") in EXECUTED for r in queries)
    query = queries[0] if queries else {}
    row["valid_query_grid"] = row["ast_pass"] and query.get("status") == "ok" and isinstance(query.get("grid"), list)
    resource = GEN_RESOURCE | {"not_run"}
    row["policy_complete"] = (outcome in ("eos_raw", "token_truncated", "no_eos")
                              and task.get("status") not in resource and bool(queries)
                              and not any(r.get("status") in resource for r in demos + queries))
    if row["valid_query_grid"] and row["gold_available"]:
        row["observed_query_em"] = query["grid"] == golds[0]
    if row["policy_complete"] and row["gold_available"]:
        row["query_em_any"] = row["observed_query_em"] is True
    cached = bool(gen.get("cache_hit"))
    inferred_calls = int(bool(gen) and not cached and
                         (gen.get("gen_tokens", 0) > 0 or gen.get("elapsed_s", 0) > 0))
    row["physical_generation_calls"] = task.get("physical_generation_calls", inferred_calls)
    row["physical_new_tokens"] = task.get("physical_new_tokens", 0 if cached else gen.get("gen_tokens", 0))
    row["physical_generation_seconds"] = task.get("physical_generation_seconds", 0 if cached else gen.get("physical_elapsed_s", gen.get("elapsed_s", 0)))
    row["execution_seconds"] = p0.get("execution_seconds", 0.0)
    return row


def summarize_program(root: Path, scoring):
    records, sources, conflicts = canonical_program_tasks(root)
    rows = [program_row(tid, mode, records.get((tid, mode)), scoring, sources.get((tid, mode)))
            for tid, mode in PLAN]
    by_arm = {}
    cost_keys = ("physical_generation_calls", "physical_new_tokens", "physical_generation_seconds",
                 "execution_seconds", "repair_invoked_count")
    for arm in ("D", "L"):
        sub = [r for r in rows if r["arm"] == arm]
        scored = sum(r["query_em_any"] is not None for r in sub)
        correct = sum(r["query_em_any"] is True for r in sub)
        n_code = sum(r["complete_code"] for r in sub)
        totals = {k: sum(r[k] for r in sub) for k in cost_keys}
        by_arm[arm] = {"n_rows": 2, "n_planned_queries": 2,
                       "n_recorded_queries": sum(r["source_file"] is not None for r in sub),
                       "n_policy_complete": sum(r["policy_complete"] for r in sub),
                       "outcomes": dict(Counter(r["outcome"] for r in sub)),
                       "complete_code": n_code, "ast_pass": sum(r["ast_pass"] for r in sub),
                       "thinking_closed": sum(r["thinking_closed"] is True for r in sub) if arm == "L" else None,
                       "executed_demos": sum(r["executed_demos"] for r in sub),
                       "tasks_with_successful_demo": sum(r["demo_successes"] > 0 for r in sub),
                       "all_demos_executed_ok": sum(r["all_demos_executed_ok"] for r in sub),
                       "all_demos_matched": sum(r["all_demos_matched"] for r in sub),
                       "executed_queries": sum(r["executed_queries"] for r in sub),
                       "valid_query_grids": sum(r["valid_query_grid"] for r in sub),
                       "query_em_true": correct, "query_em_scored": scored,
                       "query_em_over_planned": correct / 2 if scored == 2 else None,
                       "observed_query_hits": sum(r["observed_query_em"] is True for r in sub),
                       "physical_generation_s_per_complete_code": totals["physical_generation_seconds"] / n_code if n_code else None,
                       **totals}
    paired = [tid for tid in dict.fromkeys(t for t, _ in PLAN)
              if all(next(r for r in rows if r["task_id"] == tid and r["arm"] == m)["policy_complete"]
                     for m in ("D", "L"))]
    return {"status": "ambiguous" if conflicts else ("ok" if records else "missing"),
            "n_planned_calls": 4, "n_rows": 4, "rows": rows, "by_arm": by_arm,
            "n_planned_paired_queries": 2, "n_policy_complete_pairs": len(paired),
            "paired_tasks": paired, "conflicts": conflicts,
            **{k: sum(r[k] for r in rows) for k in cost_keys},
            "files_used": sorted(set(sources.values())),
            "scope": "Fixed development P2, one query per task; missing/resource-interrupted results retain planned denominators and null EM."}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("session_dir", type=Path, help="Downloaded artifact directory.")
    ap.add_argument("--scoring", type=Path, default=None, help="Offline query-answer JSON (never mounted in K).")
    ap.add_argument("--out", type=Path, default=None, help="Write JSON summary here.")
    args = ap.parse_args(argv)
    if not args.session_dir.is_dir():
        ap.error("session_dir must be an existing directory")
    scoring = read_json(args.scoring) if args.scoring else None
    if args.scoring and not isinstance(scoring, dict):
        ap.error("--scoring must map task_id to query output grids")
    summary = {"session_dir": str(args.session_dir),
               "kernel_probe": summarize_kernel(args.session_dir),
               "program": summarize_program(args.session_dir, scoring),
               "scoring_provided": str(args.scoring) if args.scoring else None,
               "limitations": ["P2 tasks are development data, not unseen validation.",
                               "Absent files are 'missing', never zero evidence.",
                               "Physical cost totals cover persisted canonical P0 records; killed in-flight calls may have unrecorded cost."]}
    text = json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text, encoding="utf-8")
    else:
        print(text, end="")
    k, p = summary["kernel_probe"], summary["program"]
    print("K: %s -- %s" % (k.get("status"), k.get("reason", k.get("note", ""))))
    if p.get("status") == "ok":
        for arm in ("D", "L"):
            b = p["by_arm"][arm]
            print("P arm %s: planned=%d outcomes=%s complete_code=%d executed_q=%d EM %d/%d scored" % (
                arm, b["n_rows"], b["outcomes"], b["complete_code"], b["executed_queries"],
                b["query_em_true"], b["query_em_scored"]))
    else:
        print("P: %s -- %s" % (p.get("status"), p.get("note", "")))


if __name__ == "__main__":
    main()
