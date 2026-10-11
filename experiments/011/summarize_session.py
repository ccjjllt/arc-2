"""Manual, stdlib-only post-hoc analysis of a downloaded 011 session.

Usage: python3 summarize_session.py SESSION_DIR [--solutions solutions.json]
       [--out summary.json] [--telemetry-timezone UTC]
No GPU/model imports. Labels are used only for offline program scoring.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import statistics
from zoneinfo import ZoneInfo

MODEL_DONE = {"complete", "invalid_output"}
EXEC_DONE = {"ok", "execution_error", "execution_timeout", "output_limit", "program_rejected",
             "incomplete_thinking", "incomplete_final_program"}
RESOURCE = {"budget_interrupted", "infrastructure_error", "executor_unavailable", "worker_protocol_error"}


def read_json(path):
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def read_lines(path, errors):
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            try:
                value = json.loads(line)
                if isinstance(value, dict):
                    yield value
            except ValueError:
                errors.append({"file": str(path), "line": number, "reason": "invalid_json"})


def stats(values):
    values = sorted(v for v in values if isinstance(v, (int, float)))
    return {"n": len(values), "mean": statistics.mean(values), "median": statistics.median(values),
            "max": max(values)} if values else {"n": 0, "mean": None, "median": None, "max": None}


def number(value):
    match = re.search(r"-?\d+(?:\.\d+)?", str(value or ""))
    return float(match.group()) if match else None


def phase(row):
    return row.get("run_tag") or ("unclassified_program/" + row["program_arm"] if row.get("program_arm") else "unclassified_candidate")


def candidates(root, errors):
    rows, seen = [], set()
    paths = sorted(root.rglob("candidates.jsonl"))
    duplicates = 0
    for path in paths:
        for row in read_lines(path, errors):
            key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()
            if key in seen:
                duplicates += 1
                continue
            seen.add(key)
            rows.append(row)
    latest, intervals = {}, []
    for index, row in enumerate(sorted(rows, key=lambda r: r.get("recorded_unix_s", 0))):
        latest[row.get("identity_key", "missing:%d" % index)] = row
        end, duration = row.get("recorded_unix_s"), row.get("elapsed_s")
        if not row.get("cache_hit") and isinstance(end, (int, float)) and isinstance(duration, (int, float)) and duration > 0:
            intervals.append((end-duration, end, phase(row)))
    def totals(selected):
        return {"records": len(selected), "status_counts": dict(Counter(r.get("status", "missing") for r in selected)),
                "journal_cache_hit_records": sum(bool(r.get("cache_hit")) for r in selected),
                "physical_elapsed_s": sum(0 if r.get("cache_hit") else r.get("physical_elapsed_s", r.get("elapsed_s", 0)) or 0 for r in selected),
                "physical_output_tokens": sum(0 if r.get("cache_hit") else r.get("gen_tokens", 0) or 0 for r in selected),
                "elapsed_s": stats([r.get("elapsed_s") for r in selected]),
                "generate_s": stats([r.get("generate_s") for r in selected])}
    report = {"files": [str(p) for p in paths], "exact_duplicate_records_ignored": duplicates,
              "all_attempts": totals(rows), "latest_identity_statuses": dict(Counter(r.get("status", "missing") for r in latest.values())),
              "by_phase": {label: totals([r for r in rows if phase(r) == label]) for label in sorted({phase(r) for r in rows})}}
    return report, intervals


def telemetry(root, intervals, timezone_name, errors):
    tz = ZoneInfo(timezone_name)
    groups, seen = defaultdict(list), set()
    for path in sorted(root.rglob("*telemetry*.csv")):
        with path.open(encoding="utf-8") as handle:
            for line, raw in enumerate(csv.DictReader(handle), 2):
                row = {str(k).strip().split(" [")[0]: v for k, v in raw.items() if k is not None}
                try:
                    stamp = datetime.fromisoformat(row["timestamp"].strip().replace("/", "-"))
                    stamp = stamp.replace(tzinfo=tz) if stamp.tzinfo is None else stamp
                    epoch, gpu = stamp.timestamp(), str(row["index"]).strip()
                except (ValueError, KeyError):
                    errors.append({"file": str(path), "line": line, "reason": "invalid_telemetry_timestamp_or_index"})
                    continue
                key = (epoch, gpu, tuple(sorted((k, str(v)) for k, v in row.items())))
                if key in seen:
                    continue
                seen.add(key)
                matches = {label for start, end, label in intervals if start <= epoch <= end}
                label = next(iter(matches)) if len(matches) == 1 else ("overlapping_candidate_intervals" if matches else "unclassified_interval")
                groups[(gpu, "all")].append(row)
                groups[(gpu, label)].append(row)
    output = {}
    for (gpu, label), rows in sorted(groups.items()):
        output.setdefault(gpu, {})[label] = {"samples": len(rows),
            **{field: stats([number(row.get(field)) for row in rows]) for field in
               ("utilization.gpu", "utilization.memory", "memory.used", "memory.total", "power.draw")}}
    return {"naive_timestamp_timezone_assumption": timezone_name,
            "interval_rule": "[recorded_unix_s - elapsed_s, recorded_unix_s], non-cache journal records only",
            "aggregation": "sample-weighted means; utilization percent, memory MiB, power W; not continuous-time measurements",
            "per_gpu": output}


def stage_artifacts(root, errors):
    stages = {}
    for path in sorted(root.rglob("*.json"), key=lambda p: (p.name != "SUMMARY.json", str(p))):
        if path.name in ("scoring.json", "session_identity.json", "requested_config.json"):
            continue
        try:
            data = read_json(path)
        except (ValueError, OSError):
            continue
        if not isinstance(data, dict):
            continue
        items = data.get("stages", []) if path.name == "SUMMARY.json" else [data]
        for item in items:
            if not isinstance(item, dict) or not item.get("stage"):
                continue
            desc = item["stage"] if isinstance(item["stage"], dict) else {}
            name, kind = desc.get("name", item["stage"]), item.get("type", desc.get("type", "grid"))
            if not isinstance(name, str):
                continue
            stages[(str(path.parent), name)] = {**item, "stage": name, "type": kind}
    for path in sorted(root.rglob("*-events.jsonl")):
        key = (str(path.parent), path.name[:-len("-events.jsonl")])
        stage = stages.setdefault(key, {"stage": key[1], "type": "program", "tasks": []})
        tasks = {t["task_id"]: t for t in stage.get("tasks", [])}
        for event in read_lines(path, errors):
            if event.get("program_task"):
                task = event["program_task"]
                tasks.setdefault(task["task_id"], task)
            elif event.get("task_id") and event.get("program_arm"):
                task = tasks.setdefault(event["task_id"], {"task_id": event["task_id"], "proposals": {}})
                task.setdefault("proposals", {}).setdefault(event["program_arm"], event["proposal"])
        stage["tasks"] = list(tasks.values())
    return list(stages.values())


def proposal_result(proposals, name, qi):
    item = proposals.get(name, {})
    gen = item.get("generation", {}).get("status")
    cases = item.get("query_results", [])
    case = cases[qi] if qi < len(cases) else {}
    state, case_state = item.get("status"), case.get("status")
    grid = case.get("grid") if case_state == "ok" else None
    if gen in RESOURCE or state in RESOURCE or case_state in RESOURCE:
        return "resource_interrupted", grid
    if name == "repair" and state == "ineligible_no_complete_p0" and proposals.get("p0", {}).get("generation", {}).get("status") in MODEL_DONE:
        return "complete", None
    if gen in MODEL_DONE and case_state in EXEC_DONE:
        return "complete", grid
    if str(state).startswith("ineligible") or str(gen).startswith("ineligible"):
        return "ineligible", grid
    return "not_run", grid


def score_program(stage, solutions):
    rows, unknown, diagnostics = [], [], []
    mechanism_ready = {}
    for task in stage.get("tasks", []):
        tid, proposals = task["task_id"], task.get("proposals", {})
        available = {}
        for name in ("p0", "restart", "repair"):
            proposal = proposals.get(name, {})
            generation = proposal.get("generation", {})
            available[name] = bool(generation) and generation.get("finish_reason") not in ("admission_deadline", "input_limit") and (
                (generation.get("gen_tokens", 0) or 0) > 0 or (generation.get("elapsed_s", 0) or 0) > 0 or generation.get("finish_reason") == "generation_error")
            diagnostics.append({"task_id": tid, "proposal": name, "proposal_status": proposal.get("status", "not_run"),
                "generation_status": generation.get("status"), "finish_reason": generation.get("finish_reason", "not_invoked"),
                "gen_tokens": generation.get("gen_tokens", 0), "generation_available": available[name],
                "cache_hit": bool(generation.get("cache_hit")), "physically_invoked": available[name] and not generation.get("cache_hit", False)})
        p0 = proposals.get("p0", {})
        mechanism_ready[tid] = bool(p0.get("code") and p0.get("final") and p0.get("demo_results")) and all(
            available[name] and proposals[name].get("generation", {}).get("status") in MODEL_DONE for name in ("restart", "repair"))
        golds = solutions.get(tid) if solutions is not None else None
        counts = [len(p.get("query_results", [])) for p in proposals.values()]
        count = len(golds) if isinstance(golds, list) else max(counts + [0])
        if not count:
            unknown.append({"task_id": tid, "status": task.get("status", "unconfirmed")})
        for qi in range(count):
            gold = golds[qi] if isinstance(golds, list) else None
            arms = {}
            for alternative in ("restart", "repair"):
                pair = [proposal_result(proposals, name, qi) for name in ("p0", alternative)]
                states = [x[0] for x in pair]
                status = "complete" if all(s == "complete" for s in states) else (
                    "resource_interrupted" if "resource_interrupted" in states or task.get("status") in RESOURCE else
                    "partial" if "complete" in states else "ineligible" if "ineligible" in states else "not_run")
                observed = any(grid == gold for _, grid in pair if grid is not None) if gold is not None else None
                arms["p0+"+alternative] = {"status": status, "em": observed if status == "complete" else None, "partial_or_complete_hit": observed}
            rows.append({"task_id": tid, "query_index": qi, "gold_available": gold is not None, "arms": arms})
    categories, mechanism_categories = Counter(), Counter()
    for row in rows:
        left, right = row["arms"]["p0+restart"]["em"], row["arms"]["p0+repair"]["em"]
        row["feedback_mechanism_comparable"] = bool(mechanism_ready.get(row["task_id"]) and left is not None and right is not None)
        if left is not None and right is not None:
            category = ("both_correct" if left else "repair_gain") if right else ("repair_loss" if left else "both_wrong")
            categories[category] += 1
            if row["feedback_mechanism_comparable"]: mechanism_categories[category] += 1
    physical = {name: sum(r["physically_invoked"] for r in diagnostics if r["proposal"] == name) for name in ("p0", "restart", "repair")}
    return {"stage": stage["stage"], "query_rows": rows, "n_query_rows": len(rows), "declared_planned_queries": stage.get("planned_queries"),
            "tasks_with_unknown_query_denominator": unknown, "score_scope": "fixed-budget policy @2; skipped repair is a policy fallback, not feedback-vs-restart mechanism evidence",
            "policy_paired_complete_scored": sum(categories.values()), "policy_paired_categories": dict(categories),
            "feedback_mechanism_paired_complete_scored": sum(mechanism_categories.values()), "feedback_mechanism_categories": dict(mechanism_categories),
            "feedback_mechanism_definition": "P0 has final code and demo feedback; restart and repair generation both actually available with terminal model outcomes; both policy arms completely scored on the same query",
            "physical_generation_calls": sum(physical.values()), "physical_generation_calls_per_arm": physical, "repair_invoked_count": physical["repair"],
            "proposal_diagnostics": diagnostics,
            "proposal_status_counts_per_arm": {name: dict(Counter(r["proposal_status"] for r in diagnostics if r["proposal"] == name)) for name in physical},
            "finish_reason_counts_per_arm": {name: dict(Counter(r["finish_reason"] for r in diagnostics if r["proposal"] == name)) for name in physical},
            "arms": {arm: {"status_denominators": dict(Counter(r["arms"][arm]["status"] for r in rows)),
                            "n_scored_complete": sum(r["arms"][arm]["em"] is not None for r in rows),
                            "exact_match_complete": sum(r["arms"][arm]["em"] is True for r in rows)}
                     for arm in ("p0+restart", "p0+repair")}}


def summarize(root, solutions_path=None, telemetry_timezone="UTC"):
    errors = []
    journal, intervals = candidates(root, errors)
    stages = stage_artifacts(root, errors)
    reused = []
    def visit(value):
        if isinstance(value, dict):
            if "identity_key" in value and value.get("cache_hit"):
                reused.append(value)
            else:
                for child in value.values(): visit(child)
        elif isinstance(value, list):
            for child in value: visit(child)
    for stage in stages: visit(stage)
    solutions = read_json(Path(solutions_path)) if solutions_path else None
    if solutions is not None and not isinstance(solutions, dict): raise ValueError("solutions must map task_id to query output grids")
    return {"session_dir": str(root), "candidate_journal": journal,
            "stage_coverage": [{"stage": s["stage"], "type": s.get("type"), "declared_planned_queries": s.get("planned_queries"),
                "declared_planned_candidates": s.get("planned_candidates"), "returned_query_rows": len(s.get("queries", [])), "returned_tasks": len(s.get("tasks", [])),
                "returned_grid_candidate_statuses": dict(Counter(c.get("status", "missing") for q in s.get("queries", []) for c in q.get("candidates", [])))} for s in stages],
            "cached_stage_references": {"count": len(reused), "physical_elapsed_s": sum(r.get("physical_elapsed_s", 0) or 0 for r in reused),
                "historical_logical_elapsed_s": sum(r.get("elapsed_s", 0) or 0 for r in reused)},
            "telemetry": telemetry(root, intervals, telemetry_timezone, errors),
            "program_scoring": [score_program(stage, solutions) for stage in stages if stage.get("type") == "program"],
            "solutions_explicitly_provided": str(solutions_path) if solutions_path else None, "parse_errors": errors,
            "limitations": ["SUMMARY/scoring file presence or status is not evidence of completed candidate coverage.",
                "Telemetry outside candidate intervals is unclassified: it may be loading, training, execution, saving or idle.",
                "Candidate elapsed includes preparation/generation overhead; sparse telemetry cannot isolate pure decode or GPU duty cycle.",
                "No journal means no reliable generation cost; a process killed before journaling leaves unmeasured work.",
                "Program tasks absent from both stage artifacts and event journals have unknown per-query denominators."]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--solutions", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--telemetry-timezone", default="UTC", help="Timezone for naive nvidia-smi CSV timestamps; explicit default assumption: UTC")
    args = parser.parse_args()
    if not args.session_dir.is_dir(): parser.error("session_dir must be an existing directory")
    text = json.dumps(summarize(args.session_dir, args.solutions, args.telemetry_timezone), ensure_ascii=False, indent=2) + "\n"
    if args.out: args.out.parent.mkdir(parents=True, exist_ok=True); args.out.write_text(text, encoding="utf-8")
    else: print(text, end="")
