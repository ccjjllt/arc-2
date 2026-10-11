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
    files = find_files(root, "*probe*.json", "*kernel*.json", "*trial*.json")
    trials, used = {}, []
    for path in files:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        cand = data.get("trials", data)
        if isinstance(cand, dict) and any(k in cand for k in ("A1", "A2", "B1", "B2")):
            trials, used = cand, [str(path)]
            break
    if not trials:
        return {"status": "missing", "files_seen": [str(p) for p in files],
                "note": "no K trial record with A1/A2/B1/B2 found; nothing timed"}
    detail = {}
    for arm in ("A1", "A2", "B1", "B2"):
        t = trials.get(arm, {}) if isinstance(trials.get(arm), dict) else {}
        detail[arm] = {k: t.get(k) for k in ("total_s", "prefill_s", "decode_s", "rel_rms",
                                             "top1_diff", "n_positions", "finish", "func_source")}
    problems = [a for a in detail if detail[a].get("finish") not in (None, "ok", "complete")]
    a1, a2 = detail["A1"], detail["A2"]
    try:
        a_rms = float(a1.get("rel_rms", 0.0) if a1.get("rel_rms") is not None else
                      a2.get("rel_rms", 0.0) or 0.0)
    except (TypeError, ValueError):
        a_rms = None
    verdict, reason = "incomplete", "trial fields missing"
    if problems:
        verdict, reason = "incomplete", "finish problems: " + ",".join(problems)
    elif a_rms is None or a1.get("top1_diff") is None:
        verdict, reason = "incomplete", "reference repeat fields missing"
    elif (a1.get("top1_diff") or 0) > 0 or (a2.get("top1_diff") or 0) > 0 or a_rms > A_RMS_TOL:
        verdict, reason = "unstable_reference", "A1/A2 top1 differ or rel RMS>1e-4; no perf claim"
    else:
        drift = []
        for b, a in (("B1", "A1"), ("B2", "A2")):
            bt, tol = detail[b], max(B_REL_FLOOR, 5 * a_rms)
            try:
                br = float(bt.get("rel_rms"))
            except (TypeError, ValueError):
                drift.append(b + ":rms_missing")
                continue
            if (bt.get("top1_diff") or 0) > 0 or br > tol:
                drift.append("%s: top1=%s rms=%.5g tol=%.5g" % (b, bt.get("top1_diff"), br, tol))
        if drift:
            verdict, reason = "numerical_drift", "; ".join(drift) + "; keep fallback backend"
        else:
            try:
                g1 = 1 - detail["B1"]["total_s"] / detail["A1"]["total_s"]
                g2 = 1 - detail["B2"]["total_s"] / detail["A2"]["total_s"]
                verdict = "worth_full_output_bridge" if (g1 >= GAIN and g2 >= GAIN) else "insufficient_gain"
                reason = "gains %.1f%%/%.1f%% vs 15%% bar" % (100 * g1, 100 * g2)
            except (TypeError, ZeroDivisionError):
                verdict, reason = "incomplete", "timing fields missing"
    return {"status": verdict, "reason": reason, "files_used": used,
            "files_seen": [str(p) for p in files], "trials": detail}


def arm_of(item: dict) -> str:
    for key in ("arm", "program_arm", "mode"):
        v = str(item.get(key, "")).upper()
        if v.startswith("D"):
            return "D"
        if v.startswith("L"):
            return "L"
    name = str(item.get("stage", "") + item.get("name", "")).lower()
    if "direct" in name or "_d" in name:
        return "D"
    if "think" in name or "_l" in name:
        return "L"
    th = item.get("enable_thinking", item.get("generation", {}).get("enable_thinking"))
    if th is True:
        return "L"
    if th is False:
        return "D"
    return "?"


def summarize_program(root: Path, scoring):
    stage_files = [p for p in find_files(root, "*.json")
                   if p.name not in ("scoring.json", "session_identity.json", "requested_config.json",
                                     "supervisor.json", "SUMMARY.json") or True]
    rows, gen_counts, exec_counts, files_used = [], Counter(), Counter(), []
    for path in stage_files:
        data = read_json(path)
        if not isinstance(data, dict):
            continue
        items = data.get("stages", []) if path.name == "SUMMARY.json" else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            tasks = item.get("tasks", [])
            if not tasks or not any(isinstance(t, dict) and "proposals" in t for t in tasks):
                continue
            files_used.append(str(path))
            for task in tasks:
                tid = task.get("task_id", "?")
                props = task.get("proposals", {})
                p0 = props.get("p0", {}) if isinstance(props, dict) else {}
                if not isinstance(p0, dict) or not p0:
                    rows.append({"task_id": tid, "arm": arm_of({**item, **task}),
                                 "outcome": "not_run", "em": None})
                    continue
                gen = p0.get("generation", {}) if isinstance(p0.get("generation"), dict) else {}
                gstat, finish = gen.get("status"), gen.get("finish_reason", "")
                arm = arm_of({**item, **task, **p0, "generation": gen})
                if gstat in GEN_RESOURCE or finish == "admission_deadline":
                    outcome = "resource_or_not_admitted"
                elif gstat == "truncated" or finish == "length":
                    outcome = "token_truncated"
                elif gstat == "time_interrupted" or finish == "timeout":
                    outcome = "time_interrupted"
                elif gstat in GEN_DONE and finish in ("stop", "eos", "", None):
                    outcome = "eos_raw"
                elif gstat in GEN_DONE:
                    outcome = "eos_raw"
                elif gstat in (None, "", "not_run"):
                    outcome = "not_run"
                else:
                    outcome = "infra_or_other:%s" % gstat
                gen_counts[arm + ":" + outcome] += 1
                code = p0.get("code") or p0.get("final") or ""
                complete = outcome == "eos_raw" and bool(code)
                qr = p0.get("query_results", []) if isinstance(p0.get("query_results"), list) else []
                executed = sum(1 for q in qr if isinstance(q, dict) and q.get("status") in
                               ("ok", "execution_error", "execution_timeout", "output_limit",
                                "program_rejected", "incomplete_thinking", "incomplete_final_program"))
                exec_counts[arm + ":executed_queries"] += executed
                golds = scoring.get(tid) if isinstance(scoring, dict) else None
                em = None
                if outcome == "eos_raw" and golds:
                    for qi, q in enumerate(qr):
                        if qi < len(golds) and isinstance(q, dict) and q.get("status") == "ok":
                            em = (q.get("grid") == golds[qi]) if em is None else em
                rows.append({"task_id": tid, "arm": arm, "outcome": outcome,
                             "complete_code": complete, "executed_queries": executed,
                             "query_em_any": em, "gen_tokens": gen.get("gen_tokens"),
                             "elapsed_s": gen.get("elapsed_s")})
    if not rows and not files_used:
        events = find_files(root, "*-events.jsonl")
        return {"status": "missing", "rows": [],
                "note": "no program stage with tasks/proposals found",
                "stage_files_seen": len(stage_files), "event_files_seen": [str(p) for p in events]}
    by_arm = {}
    for arm in ("D", "L"):
        sub = [r for r in rows if r["arm"] == arm]
        by_arm[arm] = {"n_rows": len(sub),
                       "outcomes": dict(Counter(r["outcome"] for r in sub)),
                       "complete_code": sum(1 for r in sub if r["complete_code"]),
                       "executed_queries": sum(r["executed_queries"] for r in sub),
                       "query_em_true": sum(1 for r in sub if r["query_em_any"] is True),
                       "query_em_scored": sum(1 for r in sub if r["query_em_any"] is not None)}
    both = {(r["task_id"], r["arm"]) for r in rows}
    return {"status": "ok", "n_rows": len(rows), "rows": rows, "by_arm": by_arm,
            "paired_tasks": sorted({t for t, _ in both}),
            "gen_counts": dict(gen_counts), "exec_counts": dict(exec_counts),
            "files_used": sorted(set(files_used)),
            "scope": "dev P2 only (db118e2a, 291dc1e1); query EM is a development check, not a competition score"}


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
                               "Telemetry (if any) is sample-weighted, not a duty-cycle proof."]}
    text = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
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
            print("P arm %s: rows=%d outcomes=%s complete_code=%d executed_q=%d EM %d/%d" % (
                arm, b["n_rows"], b["outcomes"], b["complete_code"], b["executed_queries"],
                b["query_em_true"], b["query_em_scored"]))
    else:
        print("P: %s -- %s" % (p.get("status"), p.get("note", "")))


if __name__ == "__main__":
    main()
