"""011 manual runner: one resident base, durable candidates, supervised limits.

  python scripts/run_budgeted_grid_experiment.py --config run.json --dry-run
  python scripts/run_budgeted_grid_experiment.py --config run.json

Config contains explicit stages/query files. No evaluation/confirmation manifest
is implicitly read. A parent process enforces load and overall deadlines while
the one worker owns the model for its entire lifetime.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import sys
import time
import traceback

import solve_grid as solver
import grid_runtime as runtime

# 012 P0-only frozen plan (§7.2-7.3): one fresh P0 per call, no history carryover.
P0_CALL_SECONDS = 760.0
P0_FROZEN_LIMITS = {"max_new_tokens": 3072, "max_call_seconds": 720,
                    "max_task_seconds": 760, "max_prompt_tokens": 8192, "seed": 20261011}
P0_EXPECTED_SEQUENCE = (("db118e2a", "D"), ("db118e2a", "L"),
                        ("291dc1e1", "L"), ("291dc1e1", "D"))


def _stage_mode(stage):
    """D/L mode for program stages; L when thinking enabled."""
    return "L" if bool(stage.get("enable_thinking", True)) else "D"


def _program_call_plan(config):
    """File-order P0 call plan: one entry per distinct task per program stage."""
    plan = []
    for stage in config.get("stages", []):
        if stage.get("type", "grid") != "program":
            continue
        pairs = solver.parse_query_file(stage["query_file"])
        tids = list(dict.fromkeys(tid for tid, _ in pairs))
        for tid in tids:
            plan.append({"stage": stage["name"], "task_id": tid,
                         "mode": _stage_mode(stage),
                         "policy": stage.get("policy", "legacy"),
                         "enable_thinking": bool(stage.get("enable_thinking", True)),
                         "reasoning_effort": stage.get("reasoning_effort", "low"),
                         "preserve_thinking": bool(stage.get("preserve_thinking", True)),
                         "limits": dict(stage.get("limits", {}))})
    return plan


def _validate_program_stages(config):
    """Pre-load (§7.1.5): policy/limits/inputs/parse/order; old 1536/360 fails here."""
    import solve_program as program
    stages = config.get("stages", [])
    program_stages = [s for s in stages if s.get("type") == "program"]
    for stage in program_stages:
        policy = stage.get("policy", "legacy")
        if policy not in ("p0_only", "legacy"):
            raise ValueError("stage %s: unknown policy %r (expected p0_only)" % (stage["name"], policy))
        if policy == "p0_only":
            if not isinstance(stage.get("enable_thinking", None), bool):
                raise ValueError("stage %s: p0_only requires explicit enable_thinking bool (D=False/L=True)" % stage["name"])
            if stage.get("reasoning_effort", "low") != "low":
                raise ValueError("stage %s: p0_only reasoning_effort must be low" % stage["name"])
            if bool(stage.get("preserve_thinking", True)) is not True:
                raise ValueError("stage %s: p0_only preserve_thinking must be true" % stage["name"])
            if "arms" in stage and list(stage["arms"]) != ["p0"]:
                raise ValueError("stage %s: p0_only arms must be [\"p0\"]" % stage["name"])
            limits = dict(stage.get("limits", {}))
            for key, want in P0_FROZEN_LIMITS.items():
                if key in limits and limits[key] != want:
                    raise ValueError("stage %s: p0_only %s=%r must be %r (§7.2 frozen; old 1536/360 rejected pre-load)" % (
                        stage["name"], key, limits[key], want))
            # Seed freezing: same task + same p0 seed base, D/L share seed.
            merged = dict(P0_FROZEN_LIMITS)
            merged.update(limits)
            if int(merged.get("seed", P0_FROZEN_LIMITS["seed"])) != 20261011:
                raise ValueError("stage %s: p0_only seed must be 20261011" % stage["name"])
            pairs = solver.parse_query_file(stage["query_file"])
            if len(pairs) != 1 or len(set(t for t, _ in pairs)) != 1:
                raise ValueError("stage %s: p0_only stages must be single-task single-query (012 §7.2)" % stage["name"])
            # p0_only new ceilings (§7.1.2): 3072/720/760/8192. Old
            # ProgramLimits.validate still caps at 1536/360, so validate the
            # frozen values here instead of delegating (explicit old values
            # already rejected above, pre-load).
            _validate_p0_limits(merged, stage["name"])
        else:
            # Legacy three-proposal mode keeps original ceilings; validate early.
            program.ProgramLimits(**stage.get("limits", {})).validate()
    p0_stages = [s for s in program_stages if s.get("policy") == "p0_only"]
    if len(p0_stages) == 4:
        got = []
        for stage in p0_stages:
            tid = solver.parse_query_file(stage["query_file"])[0][0]
            got.append((tid, _stage_mode(stage)))
        if tuple(got) != P0_EXPECTED_SEQUENCE:
            raise ValueError("p0_only order must be db118e2a D→db118e2a L→291dc1e1 L→291dc1e1 D, got %r" % (got,))
        first = p0_stages[0]
        if "time_budget_s" in first and float(first["time_budget_s"]) <= 760:
            raise ValueError("first p0_only stage must include loading (4500 or omitted), not 760")


def _validate_p0_limits(merged, stage_name):
    """p0_only ceilings (§7.1.2/§7.2): 3072/720/760/8192; rest keep 011 bounds."""
    import math
    ceilings = {"max_new_tokens": 3072, "max_prompt_tokens": 8192,
                "max_call_seconds": 720, "max_task_seconds": 760,
                "max_feedback_chars": 3000, "per_grid_seconds": 2,
                "per_program_seconds": 20, "memory_mb": 512}
    for key, ceiling in ceilings.items():
        value = merged.get(key, P0_FROZEN_LIMITS.get(key))
        if value is None:
            continue
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 < value <= ceiling:
            raise ValueError("stage %s: invalid p0_only limit %s=%r (ceiling %r)" % (stage_name, key, value, ceiling))
        if key in ("max_new_tokens", "max_prompt_tokens", "max_feedback_chars", "memory_mb") and type(value) is not int:
            raise ValueError("stage %s: p0_only limit %s must be an integer" % (stage_name, key))


def _global_work_remaining_s(session):
    """Global admission clock: session.deadline-save_window-now (§7.3)."""
    return float(session.deadline) - float(session.save_window_s) - time.monotonic()


def _call_program_task(program, session, tid, task, limits, deadline, task_index, journal, stage):
    """Forward p0_only/thinking compatibly; old solve_program keeps working."""
    import inspect
    kwargs = {}
    try:
        params = inspect.signature(program.run_program_task).parameters
    except (TypeError, ValueError):
        params = {}
    if "policy" in params and "policy" in stage:
        kwargs["policy"] = stage["policy"]
    for key in ("enable_thinking", "reasoning_effort", "preserve_thinking", "mode"):
        if key in params:
            if key == "mode" and "mode" not in stage:
                kwargs[key] = _stage_mode(stage)
            elif key in stage:
                kwargs[key] = stage[key]
    return program.run_program_task(session, tid, task, limits, deadline=deadline,
                                    task_index=task_index, journal=journal, **kwargs)


def atomic_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, default=str)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def read_config(path):
    with open(path, encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError("config must be an object")
    for field in ("model_dir", "challenges", "out_dir"):
        if not config.get(field):
            raise ValueError("config.%s required" % field)
    budget = float(config.get("time_budget_s", 7200))
    load_limit = float(config.get("load_limit_s", 1200))
    if not 0 < budget <= 7200:
        raise ValueError("this manual session must be <=7200 wall-clock seconds")
    if not 0 < load_limit <= min(1200, budget):
        raise ValueError("load_limit_s must be positive and <=1200 and session budget")
    if "notebook_start_monotonic" in config and "budget_elapsed_s" in config:
        raise ValueError("use notebook_start_monotonic OR budget_elapsed_s, not both")
    if float(config.get("budget_elapsed_s", 0)) < 0:
        raise ValueError("budget_elapsed_s must be nonnegative")
    cfg = config.get("runtime", {})
    if not cfg.get("base_source") and not cfg.get("base_weight_manifest"):
        raise ValueError("runtime.base_source or base_weight_manifest required")
    if not 0 <= float(cfg.get("save_window_s", 300)) < budget:
        raise ValueError("invalid save_window_s")
    stages = config.get("stages")
    if not isinstance(stages, list) or not stages:
        raise ValueError("nonempty stages required")
    names = set()
    for stage in stages:
        if not isinstance(stage, dict) or not stage.get("name") or stage["name"] in names:
            raise ValueError("stages need unique nonempty names")
        names.add(stage["name"])
        if Path(stage["name"]).name != stage["name"]:
            raise ValueError("stage name must not contain directories")
        if stage.get("type", "grid") not in ("grid", "ttt", "program"):
            raise ValueError("unsupported stage type")
        if not stage.get("query_file"):
            raise ValueError("every stage needs an explicit query_file")
        solver.parse_query_file(stage["query_file"])
        if stage.get("adapter"):
            adapter = Path(stage["adapter"])
            if not (adapter / "adapter_config.json").is_file() or not (
                    list(adapter.glob("adapter_model*.safetensors")) or list(adapter.glob("adapter_model*.bin"))):
                raise ValueError("adapter files must exist before GPU loading: " + str(adapter))
        if float(stage.get("time_budget_s", budget)) <= 0:
            raise ValueError("stage time_budget_s must be positive")
        if stage.get("type", "grid") == "grid":
            if int(stage.get("max_new_tokens", 960)) <= 0:
                raise ValueError("max_new_tokens must be positive")
            solver._normalize_formatter(stage.get("formatter", "f1_compact"))
            specs = stage.get("candidates", [{"view": "original", "do_sample": False}])
            if not specs:
                raise ValueError("empty candidates")
            for item in specs:
                if item.get("view", "original") not in ("original", "transpose"):
                    raise ValueError("only original/transpose are implemented")
        if stage.get("type") == "ttt" and int(stage.get("updates", 4)) != 4:
            raise ValueError("011 first TTT implementation supports A4 only, with explicit A0 pair")
        if stage.get("type") == "ttt" and any(float(stage.get(key, 300)) <= 0 for key in (
                "a4_adaptation_estimate_s", "a4_generation_reserve_s_per_query")):
            raise ValueError("TTT admission and generation reserve estimates must be positive")
    _validate_program_stages(config)
    return config


def load_inputs(config):
    with open(config["challenges"], encoding="utf-8") as handle:
        original = solver.as_challenge_map(json.load(handle))
    # Strip test outputs even if a caller supplies a training-format task map.
    tasks = {tid: {"train": [{"input": p["input"], "output": p["output"]}
                            for p in task.get("train", [])],
                   "test": [{"input": q["input"]} for q in task.get("test", [])]}
             for tid, task in original.items()}
    for stage in config["stages"]:
        for tid, qidx in solver.parse_query_file(stage["query_file"]):
            if tid not in tasks or not 0 <= qidx < len(tasks[tid]["test"]):
                raise ValueError("unknown task/query: %s/%s" % (tid, qidx))
    return tasks


def status_counts(rows):
    counts = {}
    for row in rows:
        key = row.get("status", "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def pick_two(candidates):
    """Input-free selection: valid complete grids, frequency then queue order."""
    groups = {}
    for pos, candidate in enumerate(candidates):
        if candidate.get("status") != "complete" or candidate.get("grid") is None:
            continue
        key = runtime.canonical_hash(candidate["grid"])
        entry = groups.setdefault(key, {"grid": candidate["grid"], "count": 0, "first": pos})
        entry["count"] += 1
    return [item["grid"] for item in sorted(groups.values(), key=lambda x: (-x["count"], x["first"]))[:2]]


def run_grid_stage(session, stage, tasks, deadline, out_dir):
    runtime.activate_adapter(session, stage.get("adapter"), name=stage.get("adapter_name"))
    formatter = solver._normalize_formatter(stage.get("formatter", "f1_compact"))
    if stage.get("adapter"):
        detected, source, fallback = solver.resolve_formatter("auto", stage["adapter"])
        if not fallback and detected != formatter:
            raise ValueError("adapter formatter mismatch: %s vs %s from %s" % (detected, formatter, source))
    queries = solver.parse_query_file(stage["query_file"])
    specs = stage.get("candidates", [{"view": "original", "do_sample": False}])
    results = []
    started = time.monotonic()
    for tid, qidx in queries:
        task = tasks[tid]
        candidates = []
        for position, spec in enumerate(specs):
            view = spec.get("view", "original")
            sample_index = int(spec.get("sample_index", position))
            seed = solver.task_view_seed(int(stage.get("seed", 20261011)), tid, qidx, view, sample_index)
            row = runtime.generate_grid(session, tid, qidx, task["train"], task["test"][qidx]["input"],
                formatter=formatter, view=view, sample_index=sample_index,
                do_sample=bool(spec.get("do_sample", False)),
                max_new_tokens=int(stage.get("max_new_tokens", 960)), seed=seed,
                deadline=deadline, run_tag=stage["name"],
                estimate_s=float(spec.get("estimate_s", stage.get("estimate_s", 0))),
                max_prompt_tokens=stage.get("max_prompt_tokens"),
                max_time_s=spec.get("max_time_s", stage.get("max_time_s")))
            candidates.append(row)
        results.append({"task_id": tid, "query_index": qidx, "candidates": candidates,
                        "picked_grids": pick_two(candidates)})
        atomic_json(out_dir / (stage["name"] + ".json"), {"stage": stage, "queries": results,
                    "planned_queries": len(queries), "weight_identity": session.active_identity})
    flat = [c for q in results for c in q["candidates"]]
    return {"stage": stage["name"], "type": "grid", "queries": results,
            "planned_queries": len(queries), "planned_candidates": len(queries) * len(specs),
            "status_counts": status_counts(flat), "cache_hits": sum(bool(x.get("cache_hit")) for x in flat),
            "elapsed_s": time.monotonic() - started, "weight_identity": session.active_identity}


def run_ttt_stage(session, stage, tasks, deadline, out_dir):
    """Optional explicit A0/A4 stage; not selected by the default first session."""
    import ttt_per_task as ttt
    runtime.activate_adapter(session, stage.get("adapter"), name=stage.get("adapter_name"))
    source = None if session.base_only else session.current_adapter
    session.model, initial = ttt.ensure_ttt_adapter(session.model,
        adapter_name=stage.get("ttt_adapter_name", "ttt_base"), source_adapter=source,
        seed=int(stage.get("seed", 20261011)), torch_module=session.torch)
    pairs = solver.parse_query_file(stage["query_file"])
    by_task = {}
    for tid, qi in pairs:
        by_task.setdefault(tid, []).append(qi)
    result = []
    def log_update(task_id, update):
        with open(out_dir / (stage["name"] + "-updates.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"task_id": task_id, "stage": stage["name"],
                                     "recorded_unix_s": time.time(), **update},
                                    ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    for tid, indices in by_task.items():
        if time.monotonic() >= deadline:
            result.append({"task_id": tid, "query_indices": indices, "status": "budget_interrupted"})
            continue
        # Restore the shared initial state and measure the same O+T at zero
        # updates. Query gold is never available to either adaptation call.
        initial_report = ttt.adapt_task(session.model, session.tokenizer, tid, tasks[tid]["train"], initial,
            formatter=stage.get("formatter", "f1_compact"), updates=0, grad_accum=4,
            deadline=deadline, eos_id=session.tokenizer.eos_token_id, torch_module=session.torch)
        runtime.mark_adapter_state(session, {"kind": "ttt", "task_id": tid,
            "applied_updates": 0, "adapter_sha256": initial_report.get("adapter_sha256")})
        a0_rows = []
        for qi in indices:
            a0_candidates = [runtime.generate_grid(session, tid, qi, tasks[tid]["train"],
                tasks[tid]["test"][qi]["input"], formatter=stage.get("formatter", "f1_compact"),
                view=view, do_sample=False, max_new_tokens=int(stage.get("max_new_tokens", 960)),
                deadline=deadline, run_tag=stage["name"] + "/A0") for view in ("original", "transpose")]
            a0_rows.append({"task_id": tid, "query_index": qi, "candidates": a0_candidates,
                            "picked_grids": pick_two(a0_candidates)})
        # Reserve enough time to compare the full A4 against the completed A0;
        # these configurable estimates are admission policy, not measured speed.
        reserve = float(stage.get("a4_generation_reserve_s_per_query", 300)) * len(indices)
        estimate = float(stage.get("a4_adaptation_estimate_s", 300))
        a0_complete = (initial_report.get("status") == "complete" and all(
            c.get("status") in MODEL_OUTCOMES for q in a0_rows for c in q["candidates"]))
        admitted = a0_complete and deadline - time.monotonic() >= estimate + reserve
        if admitted:
            adapted = ttt.adapt_task(session.model, session.tokenizer, tid, tasks[tid]["train"], initial,
                formatter=stage.get("formatter", "f1_compact"), updates=int(stage.get("updates", 4)),
                grad_accum=4, deadline=deadline-reserve, eos_id=session.tokenizer.eos_token_id,
                torch_module=session.torch, on_update=lambda update, task_id=tid: log_update(task_id, update))
        else:
            a0_budget_hit = any(c.get("status") == "budget_interrupted" for q in a0_rows for c in q["candidates"])
            adapted = {"task_id": tid, "status": "budget_interrupted" if a0_complete or a0_budget_hit else "infrastructure_error",
                       "reason": "a4_plus_generation_not_admitted" if a0_complete else "a0_pair_incomplete",
                       "requested_updates": 4, "completed_updates": 0, "applied_updates": 0,
                       "fallback_to_a0": True, "adapter_sha256": initial_report.get("adapter_sha256"),
                       "elapsed_s": 0.0, "train_s": 0.0}
        adapted["admission"] = {"admitted": admitted, "adaptation_estimate_s": estimate,
                                "generation_reserve_s": reserve, "adaptation_deadline": deadline-reserve}
        identity = {"kind": "ttt", "task_id": tid, "base_adapter": source,
                    "applied_updates": adapted.get("applied_updates"),
                    "adapter_sha256": adapted.get("adapter_sha256")}
        runtime.mark_adapter_state(session, identity)
        rows = []
        for qi in indices:
            if adapted.get("fallback_to_a0"):
                # Exact restoration gives the already evaluated A0 state.
                # Preserve source identity and count zero new generation cost.
                prior = next(q for q in a0_rows if q["query_index"] == qi)
                candidates = [{**c, "cache_hit": True, "physical_elapsed_s": 0.0,
                               "reused_from_a0": True} for c in prior["candidates"]]
                rows.append({"task_id": tid, "query_index": qi, "candidates": candidates,
                             "picked_grids": pick_two(candidates), "reused_from_a0": True})
                continue
            candidates = []
            for view in ("original", "transpose"):
                candidates.append(runtime.generate_grid(session, tid, qi, tasks[tid]["train"],
                    tasks[tid]["test"][qi]["input"], formatter=stage.get("formatter", "f1_compact"),
                    view=view, do_sample=False, max_new_tokens=int(stage.get("max_new_tokens", 960)),
                    deadline=deadline, run_tag=stage["name"] + "/A4"))
            rows.append({"task_id": tid, "query_index": qi, "candidates": candidates,
                         "picked_grids": pick_two(candidates)})
        result.append({"task_id": tid, "adaptation": adapted, "A0": a0_rows, "A4": rows,
                       "a0_state": initial_report})
        atomic_json(out_dir / (stage["name"] + ".json"), {"stage": stage, "tasks": result})
    return {"stage": stage["name"], "type": "ttt", "tasks": result,
            "planned_queries": len(pairs), "query_plan": [
                {"task_id": tid, "query_index": qi,
                 "source": stage.get("source_by_task", {}).get(tid, "unconfirmed")}
                for tid, qi in pairs]}


def run_program_stage(session, stage, tasks, deadline, out_dir, remaining_later_calls=0):
    """p0_only: fresh P0 per task, global admission, no history carryover."""
    import solve_program as program
    result = []
    pairs = solver.parse_query_file(stage["query_file"])
    tids = list(dict.fromkeys(tid for tid, _ in pairs))
    policy = stage.get("policy", "legacy")
    merged_limits = dict(P0_FROZEN_LIMITS) if policy == "p0_only" else {}
    merged_limits.update(stage.get("limits", {}))
    limits = program.ProgramLimits(**merged_limits)
    # Pre-load validation already ran in read_config; re-validate here so a
    # directly-invoked worker still fails fast.
    if policy == "p0_only":
        _validate_p0_limits(merged_limits, stage.get("name"))
    else:
        try:
            limits.validate()
        except ValueError as exc:
            raise ValueError("stage %s limits rejected: %s" % (stage.get("name"), exc))
    global_deadline = float(session.deadline) - float(session.save_window_s)
    mode = _stage_mode(stage)
    def persist(item):
        with open(out_dir / (stage["name"] + "-events.jsonl"), "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"stage": stage["name"], "recorded_unix_s": time.time(),
                                     **item}, ensure_ascii=False, default=str) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
    for task_index, tid in enumerate(tids):
        remaining_calls = (len(tids) - task_index) + int(remaining_later_calls or 0)
        remaining_s = global_deadline - time.monotonic()
        # Global admission (§7.3): never use the local 760 stage deadline here.
        if policy == "p0_only" and remaining_s < remaining_calls * P0_CALL_SECONDS:
            for remaining in tids[task_index:]:
                result.append({"task_id": remaining, "status": "not_run",
                               "reason": "global_admission_insufficient",
                               "remaining_s": round(remaining_s, 3),
                               "required_s": remaining_calls * P0_CALL_SECONDS,
                               "remaining_calls": remaining_calls,
                               "policy": policy, "mode": mode})
                persist({"task_id": remaining, "program_arm": "p0",
                         "proposal": result[-1]})
            break
        if time.monotonic() >= deadline:
            result.append({"task_id": tid, "status": "budget_interrupted"})
            continue
        runtime.event("program_p0_start", stage=stage["name"], task_id=tid, mode=mode,
                      policy=policy, remaining_calls=remaining_calls,
                      remaining_s=round(global_deadline - time.monotonic(), 2))
        item = _call_program_task(program, session, tid, tasks[tid], limits,
                                  deadline=deadline, task_index=task_index,
                                  journal=persist, stage={**stage, "mode": mode})
        item.setdefault("policy", policy)
        item.setdefault("mode", mode)
        result.append(item)
        atomic_json(out_dir / (stage["name"] + ".json"), {"stage": stage, "tasks": result})
        runtime.event("program_p0_done", stage=stage["name"], task_id=tid, mode=mode,
                      policy=policy, status=item.get("status"))
        if policy != "p0_only" and task_index == 1 and int(stage.get("pilot_tasks", 2)) == 2:
            executable = [r.get("proposals", {}).get("p0", {}).get("status") == "program_ready"
                          and any(d.get("status") == "ok" for d in r.get("proposals", {}).get("p0", {}).get("demo_results", []))
                          for r in result[:2]]
            if not any(executable):
                result.extend({"task_id": remaining, "status": "not_run", "reason": "p2_no_executable_initial_program"}
                              for remaining in tids[2:])
                break
    return {"stage": stage["name"], "type": "program", "policy": policy, "mode": mode,
            "tasks": result,
            "planned_tasks": len(tids), "p0_only": policy == "p0_only",
            "admission": {"remaining_later_calls": int(remaining_later_calls or 0),
                          "call_seconds": P0_CALL_SECONDS,
                          "global_deadline": global_deadline},
            "planned_queries": sum(len(tasks[tid]["test"]) for tid in tids)}


MODEL_OUTCOMES = {"complete", "invalid_output"}


def _ttt_query_rows(stage, solutions=None):
    """Mechanism A4 is scored only after all four updates and both views.

    Explicit ineligible->A0 fallback may have a complete *policy* result. It is
    kept separately and never represented as a successfully adapted A4 arm.
    """
    tasks = {item["task_id"]: item for item in stage.get("tasks", [])}
    plan = stage.get("query_plan")
    if plan is None:
        # Compatibility for manually assembled records, without reading any
        # external manifest or guessing a skipped multi-query task's size.
        plan = []
        for item in tasks.values():
            indices = set(item.get("query_indices", []))
            indices.update(q["query_index"] for key in ("A0", "A4") for q in item.get(key, []))
            plan.extend({"task_id": item["task_id"], "query_index": qi, "source": "unconfirmed"}
                        for qi in sorted(indices))
    rows = []
    for job in plan:
        tid, qi = job["task_id"], int(job["query_index"])
        task = tasks.get(tid, {})
        adaptation = task.get("adaptation", {})
        initial = task.get("a0_state", {})
        a0_state_ok = initial.get("status") == "complete" and initial.get("applied_updates") == 0
        a4_state_ok = (adaptation.get("status") == "complete"
                       and all(adaptation.get(k) == 4 for k in ("requested_updates", "completed_updates", "applied_updates"))
                       and not adaptation.get("fallback_to_a0"))
        fallback = adaptation.get("status") == "ineligible" and adaptation.get("fallback_to_a0") and adaptation.get("applied_updates") == 0
        gold = solver.gold_for(solutions, tid, qi) if solutions is not None else None
        arms = {}
        any_model_result = False
        for label, state_ok in (("A0", a0_state_ok), ("A4", a4_state_ok)):
            query = next((q for q in task.get(label, []) if int(q["query_index"]) == qi), {})
            candidates = query.get("candidates", [])
            statuses = [c.get("status") for c in candidates]
            views = [c.get("view") for c in candidates]
            generated = len(candidates) == 2 and sorted(views) == ["original", "transpose"] and all(s in MODEL_OUTCOMES for s in statuses)
            any_model_result = any_model_result or any(s in MODEL_OUTCOMES for s in statuses)
            observed = any(solver.grids_equal(g, gold) for g in pick_two(candidates)) if gold is not None else None
            complete = bool(state_ok and generated)
            arms[label] = {"state_is_requested": bool(state_ok), "generation_complete": generated,
                           "evaluation_complete": complete, "candidate_statuses": statuses,
                           "picked_em": observed if complete else None,
                           "observed_pick_em": observed,
                           "candidate_em": [(c.get("status") == "complete" and solver.grids_equal(c.get("grid"), gold))
                                            if gold is not None and c.get("status") in MODEL_OUTCOMES else None
                                            for c in candidates]}
        paired = arms["A0"]["evaluation_complete"] and arms["A4"]["evaluation_complete"]
        pair_completion = "complete" if paired else ("partial" if any_model_result else "none")
        a0_em, a4_em = arms["A0"]["picked_em"], arms["A4"]["picked_em"]
        pair_category = None
        if paired and gold is not None:
            pair_category = ("both_correct" if a0_em else "gained") if a4_em else ("lost" if a0_em else "both_wrong")
        policy_complete = bool(arms["A4"]["evaluation_complete"] or (fallback and arms["A4"]["generation_complete"]))
        rows.append({"task_id": tid, "query_index": qi, "source": job.get("source", "unconfirmed"),
                     "gold_available": gold is not None, "A0": arms["A0"], "A4": arms["A4"],
                     "adaptation_status": adaptation.get("status", task.get("status", "not_run")),
                     "applied_updates": adaptation.get("applied_updates"),
                     "fallback_to_a0": bool(adaptation.get("fallback_to_a0")),
                     "pair_completion": pair_completion, "pair_category": pair_category,
                     "fallback_policy_complete": policy_complete,
                     "fallback_policy_em": arms["A4"]["observed_pick_em"] if policy_complete else None})
    return rows


def score_ttt_results(stage_results, solutions_path):
    """Post-hoc A0/A4 paired EM, including all explicitly planned queries."""
    solutions = solver.load_solutions(solutions_path)
    results = []
    def aggregate(rows):
        categories = {key: sum(r["pair_category"] == key for r in rows)
                      for key in ("gained", "lost", "both_correct", "both_wrong")}
        return {"n_planned": len(rows), "n_with_gold": sum(r["gold_available"] for r in rows),
                "pair_completion": {key: sum(r["pair_completion"] == key for r in rows)
                                    for key in ("complete", "partial", "none")},
                "n_paired_scored": sum(r["pair_category"] is not None for r in rows),
                "pair_categories": categories, "net_gained": categories["gained"] - categories["lost"],
                "arms": {label: {"n_scored": sum(r[label]["picked_em"] is not None for r in rows),
                                  "exact_match": sum(r[label]["picked_em"] is True for r in rows)}
                         for label in ("A0", "A4")},
                "n_fallback": sum(r["fallback_to_a0"] for r in rows),
                "fallback_policy_n_scored": sum(r["fallback_policy_em"] is not None for r in rows),
                "fallback_policy_exact_match": sum(r["fallback_policy_em"] is True for r in rows)}
    for stage in stage_results:
        if stage.get("type") != "ttt":
            continue
        rows = _ttt_query_rows(stage, solutions)
        results.append({"stage": stage["stage"], "type": "ttt", "queries": rows,
                        **aggregate(rows), "by_source": {source: aggregate([r for r in rows if r["source"] == source])
                            for source in sorted(set(r["source"] for r in rows))}})
    return results


def stage_work_complete(stage):
    """Completion means all planned work, never the absence of grid stages."""
    if stage.get("type") == "grid":
        queries = stage.get("queries", [])
        candidates = [c for q in queries for c in q.get("candidates", [])]
        return (bool(queries) and len(queries) == stage.get("planned_queries", len(queries))
                and len(candidates) == stage.get("planned_candidates", len(candidates))
                and bool(candidates) and all(c.get("status") in MODEL_OUTCOMES for c in candidates))
    if stage.get("type") == "ttt":
        rows = _ttt_query_rows(stage)
        return bool(rows) and len(rows) == stage.get("planned_queries", len(rows)) and all(r["pair_completion"] == "complete" for r in rows)
    if stage.get("type") == "program":
        tasks = stage.get("tasks", [])
        if not tasks or len(tasks) != stage.get("planned_tasks", len(tasks)):
            return False
        execution_outcomes = {"ok", "execution_error", "execution_timeout", "output_limit", "program_rejected",
                              "incomplete_thinking", "incomplete_final_program"}
        # 012 p0_only: completeness looks only at planned p0 proposals.
        if stage.get("policy") == "p0_only" or stage.get("p0_only"):
            planned_arms = ("p0",)
        else:
            planned_arms = ("p0", "restart", "repair")
        for task in tasks:
            if task.get("status") not in ("complete", "completed"):
                return False
            proposals = task.get("proposals", {})
            for arm in planned_arms:
                proposal = proposals.get(arm, {})
                generation_status = proposal.get("generation", {}).get("status")
                cases = proposal.get("query_results", [])
                if (arm == "repair" and proposal.get("status") == "ineligible_no_complete_p0"
                        and proposals.get("p0", {}).get("generation", {}).get("status") in MODEL_OUTCOMES):
                    if not cases or any(c.get("status") != "ineligible_no_complete_p0" for c in cases):
                        return False
                    continue
                if generation_status not in MODEL_OUTCOMES or not cases or any(c.get("status") not in execution_outcomes for c in cases):
                    return False
        return True
    return False


def score_grid_results(stage_results, solutions_path):
    """Read gold only after generation stages finish; never passed to runtime."""
    solutions = solver.load_solutions(solutions_path)
    scored = []
    for stage in stage_results:
        if stage.get("type") != "grid":
            continue
        queries = []
        for query in stage["queries"]:
            gold = solver.gold_for(solutions, query["task_id"], query["query_index"])
            outcomes = ("complete", "invalid_output")
            evaluated = all(c.get("status") in outcomes for c in query["candidates"])
            matches = [(c.get("status") == "complete" and solver.grids_equal(c.get("grid"), gold))
                       if c.get("status") in outcomes else None
                       for c in query["candidates"]] if gold is not None else None
            partial = any(solver.grids_equal(g, gold) for g in query["picked_grids"]) if gold is not None else None
            picked = partial if evaluated else None
            pair_results = {}
            if matches is not None:
                original = next((i for i, c in enumerate(query["candidates"])
                    if c.get("view") == "original" and c.get("kind") == "greedy"), None)
                for label, predicate in (("O+S", lambda c: c.get("view") == "original" and c.get("kind") == "sample"),
                                         ("O+T", lambda c: c.get("view") == "transpose")):
                    other = next((i for i, c in enumerate(query["candidates"]) if predicate(c)), None)
                    if original is not None and other is not None:
                        values = [matches[original], matches[other]]
                        pair_results[label] = any(values) if all(v is not None for v in values) else None
            queries.append({"task_id": query["task_id"], "query_index": query["query_index"],
                            "candidate_em": matches, "picked_em": picked,
                            "evaluation_complete": evaluated, "partial_pick_em": partial,
                            "pair_em": pair_results, "gold_available": gold is not None,
                            "candidate_statuses": [c.get("status") for c in query["candidates"]]})
        scored.append({"stage": stage["stage"], "queries": queries,
                       "n_planned": len(queries), "n_with_gold": sum(x["gold_available"] for x in queries),
                       "n_incomplete": sum(not x["evaluation_complete"] for x in queries),
                       "n_scored": sum(x["picked_em"] is not None for x in queries),
                       "picked_exact_match": sum(x["picked_em"] is True for x in queries)})
    return scored


def worker(config, absolute_deadline):
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    start = absolute_deadline - float(config.get("time_budget_s", 7200))
    # 012 §7.3: notebook start is never reset per stage; first stage includes load.
    summary = {"status": "running", "stages": [], "started_unix_s": time.time(),
               "notebook_start_monotonic": start, "absolute_deadline": absolute_deadline,
               "time_budget_s": config.get("time_budget_s", 7200),
               "save_window_s": float(config.get("runtime", {}).get("save_window_s", 300))}
    # Suffix P0 call counts for global admission (remaining_calls × 760).
    p0_counts = []
    for stage in config["stages"]:
        if stage.get("type") == "program" and stage.get("policy") == "p0_only":
            tids = list(dict.fromkeys(t for t, _ in solver.parse_query_file(stage["query_file"])))
            p0_counts.append(len(tids))
        else:
            p0_counts.append(0)
    suffix = [0] * (len(p0_counts) + 1)
    for i in range(len(p0_counts) - 1, -1, -1):
        suffix[i] = suffix[i + 1] + p0_counts[i]
    try:
        tasks = load_inputs(config)
        session = runtime.load_session(config["model_dir"], deadline=absolute_deadline,
            journal_path=str(out / "candidates.jsonl"), runtime_config=config.get("runtime"))
        atomic_json(out / "session_identity.json", {"base": session.base_identity,
                    "runtime": session.runtime_identity})
        summary["save_window_s"] = float(session.save_window_s)
        atomic_json(out / "SUMMARY.json", summary)
        for position, stage in enumerate(config["stages"]):
            stage_start = start if position == 0 else time.monotonic()
            deadline = min(absolute_deadline - session.save_window_s,
                           stage_start + float(stage.get("time_budget_s", config.get("time_budget_s", 7200))))
            runtime.event("stage_start", name=stage["name"], deadline=deadline,
                          notebook_start_monotonic=start)
            kind = stage.get("type", "grid")
            function = {"grid": run_grid_stage, "ttt": run_ttt_stage, "program": run_program_stage}[kind]
            if kind == "program":
                result = function(session, stage, tasks, deadline, out,
                                  remaining_later_calls=suffix[position + 1])
            else:
                result = function(session, stage, tasks, deadline, out)
            atomic_json(out / (stage["name"] + ".json"), result)
            summary["stages"].append(result)
            atomic_json(out / "SUMMARY.json", summary)
            runtime.event("stage_done", name=stage["name"], elapsed_s=time.monotonic()-stage_start)
        if config.get("solutions"):
            atomic_json(out / "scoring.json", score_grid_results(summary["stages"], config["solutions"])
                        + score_ttt_results(summary["stages"], config["solutions"]))
        summary["stage_completion"] = {stage["stage"]: stage_work_complete(stage) for stage in summary["stages"]}
        summary["work_complete"] = len(summary["stages"]) == len(config["stages"]) and all(summary["stage_completion"].values())
        summary["status"] = "complete" if summary["work_complete"] else "partial"
    except Exception as exc:
        summary["status"] = "infrastructure_error"
        summary["error"] = "%s: %s" % (type(exc).__name__, str(exc))
        traceback.print_exc()
    summary["elapsed_s"] = time.monotonic() - start
    atomic_json(out / "SUMMARY.json", summary)
    runtime.event("worker_done", status=summary["status"], elapsed_s=summary["elapsed_s"])
    return 0 if summary["status"] in ("complete", "partial") else 1


def supervise(config, config_path):
    """Hard process limits include imports/loading; one worker, no retries."""
    out = Path(config["out_dir"])
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "requested_config.json", config)
    process_start = time.monotonic()
    start = float(config.get("notebook_start_monotonic", process_start - float(config.get("budget_elapsed_s", 0))))
    if start > process_start:
        raise ValueError("notebook_start_monotonic cannot be in the future")
    deadline = start + float(config.get("time_budget_s", 7200))
    if process_start >= deadline:
        atomic_json(out / "supervisor.json", {"status": "budget_interrupted",
                    "reason": "notebook_budget_expired_before_worker", "elapsed_s": process_start-start,
                    "notebook_start_monotonic": start})
        return 2
    load_limit = float(config.get("load_limit_s", 1200))
    command = [sys.executable, "-u", os.path.abspath(__file__), "--config", os.path.abspath(config_path),
               "--worker", "--absolute-deadline", str(deadline)]
    env = dict(os.environ, PYTHONUNBUFFERED="1")
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               env=env, start_new_session=True)
    loaded = False
    reason = None
    buffer = b""
    last_heartbeat = process_start
    log = open(out / "worker.log", "ab", buffering=0)
    try:
        while True:
            now = time.monotonic()
            if now >= deadline:
                reason = "session_deadline"
                break
            if not loaded and now - process_start >= load_limit:
                reason = "load_deadline"
                break
            if now - last_heartbeat >= 30:
                print("SUPERVISOR_HEARTBEAT " + json.dumps({"elapsed_s": round(now-start, 1),
                      "loaded": loaded, "remaining_s": round(deadline-now, 1)}), flush=True)
                last_heartbeat = now
            ready, _, _ = select.select([process.stdout], [], [], .5)
            if ready:
                block = os.read(process.stdout.fileno(), 65536)
                if block:
                    log.write(block)
                    sys.stdout.write(block.decode("utf-8", errors="replace"))
                    sys.stdout.flush()
                    buffer += block
                    while b"\n" in buffer:
                        line, buffer = buffer.split(b"\n", 1)
                        if line.startswith(b"GRID_RUNTIME_EVENT "):
                            try:
                                ev = json.loads(line[len(b"GRID_RUNTIME_EVENT "):])
                                if ev.get("event") == "load_done":
                                    loaded = True
                            except ValueError:
                                pass
                elif process.poll() is not None:
                    break
            elif process.poll() is not None:
                break
        if reason is not None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.wait(timeout=5)
        else:
            process.wait(timeout=5)
    finally:
        log.close()
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()
    report = {"status": "budget_interrupted" if reason else ("complete" if process.returncode == 0 else "infrastructure_error"),
              "reason": reason, "worker_exit_code": process.returncode, "load_completed": loaded,
              "elapsed_s": time.monotonic()-start, "time_budget_s": config.get("time_budget_s", 7200),
              "load_limit_s": load_limit, "no_retry": True,
              "notebook_start_monotonic": start, "absolute_deadline": deadline}
    atomic_json(out / "supervisor.json", report)
    print("BUDGETED_RUN_SUMMARY=" + json.dumps(report), flush=True)
    return 2 if reason else process.returncode


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Explicit JSON stages and local paths.")
    parser.add_argument("--dry-run", action="store_true", help="Validate input jobs without importing torch/loading weights.")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--absolute-deadline", type=float, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    config = read_config(args.config)
    tasks = load_inputs(config)
    if args.dry_run:
        plan = _program_call_plan(config)
        budget = float(config.get("time_budget_s", 7200))
        save_window = float(config.get("runtime", {}).get("save_window_s", 300))
        effective = budget - save_window
        required = len(plan) * P0_CALL_SECONDS if plan else 0
        # Synthetic insufficient case (§7.3): 100s left for a 760s call → not_run.
        demo_remaining, demo_required = 100.0, P0_CALL_SECONDS
        # Identity variation (§7.4/§9.6): same prompt, D vs L must hash differently.
        base_job = {"task_id": "db118e2a", "query_index": 0, "program_arm": "p0",
                    "kind": "program", "backend": "torch_fallback"}
        base_cfg = {"enable_thinking": False, "reasoning_effort": "low",
                    "preserve_thinking": True, "max_new_tokens": 3072,
                    "max_time_s": 720, "max_prompt_tokens": 8192}
        cfg_d = dict(base_cfg, mode="D")
        cfg_l = dict(base_cfg, mode="L", enable_thinking=True)
        job_d = dict(base_job, mode="D", enable_thinking=False)
        job_l = dict(base_job, mode="L", enable_thinking=True)
        key_d = runtime.canonical_hash({"generation": cfg_d, "job": job_d})
        key_l = runtime.canonical_hash({"generation": cfg_l, "job": job_l})
        print(json.dumps({"status": "cpu_plan_complete", "model_loaded": False,
            "time_budget_s": budget,
            "load_limit_s": config.get("load_limit_s", 1200),
            "save_window_s": save_window,
            "notebook_start_monotonic_saved": True,
            "stages": [{"name": s["name"], "type": s.get("type", "grid"),
                        "queries": len(solver.parse_query_file(s["query_file"])),
                        "candidates_per_query": len(s.get("candidates", [{}]))}
                       for s in config["stages"]], "input_tasks": len(tasks),
            "program_plan": plan, "program_plan_calls": len(plan),
            "program_admission": {"required_s": required, "effective_window_s": effective,
                "admitted_zero_load": effective >= required if plan else True,
                "rule": "global(session.deadline-save_window-now) >= remaining_calls*760"},
            "admission_demo_insufficient": {"remaining_s": demo_remaining,
                "required_s": demo_required,
                "decision": "not_run" if demo_remaining < demo_required else "admitted",
                "reason": "global_admission_insufficient"},
            "identity_demo": {"key_D": key_d[:16], "key_L": key_l[:16],
                "differs": key_d != key_l,
                "rule": "mode/thinking/token/time change identity"}}, ensure_ascii=False))
        return 0
    if args.worker:
        if args.absolute_deadline is None:
            parser.error("internal worker requires --absolute-deadline")
        return worker(config, args.absolute_deadline)
    return supervise(config, args.config)


if __name__ == "__main__":
    raise SystemExit(main())
