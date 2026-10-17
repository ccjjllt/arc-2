# CELL 0 (markdown)
# Qwen3.8-27B free-form ARC runner

**Generated artifact:** edit `build_kernel.py`, then regenerate; do not hand-edit this notebook.

**Recommended layout:** keep this generated operator guide plus one atomic runtime cell. A single **Run All** preserves the required ordering—resolve mode, publish a valid floor, prove the exact serving/tool stack, run universal xhigh, spend only post-coverage time on medium, then finalize receipts—without stale state from selectively rerun setup cells.

## Runtime modes

| Mode | Trigger | Challenge file | Loads model | Output policy |
|---|---|---|---:|---|
| Saved-version gate (floor only) | `default; no opt-in environment flags` | `arc-agi_evaluation_challenges.json` | no | never manually upload |
| Explicit public diagnostic | `ARC_FREEFORM_DIAGNOSTIC=1` | `arc-agi_evaluation_challenges.json` | yes | never manually upload |
| Explicit saturation probe | `ARC_FREEFORM_DIAGNOSTIC=1 and ARC_FREEFORM_PROBE=1` | `arc-agi_evaluation_challenges.json` | yes | never manually upload |
| Competition rerun (hidden test) | `KAGGLE_IS_COMPETITION_RERUN=true` | `arc-agi_test_challenges.json` | yes | competition output |

The default saved-version path writes a complete distinct pass@2 floor and exits without loading vLLM or the Python tool. Kaggle alone should activate the competition rerun. Boolean flags accept only standard true/false spellings; ambiguous values fail loudly. A saved-version or diagnostic `submission.json` must never be manually uploaded.

## Kaggle operator checklist

1. Attach the competition data, the pinned model dataset matching `qwen3-8-27b-fp8-repacked-v1`, and the offline wheelhouse matching `arc3-vllm-h100-wheelhouse-v3`. Keep internet off and select **4 x L4**.
2. Do not set diagnostic flags for the submission artifact. Save a version normally. The saved-version log must end with `NEXT ACTION: SAVED GATE VERIFIED` and `saved_version_gate_pass`; then download its output and run the saved-gate verifier command below.
3. Open the authenticated notebook page and choose **Submit to Competition** for the exact verified saved version. Never run `kaggle competitions submit ... --kernel ...`: that path can attach the floor without launching the hidden rerun. Kaggle supplies `KAGGLE_IS_COMPETITION_RERUN=true`; do not imitate that flag in a public diagnostic.
4. Early in the rerun, verify `competition_rerun`, the exact 240-task/259-query input contract, 4 L4 GPUs, vLLM `0.19.0`, served model `Qwen/Qwen3.8-27B-FP8`, active `qwen3_coder` flags, and a successful full-cycle Python-tool probe. A mismatch refuses model serving and leaves the valid floor.
5. Download the output artifacts. Upload only when `freeform_acceptance.json` says `competition_full_run_pass` and the offline verifier exits zero.

## Competition budget contract

259 queries across 240 tasks at concurrency 24 require 11 xhigh waves. At the 60-minute episode cap that is 11.0 hours, leaving 54 minutes for all pre-generation work plus a separate 3-minute finalization reserve. After real startup, the primary episode cap is reduced only if necessary to preserve universal first-pass coverage; medium remains post-coverage and opportunistic.

## Observability and local checks

The run writes `submission.json` first and `freeform_status.json` heartbeats. Model runs append `freeform_samples.jsonl`; every completed run writes a hashed receipt plus `freeform_acceptance.json`. On restart, old artifacts move under `freeform_previous_runs/<timestamp>/` as evidence and are never reused as predictions.

```bash
python3 notebook/build_kernel.py --check
python3 notebook/build_kernel.py --describe
python3 verify_run_artifacts.py --root /path/to/saved-version-output --require-saved-gate
python3 verify_run_artifacts.py --root /path/to/downloaded/kaggle-output --require-competition-pass
```

## Recovery rules

If the session dies, preserve/download the current submission, status, JSONL telemetry, and server log before restarting. The partial submission remains structurally valid but does not prove a complete competition run. Restart with **Run All**; never resume selected cells or merge old hidden-test predictions.

# CELL 1 (code)

import atexit, datetime, glob, hashlib, importlib.metadata, importlib.util, json, math, os, re, shutil, signal, socket, subprocess, sys, threading, time
from pathlib import Path
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import urllib.request, urllib.error

ENV_TRUE_VALUES = frozenset(['1', 'on', 'true', 'yes'])
ENV_FALSE_VALUES = frozenset(['', '0', 'false', 'no', 'off'])
RUN_MODE_DEFINITIONS = {'saved_version_gate': {'label': 'Saved-version gate (floor only)', 'trigger': 'default; no opt-in environment flags', 'challenge_name': 'arc-agi_evaluation_challenges.json', 'run_model': False, 'diagnostic_model_run': False, 'probe': False, 'task_limit': 0, 'hard_wall_seconds': 300, 'finalization_reserve_seconds': 60, 'direct_file_submittable': False, 'submission_scope': 'public evaluation saved-version gate'}, 'diagnostic': {'label': 'Explicit public diagnostic', 'trigger': 'ARC_FREEFORM_DIAGNOSTIC=1', 'challenge_name': 'arc-agi_evaluation_challenges.json', 'run_model': True, 'diagnostic_model_run': True, 'probe': False, 'task_limit': 24, 'hard_wall_seconds': 5040, 'finalization_reserve_seconds': 60, 'direct_file_submittable': False, 'submission_scope': 'public evaluation diagnostic'}, 'diagnostic_probe': {'label': 'Explicit saturation probe', 'trigger': 'ARC_FREEFORM_DIAGNOSTIC=1 and ARC_FREEFORM_PROBE=1', 'challenge_name': 'arc-agi_evaluation_challenges.json', 'run_model': True, 'diagnostic_model_run': True, 'probe': True, 'task_limit': 24, 'hard_wall_seconds': 3600, 'finalization_reserve_seconds': 60, 'direct_file_submittable': False, 'submission_scope': 'public evaluation saturation probe'}, 'competition_rerun': {'label': 'Competition rerun (hidden test)', 'trigger': 'KAGGLE_IS_COMPETITION_RERUN=true', 'challenge_name': 'arc-agi_test_challenges.json', 'run_model': True, 'diagnostic_model_run': False, 'probe': False, 'task_limit': None, 'hard_wall_seconds': 43020, 'finalization_reserve_seconds': 180, 'direct_file_submittable': True, 'submission_scope': 'hidden competition test'}}
EXPECTED_COMPETITION_TASKS = 240
EXPECTED_COMPETITION_QUERIES = 259
RUNTIME_CONFIG = {'schema': 'arc-qwen38-runtime-config-v1', 'model': {'served_name': 'Qwen/Qwen3.8-27B-FP8', 'source_hint': 'qwen3-8-27b-fp8-repacked-v1'}, 'vllm': {'version': '0.19.0', 'wheelhouse_hint': 'arc3-vllm-h100-wheelhouse-v3', 'reasoning_parser': 'qwen3', 'tool_parser': 'qwen3_coder'}, 'hardware': {'gpu_count': 4, 'gpu_name_contains': 'L4', 'tensor_parallel_size': 4}, 'server': {'port': 1234, 'context_len': 65536, 'gpu_memory_utilization': 0.9}, 'scheduler': {'concurrency': 24, 'samples_per_query': 2, 'python_tool_max_concurrency': 4, 'progress_interval_s': 60}, 'sampling': {'temperature': 0.6, 'top_p': 0.95, 'top_k': 20, 'max_tokens': 57344, 'reasoning_effort': 'xhigh', 'enable_thinking': True}, 'agent': {'tools_enabled': True, 'max_turns': 10}, 'pass_policies': {'xhigh': {'reasoning_effort': 'xhigh', 'turn_max_tokens': 40960, 'direct_max_tokens': 57344, 'episode_budget_s': 3600, 'initial_predicted_s': 3600}, 'medium': {'reasoning_effort': 'medium', 'turn_max_tokens': 20480, 'direct_max_tokens': 20480, 'episode_budget_s': 1800, 'initial_predicted_s': 1800}}}
RUNTIME_CONFIG_SHA256 = "eb2c70295d0ed1c14de8fc70c40095eaacac1163a5cbb0ec9eef50f385c53ad3"
def validate_runtime_config(config):
    """Fail fast on unsafe or internally inconsistent runtime tuning."""

    if not isinstance(config, dict):
        raise TypeError("runtime config must be a mapping")
    required_sections = {
        "schema",
        "model",
        "vllm",
        "hardware",
        "server",
        "scheduler",
        "sampling",
        "agent",
        "pass_policies",
    }
    missing = required_sections - set(config)
    if missing:
        raise ValueError(f"runtime config missing sections: {sorted(missing)!r}")
    unexpected = set(config) - required_sections
    if unexpected:
        raise ValueError(
            f"runtime config has unknown sections: {sorted(unexpected)!r}"
        )
    if config.get("schema") != "arc-qwen38-runtime-config-v1":
        raise ValueError("runtime config schema is not recognized")
    for section in required_sections - {"schema"}:
        if not isinstance(config.get(section), dict):
            raise ValueError(f"runtime config section {section!r} must be a mapping")

    section_keys = {
        "model": {"served_name", "source_hint"},
        "vllm": {
            "version",
            "wheelhouse_hint",
            "reasoning_parser",
            "tool_parser",
        },
        "hardware": {
            "gpu_count",
            "gpu_name_contains",
            "tensor_parallel_size",
        },
        "server": {"port", "context_len", "gpu_memory_utilization"},
        "scheduler": {
            "concurrency",
            "samples_per_query",
            "python_tool_max_concurrency",
            "progress_interval_s",
        },
        "sampling": {
            "temperature",
            "top_p",
            "top_k",
            "max_tokens",
            "reasoning_effort",
            "enable_thinking",
        },
        "agent": {"tools_enabled", "max_turns"},
        "pass_policies": {"xhigh", "medium"},
    }
    for section, expected_keys in section_keys.items():
        observed_keys = set(config[section])
        missing_keys = expected_keys - observed_keys
        unexpected_keys = observed_keys - expected_keys
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"runtime config section {section!r} has invalid keys; "
                f"missing={sorted(missing_keys)!r}, "
                f"unknown={sorted(unexpected_keys)!r}"
            )

    def positive_int(value, path):
        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
            raise ValueError(f"{path} must be a positive integer")
        return value

    for path, value in (
        ("hardware.gpu_count", config["hardware"].get("gpu_count")),
        (
            "hardware.tensor_parallel_size",
            config["hardware"].get("tensor_parallel_size"),
        ),
        ("server.port", config["server"].get("port")),
        ("server.context_len", config["server"].get("context_len")),
        ("scheduler.concurrency", config["scheduler"].get("concurrency")),
        (
            "scheduler.samples_per_query",
            config["scheduler"].get("samples_per_query"),
        ),
        (
            "scheduler.python_tool_max_concurrency",
            config["scheduler"].get("python_tool_max_concurrency"),
        ),
        (
            "scheduler.progress_interval_s",
            config["scheduler"].get("progress_interval_s"),
        ),
        ("agent.max_turns", config["agent"].get("max_turns")),
        ("sampling.max_tokens", config["sampling"].get("max_tokens")),
    ):
        positive_int(value, path)

    if config["scheduler"]["samples_per_query"] not in {1, 2}:
        raise ValueError("scheduler.samples_per_query must be 1 or 2 for pass@2")
    if config["server"]["port"] > 65535:
        raise ValueError("server.port must be at most 65535")
    if (
        config["hardware"]["gpu_count"]
        != config["hardware"]["tensor_parallel_size"]
    ):
        raise ValueError("gpu_count must equal tensor_parallel_size")
    if config["scheduler"]["python_tool_max_concurrency"] > config["scheduler"][
        "concurrency"
    ]:
        raise ValueError("python tool concurrency cannot exceed request concurrency")
    if not 0 < float(config["server"].get("gpu_memory_utilization", 0)) <= 1:
        raise ValueError("server.gpu_memory_utilization must be in (0, 1]")
    if not isinstance(config["agent"].get("tools_enabled"), bool):
        raise ValueError("agent.tools_enabled must be boolean")
    if not isinstance(config["sampling"].get("enable_thinking"), bool):
        raise ValueError("sampling.enable_thinking must be boolean")
    temperature = config["sampling"].get("temperature")
    if (
        isinstance(temperature, bool)
        or not isinstance(temperature, (int, float))
        or not 0 <= float(temperature) <= 2
    ):
        raise ValueError("sampling.temperature must be numeric in [0, 2]")
    top_p = config["sampling"].get("top_p")
    if (
        isinstance(top_p, bool)
        or not isinstance(top_p, (int, float))
        or not 0 < float(top_p) <= 1
    ):
        raise ValueError("sampling.top_p must be numeric in (0, 1]")
    positive_int(config["sampling"].get("top_k"), "sampling.top_k")
    if config["sampling"].get("reasoning_effort") != "xhigh":
        raise ValueError("sampling.reasoning_effort must be 'xhigh'")

    for path, value in (
        ("model.served_name", config["model"].get("served_name")),
        ("model.source_hint", config["model"].get("source_hint")),
        ("vllm.version", config["vllm"].get("version")),
        ("vllm.wheelhouse_hint", config["vllm"].get("wheelhouse_hint")),
        ("vllm.reasoning_parser", config["vllm"].get("reasoning_parser")),
        ("vllm.tool_parser", config["vllm"].get("tool_parser")),
        (
            "hardware.gpu_name_contains",
            config["hardware"].get("gpu_name_contains"),
        ),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{path} must be a non-empty string")

    policies = config["pass_policies"]
    if set(policies) != {"xhigh", "medium"}:
        raise ValueError("pass_policies must contain exactly xhigh and medium")
    policy_keys = {
        "reasoning_effort",
        "turn_max_tokens",
        "direct_max_tokens",
        "episode_budget_s",
        "initial_predicted_s",
    }
    context_len = config["server"]["context_len"]
    for name, policy in policies.items():
        if not isinstance(policy, dict):
            raise ValueError(f"pass_policies.{name} must be a mapping")
        observed_keys = set(policy)
        missing_keys = policy_keys - observed_keys
        unexpected_keys = observed_keys - policy_keys
        if missing_keys or unexpected_keys:
            raise ValueError(
                f"pass_policies.{name} has invalid keys; "
                f"missing={sorted(missing_keys)!r}, "
                f"unknown={sorted(unexpected_keys)!r}"
            )
        if policy.get("reasoning_effort") != name:
            raise ValueError(
                f"pass_policies.{name}.reasoning_effort must equal {name!r}"
            )
        for field in (
            "turn_max_tokens",
            "direct_max_tokens",
            "episode_budget_s",
            "initial_predicted_s",
        ):
            positive_int(policy.get(field), f"pass_policies.{name}.{field}")
        if policy["turn_max_tokens"] > context_len:
            raise ValueError(f"pass_policies.{name}.turn_max_tokens exceeds context")
        if policy["direct_max_tokens"] > context_len:
            raise ValueError(f"pass_policies.{name}.direct_max_tokens exceeds context")
        if policy["initial_predicted_s"] > policy["episode_budget_s"]:
            raise ValueError(
                f"pass_policies.{name}.initial_predicted_s exceeds episode budget"
            )
    if policies["medium"]["episode_budget_s"] > policies["xhigh"][
        "episode_budget_s"
    ]:
        raise ValueError("medium episode budget must not exceed xhigh")
    if config["sampling"]["max_tokens"] != policies["xhigh"]["direct_max_tokens"]:
        raise ValueError("sampling.max_tokens must match xhigh direct_max_tokens")

    # This also rejects non-serializable accidental values before generation.
    json.dumps(config, sort_keys=True, separators=(",", ":"))
    return config

validate_runtime_config(RUNTIME_CONFIG)
if hashlib.sha256(
    json.dumps(RUNTIME_CONFIG, sort_keys=True, separators=(",", ":")).encode()
).hexdigest() != RUNTIME_CONFIG_SHA256:
    raise RuntimeError("generated runtime config digest mismatch")
def parse_env_flag(raw, *, name):
    """Parse a notebook environment flag without Python string truthiness."""

    if raw is None:
        return False
    normalized = str(raw).strip().lower()
    if normalized in ENV_TRUE_VALUES:
        return True
    if normalized in ENV_FALSE_VALUES:
        return False
    raise RuntimeError(
        f"{name} has ambiguous boolean value {raw!r}; expected one of "
        f"{sorted(ENV_TRUE_VALUES | ENV_FALSE_VALUES)!r}"
    )

def resolve_run_mode(environ):
    """Resolve all runtime branching in one CPU-only, directly testable helper."""

    warnings = []
    rerun = parse_env_flag(
        environ.get("KAGGLE_IS_COMPETITION_RERUN"),
        name="KAGGLE_IS_COMPETITION_RERUN",
    )
    diagnostic = parse_env_flag(
        environ.get("ARC_FREEFORM_DIAGNOSTIC"),
        name="ARC_FREEFORM_DIAGNOSTIC",
    )
    probe_requested = parse_env_flag(
        environ.get("ARC_FREEFORM_PROBE"),
        name="ARC_FREEFORM_PROBE",
    )

    if rerun:
        mode_id = "competition_rerun"
        if diagnostic or probe_requested:
            warnings.append("diagnostic/probe flags ignored during a competition rerun")
    elif diagnostic and probe_requested:
        mode_id = "diagnostic_probe"
    elif diagnostic:
        mode_id = "diagnostic"
    else:
        mode_id = "saved_version_gate"
        if probe_requested:
            warnings.append("ARC_FREEFORM_PROBE ignored without ARC_FREEFORM_DIAGNOSTIC")

    mode = dict(RUN_MODE_DEFINITIONS[mode_id])
    mode.update(
        {
            "id": mode_id,
            "competition_rerun": rerun,
            "warnings": warnings,
            "environment": {
                "KAGGLE_IS_COMPETITION_RERUN": environ.get(
                    "KAGGLE_IS_COMPETITION_RERUN"
                ),
                "ARC_FREEFORM_DIAGNOSTIC": environ.get("ARC_FREEFORM_DIAGNOSTIC"),
                "ARC_FREEFORM_PROBE": environ.get("ARC_FREEFORM_PROBE"),
            },
        }
    )
    return mode

def stable_sample_seed(task_id, query_index, slot):
    """Derive a deterministic, task-specific 31-bit sampling seed."""

    payload = f"{task_id}:{query_index}:{slot}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:4], "big") & 0x7FFFFFFF

def select_distinct_fallback(model_grid, fallback_row):
    """Choose a prevalidated fallback distinct from a first model hypothesis."""

    for name in ("attempt_2", "attempt_1"):
        candidate = fallback_row[name]
        if candidate != model_grid:
            return candidate
    raise ValueError("fallback pair unexpectedly contains no grid distinct from model output")

def schedule_budget_contract(
    *, query_target, concurrency, episode_budget_s, hard_wall_s, reserve_s
):
    """Return the label-free worst-case wave math shown in logs and receipts."""

    waves = math.ceil(query_target / concurrency)
    generation_s = waves * episode_budget_s
    generation_window_s = hard_wall_s - reserve_s
    return {
        "query_target": query_target,
        "concurrency": concurrency,
        "waves": waves,
        "episode_budget_s": episode_budget_s,
        "worst_case_generation_s": generation_s,
        "generation_window_s": generation_window_s,
        "startup_and_schedule_slack_s": generation_window_s - generation_s,
        "finalization_reserve_s": reserve_s,
        "fits_contract": generation_s <= generation_window_s,
    }

def coverage_episode_budget_seconds(
    remaining_s,
    query_count,
    concurrency,
    configured_s,
    per_wave_overhead_s=30,
):
    """Cap primary episodes so every coverage wave fits the remaining wall."""

    if query_count <= 0:
        return int(configured_s)
    if concurrency <= 0:
        raise ValueError("concurrency must be positive")
    waves = (query_count + concurrency - 1) // concurrency
    usable = max(1, float(remaining_s) - waves * per_wave_overhead_s)
    return max(1, min(int(configured_s), int(usable // waves)))

def finite_work_size(work):
    """Return the exact size of finite scheduler work, else ``None``."""

    try:
        size = len(work)
    except TypeError:
        return None
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"work length must be a non-negative integer, got {size!r}")
    return size

def primary_sample_anomaly_reasons(row):
    """Classify anything that makes a primary sample non-clean evidence."""

    if not isinstance(row, dict):
        return ("malformed_sample",)
    reasons = []
    if row.get("grid") is not True:
        reasons.append("missing_or_invalid_grid")
    if row.get("error"):
        reasons.append("sample_error")
    if row.get("finish") != "stop":
        reasons.append("finish_not_stop")
    if row.get("forced_final") is True:
        reasons.append("forced_final")
    for field in ("tool_errors", "context_retries"):
        try:
            value = int(row.get(field) or 0)
        except (TypeError, ValueError):
            value = 1
        if value > 0:
            reasons.append(field)
    if row.get("recovery_attempted") is True:
        reasons.append("recovery_attempted")
    if row.get("recovery_error"):
        reasons.append("recovery_error")
    if row.get("publish_error"):
        reasons.append("publish_error")
    return tuple(reasons)

def competition_input_contract(task_count, query_count):
    """Return the exact hidden-input contract used by the schedule math."""

    observed_tasks = int(task_count)
    observed_queries = int(query_count)
    return {
        "expected_tasks": EXPECTED_COMPETITION_TASKS,
        "expected_queries": EXPECTED_COMPETITION_QUERIES,
        "observed_tasks": observed_tasks,
        "observed_queries": observed_queries,
        "task_count_matches": observed_tasks == EXPECTED_COMPETITION_TASKS,
        "query_count_matches": observed_queries == EXPECTED_COMPETITION_QUERIES,
        "ok": (
            observed_tasks == EXPECTED_COMPETITION_TASKS
            and observed_queries == EXPECTED_COMPETITION_QUERIES
        ),
    }

def evaluate_run_acceptance(snapshot):
    """Return a fail-closed, machine-readable outcome for one notebook run."""

    if not isinstance(snapshot, dict):
        raise TypeError("acceptance snapshot must be a mapping")

    mode_id = snapshot.get("mode_id")
    phase1 = snapshot.get("phase1") or {}
    phase2 = snapshot.get("phase2") or {}
    # These are verifier-owned competition facts, not receipt-tunable policy.
    expected_tasks = EXPECTED_COMPETITION_TASKS
    expected_queries = EXPECTED_COMPETITION_QUERIES
    checks = {
        "submission_valid": snapshot.get("submission_valid") is True,
        "pass2_distinct": int(snapshot.get("duplicate_attempt_pairs") or 0) == 0,
        "publication_clean": int(snapshot.get("publication_errors") or 0) == 0,
        "submission_hash_present": bool(snapshot.get("submission_sha256")),
        "challenge_hash_present": bool(snapshot.get("challenge_sha256")),
    }
    warnings = []
    if int(snapshot.get("status_write_errors") or 0) > 0:
        warnings.append("one or more status heartbeat writes failed")
    if int(snapshot.get("sample_event_write_errors") or 0) > 0:
        warnings.append("one or more sample telemetry writes failed")

    if mode_id == "saved_version_gate":
        checks.update(
            {
                "model_not_requested": snapshot.get("run_model") is False,
                "model_work_ineligible": snapshot.get("model_work_eligible")
                is False,
                "public_evaluation_scope": snapshot.get("challenge_name")
                == "arc-agi_evaluation_challenges.json",
                "manual_upload_forbidden": snapshot.get(
                    "direct_file_submittable"
                )
                is False,
            }
        )
        gate_pass = all(checks.values())
        return {
            "schema": "arc-qwen38-run-acceptance-v2",
            "mode_id": mode_id,
            "outcome": (
                "saved_version_gate_pass"
                if gate_pass
                else "saved_version_gate_failed"
            ),
            "accepted_for_manual_upload": False,
            "full_run_proven": False,
            "checks": checks,
            "failed_checks": [name for name, ok in checks.items() if not ok],
            "warnings": warnings,
        }

    if mode_id in {"diagnostic", "diagnostic_probe"}:
        checks.update(
            {
                "public_evaluation_scope": snapshot.get("challenge_name")
                == "arc-agi_evaluation_challenges.json",
                "manual_upload_forbidden": snapshot.get(
                    "direct_file_submittable"
                )
                is False,
                "model_requested": snapshot.get("run_model") is True,
                "server_served": snapshot.get("served") is True,
                "exact_vllm_version": snapshot.get("vllm_version_match") is True,
                "tool_flags_active": snapshot.get("tool_flags_active") is True,
                "live_python_tool_probe": snapshot.get("tool_probe_ok") is True,
            }
        )
        diagnostic_pass = all(checks.values())
        return {
            "schema": "arc-qwen38-run-acceptance-v2",
            "mode_id": mode_id,
            "outcome": (
                "diagnostic_complete"
                if diagnostic_pass
                else "diagnostic_degraded_floor"
            ),
            "accepted_for_manual_upload": False,
            "full_run_proven": False,
            "checks": checks,
            "failed_checks": [name for name, ok in checks.items() if not ok],
            "warnings": warnings,
        }

    if mode_id != "competition_rerun":
        raise ValueError(f"unknown run mode for acceptance: {mode_id!r}")

    primary_complete = (
        int(phase1.get("submitted") or 0) == expected_queries
        and int(phase1.get("completed") or 0) == expected_queries
        and phase1.get("exhausted") is True
    )
    primary_clean = (
        int(phase1.get("failed") or 0) == 0
        and int(phase1.get("cancelled") or 0) == 0
        and phase1.get("deadline_reached") is False
    )
    phase2_after_primary = int(phase2.get("submitted") or 0) == 0 or (
        primary_complete and primary_clean
    )
    checks.update(
        {
            "hidden_test_scope": snapshot.get("challenge_name")
            == "arc-agi_test_challenges.json",
            "manual_upload_allowed_by_mode": snapshot.get(
                "direct_file_submittable"
            )
            is True,
            "official_task_count": int(snapshot.get("n_tasks_total") or 0)
            == expected_tasks,
            "official_query_count": int(snapshot.get("n_queries_total") or 0)
            == expected_queries,
            "canonical_task_target": int(
                snapshot.get("expected_task_target") or 0
            ) == expected_tasks,
            "canonical_query_target": int(
                snapshot.get("expected_query_target") or 0
            ) == expected_queries,
            "model_requested": snapshot.get("run_model") is True,
            "model_work_eligible": snapshot.get("model_work_eligible") is True,
            "server_served": snapshot.get("served") is True,
            "exact_vllm_version": snapshot.get("vllm_version_match") is True,
            "tool_flags_active": snapshot.get("tool_flags_active") is True,
            "live_python_tool_probe": snapshot.get("tool_probe_ok") is True,
            "primary_universal_coverage": primary_complete,
            "primary_phase_clean": primary_clean,
            "primary_records_complete": int(
                snapshot.get("primary_sample_records") or 0
            ) == expected_queries,
            "primary_records_unique": int(
                snapshot.get("primary_unique_queries") or 0
            ) == expected_queries
            and int(snapshot.get("primary_duplicate_records") or 0) == 0,
            "primary_query_set_complete": snapshot.get(
                "primary_expected_query_coverage"
            )
            is True,
            "primary_policy_xhigh": int(
                snapshot.get("primary_xhigh_records") or 0
            ) == expected_queries
            and int(snapshot.get("primary_wrong_policy_records") or 0) == 0,
            # This must be counted from actual slot-0/xhigh sample rows.  A
            # medium slot-1 grid may be published into submission attempt_1
            # when primary produced no grid; publication-slot accounting must
            # never be allowed to masquerade as universal primary coverage.
            "primary_valid_grids_complete": int(
                snapshot.get("primary_valid_grids") or 0
            ) == expected_queries,
            "primary_samples_clean": int(
                snapshot.get("primary_anomaly_count") or 0
            )
            == 0,
            "sample_telemetry_complete": (
                int(snapshot.get("sample_event_write_errors") or 0) == 0
                and int(snapshot.get("sample_event_records") or 0)
                == int(snapshot.get("sample_record_count") or 0)
            ),
            "sample_telemetry_hash_present": bool(
                snapshot.get("sample_events_sha256")
            ),
            "server_log_hash_present": bool(snapshot.get("server_log_sha256")),
            "phase2_only_after_primary": phase2_after_primary,
        }
    )
    accepted = all(checks.values())
    return {
        "schema": "arc-qwen38-run-acceptance-v2",
        "mode_id": mode_id,
        "outcome": (
            "competition_full_run_pass"
            if accepted
            else "competition_rerun_degraded_floor"
        ),
        "accepted_for_manual_upload": accepted,
        "full_run_proven": accepted,
        "checks": checks,
        "failed_checks": [name for name, ok in checks.items() if not ok],
        "warnings": warnings,
    }

def parse_vllm_scheduler_log(text):
    """Summarize active-request vLLM scheduler telemetry from server logs."""

    prompt_rates, generation_rates, running_counts, kv_values = [], [], [], []
    for line in text.splitlines():
        prompt_match = re.search(
            r"Avg prompt throughput: ([0-9.]+) tokens/s", line)
        generation_match = re.search(
            r"Avg generation throughput: ([0-9.]+) tokens/s", line)
        running_match = re.search(r"Running: ([0-9]+) reqs", line)
        kv_match = re.search(r"GPU KV cache usage: ([0-9.]+)%", line)
        running = int(running_match.group(1)) if running_match else None
        if running is not None:
            running_counts.append(running)
        if prompt_match and (running is None or running > 0):
            value = float(prompt_match.group(1))
            if value > 0:
                prompt_rates.append(value)
        if generation_match and (running is None or running > 0):
            value = float(generation_match.group(1))
            if value > 0:
                generation_rates.append(value)
        if kv_match:
            kv_values.append(float(kv_match.group(1)))

    def mean(values):
        return (sum(values) / len(values)) if values else None

    mean_prompt = mean(prompt_rates)
    mean_generation = mean(generation_rates)
    return {
        "scheduler_log_samples": len(generation_rates),
        "prefill_tokens_per_second": (round(mean_prompt, 1)
                                       if mean_prompt is not None else None),
        "decode_tokens_per_second": (round(mean_generation, 1)
                                      if mean_generation is not None else None),
        "peak_prefill_tokens_per_second": (max(prompt_rates)
                                             if prompt_rates else None),
        "peak_decode_tokens_per_second": (max(generation_rates)
                                            if generation_rates else None),
        # Backward-compatible name retained for existing receipt readers.
        "peak_vllm_gen_throughput_tok_s": (max(generation_rates)
                                             if generation_rates else None),
        "mean_running_requests": (round(mean(running_counts), 1)
                                   if running_counts else None),
        "peak_running_requests": max(running_counts) if running_counts else None,
        "peak_kv_cache_usage_pct": max(kv_values) if kv_values else None,
    }

def load_module_from_source(name, path, source):
    """Write and import an embedded module with normal import semantics."""

    path.write_text(source)
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"could not create a loader for {name!r} at {path}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return module

def parse_model_grid(text, *, require_closed_fence=False):
    """Return the last rectangular digit grid, preferring closed fences.

    A response stopped at a token limit can contain a rectangular prefix of an
    unfinished grid. Callers can require a closed fence in that case.
    """

    if not isinstance(text, str):
        return None

    def parse_block(block):
        rows = []
        for line in (block or "").strip().splitlines():
            tokens = line.strip().split()
            if tokens and all(token.isdigit() and len(token) == 1 for token in tokens):
                rows.append([int(token) for token in tokens])
            elif rows:
                break
        if not rows or any(len(row) != len(rows[0]) for row in rows):
            return None
        return rows

    fenced = re.findall(r"```(?:grid|text)?\s*\n(.*?)```", text or "", re.DOTALL)
    for candidate in reversed(fenced):
        grid = parse_block(candidate)
        if grid is not None:
            return grid

    if require_closed_fence:
        return None

    blocks = []
    current = []
    for line in (text or "").splitlines() + [""]:
        tokens = line.strip().split()
        if tokens and all(token.isdigit() and len(token) == 1 for token in tokens):
            current.append(line)
        elif current:
            blocks.append("\n".join(current))
            current = []
    for candidate in reversed(blocks):
        grid = parse_block(candidate)
        if grid is not None:
            return grid
    return None

def response_reasoning(message):
    """Read vLLM's current field while retaining older-server compatibility."""

    if not isinstance(message, dict):
        return ""
    value = message.get("reasoning") or message.get("reasoning_content") or ""
    return value if isinstance(value, str) else ""

def assistant_replay_message(message):
    """Normalize one assistant tool-call turn for vLLM history replay."""

    if not isinstance(message, dict):
        raise TypeError("assistant replay message must be an object")
    content = message.get("content") or ""
    if not isinstance(content, str):
        content = ""
    tool_calls = message.get("tool_calls")
    if tool_calls is None:
        tool_calls = []
    elif not isinstance(tool_calls, list):
        raise TypeError("assistant replay tool_calls must be a list")
    replay = {
        "role": "assistant",
        "content": content,
        "tool_calls": tool_calls,
    }
    reasoning = response_reasoning(message)
    if reasoning:
        replay["reasoning"] = reasoning
    return replay

def decode_tool_code(function):
    """Accept OpenAI JSON-string arguments and already-decoded mappings."""

    if not isinstance(function, dict):
        return ""
    arguments = function.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, json.JSONDecodeError):
            return ""
    if not isinstance(arguments, dict):
        return ""
    code = arguments.get("code", "")
    return code if isinstance(code, str) else ""

def bounded_generation_cap(
    messages, configured_max, context_len=65536, reserve_tokens=1024
):
    """Conservatively keep prompt plus generation inside the context window."""

    estimated_prompt = sum(len(json.dumps(message)) for message in messages) // 2 + 512
    available = context_len - estimated_prompt - reserve_tokens
    return max(1, min(configured_max, available))

def tool_wall_budget_seconds(deadline_s, now_s, default_wall_s=20.0):
    """Return a tool wall limit that cannot outlive its episode deadline."""

    default_wall_s = float(default_wall_s)
    if default_wall_s <= 0:
        raise ValueError("default tool wall limit must be positive")
    remaining_s = float(deadline_s) - float(now_s)
    if remaining_s <= 0:
        return None
    return min(default_wall_s, remaining_s)

def model_list_contains_expected_model(payload, expected_model):
    """Validate the `/v1/models` readiness payload against the served name."""

    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        return False
    return any(
        isinstance(row, dict) and row.get("id") == expected_model
        for row in payload["data"]
    )


WORK = Path("/kaggle/working")
SUB = WORK / "submission.json"
RECEIPT = WORK / "freeform_receipt.json"
ACCEPTANCE = WORK / "freeform_acceptance.json"
STATUS = WORK / "freeform_status.json"
SAMPLE_EVENTS = WORK / "freeform_samples.jsonl"
SERVER_LOG = WORK / "vllm_server.log"
PREVIOUS_RUNS = WORK / "freeform_previous_runs"
PORT = int(RUNTIME_CONFIG["server"]["port"])
MODEL_NAME = RUNTIME_CONFIG["model"]["served_name"]
EXPECTED_VLLM_VERSION = RUNTIME_CONFIG["vllm"]["version"]
AGGREGATOR_SOURCE = '#!/usr/bin/env python3\n"""Aggregate immutable replica candidates into a complete ARC pass@2 file."""\n\nfrom __future__ import annotations\n\nimport argparse\nimport hashlib\nimport json\nimport math\nimport os\nimport re\nimport sys\nimport tempfile\nimport traceback\nimport uuid\nfrom collections import Counter\nfrom datetime import datetime, timezone\nfrom pathlib import Path\nfrom typing import Any\n\nCOLLECTION_SCHEMA = "arc-qwen38-kaggle-collection-v1"\nCANDIDATE_SCHEMA = "arc-qwen38-kaggle-candidate-v1"\nSHA256_RE = re.compile(r"^[0-9a-f]{64}$")\n\n\ndef _stable_json(value: Any) -> str:\n    return json.dumps(value, sort_keys=True, separators=(",", ":"))\n\n\ndef _digest(value: Any) -> str:\n    return hashlib.sha256(_stable_json(value).encode()).hexdigest()\n\n\ndef _require_sha256(value: Any, *, context: str) -> str:\n    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):\n        raise RuntimeError(f"{context} must be a lowercase SHA-256 digest")\n    return value\n\n\ndef _quarantine_blocking_directory(path: Path) -> Path:\n    """Move a literal destination directory aside without deleting its contents.\n\n    Kaggle work directories can survive interrupted notebook runs. If a stale\n    run leaves a *directory* at ``submission.json``, ``os.replace`` cannot\n    publish the guaranteed floor and the failure happens before the rich-path\n    degradation handler exists. Preserve the debris under a unique sibling and\n    free the required file path. Symlinks are intentionally not followed: an\n    atomic file replace safely replaces the link itself.\n    """\n    if not path.is_dir() or path.is_symlink():\n        raise NotADirectoryError(f"not a blocking literal directory: {path}")\n    for _ in range(32):\n        quarantine = path.with_name(\n            f"{path.name}.blocking-dir.{uuid.uuid4().hex}.quarantine"\n        )\n        try:\n            os.rename(path, quarantine)\n        except FileExistsError:\n            continue\n        return quarantine\n    raise RuntimeError(f"could not allocate quarantine path for {path}")\n\n\ndef _write_json(path: Path, value: Any) -> None:\n    """Atomically publish JSON via a UNIQUE same-directory temp + fsync.\n\n    A fixed ``<name>.tmp`` sibling can be stale work-directory debris (e.g. a\n    leftover *directory* at that exact path), which makes the FIRST\n    guaranteed-floor publication raise ``IsADirectoryError`` and leave no\n    ``submission.json`` at all. A unique ``mkstemp`` temp in the same directory,\n    flushed + fsynced before ``os.replace`` (and a best-effort directory fsync\n    of the rename), makes the publish crash-durable and immune to stale\n    fixed-name debris. If the destination itself is a literal directory, it is\n    quarantined non-destructively only after the replacement payload has been\n    flushed. ``os.replace`` is atomic within one filesystem.\n    """\n    path.parent.mkdir(parents=True, exist_ok=True)\n    data = json.dumps(value, indent=2, sort_keys=True) + "\\n"\n    fd, tmp_name = tempfile.mkstemp(\n        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent))\n    tmp = Path(tmp_name)\n    try:\n        with os.fdopen(fd, "w") as handle:\n            handle.write(data)\n            handle.flush()\n            os.fsync(handle.fileno())\n        if path.is_dir() and not path.is_symlink():\n            _quarantine_blocking_directory(path)\n        os.replace(tmp, path)\n    except BaseException:\n        tmp.unlink(missing_ok=True)\n        raise\n    try:  # durability of the rename itself; failure here does not lose the file\n        dir_fd = os.open(str(path.parent), os.O_RDONLY)\n        try:\n            os.fsync(dir_fd)\n        finally:\n            os.close(dir_fd)\n    except OSError:\n        pass\n\n\ndef load_collection_contract(\n    output_root: Path,\n    challenges_path: Path,\n    *,\n    challenges: dict[str, Any] | None = None,\n    challenge_sha256: str | None = None,\n) -> dict[str, Any] | None:\n    """Load and verify the immutable sweep contract when candidates exist."""\n\n    path = output_root / "collection_contract.json"\n    candidate_paths = sorted((output_root / "candidates").glob("*/*.json"))\n    if not path.is_file():\n        if candidate_paths:\n            raise RuntimeError(\n                "candidate files exist without collection_contract.json; "\n                "refusing to mix unattributed or stale runs"\n            )\n        return None\n    try:\n        contract = json.loads(path.read_text())\n    except (OSError, json.JSONDecodeError) as exc:\n        raise RuntimeError(f"invalid collection contract: {exc}") from exc\n    if contract.get("schema") != COLLECTION_SCHEMA:\n        raise RuntimeError(f"unexpected collection schema: {contract.get(\'schema\')!r}")\n    collection_id = contract.get("collection_id")\n    payload = {key: value for key, value in contract.items() if key != "collection_id"}\n    if collection_id != _digest(payload):\n        raise RuntimeError("collection contract digest does not verify")\n    if challenges is None or challenge_sha256 is None:\n        challenges, challenge_sha256 = load_public_challenges_with_digest(\n            challenges_path\n        )\n    if contract.get("challenge_sha256") != challenge_sha256:\n        raise RuntimeError(\n            "collection contract challenge digest does not match aggregation input"\n        )\n    task_ids = sorted(str(task_id) for task_id in challenges)\n    if contract.get("task_count") != len(task_ids):\n        raise RuntimeError("collection contract task_count does not match challenges")\n    if contract.get("task_ids_sha256") != _digest(task_ids):\n        raise RuntimeError("collection contract task_ids_sha256 does not match challenges")\n    model = contract.get("model")\n    if not isinstance(model, dict):\n        raise RuntimeError("collection contract has no model identity")\n    if model.get("repo_id") != "Qwen/Qwen3.8-27B":\n        raise RuntimeError("collection contract binds the wrong model repository")\n    if model.get("revision") != "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0":\n        raise RuntimeError("collection contract binds the wrong model revision")\n    _require_sha256(model.get("source_digest"), context="model.source_digest")\n    _require_sha256(\n        model.get("source_contract_sha256"),\n        context="model.source_contract_sha256",\n    )\n    _require_sha256(model.get("receipt_sha256"), context="model.receipt_sha256")\n    if model.get("weight_hashes_verified") is not True:\n        raise RuntimeError("collection contract lacks verified model weight hashes")\n    verified_file_count = model.get("verified_file_count")\n    if (\n        isinstance(verified_file_count, bool)\n        or not isinstance(verified_file_count, int)\n        or verified_file_count < 19\n    ):\n        raise RuntimeError("collection contract has an incomplete model file inventory")\n    runtime = contract.get("runtime")\n    if not isinstance(runtime, dict):\n        raise RuntimeError("collection contract has no runtime identity")\n    _require_sha256(\n        runtime.get("receipt_sha256"), context="runtime.receipt_sha256"\n    )\n    schedule = contract.get("schedule")\n    if not isinstance(schedule, dict):\n        raise RuntimeError("collection contract has no schedule")\n    if schedule.get("world_size") != 4:\n        raise RuntimeError("collection contract must use exactly four replicas")\n    for field in ("rollouts", "max_turns"):\n        value = schedule.get(field)\n        if isinstance(value, bool) or not isinstance(value, int) or value < 1:\n            raise RuntimeError(f"collection schedule {field} must be positive")\n    return contract\n\n\ndef validate_grid(grid: Any, *, context: str = "grid") -> None:\n    if not isinstance(grid, list) or not 1 <= len(grid) <= 30:\n        raise ValueError(f"{context}: invalid height")\n    if not all(isinstance(row, list) for row in grid):\n        raise ValueError(f"{context}: rows must be lists")\n    width = len(grid[0])\n    if not 1 <= width <= 30 or any(len(row) != width for row in grid):\n        raise ValueError(f"{context}: invalid rectangle")\n    if any(\n        isinstance(value, bool)\n        or not isinstance(value, int)\n        or not 0 <= value <= 9\n        for row in grid\n        for value in row\n    ):\n        raise ValueError(f"{context}: invalid color")\n\n\ndef _reject_duplicate_object_pairs(\n    pairs: list[tuple[str, Any]],\n) -> dict[str, Any]:\n    value: dict[str, Any] = {}\n    for key, item in pairs:\n        if key in value:\n            raise ValueError(f"duplicate JSON object key: {key!r}")\n        value[key] = item\n    return value\n\n\ndef validate_public_challenges(challenges: Any) -> dict[str, Any]:\n    """Accept only public ARC train labels and unlabeled test inputs."""\n\n    if not isinstance(challenges, dict) or not challenges:\n        raise ValueError("challenge JSON must be a non-empty task map")\n    for task_id, task in challenges.items():\n        if not isinstance(task_id, str) or not task_id:\n            raise ValueError("challenge task IDs must be non-empty strings")\n        context = f"challenge[{task_id!r}]"\n        if not isinstance(task, dict) or set(task) != {"train", "test"}:\n            raise ValueError(\n                f"{context}: task keys must be exactly train/test; "\n                "hidden labels or metadata are forbidden"\n            )\n        for split in ("train", "test"):\n            pairs = task[split]\n            if not isinstance(pairs, list) or not pairs:\n                raise ValueError(f"{context}.{split}: must be a non-empty list")\n            expected_keys = {"input", "output"} if split == "train" else {"input"}\n            for index, pair in enumerate(pairs):\n                pair_context = f"{context}.{split}[{index}]"\n                if not isinstance(pair, dict) or set(pair) != expected_keys:\n                    raise ValueError(\n                        f"{pair_context}: keys must be exactly "\n                        f"{sorted(expected_keys)!r}; hidden test labels are forbidden"\n                    )\n                validate_grid(pair["input"], context=f"{pair_context}.input")\n                if split == "train":\n                    validate_grid(pair["output"], context=f"{pair_context}.output")\n    return challenges\n\n\ndef load_public_challenges_with_digest(\n    path: Path,\n) -> tuple[dict[str, Any], str]:\n    try:\n        raw = path.read_bytes()\n        value = json.loads(\n            raw.decode("utf-8"), object_pairs_hook=_reject_duplicate_object_pairs\n        )\n    except (OSError, UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:\n        raise RuntimeError(f"invalid public challenge JSON: {exc}") from exc\n    try:\n        challenges = validate_public_challenges(value)\n    except ValueError as exc:\n        raise RuntimeError(f"unsafe public challenge JSON: {exc}") from exc\n    return challenges, hashlib.sha256(raw).hexdigest()\n\n\ndef load_public_challenges(path: Path) -> dict[str, Any]:\n    return load_public_challenges_with_digest(path)[0]\n\n\ndef _valid_outputs(task_id: str, task: dict[str, Any], outputs: Any) -> bool:\n    if not isinstance(outputs, list) or len(outputs) != len(task["test"]):\n        return False\n    try:\n        for index, grid in enumerate(outputs):\n            validate_grid(grid, context=f"{task_id}.candidate[{index}]")\n    except ValueError:\n        return False\n    return True\n\n\ndef _identity(task: dict[str, Any]) -> list[list[list[int]]]:\n    return [pair["input"] for pair in task["test"]]\n\n\ndef _background(task: dict[str, Any]) -> list[list[list[int]]]:\n    counts = Counter(\n        value\n        for pair in task["train"]\n        for row in pair["output"]\n        for value in row\n    )\n    color = counts.most_common(1)[0][0]\n    same_shape = all(\n        (len(pair["input"]), len(pair["input"][0]))\n        == (len(pair["output"]), len(pair["output"][0]))\n        for pair in task["train"]\n    )\n    shapes = {\n        (len(pair["output"]), len(pair["output"][0])) for pair in task["train"]\n    }\n    fixed = next(iter(shapes)) if len(shapes) == 1 else None\n    rows = []\n    for pair in task["test"]:\n        if same_shape or fixed is None:\n            height, width = len(pair["input"]), len(pair["input"][0])\n        else:\n            height, width = fixed\n        rows.append([[color] * width for _ in range(height)])\n    return rows\n\n\ndef _geometric_grid(source: str, grid: list[list[int]]) -> list[list[int]]:\n    if source == "fallback-flip-lr":\n        return [list(reversed(row)) for row in grid]\n    if source == "fallback-flip-ud":\n        return list(reversed([row[:] for row in grid]))\n    if source == "fallback-transpose":\n        return [list(row) for row in zip(*grid)]\n    if source == "fallback-rotate-cw":\n        return [list(row) for row in zip(*reversed(grid))]\n    if source == "fallback-rotate-ccw":\n        return [list(row) for row in reversed(list(zip(*grid)))]\n    if source == "fallback-rotate-180":\n        return [list(reversed(row)) for row in reversed(grid)]\n    raise ValueError(f"unknown geometric fallback: {source}")\n\n\ndef _constant_color(task: dict[str, Any], color: int) -> list[list[list[int]]]:\n    same_shape = all(\n        (len(pair["input"]), len(pair["input"][0]))\n        == (len(pair["output"]), len(pair["output"][0]))\n        for pair in task["train"]\n    )\n    shapes = {\n        (len(pair["output"]), len(pair["output"][0])) for pair in task["train"]\n    }\n    fixed = next(iter(shapes)) if len(shapes) == 1 else None\n    rows = []\n    for pair in task["test"]:\n        if same_shape or fixed is None:\n            height, width = len(pair["input"]), len(pair["input"][0])\n        else:\n            height, width = fixed\n        rows.append([[color] * width for _ in range(height)])\n    return rows\n\n\ndef _fallback_outputs(\n    task: dict[str, Any], source: str\n) -> list[list[list[int]]]:\n    if source == "fallback-identity":\n        return _identity(task)\n    if source == "fallback-background":\n        return _background(task)\n    if source.startswith("fallback-constant-"):\n        return _constant_color(task, int(source.rsplit("-", 1)[-1]))\n    return [_geometric_grid(source, pair["input"]) for pair in task["test"]]\n\n\ndef _fallback_demo_score(\n    task: dict[str, Any], source: str\n) -> tuple[bool, float]:\n    if source == "fallback-identity":\n        matches = [pair["input"] == pair["output"] for pair in task["train"]]\n    elif source == "fallback-background":\n        counts = Counter(\n            value\n            for pair in task["train"]\n            for row in pair["output"]\n            for value in row\n        )\n        color = counts.most_common(1)[0][0]\n        same_shape = all(\n            (len(pair["input"]), len(pair["input"][0]))\n            == (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        )\n        shapes = {\n            (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        }\n        fixed = next(iter(shapes)) if len(shapes) == 1 else None\n        matches = []\n        for pair in task["train"]:\n            if same_shape or fixed is None:\n                height, width = len(pair["input"]), len(pair["input"][0])\n            else:\n                height, width = fixed\n            matches.append([[color] * width for _ in range(height)] == pair["output"])\n    elif source.startswith("fallback-constant-"):\n        color = int(source.rsplit("-", 1)[-1])\n        same_shape = all(\n            (len(pair["input"]), len(pair["input"][0]))\n            == (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        )\n        shapes = {\n            (len(pair["output"]), len(pair["output"][0]))\n            for pair in task["train"]\n        }\n        fixed = next(iter(shapes)) if len(shapes) == 1 else None\n        matches = []\n        for pair in task["train"]:\n            if same_shape or fixed is None:\n                height, width = len(pair["input"]), len(pair["input"][0])\n            else:\n                height, width = fixed\n            matches.append([[color] * width for _ in range(height)] == pair["output"])\n    else:\n        matches = [\n            _geometric_grid(source, pair["input"]) == pair["output"]\n            for pair in task["train"]\n        ]\n    return all(matches), sum(matches) / len(matches)\n\n\ndef _trajectory_id(row: dict[str, Any]) -> tuple[int, int] | None:\n    rank = row.get("rank")\n    rollout = row.get("rollout")\n    if not isinstance(rank, int) or isinstance(rank, bool):\n        return None\n    if not isinstance(rollout, int) or isinstance(rollout, bool):\n        return None\n    return rank, rollout\n\n\ndef _trajectory_candidate_key(\n    row: dict[str, Any],\n) -> tuple[bool, float, int, int, str]:\n    turn = row.get("turn")\n    turn_value = turn if isinstance(turn, int) and not isinstance(turn, bool) else -1\n    return (\n        row.get("visible_pass") is True,\n        float(row.get("demo_score") or 0.0),\n        turn_value,\n        -int(row.get("program_cost") or 0),\n        str(row.get("run_id") or ""),\n    )\n\n\ndef _validate_selection_candidates(\n    task_id: str, task: dict[str, Any], candidates: list[dict[str, Any]]\n) -> None:\n    """Reject malformed in-memory candidates before ranking or clustering."""\n\n    seen_run_ids: set[str] = set()\n    for index, row in enumerate(candidates):\n        context = f"{task_id}.candidates[{index}]"\n        if not isinstance(row, dict):\n            raise RuntimeError(f"{context}: candidate must be an object")\n        if _trajectory_id(row) is None:\n            raise RuntimeError(f"{context}: candidate has no valid rank/rollout identity")\n        turn = row.get("turn")\n        if turn is not None and (\n            isinstance(turn, bool) or not isinstance(turn, int) or turn < 0\n        ):\n            raise RuntimeError(f"{context}: turn must be a non-negative integer")\n        run_id = row.get("run_id")\n        if not isinstance(run_id, str) or not run_id:\n            raise RuntimeError(f"{context}: run_id must be non-empty")\n        if run_id in seen_run_ids:\n            raise RuntimeError(f"{context}: duplicate run_id {run_id!r}")\n        seen_run_ids.add(run_id)\n        if not isinstance(row.get("visible_pass"), bool):\n            raise RuntimeError(f"{context}: visible_pass must be boolean")\n        demo_score = row.get("demo_score")\n        if (\n            isinstance(demo_score, bool)\n            or not isinstance(demo_score, (int, float))\n            or not math.isfinite(float(demo_score))\n            or not 0.0 <= float(demo_score) <= 1.0\n        ):\n            raise RuntimeError(f"{context}: demo_score must be finite in 0..1")\n        program_cost = row.get("program_cost")\n        if (\n            isinstance(program_cost, bool)\n            or not isinstance(program_cost, int)\n            or program_cost < 0\n        ):\n            raise RuntimeError(f"{context}: program_cost must be non-negative")\n        if not _valid_outputs(task_id, task, row.get("test_outputs")):\n            raise RuntimeError(f"{context}: candidate test outputs are invalid")\n\n\ndef _trajectory_representatives(\n    candidates: list[dict[str, Any]],\n) -> list[dict[str, Any]]:\n    """Choose one vote-bearing candidate per independent trajectory."""\n\n    best: dict[tuple[int, int], dict[str, Any]] = {}\n    for row in candidates:\n        trajectory = _trajectory_id(row)\n        if trajectory is None:\n            continue\n        current = best.get(trajectory)\n        if current is None or _trajectory_candidate_key(row) > _trajectory_candidate_key(\n            current\n        ):\n            best[trajectory] = row\n    return [best[key] for key in sorted(best)]\n\n\ndef _selection_key(row: dict[str, Any]) -> tuple[Any, ...]:\n    """Order hypotheses by visible evidence without conflating repairs.\n\n    ``support`` contains only the one vote-bearing state selected from each\n    independent ``(rank, rollout)`` trajectory.  Older repair states remain\n    useful pass@2 hypotheses, but their raw provenance can break ties only\n    after independent support and can never manufacture another vote.\n    """\n\n    return (\n        row["visible_pass"],\n        row["demo_score"],\n        row["visible_pass_support"],\n        row["support"],\n        row["model_derived"],\n        row["raw_visible_pass_trajectory_support"],\n        row["raw_trajectory_support"],\n        row["replica_support"],\n        row["rollout_support"],\n        -row["program_cost"],\n        row["representative_id"],\n    )\n\n\ndef load_candidates(\n    output_root: Path,\n    task_id: str,\n    task: dict[str, Any],\n    *,\n    collection_contract: dict[str, Any] | None = None,\n) -> list[dict[str, Any]]:\n    rows = []\n    directory = output_root / "candidates" / task_id\n    if not directory.is_dir():\n        return rows\n    if collection_contract is None:\n        raise RuntimeError(\n            f"{task_id}: candidate directory exists without a verified collection contract"\n        )\n    collection_id = collection_contract["collection_id"]\n    contract_sha256 = _digest(collection_contract)\n    seen_run_ids: set[str] = set()\n    schedule = collection_contract["schedule"]\n    world_size = int(schedule["world_size"])\n    rollouts = int(schedule["rollouts"])\n    max_turns = int(schedule["max_turns"])\n    for path in sorted(directory.glob("*.json")):\n        try:\n            row = json.loads(path.read_text())\n        except (OSError, json.JSONDecodeError) as exc:\n            raise RuntimeError(f"{path}: invalid candidate JSON: {exc}") from exc\n        if not isinstance(row, dict):\n            raise RuntimeError(f"{path}: candidate must be an object")\n        if row.get("schema") != CANDIDATE_SCHEMA:\n            raise RuntimeError(f"{path}: unexpected candidate schema")\n        declared_record_sha256 = _require_sha256(\n            row.get("candidate_record_sha256"),\n            context=f"{path}: candidate_record_sha256",\n        )\n        record_payload = {\n            key: value\n            for key, value in row.items()\n            if key != "candidate_record_sha256"\n        }\n        if declared_record_sha256 != _digest(record_payload):\n            raise RuntimeError(f"{path}: candidate record digest does not verify")\n        if row.get("collection_id") != collection_id:\n            raise RuntimeError(\n                f"{path}: candidate belongs to a different or unattributed collection"\n            )\n        if row.get("collection_contract_sha256") != contract_sha256:\n            raise RuntimeError(f"{path}: candidate collection-contract digest mismatch")\n        if row.get("challenge_sha256") != collection_contract["challenge_sha256"]:\n            raise RuntimeError(f"{path}: candidate challenge digest mismatch")\n        if row.get("model") != collection_contract["model"]:\n            raise RuntimeError(f"{path}: candidate model identity mismatch")\n        if row.get("runtime_receipt_sha256") != collection_contract["runtime"].get(\n            "receipt_sha256"\n        ):\n            raise RuntimeError(f"{path}: candidate runtime identity mismatch")\n        if row.get("task_id") != task_id:\n            raise RuntimeError(f"{path}: candidate task ID does not match its directory")\n        trajectory = _trajectory_id(row)\n        if trajectory is None:\n            raise RuntimeError(f"{path}: candidate has no valid rank/rollout identity")\n        rank, rollout = trajectory\n        turn = row.get("turn")\n        if not 0 <= rank < world_size:\n            raise RuntimeError(f"{path}: rank is outside the collection world size")\n        if not 0 <= rollout < rollouts:\n            raise RuntimeError(f"{path}: rollout is outside the collection schedule")\n        if isinstance(turn, bool) or not isinstance(turn, int) or not 0 <= turn < max_turns:\n            raise RuntimeError(f"{path}: turn is outside the collection schedule")\n        expected_name = f"rank{rank}-rollout{rollout}-turn{turn}.json"\n        if path.name != expected_name:\n            raise RuntimeError(\n                f"{path}: candidate filename does not match trajectory identity"\n            )\n        expected_run_id = f"{task_id}-rank{rank}-rollout{rollout}-turn{turn}"\n        run_id = row.get("run_id")\n        if run_id != expected_run_id:\n            raise RuntimeError(f"{path}: candidate run_id does not match trajectory")\n        if run_id in seen_run_ids:\n            raise RuntimeError(f"{path}: duplicate candidate run_id {run_id!r}")\n        seen_run_ids.add(run_id)\n        if row.get("source") != "qwen38-nf4-replica":\n            raise RuntimeError(f"{path}: unexpected candidate source")\n        if not isinstance(row.get("visible_pass"), bool):\n            raise RuntimeError(f"{path}: visible_pass must be boolean")\n        demo_score = row.get("demo_score")\n        if (\n            isinstance(demo_score, bool)\n            or not isinstance(demo_score, (int, float))\n            or not math.isfinite(float(demo_score))\n            or not 0.0 <= float(demo_score) <= 1.0\n        ):\n            raise RuntimeError(f"{path}: demo_score must be finite in 0..1")\n        program = row.get("program")\n        if not isinstance(program, str) or not program.strip():\n            raise RuntimeError(f"{path}: program must be non-empty")\n        program_sha256 = _require_sha256(\n            row.get("program_sha256"), context=f"{path}: program_sha256"\n        )\n        if program_sha256 != hashlib.sha256(program.encode()).hexdigest():\n            raise RuntimeError(f"{path}: program_sha256 does not match program bytes")\n        program_cost = row.get("program_cost")\n        if (\n            isinstance(program_cost, bool)\n            or not isinstance(program_cost, int)\n            or program_cost != len(program)\n        ):\n            raise RuntimeError(f"{path}: program_cost does not match program length")\n        if not _valid_outputs(task_id, task, row.get("test_outputs")):\n            raise RuntimeError(f"{path}: candidate test outputs are invalid")\n        test_outputs_sha256 = _require_sha256(\n            row.get("test_outputs_sha256"),\n            context=f"{path}: test_outputs_sha256",\n        )\n        if test_outputs_sha256 != _digest(row["test_outputs"]):\n            raise RuntimeError(\n                f"{path}: test_outputs_sha256 does not match test outputs"\n            )\n        rows.append(row)\n    return rows\n\n\ndef validate_submission_completeness(\n    challenges: dict[str, Any], submission: Any\n) -> None:\n    """Assert a submission is structurally complete and grid-valid.\n\n    The single reusable definition of "a submittable file": exactly the\n    challenge task set, one attempt object per test query, each object exactly\n    {attempt_1, attempt_2}, every attempt a valid ARC grid (1..30 dims, colours\n    0..9), and the two attempts distinct. Reused by :func:`aggregate`, the\n    last-resort fallback path, and the offline robustness harness so the\n    guarantee has one source of truth.\n    """\n\n    if not isinstance(submission, dict):\n        raise ValueError("submission must be a task->rows object")\n    if set(submission) != set(challenges):\n        missing = sorted(set(challenges) - set(submission))\n        extra = sorted(set(submission) - set(challenges))\n        raise ValueError(\n            f"submission task set mismatch (missing={missing}, extra={extra})"\n        )\n    for task_id, task in challenges.items():\n        rows = submission[task_id]\n        if not isinstance(rows, list) or len(rows) != len(task["test"]):\n            raise ValueError(f"{task_id}: wrong submission output count")\n        for index, row in enumerate(rows):\n            if not isinstance(row, dict) or set(row) != {"attempt_1", "attempt_2"}:\n                raise ValueError(f"{task_id}[{index}]: malformed attempt object")\n            for name in ("attempt_1", "attempt_2"):\n                validate_grid(row[name], context=f"{task_id}[{index}].{name}")\n            if row["attempt_1"] == row["attempt_2"]:\n                raise ValueError(f"{task_id}[{index}]: duplicate pass@2 attempts")\n\n\ndef guaranteed_fallback_submission(\n    challenges: dict[str, Any],\n) -> dict[str, list[dict[str, Any]]]:\n    """Build a structurally-complete submission that cannot fail to be valid.\n\n    This is the robustness floor for the BASE: it depends only on the already\n    validated public test *inputs* (never on train outputs, the collection\n    contract, model candidates, or the rich selection machinery), so it can be\n    written to disk before the model even loads and can never itself raise for a\n    validated challenge set. attempt_1 is the test input (identity); attempt_2\n    is the first distinct dihedral transform. If the input is invariant under\n    every such transform, one cell is changed to keep pass@2 from wasting its\n    second slot on an exact duplicate. A better submission from\n    :func:`aggregate` overwrites it on success; if every richer path fails, this\n    remains and the run still submits a valid pass@2 file.\n    """\n\n    submission: dict[str, list[dict[str, Any]]] = {}\n    for task_id, task in challenges.items():\n        rows: list[dict[str, Any]] = []\n        for pair in task["test"]:\n            grid = pair["input"]\n            candidates = [\n                _geometric_grid("fallback-flip-lr", grid),\n                _geometric_grid("fallback-flip-ud", grid),\n                _geometric_grid("fallback-rotate-180", grid),\n                _geometric_grid("fallback-transpose", grid),\n            ]\n            second = next(\n                (candidate for candidate in candidates if candidate != grid),\n                None,\n            )\n            if second is None:\n                second = [row[:] for row in grid]\n                second[0][0] = (second[0][0] + 1) % 10\n            rows.append({"attempt_1": grid, "attempt_2": second})\n        submission[task_id] = rows\n    # Fail loud if this floor is ever not valid: it means the input contract\n    # was violated upstream, which must surface, not be silently shipped.\n    validate_submission_completeness(challenges, submission)\n    return submission\n\n\ndef select_task(\n    task_id: str, task: dict[str, Any], candidates: list[dict[str, Any]]\n) -> tuple[list[dict[str, Any]], dict[str, Any]]:\n    fallback_sources = (\n        "fallback-identity",\n        "fallback-background",\n        "fallback-flip-lr",\n        "fallback-flip-ud",\n        "fallback-transpose",\n        "fallback-rotate-cw",\n        "fallback-rotate-ccw",\n        "fallback-rotate-180",\n    ) + tuple(f"fallback-constant-{color}" for color in range(10))\n    fallback_outputs = {\n        source: _fallback_outputs(task, source) for source in fallback_sources\n    }\n    _validate_selection_candidates(task_id, task, candidates)\n    submission_rows = []\n    query_receipts = []\n    trajectory_candidates = _trajectory_representatives(candidates)\n    for query_index in range(len(task["test"])):\n        # One trajectory can revise its output over several turns. Select its\n        # strongest state before clustering so it cannot vote for competing\n        # hypotheses at the same cutoff.\n        groups: dict[str, list[dict[str, Any]]] = {}\n        for candidate in trajectory_candidates:\n            grid = candidate["test_outputs"][query_index]\n            groups.setdefault(_digest(grid), []).append(candidate)\n        raw_groups: dict[str, list[dict[str, Any]]] = {}\n        for candidate in candidates:\n            grid = candidate["test_outputs"][query_index]\n            raw_groups.setdefault(_digest(grid), []).append(candidate)\n        ranked = []\n        for output_sha256, raw_members in raw_groups.items():\n            members = groups.get(output_sha256, [])\n            # A vote-bearing representative must come from the trajectory\n            # representatives themselves.  Otherwise a superseded state with\n            # the same test output could silently inflate attempt-1\'s visible\n            # score.  Correlated-only clusters have no such member, so their\n            # best raw state is used for attempt-2 ranking only.\n            representative = max(\n                members or raw_members,\n                key=_trajectory_candidate_key,\n            )\n            replicas = sorted(\n                {\n                    int(row["rank"])\n                    for row in members\n                    if isinstance(row.get("rank"), int)\n                }\n            )\n            rollouts = sorted(\n                {\n                    int(row["rollout"])\n                    for row in members\n                    if isinstance(row.get("rollout"), int)\n                }\n            )\n            trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in members\n            }\n            visible_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in members\n                if row.get("visible_pass") is True\n            }\n            raw_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in raw_members\n            }\n            raw_visible_trajectories = {\n                (int(row.get("rank", -1)), int(row.get("rollout", -1)))\n                for row in raw_members\n                if row.get("visible_pass") is True\n            }\n            correlated_only = not members\n            ranked.append(\n                {\n                    "grid": representative["test_outputs"][query_index],\n                    "output_sha256": output_sha256,\n                    "source": str(\n                        representative.get("source") or "qwen38-nf4-replica"\n                    ),\n                    "representative_id": str(representative.get("run_id") or ""),\n                    "visible_pass": representative.get("visible_pass") is True,\n                    "demo_score": float(representative.get("demo_score") or 0.0),\n                    "program_cost": int(representative.get("program_cost") or 0),\n                    "model_derived": True,\n                    "correlated_only": correlated_only,\n                    "attempt_1_eligible": not correlated_only,\n                    "hypothesis_origin": (\n                        "correlated-repair-state"\n                        if correlated_only\n                        else "independent-trajectory-vote"\n                    ),\n                    # Multiple repair turns within one rollout are correlated\n                    # and count as one vote. Consensus means independent\n                    # rank/rollout trajectories, not raw candidate files.\n                    "support": len(trajectories),\n                    "independent_support": len(trajectories),\n                    "raw_candidate_support": len(raw_members),\n                    "raw_trajectory_support": len(raw_trajectories),\n                    "raw_visible_pass_trajectory_support": len(\n                        raw_visible_trajectories\n                    ),\n                    "visible_pass_support": len(visible_trajectories),\n                    "replica_support": len(replicas),\n                    "rollout_support": len(rollouts),\n                    "supporting_replicas": replicas,\n                    "supporting_rollouts": rollouts,\n                    "supporting_ids": sorted(\n                        str(row.get("run_id") or "") for row in members\n                    ),\n                    "raw_supporting_ids": sorted(\n                        str(row.get("run_id") or "") for row in raw_members\n                    ),\n                }\n            )\n        # Visible evidence is primary.  Among equally fitting programs, the\n        # output with the most independent support wins attempt 1.  Attempt 2\n        # is the strongest distinct cluster, maximizing pass@2 diversity.\n        ranked.sort(key=_selection_key, reverse=True)\n        for source in fallback_sources:\n            grid = fallback_outputs[source][query_index]\n            fallback_pass, fallback_score = _fallback_demo_score(task, source)\n            ranked.append(\n                {\n                    "grid": grid,\n                    "output_sha256": _digest(grid),\n                    "source": source,\n                    "representative_id": source,\n                    "visible_pass": fallback_pass,\n                    "demo_score": fallback_score,\n                    "program_cost": 0,\n                    "model_derived": False,\n                    "correlated_only": False,\n                    "attempt_1_eligible": True,\n                    "hypothesis_origin": "deterministic-fallback",\n                    "support": 0,\n                    "independent_support": 0,\n                    "raw_candidate_support": 0,\n                    "raw_trajectory_support": 0,\n                    "raw_visible_pass_trajectory_support": 0,\n                    "visible_pass_support": 0,\n                    "replica_support": 0,\n                    "rollout_support": 0,\n                    "supporting_replicas": [],\n                    "supporting_rollouts": [],\n                    "supporting_ids": [],\n                    "raw_supporting_ids": [],\n                }\n            )\n        ranked.sort(key=_selection_key, reverse=True)\n\n        # A superseded repair state is deliberately ineligible for attempt 1:\n        # its trajectory cast its sole independent vote for another state.\n        # It remains eligible for attempt 2, where preserving a strong,\n        # distinct model-derived hypothesis is preferable to erasing it and\n        # falling back to an unrelated geometric guess.\n        attempt_1_pool = [row for row in ranked if row["attempt_1_eligible"]]\n        if not attempt_1_pool:\n            raise RuntimeError(\n                f"{task_id}.test[{query_index}]: no attempt-1-eligible hypothesis"\n            )\n        selected = [attempt_1_pool[0]]\n        seen = {attempt_1_pool[0]["output_sha256"]}\n        for row in ranked:\n            if row["output_sha256"] in seen:\n                continue\n            selected.append(row)\n            seen.add(row["output_sha256"])\n            if len(selected) == 2:\n                break\n        if len(selected) != 2:\n            raise RuntimeError(\n                f"{task_id}.test[{query_index}]: fewer than two distinct "\n                "evidence-ranked output hypotheses survived selection"\n            )\n        submission_rows.append(\n            {"attempt_1": selected[0]["grid"], "attempt_2": selected[1]["grid"]}\n        )\n        query_receipts.append(\n            {\n                "query_index": query_index,\n                "candidate_clusters": len(groups),\n                "raw_candidate_clusters": len(raw_groups),\n                "correlated_only_candidate_clusters": len(\n                    set(raw_groups) - set(groups)\n                ),\n                "selected": [\n                    {key: row[key] for key in row if key != "grid"}\n                    for row in selected\n                ],\n            }\n        )\n    return submission_rows, {\n        "candidate_count": len(candidates),\n        "trajectory_candidate_count": len(trajectory_candidates),\n        "visible_pass_candidates": sum(\n            row.get("visible_pass") is True for row in candidates\n        ),\n        "queries": query_receipts,\n    }\n\n\ndef aggregate(\n    challenges_path: Path, output_root: Path, submission_path: Path\n) -> tuple[dict[str, Any], dict[str, Any]]:\n    challenges, challenge_sha256 = load_public_challenges_with_digest(\n        challenges_path\n    )\n    collection_contract = load_collection_contract(\n        output_root,\n        challenges_path,\n        challenges=challenges,\n        challenge_sha256=challenge_sha256,\n    )\n    submission = {}\n    tasks = {}\n    for task_id, task in challenges.items():\n        candidates = load_candidates(\n            output_root,\n            task_id,\n            task,\n            collection_contract=collection_contract,\n        )\n        rows, receipt = select_task(task_id, task, candidates)\n        submission[task_id] = rows\n        tasks[task_id] = receipt\n    validate_submission_completeness(challenges, submission)\n    _write_json(submission_path, submission)\n    selected_sources = Counter(\n        selected["source"]\n        for task in tasks.values()\n        for query in task["queries"]\n        for selected in query["selected"]\n    )\n    receipt = {\n        "schema": "arc-qwen38-kaggle-pass2-aggregation-v1",\n        "observed_utc": datetime.now(timezone.utc).isoformat(),\n        "selection_scope": "per-test-query",\n        "collection_id": (\n            collection_contract["collection_id"] if collection_contract else None\n        ),\n        "collection_contract_sha256": (\n            _digest(collection_contract) if collection_contract else None\n        ),\n        "selection_policy": (\n            "attempt-1 excludes superseded repair states; then visible-pass, "\n            "demo-score, independent visible-pass support, independent "\n            "trajectory support, model-derived provenance, raw trajectory "\n            "provenance, replica/rollout spread, shorter-program; attempt-2 "\n            "is the strongest distinct hypothesis including zero-vote "\n            "superseded repair states"\n        ),\n        "challenge_sha256": challenge_sha256,\n        "submission_sha256": hashlib.sha256(submission_path.read_bytes()).hexdigest(),\n        "task_count": len(challenges),\n        "query_count": sum(len(task["test"]) for task in challenges.values()),\n        "candidate_count": sum(row["candidate_count"] for row in tasks.values()),\n        "tasks_with_candidate": sum(row["candidate_count"] > 0 for row in tasks.values()),\n        "tasks_with_visible_pass_candidate": sum(\n            row["visible_pass_candidates"] > 0 for row in tasks.values()\n        ),\n        "selected_source_counts": dict(sorted(selected_sources.items())),\n        "tasks": tasks,\n        "status": "pass",\n    }\n    _write_json(output_root / "aggregation_receipt.json", receipt)\n    return submission, receipt\n\n\ndef _emit_summary(summary: dict[str, Any]) -> None:\n    print(\n        "QWEN38_PASS2_AGGREGATION=" + json.dumps(summary, sort_keys=True),\n        flush=True,\n    )\n\n\ndef main() -> int:\n    parser = argparse.ArgumentParser(description=__doc__)\n    parser.add_argument("--challenges", type=Path, required=True)\n    parser.add_argument("--output-root", type=Path, required=True)\n    parser.add_argument("--submission", type=Path, required=True)\n    parser.add_argument(\n        "--guarantee-only",\n        action="store_true",\n        help=(\n            "Write only the dependency-free guaranteed-fallback submission and "\n            "exit 0. The kernel calls this first so a valid, complete "\n            "submission exists on disk before the model loads or any candidate "\n            "is produced."\n        ),\n    )\n    args = parser.parse_args()\n\n    # ROBUSTNESS FLOOR: a valid, complete submission must exist on disk no\n    # matter what fails later.  Build it from the validated public inputs and\n    # write it before anything richer is attempted.  If the challenges\n    # themselves are unreadable there is genuinely nothing to submit, so that\n    # (and only that) is allowed to raise.\n    challenges = load_public_challenges(args.challenges)\n    _write_json(args.submission, guaranteed_fallback_submission(challenges))\n\n    base_summary = {\n        "schema": "arc-qwen38-kaggle-pass2-aggregation-v1",\n        "task_count": len(challenges),\n        "query_count": sum(len(task["test"]) for task in challenges.values()),\n    }\n    if args.guarantee_only:\n        _emit_summary({**base_summary, "status": "guaranteed-fallback"})\n        return 0\n\n    try:\n        _, receipt = aggregate(args.challenges, args.output_root, args.submission)\n        # Defensive: whatever landed on disk must be a complete, valid file.\n        validate_submission_completeness(\n            challenges, json.loads(args.submission.read_text())\n        )\n    except Exception as exc:  # never let aggregation abort the submit\n        # Fail LOUD but keep submitting: rewrite the guaranteed floor and record\n        # the failure so a degraded run is visible, not silently shipped.\n        degraded = {\n            "schema": "arc-qwen38-kaggle-aggregation-degraded-v1",\n            "observed_utc": datetime.now(timezone.utc).isoformat(),\n            "error": f"{type(exc).__name__}: {exc}",\n            "traceback": traceback.format_exc(),\n            "fallback": "guaranteed-fallback-submission-retained",\n        }\n        try:\n            _write_json(\n                args.output_root / "aggregation_degraded_receipt.json", degraded\n            )\n        except OSError:\n            pass\n        _write_json(args.submission, guaranteed_fallback_submission(challenges))\n        print(\n            "LOUD: aggregation failed; retained guaranteed-fallback submission. "\n            f"{type(exc).__name__}: {exc}",\n            file=sys.stderr,\n            flush=True,\n        )\n        _emit_summary(\n            {\n                **base_summary,\n                "status": "degraded-guaranteed-fallback",\n                "error": f"{type(exc).__name__}: {exc}",\n            }\n        )\n        return 0\n\n    _emit_summary(\n        {\n            key: receipt[key]\n            for key in (\n                "schema",\n                "observed_utc",\n                "collection_id",\n                "submission_sha256",\n                "task_count",\n                "query_count",\n                "candidate_count",\n                "tasks_with_candidate",\n                "tasks_with_visible_pass_candidate",\n                "selected_source_counts",\n                "status",\n            )\n        }\n    )\n    return 0\n\n\nif __name__ == "__main__":\n    raise SystemExit(main())\n'
AGGREGATOR_SHA256 = "b799da0fd4da8ac7ff3f0e06e822a192646ce877e9b873f10dee379d2f0a10da"
SANDBOX_SOURCE = '#!/usr/bin/env python3\n"""Resource-bounded Python executor for the model\'s code-interpreter tool.\n\nThis is a resource sandbox, not a security boundary: model-written code retains\nnormal filesystem and network access. Each call runs in a fresh process session\nand temporary directory with bounded address space, CPU time, wall time,\nper-file size, open files, and captured output.\nThe parent kills the call\'s original process group after the leader exits, so\nordinary forked background work is cleaned up after both success and timeout.\nDeliberately evasive code can escape these controls because this is not a\nsecurity boundary.\n\nThe file-size rlimit is per file rather than an aggregate disk quota. Combined\nwith the throwaway working directory, process-group cleanup, and wall deadline,\nit limits the practical damage from accidental writes but should not be treated\nas filesystem isolation. RLIMIT_AS is per process rather than an aggregate\ncgroup memory cap, and same-UID processes are not isolated. Defaults are sized\nwell under a 125 GB host:\n4 GiB address space, 12 CPU-s, 20 s wall, 16 MiB per file, and 8,000 retained\noutput characters.\n\nRLIMIT_NPROC is deliberately not used. On Linux it counts tasks for the whole\nUID, not descendants of this tool call; a busy notebook/vLLM user can already\nbe above a small limit and then lose even benign subprocess support. Aggregate\ntool memory is instead bounded by the notebook-wide tool-call semaphore.\n"""\nfrom __future__ import annotations\n\nimport json\nimport os\nimport signal\nimport subprocess\nimport sys\nimport tempfile\nimport threading\nimport time\nfrom dataclasses import dataclass\n\nDEFAULT_LIMITS = {\n    "address_space_bytes": 4 * 1024**3,\n    "cpu_seconds": 12,\n    "wall_seconds": 20,\n    "file_size_bytes": 16 * 1024**2,\n    "max_output_chars": 8000,\n    "max_open_files": 64,\n}\n\n_INTEGER_LIMITS = {\n    "address_space_bytes",\n    "cpu_seconds",\n    "file_size_bytes",\n    "max_output_chars",\n    "max_open_files",\n}\n\n# The child bootstrap sets rlimits before loading task context. User stdout and\n# stderr are written into a bounded stream, not an unbounded StringIO later\n# sliced down. A dedicated inherited fd carries one authenticated-by-structure\n# result record; this is protocol separation, not protection from hostile code.\n_CHILD = r\'\'\'\nimport sys, json, io, os, resource, contextlib\n\nlimits = json.loads(sys.argv[1])\nresult_fd = int(sys.argv[2])\n\nrlimits = [\n    ("RLIMIT_AS", "address_space_bytes"),\n    ("RLIMIT_CPU", "cpu_seconds"),\n    ("RLIMIT_FSIZE", "file_size_bytes"),\n    ("RLIMIT_NOFILE", "max_open_files"),\n]\nfor resource_name, key in rlimits:\n    resource_id = getattr(resource, resource_name, None)\n    if resource_id is None:\n        continue\n    value = int(limits[key])\n    try:\n        resource.setrlimit(resource_id, (value, value))\n    except (ValueError, OSError):\n        pass\nif hasattr(resource, "RLIMIT_CORE"):\n    try:\n        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))\n    except (ValueError, OSError):\n        pass\n\nclass CappedTextIO(io.TextIOBase):\n    encoding = "utf-8"\n\n    def __init__(self, cap):\n        self.cap = int(cap)\n        self.parts = []\n        self.retained = 0\n        self.truncated = False\n\n    def writable(self):\n        return True\n\n    def write(self, value):\n        if not isinstance(value, str):\n            value = str(value)\n        original_length = len(value)\n        remaining = self.cap - self.retained\n        if remaining > 0:\n            kept = value[:remaining]\n            self.parts.append(kept)\n            self.retained += len(kept)\n        if original_length > max(remaining, 0):\n            self.truncated = True\n        return original_length\n\n    def flush(self):\n        return None\n\n    def getvalue(self):\n        value = "".join(self.parts)\n        if self.truncated:\n            value += "\\n...[output truncated at %d chars]" % self.cap\n        return value\n\npayload = json.load(sys.stdin)\ng = {"__name__": "__main__"}\ng.update(payload.get("context") or {})\nbuf = CappedTextIO(limits["max_output_chars"])\nresult = {"ok": True, "error": None, "stdout": ""}\ntry:\n    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):\n        exec(compile(payload["code"], "<model_code>", "exec"), g)\nexcept MemoryError:\n    result["ok"] = False\n    result["error"] = "MemoryError: hit the memory cap"\nexcept SystemExit as exc:\n    result["ok"] = False\n    result["error"] = "SystemExit(%r)" % (exc.code,)\nexcept BaseException as exc:\n    result["ok"] = False\n    result["error"] = "%s: %s" % (type(exc).__name__, str(exc)[:800])\nresult["stdout"] = buf.getvalue()\nrecord = ("RESULT:" + json.dumps(result, separators=(",", ":"))).encode()\nos.write(result_fd, record)\n\'\'\'\n\n\n@dataclass\nclass ExecResult:\n    ok: bool\n    stdout: str\n    error: str | None\n    timed_out: bool\n    killed_signal: int | None\n    wall_s: float\n\n\ndef _normalize_limits(overrides: dict | None) -> dict:\n    if overrides is not None and not isinstance(overrides, dict):\n        raise TypeError("limits must be a mapping or None")\n    unknown = sorted(set(overrides or {}) - set(DEFAULT_LIMITS))\n    if unknown:\n        raise ValueError(f"unknown resource limit(s): {unknown}")\n    limits = {**DEFAULT_LIMITS, **(overrides or {})}\n    for name in _INTEGER_LIMITS:\n        value = limits[name]\n        if isinstance(value, bool) or not isinstance(value, int) or value <= 0:\n            raise ValueError(f"{name} must be a positive integer, got {value!r}")\n    wall = limits["wall_seconds"]\n    if isinstance(wall, bool) or not isinstance(wall, (int, float)) or wall <= 0:\n        raise ValueError(f"wall_seconds must be positive, got {wall!r}")\n    limits["wall_seconds"] = float(wall)\n    return limits\n\n\ndef _kill_process_group(process_group_id: int) -> None:\n    """Best-effort cleanup for the leader and descendants still in its session."""\n\n    try:\n        os.killpg(process_group_id, signal.SIGKILL)\n    except ProcessLookupError:\n        pass\n    except PermissionError:\n        # The session is ours, but fail safely if a platform/container policy\n        # unexpectedly blocks group signaling.\n        pass\n\n\ndef _child_environment() -> dict[str, str]:\n    """Return a minimal deterministic environment with one native thread."""\n\n    environment = {\n        "PATH": os.environ.get("PATH", ""),\n        "PYTHONHASHSEED": "0",\n        "OMP_NUM_THREADS": "1",\n        "OPENBLAS_NUM_THREADS": "1",\n        "MKL_NUM_THREADS": "1",\n        "NUMEXPR_NUM_THREADS": "1",\n    }\n    # Some installations need the dynamic-loader path to import NumPy wheels.\n    if os.environ.get("LD_LIBRARY_PATH"):\n        environment["LD_LIBRARY_PATH"] = os.environ["LD_LIBRARY_PATH"]\n    return environment\n\n\ndef _drain_result_pipe(\n    result_fd: int,\n    *,\n    retain_bytes: int,\n    chunks: list[bytes],\n    overflow: list[bool],\n) -> None:\n    """Drain the child protocol pipe without letting it block on a full pipe.\n\n    The child emits its result only after model code finishes. Waiting for the\n    child to exit before reading can therefore deadlock when JSON escaping\n    expands retained Unicode output beyond the platform pipe capacity. Keep a\n    bounded prefix for decoding, but continue draining/discarding until EOF so\n    the child can always finish its write.\n    """\n\n    retained = 0\n    try:\n        while True:\n            chunk = os.read(result_fd, 65536)\n            if not chunk:\n                return\n            remaining = retain_bytes - retained\n            if remaining > 0:\n                kept = chunk[:remaining]\n                chunks.append(kept)\n                retained += len(kept)\n            if len(chunk) > max(remaining, 0):\n                overflow[0] = True\n    except OSError:\n        # Parent teardown may close the descriptor after process cleanup. Any\n        # incomplete record is handled by the normal "no result" path.\n        return\n\n\ndef run_python(\n    code: str,\n    context: dict | None = None,\n    limits: dict | None = None,\n) -> ExecResult:\n    """Execute ``code`` in a fresh resource-bounded child process.\n\n    ``context`` values are JSON-serialized and become globals in the child.\n    The function is designed for trusted orchestration around untrusted model\n    snippets; it bounds resources but does not prevent data or network access.\n    """\n\n    if not isinstance(code, str):\n        raise TypeError("code must be a string")\n    if context is not None and not isinstance(context, dict):\n        raise TypeError("context must be a mapping or None")\n    lim = _normalize_limits(limits)\n    payload = json.dumps(\n        {"code": code, "context": context or {}}, separators=(",", ":")\n    ).encode()\n    started = time.monotonic()\n    r_fd, w_fd = os.pipe()\n    proc = None\n    process_group_id = None\n    raw = b""\n    timed_out = False\n    killed = None\n    reader = None\n    result_chunks: list[bytes] = []\n    result_overflow = [False]\n\n    try:\n        with tempfile.TemporaryDirectory(prefix="pytool-") as cwd:\n            try:\n                proc = subprocess.Popen(\n                    [\n                        sys.executable,\n                        "-I",\n                        "-c",\n                        _CHILD,\n                        json.dumps(lim, separators=(",", ":")),\n                        str(w_fd),\n                    ],\n                    stdin=subprocess.PIPE,\n                    stdout=subprocess.DEVNULL,\n                    stderr=subprocess.DEVNULL,\n                    pass_fds=(w_fd,),\n                    cwd=cwd,\n                    start_new_session=True,\n                    env=_child_environment(),\n                )\n                process_group_id = proc.pid  # start_new_session => pgid == pid\n            finally:\n                os.close(w_fd)\n\n            # ``ensure_ascii=True`` can encode one non-BMP output character as\n            # a 12-byte surrogate-pair escape. Retain enough for the largest\n            # legitimate record and drain any excess without allocating it.\n            max_result_bytes = lim["max_output_chars"] * 12 + 65536\n            reader = threading.Thread(\n                target=_drain_result_pipe,\n                kwargs={\n                    "result_fd": r_fd,\n                    "retain_bytes": max_result_bytes,\n                    "chunks": result_chunks,\n                    "overflow": result_overflow,\n                },\n                name="safe-python-result-drain",\n                daemon=True,\n            )\n            reader.start()\n\n            try:\n                proc.communicate(input=payload, timeout=lim["wall_seconds"])\n            except subprocess.TimeoutExpired:\n                timed_out = True\n                _kill_process_group(process_group_id)\n                proc.wait()\n            finally:\n                # Also run after a normal leader exit: snippets may fork a child\n                # and return while that child keeps consuming resources.\n                if process_group_id is not None:\n                    _kill_process_group(process_group_id)\n\n            if proc.returncode and proc.returncode < 0:\n                killed = -proc.returncode\n\n            reader.join(timeout=2)\n            if reader.is_alive():\n                # All writers should be gone after process-group cleanup. Close\n                # our read descriptor as a final unblock and treat the record as\n                # incomplete rather than hanging the notebook process.\n                try:\n                    os.close(r_fd)\n                except OSError:\n                    pass\n                reader.join(timeout=1)\n            raw = b"".join(result_chunks)\n    finally:\n        try:\n            os.close(r_fd)\n        except OSError:\n            pass\n        if proc is not None and proc.poll() is None:\n            if process_group_id is not None:\n                _kill_process_group(process_group_id)\n            proc.wait()\n\n    wall = round(time.monotonic() - started, 2)\n    if timed_out:\n        return ExecResult(\n            False,\n            "",\n            f"timeout: exceeded {lim[\'wall_seconds\']:g}s wall limit",\n            True,\n            killed,\n            wall,\n        )\n\n    marker = raw.split(b"RESULT:", 1)\n    if len(marker) == 2:\n        try:\n            result = json.loads(marker[1].decode())\n            return ExecResult(\n                bool(result["ok"]),\n                result.get("stdout", ""),\n                result.get("error"),\n                False,\n                killed,\n                wall,\n            )\n        except (KeyError, TypeError, UnicodeDecodeError, json.JSONDecodeError):\n            pass\n\n    reason = f"killed by signal {killed}" if killed else "child produced no result"\n    if result_overflow[0]:\n        reason = "child result exceeded parent protocol bound"\n    if killed == signal.SIGKILL:\n        reason = "killed (CPU/memory limit)"\n    return ExecResult(False, "", reason, False, killed, wall)\n\n\n# --- OpenAI tool schema the model sees -------------------------------------\nTOOL_SCHEMA = {\n    "type": "function",\n    "function": {\n        "name": "run_python",\n        "description": (\n            "Execute Python 3 code in a fresh, resource-limited sandbox and get "\n            "its stdout back. The ARC task is preloaded as globals: `train` (a "\n            "list of {\\"input\\": grid, \\"output\\": grid} pairs) and `test_input` "\n            "(the grid to solve); grids are lists of lists of ints 0-9. Import "\n            "anything you need (e.g. numpy). PRINT what you want to inspect -- only "\n            "stdout is returned. State does NOT persist between calls; include all "\n            "needed code each time. Limits: ~4 GiB RAM, ~12 CPU-s, 20 s wall. "\n            "This is resource isolation, not filesystem/network isolation."\n        ),\n        "parameters": {\n            "type": "object",\n            "properties": {\n                "code": {\n                    "type": "string",\n                    "description": "Python 3 source to run.",\n                }\n            },\n            "required": ["code"],\n            "additionalProperties": False,\n        },\n    },\n}\n\n\ndef _selftest() -> int:\n    cases = [\n        (\n            "benign",\n            "print(\'rows:\', len(test_input)); print(sum(sum(r) for r in test_input))",\n            {"test_input": [[1, 2], [3, 4]]},\n            {},\n            lambda result: result.ok and "10" in result.stdout,\n        ),\n        (\n            "memory_bomb",\n            "x = [0] * (10**11)",\n            {},\n            {},\n            lambda result: not result.ok,\n        ),\n        (\n            "infinite_loop",\n            "while True:\\n    pass",\n            {},\n            {},\n            lambda result: not result.ok,\n        ),\n        (\n            "output_bomb",\n            "print(\'x\' * (10**8))",\n            {},\n            {},\n            lambda result: result.ok and "[output truncated" in result.stdout,\n        ),\n        (\n            "unicode_pipe",\n            "print(\'😀\' * 6000)",\n            {},\n            {"wall_seconds": 3, "cpu_seconds": 2},\n            lambda result: result.ok\n            and not result.timed_out\n            and result.stdout.count("😀") == 6000,\n        ),\n        (\n            "numpy",\n            "import numpy as np; print(np.array(test_input).T.tolist())",\n            {"test_input": [[1, 2], [3, 4]]},\n            {},\n            lambda result: result.ok and "[[1, 3], [2, 4]]" in result.stdout,\n        ),\n        (\n            "user_error",\n            "raise ValueError(\'boom\')",\n            {},\n            {},\n            lambda result: not result.ok and "ValueError" in (result.error or ""),\n        ),\n    ]\n    failures = 0\n    for name, source, context, limit_overrides, predicate in cases:\n        result = run_python(\n            source,\n            context,\n            limits={\n                "wall_seconds": 8,\n                "cpu_seconds": 5,\n                **limit_overrides,\n            },\n        )\n        passed = bool(predicate(result))\n        failures += not passed\n        preview = (result.stdout or result.error or "").replace("\\n", " ")[:80]\n        print(\n            f"{name:14s} pass={passed!s:5s} ok={result.ok!s:5s} "\n            f"timed_out={result.timed_out!s:5s} "\n            f"wall={result.wall_s:5.2f}s -> {preview}"\n        )\n    if failures:\n        print(f"SELFTEST FAILED: {failures} case(s)", file=sys.stderr)\n        return 1\n    print("SELFTEST PASS")\n    return 0\n\n\nif __name__ == "__main__":\n    sys.exit(_selftest())\n'
SANDBOX_SHA256 = "85a10c0dc79caf670cf7f44e4de10bc57b554d5a72a205bd289950f9d581eeea"
RUN_STARTED_S = time.time()
RUN_STARTED_UTC = datetime.datetime.now(datetime.timezone.utc).isoformat()
RUN_ID = hashlib.sha256(
    (RUN_STARTED_UTC + ":" + str(os.getpid())).encode()
).hexdigest()[:16]

# ---- EXACT reproduced setting + budgeting knobs (surfaced for review/tuning) ----
# Up to 57,344 reasoning+output tokens, dynamically reduced so each exact prompt
# remains inside max_model_len=65,536. Public prompts reach ~16.4K tokenizer
# tokens, so a fixed 57,344 cap can be rejected before inference. Agent turns use
# a lower cap and recover_grid backstops a response that ends before its grid.
SETTINGS = json.loads(json.dumps(RUNTIME_CONFIG["sampling"]))
# ---- AGENTIC TOOL MODE (v11): xhigh reasoning + a python code interpreter ----
# The model MAY call run_python (task preloaded as `train`/`test_input`) to
# inspect grids, test a rule against every demo, and compute the answer -- the
# locally measured behavior is hypothesize->execute->verify->answer, which
# solved tasks xhigh-alone missed. Tool execs run in the embedded resource
# sandbox (throwaway subprocess, 4 GiB/12 CPU-s/20 s caps) -- fully offline.
# If the served vLLM lacks tool-call support this degrades LOUDLY to the
# proven direct free-form lane (TOOLS_OK False in the receipt).
TOOLS = bool(RUNTIME_CONFIG["agent"]["tools_enabled"])
TOOL_PARSER = RUNTIME_CONFIG["vllm"]["tool_parser"]
MAX_TURNS = int(RUNTIME_CONFIG["agent"]["max_turns"])
CONTEXT_LEN = int(RUNTIME_CONFIG["server"]["context_len"])
# The pinned tokenizer consumes reasoning_effort through chat_template_kwargs;
# vLLM rejects it as a top-level OpenAI request field. Every query gets the
# primary xhigh policy. Only after universal coverage may the medium policy use
# spare wall time to propose the diversity hypothesis for pass@2.
PASS_POLICIES = json.loads(json.dumps(RUNTIME_CONFIG["pass_policies"]))
CONFIGURED_PASS_POLICIES = json.loads(json.dumps(PASS_POLICIES))

def log(*a):
    print("[+%7.1fs]" % (time.time() - RUN_STARTED_S), *a, flush=True)

def log_event(event, **fields):
    payload = {
        "event": event,
        "run_id": RUN_ID,
        "elapsed_s": round(time.time() - RUN_STARTED_S, 1),
        **fields,
    }
    print("EVENT " + json.dumps(payload, sort_keys=True, default=str), flush=True)

MODE = resolve_run_mode(os.environ)
RERUN_RAW = MODE["environment"]["KAGGLE_IS_COMPETITION_RERUN"]
RERUN = MODE["competition_rerun"]
DIAGNOSTIC_MODEL_RUN = MODE["diagnostic_model_run"]
RUN_MODEL = MODE["run_model"]
PROBE = MODE["probe"]
TASK_LIMIT = MODE["task_limit"]
# Concurrency: the v7 probe measured max-num-seqs 24 as SAFE and effective on the
# 4xL4 -- vLLM sustained ~24 concurrent with ZERO preemption at 97.2% KV, ~255
# tok/s sustained (~1.6x v6's 158 at C=8). 97.2% KV is the ceiling for these long
# ~27K-token xhigh samples, so 24 is about the max for full xhigh.
CONCURRENCY = int(RUNTIME_CONFIG["scheduler"]["concurrency"])
# Pass 1 is universal xhigh+Python. Pass 2 is medium+Python and strictly
# spare-time: it cannot launch until every query completed pass 1, and the
# predictive deadline gate must still admit it. At most two model hypotheses
# are useful because the competition score is pass@2.
SAMPLES_PER_TASK = int(RUNTIME_CONFIG["scheduler"]["samples_per_query"])
FULL_RUN_TASK_TARGET = EXPECTED_COMPETITION_TASKS
FULL_RUN_QUERY_TARGET = EXPECTED_COMPETITION_QUERIES
# The floor is incremental and crash-durable, so reserve minutes rather than a
# fixed half-hour. New calls are admitted only when a rolling runtime quantile
# says they should finish before GENERATION_DEADLINE_S. A hard cutoff still
# terminates vLLM if a tail request violates that prediction.
HARD_WALL_S = RUN_STARTED_S + MODE["hard_wall_seconds"]
FINALIZATION_RESERVE_S = MODE["finalization_reserve_seconds"]
GENERATION_DEADLINE_S = HARD_WALL_S - FINALIZATION_RESERVE_S
COMPETITION_MODE = RUN_MODE_DEFINITIONS["competition_rerun"]
BUDGET_CONTRACT = schedule_budget_contract(
    query_target=FULL_RUN_QUERY_TARGET,
    concurrency=CONCURRENCY,
    episode_budget_s=PASS_POLICIES["xhigh"]["episode_budget_s"],
    hard_wall_s=COMPETITION_MODE["hard_wall_seconds"],
    reserve_s=COMPETITION_MODE["finalization_reserve_seconds"],
)
ACTIVE_MODE_WINDOW = {
    "mode_id": MODE["id"],
    "hard_wall_s": MODE["hard_wall_seconds"],
    "finalization_reserve_s": FINALIZATION_RESERVE_S,
    "generation_window_s": MODE["hard_wall_seconds"] - FINALIZATION_RESERVE_S,
}

OPERATOR_BANNER = {
    "mode_id": MODE["id"],
    "mode_label": MODE["label"],
    "challenge_name": MODE["challenge_name"],
    "run_model": RUN_MODEL,
    "direct_file_submittable": MODE["direct_file_submittable"],
    "model": MODEL_NAME if RUN_MODEL else None,
    "expected_vllm": EXPECTED_VLLM_VERSION if RUN_MODEL else None,
    "concurrency": CONCURRENCY if RUN_MODEL else 0,
    "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
    "generation_window_s": ACTIVE_MODE_WINDOW["generation_window_s"],
    "finalization_reserve_s": FINALIZATION_RESERVE_S,
}
log("=" * 72)
log("RUN CONTRACT:", MODE["label"])
log("challenge:", MODE["challenge_name"], "| model:",
    MODEL_NAME if RUN_MODEL else "OFF (validated floor only)")
log("output policy:",
    "eligible only after competition_full_run_pass + offline verifier"
    if MODE["direct_file_submittable"] else
    "NEVER manually upload this mode's submission.json")
log("runtime config:", RUNTIME_CONFIG_SHA256)
log("artifacts:", str(SUB), str(STATUS), str(RECEIPT), str(ACCEPTANCE))
log("=" * 72)
log_event("operator_banner", **OPERATOR_BANNER)
log_event(
    "mode_resolved",
    mode_id=MODE["id"],
    label=MODE["label"],
    challenge_name=MODE["challenge_name"],
    run_model=RUN_MODEL,
    task_limit=TASK_LIMIT,
    direct_file_submittable=MODE["direct_file_submittable"],
)
for warning in MODE["warnings"]:
    log("MODE WARNING:", warning)
log_event("competition_budget_contract", **BUDGET_CONTRACT)
log_event("active_mode_window", **ACTIVE_MODE_WINDOW)

# ---------- 1. L4x4 gate ----------
def gpu_names():
    try:
        out = subprocess.run(["nvidia-smi","--query-gpu=name","--format=csv,noheader"],
                             check=True, capture_output=True, text=True,
                             timeout=10).stdout
        return [l.strip() for l in out.splitlines() if l.strip()]
    except Exception:
        return []
names = gpu_names() if RUN_MODEL else []
log("GPUs:", names if RUN_MODEL else "not probed in floor-only mode")

# Sample driver-reported resident memory throughout model load + generation.
# A final snapshot alone misses the true peak; server metadata is not sufficient.
GPU_MEMORY_LOCK = threading.Lock()
GPU_MEMORY_STOP = threading.Event()
GPU_MEMORY_PEAKS = {}
GPU_MEMORY_SAMPLE_COUNT = [0]
GPU_MEMORY_ERRORS = []

def _sample_gpu_memory():
    try:
        out = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total",
             "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True, timeout=5,
        ).stdout
        rows = []
        for raw in out.splitlines():
            fields = [field.strip() for field in raw.split(",")]
            if len(fields) != 4:
                continue
            try:
                index, used, total = int(fields[0]), int(fields[2]), int(fields[3])
            except ValueError:
                continue
            rows.append({"index": index, "name": fields[1],
                         "peak_used_mib": used, "total_mib": total})
        with GPU_MEMORY_LOCK:
            GPU_MEMORY_SAMPLE_COUNT[0] += 1
            for row in rows:
                old = GPU_MEMORY_PEAKS.get(row["index"])
                if old is None or row["peak_used_mib"] > old["peak_used_mib"]:
                    GPU_MEMORY_PEAKS[row["index"]] = row
    except Exception as exc:
        with GPU_MEMORY_LOCK:
            if len(GPU_MEMORY_ERRORS) < 8:
                GPU_MEMORY_ERRORS.append(type(exc).__name__ + ":" + str(exc)[:160])

def _gpu_memory_loop():
    _sample_gpu_memory()
    while not GPU_MEMORY_STOP.wait(5.0):
        _sample_gpu_memory()

def gpu_memory_summary():
    if GPU_MEMORY_THREAD is None:
        return {"status": "not_requested", "sampler": "nvidia-smi-driver-memory",
                "interval_seconds": 5.0, "sample_count": 0,
                "devices": [], "errors": []}
    _sample_gpu_memory()
    with GPU_MEMORY_LOCK:
        devices = [dict(GPU_MEMORY_PEAKS[index]) for index in sorted(GPU_MEMORY_PEAKS)]
        count = int(GPU_MEMORY_SAMPLE_COUNT[0])
        errors = list(GPU_MEMORY_ERRORS)
    return {"status": "measured" if devices else "unavailable",
            "sampler": "nvidia-smi-driver-memory",
            "interval_seconds": 5.0,
            "sample_count": count, "devices": devices, "errors": errors}

def stop_gpu_memory_sampler():
    if GPU_MEMORY_THREAD is None:
        return
    GPU_MEMORY_STOP.set()
    GPU_MEMORY_THREAD.join(timeout=10)

GPU_MEMORY_THREAD = None

# ---------- 2. strict challenge snapshot + GUARANTEED FLOOR first ----------
# Import the single-source hardened finalizer embedded by the builder. Do not
# duplicate challenge/submission invariants in this kernel.
AGG_PATH = Path("/tmp/qwen38_freeform_aggregate_submission.py")
# The model's resource-bounded python code-interpreter (agentic tool mode).
SANDBOX_PATH = Path("/tmp/qwen38_safe_python_tool.py")
# The Python tool is optional. Delay importing it until after the complete
# pass@2 floor is on disk, so a damaged optional module cannot prevent the one
# artifact this notebook must always leave behind.
SANDBOX = None
SANDBOX_LOAD_ERROR = None
TOOLS_AVAILABLE = False
AGG = load_module_from_source(
    "qwen38_freeform_aggregator", AGG_PATH, AGGREGATOR_SOURCE)

# Each top-level call is capped at 4 GiB. Limit concurrent top-level calls to
# reduce pressure; this is not an aggregate descendant-memory cgroup.
PYTHON_TOOL_MAX_CONCURRENCY = int(
    RUNTIME_CONFIG["scheduler"]["python_tool_max_concurrency"])
PYTHON_TOOL_SLOTS = threading.BoundedSemaphore(PYTHON_TOOL_MAX_CONCURRENCY)

def run_python_tool(code, context, deadline_s):
    if SANDBOX is None:
        raise RuntimeError("Python tool requested before its sandbox loaded")
    started_s = time.time()
    wait_s = max(0.0, min(20.0, deadline_s - started_s))
    if wait_s <= 0 or not PYTHON_TOOL_SLOTS.acquire(timeout=wait_s):
        return SANDBOX.ExecResult(
            False, "", "tool concurrency queue timeout", True, None,
            round(time.time() - started_s, 2))
    try:
        wall_budget_s = tool_wall_budget_seconds(
            deadline_s,
            time.time(),
            SANDBOX.DEFAULT_LIMITS["wall_seconds"],
        )
        if wall_budget_s is None:
            return SANDBOX.ExecResult(
                False, "", "tool episode deadline expired in queue", True, None,
                round(time.time() - started_s, 2))
        return SANDBOX.run_python(
            code, context, limits={"wall_seconds": wall_budget_s})
    finally:
        PYTHON_TOOL_SLOTS.release()

STATUS_SEQUENCE = [0]
STATUS_ERRORS = []
SAMPLE_EVENT_SEQUENCE = [0]
SAMPLE_EVENT_WRITTEN = [0]
SAMPLE_EVENT_ERRORS = []
SAMPLE_EVENT_LOCK = threading.Lock()

def write_status(stage, **details):
    """Best-effort atomic heartbeat; never endanger the submission floor."""
    STATUS_SEQUENCE[0] += 1
    payload = {
        "schema": "arc-qwen38-freeform-status-v1",
        "run_id": RUN_ID,
        "sequence": STATUS_SEQUENCE[0],
        "stage": stage,
        "complete": stage == "complete",
        "updated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "elapsed_s": round(time.time() - RUN_STARTED_S, 1),
        "mode_id": MODE["id"],
        "run_model": RUN_MODEL,
        **details,
    }
    try:
        AGG._write_json(STATUS, payload)
    except Exception as exc:
        message = type(exc).__name__ + ":" + str(exc)[:200]
        if len(STATUS_ERRORS) < 8:
            STATUS_ERRORS.append(message)
        log("status heartbeat failed:", message)

def append_sample_event(payload):
    """Persist one compact sample record even if final receipt creation fails."""
    try:
        with SAMPLE_EVENT_LOCK:
            SAMPLE_EVENT_SEQUENCE[0] += 1
            record = {
                "schema": "arc-qwen38-freeform-sample-event-v2",
                "run_id": RUN_ID,
                "sequence": SAMPLE_EVENT_SEQUENCE[0],
                "utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
                "elapsed_s": round(time.time() - RUN_STARTED_S, 1),
                **payload,
            }
            with open(SAMPLE_EVENTS, "a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"))
                    + "\n"
                )
                handle.flush()
                os.fsync(handle.fileno())
            SAMPLE_EVENT_WRITTEN[0] += 1
    except Exception as exc:
        message = type(exc).__name__ + ":" + str(exc)[:200]
        if len(SAMPLE_EVENT_ERRORS) < 8:
            SAMPLE_EVENT_ERRORS.append(message)
        log("sample telemetry append failed:", message)

DEFAULT_EXCEPTHOOK = sys.excepthook

def record_unhandled_exception(exc_type, exc, traceback):
    """Leave an actionable heartbeat before rendering an uncaught error."""
    message = type(exc).__name__ + ":" + str(exc)[:500]
    try:
        write_status(
            "fatal_error",
            exception_type=getattr(exc_type, "__name__", str(exc_type)),
            error=message,
            submission_exists=SUB.is_file(),
            receipt_exists=RECEIPT.is_file(),
            acceptance_exists=ACCEPTANCE.is_file(),
        )
        log_event(
            "fatal_error",
            exception_type=getattr(exc_type, "__name__", str(exc_type)),
            error=message,
            submission_exists=SUB.is_file(),
        )
    except Exception as hook_error:
        log("fatal-error heartbeat failed:",
            type(hook_error).__name__, str(hook_error)[:200])
    DEFAULT_EXCEPTHOOK(exc_type, exc, traceback)

sys.excepthook = record_unhandled_exception

def archive_previous_run_artifacts():
    """Preserve prior evidence, but never reuse old predictions in a new run."""

    candidates = (SUB, RECEIPT, ACCEPTANCE, STATUS, SAMPLE_EVENTS, SERVER_LOG)
    existing = [
        path for path in candidates if path.exists() or path.is_symlink()
    ]
    if not existing:
        return {"status": "none", "path": None, "files": [], "errors": []}
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%S%fZ"
    )
    destination = PREVIOUS_RUNS / (
        stamp + "-" + RUN_ID[:8] + "-pid" + str(os.getpid())
    )
    moved, errors = [], []
    try:
        destination.mkdir(parents=True, exist_ok=False)
    except Exception as exc:
        errors.append("mkdir:%s:%s" % (type(exc).__name__, str(exc)[:160]))
        return {
            "status": "failed",
            "path": str(destination),
            "files": moved,
            "errors": errors,
        }
    for path in existing:
        try:
            # Keep the previous submission until the new floor is atomically
            # published; copy it into the evidence archive rather than moving.
            if path == SUB and path.is_file() and not path.is_symlink():
                shutil.copy2(path, destination / path.name)
            else:
                os.replace(path, destination / path.name)
            moved.append(path.name)
        except Exception as exc:
            errors.append(
                "%s:%s:%s" % (path.name, type(exc).__name__, str(exc)[:160])
            )
    return {
        "status": "archived" if moved and not errors else "partial",
        "path": str(destination),
        "files": moved,
        "errors": errors,
    }


PREVIOUS_RUN_ARCHIVE = archive_previous_run_artifacts()
if PREVIOUS_RUN_ARCHIVE["status"] != "none":
    log("previous run artifacts:", json.dumps(PREVIOUS_RUN_ARCHIVE, sort_keys=True))
for archive_error in PREVIOUS_RUN_ARCHIVE["errors"]:
    STATUS_ERRORS.append("previous_run_archive:" + archive_error)

def reset_runtime_artifact(path):
    """Clear a leftover artifact without following symlinks."""

    try:
        if path.is_dir() and not path.is_symlink():
            AGG._quarantine_blocking_directory(path)
        elif path.exists() or path.is_symlink():
            path.unlink(missing_ok=True)
    except Exception as exc:
        STATUS_ERRORS.append(
            "artifact_reset:%s:%s:%s" % (
                path.name, type(exc).__name__, str(exc)[:160]
            )
        )

for stale_path in (RECEIPT, ACCEPTANCE, STATUS, SAMPLE_EVENTS, SERVER_LOG):
    reset_runtime_artifact(stale_path)
try:
    SERVER_LOG.write_text("")
except Exception as exc:
    STATUS_ERRORS.append(
        "server_log_reset:%s:%s" % (type(exc).__name__, str(exc)[:160]))

challenge_name = MODE["challenge_name"]
write_status(
    "initializing",
    expected_challenge_name=challenge_name,
    task_limit=TASK_LIMIT,
    direct_file_submittable=MODE["direct_file_submittable"],
)
challenge_hits = sorted(glob.glob(f"/kaggle/input/**/{challenge_name}", recursive=True))
if len(challenge_hits) != 1:
    raise RuntimeError(
        f"expected exactly one challenge source for {challenge_name!r}; "
        f"found {challenge_hits!r}"
    )
ch_path = challenge_hits[0]
challenges, challenge_sha256 = AGG.load_public_challenges_with_digest(Path(ch_path))
challenge_query_count = sum(len(task["test"]) for task in challenges.values())
raw_competition_input_contract = competition_input_contract(
    len(challenges), challenge_query_count
)
if RERUN:
    COMPETITION_INPUT_CONTRACT = {
        **raw_competition_input_contract,
        "applicable": True,
        "status": (
            "pass" if raw_competition_input_contract["ok"] else "mismatch"
        ),
    }
else:
    # Public evaluation cardinality is intentionally different from the hidden
    # competition contract.  Report inventory without making a healthy saved
    # gate look like a failed hidden-input preflight.
    COMPETITION_INPUT_CONTRACT = {
        **raw_competition_input_contract,
        "applicable": False,
        "status": "not_applicable",
        "ok": None,
        "task_count_matches": None,
        "query_count_matches": None,
    }
MODEL_WORK_ELIGIBLE = RUN_MODEL and (
    not RERUN or raw_competition_input_contract["ok"]
)
log_event(
    "challenge_contract",
    mode_id=MODE["id"],
    model_work_eligible=MODEL_WORK_ELIGIBLE,
    **COMPETITION_INPUT_CONTRACT,
)
if RERUN:
    log(
        "competition challenge contract:",
        COMPETITION_INPUT_CONTRACT["status"],
        "| observed:", len(challenges), "tasks /", challenge_query_count,
        "queries",
    )
else:
    log(
        "challenge inventory:", len(challenges), "tasks /",
        challenge_query_count,
        "queries | hidden competition cardinality check: not applicable",
    )
task_ids = list(challenges)
if TASK_LIMIT is not None:
    task_ids = task_ids[:TASK_LIMIT]
log(
    "challenges:", ch_path,
    "| available tasks:", len(challenges),
    "| scheduled model tasks:", len(task_ids),
)

submission = AGG.guaranteed_fallback_submission(challenges)
fallback_submission = json.loads(json.dumps(submission))
def save_submission():
    # Never allow a malformed candidate to destroy the already-valid floor.
    AGG.validate_submission_completeness(challenges, submission)
    AGG._write_json(SUB, submission)
save_submission()            # a complete submission exists before any model work
log("guaranteed distinct pass@2 floor written:", SUB, "tasks:", len(submission))
write_status(
    "floor_written",
    challenge_path=ch_path,
    challenge_sha256=challenge_sha256,
    challenge_tasks=len(challenges),
    challenge_queries=challenge_query_count,
    competition_input_contract=COMPETITION_INPUT_CONTRACT,
    model_work_eligible=MODEL_WORK_ELIGIBLE,
    scheduled_tasks=len(task_ids),
    submission_tasks=len(submission),
)

# Optional dependency boundary: a sandbox import failure disables tool calls
# but preserves both the on-disk floor and the proven direct free-form lane.
if MODEL_WORK_ELIGIBLE and TOOLS:
    try:
        SANDBOX = load_module_from_source(
            "qwen38_safe_python_tool", SANDBOX_PATH, SANDBOX_SOURCE)
        TOOLS_AVAILABLE = True
        log_event("python_tool_sandbox", status="loaded", sha256=SANDBOX_SHA256)
    except Exception as exc:
        SANDBOX_LOAD_ERROR = type(exc).__name__ + ":" + str(exc)[:240]
        log("Python tool sandbox unavailable; direct lane remains:",
            SANDBOX_LOAD_ERROR)
        log_event(
            "python_tool_sandbox", status="unavailable",
            error=SANDBOX_LOAD_ERROR, sha256=SANDBOX_SHA256)
write_status(
    "optional_tools_ready",
    sandbox_requested=bool(MODEL_WORK_ELIGIBLE and TOOLS),
    sandbox_loaded=TOOLS_AVAILABLE,
    sandbox_load_error=SANDBOX_LOAD_ERROR,
)

if MODEL_WORK_ELIGIBLE and names:
    GPU_MEMORY_THREAD = threading.Thread(
        target=_gpu_memory_loop, name="qwen38-gpu-memory", daemon=True)
    GPU_MEMORY_THREAD.start()

# ---------- 3. install vLLM offline + serve FP8 TP4 ----------
MODEL_SOURCE_HINT = RUNTIME_CONFIG["model"]["source_hint"]
WHEELHOUSE_SOURCE_HINT = RUNTIME_CONFIG["vllm"]["wheelhouse_hint"]

def _select_unique_path(paths, hint, label):
    unique = sorted({str(Path(path)) for path in paths})
    normalized_hint = re.sub(r"[^a-z0-9]+", "", hint.lower())
    preferred = [
        path for path in unique
        if normalized_hint in re.sub(r"[^a-z0-9]+", "", path.lower())
    ]
    if len(preferred) == 1:
        return preferred[0]
    if not preferred:
        log(label, "pinned source hint not found; refusing candidates:", unique)
    else:
        log(label, "pinned source is ambiguous; candidates:", preferred)
    return None

def find_model_root():
    roots = []
    for cfg in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = Path(cfg).parent
        if list(d.glob("*.safetensors")):
            roots.append(str(d))
    return _select_unique_path(roots, MODEL_SOURCE_HINT, "model")

def find_wheelhouse():
    hits = glob.glob("/kaggle/input/**/vllm-*.whl", recursive=True) + glob.glob("/kaggle/input/**/vllm_*.whl", recursive=True)
    parents = [str(Path(hit).parent) for hit in hits]
    return _select_unique_path(parents, WHEELHOUSE_SOURCE_HINT, "wheelhouse")

def file_sha256(path):
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except Exception:
        return None
model_root, wheelhouse = (
    (find_model_root(), find_wheelhouse())
    if MODEL_WORK_ELIGIBLE else (None, None)
)
model_provenance = {
    "status": (
        "resolved" if model_root else
        "not_requested" if not RUN_MODEL else
        "ineligible" if not MODEL_WORK_ELIGIBLE else
        "unresolved"
    ),
    "source_hint": MODEL_SOURCE_HINT,
    "selected_root": model_root,
    "config_sha256": file_sha256(Path(model_root) / "config.json") if model_root else None,
    "tokenizer_config_sha256": file_sha256(Path(model_root) / "tokenizer_config.json") if model_root else None,
    "generation_config_sha256": file_sha256(Path(model_root) / "generation_config.json") if model_root else None,
}
runtime_versions = {
    "python": sys.version.split()[0],
    "vllm": None,
    "expected_vllm": EXPECTED_VLLM_VERSION,
}
served = False
server = None
CLEANED_SERVER_OBJECTS = set()
serving_status = {
    "mode_requested": RUN_MODEL,
    "requested": MODEL_WORK_ELIGIBLE,
    "model_work_eligible": MODEL_WORK_ELIGIBLE,
    "install_attempted": False,
    "install_returncode": None,
    "install_timed_out": False,
    "install_stderr_tail": None,
    "vllm_version_match": None,
    "version_gate_passed": False,
    "startup_attempts": [],
    "shutdown_events": [],
    "wheelhouse": wheelhouse,
    "wheelhouse_source_hint": WHEELHOUSE_SOURCE_HINT,
}

def terminate_server(reason):
    """Best-effort process-group cleanup for vLLM and all worker children."""
    if server is None or id(server) in CLEANED_SERVER_OBJECTS:
        return
    server_key = id(server)
    event = {"reason": reason, "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat()}
    try:
        try:
            os.killpg(server.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        except PermissionError:
            if server.poll() is None:
                server.terminate()
        if server.poll() is None:
            try:
                server.wait(timeout=20)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    server.kill()
                server.wait(timeout=10)
        # The API process can exit before one of its TP workers. Reap the group
        # even when poll() already has a return code.
        try:
            os.killpg(server.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        event["returncode"] = server.poll()
        event["ok"] = True
    except Exception as exc:
        event["ok"] = False
        event["error"] = type(exc).__name__ + ":" + str(exc)[:200]
    finally:
        CLEANED_SERVER_OBJECTS.add(server_key)
        serving_status["shutdown_events"].append(event)

def cleanup_server_at_exit():
    """Reap a live server group if an exception skips normal finalization."""
    try:
        terminate_server("atexit")
    except Exception:
        pass

atexit.register(cleanup_server_at_exit)

EXPECTED_GPU_COUNT = int(RUNTIME_CONFIG["hardware"]["gpu_count"])
EXPECTED_GPU_NAME = str(RUNTIME_CONFIG["hardware"]["gpu_name_contains"])
TENSOR_PARALLEL_SIZE = int(RUNTIME_CONFIG["hardware"]["tensor_parallel_size"])
if (MODEL_WORK_ELIGIBLE and model_root and wheelhouse
        and len(names) == EXPECTED_GPU_COUNT
        and all(EXPECTED_GPU_NAME.upper() in n.upper() for n in names)):
    os.environ["HF_HUB_OFFLINE"]="1"; os.environ["TRANSFORMERS_OFFLINE"]="1"
    serving_status["install_attempted"] = True
    try:
        ins = subprocess.run(
            [sys.executable,"-m","pip","install","--no-index","--no-warn-conflicts",
             "--disable-pip-version-check","--find-links",wheelhouse,
             "vllm==" + EXPECTED_VLLM_VERSION],
            capture_output=True, text=True,
            timeout=max(1, min(20 * 60, GENERATION_DEADLINE_S - time.time())),
        )
    except subprocess.TimeoutExpired as exc:
        ins = None
        serving_status["install_timed_out"] = True
        serving_status["install_stderr_tail"] = str(exc)[-1500:]
        log("vLLM install timed out; floor-only submission stands")
    if ins is not None:
        serving_status["install_returncode"] = ins.returncode
        serving_status["install_stderr_tail"] = ins.stderr[-1500:]
    if ins is not None and ins.returncode == 0:
        try:
            runtime_versions["vllm"] = importlib.metadata.version("vllm")
        except importlib.metadata.PackageNotFoundError:
            pass
        serving_status["vllm_version_match"] = (
            runtime_versions["vllm"] == EXPECTED_VLLM_VERSION)
        serving_status["version_gate_passed"] = bool(
            serving_status["vllm_version_match"])
        if not serving_status["vllm_version_match"]:
            log("VLLM VERSION MISMATCH: expected", EXPECTED_VLLM_VERSION,
                "but imported", runtime_versions["vllm"],
                "- refusing to serve; validated floor remains")
        os.environ["VLLM_USE_FLASHINFER_SAMPLER"]="0"
        # 65,536 context is the proven L4x4 FP8 fit. Request caps are adjusted so
        # prompt + output stays inside it. max-num-seqs equals client concurrency.
        base_serve = [sys.executable,"-m","vllm.entrypoints.openai.api_server",
                 "--model", model_root, "--served-model-name", MODEL_NAME,
                 "--tensor-parallel-size",str(TENSOR_PARALLEL_SIZE),
                 "--max-model-len",str(CONTEXT_LEN),
                 "--gpu-memory-utilization",
                 str(RUNTIME_CONFIG["server"]["gpu_memory_utilization"]),
                 "--enable-prefix-caching",
                 "--max-num-seqs", str(CONCURRENCY),
                 "--reasoning-parser",RUNTIME_CONFIG["vllm"]["reasoning_parser"],
                 "--language-model-only",
                 "--dtype","bfloat16","--port",str(PORT)]
        # The exact pinned tokenizer emits Qwen XML tool calls. vLLM's
        # qwen3_coder parser maps those to the OpenAI `tool_calls` structure.
        # If this build rejects the flags, retry without them; the live probe
        # below then prevents a false claim that Python was available.
        tool_flags = ["--enable-auto-tool-choice","--tool-call-parser",TOOL_PARSER]
        def ready(proc, deadline):
            while time.time() < deadline:
                if proc.poll() is not None: return False
                try:
                    with socket.create_connection(("127.0.0.1",PORT),timeout=3):
                        with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",timeout=5) as r:
                            if r.status < 500:
                                payload = json.loads(r.read().decode())
                                if model_list_contains_expected_model(
                                    payload, MODEL_NAME
                                ):
                                    return True
                except Exception: pass
                time.sleep(5)
            return False
        TOOL_FLAGS_ACTIVE = False
        attempts = (([tool_flags] if TOOLS else []) + [[]])
        if not serving_status["version_gate_passed"]:
            attempts = []
        for attempt_flags in attempts:
            serve = base_serve + attempt_flags
            log("serving:", " ".join(serve))
            attempt_started = time.time()
            try:
                with open(SERVER_LOG,"a") as srv_log:
                    server = subprocess.Popen(
                        serve, stdout=srv_log, stderr=subprocess.STDOUT,
                        start_new_session=True)
            except Exception as exc:
                serving_status["startup_attempts"].append({
                    "tool_flags": list(attempt_flags),
                    "served": False,
                    "elapsed_s": round(time.time() - attempt_started, 1),
                    "returncode": None,
                    "error": type(exc).__name__ + ":" + str(exc)[:200],
                })
                server = None
                log("server process launch failed:", type(exc).__name__, str(exc)[:200])
                continue
            served = ready(server, min(time.time()+1200, GENERATION_DEADLINE_S))
            serving_status["startup_attempts"].append({
                "tool_flags": list(attempt_flags),
                "served": served,
                "elapsed_s": round(time.time() - attempt_started, 1),
                "returncode": server.poll(),
            })
            if served:
                TOOL_FLAGS_ACTIVE = bool(attempt_flags)
                break
            log("server failed to start with flags:", attempt_flags or "(none)")
            terminate_server("startup_failed")
            server = None
        log("vLLM ready:", served, "| tool flags active:", TOOL_FLAGS_ACTIVE)
    elif ins is not None:
        log("vLLM install failed:", ins.stderr[-1500:])
else:
    if not RUN_MODEL:
        log("saved-version gate complete: model execution is intentionally NOT REQUESTED; pass@2 floor stands")
    elif not MODEL_WORK_ELIGIBLE:
        log("competition input contract mismatch; refusing model install/start. Floor stands.")
    else:
        log(
            "serving prerequisites not met (model/wheelhouse/%dx%s). "
            "Floor-only submission stands."
            % (EXPECTED_GPU_COUNT, EXPECTED_GPU_NAME)
        )
if "TOOL_FLAGS_ACTIVE" not in dir():
    TOOL_FLAGS_ACTIVE = False   # server never launched -> no tools either
write_status(
    "server_ready" if served else (
        "model_not_requested" if not RUN_MODEL else
        "model_work_ineligible" if not MODEL_WORK_ELIGIBLE else
        "server_unavailable"
    ),
    served=served,
    model_work_eligible=MODEL_WORK_ELIGIBLE,
    tool_flags_active=TOOL_FLAGS_ACTIVE,
    runtime_versions=runtime_versions,
    model_status=model_provenance["status"],
)

# ---------- 4. free-form direct-answer solve ----------
def render(g): return "\n".join(" ".join(str(v) for v in row) for row in g)
def build_prompt(task):
    parts = ["You are solving an abstract reasoning puzzle. Each example has an "
             "INPUT grid and its OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n"]
    for i, p in enumerate(task["train"]):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i+1, render(p["input"]), i+1, render(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render(task["test_input"]))
    parts.append("Reason step by step however you like. Then give ONLY the final "
                 "output grid as rows of space-separated digits inside a fenced "
                 "block:\n```grid\n<rows>\n```\nOutput nothing after the block.")
    return "\n".join(parts)
def validated_model_grid(grid):
    if grid is None:
        return None
    try:
        AGG.validate_grid(grid, context="freeform model grid")
    except Exception as exc:
        log("rejecting malformed model grid:", type(exc).__name__, str(exc)[:200])
        return None
    return grid

def request_timeout_s(default_s, deadline_s=None):
    # Bound each HTTP wait by its episode deadline as well as the global wall.
    deadline_s = min(deadline_s or GENERATION_DEADLINE_S, GENERATION_DEADLINE_S)
    remaining = deadline_s - time.time()
    if remaining <= 0:
        raise TimeoutError("request deadline already expired")
    return max(1, min(default_s, remaining))

def ask(prompt, seed, policy, deadline_s=None):
    # vLLM rejects reasoning_effort as a top-level OpenAI field. The pinned
    # tokenizer consumes it as a chat-template kwarg alongside enable_thinking.
    messages = [{"role":"user","content":prompt}]
    max_tokens = bounded_generation_cap(
        messages, policy["direct_max_tokens"], CONTEXT_LEN)
    deadline_s = min(
        deadline_s or GENERATION_DEADLINE_S,
        GENERATION_DEADLINE_S,
        time.time() + policy["episode_budget_s"],
    )
    t0=time.time()
    context_retries = 0
    while True:
        body = {"model": MODEL_NAME, "messages":messages,
                "temperature":SETTINGS["temperature"], "top_p":SETTINGS["top_p"],
                "top_k":SETTINGS["top_k"], "max_tokens":max_tokens, "seed":seed,
                "chat_template_kwargs":{"enable_thinking":SETTINGS["enable_thinking"],
                                        "reasoning_effort":policy["reasoning_effort"]}}
        req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
            data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
        try:
            with urllib.request.urlopen(
                req, timeout=request_timeout_s(3600, deadline_s)) as r:
                p = json.loads(r.read().decode())
            break
        except urllib.error.HTTPError as exc:
            try:
                error_body = exc.read().decode()[:600]
            except Exception:
                error_body = ""
            context_error = exc.code == 400 and (
                "context" in error_body.lower()
                or "maximum" in error_body.lower()
            )
            if context_error and max_tokens > 1024 and context_retries < 3:
                max_tokens = max(1024, max_tokens // 2)
                context_retries += 1
                log("direct request context retry:", context_retries,
                    "new max_tokens:", max_tokens)
                continue
            raise RuntimeError(
                "HTTP %d from chat endpoint: %s" % (exc.code, error_body)) from exc
    try:
        ch=p["choices"][0]; msg=ch["message"]
        if not isinstance(msg, dict):
            raise TypeError("message is not an object")
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("malformed chat completion response") from exc
    content = msg.get("content") or ""
    if not isinstance(content, str):
        content = ""
    finish = ch.get("finish_reason")
    return {"grid": validated_model_grid(parse_model_grid(
                content, require_closed_fence=finish == "length")),
            "finish": finish,
            "content": content, "reasoning": response_reasoning(msg),
            "completion_tokens": (p.get("usage") or {}).get("completion_tokens"),
            "wall_s": round(time.time()-t0,1), "request_max_tokens": max_tokens,
            "context_retries": context_retries}

def recover_grid(prompt, prior_reasoning, seed, deadline_s=None):
    """A sample truncated at the token limit did all the reasoning but never
    emitted the grid. Feed the tail of that reasoning back and force ONLY the
    grid out, thinking OFF (cheap) — recover the answer instead of wasting it."""
    messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": "[my reasoning, truncated at the token limit]\n" + prior_reasoning[-8000:]},
                {"role": "user", "content": "You hit the token limit mid-reasoning. Do NOT reason further. "
                                            "Based on the reasoning above, output ONLY your best-guess final "
                                            "grid now as rows of space-separated digits in a fenced block:\n"
                                            "```grid\n<rows>\n```"}]
    body = {"model": MODEL_NAME, "messages": messages,
            "temperature": SETTINGS["temperature"], "top_p": SETTINGS["top_p"], "top_k": SETTINGS["top_k"],
            "max_tokens": bounded_generation_cap(messages, 4096, CONTEXT_LEN),
            "seed": seed,
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=request_timeout_s(600, deadline_s)) as r:
        p = json.loads(r.read().decode())
    choice = p["choices"][0]
    finish = choice.get("finish_reason")
    return validated_model_grid(parse_model_grid(
        choice["message"].get("content") or "",
        require_closed_fence=finish == "length",
    ))

# ---------- 4b. AGENTIC MODE: xhigh reasoning + run_python tool ----------
def build_agent_prompt(task):
    parts = ["You are solving an abstract reasoning puzzle. Each training example maps "
             "an INPUT grid to an OUTPUT grid (digits 0-9 are colors; 0 is usually "
             "background). Infer the ONE rule that maps every input to its output, "
             "then apply it to the FINAL INPUT.\n",
             "You have a Python tool `run_python`. The task is preloaded there as "
             "`train` (a list of {\"input\": grid, \"output\": grid}) and `test_input` "
             "(the grid to solve). Use it however you find helpful -- to inspect grids, "
             "test a transformation against every example, or compute the answer -- or "
             "just reason directly. It is entirely your choice.\n"]
    for i, p in enumerate(task["train"]):
        parts.append("Example %d INPUT:\n%s\nExample %d OUTPUT:\n%s\n" %
                     (i+1, render(p["input"]), i+1, render(p["output"])))
    parts.append("FINAL INPUT:\n%s\n" % render(task["test_input"]))
    parts.append("When you are confident, give the final output grid as rows of "
                 "space-separated digits inside a fenced block:\n```grid\n<rows>\n```")
    return "\n".join(parts)

def _turn_cap(messages, policy):
    return bounded_generation_cap(
        messages, policy["turn_max_tokens"], CONTEXT_LEN)

def _agent_chat(messages, seed, max_tokens, reasoning_effort, deadline_s=None,
                allow_tools=True, enable_thinking=None):
    if enable_thinking is None:
        enable_thinking = SETTINGS["enable_thinking"]
    body = {"model": MODEL_NAME, "messages": messages,
            "temperature": SETTINGS["temperature"], "top_p": SETTINGS["top_p"],
            "top_k": SETTINGS["top_k"], "max_tokens": max_tokens, "seed": seed,
            "tools": [SANDBOX.TOOL_SCHEMA],
            "tool_choice": "auto" if allow_tools else "none",
            "chat_template_kwargs": {
                "enable_thinking": enable_thinking,
                "reasoning_effort": reasoning_effort,
            }}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
        data=json.dumps(body).encode(), headers={"Content-Type":"application/json"})
    with urllib.request.urlopen(req, timeout=request_timeout_s(3600, deadline_s)) as r:
        return json.loads(r.read().decode())

def probe_tool_support():
    """LIVE preflight: prove call, decode, execute, replay, and final answer."""
    result = {"ok": False, "tool_calls": 0, "sandbox_ok": False,
              "replay_final_ok": False, "error": None}
    if not TOOLS_AVAILABLE:
        result["error"] = "Python tool sandbox unavailable"
        return result
    if not (TOOLS and TOOL_FLAGS_ACTIVE):
        result["error"] = "tool flags inactive"
        return result
    try:
        deadline_s = min(GENERATION_DEADLINE_S, time.time() + 5 * 60)
        messages = [{"role":"user","content":
            "Call run_python exactly once to print 2+2, then reply with the result."}]
        p = _agent_chat(messages, seed=7, max_tokens=512,
                        reasoning_effort="low", deadline_s=deadline_s)
        msg = p["choices"][0]["message"]
        tool_calls = msg.get("tool_calls") or []
        result["tool_calls"] = len(tool_calls)
        if not tool_calls:
            if "<tool_call>" in response_reasoning(msg):
                raise RuntimeError(
                    "tool-call markup remained inside reasoning; parser did not expose tool_calls")
            raise RuntimeError("model emitted no tool call")
        messages.append(assistant_replay_message(msg))
        for tool_call in tool_calls:
            function = tool_call.get("function") or {}
            if function.get("name") != "run_python":
                raise RuntimeError("model requested an unknown tool")
            code = decode_tool_code(function)
            if not code:
                raise RuntimeError("tool arguments did not contain Python code")
            execution = run_python_tool(code, {}, deadline_s)
            if not execution.ok:
                raise RuntimeError("sandbox execution failed: " + str(execution.error))
            result["sandbox_ok"] = True
            messages.append({"role":"tool", "tool_call_id":tool_call.get("id"),
                             "name":function.get("name"),
                             "content":execution.stdout or "(no output)"})
        followup = _agent_chat(messages, seed=7, max_tokens=256,
                               reasoning_effort="low", deadline_s=deadline_s)
        final_message = followup["choices"][0]["message"]
        result["replay_final_ok"] = (
            not final_message.get("tool_calls")
            and "4" in (final_message.get("content") or ""))
        result["ok"] = result["sandbox_ok"] and result["replay_final_ok"]
        if not result["ok"]:
            result["error"] = "tool replay did not produce the expected final answer"
    except Exception as exc:
        result["error"] = type(exc).__name__ + ":" + str(exc)[:200]
    log("tool-support full-cycle probe:", json.dumps(result, sort_keys=True))
    return result

def solve_one_agent(tid, qi, test_input, slot, policy_name):
    """Agent loop: reason at the named tier, optionally run_python, emit a grid.
    Same result shape as ask() so record_sample/telemetry work unchanged."""
    task = {"train": challenges[tid]["train"], "test_input": test_input}
    policy = PASS_POLICIES[policy_name]
    prompt = build_agent_prompt(task)
    messages = [{"role":"user","content":prompt}]
    context = {"train": task["train"], "test_input": test_input}
    seed = stable_sample_seed(tid, qi, slot)
    t0 = time.time()
    episode_deadline_s = min(
        GENERATION_DEADLINE_S, t0 + policy["episode_budget_s"])
    total_tokens = 0; tool_calls_n = 0; turns = 0
    tool_successes = 0; tool_errors = 0
    context_retries = 0
    forced_final = False
    final = ""; reasoning_tail = ""; finish = "max_turns"
    best_grid = None
    recovery_attempted = False
    recovery_error = None
    last_request_max_tokens = None
    for _turn in range(MAX_TURNS):
        if time.time() >= episode_deadline_s:
            finish = "episode_budget"; break
        remaining_s = episode_deadline_s - time.time()
        force_final = (_turn == MAX_TURNS - 1 or (_turn > 0 and remaining_s < 8 * 60))
        if force_final:
            forced_final = True
            messages.append({"role":"user", "content":
                "Final answer turn: do not call tools or do more analysis. Using your work above, "
                "output only your best final grid in a ```grid fenced block."})
        cap = min(_turn_cap(messages, policy), 4096) if force_final else _turn_cap(messages, policy)
        if cap < 256:
            finish = "context_budget"; break
        p = None
        turn_context_retries = 0
        while True:
            last_request_max_tokens = cap
            try:
                p = _agent_chat(
                    messages, seed=seed, max_tokens=cap,
                    reasoning_effort=(
                        "low" if force_final else policy["reasoning_effort"]
                    ),
                    deadline_s=episode_deadline_s, allow_tools=not force_final,
                    enable_thinking=False if force_final else None)
                break
            except urllib.error.HTTPError as exc:
                body = ""
                try:
                    body = exc.read().decode()[:300]
                except Exception:
                    pass
                body_lower = body.lower()
                context_error = exc.code == 400 and (
                    "context" in body_lower or "maximum" in body_lower
                )
                if context_error and cap > 1024 and turn_context_retries < 3:
                    cap = max(1024, cap // 2)
                    context_retries += 1
                    turn_context_retries += 1
                    log("agent request context retry:", context_retries,
                        "new max_tokens:", cap)
                    continue
                finish = "http%d:%s" % (exc.code, body)
                break
            except Exception as exc:
                finish = "error:%s:%s" % (
                    type(exc).__name__, str(exc)[:160])
                break
        if p is None:
            break
        turns += 1
        try:
            ch = p["choices"][0]; msg = ch["message"]
            if not isinstance(msg, dict):
                raise TypeError("message is not an object")
        except (KeyError, IndexError, TypeError) as exc:
            finish = "protocol_error:%s" % type(exc).__name__
            break
        total_tokens += (p.get("usage") or {}).get("completion_tokens") or 0
        current_reasoning = response_reasoning(msg)
        if current_reasoning:
            reasoning_tail = current_reasoning
        message_content = msg.get("content") or ""
        if not isinstance(message_content, str):
            message_content = ""
        turn_finish = ch.get("finish_reason")
        candidate_grid = validated_model_grid(parse_model_grid(
            message_content,
            require_closed_fence=turn_finish == "length",
        ))
        if candidate_grid is not None:
            # Some servers/models emit a usable grid alongside a tool call. Keep
            # that partial work as a backstop rather than discarding it.
            best_grid = candidate_grid
        tcs = msg.get("tool_calls") or []
        if not isinstance(tcs, list):
            finish = "protocol_error:tool_calls_not_list"
            tool_errors += 1
            final = message_content
            break
        if not tcs and "<tool_call>" in current_reasoning:
            finish = "protocol_error:tool_call_markup_in_reasoning"
            tool_errors += 1
            final = message_content
            break
        if not tcs and "<tool_call>" in message_content:
            finish = "protocol_error:tool_call_markup_in_content"
            tool_errors += 1
            final = message_content
            break
        if force_final and tcs:
            # The explicit final turn is tool-free. Preserve any grid parsed
            # above, but never execute a surprise tool call inside the answer
            # reserve after the episode has entered finalization.
            finish = "protocol_error:tool_call_during_forced_final"
            tool_errors += len(tcs)
            final = message_content
            break
        if tcs:
            malformed = any(
                not isinstance(tc, dict)
                or not isinstance(tc.get("function"), dict)
                or not isinstance(tc.get("id"), str)
                or not tc.get("id")
                for tc in tcs
            )
            if malformed:
                finish = "protocol_error:malformed_tool_call"
                tool_errors += 1
                final = message_content
                break
            messages.append(assistant_replay_message(msg))
            for tc in tcs:
                fn = tc.get("function") or {}
                if fn.get("name") != "run_python":
                    out = "error: unknown tool %r" % fn.get("name")
                    tool_errors += 1
                else:
                    code = decode_tool_code(fn)
                    tool_calls_n += 1
                    if not code:
                        out = "ERROR: run_python arguments did not contain non-empty code"
                        tool_errors += 1
                    else:
                        res = run_python_tool(code, context, episode_deadline_s)
                        out = res.stdout if res.ok else ("ERROR: " + (res.error or ""))
                        if res.ok: tool_successes += 1
                        else: tool_errors += 1
                messages.append({"role":"tool","tool_call_id":tc.get("id"),
                                 "name":fn.get("name") or "unknown_tool",
                                 "content":out or "(no output)"})
            continue
        final = message_content
        finish = ch.get("finish_reason") or "stop"
        break
    grid = validated_model_grid(parse_model_grid(
        final, require_closed_fence=finish == "length")) or best_grid
    if grid is None and finish == "length" and reasoning_tail:
        recovery_attempted = True
        try:
            rg = recover_grid(prompt, reasoning_tail, seed, episode_deadline_s)
            if rg: grid = rg; finish = "length+recovered"
        except Exception as exc:
            recovery_error = type(exc).__name__ + ":" + str(exc)[:160]
    return {"grid": grid, "finish": finish, "content": final,
            "reasoning": reasoning_tail,
            "completion_tokens": total_tokens or None,
            "wall_s": round(time.time()-t0, 1),
            "method": "python-tool" if tool_calls_n else "direct",
            "tool_calls": tool_calls_n, "tool_successes": tool_successes,
            "tool_errors": tool_errors, "turns": turns,
            "context_retries": context_retries,
            "forced_final": forced_final,
            "recovery_attempted": recovery_attempted,
            "recovery_error": recovery_error,
            "policy": policy_name,
            "reasoning_effort": policy["reasoning_effort"],
            "turn_max_tokens": policy["turn_max_tokens"],
            "request_max_tokens": last_request_max_tokens}

per_task = {}
model_answered = set()
model_attempt_slots = {"attempt_1": set(), "attempt_2": set()}
observed_walls = []
observed_walls_by_policy = {name: [] for name in PASS_POLICIES}
# --- throughput probe accounting ---
GEN_START_S = [None]            # set when the first sample is launched (post-serve)
SAMPLE_LOG = []                 # (t_complete, completion_tokens) per finished sample
PEAK_CONCURRENCY = [0]          # max in-flight requests actually reached

def predicted_sample_s(policy_name):
    """Rolling near-wall admission cost, intentionally conservative at first.

    Until this run has enough measurements, reserve the complete episode cap.
    Thereafter use a padded empirical p90, which adapts to the real
    saved-version/rerun workload rather than launching blindly near the kill.
    """
    policy = PASS_POLICIES[policy_name]
    policy_walls = observed_walls_by_policy[policy_name]
    if len(policy_walls) < 8:
        return policy["initial_predicted_s"]
    s = sorted(policy_walls)
    q90 = s[max(0, math.ceil(0.90 * len(s)) - 1)]
    return max(10 * 60, min(policy["episode_budget_s"], 1.10 * q90 + 60))

def safe_to_launch_sample(policy_name):
    # The probe is a throughput measurement: keep the batch full until the hard
    # 1h cutoff kills the server (partial tail samples are expected and fine).
    if PROBE:
        return time.time() < GENERATION_DEADLINE_S
    return time.time() + predicted_sample_s(policy_name) < GENERATION_DEADLINE_S

def solve_one(tid, qi, test_input, slot, policy_name):
    t0 = time.time()
    prompt = build_prompt({"train": challenges[tid]["train"], "test_input": test_input})
    policy = PASS_POLICIES[policy_name]
    episode_deadline_s = min(
        GENERATION_DEADLINE_S, time.time() + policy["episode_budget_s"])
    try:
        if TOOLS_OK:
            # agentic lane handles its own HTTP/recovery internally
            return tid, qi, slot, solve_one_agent(
                tid, qi, test_input, slot, policy_name)
        o = ask(prompt, seed=stable_sample_seed(tid, qi, slot), policy=policy,
                deadline_s=episode_deadline_s)
        o["recovery_attempted"] = False
        o["recovery_error"] = None
        if o.get("finish") == "length" and o.get("grid") is None and o.get("reasoning"):
            o["recovery_attempted"] = True
            try:
                rg = recover_grid(
                    prompt,
                    o["reasoning"],
                    stable_sample_seed(tid, qi, slot),
                    episode_deadline_s,
                )
                if rg: o["grid"] = rg; o["finish"] = "length+recovered"
            except Exception as exc:
                o["recovery_error"] = type(exc).__name__ + ":" + str(exc)[:160]
    except urllib.error.HTTPError as exc:
        body = ""
        try: body = exc.read().decode()[:300]
        except Exception: pass
        o = {"grid": None, "finish": "http%d:%s" % (exc.code, body),
             "completion_tokens": None, "wall_s": round(time.time()-t0, 1),
             "error": "HTTPError", "recovery_attempted": False,
             "recovery_error": None}
    except Exception as exc:
        o = {"grid": None,
             "finish": "error:%s:%s" % (type(exc).__name__, str(exc)[:160]),
             "completion_tokens": None, "wall_s": round(time.time()-t0, 1),
             "error": type(exc).__name__, "recovery_attempted": False,
             "recovery_error": None}
    o.setdefault("policy", policy_name)
    o.setdefault("reasoning_effort", policy["reasoning_effort"])
    o.setdefault("turn_max_tokens", policy["direct_max_tokens"])
    return tid, qi, slot, o

def publish_model_grid(tid, qi, grid):
    """Transactionally publish one grid while preserving distinct pass@2."""
    key = (tid, qi)
    old_row = submission[tid][qi]
    if key not in model_answered:
        new_row = {
            "attempt_1": grid,
            "attempt_2": select_distinct_fallback(
                grid, fallback_submission[tid][qi]),
        }
        published_slot = "attempt_1"
    else:
        if grid == old_row["attempt_1"] or grid == old_row["attempt_2"]:
            return None
        new_row = {"attempt_1": old_row["attempt_1"], "attempt_2": grid}
        published_slot = "attempt_2"
    submission[tid][qi] = new_row
    try:
        save_submission()
    except BaseException:
        submission[tid][qi] = old_row
        raise
    model_answered.add(key)
    model_attempt_slots[published_slot].add(key)
    return published_slot

def record_sample(tid, qi, slot, o):
    """Publish a model answer without ever weakening submission validity."""
    grid = validated_model_grid(o.get("grid"))
    published_slot = None
    publish_error = None
    if grid is not None:
        try:
            published_slot = publish_model_grid(tid, qi, grid)
        except Exception as exc:
            publish_error = type(exc).__name__ + ":" + str(exc)[:200]
            log("model grid publish failed:", tid, qi, publish_error)
    wall = o.get("wall_s")
    if isinstance(wall, (int, float)) and wall > 0:
        observed_walls.append(float(wall))
        observed_walls_by_policy[o.get("policy") or "xhigh"].append(float(wall))
    ct = o.get("completion_tokens")
    if isinstance(ct, (int, float)) and ct > 0:
        SAMPLE_LOG.append((time.time(), int(ct)))
    row = {"slot": slot, "finish": o.get("finish"),
           "completion_tokens": o.get("completion_tokens"),
           "wall_s": wall, "grid": grid is not None,
           "published_slot": published_slot,
           "publish_error": publish_error,
           "method": o.get("method"), "tool_calls": o.get("tool_calls"),
           "tool_successes": o.get("tool_successes"),
           "tool_errors": o.get("tool_errors"),
           "turns": o.get("turns"), "forced_final": o.get("forced_final"),
           "recovery_attempted": o.get("recovery_attempted"),
           "recovery_error": o.get("recovery_error"),
           "error": o.get("error"),
           "policy": o.get("policy"),
           "reasoning_effort": o.get("reasoning_effort"),
           "turn_max_tokens": o.get("turn_max_tokens"),
           "request_max_tokens": o.get("request_max_tokens"),
           "context_retries": o.get("context_retries"),
           "grid_sha256": (hashlib.sha256(
               json.dumps(grid, separators=(",", ":")).encode()).hexdigest()
               if grid is not None else None)}
    per_task.setdefault(tid, {}).setdefault(qi, []).append(row)
    append_sample_event({"task_id": tid, "query_index": qi, **row})

def stop_server_for_deadline():
    if not served:
        return
    log("hard generation cutoff reached; terminating vLLM process group to release in-flight requests")
    terminate_server("generation_deadline")

def run_phase(work, *, phase_name, policy_name, total_work=None):
    """Run one scheduling phase with predictive admission and a hard wall."""
    if total_work is None:
        total_work = finite_work_size(work)
    iterator = iter(work)
    submitted = completed = failed = cancelled = 0
    exhausted = total_work == 0
    deadline_reached = False
    ex = ThreadPoolExecutor(max_workers=CONCURRENCY)
    futs = {}
    last_heartbeat_s = [0.0]

    def progress_fields(now=None):
        now = time.time() if now is None else now
        finished = completed + failed + cancelled
        remaining = (
            max(0, int(total_work) - finished)
            if total_work is not None else None
        )
        completion_pct = (
            round(100.0 * finished / int(total_work), 1)
            if total_work else None
        )
        return {
            "total_work": total_work,
            "finished": finished,
            "remaining": remaining,
            "completion_pct": completion_pct,
            "submitted": submitted,
            "completed": completed,
            "failed": failed,
            "cancelled": cancelled,
            "in_flight": len(futs),
            "model_attempt_1_published": len(model_attempt_slots["attempt_1"]),
            "model_attempt_2_published": len(model_attempt_slots["attempt_2"]),
            "predicted_next_s": round(predicted_sample_s(policy_name), 1),
            "seconds_remaining": round(GENERATION_DEADLINE_S - now, 1),
        }

    def consume_done(done_set):
        """Record every finished response, including work completed at cutoff."""
        nonlocal completed, failed, cancelled
        for fut in done_set:
            w = futs.pop(fut)
            if fut.cancelled():
                cancelled += 1
                continue
            try:
                tid, qi, slot, o = fut.result()
                record_sample(tid, qi, slot, o)
                completed += 1
            except Exception as exc:
                failed += 1
                log("sample failed:", w[0], type(exc).__name__, str(exc)[:160])
                append_sample_event({
                    "task_id": w[0], "query_index": w[1], "slot": w[3],
                    "policy": w[4], "grid": False,
                    "finish": "future_error:%s:%s" % (
                        type(exc).__name__, str(exc)[:160]),
                })

    try:
        def maybe_launch():
            nonlocal submitted, exhausted
            while (len(futs) < CONCURRENCY and not exhausted
                   and safe_to_launch_sample(policy_name)):
                try:
                    w = next(iterator)
                except StopIteration:
                    exhausted = True
                    return
                if GEN_START_S[0] is None:
                    GEN_START_S[0] = time.time()
                futs[ex.submit(solve_one, *w)] = w
                submitted += 1
                if total_work is not None and submitted >= total_work:
                    # A finite source is exhausted as soon as its final item is
                    # admitted. Do not require an extra next() probe that a
                    # deadline gate may prevent after full submission.
                    exhausted = True
            if len(futs) > PEAK_CONCURRENCY[0]:
                PEAK_CONCURRENCY[0] = len(futs)

        maybe_launch()
        write_status(
            "phase_started", phase=phase_name, policy=policy_name,
            **progress_fields())
        log_event(
            "phase_progress", stage="started", phase=phase_name,
            policy=policy_name, **progress_fields())
        last_heartbeat_s[0] = time.time()
        while futs:
            now = time.time()
            if now - last_heartbeat_s[0] >= int(
                RUNTIME_CONFIG["scheduler"]["progress_interval_s"]
            ):
                write_status(
                    "phase_running", phase=phase_name, policy=policy_name,
                    **progress_fields(now))
                log_event(
                    "phase_progress", stage="running", phase=phase_name,
                    policy=policy_name, **progress_fields(now))
                last_heartbeat_s[0] = now
            if time.time() >= GENERATION_DEADLINE_S:
                stop_server_for_deadline()
                deadline_reached = True
                break
            done_set, _ = wait(
                list(futs), timeout=min(5, max(0.1, GENERATION_DEADLINE_S-time.time())),
                return_when=FIRST_COMPLETED,
            )
            if not done_set:
                continue
            consume_done(done_set)
            if completed and completed % 20 == 0:
                log("  %s: %d completed / %d submitted; %.1fh left; predicted next %.1fmin" %
                     (phase_name, completed, submitted,
                      (GENERATION_DEADLINE_S-time.time())/3600,
                     predicted_sample_s(policy_name)/60))
            maybe_launch()
    finally:
        # At a hard cutoff vLLM has been terminated above, so running HTTP calls
        # fail promptly. Cancel work that never started, reap the pool, and then
        # consume all completed responses so useful grids/telemetry are not lost
        # merely because they crossed the scheduler's final deadline check.
        for fut in futs:
            fut.cancel()
        ex.shutdown(wait=True, cancel_futures=True)
        consume_done(set(futs))

    result = {
        "total_work": total_work,
        "submitted": submitted,
        "completed": completed,
        "failed": failed,
        "cancelled": cancelled,
        "exhausted": exhausted,
        "deadline_reached": deadline_reached,
    }
    write_status(
        "phase_complete", phase=phase_name, policy=policy_name,
        **{**progress_fields(), **result})
    log_event(
        "phase_progress", stage="complete", phase=phase_name,
        policy=policy_name, **progress_fields(), exhausted=exhausted,
        deadline_reached=deadline_reached)
    return result

def attempt2_risk_key(item):
    tid, qi, *_ = item
    rows = per_task.get(tid, {}).get(qi, [])
    first = next((row for row in rows if row.get("slot") == 0), None)
    if first is None:
        return (5, 0, 0, 0)
    finish = str(first.get("finish") or "")
    no_grid = not first.get("grid")
    instability = (
        finish != "stop"
        or bool(first.get("forced_final"))
        or int(first.get("tool_errors") or 0) > 0
        or int(first.get("context_retries") or 0) > 0
    )
    return (
        4 if no_grid else (3 if instability else 1),
        int(first.get("tool_errors") or 0)
        + int(first.get("context_retries") or 0),
        int(first.get("completion_tokens") or 0),
        float(first.get("wall_s") or 0),
    )

def empty_phase_result():
    return {
        "total_work": 0, "submitted": 0, "completed": 0,
        "failed": 0, "cancelled": 0,
        "exhausted": True, "deadline_reached": False,
    }

# LIVE tool preflight: agentic mode only when the endpoint proves tool_calls.
if served:
    TOOL_PROBE = probe_tool_support()
elif not RUN_MODEL:
    TOOL_PROBE = {
        "status": "not_applicable", "ok": False, "tool_calls": 0,
        "sandbox_ok": False, "replay_final_ok": False,
        "error": None, "reason": "mode_does_not_run_model"}
elif not MODEL_WORK_ELIGIBLE:
    TOOL_PROBE = {
        "status": "ineligible", "ok": False, "tool_calls": 0,
        "sandbox_ok": False, "replay_final_ok": False,
        "error": None, "reason": "competition_input_contract_mismatch"}
else:
    TOOL_PROBE = {
        "status": "unavailable", "ok": False, "tool_calls": 0,
        "sandbox_ok": False, "replay_final_ok": False,
        "error": "server unavailable"}
TOOLS_OK = bool(TOOL_PROBE["ok"])
if not RUN_MODEL:
    log("AGENTIC MODE: not applicable in saved-version floor-only mode")
elif not MODEL_WORK_ELIGIBLE:
    log("AGENTIC MODE: ineligible because the competition input contract mismatched")
else:
    log("AGENTIC MODE:", "ON (xhigh + run_python)" if TOOLS_OK else
        "OFF -> direct free-form fallback")

COVERAGE_BUDGET_ADJUSTMENT = {
    "applied": False,
    "query_count": 0,
    "waves": 0,
    "configured_episode_budget_s": CONFIGURED_PASS_POLICIES["xhigh"][
        "episode_budget_s"],
    "effective_episode_budget_s": PASS_POLICIES["xhigh"]["episode_budget_s"],
    "remaining_generation_s_at_adjustment": None,
    "per_wave_overhead_s": 30,
}

if served and task_ids and PROBE:
    # 1-HOUR THROUGHPUT PROBE: keep the batch full at C=CONCURRENCY by cycling the
    # available queries with an increasing slot so we saturate the hardware and
    # measure real aggregate tokens/s + the safe KV ceiling. Ends at the 1h wall.
    queries = [(tid, qi, q["input"]) for tid in task_ids for qi, q in enumerate(challenges[tid]["test"])]
    log("PROBE: saturating C=%d over %d queries (cycling slots) until the 1h wall; cap=%d tok" %
        (CONCURRENCY, len(queries), SETTINGS["max_tokens"]))
    def _probe_work():
        slot = 0
        while True:
            for (tid, qi, inp) in queries:
                yield (tid, qi, inp, slot, "xhigh")
            slot += 1
    phase1 = run_phase(_probe_work(), phase_name="probe", policy_name="xhigh")
    phase2 = empty_phase_result()
    save_submission()
elif served and task_ids:
    queries = [(tid, qi, q["input"]) for tid in task_ids for qi, q in enumerate(challenges[tid]["test"])]
    remaining_generation_s = max(0, GENERATION_DEADLINE_S - time.time())
    effective_budget_s = coverage_episode_budget_seconds(
        remaining_generation_s,
        len(queries),
        CONCURRENCY,
        CONFIGURED_PASS_POLICIES["xhigh"]["episode_budget_s"],
        per_wave_overhead_s=COVERAGE_BUDGET_ADJUSTMENT["per_wave_overhead_s"],
    )
    PASS_POLICIES["xhigh"]["episode_budget_s"] = effective_budget_s
    PASS_POLICIES["xhigh"]["initial_predicted_s"] = min(
        PASS_POLICIES["xhigh"]["initial_predicted_s"], effective_budget_s)
    COVERAGE_BUDGET_ADJUSTMENT.update({
        "applied": (
            effective_budget_s
            < CONFIGURED_PASS_POLICIES["xhigh"]["episode_budget_s"]),
        "query_count": len(queries),
        "waves": math.ceil(len(queries) / CONCURRENCY),
        "effective_episode_budget_s": effective_budget_s,
        "remaining_generation_s_at_adjustment": round(remaining_generation_s, 1),
    })
    log("PRIMARY COVERAGE BUDGET:", json.dumps(
        COVERAGE_BUDGET_ADJUSTMENT, sort_keys=True))
    log("solving: xhigh+Python attempt_1 coverage for all %d queries; C=%d, cap=%d tok, generation wall in %.1fh" %
        (len(queries), CONCURRENCY, PASS_POLICIES["xhigh"]["turn_max_tokens"],
         (GENERATION_DEADLINE_S - time.time())/3600))
    phase1 = run_phase(
        [(tid, qi, inp, 0, "xhigh") for (tid, qi, inp) in queries],
        phase_name="attempt_1_xhigh", policy_name="xhigh")

    # Never spend on a second hypothesis until every query got its first model
    # launch. Then spend the bank on the shakiest label-free cases first:
    # missing answer > abnormal/recovered/forced/tool-error answer > expensive
    # normal completion. This changes only pass-2 ordering, never pass-1 coverage.
    phase2 = empty_phase_result()
    if SAMPLES_PER_TASK < 2:
        log("pass@1: one xhigh attempt per query; attempt_2 stays the robust floor")
    elif (phase1["exhausted"] and phase1["completed"] == len(queries)
          and safe_to_launch_sample("medium")):
        log("xhigh coverage complete; spending only safe spare wall on risk-ordered medium+Python attempt_2")
        ordered = sorted(queries, key=attempt2_risk_key, reverse=True)
        phase2 = run_phase(
            [(tid, qi, inp, 1, "medium") for (tid, qi, inp) in ordered],
            phase_name="attempt_2_medium", policy_name="medium")
    else:
        log("no attempt_2 bank: attempt_1 coverage incomplete or no safely-admissible sample remains")
    save_submission()
else:
    phase1 = empty_phase_result()
    phase2 = empty_phase_result()

# ---------- 5. finalize + budgeting receipt ----------
GENERATION_FINISHED_S = time.time()
write_status(
    "finalizing", phase1=phase1, phase2=phase2,
    model_backed_attempt_1=len(model_attempt_slots["attempt_1"]),
    model_backed_attempt_2=len(model_attempt_slots["attempt_2"]))
if server is not None:
    terminate_server("normal_finalization")

walls = [o["wall_s"] for t in per_task.values() for q in t.values() for o in q if o.get("wall_s")]
toks = [o["completion_tokens"] for t in per_task.values() for q in t.values() for o in q if o.get("completion_tokens")]
sample_rows = [o for task in per_task.values() for query in task.values() for o in query]
tool_call_total = sum(int(row.get("tool_calls") or 0) for row in sample_rows)
tool_success_total = sum(int(row.get("tool_successes") or 0) for row in sample_rows)
tool_error_total = sum(int(row.get("tool_errors") or 0) for row in sample_rows)
valid_model_grid_total = sum(row.get("grid") is True for row in sample_rows)
published_model_grid_total = sum(row.get("published_slot") is not None for row in sample_rows)
publication_error_total = sum(row.get("publish_error") is not None for row in sample_rows)
duplicate_attempt_pairs = sum(
    row["attempt_1"] == row["attempt_2"]
    for task_rows in submission.values() for row in task_rows)
if duplicate_attempt_pairs:
    raise RuntimeError("final submission unexpectedly contains duplicate pass@2 attempts")

def _server_log_stats():
    """Parse scheduler telemetry, including tokens still in flight."""
    try:
        text = SERVER_LOG.read_text(errors="ignore")
    except Exception:
        text = ""
    return parse_vllm_scheduler_log(text)

gpu_memory = gpu_memory_summary()
stop_gpu_memory_sampler()

def _throughput():
    finished_tokens = sum(ct for _, ct in SAMPLE_LOG)
    window = None
    if GEN_START_S[0] is not None:
        window = max(1e-6, GENERATION_FINISHED_S - GEN_START_S[0])
    finished_lower_bound = (round(finished_tokens / window, 1)
                            if window else None)
    server_stats = _server_log_stats()
    scheduler_rate = server_stats.get("decode_tokens_per_second")
    aggregate_rate = scheduler_rate or finished_lower_bound
    rate_source = ("vllm_scheduler_log_mean" if scheduler_rate
                   else ("finished_responses_lower_bound"
                         if finished_lower_bound else None))
    estimated_all_tokens = (round(aggregate_rate * window)
                            if aggregate_rate and window else finished_tokens)
    primary_toks = [
        o["completion_tokens"] for t in per_task.values() for q in t.values()
        for o in q if o.get("policy") == "xhigh" and o.get("completion_tokens")
    ]
    mean_ct = (sum(primary_toks) / len(primary_toks)) if primary_toks else None
    projection_h = None
    if aggregate_rate and mean_ct:
        projection_h = round(
            FULL_RUN_QUERY_TARGET * mean_ct / aggregate_rate / 3600, 2)
    return {
        "measured_aggregate_tokens_per_s": aggregate_rate,
        "throughput_measurement_source": rate_source,
        "lower_bound_finished_response_tokens_per_s": finished_lower_bound,
        "prefill_tokens_per_second": server_stats["prefill_tokens_per_second"],
        "decode_tokens_per_second": server_stats["decode_tokens_per_second"],
        "peak_concurrency_observed": PEAK_CONCURRENCY[0],
        "max_num_seqs_configured": CONCURRENCY,
        "wall_seconds": round(window, 1) if window else None,
        "generation_window_s": round(window, 1) if window else None,
        "finished_completion_tokens": finished_tokens,
        "estimated_generated_tokens_including_inflight": estimated_all_tokens,
        "generated_token_accounting": (
            "scheduler-rate estimate includes in-flight work; "
            "finished_completion_tokens is the exact completed-response floor"),
        "samples_finished": len(SAMPLE_LOG),
        "projection_samples_per_task": 1,
        "full_run_query_target": FULL_RUN_QUERY_TARGET,
        "projected_full_xhigh_wall_h_at_this_rate": projection_h,
        "projected_full_medium_wall_h_at_this_rate": None,
        "fits_12h_wall": (projection_h is not None and projection_h <= 11.5),
        "peak_per_device_memory_mib": gpu_memory["devices"],
        "gpu_memory_sampler": gpu_memory,
        **server_stats,
        "note": "Use the scheduler-log mean for saturated probes because it "
                "includes running requests; finished-response usage alone is "
                "only a lower bound. Projection covers the universal xhigh pass; "
                "the medium pass is opportunistic and is not projected as universal.",
    }
throughput = _throughput()
if RUN_MODEL:
    log("THROUGHPUT:", json.dumps(throughput, indent=1))
else:
    log("THROUGHPUT: not applicable in saved-version floor-only mode")
n_queries_total = challenge_query_count
primary_entries = [
    (tid, qi, row)
    for tid, query_map in per_task.items()
    for qi, rows in query_map.items()
    for row in rows
    if row.get("slot") == 0
]
primary_rows = [row for _, _, row in primary_entries]
primary_query_keys = [(str(tid), str(qi)) for tid, qi, _ in primary_entries]
primary_unique_queries = len(set(primary_query_keys))
primary_duplicate_records = len(primary_query_keys) - primary_unique_queries
expected_query_keys = {
    (str(tid), str(qi))
    for tid, rows in submission.items()
    for qi, _ in enumerate(rows)
}
primary_expected_query_coverage = set(primary_query_keys) == expected_query_keys
primary_xhigh_records = sum(
    row.get("policy") == "xhigh" for row in primary_rows
)
primary_wrong_policy_records = len(primary_rows) - primary_xhigh_records
primary_valid_grids = sum(row.get("grid") is True for row in primary_rows)
primary_anomaly_reason_counts = {}
primary_anomaly_count = 0
for row in primary_rows:
    reasons = primary_sample_anomaly_reasons(row)
    if reasons:
        primary_anomaly_count += 1
        for reason in reasons:
            primary_anomaly_reason_counts[reason] = (
                primary_anomaly_reason_counts.get(reason, 0) + 1)

# Revalidate the exact on-disk artifact before hashing and classifying the run.
save_submission()
AGG.validate_submission_completeness(challenges, json.loads(SUB.read_text()))
artifact_hashes = {
    "submission_sha256": file_sha256(SUB),
    "sample_events_sha256": file_sha256(SAMPLE_EVENTS),
    "server_log_sha256": file_sha256(SERVER_LOG),
}
acceptance_inputs = {
    "mode_id": MODE["id"],
    "challenge_name": MODE["challenge_name"],
    "direct_file_submittable": MODE["direct_file_submittable"],
    "submission_valid": True,
    "submission_sha256": artifact_hashes["submission_sha256"],
    "challenge_sha256": challenge_sha256,
    "n_tasks_total": len(challenges),
    "n_queries_total": n_queries_total,
    "expected_task_target": FULL_RUN_TASK_TARGET,
    "expected_query_target": FULL_RUN_QUERY_TARGET,
    "run_model": RUN_MODEL,
    "model_work_eligible": MODEL_WORK_ELIGIBLE,
    "served": served,
    "vllm_version_match": serving_status["vllm_version_match"],
    "tool_flags_active": TOOL_FLAGS_ACTIVE,
    "tool_probe_ok": TOOLS_OK,
    "phase1": phase1,
    "phase2": phase2,
    "primary_sample_records": len(primary_rows),
    "primary_unique_queries": primary_unique_queries,
    "primary_duplicate_records": primary_duplicate_records,
    "primary_expected_query_coverage": primary_expected_query_coverage,
    "primary_xhigh_records": primary_xhigh_records,
    "primary_wrong_policy_records": primary_wrong_policy_records,
    "primary_valid_grids": primary_valid_grids,
    "primary_anomaly_count": primary_anomaly_count,
    "primary_anomaly_reason_counts": primary_anomaly_reason_counts,
    "sample_record_count": len(sample_rows),
    "sample_event_records": SAMPLE_EVENT_WRITTEN[0],
    "sample_event_write_errors": len(SAMPLE_EVENT_ERRORS),
    "duplicate_attempt_pairs": duplicate_attempt_pairs,
    "publication_errors": publication_error_total,
    "status_write_errors": len(STATUS_ERRORS),
    "sample_events_sha256": artifact_hashes["sample_events_sha256"],
    "server_log_sha256": artifact_hashes["server_log_sha256"],
}
acceptance = evaluate_run_acceptance(acceptance_inputs)
if acceptance["accepted_for_manual_upload"]:
    operator_action = {
        "code": "VERIFY_THEN_UPLOAD_COMPETITION_OUTPUT",
        "headline": "COMPETITION OUTPUT ELIGIBLE AFTER OFFLINE VERIFIER",
        "instruction": (
            "Download the complete output directory and require "
            "competition_full_run_pass with verify_run_artifacts.py before upload."
        ),
    }
elif acceptance["outcome"] == "saved_version_gate_pass":
    operator_action = {
        "code": "WEB_UI_SUBMIT_TO_COMPETITION",
        "headline": "SAVED GATE VERIFIED",
        "instruction": (
            "Verify this saved output with --require-saved-gate, then use Kaggle's "
            "authenticated Submit to Competition menu for this exact version. "
            "Never use kaggle competitions submit --kernel."
        ),
    }
else:
    operator_action = {
        "code": "DO_NOT_UPLOAD",
        "headline": "DO NOT UPLOAD THIS OUTPUT",
        "instruction": (
            "Preserve all artifacts for diagnosis; failed acceptance checks are "
            "listed in freeform_acceptance.json."
        ),
    }
log_event(
    "run_acceptance",
    outcome=acceptance["outcome"],
    accepted_for_manual_upload=acceptance["accepted_for_manual_upload"],
    full_run_proven=acceptance["full_run_proven"],
    failed_checks=acceptance["failed_checks"],
    warnings=acceptance["warnings"],
    operator_action=operator_action["code"],
)
receipt = {
    "schema": "arc-qwen38-freeform-direct-v5",
    "status": acceptance["outcome"],
    "execution_complete": True,
    "run_id": RUN_ID,
    "runtime_config": RUNTIME_CONFIG,
    "runtime_config_sha256": RUNTIME_CONFIG_SHA256,
    "operator_action": operator_action,
    "acceptance": acceptance,
    "acceptance_inputs": acceptance_inputs,
    "artifact_hashes": artifact_hashes,
    "run_started_utc": RUN_STARTED_UTC,
    "run_finished_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "elapsed_s": round(time.time() - RUN_STARTED_S, 1),
    "mode": MODE,
    "rerun": RERUN, "rerun_env_raw": RERUN_RAW,
    "run_model": RUN_MODEL,
    "model_work_eligible": MODEL_WORK_ELIGIBLE,
    "diagnostic_model_run": DIAGNOSTIC_MODEL_RUN,
    "minimal_saved_version": not RUN_MODEL,
    "probe": bool(PROBE), "gpus": names, "served": served,
    "runtime_versions": runtime_versions,
    "competition_input_contract": COMPETITION_INPUT_CONTRACT,
    "model_provenance": model_provenance,
    "serving": serving_status,
    "agentic_tools": {
        "configured": TOOLS,
        "mode_requested": bool(RUN_MODEL and TOOLS),
        "requested": bool(MODEL_WORK_ELIGIBLE and TOOLS),
        "sandbox_loaded": TOOLS_AVAILABLE,
        "sandbox_load_error": SANDBOX_LOAD_ERROR,
        "server_tool_flags_active": TOOL_FLAGS_ACTIVE,
        "live_probe_tools_ok": TOOLS_OK, "live_probe": TOOL_PROBE,
        "mode": ("not_applicable" if not RUN_MODEL else
                 ("ineligible_input_contract" if not MODEL_WORK_ELIGIBLE else
                 ("xhigh + run_python agent" if TOOLS_OK else "direct free-form fallback"))),
        "tool_parser": TOOL_PARSER, "max_turns": MAX_TURNS,
        "pass_policies": PASS_POLICIES, "sandbox_sha256": SANDBOX_SHA256,
        "python_tool_max_concurrency": PYTHON_TOOL_MAX_CONCURRENCY,
        "resource_sandbox_not_security_boundary": True,
        "filesystem_isolation": False,
        "same_uid_process_isolation": False,
        "address_space_limit_scope": "per process; no aggregate cgroup memory cap",
        "file_limit_scope": "per regular file, not aggregate disk quota",
        "process_limit_scope": (
            "no RLIMIT_NPROC: it is UID-wide on Linux; the notebook semaphore "
            "limits concurrent top-level calls but is not an aggregate descendant "
            "memory or process cgroup"),
        "descendant_process_group_reaped_after_every_call": True,
        "completed_tool_calls": tool_call_total,
        "successful_tool_calls": tool_success_total,
        "failed_tool_calls": tool_error_total,
    },
    "settings": SETTINGS, "concurrency": CONCURRENCY, "samples_per_task": SAMPLES_PER_TASK,
    "throughput": throughput,
    "challenge_path": ch_path, "n_tasks_total": len(challenges), "n_tasks_attempted": len(per_task),
    "n_queries_total": n_queries_total,
    "n_sample_records": len(sample_rows),
    "challenge_sha256": challenge_sha256,
    "aggregator_sha256": AGGREGATOR_SHA256,
    "prediction_accounting": {
        "valid_model_grids": valid_model_grid_total,
        "published_model_grids": published_model_grid_total,
        "publication_errors": publication_error_total,
        "model_backed_attempt_1": len(model_attempt_slots["attempt_1"]),
        "model_backed_attempt_2": len(model_attempt_slots["attempt_2"]),
        "primary_sample_records": len(primary_rows),
        "primary_unique_queries": primary_unique_queries,
        "primary_duplicate_records": primary_duplicate_records,
        "primary_expected_query_coverage": primary_expected_query_coverage,
        "primary_xhigh_records": primary_xhigh_records,
        "primary_wrong_policy_records": primary_wrong_policy_records,
        "primary_valid_grids": primary_valid_grids,
        "primary_anomaly_count": primary_anomaly_count,
        "primary_anomaly_reason_counts": primary_anomaly_reason_counts,
        "duplicate_attempt_pairs": duplicate_attempt_pairs,
    },
    "submission_robustness": {
        "floor": "single-source guaranteed_fallback_submission (identity+distinct dihedral)",
        "challenge_validation": "single-source load_public_challenges_with_digest",
        "every_publish_validated": True,
        "atomic_writer": "single-source aggregate_submission._write_json",
        "distinct_attempts_enforced_by_validator": True,
        "direct_file_submittable": MODE["direct_file_submittable"],
        "scope": MODE["submission_scope"],
        "diagnostic_upload_forbidden": not MODE["direct_file_submittable"],
    },
    "observability": {
        "status_path": str(STATUS),
        "acceptance_path": str(ACCEPTANCE),
        "sample_events_path": str(SAMPLE_EVENTS),
        "sample_events_format": (
            "append-only JSONL with run_id and contiguous sequence; a completed "
            "run must parse without a partial line"
        ),
        "sample_event_records_written": SAMPLE_EVENT_WRITTEN[0],
        "sample_event_write_errors": SAMPLE_EVENT_ERRORS,
        "status_write_errors": STATUS_ERRORS,
        "previous_run_archive": PREVIOUS_RUN_ARCHIVE,
    },
    "deadline_policy": {
        "hard_wall_s_from_start": round(HARD_WALL_S - RUN_STARTED_S, 1),
        "finalization_reserve_s": FINALIZATION_RESERVE_S,
        "generation_deadline_s_from_start": round(GENERATION_DEADLINE_S - RUN_STARTED_S, 1),
        "launch_admission": "episode cap until 8 samples, then padded empirical p90",
        "coverage_budget_adjustment": COVERAGE_BUDGET_ADJUSTMENT,
        "phase1": phase1,
        "phase2": phase2,
        "attempt2_priority": (
            "missing-grid > non-stop/recovered/forced-final/tool-or-context-error "
            "> completion_tokens > wall_s"
        ),
        "last_predicted_sample_s": {
            name: round(predicted_sample_s(name), 1) for name in PASS_POLICIES
        },
    },
    "budgeting": {
        "contract_scope": "competition_rerun",
        "worst_case_contract": BUDGET_CONTRACT,
        "active_mode_window": ACTIVE_MODE_WINDOW,
        "full_run_task_target": FULL_RUN_TASK_TARGET,
        "full_run_query_target": FULL_RUN_QUERY_TARGET,
        "worst_case_xhigh_waves": math.ceil(FULL_RUN_QUERY_TARGET / CONCURRENCY),
        "worst_case_xhigh_generation_h": round(
            math.ceil(FULL_RUN_QUERY_TARGET / CONCURRENCY)
            * PASS_POLICIES["xhigh"]["episode_budget_s"] / 3600, 2),
        "mean_wall_s_per_sample": round(sum(walls)/len(walls),1) if walls else None,
        "max_wall_s_per_sample": max(walls) if walls else None,
        "mean_completion_tokens": round(sum(toks)/len(toks)) if toks else None,
        "max_completion_tokens": max(toks) if toks else None,
        "samples_completed": len(walls),
        "note": "Use mean/max wall and saturated scheduler throughput to project the 12h full run. Diagnostic cohorts smaller than max_num_seqs understate full-batch throughput; length finishes indicate recovery/budget pressure.",
    },
    "per_task": per_task,
}
AGG._write_json(RECEIPT, receipt)
receipt_sha256 = file_sha256(RECEIPT)
if receipt_sha256 is None:
    raise RuntimeError("could not hash final receipt artifact")
acceptance_artifact = {
    "schema": "arc-qwen38-freeform-acceptance-v3",
    "run_id": RUN_ID,
    "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "outcome": acceptance["outcome"],
    "accepted_for_manual_upload": acceptance["accepted_for_manual_upload"],
    "full_run_proven": acceptance["full_run_proven"],
    "failed_checks": acceptance["failed_checks"],
    "warnings": acceptance["warnings"],
    "operator_action": operator_action,
    "acceptance": acceptance,
    "acceptance_inputs": acceptance_inputs,
    "artifact_hashes": {
        **artifact_hashes,
        "receipt_sha256": receipt_sha256,
    },
}
AGG._write_json(ACCEPTANCE, acceptance_artifact)
acceptance_sha256 = file_sha256(ACCEPTANCE)
if acceptance_sha256 is None:
    raise RuntimeError("could not hash final acceptance artifact")
final_artifact_hashes = {
    **acceptance_artifact["artifact_hashes"],
    "acceptance_sha256": acceptance_sha256,
}
write_status(
    "complete", receipt_path=str(RECEIPT), submission_path=str(SUB),
    acceptance_path=str(ACCEPTANCE),
    served=served, n_sample_records=len(sample_rows),
    duplicate_attempt_pairs=duplicate_attempt_pairs,
    outcome=acceptance["outcome"],
    accepted_for_manual_upload=acceptance["accepted_for_manual_upload"],
    full_run_proven=acceptance["full_run_proven"],
    operator_action=operator_action,
    failed_checks=acceptance["failed_checks"],
    artifact_hashes=final_artifact_hashes)
log_event(
    "run_complete",
    outcome=acceptance["outcome"],
    accepted_for_manual_upload=acceptance["accepted_for_manual_upload"],
    full_run_proven=acceptance["full_run_proven"],
    operator_action=operator_action["code"],
    receipt_sha256=receipt_sha256,
    acceptance_sha256=acceptance_sha256,
    submission_sha256=artifact_hashes["submission_sha256"],
)
log("RECEIPT:", json.dumps({
    "status": receipt["status"],
    "mode": receipt["mode"],
    "served": receipt["served"],
    "n_tasks_attempted": receipt["n_tasks_attempted"],
    "acceptance": receipt["acceptance"],
    "artifact_hashes": final_artifact_hashes,
}, indent=1))
log("=" * 72)
log("NEXT ACTION:", operator_action["headline"])
log(operator_action["instruction"])
log("=" * 72)
log("submission tasks:", len(json.load(open(SUB))))

