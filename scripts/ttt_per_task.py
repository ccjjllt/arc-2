"""TTT per-task entry (010 S3 / 008 6.2). CPU plan + dry-run only; no GPU launch here.

Covers docs/plan/010-gpu-budget-experiments.md S3 TTT paragraph and
docs/plan/008-generalization-adaptation.md 6.2 (0/8-step comparison on the same
offline checkpoint, same candidates/views). Reuses torch-free helpers from the
three existing scripts; modifies none of them:

  solve_grid.py     -> grids_equal, task_view_seed, parse_query_file (scoring/seed)
  train_grid_sft.py -> build_sequence, lr_multiplier (answer-only labels, LR shape)
  arc_episode_data.py -> build_messages (prompt building, GPU path only)

Dry-run (pure CPU, no model, no training):

  python3 scripts/ttt_per_task.py --help
  python3 scripts/ttt_per_task.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

# Torch-free reuse (both modules import no torch at top level).
from solve_grid import grids_equal as _solve_grids_equal  # noqa: E402
from solve_grid import task_view_seed as _task_view_seed  # noqa: E402
from train_grid_sft import build_sequence as _train_build_sequence  # noqa: E402
from train_grid_sft import lr_multiplier as _train_lr_multiplier  # noqa: E402

DEFAULT_MANIFEST = os.path.join(
    os.path.dirname(_SCRIPT_DIR), "experiments", "010", "evaluation-manifest.json"
)
# Fallback when invoked from another cwd: path relative to this file.
if not os.path.isfile(DEFAULT_MANIFEST):
    DEFAULT_MANIFEST = "experiments/010/evaluation-manifest.json"
DEFAULT_MANIFEST_KEY = "TTT_P6"
DEFAULT_STEPS = "0,8"
ALLOWED_STEPS = (0, 8, 16)
DEFAULT_GRAD_ACCUM = 4
DEFAULT_SEED = 20261003
PROMPT_LABEL = -100


def parse_ttt_steps(text):
    """Parse comma-separated TTT update counts. Allowed {0,8,16}; default 0,8.

    16 is accepted as an explicit opt-in only (010 S3: run it only when 8 steps
    already show a signal worth following). The default plan never runs 16.
    """
    parts = [p.strip() for p in str(text or "").split(",") if p.strip() != ""]
    if not parts:
        raise ValueError("--ttt-steps is empty")
    steps = []
    for part in parts:
        try:
            value = int(part)
        except ValueError:
            raise ValueError("--ttt-steps must be integers from {0,8,16}, got %r" % part)
        if value not in ALLOWED_STEPS:
            raise ValueError("--ttt-steps allows only 0/8/16, got %d" % value)
        if value not in steps:
            steps.append(value)
    return sorted(steps)


def load_ttt_p6(manifest_path, key=DEFAULT_MANIFEST_KEY):
    """Load the frozen TTT P6 slice: 6 tasks / 6 queries (H_new 3/3 + E24 3/3).

    Reads input-side metadata only (task_id, n_demo, n_test_queries, tier).
    Never touches query answers; solutions stay in the scorer, not here.
    """
    with open(manifest_path, "r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if key not in manifest:
        raise ValueError("manifest key %r not found in %s" % (key, manifest_path))
    block = manifest[key]
    entries = []
    for group in ("H_new", "E24"):
        for item in block.get(group, {}).get("entries", []):
            entries.append(
                {
                    "task_id": str(item["task_id"]),
                    "group": group,
                    "tier": str(item.get("tier", "")),
                    "n_demo": int(item.get("n_demo", 0)),
                    "n_test_queries": int(item.get("n_test_queries", 0)),
                }
            )
    total_tasks = int(block.get("total_tasks", len(entries)))
    total_queries = int(block.get("total_queries", 0))
    if len(entries) != total_tasks:
        raise ValueError(
            "TTT slice task count diverged: %d entries vs total_tasks=%d"
            % (len(entries), total_tasks)
        )
    return {
        "manifest": os.path.abspath(manifest_path),
        "key": key,
        "entries": entries,
        "n_tasks": len(entries),
        "n_queries": sum(e["n_test_queries"] for e in entries),
        "manifest_total_queries": total_queries,
    }


def per_task_seed(base_seed, task_id):
    """Fixed per-task seed derived from the base seed + task_id (stable hash)."""
    digest = hashlib.sha256(
        ("%d|%s" % (int(base_seed), str(task_id))).encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") & 0x7FFFFFFFFFFFFFFF


def fresh_optimizer_spec(seed):
    """Fresh-optimizer marker: rebuilt from scratch per task, never carried over.

    Returns the config dict the GPU loop must use when constructing a new
    optimizer/scheduler/cursor on top of the same offline checkpoint.
    """
    return {
        "reset": True,
        "carryover_updates": 0,
        "seed": int(seed),
        "note": "same offline checkpoint, fresh optimizer/scheduler/cursor per task",
    }


def plan_runs(entries, steps, grad_accum, base_seed):
    """Per-task adaptation plan. Multi-query tasks share one adaptation.

    Cost is counted per task (one reset + one adaptation), not per query.
    microsteps = updates * grad_accum (microbatch is always 1).
    """
    plans = []
    for entry in entries:
        task_seed = per_task_seed(base_seed, entry["task_id"])
        per_step = {}
        for step in steps:
            per_step[str(step)] = {
                "optimizer_updates": int(step),
                "grad_accum": int(grad_accum),
                "microsteps": int(step) * int(grad_accum),
                "optimizer": fresh_optimizer_spec(task_seed),
                "view_seed": int(_task_view_seed(base_seed, entry["task_id"], 0, "orig", 0)),
            }
        plans.append(
            {
                "task_id": entry["task_id"],
                "group": entry["group"],
                "tier": entry["tier"],
                "n_demo": entry["n_demo"],
                "n_test_queries": entry["n_test_queries"],
                "task_seed": task_seed,
                "cost_unit": "task",
                "steps": per_step,
            }
        )
    return plans


def build_loo_train_contexts(demos, task_id=""):
    """LOO demonstration contexts for TTT. Query outputs never enter here.

    Each held-out demo becomes the supervised target once; the remaining demos
    form the context. Unlike the offline min_loo_demos=2 filter, a 2-demo task
    leaving a single context demo is still usable: it is kept and flagged with
    used_single_context=True instead of skipping the whole task. Tasks with
    fewer than 2 demos yield no contexts (nothing to leave out).
    """
    demos = list(demos or [])
    if len(demos) < 2:
        return []
    contexts = []
    for held in range(len(demos)):
        context = [demos[i] for i in range(len(demos)) if i != held]
        contexts.append(
            {
                "task_id": str(task_id),
                "held_out_index": int(held),
                "n_context": len(context),
                "used_single_context": len(context) == 1,
                "target_demo_index": int(held),
            }
        )
    return contexts


def mask_query_targets(labels, prompt_len):
    """Training collator rule: hidden query outputs must stay masked.

    Query spans keep label -100 so no query gold ever trains. Pure function so
    the CPU dry-run can exercise it without torch. Reuses the same -100
    prompt convention as train_grid_sft.build_sequence.
    """
    masked = [PROMPT_LABEL if i < int(prompt_len) else int(v) for i, v in enumerate(labels)]
    # Everything at/after prompt_len is a query-output position here -> mask it.
    for i in range(int(prompt_len), len(masked)):
        masked[i] = PROMPT_LABEL
    return masked


def collate_ttt_example(prompt_ids, target_ids, eos_id, is_query_without_target=False):
    """Build one TTT training row, or skip (None) when the target is missing.

    Demonstrations/LOO go through build_sequence (answer-only supervision).
    A query row without a target (hidden gold unavailable at adapt time) is
    skipped without error -> returns None. A query row *with* a target is only
    used for post-hoc scoring; its labels are fully masked for training.
    """
    if is_query_without_target or target_ids is None:
        return None
    input_ids, labels, attention = _train_build_sequence(prompt_ids, target_ids, eos_id)
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention,
    }


def empty_trial_record(step):
    """Shared per-step trial schema (also used by the no-TTT second-view arm).

    Timing is split into train_s / gen_s / rescore_s so the equal-wall-clock
    comparison in 008 6.2 can trade TTT updates against one extra view.
    final_train_loss is recorded only; low loss never routes or filters
    (low_loss_routed stays False).
    """
    return {
        "optimizer_updates": int(step),
        "exact_match": None,
        "train_s": None,
        "gen_s": None,
        "rescore_s": None,
        "final_train_loss": None,
        "low_loss_routed": False,
    }


def summarize_task(task_plan, used_single_context):
    """Per-task result row sharing the second-view comparison schema."""
    trials = {name: empty_trial_record(name) for name in task_plan["steps"]}
    return {
        "task_id": task_plan["task_id"],
        "group": task_plan["group"],
        "tier": task_plan["tier"],
        "n_demo": task_plan["n_demo"],
        "n_test_queries": task_plan["n_test_queries"],
        "task_seed": task_plan["task_seed"],
        "used_single_context": bool(used_single_context),
        "trials": trials,
        "net_new": None,
        "lost": None,
    }


def build_output_payload(slice_info, plans, steps, grad_accum, base_seed, out_path=""):
    """Output schema shared with the no-TTT second-view comparison arm."""
    per_task = []
    for plan, entry in zip(plans, slice_info["entries"]):
        loo = build_loo_train_contexts([{"demo": i} for i in range(entry["n_demo"])],
                                       task_id=entry["task_id"])
        single = any(c["used_single_context"] for c in loo)
        per_task.append(summarize_task(plan, single))
    total_micro = {}
    for step in steps:
        total_micro[str(step)] = sum(
            p["steps"][str(step)]["microsteps"] for p in plans
        )
    return {
        "experiment": "ttt_per_task",
        "spec": "010-S3 + 008-6.2: 0/8 updates, accum %d, per-task reset" % int(grad_accum),
        "manifest": slice_info["manifest"],
        "manifest_key": slice_info["key"],
        "n_tasks": slice_info["n_tasks"],
        "n_queries": slice_info["n_queries"],
        "ttt_steps": [int(s) for s in steps],
        "grad_accum": int(grad_accum),
        "seed": int(base_seed),
        "cost_unit": "task",
        "conditional_microsteps": total_micro,
        "per_task": per_task,
        "aggregates": {
            "net_new": None,
            "lost": None,
            "note": "net_new/lost vs 0-step on identical queries; scored post-hoc only",
        },
        "scoring_note": "query targets used for post-hoc scoring only, never in adaptation",
        "out": out_path,
    }


def ttt_adapt_and_generate(*args, **kwargs):
    """GPU TTT loop (NOT implemented in this CPU entry).

    The GPU session must: reload the same offline checkpoint per task, build a
    fresh optimizer (fresh_optimizer_spec), train demonstrations/LOO (+
    consistent augments) with query targets masked (mask_query_targets /
    collate_ttt_example), then generate + rescore, filling train_s/gen_s/
    rescore_s per task. Raises until that session lands.
    """
    raise RuntimeError(
        "GPU TTT loop not implemented in this CPU entry; "
        "needs a GPU session (010 S3). Plan/dry-run only."
    )


def build_parser():
    parser = argparse.ArgumentParser(
        description=(
            "TTT per-task entry (010 S3 + 008 6.2). CPU dry-run plans the frozen "
            "TTT P6 slice (6 tasks / 6 queries); GPU adaptation is not run here."
        )
    )
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST,
                        help="evaluation-manifest.json path.")
    parser.add_argument("--manifest-key", default=DEFAULT_MANIFEST_KEY,
                        help="Manifest slice key (default TTT_P6).")
    parser.add_argument("--ttt-steps", default=DEFAULT_STEPS,
                        help="Comma-separated updates from {0,8,16}. Default '0,8'; "
                             "16 is explicit opt-in only, never default.")
    parser.add_argument("--grad-accum", type=int, default=DEFAULT_GRAD_ACCUM,
                        help="Gradient accumulation (default 4; microbatch is 1).")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED,
                        help="Base seed; per-task seeds derive via sha256(seed|task_id).")
    parser.add_argument("--checkpoint", default="",
                        help="Offline checkpoint dir (same init for every task; "
                             "required unless --dry-run).")
    parser.add_argument("--challenges", default="",
                        help="ARC challenges JSON (task inputs; required unless --dry-run).")
    parser.add_argument("--out", default="",
                        help="Output JSON path for the shared-schema result skeleton.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Pure-CPU plan: parse the P6 slice, print the 0/8 plan, "
                             "no model load, no training.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    steps = parse_ttt_steps(args.ttt_steps)
    if int(args.grad_accum) < 1:
        raise SystemExit("--grad-accum must be >= 1")
    if not os.path.isfile(args.manifest):
        raise SystemExit("manifest not found: %s" % args.manifest)
    slice_info = load_ttt_p6(args.manifest, args.manifest_key)
    plans = plan_runs(slice_info["entries"], steps, int(args.grad_accum), int(args.seed))
    if args.dry_run:
        lines = [
            "TTT dry-run: %s tasks=%d queries=%d steps=%s accum=%d seed=%d"
            % (slice_info["key"], slice_info["n_tasks"], slice_info["n_queries"],
               ",".join(str(s) for s in steps), int(args.grad_accum), int(args.seed))
        ]
        for plan in plans:
            micro = ",".join(
                "%s:%d" % (s, plan["steps"][s]["microsteps"]) for s in plan["steps"]
            )
            loo_n = max(0, plan["n_demo"] - 1)
            single = " used_single_context" if plan["n_demo"] == 2 else ""
            lines.append(
                "  %s [%s/%s] demos=%d queries=%d seed=%d {%s} loo_ctx=%d%s"
                % (plan["task_id"], plan["group"], plan["tier"], plan["n_demo"],
                   plan["n_test_queries"], plan["task_seed"], micro, loo_n, single)
            )
        total8 = sum(p["steps"].get("8", {}).get("microsteps", 0) for p in plans)
        lines.append("conditional_microsteps_if_8_updates_accum%d=%d" % (int(args.grad_accum), total8))
        lines.append("no model loaded; no training; query targets never used in adaptation")
        print("\n".join(lines), flush=True)
        if args.out:
            payload = build_output_payload(
                slice_info, plans, steps, int(args.grad_accum), int(args.seed), args.out)
            parent = os.path.dirname(os.path.abspath(args.out))
            if parent:
                os.makedirs(parent, exist_ok=True)
            with open(args.out, "w", encoding="utf-8") as handle:
                json.dump(payload, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
            print("wrote skeleton: %s" % args.out, flush=True)
        return 0
    # Non-dry-run: validate presence of the offline init + inputs, then stop.
    # Model load / optimizer / generation stay in the GPU session.
    if not args.checkpoint or not os.path.isdir(args.checkpoint):
        raise SystemExit("needs --checkpoint <offline adapter dir> (same init every task)")
    if not args.challenges or not os.path.isfile(args.challenges):
        raise SystemExit("needs --challenges <ARC challenges JSON>")
    raise SystemExit(
        "GPU TTT loop not implemented in this CPU entry "
        "(ttt_adapt_and_generate raises); dry-run plan above is the deliverable."
    )


if __name__ == "__main__":
    raise SystemExit(main())
