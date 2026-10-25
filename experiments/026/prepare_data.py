#!/usr/bin/env python3
"""Manually invoked CPU preparation of the 026 NVARC continuation data.

Only official *training* files and a fixed extra-training generator whitelist are
used. No evaluation solution file is accepted by this entry point. No GPU code.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import importlib
import json
import os
import platform
import random
import signal
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
REVISION = "d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0"
SOURCE = "water-vapor/synth-rearc"
QUARANTINE = ("17829a00", "b74ca5d1", "f18ec8cc")
PILOT_FAMILIES = (
    "14754a24", "18419cfa", "2037f2c7", "2de01db2", "3194b014",
    "396d80d7", "5587a8d0", "6df30ad6", "7bb29440", "aa62e3f4",
    "aaecdb9a", "b15fca0b", "cd3c21df", "d56f2372", "e729b7be", "f5c89df1",
)
TRAINING_DIR = REPO / "experiments/007/raw/arc-prize-2026"
DELTA_FILE = REPO / "experiments/021/evidence/rearc2-generator-delta.json"
EVAL_CHALLENGES = REPO / "experiments/025/cohort-v1/challenges.json"


def digest(obj):
    return hashlib.sha256(json.dumps(obj, separators=(",", ":"), sort_keys=True).encode()).hexdigest()


def file_digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def grid(value):
    value = tuple(tuple(row) for row in value)
    if not (1 <= len(value) <= 30 and 1 <= len(value[0]) <= 30):
        raise ValueError("grid dimensions out of ARC range")
    if any(len(row) != len(value[0]) for row in value):
        raise ValueError("nonrectangular grid")
    if any(type(x) is not int or not 0 <= x <= 9 for row in value for x in row):
        raise ValueError("invalid ARC color")
    return value


def _alarm(_signum, _frame):
    raise TimeoutError("generator/verifier CPU time allowance reached")


def bounded_call(seconds, func, *args):
    signal.setitimer(signal.ITIMER_REAL, max(0.001, seconds))
    try:
        return func(*args)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def branch_signature(task_id, value):
    """Only source-audited categorical cues; no generic complexity labels."""
    if task_id in ("009d5c81", "9bebae7a"):
        color = 1 if task_id == "009d5c81" else 6
        cells = [(i, j) for i, row in enumerate(value) for j, x in enumerate(row) if x == color]
        mi, mj = min(i for i, _ in cells), min(j for _, j in cells)
        return tuple(sorted((i - mi, j - mj) for i, j in cells))
    if task_id == "a6953f00":
        return len(value[0]) % 2
    return None


def episode_flags(pairs):
    outputs = [digest(p["output"]) for p in pairs]
    return {
        "query_output_equals_demo_output": outputs[-1] in outputs[:-1],
        "identity_pairs": sum(p["input"] == p["output"] for p in pairs),
        "distinct_demo_outputs": len(set(outputs[:-1])),
    }


def make_row(task_id, pairs, split, base_id, source, revision, **extra):
    flags = episode_flags(pairs)
    row = {
        "episode_id": base_id, "base_id": base_id, "family_id": task_id,
        "source": source, "source_revision": revision, "ancestor_task_ids": [task_id],
        "split": split, "train": pairs[:-1], "query_input": pairs[-1]["input"],
        "query_output": pairs[-1]["output"], "query_hash": digest(pairs[-1]["input"]),
        "augmentation": "identity; no stored geometric/color expansions",
        "quality_flags": flags,
        # Each digit and row separator has one token in the original 16-token
        # NVARC vocabulary. This omits prompt separators, not actual tokenizer IO.
        "grid_token_proxy": sum(len(p[k]) * (len(p[k][0]) + 1) for p in pairs for k in ("input", "output")),
    }
    row.update(extra)
    return row


def load_assets(source_root, seed):
    head = (source_root / ".git/HEAD").read_text().strip()
    if head != REVISION:
        raise ValueError(f"source must be detached at {REVISION}; found {head}")
    whitelist = set((source_root / "data/task_lists/arc2_training_minus_arc1_task_ids.txt").read_text().split())
    delta = json.loads(DELTA_FILE.read_text())
    ids = delta["delta_ids"]
    if delta["synth_rearc_commit"] != REVISION or not set(ids).issubset(whitelist):
        raise ValueError("fixed source whitelist mismatch")
    challenges_path = TRAINING_DIR / "arc-agi_training_challenges.json"
    solutions_path = TRAINING_DIR / "arc-agi_training_solutions.json"
    challenges = json.loads(challenges_path.read_text())
    # Read challenges first, prove source task IDs are training IDs before targets.
    if not set(ids).issubset(challenges):
        raise ValueError("generator ancestors not all in official training")
    eval_ids = set(json.loads(EVAL_CHALLENGES.read_text()))
    if set(challenges) & eval_ids:
        raise ValueError("training/evaluation ancestor collision")
    solutions = json.loads(solutions_path.read_text())
    qualified = sorted(set(ids) - set(QUARANTINE))
    heldout = sorted(qualified, key=lambda t: digest(["026-family-v1", seed, t]))[:16]
    identities = {str(p.relative_to(REPO)): file_digest(p) for p in (DELTA_FILE, challenges_path, solutions_path, EVAL_CHALLENGES)}
    source_files = [source_root / "dsl.py", source_root / "utils.py", source_root / "synth_rearc/core.py", source_root / "synth_rearc/dsl_ext.py"]
    source_files += sorted(p for t in ids for p in (source_root / f"synth_rearc/tasks/task_{t}").rglob("*.py"))
    hashes = {str(p.relative_to(source_root)): file_digest(p) for p in source_files}
    return challenges, solutions, qualified, set(heldout), identities, hashes


def generate_family(task_id, count, args, split, global_inputs):
    mod = importlib.import_module(f"synth_rearc.tasks.task_{task_id}")
    start = time.monotonic()
    rows, rejected = [], Counter()
    draws = 0
    used_inputs = set()

    def pair():
        nonlocal draws
        remaining = args.family_seconds - (time.monotonic() - start)
        if remaining <= 0:
            raise TimeoutError("family CPU time allowance reached")
        draw_seed = int(digest([args.seed, task_id, draws])[:16], 16)
        draws += 1
        random.seed(draw_seed)
        value = bounded_call(min(args.pair_seconds, remaining), mod.generate, 0.0, 1.0)
        value = {k: grid(value[k]) for k in ("input", "output")}
        answer = bounded_call(min(args.pair_seconds, remaining), mod.verify, value["input"])
        if grid(answer) != value["output"]:
            raise ValueError("generated pair fails fixed verifier")
        key = digest(value["input"])
        if key in used_inputs or key in global_inputs:
            rejected["duplicate_input"] += 1
            return None
        return value, draw_seed

    try:
        while len(rows) < count and draws < count * 80:
            # Draw the query first, then ensure any audited arbitrary categorical
            # cue also appears among its demonstrations. All four scenes are fresh.
            q = pair()
            if q is None:
                continue
            picked = [q]
            signature = branch_signature(task_id, q[0]["input"])
            for _ in range(3):
                for attempt in range(100):
                    d = pair()
                    if d is None or any(digest(d[0]["input"]) == digest(x[0]["input"]) for x in picked):
                        continue
                    if signature is not None and len(picked) == 1 and branch_signature(task_id, d[0]["input"]) != signature:
                        rejected["missing_query_cue_in_first_demo"] += 1
                        continue
                    picked.append(d)
                    break
                else:
                    raise ValueError("could not assemble unique demonstrations")
            pairs = [x[0] for x in picked[1:]] + [q[0]]
            if len({digest(x["output"]) for x in pairs[:-1]}) < 2:
                rejected["constant_demo_output"] += 1
                continue
            if all(x["input"] == x["output"] for x in pairs):
                rejected["all_identity"] += 1
                continue
            for value in pairs:
                used_inputs.add(digest(value["input"]))
            base_id = f"026|rearc2|{task_id}|seed{args.seed}|base{len(rows):04d}"
            rows.append(make_row(task_id, pairs, split, base_id, SOURCE, REVISION,
                                 generator_seeds=[x[1] for x in picked[1:]] + [q[1]],
                                 verified_pairs=4,
                                 semantic_review="pilot_source_review" if task_id in PILOT_FAMILIES else "official_demo_and_pair_verifier_only",
                                 categorical_query_cue_covered=signature is not None))
    except Exception as exc:
        error = type(exc).__name__ + ": " + str(exc)
    else:
        error = None
    global_inputs.update(used_inputs)
    return rows, {"task_id": task_id, "requested": count, "episodes": len(rows), "draws": draws,
                  "rejected": dict(rejected), "error": error, "seconds": round(time.monotonic() - start, 3)}


def quantiles(values):
    values = sorted(values)
    return {"min": values[0], "median": statistics.median(values), "p90": values[int((len(values)-1)*.9)], "max": values[-1]} if values else {}


def canonical_grid(value):
    """Diagnostic only: identify color/dihedral query similarities per family."""
    variants = []
    value = tuple(map(tuple, value))
    for _ in range(4):
        for candidate in (value, tuple(row[::-1] for row in value)):
            palette = {}
            variants.append(tuple(tuple(palette.setdefault(x, len(palette)) for x in row) for row in candidate))
        value = tuple(zip(*value[::-1]))
    return min(variants)


def freeze_embedded(pool, out, row_count, seed):
    """Freeze a source/family/length-balanced subset before any model run."""
    pool = pool.resolve()
    if pool.parent != REPO / "experiments/026" or not pool.name.startswith("data"):
        raise ValueError("freeze source must be an experiments/026/data* pool")
    parent = json.loads((pool / "manifest.json").read_text())
    original = [json.loads(line) for line in (pool / "train.jsonl").read_text().splitlines()]
    heldout = set(parent["family_holdout_ids"])
    if any(r["split"] != "train" or set(r["ancestor_task_ids"]) & heldout for r in original):
        raise ValueError("holdout ancestor in training pool")
    # For the pinned 16-token WordLevel tokenizer and the original QwenFormatter,
    # grid_proxy omits -1 final newline per grid; each pair adds 8 role/newline/
    # boundary tokens. Thus full text has proxy + 6*(demo_count+1) tokens. The
    # runtime also checks this with the real tokenizer before any model update.
    for row in original:
        row["formatter_length_formula"] = row["grid_token_proxy"] + 6 * (len(row["train"]) + 1)
    too_long = [r["episode_id"] for r in original if r["formatter_length_formula"] > 8192]
    eligible = [r for r in original if r["formatter_length_formula"] <= 8192]
    synthetic = [r for r in eligible if r["source"] == SOURCE]
    replay = [r for r in eligible if r["source"] != SOURCE]
    by_family = {}
    for row in synthetic:
        by_family.setdefault(row["family_id"], []).append(row)
    families = sorted(by_family, key=lambda f: digest(["026-embedded-family-v1", seed, f]))
    queues = {}
    bucket_counts = Counter()
    for family in families:
        ordered = sorted(by_family[family], key=lambda r: (r["formatter_length_formula"], r["episode_id"]))
        buckets = [ordered[len(ordered)*i//4:len(ordered)*(i+1)//4] for i in range(4)]
        for i, bucket in enumerate(buckets):
            random.Random(digest(["026-within-length-bucket", seed, family, i])).shuffle(bucket)
        queues[family] = buckets
    n_each = row_count // 2
    picked_synth = []
    round_id = 0
    while len(picked_synth) < n_each:
        changed = False
        for family_id, family in enumerate(families):
            bucket_id = (round_id + family_id) % 4
            available = next((j for j in [(bucket_id + offset) % 4 for offset in range(4)] if queues[family][j]), None)
            if available is None:
                continue
            row = queues[family][available].pop()
            picked_synth.append(row)
            bucket_counts[available] += 1
            changed = True
            if len(picked_synth) == n_each:
                break
        if not changed:
            raise ValueError("not enough independent synthetic rows")
        round_id += 1
    # Official replay first contributes one untouched original query per ancestor,
    # then is stratified into equal count quartiles of the original full length.
    one_per_family = {}
    for row in sorted(replay, key=lambda r: digest(["026-official-query-v1", seed, r["episode_id"]])):
        one_per_family.setdefault(row["family_id"], row)
    official_order = sorted(one_per_family.values(), key=lambda r: (r["formatter_length_formula"], r["episode_id"]))
    official_buckets = [official_order[len(official_order)*i//4:len(official_order)*(i+1)//4] for i in range(4)]
    picked_official = []
    official_bucket_counts = []
    for i, bucket in enumerate(official_buckets):
        bucket.sort(key=lambda r: digest(["026-official-length-v1", seed, i, r["episode_id"]]))
        quota = n_each // 4 + int(i < n_each % 4)
        if len(bucket) < quota:
            raise ValueError("not enough official replay families in length bucket")
        picked_official += bucket[:quota]
        official_bucket_counts.append(quota)
    # Keep official buckets mixed in the consumption prefix as well as the pool.
    random.Random(digest(["026-official-order-v1", seed])).shuffle(picked_official)
    rows = [r for pair in zip(picked_synth, picked_official) for r in pair]
    path = out / "train.jsonl"
    path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in rows))
    compressed = out / "train.jsonl.gz"
    with compressed.open("wb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
            zipped.write(path.read_bytes())
    stats = {
        "name": "026 fixed embedded continuation data", "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seed": seed, "pool_path": str(pool.relative_to(REPO)), "pool_manifest_sha256": file_digest(pool / "manifest.json"),
        "pool_train_sha256": file_digest(pool / "train.jsonl"), "code_sha256": file_digest(Path(__file__)),
        "train_jsonl_sha256": file_digest(path), "train_gzip_sha256": file_digest(compressed),
        "train_jsonl_bytes": path.stat().st_size, "train_gzip_bytes": compressed.stat().st_size,
        "base64_characters": 4 * ((compressed.stat().st_size + 2) // 3), "episodes": len(rows),
        "synthetic_episodes": len(picked_synth), "synthetic_families": len(families),
        "synthetic_episodes_per_family_histogram": dict(Counter(Counter(r['family_id'] for r in picked_synth).values())),
        "synthetic_selected_within_family_length_quartiles": dict(bucket_counts),
        "official_replay_episodes": len(picked_official), "official_replay_families": len({r['family_id'] for r in picked_official}),
        "official_selected_global_length_quartiles": official_bucket_counts,
        "family_holdout_ids": sorted(heldout), "holdout_ancestor_intersection": [],
        "quarantined_families": parent['quarantined_families'], "overlength_excluded_from_pool": too_long,
        "full_length_formula_distribution": quantiles([r['formatter_length_formula'] for r in rows]),
        "full_length_formula_by_source": {s: {"sum": sum(r['formatter_length_formula'] for r in rows if r['source'] == s), "distribution": quantiles([r['formatter_length_formula'] for r in rows if r['source'] == s])} for s in sorted({r['source'] for r in rows})},
        "selected_episode_ids": [r['episode_id'] for r in rows],
        "selection": "512/512 sources (or equal halves if explicitly reduced); synthetic round-robin across all 90 training families and within-family length quartiles; official one original query per ancestor, equal global length quartiles; deterministic hash order; no model outputs used",
        "length_formula_contract": "original QwenFormatter + pinned 16-token WordLevel grids15 tokenizer; full = grid_token_proxy + 6*(n_demos+1); runtime must compare real tokenizer lengths before training",
        "limitations": parent['limitations'],
    }
    (out / "manifest.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps({k: v for k, v in stats.items() if k in ('episodes', 'synthetic_episodes', 'official_replay_episodes', 'train_gzip_bytes', 'base64_characters', 'full_length_formula_distribution')}, indent=2))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source-root", type=Path)
    ap.add_argument("--freeze-from", type=Path)
    ap.add_argument("--embedded-rows", type=int, choices=(512, 768, 1024), default=1024)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--profile", choices=("pilot", "expanded"), default="pilot")
    ap.add_argument("--seed", type=int, default=20261025)
    ap.add_argument("--episodes-per-family", type=int)
    ap.add_argument("--family-seconds", type=float, default=30)
    ap.add_argument("--pair-seconds", type=float, default=4)
    args = ap.parse_args()
    count = args.episodes_per_family or (4 if args.profile == "pilot" else 12)
    if not 1 <= count <= 32 or not 0 < args.pair_seconds <= 10 or not 0 < args.family_seconds <= 120:
        ap.error("small CPU preparation bounds exceeded")
    out = args.output_dir.resolve()
    if out.parent != REPO / "experiments/026" or not out.name.startswith("data"):
        ap.error("output directory must be experiments/026/data*")
    out.mkdir(exist_ok=True, parents=True)
    if any(out.iterdir()):
        raise FileExistsError(f"refusing to overwrite nonempty {out}")
    if args.freeze_from is not None:
        freeze_embedded(args.freeze_from, out, args.embedded_rows, args.seed)
        return
    if args.source_root is None:
        ap.error("--source-root is required for generator preparation")
    os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/arc026-matplotlib")
    os.environ.setdefault("MPLBACKEND", "Agg")
    signal.signal(signal.SIGALRM, _alarm)
    sys.path.insert(0, str(args.source_root.resolve()))
    started = time.monotonic()
    challenges, solutions, qualified, heldout, identities, hashes = load_assets(args.source_root, args.seed)
    official_checks, usable = [], []
    for task_id in qualified + list(QUARANTINE):
        try:
            mod = importlib.import_module(f"synth_rearc.tasks.task_{task_id}")
            checks = [grid(bounded_call(args.pair_seconds, mod.verify, grid(p["input"]))) == grid(p["output"]) for p in challenges[task_id]["train"]]
            good = all(checks)
            official_checks.append({"task_id": task_id, "matched": sum(checks), "total": len(checks), "all_match": good})
            if good and task_id not in QUARANTINE:
                usable.append(task_id)
        except Exception as exc:
            official_checks.append({"task_id": task_id, "error": type(exc).__name__ + ": " + str(exc), "all_match": False})
    selected = sorted(set(PILOT_FAMILIES) & set(usable)) if args.profile == "pilot" else sorted(usable)
    rows, family_runs, global_inputs = [], [], set()
    for task_id in selected:
        split = "family_holdout" if task_id in heldout else "train"
        generated, report = generate_family(task_id, count, args, split, global_inputs)
        rows += generated
        family_runs.append(report)
        print(json.dumps({"family": task_id, "split": split, "episodes": len(generated), "error": report["error"]}), flush=True)
    synthetic_train = [r for r in rows if r["split"] == "train"]
    replay, replay_rejected = [], Counter()
    # Match the number of synthetic training episodes as closely as the finite
    # official training query supply allows. No repeat/LOO row inflation.
    replay_candidates = sorted(challenges, key=lambda t: digest(["026-replay-v1", args.seed, t]))
    for task_id in replay_candidates:
        if task_id in heldout or task_id in QUARANTINE:
            continue
        task = challenges[task_id]
        for query_id, query in enumerate(task["test"]):
            pairs = [{k: grid(p[k]) for k in ("input", "output")} for p in task["train"]]
            pairs += [{"input": grid(query["input"]), "output": grid(solutions[task_id][query_id])}]
            inputs = [digest(p["input"]) for p in pairs]
            if len(set(inputs)) != len(inputs):
                replay_rejected["within_episode_duplicate_input"] += 1
                continue
            if inputs[-1] in global_inputs:
                replay_rejected["query_repeated_in_other_episode"] += 1
                continue
            global_inputs.add(inputs[-1])
            base_id = f"026|official_training|{task_id}|q{query_id}"
            replay.append(make_row(task_id, pairs, "train", base_id, "Kaggle ARC Prize 2026 official training", identities[str((TRAINING_DIR / 'arc-agi_training_challenges.json').relative_to(REPO))],
                                   original_query_index=query_id, verified_pairs=0,
                                   semantic_review="original_training_episode; no reassembly",
                                   upstream_checkpoint_exposure="training ancestor may have been used by NVARC"))
            if len(replay) >= len(synthetic_train):
                break
        if len(replay) >= len(synthetic_train):
            break
    all_train = synthetic_train + replay
    # A fixed balanced interleave preserves both sources in a 1024-row prefix.
    rng = random.Random(args.seed)
    rng.shuffle(synthetic_train)
    rng.shuffle(replay)
    all_train = []
    for i in range(max(len(synthetic_train), len(replay))):
        if i < len(synthetic_train): all_train.append(synthetic_train[i])
        if i < len(replay): all_train.append(replay[i])
    holdout_rows = [r for r in rows if r["split"] == "family_holdout"]
    files = {}
    for name, content in (("train.jsonl", all_train), ("family-holdout.jsonl", holdout_rows)):
        path = out / name
        path.write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in content))
        files[name] = {"sha256": file_digest(path), "episodes": len(content), "bytes": path.stat().st_size}
        compressed = out / (name + ".gz")
        with compressed.open("wb") as raw:
            with gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as zipped:
                zipped.write(path.read_bytes())
        files[name + ".gz"] = {"sha256": file_digest(compressed), "episodes": len(content), "bytes": compressed.stat().st_size, "uncompressed_sha256": file_digest(path)}
    manifest = {
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "name": "026 ARC2 extra training rule coverage challenger", "profile": args.profile,
        "code_sha256": file_digest(Path(__file__)), "seed": args.seed, "source_repository": SOURCE,
        "source_revision": REVISION, "source_root": str(args.source_root.resolve()),
        "source_file_sha256": hashes, "input_file_sha256": identities,
        "qualified_generator_families": sorted(usable), "quarantined_families": list(QUARANTINE),
        "quarantine_reason": "fixed verifier disagrees with one local official training demonstration in each family; no assertion that official data is wrong",
        "official_demo_checks": official_checks, "selected_generator_families": selected,
        "family_holdout_ids": sorted(heldout), "family_split_method": "lowest 16 sha256(JSON(['026-family-v1',seed,task_id])) among 106 non-quarantine training ancestors",
        "family_runs": family_runs, "files": files,
        "counts": {"synthetic_train_episodes": len(synthetic_train), "synthetic_train_families": len({r['family_id'] for r in synthetic_train}),
                   "synthetic_holdout_episodes": len(holdout_rows), "synthetic_holdout_families": len({r['family_id'] for r in holdout_rows}),
                   "official_replay_episodes": len(replay), "official_replay_families": len({r['family_id'] for r in replay}),
                   "total_train_episodes": len(all_train), "unique_train_base_ids": len({r['base_id'] for r in all_train}),
                   "unique_train_query_inputs": len({r['query_hash'] for r in all_train})},
        "replay_rejections": dict(replay_rejected),
        "training_grid_token_proxy": quantiles([r["grid_token_proxy"] for r in all_train]),
        "grid_token_proxy_by_source": {source: {"sum": sum(r['grid_token_proxy'] for r in all_train if r['source'] == source), "distribution": quantiles([r['grid_token_proxy'] for r in all_train if r['source'] == source])} for source in sorted({r['source'] for r in all_train})},
        "grid_proxy_above_8192": sum(r["grid_token_proxy"] > 8192 for r in all_train),
        "synthetic_train_pair_inputs": len(synthetic_train) * 4,
        "unique_synthetic_train_pair_inputs": len({digest(p['input']) for r in synthetic_train for p in r['train'] + [{'input': r['query_input']}]}),
        "synthetic_canonical_query_duplicate_count_within_family": len(synthetic_train) - len({(r['family_id'], canonical_grid(r['query_input'])) for r in synthetic_train}),
        "quality_flag_counts": dict(Counter(k for r in all_train for k, v in r['quality_flags'].items() if k != 'distinct_demo_outputs' and v)),
        "preview_prefix_1024": {"episodes": len(all_train[:1024]), "source_counts": dict(Counter(r['source'] for r in all_train[:1024])), "families": len({r['family_id'] for r in all_train[:1024]})},
        "runtime": {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(), "pythonhashseed": os.environ.get('PYTHONHASHSEED'), "elapsed_seconds": round(time.monotonic() - started, 3), "gpu": "none"},
        "limitations": [
            "No evaluation query target grids were read. The frozen 025 challenges file is used for ancestor-ID exclusion only.",
            "Held-out ancestors are excluded from every newly prepared training source, including official replay. This is new-continuation family holdout; NVARC upstream exposure is not absent or known completely.",
            "Different task IDs can share semantics; cross-ID semantic family equivalence is not established.",
            "109 extra is relative to fixed ARC-GEN coverage, not 109 previously unseen rules.",
            "Verifier agreement and original-demo matches do not prove human rule correctness or that three demonstrations uniquely identify every query rule.",
            "Only the 16 pilot families receive this round's source-level semantic review. Expanded families retain explicit weaker review labels.",
            "The token proxy counts grid digits and row separators only. Training must use the original NVARC formatter and check actual full-sequence length without truncation.",
            "No stored rotation/color augmentation inflates independent episode counts. Official replay repeats demonstrations across queries of the same task and retains task lineage.",
            "Synthetic categorical cue coverage is explicitly implemented only for 009d5c81,9bebae7a,a6953f00; other ambiguous branches have not been exhaustively certified.",
            "No model training, held-out evaluation or score gain is established by data preparation.",
        ],
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"output_dir": str(out), "counts": manifest['counts'], "elapsed_seconds": manifest['runtime']['elapsed_seconds']}, indent=2))


if __name__ == "__main__":
    main()
