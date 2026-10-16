import os
import time
import json
import random
import bz2
import pickle
import torch
import argparse
import torch.multiprocessing as mp
from collections import defaultdict


def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_str(name, default):
    value = os.getenv(name)
    return default if value is None or value == "" else value


def env_bool(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def parse_key_list(raw):
    if raw is None or raw.strip() == "":
        return None
    return [x.strip() for x in raw.split(",") if x.strip()]


def _parse_output_key(filename):
    base = filename.split(".")[0]
    if "_" not in base:
        return None
    task_key, maybe_sub = base.rsplit("_", 1)
    if not maybe_sub.isdigit():
        return None
    return task_key, int(maybe_sub), f"{task_key}_{int(maybe_sub)}"


def _solution_key(solution):
    try:
        if hasattr(solution, "tolist"):
            solution = solution.tolist()
        return tuple(tuple(int(x) for x in row) for row in solution)
    except Exception:
        return None


def infer_output_quality(output_dir, data):
    if not os.path.isdir(output_dir):
        return {}

    per_output = defaultdict(lambda: {
        "files": 0,
        "candidates": 0,
        "unique": set(),
        "best_beam": None,
    })

    for filename in sorted(os.listdir(output_dir)):
        parsed = _parse_output_key(filename)
        if parsed is None:
            continue
        _, _, output_key = parsed
        stats = per_output[output_key]
        stats["files"] += 1

        path = os.path.join(output_dir, filename)
        if not os.path.isfile(path):
            continue
        try:
            with bz2.BZ2File(path) as f:
                outputs = pickle.load(f)
        except Exception:
            continue

        if not isinstance(outputs, list):
            continue
        stats["candidates"] += len(outputs)
        for sample in outputs:
            if not isinstance(sample, dict):
                continue
            skey = _solution_key(sample.get("solution"))
            if skey is not None:
                stats["unique"].add(skey)
            try:
                beam = float(sample.get("beam_score"))
            except Exception:
                beam = None
            if beam is not None and (stats["best_beam"] is None or beam < stats["best_beam"]):
                stats["best_beam"] = beam

    quality = {}
    for task_key, task in data.items():
        need = len(task.get("test", []))
        if need <= 0:
            continue
        rows = []
        for i in range(need):
            output_key = f"{task_key}_{i}"
            raw = per_output.get(output_key)
            if raw is None:
                rows.append({"files": 0, "candidates": 0, "unique": 0, "best_beam": None})
            else:
                rows.append({
                    "files": raw["files"],
                    "candidates": raw["candidates"],
                    "unique": len(raw["unique"]),
                    "best_beam": raw["best_beam"],
                })

        files = [r["files"] for r in rows]
        candidates = [r["candidates"] for r in rows]
        unique = [r["unique"] for r in rows]
        beams = [r["best_beam"] for r in rows if r["best_beam"] is not None]
        quality[task_key] = {
            "min_files": min(files) if files else 0,
            "mean_files": (sum(files) / len(files)) if files else 0.0,
            "min_candidates": min(candidates) if candidates else 0,
            "mean_candidates": (sum(candidates) / len(candidates)) if candidates else 0.0,
            "min_unique": min(unique) if unique else 0,
            "mean_unique": (sum(unique) / len(unique)) if unique else 0.0,
            "best_beam": min(beams) if beams else None,
        }
    return quality


def infer_completed_keys(
    output_dir,
    data,
    min_files_per_test=1,
    min_unique_per_test=0,
    min_candidates_per_test=0,
    mode="files",
):
    quality = infer_output_quality(output_dir, data)

    completed = set()
    for task_key, task in data.items():
        need = len(task.get("test", []))
        if need <= 0:
            continue
        task_quality = quality.get(task_key, {})
        ok = True
        if task_quality.get("min_files", 0) < min_files_per_test:
            ok = False
        if mode == "quality":
            if task_quality.get("min_unique", 0) < min_unique_per_test:
                ok = False
            if task_quality.get("min_candidates", 0) < min_candidates_per_test:
                ok = False
        if ok:
            completed.add(task_key)
    return completed


def infer_key_coverage(output_dir, data):
    return infer_output_quality(output_dir, data)


def task_complexity(task):
    total_cells = 0
    total_pairs = len(task.get("train", [])) + len(task.get("test", []))
    for section in ("train", "test"):
        for sample in task.get(section, []):
            inp = sample.get("input", [])
            total_cells += len(inp) * (len(inp[0]) if len(inp) else 0)
            if section == "train":
                out = sample.get("output", [])
                total_cells += len(out) * (len(out[0]) if len(out) else 0)
    return total_cells + 50 * total_pairs


def order_keys(data, mode):
    keys = list(data.keys())
    if mode == "lexicographic":
        return sorted(keys)
    scored = [(k, task_complexity(data[k])) for k in keys]
    if mode == "hard_first":
        scored.sort(key=lambda x: x[1], reverse=True)
        return [k for k, _ in scored]
    if mode == "random":
        rnd = random.Random(env_int("ARC_TASK_ORDER_SEED", 42))
        rnd.shuffle(keys)
        return keys
    scored.sort(key=lambda x: x[1])
    return [k for k, _ in scored]


def parse_gpu_ids():
    raw = parse_key_list(os.getenv("ARC_GPU_IDS"))
    if raw is not None:
        gpu_ids = [int(x) for x in raw]
    else:
        gpu_count = torch.cuda.device_count()
        gpu_ids = list(range(gpu_count))
    if len(gpu_ids) == 0:
        return [0]
    return gpu_ids


def worker_marker_path(marker_tag, rank):
    return f"/kaggle/worker_{marker_tag}_{rank}"


def clear_worker_markers(marker_tag, nprocs):
    for r in range(nprocs):
        marker = worker_marker_path(marker_tag, r)
        if os.path.exists(marker):
            os.remove(marker)


def local_worker(rank, queue, end_time, gpu_ids, marker_tag):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_ids[rank])
    torch.set_default_device("cpu")

    # Unsloth patching conflict workaround.
    if rank > 0:
        prev_marker = worker_marker_path(marker_tag, rank - 1)
        while not os.path.exists(prev_marker):
            time.sleep(5)

    from arc_solver import worker

    current_marker = worker_marker_path(marker_tag, rank)
    with open(current_marker, "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")
    worker(rank, queue, end_time)
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    os.environ.setdefault("PYTHONHASHSEED", str(env_int("ARC_PYTHONHASHSEED", 42)))

    task_order = env_str("ARC_TASK_ORDER", "lexicographic").strip().lower()
    keys = order_keys(data, task_order)

    key_filter = parse_key_list(os.getenv("ARC_KEYS"))
    if key_filter is not None:
        allowed = set(key_filter)
        keys = [k for k in keys if k in allowed]

    eval_default_keys = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]
    eval_keys = parse_key_list(os.getenv("ARC_EVAL_KEYS"))
    use_default_eval_subset = env_bool("ARC_USE_DEFAULT_EVAL_SUBSET", True)
    if not rerun_mode and use_default_eval_subset:
        keep = set(eval_default_keys if eval_keys is None else eval_keys)
        keys = [k for k in keys if k in keep]

    task_limit = env_int("ARC_TASK_LIMIT", 0)
    if task_limit > 0:
        keys = keys[:task_limit]

    skip_done = env_bool("ARC_SKIP_DONE", False)
    output_dir = env_str("ARC_OUTPUT_DIR", "/kaggle/inference_outputs")
    skip_done_mode = env_str("ARC_SKIP_DONE_MODE", "files").strip().lower()
    skip_done_min_files_per_test = max(1, env_int("ARC_SKIP_DONE_MIN_FILES_PER_TEST", 1))
    skip_done_min_unique_per_test = max(0, env_int("ARC_SKIP_DONE_MIN_UNIQUE_PER_TEST", 0))
    skip_done_min_candidates_per_test = max(0, env_int("ARC_SKIP_DONE_MIN_CANDIDATES_PER_TEST", 0))
    if skip_done:
        done_keys = infer_completed_keys(
            output_dir,
            data,
            min_files_per_test=skip_done_min_files_per_test,
            min_unique_per_test=skip_done_min_unique_per_test,
            min_candidates_per_test=skip_done_min_candidates_per_test,
            mode=skip_done_mode,
        )
        before = len(keys)
        keys = [k for k in keys if k not in done_keys]
        print(
            f"[Starter] skip_done={skip_done}, output_dir='{output_dir}', "
            f"mode={skip_done_mode}, "
            f"min_files_per_test={skip_done_min_files_per_test}, "
            f"min_unique_per_test={skip_done_min_unique_per_test}, "
            f"min_candidates_per_test={skip_done_min_candidates_per_test}, "
            f"done={before - len(keys)}, remaining={len(keys)}"
        )

    # Optional hard-focus pass: keep only the least-covered tasks among remaining keys.
    focus_hard_n = max(0, env_int("ARC_FOCUS_HARD_N", 0))
    if focus_hard_n > 0 and len(keys) > focus_hard_n:
        coverage = infer_key_coverage(output_dir, data)
        focus_mode = env_str("ARC_FOCUS_MODE", "files").strip().lower()
        # Quality mode prioritizes outputs with low diversity/candidate count, not just low file count.
        keys_sorted = sorted(
            keys,
            key=(
                (lambda k: (
                    coverage.get(k, {}).get("min_unique", 0),
                    coverage.get(k, {}).get("min_candidates", 0),
                    coverage.get(k, {}).get("min_files", 0),
                    coverage.get(k, {}).get("mean_unique", 0.0),
                    coverage.get(k, {}).get("mean_candidates", 0.0),
                    coverage.get(k, {}).get("mean_files", 0.0),
                    -task_complexity(data[k]),
                    k,
                ))
                if focus_mode == "quality"
                else (lambda k: (
                    coverage.get(k, {}).get("min_files", 0),
                    coverage.get(k, {}).get("mean_files", 0.0),
                    -task_complexity(data[k]),
                    k,
                ))
            ),
        )
        keys = keys_sorted[:focus_hard_n]
        print(
            f"[Starter] focus_hard_n={focus_hard_n}, focus_mode={focus_mode}, "
            f"selected={len(keys)} least-covered/lowest-quality tasks"
        )
        for k in keys[:12]:
            q = coverage.get(k, {})
            print(
                f"[Starter] focus {k}: files={q.get('min_files', 0)}/"
                f"{q.get('mean_files', 0.0):.1f}, unique={q.get('min_unique', 0)}/"
                f"{q.get('mean_unique', 0.0):.1f}, cand={q.get('min_candidates', 0)}/"
                f"{q.get('mean_candidates', 0.0):.1f}"
            )

    gpu_ids = parse_gpu_ids()
    max_procs = len(gpu_ids)
    requested_procs = env_int("ARC_NPROCS", max_procs)
    # Do not spawn more processes than pending tasks:
    # extra ranks still pay full model load cost before they can exit.
    nprocs = max(1, min(requested_procs, max_procs, len(keys)))
    gpu_ids = gpu_ids[:nprocs]

    os.environ.setdefault("ARC_WORLD_SIZE", str(nprocs))

    print(
        f"[Starter] tasks={len(keys)}, order={task_order}, "
        f"nprocs={nprocs}, gpu_ids={gpu_ids}, rerun={bool(rerun_mode)}"
    )

    if len(keys) == 0:
        print("[Starter] no tasks to run, exiting without spawning workers")
        raise SystemExit(0)

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    marker_tag = env_str("ARC_WORKER_MARKER_TAG", f"{int(time.time())}_{os.getpid()}")
    clear_worker_markers(marker_tag, nprocs)
    print(f"[Starter] worker_marker_tag={marker_tag}")

    mp.spawn(local_worker, args=(queue, args.end_time, gpu_ids, marker_tag), nprocs=nprocs)
