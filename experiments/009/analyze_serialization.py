"""Manually run CPU-only token-cost research; no model, training or evaluation gold.

Run from repository root: python3 experiments/009/analyze_serialization.py
Writes a separate 009 artifact; does not change training data or formatters.
"""
import hashlib
import json
import platform
import statistics
import sys
import time
from collections import Counter
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import arc_episode_data as data


def dump_grid(grid, mode):
    if mode == "space":
        return data.grid_to_text(grid)
    if mode == "compact":
        return "\n".join("".join(map(str, row)) for row in grid)
    lines = []
    for row in grid:
        runs = []
        for color in row:
            if runs and runs[-1][0] == color:
                runs[-1][1] += 1
            else:
                runs.append([color, 1])
        lines.append(" ".join("%d:%d" % tuple(run) for run in runs))
    return "\n".join(lines)


def read_grid(text, mode):
    if mode == "space":
        return data.text_to_grid(text)
    if mode == "compact":
        return [[int(c) for c in row] for row in text.splitlines()]
    rows = []
    for row in text.splitlines():
        cells = []
        for run in row.split():
            color, count = map(int, run.split(":"))
            cells.extend([color] * count)
        rows.append(cells)
    return rows


SYSTEM = {
    "space": data.SYSTEM_PROMPT,
    "compact": "Output only the answer grid. Each row is digits 0-9 without spaces. "
               "Put one row on each line. Do not write an explanation.",
    "rle": "Output only the answer grid in row run-length format. "
           "Write each run as color:count, separate runs by single spaces, "
           "and put one row on each line. Colors are digits 0-9. Do not write an explanation.",
}


def messages(demos, query, mode):
    blocks = []
    for i, pair in enumerate(demos):
        blocks.extend(["[demo %d] in:" % i, dump_grid(pair["input"], mode),
                       "[demo %d] out:" % i, dump_grid(pair["output"], mode)])
    blocks.extend(["[query] in:", dump_grid(query, mode), "[query] out:"])
    return [{"role": "system", "content": SYSTEM[mode]},
            {"role": "user", "content": "\n".join(blocks)}]


def describe(values):
    s = sorted(values)
    return {"n": len(s), "min": min(s), "median": statistics.median(s),
            "p90_nearest_rank": s[max(0, (9 * len(s) + 9) // 10 - 1)],
            "max": max(s), "sum": sum(s)}


def main():
    start = time.perf_counter()
    cache = Path.home() / ".cache/qwen38-tokenizer"
    cfg = json.loads((cache / "tokenizer_config.json").read_text())
    tok = PreTrainedTokenizerFast(
        tokenizer_file=str(cache / "tokenizer.json"), eos_token=cfg["eos_token"],
        pad_token=cfg["pad_token"], chat_template=(cache / "chat_template.jinja").read_text())
    encode = lambda text: tok.encode(text, add_special_tokens=False)
    paths = [ROOT / "experiments/007/raw/arc-prize-2026/arc-agi_training_challenges.json",
             ROOT / "experiments/007/raw/arc-prize-2026/arc-agi_training_solutions.json",
             ROOT / "experiments/007/g1/split.json",
             ROOT / "experiments/008/data/gen_newbases.jsonl",
             ROOT / "experiments/008/raw-eval/arc-agi_evaluation_challenges.json",
             ROOT / "experiments/008/e0_e24_24.json"]
    train, solutions, split = [json.loads(p.read_text()) for p in paths[:3]]
    groups = {"official_train_test_allquery": [], "new_generated_bases": [],
              "public_eval_prompt_only": []}
    for tid in split["train"]:
        for qi, pair in enumerate(train[tid]["test"]):
            groups["official_train_test_allquery"].append(
                (tid, "test:%d" % qi, train[tid]["train"], pair["input"], solutions[tid][qi]))
    for line in paths[3].read_text().splitlines():
        b = json.loads(line)
        groups["new_generated_bases"].append(
            (b["task_id"], b["base_instance_id"], b["demos"], b["query_input"], b["target"]))
    public = json.loads(paths[4].read_text())
    e24 = {x["task_id"] for x in json.loads(paths[5].read_text())["selected24"]}
    for tid, task in sorted(public.items()):
        for qi, pair in enumerate(task["test"]):
            if "output" in pair:
                raise ValueError("Public evaluation file unexpectedly contains query outputs")
            groups["public_eval_prompt_only"].append(
                (tid, "query:%d" % qi, task["train"], pair["input"], None))
    rows, checks = [], Counter()
    for group, items in groups.items():
        for tid, bid, demos, query, target in items:
            rec = {"group": group, "task_id": tid, "base_id": bid, "formats": {}}
            if group == "public_eval_prompt_only":
                rec["e24_selected"] = tid in e24
            grids = [g for d in demos for g in [d["input"], d["output"]]] + [query]
            if target is not None:
                grids.append(target)
            for mode in SYSTEM:
                for grid in grids:
                    if read_grid(dump_grid(grid, mode), mode) != grid:
                        raise ValueError("Lossless roundtrip failed")
                    checks[mode] += 1
                msgs = messages(demos, query, mode)
                if mode == "space" and msgs != data.build_messages(demos, query):
                    raise ValueError("Space reference differs from current formatter")
                prompt = tok.apply_chat_template(msgs, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
                p = len(encode(prompt))
                t = len(encode(dump_grid(target, mode))) + 1 if target is not None else None
                rec["formats"][mode] = {"prompt": p, "target_eos": t,
                                        "full": p + t if t is not None else None}
            rows.append(rec)
    summaries = {}
    for group in groups:
        subsets = {group: [r for r in rows if r["group"] == group]}
        if group == "public_eval_prompt_only":
            subsets["e24_prompt_only"] = [r for r in subsets[group] if r["e24_selected"]]
        for label, subset in subsets.items():
            out = {"rows": len(subset), "tasks": len({r["task_id"] for r in subset}), "formats": {}}
            for mode in SYSTEM:
                stats = {}
                for field in ["prompt", "target_eos", "full"]:
                    vals = [r["formats"][mode][field] for r in subset if r["formats"][mode][field] is not None]
                    if vals:
                        stats[field] = describe(vals)
                key = "full" if "full" in stats else "prompt"
                stats["limit_basis"] = key
                stats["coverage"] = {str(limit): {
                    "rows": sum(r["formats"][mode][key] <= limit for r in subset),
                    "tasks_with_one_fitting_row": len({r["task_id"] for r in subset
                                                       if r["formats"][mode][key] <= limit})}
                    for limit in [1024, 2048, 4096, 8192]}
                out["formats"][mode] = stats
            summaries[label] = out
    probe = {s: encode(s) for s in ["0123456789", "000000000000000000000000000000", "9876543210"]}
    output = {"date": "2026-10-01", "purpose": "CPU serialization cost; no model quality measured",
              "hardware": platform.machine(), "python": sys.version,
              "tokenizer_revision": "1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0",
              "tokenizer_sha256": hashlib.sha256((cache / "tokenizer.json").read_bytes()).hexdigest(),
              "inputs_sha256": {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
              "public_eval_query_outputs_read": False, "grid_roundtrips_by_format": dict(checks),
              "format_system_prompts": SYSTEM, "digit_token_probe": probe,
              "summaries": summaries, "rows": rows, "elapsed_seconds": time.perf_counter() - start}
    dest = ROOT / "experiments/009/serialization-analysis.json"
    dest.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"summaries": summaries, "elapsed_seconds": output["elapsed_seconds"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
