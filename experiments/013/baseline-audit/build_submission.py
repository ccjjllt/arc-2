import os
import json
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import (
    ArcDecoder,
    score_kgmon,
    score_full_probmul_3,
    score_hybrid_consensus,
    score_round_consensus,
)


def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def parse_dirs(raw, default):
    if raw is None or raw.strip() == "":
        return [default]
    return [x.strip() for x in raw.split(",") if x.strip()]


def write_validation_report(path, split_data, decoder, selected_results):
    rows = []
    for key in split_data.keys:
        label = split_data.replies.get(key, [None])[0]
        decoded = decoder.decoded_results.get(key, {})
        candidates = list(decoded.values())
        selected = selected_results.get(key, [])

        correct_candidates = []
        if label is not None:
            for sample in candidates:
                if np.array_equal(sample["solution"], label):
                    correct_candidates.append(sample)

        attempt_correct = []
        if label is not None:
            attempt_correct = [bool(np.array_equal(guess, label)) for guess in selected[:2]]
        while len(attempt_correct) < 2:
            attempt_correct.append(False)

        # Keep the hashing explicit; candidate grids are small and this is report-only.
        unique_candidates = set()
        for sample in candidates:
            try:
                unique_candidates.add(tuple(tuple(int(x) for x in row) for row in np.asarray(sample["solution"]).tolist()))
            except Exception:
                pass

        rows.append({
            "key": key,
            "num_candidates": len(candidates),
            "num_unique_candidates": len(unique_candidates),
            "candidate_hit": len(correct_candidates) > 0,
            "attempt_1_correct": attempt_correct[0],
            "attempt_2_correct": attempt_correct[1],
            "pass2_correct": any(attempt_correct),
            "best_correct_beam": (
                min(float(sample["beam_score"]) for sample in correct_candidates)
                if correct_candidates else None
            ),
        })

    summary = {
        "outputs": len(rows),
        "candidate_hit": sum(1 for row in rows if row["candidate_hit"]),
        "pass2_correct": sum(1 for row in rows if row["pass2_correct"]),
        "attempt_1_correct": sum(1 for row in rows if row["attempt_1_correct"]),
        "attempt_2_correct": sum(1 for row in rows if row["attempt_2_correct"]),
        "zero_candidate_outputs": sum(1 for row in rows if row["num_candidates"] == 0),
        "rows": rows,
    }
    with open(path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"*** Validation report saved to '{path}'")
    print(
        "*** Validation report summary:",
        f"pass2={summary['pass2_correct']}/{summary['outputs']}",
        f"candidate_hit={summary['candidate_hit']}/{summary['outputs']}",
        f"zero_candidates={summary['zero_candidate_outputs']}",
    )


def main():
    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")
    n_guesses = max(1, env_int("ARC_N_GUESSES", 2))
    inference_dirs = parse_dirs(os.getenv("ARC_INFERENCE_DIRS"), "/kaggle/inference_outputs")
    output_path = os.getenv("ARC_SUBMISSION_PATH", "/kaggle/working/submission.json")
    selection_mode = os.getenv("ARC_SELECTION_MODE", "kgmon").strip().lower()

    if rerun_mode:
        data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
    else:
        data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
        data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

    split_data = data.split_multi_replies()
    decoder = ArcDecoder(split_data, n_guesses=n_guesses)

    loaded = 0
    for i, store in enumerate(inference_dirs):
        if not os.path.isdir(store):
            print(f"[WARN] Skip missing inference dir: {store}")
            continue
        run_name = f".run{i}"
        print(f"*** Loading decoded outputs from '{store}' ({run_name})")
        decoder.load_decoded_results(store, run_name=run_name)
        loaded += 1

    if loaded == 0:
        raise RuntimeError("No inference outputs loaded. Check ARC_INFERENCE_DIRS.")
    if len(decoder.decoded_results) == 0:
        raise RuntimeError("Inference directory exists but no decoded result files were parsed.")

    if selection_mode == "kgmon":
        selected_results = decoder.run_selection_algo(score_kgmon)
    elif selection_mode == "full":
        selected_results = decoder.run_selection_algo(score_full_probmul_3)
    elif selection_mode == "hybrid":
        selected_results = decoder.run_selection_algo(score_hybrid_consensus)
    elif selection_mode == "round":
        selected_results = decoder.run_selection_algo(score_round_consensus)
    elif selection_mode == "ensemble":
        selected_results = decoder.run_selection_ensemble()
    elif selection_mode == "dual":
        selected_results = decoder.run_selection_dual()
    elif selection_mode == "dual_round":
        selected_results = decoder.run_selection_dual_round()
    elif selection_mode == "guarded_dual":
        selected_results = decoder.run_selection_guarded_dual()
    elif selection_mode == "diverse_dual":
        selected_results = decoder.run_selection_diverse_dual()
    else:
        print(f"[WARN] Unknown ARC_SELECTION_MODE='{selection_mode}', fallback to kgmon")
        selected_results = decoder.run_selection_algo(score_kgmon)

    submission = data.get_submission(selected_results)

    with open(output_path, "w") as f:
        json.dump(submission, f)
    print(f"*** Submission saved to '{output_path}' (selection_mode={selection_mode})")

    if not rerun_mode:
        report_path = os.getenv("ARC_REPORT_PATH", "").strip()
        if report_path:
            write_validation_report(report_path, split_data, decoder, selected_results)
        decoder.benchmark_selection_algos()
        with open(output_path, "r") as f:
            reload_submission = json.load(f)
        print("*** Reload score:", data.validate_submission(reload_submission))


if __name__ == "__main__":
    main()
