import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def _safe_mean(values):
    return float(np.mean(values)) if len(values) else 0.0


def _grid_signature(guess):
    arr = np.asarray(guess)
    shape = tuple(arr.shape)
    flat = tuple(arr.reshape(-1).tolist())
    return shape, flat


def _is_base_guess(sample):
    source = f"{sample.get('_source_key', '')}{sample.get('_source_run', '')}"
    return ".base." in source or ".base.run" in source or source.endswith(".base")


def _unique_solution_count(values):
    return len({hashable(sample["solution"]) for sample in values.values()})

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    ranked = []
    for sc, output in scores.values():
        metric = float(getter(sc))
        support = len(sc)
        beam_best = float(np.min([g["beam_score"] for g in sc])) if support else float("inf")
        aug_mean = _safe_mean([_safe_mean(g["score_aug"]) for g in sc])
        # Deterministic tie-breaks reduce submission variance from file iteration order.
        ranked.append((metric, support, beam_best, aug_mean, _grid_signature(output), output))
    ranked = sorted(ranked, key=lambda x: (-x[0], -x[1], x[2], x[3], x[4]))
    ordered_outputs = [x[-1] for x in ranked]
    return ordered_outputs

def getter_full_probmul_3(guesses, baseline=3):
    inf_score = np.sum([baseline-g["beam_score"] for g in guesses])
    aug_score = np.mean([np.sum([baseline-s for s in g["score_aug"]]) for g in guesses])
    return inf_score + aug_score

def score_full_probmul_3(guesses):
    return score_sum(guesses, getter_full_probmul_3)

def getter_kgmon(guesses):
    inf_score = len(guesses)
    aug_score = np.mean([np.mean(g["score_aug"]) for g in guesses])
    return inf_score - aug_score

def score_kgmon(guesses):
    return score_sum(guesses, getter_kgmon)

def getter_hybrid_consensus(guesses):
    support = len(guesses)
    beam_scores = [g["beam_score"] for g in guesses]
    aug_scores = [_safe_mean(g["score_aug"]) for g in guesses]
    beam_best = float(np.min(beam_scores))
    beam_mean = _safe_mean(beam_scores)
    aug_mean = _safe_mean(aug_scores)
    aug_std = float(np.std(aug_scores)) if len(aug_scores) else 0.0
    return (2.5 * support) - (1.2 * beam_best) - (0.6 * beam_mean) - (0.8 * aug_mean) - (0.1 * aug_std)

def score_hybrid_consensus(guesses):
    return score_sum(guesses, getter_hybrid_consensus)

def getter_round_consensus(guesses):
    support = len(guesses)
    unique_rounds = len({g.get("_round", -1) for g in guesses})
    unique_runs = len({g.get("_source_run", "") for g in guesses})
    beam_scores = [g["beam_score"] for g in guesses]
    aug_scores = [_safe_mean(g["score_aug"]) for g in guesses]
    beam_best = float(np.min(beam_scores))
    beam_mean = _safe_mean(beam_scores)
    aug_mean = _safe_mean(aug_scores)
    return (2.8 * unique_rounds) + (1.4 * unique_runs) + (1.2 * support) - (1.0 * beam_best) - (0.5 * beam_mean) - (0.8 * aug_mean)

def score_round_consensus(guesses):
    return score_sum(guesses, getter_round_consensus)


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
    score_hybrid_consensus,
    score_round_consensus,
]


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        loaded_files = 0
        for key in sorted(os.listdir(store)):
            path = os.path.join(store, key)
            if not os.path.isfile(path):
                continue
            try:
                with bz2.BZ2File(path) as f:
                    outputs = pickle.load(f)
            except Exception:
                # Ignore non-bz2 / unreadable files in the output directory.
                continue
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                enriched = dict(sample)
                enriched["_source_key"] = key
                enriched["_source_run"] = run_name or ".run0"
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = enriched
            loaded_files += 1
        print(f"*** Loaded {loaded_files} decoded files from '{store}'")

    @staticmethod
    def merge_rankings(rankings):
        merged = {}
        for ranking in rankings:
            n = max(1, len(ranking))
            for rank, guess in enumerate(ranking):
                h = hashable(guess)
                item = merged.get(h, {"guess": guess, "score": 0.0, "best_rank": rank})
                item["score"] += (n - rank)
                item["best_rank"] = min(item["best_rank"], rank)
                merged[h] = item
        ordered = sorted(merged.values(), key=lambda x: (x["score"], -x["best_rank"]), reverse=True)
        return [x["guess"] for x in ordered]

    def run_selection_ensemble(self, selection_algorithms_local=None):
        algos = selection_algorithms if selection_algorithms_local is None else selection_algorithms_local
        selected = {}
        for bk, values in self.decoded_results.items():
            guesses = {k: g for k, g in values.items()}
            rankings = [algo(guesses) for algo in algos]
            selected[bk] = self.merge_rankings(rankings)
        return selected

    def run_selection_dual(self, primary_algorithm=score_kgmon, secondary_algorithm=score_full_probmul_3):
        selected = {}
        for bk, values in self.decoded_results.items():
            guesses = {k: g for k, g in values.items()}
            primary = primary_algorithm(guesses)
            secondary = secondary_algorithm(guesses)
            merged = []
            seen = set()

            def append_unique(candidates):
                for guess in candidates:
                    h = hashable(guess)
                    if h not in seen:
                        seen.add(h)
                        merged.append(guess)

            if primary:
                append_unique([primary[0]])
            if secondary:
                append_unique([secondary[0]])
            append_unique(primary[1:])
            append_unique(secondary[1:])
            selected[bk] = merged
        return selected

    def run_selection_dual_round(self):
        return self.run_selection_dual(primary_algorithm=score_round_consensus, secondary_algorithm=score_kgmon)

    def run_selection_guarded_dual(
        self,
        primary_algorithm=score_kgmon,
        secondary_algorithm=score_full_probmul_3,
        min_base_candidates=1,
        min_base_unique=1,
    ):
        selected = {}
        for bk, values in self.decoded_results.items():
            all_guesses = {k: g for k, g in values.items()}
            base_guesses = {k: g for k, g in all_guesses.items() if _is_base_guess(g)}
            use_base_primary = (
                len(base_guesses) >= min_base_candidates
                and _unique_solution_count(base_guesses) >= min_base_unique
            )
            primary_guesses = base_guesses if use_base_primary else all_guesses

            primary = primary_algorithm(primary_guesses)
            secondary = secondary_algorithm(all_guesses)
            merged = []
            seen = set()

            def append_unique(candidates, max_new=None):
                added = 0
                for guess in candidates:
                    h = hashable(guess)
                    if h in seen:
                        continue
                    seen.add(h)
                    merged.append(guess)
                    added += 1
                    if max_new is not None and added >= max_new:
                        break

            # Attempt 1 stays anchored to kgmon, preferably using the base pass only.
            append_unique(primary[:1])
            # Attempt 2 is the best full-probability alternative, scanning past duplicates.
            append_unique(secondary, max_new=1)
            append_unique(primary[1:])
            append_unique(secondary)
            selected[bk] = merged
        return selected

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

    def run_selection_diverse_dual(
        self,
        primary_algorithm=score_kgmon,
        secondary_algorithm=score_full_probmul_3,
        min_base_candidates=1,
        min_base_unique=1,
    ):
        """Like guarded_dual but attempt_2 prefers solutions from diverse augmentation sources.

        When multiple candidates have similar scores, the one that appears across more
        distinct geometric transformations (rot90, transpose, flip_lr, ...) is preferred.
        """
        selected = {}
        for bk, values in self.decoded_results.items():
            all_guesses = {k: g for k, g in values.items()}
            base_guesses = {k: g for k, g in all_guesses.items() if _is_base_guess(g)}
            use_base_primary = (
                len(base_guesses) >= min_base_candidates
                and _unique_solution_count(base_guesses) >= min_base_unique
            )
            primary_guesses = base_guesses if use_base_primary else all_guesses

            primary = primary_algorithm(primary_guesses)

            # Build diversity-augmented scoring for secondary ranking
            by_solution = {}
            for k, g in all_guesses.items():
                h = hashable(g["solution"])
                entry = by_solution.get(h, {"samples": [], "solution": g["solution"]})
                entry["samples"].append(g)
                # Extract transformation family from source key
                source = g.get("_source_key", "")
                for op in source.split("."):
                    if op in ("transpose", "rot90", "flip_lr", "flip_ud", "permute", "colors"):
                        entry.setdefault("ops", set()).add(op.split("0")[0] if op.startswith("rot") else op)
                by_solution[h] = entry

            # Diversity-aware secondary ordering
            diverse_ranked = []
            for h, entry in by_solution.items():
                samples = entry["samples"]
                solution = entry["solution"]
                n_sources = len(entry.get("ops", set()))
                support = len(samples)
                beam_best = float(np.min([g["beam_score"] for g in samples])) if support else float("inf")
                aug_mean = _safe_mean([_safe_mean(g["score_aug"]) for g in samples])
                # Score: full_probmul_3 metric + diversity bonus from distinct augmentation ops
                probmul_score = getter_full_probmul_3(samples)
                diversity_bonus = n_sources * 0.5
                metric = probmul_score + diversity_bonus
                diverse_ranked.append((metric, support, beam_best, aug_mean, _grid_signature(solution), solution))
            diverse_ranked = sorted(diverse_ranked, key=lambda x: (-x[0], -x[1], x[2], x[3], x[4]))
            secondary = [x[-1] for x in diverse_ranked]

            merged = []
            seen = set()

            def append_unique(candidates, max_new=None):
                added = 0
                for guess in candidates:
                    h = hashable(guess)
                    if h in seen:
                        continue
                    seen.add(h)
                    merged.append(guess)
                    added += 1
                    if max_new is not None and added >= max_new:
                        break

            append_unique(primary[:1])
            append_unique(secondary, max_new=1)
            append_unique(primary[1:])
            append_unique(secondary)
            selected[bk] = merged
        return selected

    def benchmark_selection_algos(self):
        print("*** Benchmark selection algorithms...")

        labels = {}
        num_tasks_per_puzzle = {}
        num_solved_keys = 0
        num_total_keys = 0

        correct_beam_scores = []

        for basekey, basevalues in self.decoded_results.items():

            mult_key, mult_sub = basekey.split("_")
            num_tasks_per_puzzle[mult_key] = max(num_tasks_per_puzzle.get(mult_key, 0), int(mult_sub) + 1)

            labels[basekey] = correct_solution = self.dataset.replies[basekey][0]

            for subkey, sample in basevalues.items():

                solution = sample["solution"]
                beam_score = sample["beam_score"]
                aug_mean = np.mean(sample["score_aug"])

                if np.shape(correct_solution) != np.shape(solution):
                    corr_str = "bad_xy_size"
                elif np.array_equal(correct_solution, solution):
                    corr_str = "ALL_CORRECT"
                    num_solved_keys += 1
                    correct_beam_scores.append(beam_score)
                else:
                    corr_str = "bad_content"

                output_len = f"{solution.shape[0]}x{solution.shape[1]}"

                if corr_str == "ALL_CORRECT":
                    print(f"{corr_str}:{beam_score:8.5f} - {aug_mean:8.5f} {output_len:5s} [{subkey}]")
                num_total_keys += 1

        if num_total_keys == 0:
            print(" subkeys: 0/0")
            print(" avg correct beam score:      nan")
            print(" max correct beam score:      nan")
            print(" [WARN] No decoded outputs loaded; skip benchmark.")
            return

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if len(correct_beam_scores):
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")
        else:
            print(" avg correct beam score:      nan")
            print(" max correct beam score:      nan")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")

        selected_ensemble = self.run_selection_ensemble()
        correct_puzzles = {k for k, v in selected_ensemble.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
        score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
        print(f" acc: {score:5.1f}/{num_puzzles:3} ('ensemble')")

        selected_dual = self.run_selection_dual()
        correct_puzzles = {k for k, v in selected_dual.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
        score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
        print(f" acc: {score:5.1f}/{num_puzzles:3} ('dual')")

        selected_guarded_dual = self.run_selection_guarded_dual()
        correct_puzzles = {k for k, v in selected_guarded_dual.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
        score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
        print(f" acc: {score:5.1f}/{num_puzzles:3} ('guarded_dual')")

        selected_diverse_dual = self.run_selection_diverse_dual()
        correct_puzzles = {k for k, v in selected_diverse_dual.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
        score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
        print(f" acc: {score:5.1f}/{num_puzzles:3} ('diverse_dual')")
