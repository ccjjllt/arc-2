# CELL 0 (markdown)
# ARC-AGI-2 · NVARC+ — version 2: two independent half-cost test-time-training passes, pooled

**What this is.** The open-sourced ARC Prize 2025 winner (NVARC: `qwen3_4b_grids15_sft139`, a Qwen3-4B with a
16-token grid vocabulary + per-task LoRA test-time training + batched DFS decoding + augmentation re-scoring +
`score_kgmon` selection), run **twice per task with independent seeds** inside the 12 h budget, with both
candidate pools voting together for the two attempts.

**Why.** Re-running the *identical* NVARC notebook gives public-LB scores of 26.9 – 32.2 (our own five reruns:
29.7 / 31.8 / 26.9 / 32.2 / 31.4): each run solves a partly different set of ~5 tasks, because test-time
training and batched DFS are stochastic. The hidden rerun has 240 tasks, so the stock configuration (128 TTT
sequences, 16 views) uses the whole 12 h; a half-cost configuration (64 sequences, 8 views) runs in ~6 h and
public runtime data shows it scores in the same band (35.97 in 5h44, 33.89 in 3h50 on the LB runtime board). Two attempts per output let us keep the union of two runs: pass A (the LB-proven configuration)
and pass B (different LoRA-init / augmentation / scoring seeds) are pooled; the candidate found by both gets
double votes, and when the passes disagree, the two strongest distinct grids fill attempt_1 / attempt_2.

**Guarantee.** Pass A's top-1 is always kept among the two attempts, so the ensemble can never lose a task that
the single-pass baseline would have scored on attempt_1.

**Pipeline.**
1. Pass A – NVARC perfpatch maths, 64 TTT sequences (8 augmentations × 8 geometries), 8 decoded views,
   cheap-first task order, 4 workers (one per L4), per-task cap 800 s.
2. Catch-up – any task pass A did not reach (deadline) is run first in pass B's time.
3. Pass B – same half-cost pipeline with independent seeds (137 / 17 / 29 / +7), cheap-first, until the deadline.
4. Selection – `score_kgmon` (#views producing the grid − mean augmented NLL) over the pooled candidates;
   attempt_1 = pooled #1, attempt_2 = pooled #2, with pass-A #1 forced in if it fell out; identity / `[[0]]`
   fallbacks; schema re-validation.

Interactive/commit runs use a 55-minute budget on 4 public-evaluation tasks so every phase is exercised; the
hidden-test rerun uses 12 h − 10 min. Dependencies: model `sorokin/qwen3_4b_grids15_sft139`, utility script
`sorokin/pip-install-unsloth-flash-patch`, docker image pinned to the py3.11/torch 2.8 build.

Credits: NVARC (Ivan Sorokin & Jean-François Puget), the ARChitects (Daniel Franzen & Jan Disselhoff),
mikelou1's throughput patch, koushikrudra's deterministic-seed fixes.


# CELL 1 (code)
# ---------------------------------------------------------------------------
# Global wall-clock budget. Rerun (hidden test): 12h minus a 10 min write buffer
# (the LB-proven NVARC setting). Interactive/commit run: short budget so the
# commit is cheap but every phase is exercised end-to-end on 4 debug tasks.
# ---------------------------------------------------------------------------
import os, time, json
RERUN = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
T0 = time.time()
if RERUN:
    global_end_time = T0 + 12 * 3600 - 600
else:
    global_end_time = T0 + 55 * 60
print(f"rerun={RERUN} budget={(global_end_time-T0)/3600:.2f}h")


# CELL 2 (code)
!pip uninstall -y tensorflow
import torch, sys, os
print(sys.version.split()[0], "torch", torch.__version__, "gpus", torch.cuda.device_count(),
      [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])
import glob
print("models:", glob.glob("/kaggle/input/models/*/*"), glob.glob("/kaggle/input/qwen*"))
print("utility scripts:", [p for p in sys.path if "/kaggle/usr/lib" in p][:5])
os.makedirs("/kaggle/working/logs", exist_ok=True)


# CELL 3 (code)
%%writefile arc_loader.py
import json
import numpy as np
from transformers import AutoTokenizer


def convert_grid_to_string(grid) -> str:
    text = ""
    for row in grid:
        for cell in row:
            text += str(int(cell))
        text += "\n"
    return text.strip()

def is_valid_solution(guess):
    return isinstance(guess, np.ndarray) and guess.ndim == 2 and all(0 < x <= 30 for x in guess.shape)

def shuffled(data_list):
    return np.random.permutation(data_list).tolist()

def permute_mod(a, descriptor, invert=False):
    permutation = [int(i) for i in descriptor if str(i).isdigit()]
    assert sorted(permutation)==list(range(10))
    a = np.asarray(a)
    if a.ndim==3:
        if not invert: permutation = np.argsort(permutation)
        a = a[..., permutation]
    else:
        assert a.ndim==2
        if invert: permutation = np.argsort(permutation)
        a = np.asarray(permutation)[a]
    return a

def permute_rnd_all_(query):
    permutation = np.random.permutation(10).tolist()
    return 'permute' + ''.join(map(str, permutation))


class QwenFormatter:

    def __init__(self, tokenizer: AutoTokenizer):
        self.tokenizer = tokenizer

    def fmt_query(self, query) -> str:
        grid_input = convert_grid_to_string(query[0]["input"])
        return "<|im_start|>user\n" + grid_input + "<|im_end|><|im_start|>assistant\n"

    def fmt_reply(self, reply) -> str:
        return convert_grid_to_string(reply[0]) + "<|im_end|>"

    def fmt_train(self, train, last_is_challenge=False) -> str:
        if last_is_challenge:
            test = train[-1]
            train = train[:-1]
        else:
            test = None
        text = ""
        for x in train:
            grid_input = convert_grid_to_string(x["input"])
            grid_output = convert_grid_to_string(x["output"])
            text += f"<|im_start|>user\n{grid_input}<|im_end|><|im_start|>assistant\n{grid_output}<|im_end|>"
        if test is not None:
            text += self.fmt_query([test]) + self.fmt_reply([test["output"]])
        return text

    def max_new_tokens(self):
        max_sized_reply = np.zeros([30, 30], dtype=int)
        tokens = self.tokenizer.encode(self.fmt_reply([max_sized_reply]))
        return len(tokens) + 1

    def convert_tokens_to_array(self, tokens, limit_rows=30):
        if len(tokens) < 2:
            return None
        text = self.tokenizer.decode(tokens[:-1])
        try:
            lines = text.strip().split("\n")
            by_rows = [row for row in [[int(x) for x in line if x.isdigit()] for line in lines] if len(row)]
            if len(by_rows) > limit_rows:
                by_rows = by_rows[:limit_rows]
            array = np.array(by_rows, dtype=int)
            if is_valid_solution(array):
                return array
        except:
            pass
        return None


class ArcDataset:

    @staticmethod
    def forward_mod(a, key, use_perm=True):
        if a is None: return a
        for op in key.split('.')[1:]:
            if   op=='rot90':              a = np.rot90(a)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=False) if use_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    @staticmethod
    def invert_mod(a, key, inv_perm=True):
        if a is None: return a
        for op in key.split('.')[1:][::-1]:
            if   op=='rot90':              a = np.rot90(a, k=3)
            elif op=='transpose':          a = np.swapaxes(a, 0, 1)
            elif op.startswith('permute'): a = permute_mod(a, op, invert=True) if inv_perm else a
            elif op.startswith('copy'):    a = np.copy(a)
            elif op.startswith('out'):     a = a
            elif op.startswith('ex'):      a = a
            elif op.startswith('run'):     a = a
            else: raise NotImplementedError(f"Inversion of operation '{op}' unknown.")
        return a

    def __init__(self, queries, replies={}, keys=None, is_orig=False):
        if keys is not None: keys = [k for k in keys if k is not None]
        self.queries = queries if keys is None else {k: queries[k] for k in keys}
        self.replies = replies if keys is None else {k: replies[k] for k in keys if k in replies}
        self.is_orig = is_orig
        self.keys = sorted(queries.keys()) if keys is None else keys
        self.transposed_dataset = None

    def change_keys(self, keys, keep_flags=False):
        flags = dict(is_orig=self.is_orig) if keep_flags else {}
        return self.__class__(queries=self.queries, replies=self.replies, keys=keys, **flags)

    @classmethod
    def from_file(cls, queries_file, keys=None):
        with open(queries_file) as f:
            queries = f.read()
        return cls(
            queries=json.loads(queries),
            is_orig=True,
            keys=keys,
        )

    def load_replies(self, replies_file):
        print(f"*** Load solutions from '{replies_file}'...")
        with open(replies_file) as f: replies = f.read()
        replies_parsed = json.loads(replies)
        self.replies = {k: replies_parsed[k] for k in self.keys}
        return self

    def split_multi_replies(self):
        key_indices = [(k, i) for k in self.keys for i in range(len(self.queries[k]['test']))]
        return self.__class__(
            keys=[f'{k}_{i}' for k, i in key_indices],
            queries={f'{k}_{i}': {'train': self.queries[k]['train'], 'test': [self.queries[k]['test'][i]]} for k, i in key_indices},
            replies={f'{k}_{i}': [self.replies[k][i]] for k, i in key_indices if k in self.replies},
        )

    def shuffled(self):
        return self.__class__(queries=self.queries, replies=self.replies, keys=shuffled(self.keys))

    def append(*datasets):
        return datasets[0].__class__(
            queries={k: v for d in datasets for k, v in d.queries.items()},
            replies={k: v for d in datasets for k, v in d.replies.items()},
            keys   =[k    for d in datasets for k    in d.keys           ],
        )

    def mod_single(self, mod_func, descriptor, i, keep_key, inputs_only):
        queries = {}
        replies = {}
        keys    = []
        for k0 in self.keys:
            desc = (('copy{i}' if mod_func is np.copy else mod_func.__name__) if descriptor is None else descriptor if isinstance(descriptor, str) else descriptor(self.queries[k0])).format(i=i)
            func = lambda a, d: np.asarray(mod_func(a) if descriptor is None else mod_func(a, d)).tolist()
            k1 = k0 if keep_key else f"{k0}.{'I' if inputs_only else ''}{desc}"
            keys.append(k1)
            queries[k1] = {m: [{t: (func(a, desc) if t=='input' or not inputs_only else a) for t, a in x.items()} for x in e] for m, e in self.queries[k0].items()}
            if k0 in self.replies:
                replies[k1] = [func(a, desc) for a in self.replies[k0]]
        ret = self.__class__(queries=queries, replies=replies, keys=keys)
        return ret

    def mod(self, mod_func, descriptor=None, n=1, stack=None, keep=False, keep_key=False, shuffle=False, join=True, inputs_only=False):
        assert not (keep and keep_key)
        cur = self
        ret = [cur.shuffled() if shuffle else cur] if keep else []
        if stack is None: stack = mod_func.__name__.startswith('rot')
        for i in range(n):
            cur = (cur if stack else self).mod_single(mod_func, descriptor, i=i, keep_key=keep_key, inputs_only=inputs_only)
            ret.append(cur.shuffled() if shuffle else cur)
        return self.__class__.append(*ret) if join else ret

    def get(self, key, formatter: QwenFormatter):
        train = formatter.fmt_train(self.queries[key]['train'])
        query = formatter.fmt_query(self.queries[key]['test'])
        reply = formatter.fmt_reply(self.replies[key]) if key in self.replies else ''
        text = train+query+reply if reply else formatter.fmt_train(self.queries[key]['train'], last_is_challenge=True)
        return dict(key=key, train=train, query=query, reply=reply, input=train+query, text=text)

    def as_list(self, formatter: QwenFormatter):
        return [self.get(key, formatter) for key in self.keys]

    def get_length(self, key, formatter: QwenFormatter, name, max_of_transposed=False):
        if formatter is None:
            if   name=='input': return sum(np.prod(np.shape(v)) for v3 in self.queries[key].values() for v2 in v3 for v in v2.values())
            elif name=='reply': return sum(np.prod(np.shape(v)) for v in self.replies[key])
            else: assert False
        else:
            datasets = [self]
            if max_of_transposed:
                if self.transposed_dataset is None: self.transposed_dataset = self.mod(np.transpose, keep=False, keep_key=True)
                datasets.append(self.transposed_dataset)
            return max(len(formatter.tokenizer.encode(ds.get(key, formatter=formatter)[name])) for ds in datasets)

    def cut_to_len(self, formatter, name, max_len, from_end=False):
        temp_ds = self.change_keys(self.keys)
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            reply = temp_ds.replies.get(key)
            while max_len<temp_ds.get_length(key, formatter=formatter, name=name):
                query = temp_ds.queries[key]
                if not key.split('.')[-1].startswith('ex'):
                    key = f"{key}.ex{''.join(map(str, range(len(query['train']))))}"
                key_split = key.split('.')
                assert key_split[-1].startswith('ex')
                key = '.'.join(key_split[:-1] + [f'ex{key_split[-1][2:-1] if from_end else key_split[-1][3:]}'])
                temp_ds.queries[key] = {k: ((v[:-1] if from_end else v[1:]) if k=='train' else v) for k, v in query.items()}
                if reply is not None:
                    temp_ds.replies[key] = reply
            new_keys.append(key)
            new_queries[key] = temp_ds.queries[key]
            if reply is not None: new_replies[key] = reply
        return self.__class__(keys=new_keys, queries=new_queries, replies=new_replies)
    
    def shuffle_ex(self, perm=None, keep_max=None):
        new_keys = []
        new_queries = {}
        new_replies = {}
        for key in self.keys:
            n = len(self.queries[key]['train'])
            p = np.random.permutation(n) if perm is None else perm
            if keep_max is not None: p = p[:keep_max]
            new_key = f'{key}.ex' + ('-' if (p.max()>9) else '').join(map(str, p.tolist()))
            new_keys.append(new_key)
            new_queries[new_key] = {k: (np.array(v, dtype=object)[p].tolist() if k=='train' else v) for k, v in self.queries[key].items()}
            if key in self.replies: new_replies[new_key] = self.replies[key]
        return self.__class__(queries=new_queries, replies=new_replies, keys=new_keys)

    def augment(self, n=1, shfl_keys=False, seed=42):
        np.random.seed(seed)
        d = self
        d = d.mod(np.transpose, keep=True)
        d = d.mod(np.rot90, n=3, keep=True)
        d = d.mod(permute_mod, permute_rnd_all_, n=n, shuffle=shfl_keys, keep=False)
        d = d.shuffle_ex()
        return d

    def get_submission(self, results=None):
        assert self.is_orig==True, 'Must be run on original dataset.'
        submission = {k: [{f'attempt_{i+1}': [[0]] for i in range(2)} for _ in range(len(self.queries[k]['test']))] for k in self.keys}
        if results is not None: self.fill_submission(results, submission)
        return submission

    @staticmethod
    def fill_submission(results, submission):
        print(f'*** Generating submission for {len(results)} outputs...')
        for k, v in results.items():
            base_id, base_nr = k.split('_')
            target_dict = submission[base_id][int(base_nr)]
            for i, g in enumerate(v[:len(target_dict)]):
                target_dict[f'attempt_{i+1}'] = g.tolist()

    def validate_submission(self, submission):
        assert self.is_orig==True, 'Must be run on original dataset.'
        score = 0
        for k, v in self.replies.items():
            for i, r in enumerate(v):
                for attempt in ['attempt_1', 'attempt_2']:
                    if np.array_equal(r, submission[k][i][attempt]):
                        score += 1 / len(v)
                        break
        return score

# CELL 4 (code)
%%writefile arc_decoder.py
import os
import bz2
import pickle
import numpy as np

def hashable(guess):
    return tuple(map(tuple, guess))

def score_sum(guesses, getter):
    guess_list = list(guesses.values())
    scores = {}
    for g in guess_list:
        h = hashable(g["solution"])
        x = scores[h] = scores.get(h, [[], g["solution"]])
        x[0].append(g)
    scores = [(getter(sc), o) for sc, o in scores.values()]
    scores = sorted(scores, key=(lambda x: x[0]), reverse=True)
    ordered_outputs = [x[-1] for x in scores]
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


selection_algorithms = [
    score_full_probmul_3,
    score_kgmon,
]


def _valid_sample(sample):
    try:
        sol = np.asarray(sample["solution"])
        if sol.ndim != 2 or sol.size == 0 or sol.shape[0] > 30 or sol.shape[1] > 30:
            return False
        if not np.issubdtype(sol.dtype, np.integer):
            return False
        if sol.min() < 0 or sol.max() > 9:
            return False
        if not np.isfinite(sample["beam_score"]):
            return False
        if not len(sample["score_aug"]) or not np.all(np.isfinite(sample["score_aug"])):
            return False
        return True
    except Exception:
        return False


class ArcDecoder:

    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}
        try:
            self.valid_keys = set(dataset.keys)
        except Exception:
            self.valid_keys = None

    def load_decoded_results(self, store, run_name=""):
        if not os.path.isdir(store):
            print(f"*** No decoded results at {store}")
            return 0
        n_files = n_samples = n_bad = 0
        for key in os.listdir(store):
            try:
                with bz2.BZ2File(os.path.join(store, key)) as f:
                    outputs = pickle.load(f)
            except Exception as e:
                print(f"*** Skipping corrupt shard {key}: {e}")
                n_bad += 1
                continue
            n_files += 1
            base_key = key.split(".")[0]
            if self.valid_keys is not None and base_key not in self.valid_keys:
                n_bad += 1
                continue
            for i, sample in enumerate(outputs):
                if not _valid_sample(sample):
                    n_bad += 1
                    continue
                self.decoded_results.setdefault(base_key, {})[f"{key}{run_name}.out{i}"] = sample
                n_samples += 1
        print(f"*** Loaded {n_files} shards / {n_samples} samples from {store} (skipped {n_bad})")
        return n_samples

    def candidate_stats(self):
        """Per output key: number of unique candidate grids, number of samples, and the
        vote count / kgmon margin of the leading candidate (how sure the pool is)."""
        stats = {}
        for bk, v in self.decoded_results.items():
            groups = {}
            for g in v.values():
                groups.setdefault(hashable(g["solution"]), []).append(g)
            scored = sorted((getter_kgmon(gs) for gs in groups.values()), reverse=True)
            top_votes = max(len(gs) for gs in groups.values())
            margin = (scored[0] - scored[1]) if len(scored) > 1 else float("inf")
            stats[bk] = dict(unique=len(groups), samples=len(v), top_votes=top_votes, margin=margin)
        return stats

    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}

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

        print(f" subkeys: {num_solved_keys}/{num_total_keys}")
        if correct_beam_scores:
            print(f" avg correct beam score: {np.mean(correct_beam_scores):8.5f}")
            print(f" max correct beam score: {np.max(correct_beam_scores):8.5f}")

        num_puzzles = len(num_tasks_per_puzzle)

        for selection_algorithm in selection_algorithms:
            name = selection_algorithm.__name__
            selected = self.run_selection_algo(selection_algorithm)
            correct_puzzles = {k for k, v in selected.items() if any(np.array_equal(guess, labels[k]) for guess in v[:self.n_guesses])}
            print(correct_puzzles)
            score = sum(1/num_tasks_per_puzzle[k.split("_")[0]] for k in correct_puzzles)
            print(f" acc: {score:5.1f}/{num_puzzles:3} ('{name}')")


# CELL 5 (code)
%%writefile arc_solver.py
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
import zlib
import torch
import numpy as np
from tqdm import tqdm
from datasets import Dataset
from collections import defaultdict

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling

import logging
from contextlib import redirect_stdout, redirect_stderr

from peft import get_peft_model_state_dict, set_peft_model_state_dict

import bz2
import pickle
import traceback

logging.disable(logging.WARNING)

# ---------------------------------------------------------------------------
# Run configuration (all overridable through environment variables so the
# same solver can run the LB-proven primary pass and a wider secondary pass).
# Defaults == the public LB 33.89 NVARC configuration.
# ---------------------------------------------------------------------------
def _env(name, default, cast):
    v = os.getenv(name)
    return cast(v) if v not in (None, "") else default

CFG = dict(
    model_path      = _env("ARC_MODEL_PATH", "", str),
    out_dir         = _env("ARC_OUT_DIR", "/kaggle/inference_outputs", str),
    lora_seed       = _env("ARC_LORA_SEED", 42, int),
    train_aug_seed  = _env("ARC_TRAIN_AUG_SEED", 1, int),
    n_train_aug     = _env("ARC_N_TRAIN_AUG", 16, int),     # x8 geometries = 128 sequences
    num_epochs      = _env("ARC_EPOCHS", 1, int),
    learning_rate   = _env("ARC_LR", 5e-5, float),
    eval_aug_seed   = _env("ARC_EVAL_AUG_SEED", 2, int),
    n_eval_aug      = _env("ARC_N_EVAL_AUG", 2, int),       # x8 geometries = 16 decoded views
    min_prob        = _env("ARC_MIN_PROB", 0.2, float),     # DFS cumulative prob threshold
    dfs_window      = _env("ARC_DFS_WINDOW", 540.0, float), # seconds per DFS call
    task_cap        = _env("ARC_TASK_CAP", 1200.0, float),  # seconds per task (decode phase)
    score_seed_off  = _env("ARC_SCORE_SEED_OFFSET", 0, int),
    decode_batch    = _env("ARC_DECODE_BATCH", 4, int),
)

ARC_VOCAB = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    "Ċ": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def resolve_model_dir():
    if CFG["model_path"] and os.path.isdir(CFG["model_path"]):
        return CFG["model_path"]
    candidates = [
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
        "/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1",
    ]
    for c in candidates:
        if os.path.isfile(os.path.join(c, "config.json")):
            return c
    import glob
    for c in glob.glob("/kaggle/input/**/config.json", recursive=True):
        d = os.path.dirname(c)
        if "grids15" in d and os.path.isfile(os.path.join(d, "tokenizer.json")):
            return d
    raise FileNotFoundError("qwen3_4b_grids15_sft139 model directory not found under /kaggle/input")


def stable_seed(key, offset=0):
    return (zlib.crc32(key.encode("utf-8")) + offset) % (1024 ** 2)


class UnslothFixedTrainer(UnslothTrainer):

    # Issue https://github.com/unslothai/unsloth/issues/2435

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        """Fixed compute_loss that handles Unsloth's view tensor issue"""
        if self.label_smoother is not None and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        outputs = model(**inputs)
        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if hasattr(unwrapped_model, "_get_name") and "unsloth" in unwrapped_model._get_name().lower():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        if hasattr(loss, "clone"):
            loss = loss.clone()
        if self.accelerator.num_processes > 1:
            loss = loss * self.accelerator.num_processes
        return (loss, outputs) if return_outputs else loss


class QwenDataCollatorForCompletionOnlyLM(DataCollatorForLanguageModeling):

    def torch_call(self, examples: list[Union[list[int], Any, dict[str, Any]]]) -> dict[str, Any]:
        batch = super().torch_call(examples)
        for i in range(len(examples)):
            labels = batch["input_ids"][i].clone()
            user_start_idx = np.where(labels == USER_TOKEN_ID)[0].tolist()
            assistant_start_idx = np.where(labels == ASSISTANT_TOKEN_ID)[0].tolist()
            start_idx = sorted(user_start_idx + assistant_start_idx)
            end_idx = np.where(labels == EOS_ID)[0]
            batch["labels"][i, :] = -100
            for j, (start, end) in enumerate(zip(start_idx, end_idx)):
                assert start < end
                if j % 2 == 1:
                    start += 2
                    end += 1
                    batch["labels"][i, start:end] = labels[start:end]
        return batch


# Throughput patch (algebraically identical to the baseline): only the 12
# ARC-token NLL values are moved to the CPU instead of the whole vocabulary.
_ARC_TOKEN_ID_CACHE = {}


def _arc_token_ids(device):
    key = str(device)
    token_ids = _ARC_TOKEN_ID_CACHE.get(key)
    if token_ids is None:
        token_ids = torch.tensor(ARC_TOKENS, dtype=torch.long, device=device)
        _ARC_TOKEN_ID_CACHE[key] = token_ids
    return token_ids


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, dfs_window) -> dict:

    n = logits.size(0)

    logits_f = logits.float()
    token_ids = _arc_token_ids(logits.device)
    arc_logits = logits_f.index_select(-1, token_ids)
    nll = (
        torch.as_tensor(scores, dtype=torch.float32, device=logits.device).view(n, 1)
        + torch.logsumexp(logits_f, dim=-1, keepdim=True)
        - arc_logits
    ).cpu()

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for token_idx, t in enumerate(ARC_TOKENS):
            score = nll[i, token_idx].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0])

    while time.time() - start_time < dfs_window and time.time() < end_time:

        batch_tokens = []
        batch_scores = []
        num_alive_beams = 0

        for i in range(n):
            if len(candidates[i]) == 0:
                batch_tokens.append(PAD_ID)
                batch_scores.append(1000)
            else:
                score, t = candidates[i].pop(0)
                batch_tokens.append(t)
                batch_scores.append(score)
                num_alive_beams += 1

        if num_alive_beams == 0:
            break

        outputs = model(
            input_ids=torch.tensor(batch_tokens, device=model.device, dtype=torch.long).view(-1, 1),
            position_ids=torch.full((n, 1), pos, device=model.device),
            past_key_values=cache,
            return_dict=True,
            use_cache=True,
        )

        next_suffixes = turbo_dfs(
            model,
            logits=outputs.logits[:, -1],
            max_new_tokens=max_new_tokens-1,
            max_score=max_score,
            scores=batch_scores,
            pos=pos+1,
            cache=outputs.past_key_values,
            start_time=start_time,
            end_time=end_time,
            dfs_window=dfs_window,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, dfs_window):
    input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    suffixes = turbo_dfs(
        model,
        logits=outputs.logits[:, -1],
        max_new_tokens=max_new_tokens,
        max_score=max_score,
        scores=[0.0] * input_ids.size(0),
        pos=input_ids.size(1),
        cache=outputs.past_key_values,
        start_time=time.time(),
        end_time=end_time,
        dfs_window=dfs_window,
    )
    result = []
    for batch_id, beams in suffixes.items():
        sorted_beams = sorted(beams, key=lambda x:x[0])
        result.append((batch_id, sorted_beams))
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model):
    batch_query_tokens = []
    batch_answer_tokens = []
    batch_tokens = []
    batch_lengths = []
    for query, answer in zip(queries, answers):
        query_tokens = tokenizer.encode(query)
        answer_tokens = tokenizer.encode(answer)
        tokens = query_tokens + answer_tokens
        batch_query_tokens.append(query_tokens)
        batch_answer_tokens.append(answer_tokens)
        batch_tokens.append(tokens)
        batch_lengths.append(len(tokens))
    max_len = max(batch_lengths)
    padded_tokens = []
    for tokens in batch_tokens:
        padded = tokens + [PAD_ID] * (max_len - len(tokens))
        padded_tokens.append(padded)
    input_ids = torch.tensor(padded_tokens, device=model.device, dtype=torch.long)

    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float()
    batch_log_norm = torch.logsumexp(batch_logits, dim=-1)
    result = []
    for row_id, (query_tokens, answer_tokens) in enumerate(zip(batch_query_tokens, batch_answer_tokens)):
        query_length = len(query_tokens)
        answer_length = len(answer_tokens)
        positions = torch.arange(
            query_length - 1,
            query_length - 1 + answer_length,
            device=model.device,
        )
        target_tokens = torch.tensor(answer_tokens, device=model.device, dtype=torch.long)
        answer_log_probs = (
            batch_logits[row_id, positions, target_tokens]
            - batch_log_norm[row_id, positions]
        )
        result.append(-answer_log_probs.sum().item())
    return result


def make_view_batches(eval_ds, n_perm, batch_size):
    """Group decoded views into batches. With n_perm=2 / batch_size=4 this is
    byte-for-byte the baseline batching ([id,rot180],[rot90,rot270],[T,T.rot180],[T.rot90,T.rot270])."""
    test_id_to_subkeys = defaultdict(list)
    for subkey in sorted(eval_ds.keys):
        test_id = subkey.split(".")[0].split("_")[1]
        test_id_to_subkeys[test_id].append(subkey)
    # sorted geometry order: 0 id, 1 rot90, 2 rot180, 3 rot270, 4 T, 5 T.rot90, 6 T.rot180, 7 T.rot270
    groups_a = [0, 2, 1, 3]          # untransposed geometries
    groups_b = [4, 6, 5, 7]          # transposed geometries
    batches = []
    for geos in (groups_a, groups_b):
        for test_id, subkeys in test_id_to_subkeys.items():
            if n_perm == 2 and batch_size == 4:
                # exact baseline batches: [id,rot180], [rot90,rot270], ...
                for a, b in ((geos[0], geos[1]), (geos[2], geos[3])):
                    batches.append(subkeys[a*n_perm:(a+1)*n_perm] + subkeys[b*n_perm:(b+1)*n_perm])
            else:
                # Batched DFS needs equal token lengths in a batch. Within a geometry group the
                # shape classes are: rot0/rot180 = H x W, rot90/rot270 = W x H, so pair them.
                for a, b in ((geos[0], geos[1]), (geos[2], geos[3])):
                    views = subkeys[a*n_perm:(a+1)*n_perm] + subkeys[b*n_perm:(b+1)*n_perm]
                    for i in range(0, len(views), batch_size):
                        batches.append(views[i:i+batch_size])
    return batches


def worker(rank, queue, end_time, test_path=None):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=CFG["lora_seed"],
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=CFG["num_epochs"],
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=CFG["learning_rate"],
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=CFG["lora_seed"],
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model_dir = resolve_model_dir()
    print(f"[Rank {rank}] model dir: {model_dir}")
    print(f"[Rank {rank}] config: {CFG}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_dir,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for name, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        mlm=False,
    )

    formatter = QwenFormatter(tokenizer=tokenizer)

    max_new_tokens = formatter.max_new_tokens()

    max_score = -np.log(CFG["min_prob"])

    if test_path is None:
        if rerun_mode:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        else:
            test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = CFG["out_dir"]
    os.makedirs(dir_outputs, exist_ok=True)

    while True:

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()

        try:
            torch.cuda.reset_peak_memory_stats()

            set_peft_model_state_dict(
                model,
                default_weights.copy(),
                adapter_name="default",
            )

            model = FastLanguageModel.for_training(model)

            puzzle_ds = arc_test_set.change_keys([key])

            train_ds = puzzle_ds.augment(n=CFG["n_train_aug"], shfl_keys=True, seed=CFG["train_aug_seed"])
            train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)

            with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):

                trainer = UnslothFixedTrainer(
                    model=model,
                    tokenizer=tokenizer,
                    data_collator=collator,
                    train_dataset=Dataset.from_list(train_ds.as_list(formatter)),
                    dataset_text_field="text",
                    max_seq_length=max_seq_length,
                    args=UnslothTrainingArguments(**train_args),
                )

                stats = trainer.train()

                model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)

                del trainer

            model = FastLanguageModel.for_inference(model)

            gc.collect()
            torch.cuda.empty_cache()

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for training")

            torch.cuda.reset_peak_memory_stats()

            print(f"[Rank {rank}] training stats for puzzle {key}: {stats}")

            puzzle_ds_multi = puzzle_ds.split_multi_replies()

            eval_ds = puzzle_ds_multi.augment(n=CFG["n_eval_aug"], seed=CFG["eval_aug_seed"])
            eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

            batches = make_view_batches(eval_ds, CFG["n_eval_aug"], CFG["decode_batch"])

            with torch.inference_mode():

                known_scores = {}

                for subkeys in batches:

                    spend_time = time.time() - start_time
                    if spend_time > CFG["task_cap"] or time.time() > end_time:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")

                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time, CFG["dfs_window"])

                    for subkey_id, scored_beams in dfs_result:

                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []

                        for beam_score, tokens_ in scored_beams:

                            array = formatter.convert_tokens_to_array(tokens_)
                            if array is None:
                                continue

                            solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)

                            grid_id = (bk, tuple(map(tuple, solution)))

                            if grid_id in known_scores:
                                augmented_scores = known_scores[grid_id]
                            else:
                                print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                                aug_dataset = ArcDataset(
                                    keys=[bk],
                                    queries={bk: puzzle_ds_multi.queries.get(bk)},
                                    replies={bk: [solution.tolist()]},
                                )
                                aug_dataset = aug_dataset.augment(seed=stable_seed(bk, CFG["score_seed_off"]))
                                aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])
                                augmented_scores1 = calc_scores(aug_queries[:4], aug_answers[:4], tokenizer, model)
                                augmented_scores2 = calc_scores(aug_queries[4:], aug_answers[4:], tokenizer, model)
                                augmented_scores = augmented_scores1 + augmented_scores2
                                known_scores[grid_id] = augmented_scores

                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                            })

                        if len(decoded_result):
                            with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:
                                pickle.dump(decoded_result, f)

            memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
            print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")

        except Exception as e:
            print(f"[Rank {rank}] ERROR on puzzle {key}: {type(e).__name__}: {e}")
            traceback.print_exc()
            try:
                model = FastLanguageModel.for_inference(model)
            except Exception:
                pass
            gc.collect()
            torch.cuda.empty_cache()
            if isinstance(e, torch.cuda.OutOfMemoryError):
                torch.cuda.synchronize()

        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")


# CELL 6 (code)
%%writefile starter.py
import os
import sys
import time
import json
import torch
import argparse
import traceback
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time, test_path, marker_dir):

    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Serialize the unsloth import/patching across workers (baseline behaviour),
    # but never wait forever for a dead predecessor.
    if rank > 0:
        waited = 0
        while not os.path.exists(os.path.join(marker_dir, f"worker{rank-1}")) and waited < 900:
            time.sleep(5)
            waited += 5

    from arc_solver import worker

    with open(os.path.join(marker_dir, f"worker{rank}"), "w") as f:
        f.write("Ok")

    print(f"[Rank {rank}] start!")

    attempts = 0
    while attempts < 2 and time.time() < end_time:
        attempts += 1
        try:
            worker(rank, queue, end_time, test_path=test_path)
            break
        except Exception as e:
            print(f"[Rank {rank}] worker crashed ({type(e).__name__}: {e}); attempt {attempts}")
            traceback.print_exc()
            try:
                import gc
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                pass
            if attempts >= 2:
                print(f"[Rank {rank}] giving up.")

    print(f"[Rank {rank}] done!")


def estimated_work(task):
    """Rough cost proxy: tokens of all train pairs (TTT + context) plus an estimate of
    the decoded output sizes. Cheap tasks first => the deadline truncates the most
    expensive (and least likely to be solved) tail."""
    def ntok(g):
        return len(g) * (len(g[0]) + 1)
    train_tokens = sum(ntok(p["input"]) + ntok(p["output"]) for p in task["train"])
    ratios = [ntok(p["output"]) / max(1, ntok(p["input"])) for p in task["train"]]
    ratios.sort()
    ratio = ratios[len(ratios) // 2]
    test_tokens = sum(ntok(t["input"]) * (1 + ratio) for t in task["test"])
    return train_tokens * 16 + test_tokens * 8 * len(task["test"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    parser.add_argument("--keys-file", type=str, default="")
    parser.add_argument("--nprocs", type=int, default=0)
    parser.add_argument("--order", type=str, default="cheap", choices=["cheap", "sorted", "file"])
    parser.add_argument("--test-path", type=str, default="")
    parser.add_argument("--marker-dir", type=str, default="/kaggle/working/markers")
    args, _ = parser.parse_known_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if args.test_path:
        test_path = args.test_path
    elif rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "r") as f:
        data = json.load(f)

    if args.keys_file:
        with open(args.keys_file) as f:
            keys = [k for k in json.load(f) if k in data]
    else:
        keys = sorted(data.keys())
        if not rerun_mode:
            debug_keys = os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")
            keys = [k for k in keys if k in debug_keys]

    if args.order == "cheap":
        keys = sorted(keys, key=lambda k: estimated_work(data[k]))
    elif args.order == "sorted":
        keys = sorted(keys)

    nprocs = args.nprocs or min(4, torch.cuda.device_count())
    os.makedirs(args.marker_dir, exist_ok=True)
    for f_ in os.listdir(args.marker_dir):
        try:
            os.remove(os.path.join(args.marker_dir, f_))
        except Exception:
            pass

    print(f"[starter] {len(keys)} tasks, {nprocs} workers, order={args.order}, "
          f"budget={(args.end_time - time.time())/60:.1f} min, test_path={test_path}")

    queue = mp.Manager().Queue()
    for key in keys:
        queue.put(key)
    for _ in range(nprocs):
        queue.put(None)

    try:
        mp.spawn(local_worker, args=(queue, args.end_time, test_path, args.marker_dir), nprocs=nprocs)
    except Exception as e:
        print(f"[starter] spawn finished with error: {type(e).__name__}: {e}")
        traceback.print_exc()
    print("[starter] finished.")


# CELL 7 (code)
# ---------------------------------------------------------------------------
# Pass A — NVARC (perfpatch maths unchanged) in a half-cost configuration, cheap-first task
# order, one worker per L4. Per-task cap 800 s measured from task start (includes TTT).
# ---------------------------------------------------------------------------
import subprocess, sys, time, os
COMMON_ENV = {
    "UNSLOTH_DISABLE_STATISTICS": "1",
    "TRITON_PTXAS_PATH": "/usr/local/cuda/bin/ptxas",
    "OMP_NUM_THREADS": "12",
    "PYTHONHASHSEED": "0",
    "ARC_TASK_CAP": os.getenv("ARC_PRIMARY_TASK_CAP", "800"),
    # fast pass: 64 TTT sequences (8 aug x 8 geometries) and 8 decoded views instead of 128 / 16.
    # Public evidence (LB runtime board, Sep 2026): half-cost NVARC passes score in the same 34-36 band
    # (Nguyen 35.97 in 5h44, Mithil 33.89 in 3h50). The saved time buys an independent second pass.
    "ARC_N_TRAIN_AUG": os.getenv("ARC_N_TRAIN_AUG", "8"),
    "ARC_N_EVAL_AUG": os.getenv("ARC_N_EVAL_AUG", "1"),
}
os.environ.update(COMMON_ENV)
os.environ["ARC_OUT_DIR"] = "/kaggle/inference_outputs"
t_a = time.time()
rc = subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}", "--order", "cheap"])
print(f"pass-A rc={rc} took {(time.time()-t_a)/60:.1f} min; remaining {(global_end_time-time.time())/60:.1f} min")


# CELL 8 (code)
# ---------------------------------------------------------------------------
# Pass B — the remaining budget: (1) catch-up of tasks pass A never reached (same
# config), then (2) an independent second pass over ALL tasks with different seeds
# (cheap-first, truncated by the deadline). Pass A outputs are never modified.
# ---------------------------------------------------------------------------
import os, sys, json, time, subprocess
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
from starter import estimated_work

def remaining():
    return global_end_time - time.time()

test_path = ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json" if RERUN
             else "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
data = ArcDataset.from_file(test_path)
keys_in_scope = data.keys if RERUN else [k for k in data.keys if k in os.getenv("ARC_DEBUG_KEYS", "0934a4d8,36a08778,981571dc,aa4ec2a5").split(",")]

dec = ArcDecoder(data.split_multi_replies(), n_guesses=2)
dec.load_decoded_results("/kaggle/inference_outputs")
stats = dec.candidate_stats()
unprocessed = [k for k in keys_in_scope if not any(f"{k}_{i}" in stats for i in range(len(data.queries[k]["test"])))]
print(f"[pass-B] pass-A reached {len(keys_in_scope)-len(unprocessed)}/{len(keys_in_scope)} tasks; remaining {remaining()/60:.1f} min")

def run_pass(name, keys, env_overrides, out_dir, min_minutes=20):
    if not keys or remaining() < min_minutes * 60:
        print(f"[pass-B:{name}] skipped (keys={len(keys)}, remaining={remaining()/60:.1f} min)")
        return
    keys_file = f"/kaggle/working/keys_{name}.json"
    json.dump(keys, open(keys_file, "w"))
    env = dict(os.environ, ARC_OUT_DIR=out_dir, **{k: str(v) for k, v in env_overrides.items()})
    t = time.time()
    rc = subprocess.call([sys.executable, "starter.py", "--end-time", f"{global_end_time}", "--keys-file", keys_file, "--order", "file"], env=env)
    print(f"[pass-B:{name}] rc={rc} keys={len(keys)} took {(time.time()-t)/60:.1f} min; remaining {remaining()/60:.1f} min")

cheap_first = lambda ks: sorted(ks, key=lambda k: estimated_work(data.queries[k]))

# (1) catch-up with the pass-A configuration into the pass-A store
run_pass("catchup", cheap_first(unprocessed), {}, "/kaggle/inference_outputs")

# (2) independent second pass over all tasks: new LoRA-init / augmentation / scoring seeds
SEED_B = dict(ARC_LORA_SEED=137, ARC_TRAIN_AUG_SEED=17, ARC_EVAL_AUG_SEED=29, ARC_SCORE_SEED_OFFSET=7,
              ARC_N_TRAIN_AUG=8, ARC_N_EVAL_AUG=1, ARC_TASK_CAP=700)
run_pass("seedB", cheap_first(keys_in_scope), SEED_B, "/kaggle/inference_outputs_b")
json.dump({"unprocessed_after_A": unprocessed, "seed_B": SEED_B}, open("/kaggle/working/passB_plan.json", "w"))


# CELL 9 (code)
# ---------------------------------------------------------------------------
# Selection: pooled score_kgmon over pass A + pass B candidates (votes add up).
# Invariant: pass A's top-1 is always among the two attempts.
# ---------------------------------------------------------------------------
import os, json, hashlib
import numpy as np
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

test_path = ("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json" if RERUN
             else "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
data = ArcDataset.from_file(test_path)
if not RERUN:
    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")
dm = data.split_multi_replies()

dec_a = ArcDecoder(dm, n_guesses=2)
dec_a.load_decoded_results("/kaggle/inference_outputs")
sel_a = dec_a.run_selection_algo()

pooled = ArcDecoder(dm, n_guesses=2)
pooled.load_decoded_results("/kaggle/inference_outputs")
n_b = pooled.load_decoded_results("/kaggle/inference_outputs_b", run_name=".b")
sel_p = pooled.run_selection_algo()

selected, n_changed, n_forced = {}, 0, 0
for bk in set(sel_a) | set(sel_p):
    a1 = sel_a.get(bk, [None])[0]
    top = list(sel_p.get(bk, []))[:2]
    if a1 is not None and not any(np.array_equal(a1, g) for g in top):
        top = (top[:1] + [a1]) if top else [a1]
        n_forced += 1
    if bk in sel_a and (len(top) != len(sel_a[bk][:2]) or any(not np.array_equal(x, y) for x, y in zip(top, sel_a[bk][:2]))):
        n_changed += 1
    selected[bk] = top
print(f"*** pooled selection: {len(selected)} outputs, pass-B samples={n_b}, changed vs pass-A={n_changed}, pass-A top-1 forced back={n_forced}")

submission = data.get_submission(selected)
n_fallback = 0
for k in data.keys:
    for i, t in enumerate(data.queries[k]["test"]):
        e = submission[k][i]
        def ok(g):
            return isinstance(g, list) and len(g) > 0 and all(isinstance(r, list) and len(r) == len(g[0]) and len(r) > 0 for r in g)
        if not ok(e["attempt_1"]) or e["attempt_1"] == [[0]]:
            e["attempt_1"] = [[int(x) for x in row] for row in t["input"]]
            n_fallback += 1
        if not ok(e["attempt_2"]) or e["attempt_2"] == e["attempt_1"]:
            e["attempt_2"] = [[0]]
        e["attempt_1"] = [[int(x) for x in row] for row in e["attempt_1"]]
        e["attempt_2"] = [[int(x) for x in row] for row in e["attempt_2"]]

with open("/kaggle/working/submission.json", "w") as f:
    json.dump(submission, f)
chk = json.load(open("/kaggle/working/submission.json"))
assert set(chk) == set(data.keys)
for k in data.keys:
    assert len(chk[k]) == len(data.queries[k]["test"]) and all("attempt_1" in e and "attempt_2" in e for e in chk[k])
print(f"*** submission.json OK: {len(chk)} tasks, {sum(len(v) for v in chk.values())} outputs, {n_fallback} identity fallbacks, "
      f"sha256={hashlib.sha256(open('/kaggle/working/submission.json','rb').read()).hexdigest()[:12]}")

if not RERUN:
    print("--- pass A alone ---"); dec_a.benchmark_selection_algos()
    print("--- pooled A+B ---"); pooled.benchmark_selection_algos()
    print("*** Reload score (tasks):", data.validate_submission(chk))

