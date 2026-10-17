# CELL 0 (code)
import time
global_end_time = time.time() + 12 * 3600 - 600

# CELL 1 (code)
!pip uninstall -y tensorflow

# CELL 2 (code)
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

# CELL 3 (code)
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


class ArcDecoder:
    
    def __init__(self, dataset, n_guesses):
        self.dataset = dataset
        self.n_guesses = n_guesses
        self.decoded_results = {}

    def load_decoded_results(self, store, run_name=""):
        for key in os.listdir(store):
            with bz2.BZ2File(os.path.join(store, key)) as f:
                outputs = pickle.load(f)
            base_key = key.split(".")[0]
            self.decoded_results[base_key] = self.decoded_results.get(base_key, {})
            for i, sample in enumerate(outputs):
                self.decoded_results[base_key][f"{key}{run_name}.out{i}"] = sample

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

# CELL 4 (code)
%%writefile arc_solver.py
from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import time
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

logging.disable(logging.WARNING)

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
        # 🔧 KEY FIX: Clone the loss tensor before in-place operations
        if hasattr(loss, "clone"):
            loss = loss.clone()  # Converts view tensor to independent tensor
        # Now safe for DDP gradient scaling
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


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time) -> dict:

    n = logits.size(0)

    nll = torch.tensor(scores, dtype=torch.float32).view(n, 1) - logits.float().cpu().log_softmax(-1)

    suffixes = defaultdict(list)

    candidates = dict()

    for i in range(n):
        candidates[i] = []
        for t in ARC_TOKENS:
            score = nll[i, t].item()
            if score < max_score:
                if t == EOS_ID:
                    suffixes[i].append((score, [t]))
                elif max_new_tokens > 1:
                    candidates[i].append((score, t))

    for i in range(n):
        candidates[i] = sorted(candidates[i], key=lambda x:x[0]) #[:5]
    
    while time.time() - start_time < 540 and time.time() < end_time:

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
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time):
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
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=True)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def worker(rank, queue, end_time):

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    peft_params = dict(
        r=256,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "embed_tokens", "lm_head"],
        lora_alpha=32,
        lora_dropout=0.0,
        bias="none",
        use_gradient_checkpointing=False,
        random_state=42,
        use_rslora=True,
        loftq_config=None,
    )

    train_args = dict(
        per_device_eval_batch_size=1,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        num_train_epochs=1,
        warmup_steps=0,
        warmup_ratio=0.1,
        max_grad_norm=1.0,
        learning_rate=5e-5,
        optim="adamw_torch",
        weight_decay=0.0,
        lr_scheduler_type="cosine",
        seed=42,
        report_to="none",
        save_strategy="no",
        eval_strategy="no",
        logging_strategy="no",
        fp16=False,
        bf16=True,
        # Disable FSDP (use standard DDP)
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    max_seq_length = 8192

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1",
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

    max_score = -np.log(0.2)

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = "/kaggle/inference_outputs"
    os.makedirs(dir_outputs, exist_ok=True)

    while not queue.empty():

        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break
        
        start_time = time.time()
        
        torch.cuda.reset_peak_memory_stats()

        load_result = set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)

        puzzle_ds = arc_test_set.change_keys([key])

        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)
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

        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
        eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

        test_id_to_subkeys = defaultdict(list)
        for subkey in sorted(eval_ds.keys):
            test_id = subkey.split(".")[0].split("_")[1]
            test_id_to_subkeys[test_id].append(subkey)

        batches = []
        for test_id, subkeys in test_id_to_subkeys.items():
            # 0: permute x 2
            # 4: rot90.rot90.permute x 2
            batch = []
            for offset in [0, 4]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 2: permute.rot90 x 2
            # 6: rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [2, 6]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
        for test_id, subkeys in test_id_to_subkeys.items():
            # 8: transpose.permute x 2
            # 12: transpose.rot90.rot90.permute x 2
            batch = []
            for offset in [8, 12]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)
            # 10: transpose.rot90.permute x 2
            # 14: transpose.rot90.rot90.rot90.permute x 2
            batch = []
            for offset in [10, 14]:
                batch.extend(subkeys[offset:offset+2])
            batches.append(batch)

        with torch.inference_mode():
                
            known_scores = {}

            for subkeys in batches:

                spend_time = time.time() - start_time
                if spend_time > 1200 or time.time() > end_time:
                    print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                    break

                print(f"[Rank {rank}] decoding {subkeys}")

                tokens = []
                for subkey in subkeys:
                    data = eval_ds.get(subkey, formatter)
                    tokens.append(tokenizer.encode(data["input"]))

                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)

                for subkey_id, scored_beams in dfs_result:

                    subkey = subkeys[subkey_id]
                    bk = subkey.split(".")[0]
                    decoded_result = []

                    for beam_score, tokens in scored_beams:

                        array = formatter.convert_tokens_to_array(tokens)
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
                            aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
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
        
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")

# CELL 5 (code)
%%writefile starter.py
import os
import time
import json
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
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

    legc_skip = set()
    legc_path = "/kaggle/working/induction_results.json"
    if os.path.exists(legc_path):
        with open(legc_path) as f:
            legc_skip = {k for k, r in json.load(f).items()
                         if isinstance(r, dict) and r.get("verified")}
        print(f"[LegC] skipping {len(legc_skip)} verified task(s) in the base queue")

    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        if key in legc_skip:
            continue
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)

# CELL 6 (code)
%%writefile arc_induction_v2.py
#!/usr/bin/env python3
# =============================================================================
#  ARC-AGI-2  Leg C : Verified-Induction Pre-Pass  (v2, base-notebook-aligned)
#  -------------------------------------------------------------
#  Role      : runs BEFORE the base 4B TTT pipeline (failed-in-aimo line).
#              Samples DSL programs with a 7B coder (one per L4, 4 workers),
#              verifies them against ALL train pairs in an isolated,
#              torch-free sandbox, and emits train-verified test outputs.
#              A separate merge cell then lets verified outputs pre-empt
#              attempt_1 while attempt_2 keeps the base (transduction)
#              candidate -> monotonic: final score can only add coverage.
#  Env facts : Kaggle AGI-2 = Python 3.11, L4 22GB x4, offline.
#              -> NO vLLM / cp312 wheelhouse dependency (27B v4 incident).
#              -> 7B bf16 fits a single L4; 4 independent workers.
#              -> heavy imports live INSIDE worker fns (spawn re-import cheap).
#  Fixes vs the "hacker plan":
#     1. exec single-namespace (two-dict exec made the DSL unreachable)
#     2. `import re` present; no undefined names in the submit loop
#     3. DSL is ENFORCED via AST whitelist, not politely requested
#     4. 35-primitive library instead of 3; per-program SIGALRM timeout
#     5. batched sampling (K per call) + 1 repair round, not 3 blind shots
#     6. official 2026 submission format {tid: [{attempt_1, attempt_2}, ..]}
#  Usage (Kaggle):
#     !python arc_induction_v1.py --budget-h 1.5
#     !python arc_induction_v1.py --selftest        # sandbox unit tests, no GPU
#     !python arc_induction_v1.py --smoke 4         # 4 easiest tasks, verbose
# =============================================================================

import os
import sys
import re
import json
import time
import math
import random
import hashlib
import argparse
import subprocess
import multiprocessing as mp

# ----------------------------------------------------------------------------
# Paths / defaults
# ----------------------------------------------------------------------------
DATA_DIR_CANDIDATES = [
    "/kaggle/input/competitions/arc-prize-2026-arc-agi-2",  # path used by the LB-proven base notebook
    "/kaggle/input/arc-prize-2026-arc-agi-2",               # standard mount fallback
]
WORK_DIR      = os.environ.get("ARC_WORK_DIR", "/kaggle/working")
RUNTIME_PATH  = os.path.join(WORK_DIR, "_dsl_runtime.py")
RESULTS_PATH  = os.path.join(WORK_DIR, "induction_results.json")
AUDIT_PATH    = os.path.join(WORK_DIR, "induction_audit.json")
SUB_PATH      = os.path.join(WORK_DIR, "submission_induction_only.json")
PARTIAL_TPL   = os.path.join(WORK_DIR, "_legc_partial_{rank}.jsonl")

MODEL_PATH_CANDIDATES = [
    "/kaggle/input/models/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1",  # user's attach path
    "/kaggle/input/qwen-lm/qwen2.5-coder/transformers/7b-instruct/1",
    "/kaggle/input/qwen2.5-coder-7b-instruct/transformers/default/1",
    "/kaggle/input/qwen2.5-coder/transformers/7b-instruct/1",
    "/kaggle/input/qwen2.5-coder-7b-instruct",
]

DEFAULTS = dict(
    budget_h        = 1.5,    # wall budget of the whole induction phase
    per_task_cap    = 200.0,  # seconds per task (both rounds included)
    k1              = 8,      # round-1 samples per task
    k2              = 6,      # repair-round samples
    temp1           = 0.8,
    temp2           = 0.7,
    max_new_tokens  = 640,
    max_prompt_chars= 26000,
    prog_timeout    = 4.0,    # per-candidate CPU seconds in the sandbox
    seed            = 1234,
    print_grids     = 8,      # dump grids for first N verified tasks
)

# ----------------------------------------------------------------------------
# The sandbox runtime (torch-free).  Written to _dsl_runtime.py and executed
# as a subprocess:  python _dsl_runtime.py in.json out.json
# Candidate code is exec'd into ONE namespace that already CONTAINS the DSL
# functions as values -> transform.__globals__ sees them.  (This is the
# structural fix for the two-dict exec bug.)
# ----------------------------------------------------------------------------
RUNTIME_CODE = r'''
import sys, json, ast, signal, builtins
import numpy as np
from scipy.ndimage import label as _cc_label

MAXD = 30
_ST4 = np.array([[0,1,0],[1,1,1],[0,1,0]])
_ST8 = np.ones((3,3), dtype=int)

def _np(g): return np.asarray(g, dtype=int)
def _ls(a): return [[int(x) for x in row] for row in np.asarray(a)]

# ---------------- grid basics ----------------
def grid_shape(g):
    a = _np(g); return (int(a.shape[0]), int(a.shape[1]))
def new_grid(h, w, fill=0):
    return [[int(fill)] * int(w) for _ in range(int(h))]
def rot90(g):  return _ls(np.rot90(_np(g), 3))     # clockwise
def rot180(g): return _ls(np.rot90(_np(g), 2))
def rot270(g): return _ls(np.rot90(_np(g), 1))     # counter-clockwise
def flip_h(g): return _ls(np.fliplr(_np(g)))       # mirror left-right
def flip_v(g): return _ls(np.flipud(_np(g)))       # mirror up-down
def transpose(g): return _ls(_np(g).T)
def crop(g, r, c, h, w):
    return _ls(_np(g)[int(r):int(r)+int(h), int(c):int(c)+int(w)])

# ---------------- colors ----------------
def bg_color(g):
    v, n = np.unique(_np(g), return_counts=True); return int(v[np.argmax(n)])
def palette(g): return [int(x) for x in np.unique(_np(g))]
def color_counts(g):
    v, n = np.unique(_np(g), return_counts=True)
    return {int(a): int(b) for a, b in zip(v, n)}
def replace_color(g, a, b):
    arr = _np(g).copy(); arr[arr == int(a)] = int(b); return _ls(arr)
def swap_colors(g, a, b):
    arr = _np(g).copy(); ma = arr == int(a); mb = arr == int(b)
    arr[ma] = int(b); arr[mb] = int(a); return _ls(arr)
def map_colors(g, mapping):
    arr = _np(g); out = arr.copy()
    for k, v in mapping.items(): out[arr == int(k)] = int(v)
    return _ls(out)
def find_color_cells(g, color):
    ys, xs = np.where(_np(g) == int(color))
    return [(int(r), int(c)) for r, c in zip(ys, xs)]
def crop_to_content(g, bg=None):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    ys, xs = np.where(a != b)
    if len(ys) == 0: return _ls(a)
    return _ls(a[ys.min():ys.max()+1, xs.min():xs.max()+1])

# ---------------- assembly ----------------
def paste(dst, src, r, c, transparent=None):
    D = _np(dst).copy(); S = _np(src)
    r0, c0 = int(r), int(c); h, w = S.shape
    r1, c1 = max(r0, 0), max(c0, 0)
    r2, c2 = min(r0 + h, D.shape[0]), min(c0 + w, D.shape[1])
    if r2 <= r1 or c2 <= c1: return _ls(D)
    sub = S[r1-r0:r2-r0, c1-c0:c2-c0]
    if transparent is None:
        D[r1:r2, c1:c2] = sub
    else:
        m = sub != int(transparent)
        D[r1:r2, c1:c2][m] = sub[m]
    return _ls(D)
def concat_h(a, b): return _ls(np.concatenate([_np(a), _np(b)], axis=1))
def concat_v(a, b): return _ls(np.concatenate([_np(a), _np(b)], axis=0))
def tile(g, ny, nx): return _ls(np.tile(_np(g), (int(ny), int(nx))))
def upscale(g, fy, fx):
    return _ls(np.kron(_np(g), np.ones((int(fy), int(fx)), dtype=int)))
def downscale(g, fy, fx):
    a = _np(g); fy, fx = int(fy), int(fx)
    h, w = a.shape[0] // fy, a.shape[1] // fx
    out = np.zeros((h, w), dtype=int)
    for i in range(h):
        for j in range(w):
            blk = a[i*fy:(i+1)*fy, j*fx:(j+1)*fx].ravel()
            v, n = np.unique(blk, return_counts=True)
            out[i, j] = v[np.argmax(n)]
    return _ls(out)
def cellwise(a, b, f):
    A, B = _np(a), _np(b)
    if A.shape != B.shape: raise ValueError("cellwise: shape mismatch")
    return [[int(f(int(A[i, j]), int(B[i, j]))) for j in range(A.shape[1])]
            for i in range(A.shape[0])]
def grid_eq(a, b):
    A, B = _np(a), _np(b)
    return A.shape == B.shape and bool((A == B).all())
def split_rows(g, n):
    a = _np(g); h = a.shape[0] // int(n)
    return [_ls(a[i*h:(i+1)*h, :]) for i in range(int(n))]
def split_cols(g, n):
    a = _np(g); w = a.shape[1] // int(n)
    return [_ls(a[:, i*w:(i+1)*w]) for i in range(int(n))]
def split_by_separator(g):
    # splits on full uniform rows/cols of the dominant separator color;
    # returns row-major list of sub-grids (or [g] when no separator found)
    a = _np(g); H, W = a.shape
    frows = [i for i in range(H) if len(set(a[i, :].tolist())) == 1]
    fcols = [j for j in range(W) if len(set(a[:, j].tolist())) == 1]
    colors = [int(a[i, 0]) for i in frows] + [int(a[0, j]) for j in fcols]
    if not colors: return [_ls(a)]
    v, n = np.unique(np.array(colors), return_counts=True)
    sep = int(v[np.argmax(n)])
    rs = set(i for i in frows if a[i, 0] == sep)
    cs = set(j for j in fcols if a[0, j] == sep)
    def runs(banned, N):
        out, s = [], None
        for i in range(N):
            if i in banned:
                if s is not None: out.append((s, i)); s = None
            else:
                if s is None: s = i
        if s is not None: out.append((s, N))
        return out
    parts = []
    for (r0, r1) in runs(rs, H):
        for (c0, c1) in runs(cs, W):
            parts.append(_ls(a[r0:r1, c0:c1]))
    return parts if len(parts) > 1 else [_ls(a)]

# ---------------- objects ----------------
def _mk_obj(cells):
    rs  = [r for r, _, _ in cells]; csx = [c for _, c, _ in cells]
    col = [v for _, _, v in cells]
    v, n = np.unique(np.array(col), return_counts=True)
    return {'cells': cells,
            'r': min(rs), 'c': min(csx),
            'h': max(rs) - min(rs) + 1, 'w': max(csx) - min(csx) + 1,
            'size': len(cells),
            'color': int(v[np.argmax(n)]),
            'colors': [int(x) for x in v]}
def get_objects(g, bg=None, diagonal=False, multicolor=True):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    st = _ST8 if diagonal else _ST4
    objs = []
    if multicolor:
        lab, n = _cc_label(a != b, structure=st)
        for i in range(1, n + 1):
            ys, xs = np.where(lab == i)
            objs.append(_mk_obj([(int(r), int(c), int(a[r, c]))
                                 for r, c in zip(ys, xs)]))
    else:
        for col in palette(g):
            if col == b: continue
            lab, n = _cc_label(a == col, structure=st)
            for i in range(1, n + 1):
                ys, xs = np.where(lab == i)
                objs.append(_mk_obj([(int(r), int(c), col)
                                     for r, c in zip(ys, xs)]))
    return objs
def obj_size(o):  return o['size']
def obj_color(o): return o['color']
def obj_grid(o, bg=0):
    g = np.full((o['h'], o['w']), int(bg), dtype=int)
    for r, c, v in o['cells']: g[r - o['r'], c - o['c']] = v
    return _ls(g)
def subgrid_at(g, o):
    return crop(g, o['r'], o['c'], o['h'], o['w'])
def largest_object(objs):
    return max(objs, key=lambda o: (o['size'], -o['r'], -o['c']))
def smallest_object(objs):
    return min(objs, key=lambda o: (o['size'], o['r'], o['c']))
def paint_objects(g, objs, color):
    a = _np(g).copy()
    for o in objs:
        for r, c, _ in o['cells']: a[r, c] = int(color)
    return _ls(a)
def erase_objects(g, objs, bg=0):
    a = _np(g).copy()
    for o in objs:
        for r, c, _ in o['cells']: a[r, c] = int(bg)
    return _ls(a)
def move_object(g, o, dr, dc, bg=0):
    a = _np(erase_objects(g, [o], bg)); H, W = a.shape
    for r, c, v in o['cells']:
        nr, nc = r + int(dr), c + int(dc)
        if 0 <= nr < H and 0 <= nc < W: a[nr, nc] = v
    return _ls(a)
def fold_repaint(g, objs, color_fn):
    a = _np(g).copy()
    for o in objs:
        col = color_fn(o)
        if col is None: continue
        for r, c, _ in o['cells']: a[r, c] = int(col)
    return _ls(a)
def count_holes(o):
    m = np.zeros((o['h'], o['w']), dtype=int)
    for r, c, _ in o['cells']: m[r - o['r'], c - o['c']] = 1
    lab, n = _cc_label(m == 0, structure=_ST4)
    holes = 0
    for i in range(1, n + 1):
        ys, xs = np.where(lab == i)
        if ys.min() > 0 and xs.min() > 0 and \
           ys.max() < o['h'] - 1 and xs.max() < o['w'] - 1:
            holes += 1
    return holes

# ---------------- physics-ish ----------------
def flood_fill(g, r, c, color):
    a = _np(g).copy(); H, W = a.shape
    r0, c0 = int(r), int(c); tgt = int(a[r0, c0]); col = int(color)
    if tgt == col: return _ls(a)
    stack = [(r0, c0)]
    while stack:
        y, x = stack.pop()
        if 0 <= y < H and 0 <= x < W and a[y, x] == tgt:
            a[y, x] = col
            stack.extend([(y+1, x), (y-1, x), (y, x+1), (y, x-1)])
    return _ls(a)
def gravity(g, direction, bg=None):
    a = _np(g); b = bg_color(g) if bg is None else int(bg)
    H, W = a.shape
    out = np.full_like(a, b)
    if direction in ('down', 'up'):
        for j in range(W):
            col = [v for v in a[:, j].tolist() if v != b]
            if direction == 'down': out[H - len(col):, j] = col
            else:                   out[:len(col), j] = col
    elif direction in ('left', 'right'):
        for i in range(H):
            row = [v for v in a[i, :].tolist() if v != b]
            if direction == 'left': out[i, :len(row)] = row
            else:                   out[i, W - len(row):] = row
    else:
        raise ValueError("gravity: direction must be up/down/left/right")
    return _ls(out)
def is_symmetric_h(g): return grid_eq(g, flip_h(g))
def is_symmetric_v(g): return grid_eq(g, flip_v(g))

EXPORTS = {k: v for k, v in list(globals().items())
           if not k.startswith('_') and callable(v)
           and k not in ('main',)}

# ---------------- enforcement ----------------
_ALLOWED_BUILTINS = ['abs','all','any','bool','dict','divmod','enumerate',
                     'filter','float','frozenset','int','isinstance','len',
                     'list','map','max','min','pow','print','range','repr',
                     'reversed','round','set','sorted','str','sum','tuple','zip']
SAFE_BUILTINS = {k: getattr(builtins, k) for k in _ALLOWED_BUILTINS}

_BANNED_NODES = (ast.While, ast.Import, ast.ImportFrom, ast.Global,
                 ast.Nonlocal, ast.ClassDef, ast.AsyncFunctionDef,
                 ast.AsyncFor, ast.AsyncWith, ast.Await)
_BANNED_NAMES = {'exec','eval','open','__import__','compile','globals',
                 'locals','vars','getattr','setattr','delattr','input',
                 'breakpoint','exit','quit','help','super','type','object',
                 'memoryview','bytearray','bytes'}

def validate_source(src):
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return 'syntax error: %s' % e
    has_transform = False
    for node in ast.walk(tree):
        if isinstance(node, _BANNED_NODES):
            return 'banned construct: %s' % type(node).__name__
        if isinstance(node, ast.Name) and node.id in _BANNED_NAMES:
            return 'banned name: %s' % node.id
        if isinstance(node, ast.Attribute) and node.attr.startswith('__'):
            return 'banned dunder attribute: %s' % node.attr
        if isinstance(node, ast.FunctionDef) and node.name == 'transform':
            has_transform = True
    if not has_transform:
        return 'missing def transform(grid)'
    return None

class _Timeout(Exception): pass
def _on_alarm(signum, frame): raise _Timeout()
signal.signal(signal.SIGALRM, _on_alarm)

def _norm_out(g):
    a = np.asarray(g)
    if a.dtype == object or a.ndim != 2 or a.size == 0:
        raise ValueError('output must be a non-empty rectangular 2D grid')
    if a.shape[0] > MAXD or a.shape[1] > MAXD:
        raise ValueError('output larger than %dx%d' % (MAXD, MAXD))
    a = a.astype(int)
    if a.min() < 0 or a.max() > 9:
        raise ValueError('colors must be integers 0..9')
    return a.tolist()

def _copy_grid(g): return [row[:] for row in g]

def run_one(src, train, test, tmo):
    err = validate_source(src)
    if err:
        return {'status': 'rejected', 'error': err, 'pairs_passed': 0}
    env = dict(EXPORTS)
    env['np'] = np                      # numpy allowed read-only by convention
    env['__builtins__'] = SAFE_BUILTINS
    try:
        signal.setitimer(signal.ITIMER_REAL, float(tmo))
        exec(compile(src, '<candidate>', 'exec'), env)   # SINGLE namespace
        tf = env.get('transform')
        if not callable(tf):
            return {'status': 'rejected', 'error': 'transform not callable',
                    'pairs_passed': 0}
        pairs_passed, fail = 0, None
        for i, p in enumerate(train):
            try:
                got = _norm_out(tf(_copy_grid(p['input'])))
            except _Timeout:
                raise
            except Exception as e:
                if fail is None:
                    fail = {'idx': i, 'err': repr(e)[:200]}
                continue
            if got == p['output']:
                pairs_passed += 1
            elif fail is None:
                fail = {'idx': i, 'got': got}
        if pairs_passed == len(train):
            touts = [_norm_out(tf(_copy_grid(t))) for t in test]
            return {'status': 'verified', 'pairs_passed': pairs_passed,
                    'n_pairs': len(train), 'test_outputs': touts}
        out = {'status': 'partial', 'pairs_passed': pairs_passed,
               'n_pairs': len(train)}
        if fail: out.update(fail)
        return out
    except _Timeout:
        return {'status': 'timeout', 'pairs_passed': 0,
                'error': 'exceeded %.1fs' % tmo}
    except Exception as e:
        return {'status': 'error', 'pairs_passed': 0,
                'error': repr(e)[:300]}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)

def main():
    sys.setrecursionlimit(300)
    inp, outp = sys.argv[1], sys.argv[2]
    with open(inp) as f:
        req = json.load(f)
    results = [run_one(src, req['train'], req['test'], req.get('timeout', 4.0))
               for src in req['programs']]
    with open(outp, 'w') as f:
        json.dump({'results': results}, f)

if __name__ == '__main__':
    main()
'''

# ----------------------------------------------------------------------------
# Prompting
# ----------------------------------------------------------------------------
SYSTEM_MSG = (
    "You are an expert ARC puzzle solver. You write short Python solutions "
    "using ONLY the provided DSL helpers plus basic pure Python "
    "(for-loops, comprehensions, if/else, lambda). Follow the rules exactly."
)

CHEAT_SHEET = """AVAILABLE DSL (grids are list[list[int]], colors 0-9):
grid_shape(g)->(h,w) | new_grid(h,w,fill=0) | crop(g,r,c,h,w)
rot90(g) rot180(g) rot270(g) | flip_h(g) flip_v(g) | transpose(g)
bg_color(g) | palette(g) | color_counts(g)->dict | find_color_cells(g,color)
replace_color(g,a,b) | swap_colors(g,a,b) | map_colors(g,{a:b,...})
crop_to_content(g,bg=None)
paste(dst,src,r,c,transparent=None) | concat_h(a,b) concat_v(a,b)
tile(g,ny,nx) | upscale(g,fy,fx) | downscale(g,fy,fx)
cellwise(a,b,f) | grid_eq(a,b) | split_rows(g,n) split_cols(g,n)
split_by_separator(g)->list of sub-grids (row-major)
get_objects(g,bg=None,diagonal=False,multicolor=True)->list of obj
  obj = {'cells':[(r,c,color)..],'r','c','h','w','size','color','colors'}
obj_size(o) obj_color(o) | obj_grid(o,bg=0) | subgrid_at(g,o)
largest_object(objs) smallest_object(objs)
paint_objects(g,objs,color) | erase_objects(g,objs,bg=0)
move_object(g,o,dr,dc,bg=0) | fold_repaint(g,objs,color_fn)
count_holes(o) | flood_fill(g,r,c,color) | gravity(g,'down'|'up'|'left'|'right')
is_symmetric_h(g) is_symmetric_v(g)

RULES:
- Define exactly one function: def transform(grid) -> grid.
- NO imports. NO while loops. NO classes. Use only helpers above + pure python.
- Return a new grid (list of lists of ints 0-9), never mutate weird state.
"""

FEWSHOT = """Two tiny examples of the expected style:

Example task: recolor every object to the color of the largest object.
```python
def transform(grid):
    objs = get_objects(grid)
    big = largest_object(objs)
    return fold_repaint(grid, objs, lambda o: obj_color(big))
```

Example task: output = input rotated 90 degrees clockwise, colors 1<->3 swapped.
```python
def transform(grid):
    return swap_colors(rot90(grid), 1, 3)
```
"""

def render_grid(g):
    return "\n".join("".join(str(int(v)) for v in row) for row in g)

def shape_hint(task):
    ins  = [(len(p['input']),  len(p['input'][0]))  for p in task['train']]
    outs = [(len(p['output']), len(p['output'][0])) for p in task['train']]
    if all(a == b for a, b in zip(ins, outs)):
        return "Observed shape rule: every output has the SAME shape as its input."
    if len(set(outs)) == 1:
        h, w = outs[0]
        return f"Observed shape rule: every output is exactly {h}x{w}."
    ratios = set()
    ok = True
    for (ih, iw), (oh, ow) in zip(ins, outs):
        if oh % ih == 0 and ow % iw == 0:
            ratios.add((oh // ih, ow // iw))
        else:
            ok = False
    if ok and len(ratios) == 1:
        fy, fx = next(iter(ratios))
        return f"Observed shape rule: output = input scaled by ({fy}x, {fx}x)."
    return "Observed shape rule: varies -- infer it from the pairs. " + \
           f"in->out shapes: {list(zip(ins, outs))}"

def build_prompt(task, repair=None):
    lines = [CHEAT_SHEET, FEWSHOT, "--- TASK ---"]
    for i, p in enumerate(task['train'], 1):
        ih, iw = len(p['input']), len(p['input'][0])
        oh, ow = len(p['output']), len(p['output'][0])
        lines.append(f"Train {i} Input ({ih}x{iw}):\n{render_grid(p['input'])}")
        lines.append(f"Train {i} Output ({oh}x{ow}):\n{render_grid(p['output'])}")
    for j, t in enumerate(task['test'], 1):
        th, tw = len(t['input']), len(t['input'][0])
        lines.append(f"Test {j} Input ({th}x{tw}):\n{render_grid(t['input'])}")
    lines.append(shape_hint(task))
    if repair:
        lines.append(
            "### PREVIOUS ATTEMPT (best so far) ###\n"
            f"```python\n{repair['code']}\n```\n"
            f"It passed {repair['pairs_passed']}/{repair['n_pairs']} train pairs "
            f"but FAILED on Train {repair['fail_idx'] + 1}.\n"
            f"{repair['diff']}\n"
            "Fix the LOGIC. Keep the DSL rules. Output the corrected full solution."
        )
    lines.append("Think briefly, then answer with ONE ```python code block "
                 "containing def transform(grid).")
    return "\n\n".join(lines)

def repair_diff(task, part):
    i = part.get('idx', part.get('fail_idx', 0))
    exp = task['train'][i]['output']
    if 'got' in part and part['got'] is not None:
        got = part['got']
        gh, gw = len(got), len(got[0])
        eh, ew = len(exp), len(exp[0])
        if (gh, gw) != (eh, ew):
            return f"Your output shape was {gh}x{gw}, expected {eh}x{ew}."
        if gh <= 15 and gw <= 15:
            return ("Expected:\n" + render_grid(exp) +
                    "\nYour output:\n" + render_grid(got))
        wrong = sum(1 for r in range(eh) for c in range(ew)
                    if got[r][c] != exp[r][c])
        return f"Shape correct ({eh}x{ew}) but {wrong} cells are wrong."
    if 'err' in part:
        return f"It raised: {part['err']}"
    return "Output did not match."

def extract_code(text):
    m = re.findall(r"```python(.*?)```", text, re.DOTALL)
    if m: return m[-1].strip()
    m = re.findall(r"```(.*?)```", text, re.DOTALL)
    if m: return m[-1].strip()
    idx = text.find("def transform")
    return text[idx:].strip() if idx >= 0 else text.strip()

def code_key(src):
    return hashlib.md5(re.sub(r"\s+", "", src).encode()).hexdigest()

# ----------------------------------------------------------------------------
# Sandbox client (used by workers AND selftest; torch-free)
# ----------------------------------------------------------------------------
def run_verifier(programs, train, test, prog_timeout, tag):
    inp  = os.path.join(WORK_DIR, f"_vin_{tag}.json")
    outp = os.path.join(WORK_DIR, f"_vout_{tag}.json")
    with open(inp, 'w') as f:
        json.dump({'programs': programs, 'train': train,
                   'test': test, 'timeout': prog_timeout}, f)
    total = 10 + prog_timeout * len(programs) + 5
    try:
        subprocess.run([sys.executable, RUNTIME_PATH, inp, outp],
                       timeout=total,
                       stdout=subprocess.DEVNULL,
                       stderr=subprocess.DEVNULL)
    except subprocess.TimeoutExpired:
        return [{'status': 'timeout', 'pairs_passed': 0,
                 'error': 'verifier batch timeout'}] * len(programs)
    finally:
        try: os.remove(inp)
        except OSError: pass
    if not os.path.exists(outp):
        return [{'status': 'error', 'pairs_passed': 0,
                 'error': 'verifier crashed'}] * len(programs)
    with open(outp) as f:
        res = json.load(f)['results']
    try: os.remove(outp)
    except OSError: pass
    return res

# ----------------------------------------------------------------------------
# Per-task solve (runs inside a GPU worker)
# ----------------------------------------------------------------------------
def gen_batch(model, tok, prompt, k, temp, max_new):
    import torch
    msgs = [{'role': 'system', 'content': SYSTEM_MSG},
            {'role': 'user',   'content': prompt}]
    text = tok.apply_chat_template(msgs, tokenize=False,
                                   add_generation_prompt=True)
    enc = tok(text, return_tensors='pt').to(model.device)
    try:
        with torch.inference_mode():
            out = model.generate(**enc,
                                 do_sample=True,
                                 temperature=max(temp, 1e-3),
                                 top_p=0.95,
                                 num_return_sequences=k,
                                 max_new_tokens=max_new,
                                 pad_token_id=tok.eos_token_id)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        if k > 2:
            return gen_batch(model, tok, prompt, k // 2, temp, max_new)
        raise
    L = enc['input_ids'].shape[1]
    return [tok.decode(o[L:], skip_special_tokens=True) for o in out]

def vote_outputs(verified, n_test):
    # verified: list of (src, result) ; result['test_outputs'] aligned by index
    outs = []
    for t in range(n_test):
        buckets = {}
        for src, res in verified:
            key = json.dumps(res['test_outputs'][t])
            buckets.setdefault(key, {'votes': 0, 'srcs': []})
            buckets[key]['votes'] += 1
            buckets[key]['srcs'].append(src)
        ranked = sorted(buckets.items(),
                        key=lambda kv: (-kv[1]['votes'],
                                        min(len(s) for s in kv[1]['srcs'])))
        top_key, top = ranked[0]
        alt = json.loads(ranked[1][0]) if len(ranked) > 1 else None
        outs.append({'attempt': json.loads(top_key),
                     'votes': top['votes'],
                     'alt': alt,
                     'program': min(top['srcs'], key=len)})
    return outs

def solve_task(tid, task, cfg, deadline_ts, rank, log):
    t0 = time.time()
    task_end = min(t0 + cfg['per_task_cap'], deadline_ts)
    rec = {'verified': False, 'n_verified': 0, 'outputs': [],
           'best_partial': 0, 'n_pairs': len(task['train']),
           'n_cand': 0, 'elapsed': 0.0}
    prompt = build_prompt(task)
    if len(prompt) > cfg['max_prompt_chars']:
        rec['skip'] = 'prompt_too_long(%d)' % len(prompt)
        log(f"[W{rank}] {tid}  SKIP prompt too long ({len(prompt)} chars)")
        return rec
    model, tok = cfg['_model'], cfg['_tok']
    test_inputs = [t['input'] for t in task['test']]

    seen, all_codes, all_res = set(), [], []
    def run_round(k, temp, repair=None):
        texts = gen_batch(model, tok, build_prompt(task, repair),
                          k, temp, cfg['max_new_tokens'])
        codes = []
        for txt in texts:
            c = extract_code(txt)
            h = code_key(c)
            if c and h not in seen:
                seen.add(h); codes.append(c)
        if not codes: return
        res = run_verifier(codes, task['train'], test_inputs,
                           cfg['prog_timeout'], f"{rank}_{tid}")
        all_codes.extend(codes); all_res.extend(res)

    run_round(cfg['k1'], cfg['temp1'])
    verified = [(c, r) for c, r in zip(all_codes, all_res)
                if r['status'] == 'verified']
    if not verified and time.time() < task_end - 30:
        partials = [(c, r) for c, r in zip(all_codes, all_res)
                    if r['status'] == 'partial' and r['pairs_passed'] >= 1]
        if partials:
            c, r = max(partials, key=lambda cr: (cr[1]['pairs_passed'],
                                                 -len(cr[0])))
            rep = {'code': c[:4000],
                   'pairs_passed': r['pairs_passed'],
                   'n_pairs': r['n_pairs'],
                   'fail_idx': r.get('idx', 0),
                   'diff': repair_diff(task, r)}
            run_round(cfg['k2'], cfg['temp2'], repair=rep)
            verified = [(c2, r2) for c2, r2 in zip(all_codes, all_res)
                        if r2['status'] == 'verified']

    rec['n_cand'] = len(all_codes)
    rec['best_partial'] = max([r.get('pairs_passed', 0) for r in all_res],
                              default=0)
    if verified:
        rec['verified']   = True
        rec['n_verified'] = len(verified)
        rec['outputs']    = vote_outputs(verified, len(test_inputs))
        rec['program']    = rec['outputs'][0]['program']
    rec['elapsed'] = round(time.time() - t0, 1)
    tag = (f"VERIFIED {rec['n_verified']} prog(s)" if rec['verified']
           else f"no verified (best {rec['best_partial']}/{rec['n_pairs']})")
    log(f"[W{rank}] {tid}  {tag}  cand {rec['n_cand']}  {rec['elapsed']}s")
    return rec

# ----------------------------------------------------------------------------
# GPU worker process (heavy imports live here -> spawn stays cheap)
# ----------------------------------------------------------------------------
def worker_main(rank, gpu_id, items, cfg, deadline_ts, partial_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
    def log(msg): print(msg, flush=True)
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        torch.manual_seed(cfg['seed'] + rank)
        random.seed(cfg['seed'] + rank)
        log(f"[W{rank}] loading {cfg['model_path']} on gpu {gpu_id} ...")
        tok = AutoTokenizer.from_pretrained(cfg['model_path'])
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            cfg['model_path'], torch_dtype=torch.bfloat16
        ).to('cuda').eval()
        cfg = dict(cfg); cfg['_model'] = model; cfg['_tok'] = tok
        log(f"[W{rank}] ready, {len(items)} tasks")
    except Exception as e:
        with open(partial_path, 'a') as f:
            f.write(json.dumps({'tid': '__worker_error__',
                                'error': repr(e)[:500]}) + "\n")
        print(f"[W{rank}] FATAL during model load: {e!r}", flush=True)
        return
    with open(partial_path, 'a') as f:
        for tid, task in items:
            if time.time() > deadline_ts - 25:
                log(f"[W{rank}] deadline reached, stopping")
                break
            try:
                rec = solve_task(tid, task, cfg, deadline_ts, rank, log)
            except Exception as e:
                rec = {'verified': False, 'error': repr(e)[:300]}
                log(f"[W{rank}] {tid}  ERROR {e!r}")
            f.write(json.dumps({'tid': tid, **{k: v for k, v in rec.items()
                                               if not k.startswith('_')}}) + "\n")
            f.flush()

# ----------------------------------------------------------------------------
# Data / model resolution
# ----------------------------------------------------------------------------
def resolve_data_dir():
    for d in DATA_DIR_CANDIDATES:
        if os.path.exists(os.path.join(d, "arc-agi_evaluation_challenges.json")) \
           or os.path.exists(os.path.join(d, "arc-agi_test_challenges.json")):
            return d
    return DATA_DIR_CANDIDATES[0]

def load_challenges():
    data_dir = resolve_data_dir()
    rerun  = bool(os.getenv("KAGGLE_IS_COMPETITION_RERUN"))
    test_p = os.path.join(data_dir, "arc-agi_test_challenges.json")
    eval_p = os.path.join(data_dir, "arc-agi_evaluation_challenges.json")
    sol_p  = os.path.join(data_dir, "arc-agi_evaluation_solutions.json")
    if rerun or (os.path.exists(test_p) and not os.path.exists(eval_p)):
        with open(test_p) as f:
            return json.load(f), None, 'SUBMIT', data_dir
    with open(eval_p) as f:
        ch = json.load(f)
    sols = None
    if os.path.exists(sol_p):
        with open(sol_p) as f:
            sols = json.load(f)
    return ch, sols, 'LOCAL_EVAL', data_dir

def resolve_model_path(override=None):
    cands = ([override] if override else []) + MODEL_PATH_CANDIDATES
    for p in cands:
        if p and os.path.exists(os.path.join(p, "config.json")):
            return p
    root = "/kaggle/input"
    if os.path.isdir(root):
        hits = []
        for dirpath, _, files in os.walk(root):
            if dirpath.count(os.sep) - root.count(os.sep) > 7:
                continue
            if "config.json" in files and (
                "tokenizer.json" in files or "tokenizer_config.json" in files):
                low = dirpath.lower()
                if "coder" in low or "qwen" in low:
                    hits.append((0 if "7b" in low else 1, dirpath))
        if hits:
            return sorted(hits)[0][1]
    return None

def est_cost(task):
    s = 0
    for p in task['train']:
        s += len(p['input']) * len(p['input'][0])
        s += len(p['output']) * len(p['output'][0])
    for t in task['test']:
        s += len(t['input']) * len(t['input'][0])
    return s

# ----------------------------------------------------------------------------
# Selftest: proves the sandbox (and the exec fix) with zero GPU/model
# ----------------------------------------------------------------------------
def selftest():
    print("== Leg C sandbox selftest (no GPU, no model) ==")
    train = [
        {'input':  [[0,0,0],[0,3,0],[0,0,0]],
         'output': [[0,0,0],[0,2,0],[0,0,0]]},
        {'input':  [[5,0],[0,0]],
         'output': [[2,0],[0,0]]},
    ]
    test = [[[0,7,0],[0,0,0],[0,0,0]]]
    cases = [
        ("dsl_reachable (old two-dict exec bug regression)",
         "def transform(grid):\n"
         "    objs = get_objects(grid)\n"
         "    return paint_objects(grid, objs, 2)\n", 'verified'),
        ("while loop must be rejected",
         "def transform(grid):\n"
         "    while True:\n        pass\n    return grid\n", 'rejected'),
        ("import must be rejected",
         "import os\ndef transform(grid):\n    return grid\n", 'rejected'),
        ("missing transform must be rejected",
         "def solve(grid):\n    return grid\n", 'rejected'),
        ("runaway loop must time out",
         "def transform(grid):\n"
         "    x = 0\n"
         "    for i in range(10**9):\n        x += i\n"
         "    return grid\n", 'timeout'),
        ("wrong logic must be partial",
         "def transform(grid):\n    return replace_color(grid, 3, 4)\n",
         'partial'),
        ("dunder escape must be rejected",
         "def transform(grid):\n"
         "    f = (1).__class__\n    return grid\n", 'rejected'),
    ]
    programs = [src for _, src, _ in cases]
    res = run_verifier(programs, train, test, 3.0, "selftest")
    ok = True
    for (name, _, want), r in zip(cases, res):
        got = r['status']
        mark = "PASS" if got == want else "FAIL"
        if got != want: ok = False
        print(f"  [{mark}] {name:48s} -> {got} "
              f"({r.get('error','')[:60]})")
    v = res[0]
    if v['status'] == 'verified':
        pred = v['test_outputs'][0]
        want = [[0,2,0],[0,0,0],[0,0,0]]
        mark = "PASS" if pred == want else "FAIL"
        if pred != want: ok = False
        print(f"  [{mark}] verified program predicts test correctly -> {pred}")
    names = re.findall(r"([a-z_][a-z0-9_]*)\(", CHEAT_SHEET)
    import ast as _ast
    tree = _ast.parse(RUNTIME_CODE)
    defined = {n.name for n in _ast.walk(tree)
               if isinstance(n, _ast.FunctionDef)}
    missing = [n for n in set(names) - {'f', 'color_fn', 'transform'}
               if n not in defined]
    mark = "PASS" if not missing else "FAIL"
    if missing: ok = False
    print(f"  [{mark}] cheat sheet <-> runtime consistency "
          f"(missing: {missing})")
    print("== selftest", "OK ==" if ok else "FAILED ==")
    return 0 if ok else 1

# ----------------------------------------------------------------------------
# Reporting: input + guessed result + mark
# ----------------------------------------------------------------------------
def mark_and_report(results, challenges, sols, cfg):
    audit, n_ver, n_out, n_correct = {}, 0, 0, 0
    printed = 0
    for tid, rec in results.items():
        if tid == '__worker_error__':
            continue
        task = challenges[tid]
        entry = {'verified': rec.get('verified', False),
                 'n_verified': rec.get('n_verified', 0),
                 'best_partial': rec.get('best_partial', 0),
                 'n_pairs': rec.get('n_pairs', 0),
                 'elapsed': rec.get('elapsed', 0),
                 'tests': []}
        for t_idx, t in enumerate(task['test']):
            out = (rec['outputs'][t_idx]['attempt']
                   if rec.get('verified') and t_idx < len(rec.get('outputs', []))
                   else None)
            exp = sols[tid][t_idx] if sols and tid in sols else None
            mark = None
            if out is not None:
                n_out += 1
                if exp is not None:
                    mark = (out == exp)
                    n_correct += int(mark)
            entry['tests'].append({'input': t['input'],
                                   'guess': out,
                                   'expected': exp,
                                   'mark': mark})
        if rec.get('verified'):
            n_ver += 1
            if printed < cfg['print_grids']:
                printed += 1
                for t_idx, tr in enumerate(entry['tests']):
                    sym = ('?' if tr['mark'] is None
                           else ('O' if tr['mark'] else 'X'))
                    print(f"\n--- {tid} test#{t_idx}  mark={sym} "
                          f"(votes {rec['outputs'][t_idx]['votes']}"
                          f"/{rec['n_verified']}) ---")
                    print("input:");  print(render_grid(tr['input']))
                    print("guess:");  print(render_grid(tr['guess']))
                    if tr['expected'] is not None:
                        print("expected:"); print(render_grid(tr['expected']))
        audit[tid] = entry
    with open(AUDIT_PATH, 'w') as f:
        json.dump(audit, f)
    print("\n================ Leg C summary ================")
    print(f" tasks attempted      : {len(audit)}")
    print(f" tasks verified       : {n_ver}")
    print(f" test outputs guessed : {n_out}")
    if sols is not None:
        pct = 100.0 * n_correct / max(n_out, 1)
        print(f" verified & correct   : {n_correct}/{n_out}  "
              f"(precision {pct:.1f}%)")
        solved = [t for t, e in audit.items()
                  if e['verified'] and all(x['mark'] for x in e['tests'])]
        print(f" fully-correct tasks  : {len(solved)} -> {solved}")
    print(f" results -> {RESULTS_PATH}")
    print(f" audit   -> {AUDIT_PATH}")
    print("===============================================")

def write_outputs(results, challenges):
    with open(RESULTS_PATH, 'w') as f:
        json.dump(results, f)
    # standalone submission (smoke/debug only; merge path is the real one)
    sub = {}
    for tid, task in challenges.items():
        rec = results.get(tid, {})
        entries = []
        for t_idx, t in enumerate(task['test']):
            if rec.get('verified') and t_idx < len(rec.get('outputs', [])):
                o = rec['outputs'][t_idx]
                a1 = o['attempt']
                a2 = o['alt'] if o.get('alt') else a1
            else:
                a1 = a2 = t['input']
            entries.append({'attempt_1': a1, 'attempt_2': a2})
        sub[tid] = entries
    with open(SUB_PATH, 'w') as f:
        json.dump(sub, f)

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--budget-h',     type=float, default=DEFAULTS['budget_h'])
    ap.add_argument('--per-task-cap', type=float, default=DEFAULTS['per_task_cap'])
    ap.add_argument('--k1',   type=int, default=DEFAULTS['k1'])
    ap.add_argument('--k2',   type=int, default=DEFAULTS['k2'])
    ap.add_argument('--workers', type=int, default=0)
    ap.add_argument('--model-path', type=str, default=None)
    ap.add_argument('--smoke', type=int, default=0)
    ap.add_argument('--max-tasks', type=int, default=0)
    ap.add_argument('--print-grids', type=int, default=DEFAULTS['print_grids'])
    ap.add_argument('--end-ts', type=float, default=0.0,
                    help='absolute unix ts; deadline = min(now+budget, end-ts)')
    ap.add_argument('--selftest', action='store_true')
    args = ap.parse_args()

    os.makedirs(WORK_DIR, exist_ok=True)
    with open(RUNTIME_PATH, 'w') as f:
        f.write(RUNTIME_CODE)

    if args.selftest:
        sys.exit(selftest())

    challenges, sols, mode, data_dir = load_challenges()
    print(f"[LegC] mode={mode}  data={data_dir}  tasks={len(challenges)}")

    model_path = resolve_model_path(args.model_path)
    if model_path is None:
        print("[LegC] WARNING: no coder model found under /kaggle/input.")
        print("       Attach Qwen2.5-Coder-7B-Instruct, or pass --model-path.")
        print("       Writing empty results so the merge step is a no-op.")
        write_outputs({}, challenges)
        mark_and_report({}, challenges, sols,
                        {'print_grids': 0})
        return
    print(f"[LegC] model: {model_path}")

    cfg = dict(DEFAULTS)
    cfg.update(per_task_cap=args.per_task_cap, k1=args.k1, k2=args.k2,
               print_grids=args.print_grids, model_path=model_path)

    items = sorted(challenges.items(), key=lambda kv: est_cost(kv[1]))
    if args.smoke:
        items = items[:args.smoke]
        cfg['print_grids'] = max(cfg['print_grids'], args.smoke)
    elif args.max_tasks:
        items = items[:args.max_tasks]

    try:
        n_gpu = int(subprocess.run(
            ['nvidia-smi', '-L'], capture_output=True, text=True
        ).stdout.strip().count('GPU '))
    except Exception:
        n_gpu = 1
    n_workers = args.workers or max(1, min(4, n_gpu))
    shards = [items[i::n_workers] for i in range(n_workers)]

    deadline_ts = time.time() + args.budget_h * 3600
    if args.end_ts > 0:
        deadline_ts = min(deadline_ts, args.end_ts)
    remain = deadline_ts - time.time()
    if remain < 300:
        print(f"[LegC] budget too small ({remain:.0f}s) -- writing empty "
              "results and exiting so the base pipeline keeps all its time.")
        write_outputs({}, challenges)
        mark_and_report({}, challenges, sols, {'print_grids': 0})
        return
    print(f"[LegC] {len(items)} tasks, {n_workers} worker(s), "
          f"budget {remain/3600:.2f} h")

    for r in range(n_workers):
        p = PARTIAL_TPL.format(rank=r)
        if os.path.exists(p): os.remove(p)

    ctx = mp.get_context('spawn')
    procs = []
    for r in range(n_workers):
        p = ctx.Process(target=worker_main,
                        args=(r, r % max(n_gpu, 1), shards[r], cfg,
                              deadline_ts, PARTIAL_TPL.format(rank=r)))
        p.start(); procs.append(p)
    hard_stop = deadline_ts + 600
    for p in procs:
        p.join(max(1.0, hard_stop - time.time()))
    for p in procs:
        if p.is_alive():
            p.terminate(); p.join()

    results = {}
    for r in range(n_workers):
        path = PARTIAL_TPL.format(rank=r)
        if not os.path.exists(path): continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line: continue
                try:
                    rec = json.loads(line)
                    results[rec.pop('tid')] = rec
                except Exception:
                    continue

    write_outputs(results, challenges)
    mark_and_report(results, challenges, sols, cfg)

if __name__ == '__main__':
    main()


# CELL 7 (code)
# ============ Leg C sandbox selftest (CPU, ~30s) ============
# 9/9 PASS expected: DSL sandbox, AST enforcement, per-program timeouts.
!python arc_induction_v2.py --selftest


# CELL 8 (code)
# ===================== Leg C launch (BEFORE base pipeline) =====================
# Verified-induction pre-pass (Qwen2.5-Coder-7B-Instruct, 4 x L4 workers).
# Base is near time-saturated on the hidden set, so Leg C wall time is NOT free:
# keep LEGC_BUDGET_H small; the starter-level skip refunds verified tasks.
# arc_induction_v2.py exits safely (empty results) if the budget is < 5 min,
# if the coder model is not attached, or on any internal failure -> the merge
# becomes a no-op and the base score is untouched.
LEGC_BUDGET_H  = 1.0   # 0 disables Leg C (safe no-op)
BASE_RESERVE_H = 9.8   # wall hours guaranteed to remain for the base pipeline

LEGC_END_TS = f"{global_end_time - BASE_RESERVE_H * 3600:.0f}"
LEGC_BUDGET = f"{LEGC_BUDGET_H:.3f}"

!python arc_induction_v2.py --budget-h {LEGC_BUDGET} --end-ts {LEGC_END_TS}


# CELL 9 (code)
!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 python starter.py --end-time {global_end_time}

# CELL 10 (code)
import os
import json
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

if rerun_mode:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json")
else:
    data = ArcDataset.from_file("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json")
    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")

decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)

decoder.load_decoded_results("/kaggle/inference_outputs")

submission = data.get_submission(decoder.run_selection_algo())

with open("submission.json", "w") as f:
    json.dump(submission, f)

if not rerun_mode:
    decoder.benchmark_selection_algos()
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))

# CELL 11 (code)
# ===================== Leg C merge (run AFTER the submission cell) =====================
# Monotonic union: train-verified induction outputs pre-empt attempt_1, the
# base candidate is demoted to attempt_2 (cross-family decorrelation).
# Tasks without a verified program are left untouched.
import os, json

SUB = "submission.json"
IND = "/kaggle/working/induction_results.json"

with open(SUB) as f:
    sub = json.load(f)
ind = {}
if os.path.exists(IND):
    with open(IND) as f:
        ind = json.load(f)

rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

def local_score(s):
    try:
        return data.validate_submission(s)   # `data` from the submission cell
    except Exception:
        return None

before = None if rerun_mode else local_score(sub)

promoted = created = 0
for tid, rec in ind.items():
    if not isinstance(rec, dict) or not rec.get("verified"):
        continue
    for ti, o in enumerate(rec.get("outputs", [])):
        g = o.get("attempt")
        if not g:
            continue
        alt = o.get("alt")
        if tid not in sub:
            sub[tid] = []
        while len(sub[tid]) <= ti:
            sub[tid].append({"attempt_1": [[0]], "attempt_2": [[0]]})
            created += 1
        e = sub[tid][ti]
        if e["attempt_1"] != g:
            demoted = e["attempt_1"]
            e["attempt_2"] = demoted if demoted != [[0]] else (alt or demoted)
            e["attempt_1"] = g
            promoted += 1
        elif alt and e.get("attempt_2") == [[0]]:
            e["attempt_2"] = alt

with open(SUB, "w") as f:
    json.dump(sub, f)

after = None if rerun_mode else local_score(sub)
print(f"[LegC merge] promoted={promoted} created={created}")
if before is not None and after is not None:
    print(f"[LegC merge] local score: base {before:.2f} -> merged {after:.2f} (unit = tasks)")

