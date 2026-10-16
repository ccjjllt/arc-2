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
import hashlib

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
    "膴": 10,
    "<|im_end|>": 15,
}

ARC_TOKENS = list(ARC_VOCAB.values())
USER_TOKEN_ID = 11
ASSISTANT_TOKEN_ID = 12
PAD_ID = 13
EOS_ID = 15


def env_int(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def env_float(name, default):
    value = os.getenv(name)
    if value is None:
        return default
    try:
        return float(value)
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


def parse_int_list(raw, default):
    if raw is None or raw.strip() == "":
        raw = default
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(int(item))
    return result


def parse_float_list(raw, default):
    if raw is None or raw.strip() == "":
        raw = default
    result = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        result.append(float(item))
    return result


def list_value(values, idx):
    if not values:
        return None
    if idx < len(values):
        return values[idx]
    return values[-1]


def get_queue_size(queue):
    try:
        return int(queue.qsize())
    except Exception:
        return -1


def stable_seed_from_key(key, mod=1024**2):
    if key is None:
        return 0
    digest = hashlib.blake2b(str(key).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little") % mod


def compute_puzzle_budget(
    budget_mode,
    remaining_global_time,
    pending_after_pop,
    max_puzzle_seconds,
    min_puzzle_seconds,
    world_size,
):
    if pending_after_pop < 0:
        return max_puzzle_seconds
    tasks_left = max(1, pending_after_pop + 1)
    if budget_mode == "adaptive_parallel":
        adaptive_budget = remaining_global_time * max(1, world_size) / tasks_left
        return min(max_puzzle_seconds, max(min_puzzle_seconds, adaptive_budget * 1.1))
    if budget_mode == "adaptive_serial":
        adaptive_budget = remaining_global_time / tasks_left
        return min(max_puzzle_seconds, max(min_puzzle_seconds, adaptive_budget * 1.25))
    return max_puzzle_seconds


def build_decode_batches(subkeys):
    if len(subkeys) == 16:
        batches = []
        for offsets in ([0, 4], [2, 6], [8, 12], [10, 14]):
            batch = []
            for offset in offsets:
                batch.extend(subkeys[offset:offset+2])
            if len(batch):
                batches.append(batch)
        return batches
    return [subkeys[i:i+4] for i in range(0, len(subkeys), 4) if subkeys[i:i+4]]


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
        # 馃敡 KEY FIX: Clone the loss tensor before in-place operations
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


def turbo_dfs(model, logits, max_new_tokens, max_score, scores, pos, cache, start_time, end_time, dfs_time_limit) -> dict:

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
    
    while time.time() - start_time < dfs_time_limit and time.time() < end_time:

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
            dfs_time_limit=dfs_time_limit,
        )

        for batch_id, beams in next_suffixes.items():
            for score, suffix_tokens in beams:
                suffix_tokens.insert(0, batch_tokens[batch_id])
                suffixes[batch_id].append((score, suffix_tokens))

    return suffixes


@torch.no_grad()
def inference_turbo_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time, dfs_time_limit):
    # Prefix lengths can differ after augment/cut, so decode per-length bucket.
    len_to_indices = defaultdict(list)
    for idx, toks in enumerate(prefix_tokens):
        len_to_indices[len(toks)].append(idx)

    result = []
    for _, indices in sorted(len_to_indices.items(), key=lambda x: x[0], reverse=True):
        bucket = [prefix_tokens[i] for i in indices]
        input_ids = torch.tensor(bucket, device=model.device, dtype=torch.long)
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
            dfs_time_limit=dfs_time_limit,
        )
        for local_batch_id, beams in suffixes.items():
            global_batch_id = indices[local_batch_id]
            sorted_beams = sorted(beams, key=lambda x:x[0])
            result.append((global_batch_id, sorted_beams))
    result = sorted(result, key=lambda x: x[0])
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

    model_name = env_str("ARC_MODEL_NAME", "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1")
    max_seq_length = env_int("ARC_MAX_SEQ_LENGTH", 8192)
    train_aug_n = max(1, env_int("ARC_TRAIN_AUG_N", 16))
    eval_aug_n = max(1, env_int("ARC_EVAL_AUG_N", 2))
    dfs_time_limit = env_float("ARC_DFS_TIME_LIMIT", 540.0)
    min_token_prob = min(max(env_float("ARC_MIN_TOKEN_PROB", 0.2), 1e-6), 1 - 1e-6)
    budget_mode = env_str("ARC_BUDGET_MODE", "adaptive_parallel").strip().lower()
    max_puzzle_seconds = env_float("ARC_MAX_PUZZLE_SECONDS", 1200.0)
    min_puzzle_seconds = env_float("ARC_MIN_PUZZLE_SECONDS", 420.0)
    world_size = max(1, env_int("ARC_WORLD_SIZE", env_int("ARC_NPROCS", 4)))
    augment_mode = env_str("ARC_AUGMENT_MODE", "default").strip().lower()
    train_data_seed = env_int("ARC_TRAIN_DATA_SEED", 1)
    score_seed_offset = env_int("ARC_SCORE_SEED_OFFSET", 0)
    force_all_rounds = env_bool("ARC_FORCE_ALL_ROUNDS", False)
    rescue_aggressive = env_bool("ARC_RESCUE_AGGRESSIVE", False)
    round1_max_fraction = env_float("ARC_ROUND1_MAX_FRACTION", 0.72)
    round1_min_missing_ratio = env_float("ARC_ROUND1_MIN_MISSING_RATIO", 0.08)
    round1_require_incomplete = env_bool("ARC_ROUND1_REQUIRE_INCOMPLETE", True)
    max_scored_beams_round0 = max(0, env_int("ARC_MAX_SCORED_BEAMS_ROUND0", 0))
    max_scored_beams_roundn = max(0, env_int("ARC_MAX_SCORED_BEAMS_ROUNDN", 12))

    eval_seed_list = parse_int_list(os.getenv("ARC_EVAL_SEEDS"), "2")
    eval_aug_n_list = parse_int_list(os.getenv("ARC_EVAL_AUG_NS"), str(eval_aug_n))
    min_token_prob_list = parse_float_list(os.getenv("ARC_MIN_TOKEN_PROBS"), str(min_token_prob))
    dfs_time_limit_list = parse_float_list(os.getenv("ARC_DFS_TIME_LIMITS"), str(dfs_time_limit))

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
        fsdp="",
        ddp_find_unused_parameters=False,
        dataloader_num_workers=0,
        gradient_checkpointing=False,
    )

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        full_finetuning=False,
        load_in_4bit=False,
        local_files_only=True,
        use_gradient_checkpointing=False,
        max_seq_length=max_seq_length,
    )

    model = FastLanguageModel.get_peft_model(model, **peft_params)

    for _, param in model.named_parameters():
        if param.dtype == torch.float32:
            param.data = param.data.to(torch.bfloat16)

    default_weights = get_peft_model_state_dict(model, adapter_name="default")
    default_weights = {k: v.clone().detach() for k, v in default_weights.items()}

    collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
    formatter = QwenFormatter(tokenizer=tokenizer)
    max_new_tokens = formatter.max_new_tokens()

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    arc_test_set = ArcDataset.from_file(test_path)

    dir_outputs = env_str("ARC_OUTPUT_DIR", "/kaggle/inference_outputs")
    run_tag = env_str("ARC_RUN_TAG", "").strip()
    os.makedirs(dir_outputs, exist_ok=True)

    print(
        f"[Rank {rank}] budget_mode={budget_mode}, world_size={world_size}, "
        f"train_aug_n={train_aug_n}, eval_rounds={len(eval_seed_list)}, "
        f"base_eval_aug_n={eval_aug_n}, base_dfs_limit={dfs_time_limit:.0f}s, "
        f"round1_max_fraction={round1_max_fraction:.2f}, run_tag='{run_tag}', "
        f"augment_mode='{augment_mode}'"
    )

    while True:
        if time.time() > end_time:
            print(f"[Rank {rank}] stop!")
            break

        key = queue.get()
        if key is None:
            break

        start_time = time.time()
        pending_after_pop = get_queue_size(queue)
        remaining_global_time = max(1.0, end_time - start_time)
        puzzle_budget = compute_puzzle_budget(
            budget_mode=budget_mode,
            remaining_global_time=remaining_global_time,
            pending_after_pop=pending_after_pop,
            max_puzzle_seconds=max_puzzle_seconds,
            min_puzzle_seconds=min_puzzle_seconds,
            world_size=world_size,
        )
        puzzle_deadline = min(end_time, start_time + puzzle_budget)

        torch.cuda.reset_peak_memory_stats()

        set_peft_model_state_dict(
            model,
            default_weights.copy(),
            adapter_name="default",
        )

        model = FastLanguageModel.for_training(model)
        puzzle_ds = arc_test_set.change_keys([key])

        if augment_mode == "extended":
            train_ds = puzzle_ds.augment_v2(n=train_aug_n, shfl_keys=True, seed=train_data_seed)
        else:
            train_ds = puzzle_ds.augment(n=train_aug_n, shfl_keys=True, seed=train_data_seed)
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
        print(
            f"[Rank {rank}] training stats for puzzle {key}: {stats}; "
            f"budget={puzzle_budget:.1f}s pending={pending_after_pop}"
        )

        puzzle_ds_multi = puzzle_ds.split_multi_replies()

        with torch.inference_mode():
            known_scores = {}
            round0_coverage = None
            for round_id, eval_seed in enumerate(eval_seed_list):
                spend_time = time.time() - start_time
                if spend_time > puzzle_budget or time.time() > puzzle_deadline:
                    print(f"[Rank {rank}] timeout before round {round_id} after {spend_time:.1f}s for puzzle {key}")
                    break

                if round_id > 0 and not force_all_rounds:
                    if (spend_time / max(1.0, puzzle_budget)) > round1_max_fraction:
                        print(
                            f"[Rank {rank}] skip round {round_id} for {key}: "
                            f"time fraction {spend_time / max(1.0, puzzle_budget):.2f} > {round1_max_fraction:.2f}"
                        )
                        break
                    if round1_require_incomplete and round0_coverage is not None:
                        round0_missing = 1.0 - round0_coverage
                        if round0_missing < round1_min_missing_ratio:
                            print(
                                f"[Rank {rank}] skip round {round_id} for {key}: "
                                f"round0 missing {round0_missing:.2f} < {round1_min_missing_ratio:.2f}"
                            )
                            break

                round_eval_aug_n = max(1, int(list_value(eval_aug_n_list, round_id)))
                round_min_token_prob = min(max(float(list_value(min_token_prob_list, round_id)), 1e-6), 1 - 1e-6)
                round_max_score = -np.log(round_min_token_prob)
                round_dfs_time_limit = float(list_value(dfs_time_limit_list, round_id))

                eval_ds = puzzle_ds_multi.augment(n=round_eval_aug_n, seed=eval_seed)
                eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

                test_id_to_subkeys = defaultdict(list)
                for subkey in sorted(eval_ds.keys):
                    test_id = subkey.split(".")[0].split("_")[1]
                    test_id_to_subkeys[test_id].append(subkey)

                batches = []
                for _, subkeys in test_id_to_subkeys.items():
                    batches.extend(build_decode_batches(subkeys))
                round_written_subkeys = set()
                round_total_subkeys = len(eval_ds.keys)

                print(
                    f"[Rank {rank}] round={round_id} eval_seed={eval_seed} "
                    f"eval_aug_n={round_eval_aug_n} min_prob={round_min_token_prob:.3f} "
                    f"dfs_limit={round_dfs_time_limit:.0f}s batches={len(batches)}"
                )

                for subkeys in batches:
                    spend_time = time.time() - start_time
                    if spend_time > puzzle_budget or time.time() > puzzle_deadline:
                        print(f"[Rank {rank}] timeout after {spend_time:.1f}s for puzzle {key}")
                        break

                    print(f"[Rank {rank}] decoding {subkeys}")

                    tokens = []
                    for subkey in subkeys:
                        data = eval_ds.get(subkey, formatter)
                        tokens.append(tokenizer.encode(data["input"]))

                    dfs_result = inference_turbo_dfs(
                        model=model,
                        prefix_tokens=tokens,
                        max_new_tokens=max_new_tokens,
                        max_score=round_max_score,
                        end_time=puzzle_deadline,
                        dfs_time_limit=round_dfs_time_limit,
                    )

                    for subkey_id, scored_beams in dfs_result:
                        subkey = subkeys[subkey_id]
                        bk = subkey.split(".")[0]
                        decoded_result = []
                        max_scored_beams = max_scored_beams_round0 if round_id == 0 else max_scored_beams_roundn

                        for beam_score, tokens in scored_beams:
                            if max_scored_beams > 0 and len(decoded_result) >= max_scored_beams:
                                break
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
                                aug_dataset = aug_dataset.augment(seed=stable_seed_from_key(f"{bk}:{score_seed_offset}"))
                                aug_dataset = aug_dataset.cut_to_len(
                                    formatter=formatter,
                                    name="input",
                                    max_len=max_seq_length-max_new_tokens,
                                )
                                aug_queries = []
                                aug_answers = []
                                for augmented_sample in aug_dataset.as_list(formatter):
                                    aug_queries.append(augmented_sample["input"])
                                    aug_answers.append(augmented_sample["reply"])

                                if len(aug_queries) == 0:
                                    augmented_scores = [10.0]
                                else:
                                    split = max(1, len(aug_queries) // 2)
                                    augmented_scores1 = calc_scores(aug_queries[:split], aug_answers[:split], tokenizer, model)
                                    augmented_scores2 = (
                                        calc_scores(aug_queries[split:], aug_answers[split:], tokenizer, model)
                                        if split < len(aug_queries) else []
                                    )
                                    augmented_scores = augmented_scores1 + augmented_scores2

                                known_scores[grid_id] = augmented_scores

                            decoded_result.append({
                                "beam_score": beam_score,
                                "score_aug": augmented_scores,
                                "solution": solution,
                                "_round": round_id,
                                "_eval_seed": eval_seed,
                            })

                        if len(decoded_result):
                            round_written_subkeys.add(subkey)
                            tag_part = f".{run_tag}" if run_tag else ""
                            store_key = f"{subkey}{tag_part}.run{round_id}"
                            with bz2.BZ2File(os.path.join(dir_outputs, store_key), "w") as f:
                                pickle.dump(decoded_result, f)

                if round_id == 0:
                    round0_coverage = len(round_written_subkeys) / max(1, round_total_subkeys)
                    print(
                        f"[Rank {rank}] round0 coverage for {key}: "
                        f"{len(round_written_subkeys)}/{round_total_subkeys} = {round0_coverage:.2f}"
                    )

            # Auto-aggressive rescue: when coverage is zero after all rounds, retry
            # with much lower threshold before giving up entirely.
            if rescue_aggressive:
                spend_time = time.time() - start_time
                remaining_budget = puzzle_budget - spend_time
                if round0_coverage is not None and round0_coverage < 0.05 and remaining_budget > 120:
                    aggressive_min_prob = max(0.03, round_min_token_prob * 0.25)
                    aggressive_max_score = -np.log(aggressive_min_prob)
                    aggressive_dfs_limit = min(float(list_value(dfs_time_limit_list, 0)) * 0.6, remaining_budget * 0.5)
                    aggressive_eval_seed = int(list_value(eval_seed_list, 0)) + 99
                    aggressive_eval_aug_n = max(2, int(list_value(eval_aug_n_list, 0)) * 2)

                    print(
                        f"[Rank {rank}] aggressive retry for {key}: "
                        f"min_prob={aggressive_min_prob:.4f} dfs_limit={aggressive_dfs_limit:.0f}s "
                        f"eval_seed={aggressive_eval_seed} eval_aug_n={aggressive_eval_aug_n}"
                    )

                    eval_ds = puzzle_ds_multi.augment(n=aggressive_eval_aug_n, seed=aggressive_eval_seed)
                    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)

                    test_id_to_subkeys = defaultdict(list)
                    for subkey in sorted(eval_ds.keys):
                        test_id = subkey.split(".")[0].split("_")[1]
                        test_id_to_subkeys[test_id].append(subkey)

                    batches = []
                    for _, subkeys in test_id_to_subkeys.items():
                        batches.extend(build_decode_batches(subkeys))

                    for subkeys in batches:
                        spend_time = time.time() - start_time
                        if spend_time > puzzle_budget or time.time() > puzzle_deadline:
                            print(f"[Rank {rank}] aggressive timeout for {key}")
                            break

                        tokens = []
                        for subkey in subkeys:
                            data = eval_ds.get(subkey, formatter)
                            tokens.append(tokenizer.encode(data["input"]))

                        dfs_result = inference_turbo_dfs(
                            model=model,
                            prefix_tokens=tokens,
                            max_new_tokens=max_new_tokens,
                            max_score=aggressive_max_score,
                            end_time=puzzle_deadline,
                            dfs_time_limit=aggressive_dfs_limit,
                        )

                        for subkey_id, scored_beams in dfs_result:
                            subkey = subkeys[subkey_id]
                            bk = subkey.split(".")[0]
                            decoded_result = []
                            for beam_score, tokens in scored_beams:
                                array = formatter.convert_tokens_to_array(tokens)
                                if array is None:
                                    continue
                                solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                                decoded_result.append({
                                    "beam_score": beam_score,
                                    "score_aug": [beam_score],
                                    "solution": solution,
                                    "_round": -1,
                                    "_eval_seed": aggressive_eval_seed,
                                })

                            if len(decoded_result):
                                tag_part = f".{run_tag}" if run_tag else ""
                                store_key = f"{subkey}{tag_part}.run_aggr"
                                with bz2.BZ2File(os.path.join(dir_outputs, store_key), "w") as f:
                                    pickle.dump(decoded_result, f)

        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2
        print(f"[Rank {rank}] allocated {memory_allocated}MB for inference")
        spend_time = time.time() - start_time
        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")
