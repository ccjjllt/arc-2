from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
from arc_loader import ArcDataset, QwenFormatter

import gc
import os
import io
import importlib.metadata
import json
import signal
import sys
import tempfile
import time
import torch
import numpy as np
from datasets import Dataset
from collections import defaultdict
from queue import Empty

from typing import Any, Union
from transformers import DataCollatorForLanguageModeling, TrainerCallback

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


class DeadlineReached(RuntimeError):
    pass


def check_deadline(deadline, stage):
    if time.time() >= deadline:
        raise DeadlineReached(stage)


class TrainingDeadlineCallback(TrainerCallback):
    """Stop at natural training boundaries; the parent owns the hard kill."""

    def __init__(self, deadline):
        self.deadline = deadline
        self.stopped_for_deadline = False

    def _check(self, control):
        if time.time() >= self.deadline:
            self.stopped_for_deadline = True
            control.should_training_stop = True
            control.should_epoch_stop = True
        return control

    def on_train_begin(self, args, state, control, **kwargs):
        return self._check(control)

    def on_step_begin(self, args, state, control, **kwargs):
        return self._check(control)

    def on_substep_end(self, args, state, control, **kwargs):
        return self._check(control)

    def on_step_end(self, args, state, control, **kwargs):
        return self._check(control)


def atomic_save_candidates(path, candidates):
    """A killed worker must leave either the previous complete file or the new one."""
    fd, temporary_path = tempfile.mkstemp(prefix=".candidate-", suffix=".tmp", dir=os.path.dirname(path))
    os.close(fd)
    try:
        with bz2.BZ2File(temporary_path, "w") as handle:
            pickle.dump(candidates, handle, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(temporary_path, path)
    finally:
        if os.path.exists(temporary_path):
            os.unlink(temporary_path)


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
    # pending_after_pop counts real queued tasks, excluding sentinels. Account
    # conservatively for up to world_size tasks already running, including ours.
    # A minimum budget must never enlarge a fair share and starve the tail.
    remaining = max(0.0, remaining_global_time)
    if pending_after_pop < 0:
        return min(remaining, max_puzzle_seconds)
    parallelism = max(1, world_size) if budget_mode != "adaptive_serial" else 1
    tasks_left = max(parallelism, pending_after_pop + parallelism)
    fair_share = remaining * parallelism / tasks_left
    return max(0.0, min(remaining, max_puzzle_seconds, fair_share))


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
    if time.time() >= min(end_time, start_time + dfs_time_limit):
        return {}
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
    start_time = time.time()
    decode_deadline = min(end_time, start_time + dfs_time_limit)
    for _, indices in sorted(len_to_indices.items(), key=lambda x: x[0], reverse=True):
        if time.time() >= decode_deadline:
            break
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
            start_time=start_time,
            end_time=decode_deadline,
            dfs_time_limit=dfs_time_limit,
        )
        for local_batch_id, beams in suffixes.items():
            global_batch_id = indices[local_batch_id]
            sorted_beams = sorted(beams, key=lambda x:x[0])
            result.append((global_batch_id, sorted_beams))
    result = sorted(result, key=lambda x: x[0])
    return result


@torch.no_grad()
def calc_scores(queries, answers, tokenizer, model, deadline):
    check_deadline(deadline, "score_tokenize")
    if not queries or len(queries) != len(answers):
        raise ValueError("Scoring requires matched nonempty queries and answers")
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
    check_deadline(deadline, "score_forward")
    outputs = model(input_ids=input_ids, return_dict=True, use_cache=False)
    batch_logits = outputs.logits.float().cpu().log_softmax(-1)
    result = []
    for logits, query_tokens, answer_tokens in zip(batch_logits, batch_query_tokens, batch_answer_tokens):
        query_length = len(query_tokens)
        answer_logits = logits[query_length-1:query_length-1+len(answer_tokens)]
        answer_score = answer_logits[torch.arange(len(answer_tokens)), answer_tokens].sum()
        result.append(-answer_score.item())
    return result


def worker(rank, queue, end_time):
    """One independent GPU worker; end_time is an absolute Unix timestamp."""
    dir_outputs = env_str("ARC_OUTPUT_DIR", "/kaggle/inference_outputs")
    event_dir = env_str("ARC_EVENT_DIR", os.path.join(dir_outputs, "events"))
    os.makedirs(dir_outputs, exist_ok=True)
    os.makedirs(event_dir, exist_ok=True)
    event_path = os.path.join(event_dir, f"worker-{rank}.jsonl")
    active_key = None

    def emit(event, **fields):
        record = dict(event=event, timestamp=time.time(), rank=rank,
                      pid=os.getpid(), task_key=active_key, **fields)
        line = json.dumps(record, ensure_ascii=False, default=str)
        print("ARC_EVENT " + line, flush=True)
        with open(event_path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()

    def handle_signal(signum, frame):
        emit("worker_signal", signal_number=signum,
             signal_name=signal.Signals(signum).name)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, handle_signal)
    signal.signal(signal.SIGINT, handle_signal)

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower() in {"1", "true", "yes"}
    default_challenges = (
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
        if rerun_mode else
        "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"
    )
    test_path = env_str("ARC_CHALLENGES_PATH", default_challenges)
    model_name = env_str("ARC_MODEL_PATH", env_str(
        "ARC_MODEL_NAME", "/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1"))
    max_seq_length = env_int("ARC_MAX_SEQ_LENGTH", 8192)
    train_aug_n = max(1, env_int("ARC_TRAIN_AUG_N", 16))
    eval_aug_n = max(1, env_int("ARC_EVAL_AUG_N", 2))
    eval_seed = env_int("ARC_EVAL_SEED", 2)
    dfs_time_limit = max(0.0, env_float("ARC_DFS_TIME_LIMIT", 540.0))
    min_token_prob = min(max(env_float("ARC_MIN_TOKEN_PROB", 0.2), 1e-6), 1 - 1e-6)
    max_puzzle_seconds = max(0.0, env_float("ARC_MAX_PUZZLE_SECONDS", 1200.0))
    world_size = max(1, env_int("ARC_WORLD_SIZE", env_int("ARC_NPROCS", 4)))
    queue_sentinels = max(0, env_int("ARC_QUEUE_SENTINELS", world_size))
    train_data_seed = env_int("ARC_TRAIN_DATA_SEED", 1)
    score_seed_offset = env_int("ARC_SCORE_SEED_OFFSET", 0)
    train_max_fraction = min(0.8, max(0.05, env_float("ARC_TRAIN_MAX_FRACTION", 0.4)))
    train_max_seconds = max(0.0, env_float("ARC_TRAIN_MAX_SECONDS", 480.0))
    inference_reserve = max(0.0, env_float("ARC_INFERENCE_RESERVE_SECONDS", 60.0))
    scoring_reserve = max(1.0, env_float("ARC_SCORING_RESERVE_SECONDS", 30.0))
    max_scored_beams = max(0, env_int("ARC_MAX_SCORED_BEAMS", 0))
    run_tag = env_str("ARC_RUN_TAG", "base").strip()
    load_deadline = min(end_time, time.time() + max(1.0, env_float("ARC_MODEL_LOAD_MAX_SECONDS", 600.0)))
    with open(__file__, "rb") as handle:
        source_sha256 = hashlib.sha256(handle.read()).hexdigest()

    emit("worker_start", source_path=os.path.abspath(__file__), source_sha256=source_sha256,
         model_path=model_name, challenges_path=test_path, output_dir=dir_outputs,
         event_path=event_path, global_deadline_epoch=end_time,
         model_load_deadline_epoch=load_deadline, world_size=world_size,
         queue_sentinels=queue_sentinels, train_aug_n=train_aug_n, eval_aug_n=eval_aug_n,
         min_token_prob=min_token_prob, dfs_time_limit=dfs_time_limit,
         train_max_fraction=train_max_fraction, train_max_seconds=train_max_seconds,
         run_tag=run_tag, candidate_source="base_ttt", rescue_enabled=False)

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
        output_dir=os.path.join(event_dir, f"trainer-{rank}"),
    )

    try:
        check_deadline(load_deadline, "model_load_start")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            full_finetuning=False,
            load_in_4bit=False,
            local_files_only=True,
            use_gradient_checkpointing=False,
            max_seq_length=max_seq_length,
        )
        check_deadline(load_deadline, "model_loaded")
        model = FastLanguageModel.get_peft_model(model, **peft_params)
        for _, param in model.named_parameters():
            if param.dtype == torch.float32:
                param.data = param.data.to(torch.bfloat16)
        default_weights = get_peft_model_state_dict(model, adapter_name="default")
        default_weights = {k: v.clone().detach() for k, v in default_weights.items()}
        collator = QwenDataCollatorForCompletionOnlyLM(tokenizer=tokenizer, mlm=False)
        formatter = QwenFormatter(tokenizer=tokenizer)
        max_new_tokens = formatter.max_new_tokens()
        arc_test_set = ArcDataset.from_file(test_path)
        check_deadline(load_deadline, "worker_ready")
        package_versions = {}
        module_paths = {}
        for package in ("torch", "transformers", "unsloth", "unsloth_zoo", "peft"):
            try:
                package_versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                package_versions[package] = "distribution_metadata_unavailable"
            module_paths[package] = getattr(sys.modules.get(package), "__file__", None)
        cuda_device = torch.cuda.current_device()
        emit("model_loaded", task_count=len(arc_test_set.keys),
             max_seq_length=max_seq_length, max_new_tokens=max_new_tokens,
             lora_rank=peft_params["r"], model_path=model_name,
             package_versions=package_versions, module_paths=module_paths,
             torch_runtime_version=torch.__version__, cuda_runtime_version=torch.version.cuda,
             cuda_device_index=cuda_device, gpu_name=torch.cuda.get_device_name(cuda_device),
             gpu_device_capability=list(torch.cuda.get_device_capability(cuda_device)),
             gpu_total_memory_bytes=torch.cuda.get_device_properties(cuda_device).total_memory)

        while time.time() < end_time:
            try:
                key = queue.get(timeout=min(1.0, max(0.01, end_time - time.time())))
            except Empty:
                continue
            if key is None:
                break
            active_key = key
            start_time = time.time()
            queued_entries = get_queue_size(queue)
            pending_after_pop = max(0, queued_entries - queue_sentinels) if queued_entries >= 0 else -1
            puzzle_budget = compute_puzzle_budget(
                budget_mode="adaptive_parallel",
                remaining_global_time=end_time - start_time,
                pending_after_pop=pending_after_pop,
                max_puzzle_seconds=max_puzzle_seconds,
                min_puzzle_seconds=0.0,
                world_size=world_size,
            )
            puzzle_deadline = min(end_time, start_time + puzzle_budget)
            train_allowance = min(train_max_seconds, puzzle_budget * train_max_fraction,
                                  max(0.0, puzzle_budget - inference_reserve))
            train_deadline = min(puzzle_deadline, start_time + train_allowance)
            trained_steps = 0
            saved_candidates = 0
            saved_subkeys = set()
            done_status = "complete"
            emit("task_start", task_deadline_epoch=puzzle_deadline,
                 train_deadline_epoch=train_deadline, task_budget_seconds=puzzle_budget,
                 pending_tasks_after_pop=pending_after_pop, queued_entries=queued_entries,
                 test_query_count=len(arc_test_set.queries[key]["test"]))
            try:
                check_deadline(train_deadline, "train_prepare")
                torch.cuda.reset_peak_memory_stats()
                set_peft_model_state_dict(model, default_weights.copy(), adapter_name="default")
                model = FastLanguageModel.for_training(model)
                puzzle_ds = arc_test_set.change_keys([key])
                train_ds = puzzle_ds.augment(n=train_aug_n, shfl_keys=True, seed=train_data_seed)
                train_ds = train_ds.cut_to_len(formatter=formatter, name="text", max_len=max_seq_length)
                train_records = train_ds.as_list(formatter)
                check_deadline(train_deadline, "trainer_create")
                callback = TrainingDeadlineCallback(train_deadline)
                with io.StringIO() as buf, redirect_stdout(buf), redirect_stderr(buf):
                    trainer = UnslothFixedTrainer(
                        model=model,
                        tokenizer=tokenizer,
                        data_collator=collator,
                        train_dataset=Dataset.from_list(train_records),
                        dataset_text_field="text",
                        max_seq_length=max_seq_length,
                        args=UnslothTrainingArguments(**train_args),
                        callbacks=[callback],
                    )
                    try:
                        check_deadline(train_deadline, "train_start")
                        stats = trainer.train()
                        trained_steps = int(trainer.state.global_step)
                        model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)
                    finally:
                        del trainer
                emit("train_done", trained_steps=trained_steps, train_examples=len(train_records),
                     elapsed_seconds=time.time() - start_time,
                     stopped_for_deadline=callback.stopped_for_deadline,
                     train_deadline_epoch=train_deadline,
                     peak_allocated_mb=torch.cuda.max_memory_allocated() // 1024**2,
                     metrics=getattr(stats, "metrics", {}))
                if trained_steps < 1:
                    # Pilot and hidden runs both require actual task adaptation.
                    raise DeadlineReached("no_training_updates")
                check_deadline(puzzle_deadline, "inference_start")
                model = FastLanguageModel.for_inference(model)
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                puzzle_ds_multi = puzzle_ds.split_multi_replies()
                eval_ds = puzzle_ds_multi.augment(n=eval_aug_n, seed=eval_seed)
                eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input",
                                            max_len=max_seq_length-max_new_tokens)
                test_id_to_subkeys = defaultdict(list)
                for subkey in sorted(eval_ds.keys):
                    test_id_to_subkeys[subkey.split(".")[0]].append(subkey)
                batches = []
                for subkeys in test_id_to_subkeys.values():
                    batches.extend(build_decode_batches(subkeys))
                known_scores = {}
                emit("decode_start", batch_count=len(batches), subkey_count=len(eval_ds.keys),
                     eval_seed=eval_seed, task_deadline_epoch=puzzle_deadline)
                with torch.inference_mode():
                    for batch_index, subkeys in enumerate(batches):
                        check_deadline(puzzle_deadline, "decode_batch")
                        remaining = puzzle_deadline - time.time()
                        # Keep part of this batch's share for complete augmented NLL.
                        batch_share = remaining / max(1, len(batches) - batch_index)
                        reserve = min(scoring_reserve, batch_share * 0.25)
                        decode_deadline = min(puzzle_deadline, time.time() + max(0.0, batch_share - reserve))
                        tokens = [tokenizer.encode(eval_ds.get(subkey, formatter)["input"]) for subkey in subkeys]
                        dfs_result = inference_turbo_dfs(
                            model=model, prefix_tokens=tokens, max_new_tokens=max_new_tokens,
                            max_score=-np.log(min_token_prob), end_time=decode_deadline,
                            dfs_time_limit=dfs_time_limit,
                        )
                        emit("decode_batch_done", batch_index=batch_index,
                             subkeys=subkeys, complete_beams=sum(len(beams) for _, beams in dfs_result),
                             elapsed_seconds=time.time() - start_time)
                        for subkey_id, scored_beams in dfs_result:
                            subkey = subkeys[subkey_id]
                            bk = subkey.split(".")[0]
                            decoded_result = []
                            for beam_score, candidate_tokens in scored_beams:
                                check_deadline(puzzle_deadline, "candidate_rescore")
                                if max_scored_beams > 0 and len(decoded_result) >= max_scored_beams:
                                    break
                                array = formatter.convert_tokens_to_array(candidate_tokens)
                                if array is None:
                                    continue
                                solution = puzzle_ds_multi.invert_mod(array, subkey, inv_perm=True)
                                grid_id = (bk, tuple(map(tuple, solution)))
                                if grid_id in known_scores:
                                    augmented_scores, expected_score_count = known_scores[grid_id]
                                else:
                                    aug_dataset = ArcDataset(
                                        keys=[bk], queries={bk: puzzle_ds_multi.queries[bk]},
                                        replies={bk: [solution.tolist()]},
                                    )
                                    aug_dataset = aug_dataset.augment(seed=stable_seed_from_key(f"{bk}:{score_seed_offset}"))
                                    aug_dataset = aug_dataset.cut_to_len(
                                        formatter=formatter, name="input", max_len=max_seq_length-max_new_tokens)
                                    aug_samples = aug_dataset.as_list(formatter)
                                    if not aug_samples:
                                        emit("candidate_rejected", subkey=subkey, reason="empty_score_augmentation")
                                        continue
                                    aug_queries = [sample["input"] for sample in aug_samples]
                                    aug_answers = [sample["reply"] for sample in aug_samples]
                                    split = max(1, len(aug_queries) // 2)
                                    augmented_scores = []
                                    for offset in range(0, len(aug_queries), split):
                                        augmented_scores.extend(calc_scores(
                                            aug_queries[offset:offset+split], aug_answers[offset:offset+split],
                                            tokenizer, model, puzzle_deadline))
                                    if len(augmented_scores) != len(aug_samples) or not np.isfinite(augmented_scores).all():
                                        emit("candidate_rejected", subkey=subkey, reason="invalid_augmented_scores")
                                        continue
                                    expected_score_count = len(aug_samples)
                                    known_scores[grid_id] = (augmented_scores, expected_score_count)
                                decoded_result.append({
                                    "beam_score": float(beam_score), "score_aug": augmented_scores,
                                    "solution": solution, "_round": 0, "_eval_seed": eval_seed,
                                    "score_source": "augmentation_nll", "score_aug_count": len(augmented_scores),
                                    "score_aug_expected_count": expected_score_count,
                                    "candidate_source": "base_ttt", "trained_steps": trained_steps,
                                })
                                tag_part = f".{run_tag}" if run_tag else ""
                                output_path = os.path.join(dir_outputs, f"{subkey}{tag_part}.run0")
                                atomic_save_candidates(output_path, decoded_result)
                                saved_candidates += 1
                                saved_subkeys.add(subkey)
                                emit("candidate_saved", subkey=subkey, query_key=bk, path=output_path,
                                     candidates_in_file=len(decoded_result), score_source="augmentation_nll",
                                     score_aug_count=len(augmented_scores),
                                     score_aug_expected_count=expected_score_count, trained_steps=trained_steps)
            except DeadlineReached as exc:
                done_status = "deadline"
                emit("task_deadline", stage=str(exc), trained_steps=trained_steps,
                     task_deadline_epoch=puzzle_deadline, train_deadline_epoch=train_deadline)
                if trained_steps == 0:
                    emit("train_done", trained_steps=0, status="not_completed",
                         elapsed_seconds=time.time() - start_time, stopped_for_deadline=True)
            except Exception as exc:
                done_status = "error"
                emit("task_error", error_type=type(exc).__name__, error=str(exc))
                raise
            except BaseException:
                done_status = "interrupted"
                raise
            finally:
                emit("task_done", status=done_status, trained_steps=trained_steps,
                     elapsed_seconds=time.time() - start_time, saved_candidates=saved_candidates,
                     saved_subkeys=len(saved_subkeys), task_deadline_epoch=puzzle_deadline)
                active_key = None
                gc.collect()
                torch.cuda.empty_cache()
        emit("worker_done", status="deadline" if time.time() >= end_time else "queue_complete")
    except Exception as exc:
        emit("worker_error", error_type=type(exc).__name__, error=str(exc))
        raise
