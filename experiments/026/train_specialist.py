#!/usr/bin/env python3
"""Bounded, single-GPU continuation of the V1 16-token NVARC 4B model.

The original arc_loader.py and arc_solver.py must be materialized outside this
repository and supplied with --reference-dir. Only their formatter and fixed
Unsloth trainer are reused. --inspect-only needs a local tokenizer, not torch or
a GPU. The merged_model directory, rather than this run's adapter, is the input
to a separate, freshly started original rank-256 TTT worker.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict, deque
import hashlib
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time
import traceback
from types import SimpleNamespace


BASE_CONFIG_SHA256 = "932e9430395ba3eea4087cba1769241bc7d0dc8d574846862d6983659456a2d0"
TOKEN_FILES = (
    "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
    "added_tokens.json", "vocab.json", "merges.txt", "chat_template.jinja",
)
TARGET_MODULES = (
    "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
    "down_proj", "embed_tokens", "lm_head",
)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def event(output, name, **fields):
    row = {"event": name, "time_unix": time.time(), **fields}
    with (output / "events.jsonl").open("a") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(row, ensure_ascii=False), flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", "--train-jsonl", type=Path, required=True)
    parser.add_argument("--data-manifest", type=Path)
    parser.add_argument("--expected-data-sha256")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--reference-dir", "--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--merged-model-dir", type=Path,
                        help="Full plain checkpoint destination; defaults to output-dir/merged_model.")
    parser.add_argument("--inspect-only", action="store_true")
    parser.add_argument("--seed", type=int, default=26026)
    parser.add_argument("--rank", type=int, choices=(32, 64), default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=8192)
    parser.add_argument("--optimizer-steps", "--max-steps", type=int, default=256)
    parser.add_argument("--pilot-steps", type=int, default=32)
    parser.add_argument("--max-seconds", type=float, default=3600)
    parser.add_argument("--train-seconds", type=float, default=2700,
                        help="Latest optimizer-update time measured from this script's start.")
    parser.add_argument("--save-reserve-seconds", type=float, default=600)
    parser.add_argument("--stage-deadline-unix", type=float,
                        help="Optional outer SFT-stage deadline, including prior preparation.")
    parser.add_argument("--expected-base-config-sha256", default=BASE_CONFIG_SHA256)
    parser.add_argument("--no-augmentation", action="store_true")
    args = parser.parse_args()
    if not (0 < args.learning_rate <= 2e-5):
        parser.error("This bounded continuation requires 0 < learning-rate <= 2e-5")
    if not (1 <= args.gradient_accumulation <= 8):
        parser.error("gradient-accumulation must be between 1 and 8")
    if not (512 <= args.max_seq_length <= 8192):
        parser.error("max-seq-length must be between 512 and 8192")
    if not (20 <= args.pilot_steps <= 50):
        parser.error("pilot-steps must be between 20 and 50")
    if not (args.pilot_steps <= args.optimizer_steps <= 512):
        parser.error("optimizer-steps must be between pilot-steps and 512")
    if not (0 < args.save_reserve_seconds < args.max_seconds <= 3600):
        parser.error("Require 0 < save-reserve-seconds < max-seconds <= 3600")
    if not (0 < args.train_seconds < args.max_seconds):
        parser.error("Require 0 < train-seconds < max-seconds")
    return args


def grid(value, field):
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        raise ValueError(f"{field}: expected 1..30 grid rows")
    width = len(value[0]) if isinstance(value[0], list) else 0
    if not 1 <= width <= 30:
        raise ValueError(f"{field}: expected 1..30 grid columns")
    for row in value:
        if not isinstance(row, list) or len(row) != width:
            raise ValueError(f"{field}: nonrectangular grid")
        if any(type(cell) is not int or not 0 <= cell <= 9 for cell in row):
            raise ValueError(f"{field}: cells must be integer colors 0..9")
    return value


def augment_grid(value, colors, rotation, transpose):
    result = [[colors[cell] for cell in row] for row in value]
    if transpose:
        result = [list(row) for row in zip(*result)]
    for _ in range(rotation):
        result = [list(row) for row in zip(*result[::-1])]
    return result


def read_episodes(path, seed, augment):
    episodes, excluded, seen = [], Counter(), set()
    with path.open() as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            item = json.loads(line)
            split = item.get("split")
            if split != "train":
                excluded[str(split)] += 1
                continue
            episode_id = str(item["episode_id"])
            if episode_id in seen:
                raise ValueError(f"Duplicate training episode_id: {episode_id}")
            seen.add(episode_id)
            family = str(item["family_id"])
            demonstrations = item["train"]
            if not isinstance(demonstrations, list) or not demonstrations:
                raise ValueError(f"{episode_id}: at least one demonstration is required")
            pairs = [{"input": grid(pair["input"], f"{episode_id}.train.input"),
                      "output": grid(pair["output"], f"{episode_id}.train.output")}
                     for pair in demonstrations]
            query_input = grid(item["query_input"], f"{episode_id}.query_input")
            query_output = grid(item["query_output"], f"{episode_id}.query_output")
            descriptor = {"colors": list(range(10)), "rotation": 0, "transpose": False}
            if augment:
                # Stable across Python processes and independent of PYTHONHASHSEED.
                local_seed = int.from_bytes(hashlib.sha256(
                    f"{seed}:{episode_id}".encode()).digest()[:8], "big")
                rng = random.Random(local_seed)
                rng.shuffle(descriptor["colors"])
                descriptor.update(rotation=rng.randrange(4), transpose=bool(rng.randrange(2)))
                def transform(value):
                    return augment_grid(value, **descriptor)
                pairs = [{key: transform(value) for key, value in pair.items()} for pair in pairs]
                query_input, query_output = transform(query_input), transform(query_output)
            metadata = {
                "episode_id": episode_id, "family_id": family,
                "source": item.get("source", "unspecified"),
                "source_revision": item.get("source_revision"),
                "ancestor_task_ids": item.get("ancestor_task_ids", []),
                "base_id": item.get("base_id", episode_id),
                "generator_seed": item.get("generator_seed"),
                "query_hash": item.get("query_hash"),
                "line_number": line_number, "augmentation": descriptor,
            }
            episodes.append((pairs, query_input, query_output, metadata))
    if not episodes:
        raise ValueError("No split=='train' episodes; no other split is eligible for training")
    return episodes, dict(excluded)


def tokenizer_contract(tokenizer):
    expected = {**{str(i): [i] for i in range(10)}, "\n": [10],
                "user": [11], "assistant": [12], "<|endoftext|>": [13],
                "<|im_start|>": [14], "<|im_end|>": [15],
                "<|im_start|>user\n": [14, 11, 10],
                "<|im_start|>assistant\n": [14, 12, 10]}
    # The real V1 tokenizer has 13 base tokens plus three added special tokens.
    if len(tokenizer) != 16 or set(tokenizer.get_vocab().values()) != set(range(16)):
        raise ValueError(f"Expected exact compact 16-token tokenizer, got {len(tokenizer)}")
    for text, ids in expected.items():
        actual = tokenizer.encode(text, add_special_tokens=False)
        if actual != ids:
            raise ValueError(f"Tokenizer contract mismatch for {text!r}: {actual} != {ids}")
    if tokenizer.eos_token_id != 15 or tokenizer.pad_token_id != 13:
        raise ValueError("Expected EOS=15 and PAD=13")
    return {"vocab_size_including_added": 16, "base_vocab_size": tokenizer.vocab_size,
            "eos_token_id": 15, "pad_token_id": 13,
            "user_header": [14, 11, 10], "assistant_header": [14, 12, 10]}


def tokenize_episode(formatter, tokenizer, pairs, query_input, query_output, metadata, max_length):
    pairs = list(pairs)
    removed = 0
    while True:
        prompt = formatter.fmt_train(pairs) + formatter.fmt_query([{"input": query_input}])
        reply = formatter.fmt_reply([query_output])
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        reply_ids = tokenizer.encode(reply, add_special_tokens=False)
        full_ids = tokenizer.encode(prompt + reply, add_special_tokens=False)
        # Prefix matching is necessary: tokenizing a suffix in isolation is not
        # generally equivalent to slicing a tokenized whole chat transcript.
        if full_ids != prompt_ids + reply_ids:
            raise ValueError(f"{metadata['episode_id']}: prompt/full token boundary mismatch")
        if prompt_ids[-3:] != [14, 12, 10]:
            raise ValueError("Query prompt does not end at the assistant grid boundary")
        if not reply_ids or reply_ids[-1] != 15 or reply_ids.count(15) != 1:
            raise ValueError("Query target must end in exactly one EOS")
        if any(token not in range(11) for token in reply_ids[:-1]):
            raise ValueError("Query target contains a nongrid token")
        if any(not 0 <= token < 16 for token in full_ids):
            raise ValueError("A token is outside the compact vocabulary")
        if len(full_ids) <= max_length:
            break
        # Never truncate the query target or a partial demonstration. Prefer at
        # least two demonstrations; an originally single-demo task stays valid.
        if len(pairs) <= min(2, metadata["original_demo_count"]):
            return None
        pairs.pop(0)
        removed += 1
    labels = [-100] * len(prompt_ids) + reply_ids
    if any(label != -100 for label in labels[:len(prompt_ids)]):
        raise AssertionError("Prompt labels must all be ignored")
    bucket = next(limit for limit in (1024, 2048, 4096, 8192) if len(full_ids) <= limit)
    meta = {**metadata, "prompt_tokens": len(prompt_ids), "target_tokens": len(reply_ids),
            "total_tokens": len(full_ids), "length_bucket": bucket,
            "demonstrations_used": len(pairs), "demonstrations_removed": removed}
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids),
            "labels": labels, "text": prompt + reply, "metadata": meta}


def prepare_rows(episodes, formatter, tokenizer, max_length, seed):
    groups, overflow = defaultdict(list), []
    for pairs, query_input, query_output, metadata in episodes:
        metadata = {**metadata, "original_demo_count": len(pairs)}
        row = tokenize_episode(formatter, tokenizer, pairs, query_input, query_output,
                               metadata, max_length)
        if row is None:
            overflow.append(metadata["episode_id"])
        else:
            groups[(str(metadata["source"]), metadata["family_id"], row["metadata"]["length_bucket"])].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for group in groups.values():
        rng.shuffle(group)
    # Alternate sources, and within each source rotate its family/length strata.
    # A deadline prefix therefore retains both synthetic data and replay even
    # though replay has many more distinct families. No episode is repeated.
    by_source = defaultdict(deque)
    for key in keys:
        by_source[key[0]].append(deque(groups[key]))
    source_keys = sorted(by_source)
    rng.shuffle(source_keys)
    pending = deque(by_source[source] for source in source_keys)
    rows = []
    while pending:
        strata = pending.popleft()
        values = strata.popleft()
        rows.append(values.popleft())
        if values:
            strata.append(values)
        if strata:
            pending.append(strata)
    if not rows:
        raise ValueError("Every training episode exceeds the context budget")
    for index, row in enumerate(rows):
        row["episode_index"] = index
    return rows, overflow


def summarize_rows(rows):
    return {
        "episodes": len(rows),
        "families": dict(sorted(Counter(row["metadata"]["family_id"] for row in rows).items())),
        "sources": dict(sorted(Counter(str(row["metadata"]["source"]) for row in rows).items())),
        "length_buckets": dict(sorted(Counter(row["metadata"]["length_bucket"] for row in rows).items())),
        "input_tokens": sum(row["metadata"]["total_tokens"] for row in rows),
        "supervised_tokens": sum(row["metadata"]["target_tokens"] for row in rows),
        "max_length": max(row["metadata"]["total_tokens"] for row in rows),
        "demonstrations_removed": sum(row["metadata"]["demonstrations_removed"] for row in rows),
    }


def tokenizer_files(source, destination=None):
    records = {}
    for name in TOKEN_FILES:
        path = source / name
        if not path.is_file():
            continue
        digest = sha256(path)
        if destination is not None:
            shutil.copy2(path, destination / name)
            if sha256(destination / name) != digest:
                raise ValueError(f"Tokenizer copy changed bytes: {name}")
        records[name] = {"sha256": digest, "bytes": path.stat().st_size}
    if not {"tokenizer.json", "tokenizer_config.json"}.issubset(records):
        raise ValueError("The source checkpoint must contain its original tokenizer files")
    return records


def load_original_trainer(path, state):
    # Importing the whole materialized solver could execute notebook-specific
    # observer globals. Compile only the original class from its external AST;
    # no third-party solver source is copied into this repository.
    from unsloth import FastLanguageModel, UnslothTrainingArguments, UnslothTrainer
    source = path.read_text()
    tree = ast.parse(source, filename=str(path))
    definitions = [node for node in tree.body
                   if isinstance(node, ast.ClassDef) and node.name == "UnslothFixedTrainer"]
    if len(definitions) != 1:
        raise ValueError("Expected exactly one original UnslothFixedTrainer definition")
    definition = definitions[0]
    state["external_fixed_trainer_sha256"] = hashlib.sha256(
        ast.get_source_segment(source, definition).encode()).hexdigest()
    namespace = {"UnslothTrainer": UnslothTrainer}
    exec(compile(ast.Module(body=[definition], type_ignores=[]), str(path), "exec"), namespace)
    return SimpleNamespace(FastLanguageModel=FastLanguageModel,
                           UnslothTrainingArguments=UnslothTrainingArguments,
                           UnslothFixedTrainer=namespace["UnslothFixedTrainer"])


def train(args, state, rows, tokenizer, original, started, hard_deadline):
    import torch
    from datasets import Dataset
    from transformers import TrainerCallback

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("This script requires exactly one visible CUDA GPU; DDP is excluded")
    for package, expected in (("unsloth", "2025.9.7"), ("transformers", "4.55.4")):
        actual = importlib.metadata.version(package)
        if actual != expected:
            raise RuntimeError(f"Use the proven 025 image: {package}={actual}, expected {expected}")
    state["runtime"] = {name: importlib.metadata.version(name) for name in
                        ("unsloth", "transformers", "trl", "peft", "torch", "datasets")}
    state["hardware"] = {"visible_gpus": 1, "name": torch.cuda.get_device_name(0),
                         "vram_bytes": torch.cuda.get_device_properties(0).total_memory}
    steps = min(args.optimizer_steps, len(rows) // args.gradient_accumulation)
    if steps < 20:
        raise ValueError("At least 20 complete optimizer steps are required; supply >=80 train episodes at accum4")
    pilot_step = min(args.pilot_steps, steps)
    # Leave an incomplete accumulation group unused. At most one pass is made.
    rows = rows[:steps * args.gradient_accumulation]
    state["planned_optimizer_steps"] = steps
    state["planned_pilot_step"] = pilot_step
    state["scheduled_data"] = summarize_rows(rows)
    train_deadline = min(state["started_unix"] + args.train_seconds,
                         hard_deadline - args.save_reserve_seconds)
    if time.time() >= train_deadline:
        raise RuntimeError("No training time remains after the reserved save/merge budget")

    class QueryOnlyCollator:
        def __call__(self, examples):
            width = max(len(row["input_ids"]) for row in examples)
            return {
                "input_ids": torch.tensor([row["input_ids"] + [13] * (width - len(row["input_ids"]))
                                           for row in examples], dtype=torch.long),
                "attention_mask": torch.tensor([row["attention_mask"] + [0] * (width - len(row["input_ids"]))
                                                for row in examples], dtype=torch.long),
                "labels": torch.tensor([row["labels"] + [-100] * (width - len(row["input_ids"]))
                                        for row in examples], dtype=torch.long),
                "episode_indices": torch.tensor([row["episode_index"] for row in examples]),
            }

    consumption = {"microbatches": 0, "episodes": 0, "input_tokens": 0,
                   "supervised_tokens": 0, "family_episodes": Counter(),
                   "source_episodes": Counter(), "length_bucket_episodes": Counter()}
    consumption_handle = (args.output_dir / "consumption.jsonl").open("a", buffering=1)

    class SpecialistTrainer(original.UnslothFixedTrainer):
        def _prepare_dataset(self, dataset, *unused_args, **unused_kwargs):
            if not {"input_ids", "attention_mask", "labels", "episode_index"}.issubset(dataset.column_names):
                raise ValueError("Only the already audited, pretokenized dataset is accepted")
            return dataset

        def _get_train_sampler(self, train_dataset=None):
            return torch.utils.data.SequentialSampler(self.train_dataset if train_dataset is None else train_dataset)

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            indices = inputs.pop("episode_indices").detach().cpu().tolist()
            result = super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)
            loss = result[0] if return_outputs else result
            if not torch.isfinite(loss.detach()).all().item():
                raise FloatingPointError("Nonfinite pilot/continuation loss; no merged challenger is accepted")
            for index in indices:
                meta = rows[index]["metadata"]
                consumption["episodes"] += 1
                consumption["input_tokens"] += meta["total_tokens"]
                consumption["supervised_tokens"] += meta["target_tokens"]
                consumption["family_episodes"][meta["family_id"]] += 1
                consumption["source_episodes"][str(meta["source"])] += 1
                consumption["length_bucket_episodes"][meta["length_bucket"]] += 1
                consumption_handle.write(json.dumps({"optimizer_step_before": self.state.global_step,
                                                     **meta}, ensure_ascii=False) + "\n")
            consumption["microbatches"] += 1
            state["consumption"] = consumption
            return result

    class BudgetCallback(TrainerCallback):
        def on_step_end(self, training_args, trainer_state, control, **kwargs):
            state["actual_optimizer_steps"] = trainer_state.global_step
            elapsed = time.monotonic() - started
            event(args.output_dir, "optimizer_step", step=trainer_state.global_step,
                  elapsed_seconds=elapsed, input_tokens=consumption["input_tokens"],
                  supervised_tokens=consumption["supervised_tokens"],
                  peak_cuda_allocated_bytes=torch.cuda.max_memory_allocated())
            if trainer_state.global_step == pilot_step:
                # Saving does not reset the optimizer, scheduler, sampler, or adapter.
                trainer.model.save_pretrained(args.output_dir / "pilot_adapter", safe_serialization=True)
                state["pilot"] = {"optimizer_step": trainer_state.global_step,
                                  "elapsed_seconds": elapsed, "finite_loss": True,
                                  "adapter_path": str(args.output_dir / "pilot_adapter")}
                write_json(args.output_dir / "pilot.json", {**state["pilot"], "consumption": consumption})
                event(args.output_dir, "pilot_saved", **state["pilot"])
            if time.time() >= train_deadline:
                state["stop_reason"] = "training_deadline_save_reserve"
                control.should_training_stop = True
            write_json(args.output_dir / "manifest.json", state)
            return control

        def on_log(self, training_args, trainer_state, control, logs=None, **kwargs):
            if logs:
                event(args.output_dir, "trainer_log", step=trainer_state.global_step, metrics=logs)

    peft_parameters = dict(r=args.rank, target_modules=list(TARGET_MODULES), lora_alpha=32,
                           lora_dropout=0, bias="none", use_gradient_checkpointing="unsloth",
                           random_state=args.seed, use_rslora=True, loftq_config=None)
    state["offline_adapter"] = {**peft_parameters, "effective_scale": 32 / math.sqrt(args.rank)}
    model, loaded_tokenizer = original.FastLanguageModel.from_pretrained(
        model_name=str(args.model_path), full_finetuning=False, load_in_4bit=False,
        local_files_only=True, use_gradient_checkpointing=False,
        max_seq_length=args.max_seq_length, dtype=torch.bfloat16)
    tokenizer_contract(loaded_tokenizer)
    if loaded_tokenizer.get_vocab() != tokenizer.get_vocab():
        raise ValueError("Unsloth model loading changed tokenizer vocabulary")
    if model.config.vocab_size != 16:
        raise ValueError("Loaded model does not have the compact 16-token vocabulary")
    model = original.FastLanguageModel.get_peft_model(model, **peft_parameters)
    for parameter in model.parameters():
        if parameter.dtype == torch.float32:
            parameter.data = parameter.data.to(torch.bfloat16)
    state["trainable_parameters"] = sum(p.numel() for p in model.parameters() if p.requires_grad)
    model = original.FastLanguageModel.for_training(model)
    dataset = Dataset.from_list([{key: value for key, value in row.items() if key != "metadata"} for row in rows])
    training_args = original.UnslothTrainingArguments(
        output_dir=str(args.output_dir / "trainer"), per_device_train_batch_size=1,
        gradient_accumulation_steps=args.gradient_accumulation,
        num_train_epochs=1, max_steps=steps, learning_rate=args.learning_rate,
        warmup_steps=0, warmup_ratio=0.1, lr_scheduler_type="cosine",
        optim="adamw_torch", weight_decay=0.0, max_grad_norm=1.0,
        bf16=True, fp16=False, seed=args.seed, data_seed=args.seed,
        report_to="none", save_strategy="no", eval_strategy="no",
        logging_strategy="steps", logging_steps=1, logging_first_step=True,
        fsdp="", dataloader_num_workers=0, dataloader_drop_last=False,
        remove_unused_columns=False, gradient_checkpointing=True,
    )
    trainer = SpecialistTrainer(
        model=model, tokenizer=loaded_tokenizer, data_collator=QueryOnlyCollator(),
        train_dataset=dataset, dataset_text_field="text", max_seq_length=args.max_seq_length,
        packing=False, args=training_args, callbacks=[BudgetCallback()],
    )
    if trainer.accelerator.num_processes != 1:
        raise RuntimeError("Original loss scaling is not used with DDP in this experiment")
    # The original fixed loss does not pass num_items_in_batch to the model.
    # HF must therefore scale its per-microbatch mean by gradient accumulation.
    trainer.model_accepts_loss_kwargs = False
    if len(trainer.train_dataset) != len(rows):
        raise ValueError("Trainer preprocessing changed the episode count")
    for index, sample in enumerate(trainer.train_dataset):
        if sample.get("labels") != rows[index]["labels"] or sample.get("episode_index") != index:
            raise ValueError("Trainer preprocessing changed query-only labels or episode identity")
    if time.time() >= train_deadline:
        raise RuntimeError("Model loading exhausted the allowed pre-merge training window")
    state["startup_and_load_seconds"] = time.monotonic() - started
    write_json(args.output_dir / "manifest.json", state)
    optimizer_started = time.monotonic()
    try:
        stats = trainer.train()
    finally:
        consumption_handle.close()
    state["actual_optimizer_steps"] = int(stats.global_step)
    state["optimizer_phase_seconds"] = time.monotonic() - optimizer_started
    state["training_loss"] = float(stats.training_loss)
    if not math.isfinite(state["training_loss"]):
        raise FloatingPointError("Training returned a nonfinite loss")
    if state["actual_optimizer_steps"] < 20:
        raise RuntimeError("The budget ended before a 20-step pilot; no challenger is accepted")
    state.setdefault("stop_reason", "optimizer_step_cap_one_pass")
    state["training_elapsed_seconds"] = time.monotonic() - started
    model = trainer.accelerator.unwrap_model(trainer.model, keep_fp32_wrapper=False)
    model.save_pretrained(args.output_dir / "adapter", safe_serialization=True)
    original.FastLanguageModel.for_inference(model)
    probes = []
    actual_scales = set()
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or "default" not in module.lora_A:
            continue
        actual_scales.add(float(module.scaling["default"]))
        if not (name.endswith("q_proj") or name.endswith("up_proj") or name.endswith("lm_head")):
            continue
        base_layer = module.get_base_layer()
        a = module.lora_A["default"].weight.detach()
        b = module.lora_B["default"].weight.detach()
        probe_delta = (b[:64].float() @ a[:, :64].float()) * module.scaling["default"]
        probes.append({"name": name, "base_layer": base_layer,
                       "before": base_layer.weight[:64, :64].detach().cpu().clone(),
                       "adapter_delta_max_abs": float(probe_delta.abs().max().item())})
    state["offline_adapter"]["observed_module_scales"] = sorted(actual_scales)
    if not probes or not any(probe["adapter_delta_max_abs"] > 0 for probe in probes):
        raise ValueError("No nonzero offline LoRA update was found in the recorded weight probes")
    # A full merged plain checkpoint is required. Evaluation must reload this
    # directory in new processes before attaching the original rank-256 TTT.
    merged = model.merge_and_unload(safe_merge=True)
    merge_proof = []
    for probe in probes:
        after = probe["base_layer"].weight[:64, :64].detach().cpu()
        changes = after != probe["before"]
        merge_proof.append({"module": probe["name"], "probe_shape": list(after.shape),
                            "adapter_delta_max_abs": probe["adapter_delta_max_abs"],
                            "changed_bf16_values": int(changes.sum().item()),
                            "merged_change_max_abs": float((after.float() - probe["before"].float()).abs().max().item())})
    if not any(probe["changed_bf16_values"] > 0 for probe in merge_proof):
        raise ValueError("Offline LoRA did not change any recorded BF16 base-weight probe after merging")
    state["offline_merge_proof"] = merge_proof
    if any("lora_" in name for name, _ in merged.named_parameters()):
        raise ValueError("LoRA parameters remain in the intended plain merged checkpoint")
    if any(parameter.dtype != torch.bfloat16 for parameter in merged.parameters() if parameter.is_floating_point()):
        raise ValueError("Merged floating parameters must all be BF16")
    merged.config.use_cache = True
    merged_dir = args.merged_model_dir
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(merged_dir, safe_serialization=True, max_shard_size="4GB")
    state["merged_tokenizer_files"] = tokenizer_files(args.model_path, merged_dir)
    from transformers import AutoTokenizer
    tokenizer_contract(AutoTokenizer.from_pretrained(merged_dir, local_files_only=True))
    merged_config = json.loads((merged_dir / "config.json").read_text())
    if merged_config["vocab_size"] != 16 or (merged_dir / "adapter_config.json").exists():
        raise ValueError("Saved challenger must be a full 16-token model, not a PEFT adapter")
    weight_paths = sorted(merged_dir.glob("*.safetensors"))
    weight_bytes = sum(path.stat().st_size for path in weight_paths)
    if weight_bytes < 6_000_000_000:
        raise ValueError(f"Expected a full 4B BF16 checkpoint, found only {weight_bytes} weight bytes")
    state["merged_model"] = {
        "path": str(merged_dir), "config_sha256": sha256(merged_dir / "config.json"),
        "weight_files": {path.name: path.stat().st_size for path in weight_paths},
        "weight_bytes": weight_bytes, "plain_model_without_lora": True,
        "tokenizer_bytes_match_base": True, "vocab_size": 16,
        "all_floating_parameters_bfloat16": True,
        "offline_update_nonzero_after_bf16_merge": True,
        "files": {path.name: {"bytes": path.stat().st_size,
                              **({"sha256": sha256(path)} if path.stat().st_size < 1_000_000 else {})}
                  for path in sorted(merged_dir.iterdir()) if path.is_file()},
    }
    state["peak_cuda_allocated_bytes"] = torch.cuda.max_memory_allocated()
    state["elapsed_seconds"] = time.monotonic() - started
    state["gpu_hours"] = state["elapsed_seconds"] / 3600
    if time.time() > hard_deadline:
        raise RuntimeError("SFT including save/merge exceeded its 3600-second outer budget; reject the run")
    state["status"] = "merged_checkpoint_ready_for_paired_evaluation"
    state["score_effect"] = "unmeasured; continuation loss is not an ARC score"


def main():
    started = time.monotonic()
    wall_started = time.time()
    args = arguments()
    args.data = args.data.resolve()
    args.model_path = args.model_path.resolve()
    args.reference_dir = args.reference_dir.resolve()
    args.output_dir = args.output_dir.resolve()
    args.merged_model_dir = (args.merged_model_dir or args.output_dir / "merged_model").resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if any(args.output_dir.iterdir()):
        raise ValueError("Output directory must be empty; never overwrite an earlier checkpoint")
    if args.merged_model_dir == args.output_dir:
        raise ValueError("Merged model requires its own directory")
    if args.merged_model_dir.exists() and any(args.merged_model_dir.iterdir()):
        raise ValueError("Merged model destination must be new or empty")
    hard_deadline = min(wall_started + args.max_seconds, args.stage_deadline_unix or float("inf"))
    state = {"status": "initializing", "started_unix": wall_started,
             "hard_deadline_unix": hard_deadline,
             "configuration": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
             "actual_optimizer_steps": 0, "score_effect": "unmeasured"}
    try:
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise RuntimeError("Do not launch this script under torchrun/DDP")
        # Set visibility outside the process. No hidden CUDA device selection or
        # network fetching is performed by this training program.
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"
        source_paths = {name: args.reference_dir / f"{name}.py" for name in ("arc_loader", "arc_solver")}
        if not source_paths["arc_loader"].is_file():
            raise FileNotFoundError("Materialize original arc_loader.py outside the repo and supply --reference-dir")
        state["external_reference_sources"] = {name: {"path": str(path), "sha256": sha256(path)}
                                                for name, path in source_paths.items() if path.is_file()}
        sys.path.insert(0, str(args.reference_dir))
        # Unsloth must patch transformers/torch before either is imported. The
        # CPU-only formatter audit imports only arc_loader, never arc_solver.
        original = None if args.inspect_only else load_original_trainer(source_paths["arc_solver"], state)
        loader = importlib.import_module("arc_loader")
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
        state["tokenizer_contract"] = tokenizer_contract(tokenizer)
        state["base_tokenizer_files"] = tokenizer_files(args.model_path)
        config_path = args.model_path / "config.json"
        if config_path.is_file():
            state["base_config_sha256"] = sha256(config_path)
        if not args.inspect_only and state.get("base_config_sha256") != args.expected_base_config_sha256:
            raise ValueError("Base config does not match the frozen sft139 BF16 V1 checkpoint")
        state["data_sha256"] = sha256(args.data)
        if args.expected_data_sha256 and state["data_sha256"] != args.expected_data_sha256:
            raise ValueError("Training JSONL SHA256 differs from the frozen data choice")
        if args.data_manifest:
            state["data_manifest"] = {"path": str(args.data_manifest.resolve()),
                                      "sha256": sha256(args.data_manifest),
                                      "contents": json.loads(args.data_manifest.read_text())}
        episodes, excluded = read_episodes(args.data, args.seed, not args.no_augmentation)
        rows, overflow = prepare_rows(episodes, loader.QwenFormatter(tokenizer), tokenizer,
                                      args.max_seq_length, args.seed)
        state["excluded_splits"] = excluded
        state["overflow_skipped_episode_ids"] = overflow
        state["eligible_data"] = summarize_rows(rows)
        state["schedule_policy"] = "source round robin; per-source family/length stratum round robin; one pass"
        # These examples retain complete IDs and labels so a manual CPU audit can
        # verify prompt -100, final-query grid labels, and the supervised EOS.
        examples = [{"metadata": row["metadata"], "input_ids": row["input_ids"],
                     "labels": row["labels"], "text": row["text"]} for row in rows[:3]]
        write_json(args.output_dir / "mask_examples.json", examples)
        with (args.output_dir / "episode_schedule.jsonl").open("w") as handle:
            for row in rows:
                handle.write(json.dumps({"episode_index": row["episode_index"], **row["metadata"]}, ensure_ascii=False) + "\n")
        if args.inspect_only:
            state.update(status="cpu_format_and_query_mask_verified", elapsed_seconds=time.monotonic() - started)
        else:
            train(args, state, rows, tokenizer, original, started, hard_deadline)
        write_json(args.output_dir / "manifest.json", state)
        if not args.inspect_only:
            write_json(args.output_dir / "merge_ready.json", {
                "status": state["status"], "merged_model_dir": str(args.merged_model_dir),
                "manifest_sha256": sha256(args.output_dir / "manifest.json"),
                "training_process_must_exit_before_ttt": True,
                "actual_optimizer_steps": state["actual_optimizer_steps"],
                "elapsed_seconds": state["elapsed_seconds"],
                "merged_model": state["merged_model"],
            })
        event(args.output_dir, "completed", status=state["status"],
              optimizer_steps=state["actual_optimizer_steps"], elapsed_seconds=state["elapsed_seconds"])
        return 0
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     elapsed_seconds=time.monotonic() - started, traceback=traceback.format_exc())
        write_json(args.output_dir / "manifest.json", state)
        event(args.output_dir, "failed", error=state["error"], elapsed_seconds=state["elapsed_seconds"])
        raise


if __name__ == "__main__":
    raise SystemExit(main())
