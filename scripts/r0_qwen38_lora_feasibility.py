"""R0：Qwen3.8-27B LoRA 训练可行性探针（草稿，未运行）。

用途：对应 docs/plan/006-qwen38-l4-budget.md 的 R0，由用户在 Kaggle L4x4
离线 Notebook 上手动运行，回答“训练能否实际运行”，不产生分数结论。

运行环境：Kaggle GPU Notebook（建议 L4x4），关闭联网，挂载：
  - 比赛数据：arc-prize-2026-arc-agi-2
  - 模型：ugvfpdcuwfnh/qwen3-8-27b / Transformers / transformers-bf16 / v1
    （上游 revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0，约 55.56GB，18 shards）

手动步骤：
  1. 新建 Kaggle Notebook，关联比赛，挂载上述数据与模型，选 L4x4，加速器开。
  2. 把本文件内容粘贴到一个 cell 或作为 Notebook 附件上传后运行：
     python scripts/r0_qwen38_lora_feasibility.py --model-dir /kaggle/input/<mount> --data-dir /kaggle/input/<comp>
  3. 记录：每卡 nvidia-smi 峰值、tokenizer 后 episode 长度、单步耗时（如跑 --probe-forward）。
  4. 回填到 docs/progress/006-qwen38-l4-budget.md，不声称提分。

注意：本文件只做只读探针 + 配置打印，不启动全量 LoRA 训练。
实际 LoRA 训练请按 NVIDIA NeMo 官方 Qwen3.8-27B 配方（FSDP2 + activation
checkpointing + rank8/alpha32 起点）在 8xH100 之外另行实测 4xL4 显存/吞吐。
"""

import argparse
import glob
import json
import os
import sys


def find_file(data_dir, names):
    for name in names:
        hits = glob.glob(os.path.join(data_dir, "**", name), recursive=True)
        if hits:
            return hits[0]
    return None


def grid_to_text(grid):
    return "\n".join(" ".join(str(c) for c in row) for row in grid)


def build_episode(task):
    """同一任务的 demonstrations + query 组成一个 episode；target 仅用于训练 loss。"""
    lines = []
    for i, pair in enumerate(task.get("train", [])):
        lines.append(f"[demo {i}] input:\n{grid_to_text(pair['input'])}")
        lines.append(f"[demo {i}] output:\n{grid_to_text(pair['output'])}")
    tests = task.get("test", [])
    if not tests:
        raise ValueError("task has no test items")
    lines.append(f"[query] input:\n{grid_to_text(tests[0]['input'])}")
    prompt = "\n".join(lines) + "\n[query] output:\n"
    return prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--data-dir", default="/kaggle/input")
    ap.add_argument("--probe-forward", action="store_true",
                    help="额外做一次单样本 forward+loss（需 torch+transformers，本机默认不跑）")
    ap.add_argument("--max-tasks-scan", type=int, default=50)
    args = ap.parse_args()

    print("== R0 env ==")
    print("python:", sys.version.split()[0])
    try:
        import torch
        print("torch:", torch.__version__, "cuda_available:", torch.cuda.is_available(),
              "device_count:", torch.cuda.device_count() if torch.cuda.is_available() else 0)
        if torch.cuda.is_available():
            for i in range(torch.cuda.device_count()):
                print(f"  gpu{i}:", torch.cuda.get_device_name(i),
                      f"{torch.cuda.get_device_properties(i).total_memory / 1e9:.1f}GB")
    except Exception as e:  # 本地 M5 无 CUDA 时只记录，不中断
        print("torch probe skipped:", e)

    model_dir = args.model_dir
    if model_dir is None:
        cands = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
        model_dir = os.path.dirname(cands[0]) if cands else None
    print("model_dir:", model_dir)
    if model_dir and os.path.exists(os.path.join(model_dir, "config.json")):
        with open(os.path.join(model_dir, "config.json")) as f:
            cfg = json.load(f)
        print("model_type:", cfg.get("model_type"), "layers:", cfg.get("num_hidden_layers"),
              "heads:", cfg.get("num_attention_heads"), "kv_heads:", cfg.get("num_key_value_heads"),
              "vocab:", cfg.get("vocab_size"))

    ch_path = find_file(args.data_dir, ["arc-agi_training_challenges.json"])
    print("training_challenges:", ch_path)
    if not ch_path:
        print("WARN: 未找到比赛训练数据，请检查 Notebook 的数据挂载。")
        return 2
    with open(ch_path) as f:
        challenges = json.load(f)
    task_ids = sorted(challenges.keys())
    print(f"tasks: {len(task_ids)} (scan first {args.max_tasks_scan})")

    # 用第一个训练任务组一个 episode，统计文本长度与 tokenizer 长度
    tid = task_ids[0]
    prompt = build_episode(challenges[tid])
    print(f"sample task: {tid}, prompt_chars: {len(prompt)}")
    try:
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        ids = tok(prompt, return_tensors="pt")["input_ids"][0]
        print(f"sample token_len: {len(ids)} (vocab {tok.vocab_size})")
        print("首轮长度建议：先用可完整放入 2048/4096 的 episode；超长样本单独分组扩长度，不截断答案。")
    except Exception as e:
        print("tokenizer probe skipped:", e)

    print("""
== R0 训练配置起点（待实测，非结论）==
- 精度/并行：BF16 冻结底座 + LoRA + 受支持的 FSDP2 分片，4xL4
- Adapter 起点：rank8 / alpha32（官方配方）；必要时 rank16/32；不照搬 NVARC rank256
- Batch：每卡 batch1 + 梯度累积；先短/中/长各几十步测 step time
- 激活：activation checkpointing；记录激活/logits/临时 all-gather 峰值
- Loss：核对 chunked/fused linear CE 支持，避免一次物化 [seq=4096, vocab=248320] logits（单序列 BF16 约 2GB）
- 依据：NVIDIA NeMo Qwen3.8-27B 配方（8xH100 100 步，非 4xL4 实测）；Axolotl 同架构 QLoRA/FSDP2 路径
""")
    if args.probe_forward:
        print("probe-forward: 请在 L4 Notebook 上手动确认后再跑完整训练，本草稿默认不执行。")
    print("R0 探针结束：请把 nvidia-smi 峰值、step time、tokens/episode 回填到 progress/006。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
