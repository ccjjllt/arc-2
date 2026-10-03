"""010 预算实验最小手动入口：共享会话规划 + 已加载模型评估包装。

只做 §8 接口的最小入口，不建训练平台/通用调度。CPU dry-run 不加载
模型、不 import torch/transformers/peft：

  python3 scripts/run_budgeted_grid_experiment.py --help
  python3 scripts/run_budgeted_grid_experiment.py --dry-run --stage S0,S1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

PROCESS_T0 = time.time()  # Notebook 全墙钟起点（import 起），GPU 侧照抄此字段。
SCHEDULER_HORIZON = 100  # §4/§8.2：总 horizon 固定 100，不随 stop_after_update 改写。
STOP_CHOICES = (25, 50, 100)
GPU_QUOTA_FACTOR = 2.0  # L4x4 两倍扣额：配额秒 ≈ 墙钟秒 × 2。

STAGES = (
    # (阶段, 墙钟上限h, 配额上限h, 最小可回答问题)
    ("S0", 2.5, 5.0, "G1相对底座新增/丢失；F1格式输出。P12配对优先"),
    ("S1", 4.5, 9.0, "同训练例/初始化/LR曲线下格式/解题/时间差异"),
    ("S2", 1.0, 2.0, "条件续到100是否仍有收益；否则保留共同50步"),
    ("S3", 2.0, 4.0, "第二候选比较与可选TTT(P6,0/8更新)"),
    ("S4", 1.5, 3.0, "冻结方案在C12独立确认"),
    ("RESERVE", 3.5, 7.0, "保存/恢复、账差及未承诺余量"),
)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="010共享会话最小入口：dry-run只打印阶段规划；"
        "评估包装只接收已加载model/tokenizer，不内部from_pretrained。"
    )
    p.add_argument("--dry-run", action="store_true", help="纯CPU阶段规划打印，不加载模型。")
    p.add_argument("--stage", default="S0,S1",
                   help="逗号分隔的阶段子集，如 S0,S1（默认 S0,S1）。")
    p.add_argument("--stop-after-update", type=int, default=25, choices=list(STOP_CHOICES),
                   help="阶段停止点，与scheduler总horizon=100分离。")
    p.add_argument("--scheduler-horizon", type=int, default=SCHEDULER_HORIZON,
                   help="必须保持100；传入他值直接拒绝。")
    p.add_argument("--arm", action="append", default=[],
                   help="可重复：base / g1 / f0_25 / f1_25 等臂名。")
    p.add_argument("--formatter", default="f0_space", help="f0_space 或 f1_compact。")
    p.add_argument("--queries", default="", help="显式query清单JSON路径。")
    p.add_argument("--p12-out", default="experiments/010/run_p12",
                   help="P12输出目录（与C12分开）。")
    p.add_argument("--c12-out", default="experiments/010/run_c12",
                   help="C12输出目录（与P12分开）。")
    p.add_argument("--overwrite", action="store_true", help="默认不覆盖；传入才允许覆盖。")
    p.add_argument("--deadline-s", type=float, default=0.0, help="绝对deadline(秒,0=不限)。")
    p.add_argument("--cost-short-s", type=float, default=30.0)
    p.add_argument("--cost-mid-s", type=float, default=90.0)
    p.add_argument("--cost-long-s", type=float, default=240.0)
    p.add_argument("--save-window-s", type=float, default=300.0,
                   help="至少预留一个保存窗口的秒数。")
    return p.parse_args(argv)


def plan_stages(stages_text):
    """纯CPU：打印 §5 阶段预算与顺序（评估共享先，训练臂共享次之）。"""
    want = [s.strip().upper() for s in str(stages_text or "").split(",") if s.strip()]
    total_wall = 0.0
    total_quota = 0.0
    lines = ["order: 共享评估(已有compare-mode直用,runner只包一层) > 同初始化两训练臂 > 后缀logits探针钩子"]
    for name, wall, quota, question in STAGES:
        if want and name not in want:
            continue
        total_wall += wall
        total_quota += quota
        lines.append("stage %s: wall<=%.1fh quota<=%.1fh | %s" % (name, wall, quota, question))
    lines.append("selected_total: wall=%.1fh quota=%.1fh (quota≈wall*%.1f)" % (
        total_wall, total_quota, GPU_QUOTA_FACTOR))
    lines.append("notebook_wall_s: 自本文件import起计时；gpu_quota_delta_s由GPU侧填写")
    return "\n".join(lines)


def resume_cursor_from_consumed_ids(consumed_ids, pool_ids):
    """§8.2 跨会话恢复：以实际消费ID为准，不重新采样。返回应继续的cursor下标。"""
    consumed = list(consumed_ids or [])
    pool = list(pool_ids or [])
    cursor = 0
    for cid in consumed:
        if cursor < len(pool) and pool[cursor] == cid:
            cursor += 1
        elif cid in pool:
            cursor = pool.index(cid) + 1
        else:
            break
    return cursor


def stage_stop_state(optimizer_state, rng_state, cursor, consumed_ids, completed_updates):
    """§8.2 阶段停止保留的可恢复状态；只含LoRA+必要optimizer/RNG/cursor。"""
    return {
        "optimizer_state": optimizer_state,
        "rng_state": rng_state,
        "cursor": int(cursor),
        "consumed_ids": list(consumed_ids or []),
        "completed_updates": int(completed_updates),
    }


def new_fresh_lora_spec(r=8, alpha=32):
    """§8.1 fresh LoRA：A正常初始化、B为0；禁止A/B同时清零。返回声明式spec。"""
    spec = {"r": int(r), "lora_alpha": int(alpha),
            "init_A": "normal", "init_B": "zeros", "zero_A": False}
    if spec["zero_A"]:
        raise ValueError("fresh LoRA禁止A/B同时清零")
    return spec


def attach_lora_snapshot(model, snapshot_dir, adapter_name="default"):
    """§8.1 训练臂同一起始LoRA快照开始。函数级复用train加载函数，不deepcopy底座。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import train_grid_sft as trainer  # 延迟导入；dry-run不走此分支

    import torch  # noqa: WPS433

    _ = torch
    stats = trainer.load_init_adapter_weights(
        model, snapshot_dir, torch, adapter_name=adapter_name)
    return {"status": "attached", "snapshot_dir": snapshot_dir, "stats": stats}


def clear_decode_state(model):
    """§8.3 每次权重/adapter/formatter切换清空KV/GDN/卷积状态。

    只调已有清空函数，不移植DFS回退。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import solve_grid as solver  # noqa: WPS433

    solver._drop_cached_decode_state(model)
    try:
        import torch  # noqa: WPS433

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return {"status": "cleared"}


def enter_train_mode(model):
    """§8.3 训练模式：use_cache=False + gradient checkpointing。复用train函数。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import train_grid_sft as trainer  # noqa: WPS433

    trainer.set_use_cache(model, False)
    for holder in (model, getattr(model, "base_model", None)):
        enable = getattr(holder, "gradient_checkpointing_enable", None)
        if callable(enable):
            enable()
    model.train()
    clear_decode_state(model)
    return {"status": "train_mode", "use_cache": False}


def enter_eval_mode(model):
    """§8.3 评价模式：eval + 受支持cache。调用方配合torch.no_grad()生成。"""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import train_grid_sft as trainer  # noqa: WPS433

    model.eval()
    trainer.set_use_cache(model, True)
    clear_decode_state(model)
    return {"status": "eval_mode", "use_cache": True}


def admission_check(remaining_s, est_cost_s, save_window_s):
    """§8.5 准入：剩余预算须覆盖本次估计成本 + 至少一个保存窗口。"""
    return float(remaining_s) >= float(est_cost_s) + float(save_window_s)


def interleave_arms_queries(arms, queries):
    """§8.5 多臂同query优先：同一query跑完各臂再换下一query。"""
    order = []
    for q in list(queries or []):
        for arm in list(arms or []):
            order.append((arm, q))
    return order


def should_skip_completed(out_path, checkpoint_hash, formatter, task_id, query_index,
                           view, config_hash, overwrite=False):
    """§8.6 按checkpoint hash/formatter/task/query/view/config判重；默认不覆盖。"""
    if overwrite:
        return False
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import solve_grid as solver  # noqa: WPS433

    key = solver.query_dedup_key(checkpoint_hash, formatter, task_id, query_index,
                                 view, config_hash)
    done = solver.load_completed_queries(out_path)
    return key in done


def evaluate_loaded_model(model, tokenizer, arm, formatter, queries, deadline_s, out_path,
                          checkpoint_hash="", config_hash="", overwrite=False):
    """§8.4 评价函数：只接收已加载model/tokenizer；内部不再from_pretrained。

    每query持久化原文/parsed/tokens/时长/状态；每arm独立分数，
    oracle(at_2任一候选中)与最终选2分开记录。deadline触发记
    budget_interrupted，不算模型错误。
    """
    script_dir = os.path.dirname(os.path.abspath(__file__))
    if script_dir not in sys.path:
        sys.path.insert(0, script_dir)
    import solve_grid as solver  # noqa: WPS433
    import train_grid_sft as trainer  # noqa: WPS433

    _ = trainer.resolve_train_formatter
    if model is None or tokenizer is None:
        raise ValueError("evaluate_loaded_model需要已加载的model/tokenizer")
    fmt = solver.resolve_formatter(formatter, None)
    enter_eval_mode(model)
    t0 = PROCESS_T0
    per_arm_records = []
    for pos, query in enumerate(list(queries or [])):
        task_id = query.get("task_id", "")
        query_index = int(query.get("query_index", pos))
        view = str(query.get("view", "orig"))
        if should_skip_completed(out_path, checkpoint_hash, fmt, task_id, query_index,
                                 view, config_hash, overwrite=overwrite):
            continue
        if deadline_s and time.time() >= float(deadline_s):
            row = {"task_id": task_id, "query_index": query_index, "view": view,
                   "arm": arm, "formatter": fmt, "status": "budget_interrupted",
                   "raw": "", "parsed": None, "tokens": 0, "seconds": 0.0,
                   "notebook_wall_s": time.time() - t0, "gpu_quota_delta_s": None}
            solver.append_query_jsonl(out_path, row)
            break
        qs = time.time()
        # 实际生成由调用方在no_grad下执行已有compare-mode候选路径；
        # 此处只落盘包装所需的逐query字段结构（runner只包一层）。
        row = {"task_id": task_id, "query_index": query_index, "view": view,
               "arm": arm, "formatter": fmt, "status": "not_run_no_gpu",
               "raw": "", "parsed": None, "tokens": 0,
               "seconds": time.time() - qs,
               "notebook_wall_s": time.time() - t0, "gpu_quota_delta_s": None}
        solver.append_query_jsonl(out_path, row)
        per_arm_records.append(row)
    agg = solver.pair_aggregate([]) if not per_arm_records else {"n_queries": len(per_arm_records)}
    return {"arm": arm, "formatter": fmt, "aggregate": agg,
            "oracle_vs_pick2": "oracle=任一候选中(at_2)；pick2=最终选2，二者分开记录",
            "notebook_wall_s": time.time() - t0, "gpu_quota_delta_s": None}


def suffix_logits_probe_hook(prompt_len_p, answer_len_t, contiguous_suffix,
                             labels_layout_ok, logits_to_keep_supported):
    """§6.1 后缀logits探针钩子：只定义接口+拒绝条件，不做数值比较。

    满足连续后缀契约才允许GPU侧实测；否则一律拒绝并给出原因。
    """
    reason = ""
    if not contiguous_suffix:
        reason = "非连续后缀契约，拒绝后缀路径"
    elif not labels_layout_ok:
        reason = "labels非[:,P-1:]布局或被手动shift，拒绝"
    elif not logits_to_keep_supported:
        reason = "不支持logits_to_keep=T+1，拒绝"
    elif int(answer_len_t) < 1 or int(prompt_len_p) < 1:
        reason = "P/T非法，拒绝"
    if reason:
        return {"status": "rejected", "reason": reason}
    return {"status": "admitted_not_run",
            "interface": "model(input_ids,attention_mask,labels[:,P-1:],"
                         "logits_to_keep=T+1,use_cache=False)",
            "P": int(prompt_len_p), "T": int(answer_len_t),
            "note": "数值比较(loss/梯度/时间)留待GPU会话，不在此实现"}


def main(argv=None):
    args = parse_args(argv)
    if int(args.scheduler_horizon) != SCHEDULER_HORIZON:
        raise SystemExit("scheduler-horizon必须为100，stop_after_update才有 §4 前缀语义")
    if args.dry_run:
        print(plan_stages(args.stage))
        print("stops: scheduler_horizon=100 stop_after_update=%d" % int(args.stop_after_update))
        print("dedup: checkpoint_hash/formatter/task/query/view/config；默认不覆盖")
        print("dirs: P12=%s C12=%s（分开）" % (args.p12_out, args.c12_out))
        print("probe: 后缀logits仅钩子接口，未实现数值比较")
        return 0
    raise SystemExit("非dry-run需在GPU会话手动执行：先完成A/B修后再接共享评估/训练臂")


if __name__ == "__main__":
    raise SystemExit(main())
