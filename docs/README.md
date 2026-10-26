# 文档索引

当前研究入口：[027计划](plan/027-nvarc-ttt-view-ensemble.md)与[027进展](progress/027-nvarc-ttt-view-ensemble.md)，[产物导航](../experiments/027/README.md)。原权重独立TTT视图确认完成：base5/34→融合6/34，净增1、无丢失、前缀无回归；普通验证已通过并正式提交57634436，绑定V1／362150112，等待真实公榜结果。026失败续训配方保持结束，不复活其权重。

025此前B4=R2=15/58、oracle17/58，已结束R2扩展。正式交付仍为023的30.14，参见[023进展](progress/023-score-recovery-and-selection.md)。

`plan/`记录背景、假设和执行交接，`progress/`记录实际改动与实验。旧条目是历史快照；同轮后续追加及新轮次会更新或纠正早期结论。023的57582282于2026-10-25实查COMPLETE／公榜30.14，与历史最高持平，最新证据见023进展§9。

## 轮次索引

| 轮次 | 计划 | 实际进展 |
| --- | --- | --- |
| 001 基线与比赛、算力调研 | [plan/001-baseline.md](plan/001-baseline.md) | [progress/001-baseline.md](progress/001-baseline.md) |
| 002 基于现有方案的提分研究 | [plan/002-strategy-improvement.md](plan/002-strategy-improvement.md) | 未启动；路线优先级由 003/004 更新 |
| 003 深度调研与方向选择 | [plan/003-research-direction.md](plan/003-research-direction.md) | [progress/003-research-direction.md](progress/003-research-direction.md) |
| 004 NVARC 网格续训对照 | [plan/004-specialist-solver.md](plan/004-specialist-solver.md) | 保留对照；SOAR路线已移除；未运行 |
| 005 更强初始化模型比较 | [plan/005-stronger-backbones.md](plan/005-stronger-backbones.md) | [progress/005-stronger-backbones.md](progress/005-stronger-backbones.md)；已选Qwen3.8-27B |
| 006 Qwen3.8-27B与12小时方案 | [plan/006-qwen38-l4-budget.md](plan/006-qwen38-l4-budget.md) | [progress/006-qwen38-l4-budget.md](progress/006-qwen38-l4-budget.md)；首轮诊断已收官，源码/摘要已归档 |
| 007 正式网格微调与NVARC专用化 | [plan/007-grid-specialist-sft.md](plan/007-grid-specialist-sft.md) | [progress/007-grid-specialist-sft.md](progress/007-grid-specialist-sft.md)；诊断 7/8，留出 960 下 0/5、2048 下 6/20；G2 的 3 万条池已生成、未训练 |
| 008 泛化证据、数据覆盖与逐题适配 | [plan/008-generalization-adaptation.md](plan/008-generalization-adaptation.md) | [progress/008-generalization-adaptation.md](progress/008-generalization-adaptation.md)；研究复核完成，GPU实验未运行；先补代表性配对评价，再研究修正后的G2与TTT/TTA |
| 009 有效episode：表示、采样与消歧 | [plan/009-efficient-episodes.md](plan/009-efficient-episodes.md) | [progress/009-efficient-episodes.md](progress/009-efficient-episodes.md)；CPU研究完成：紧凑格式约省45% token，同预算覆盖改善，一例demo替换消歧；模型效果未运行 |
| 010 配额刷新后的GPU实验 | [plan/010-gpu-budget-experiments.md](plan/010-gpu-budget-experiments.md) | [progress/010-gpu-budget-experiments.md](progress/010-gpu-budget-experiments.md)；E0及F1-25/50 P12均0/14，F0-suffix-25随后在011评完0/14；记录纠错见011 |
| 011 有效预算与适配研究 | [plan/011-budget-efficient-adaptation.md](plan/011-budget-efficient-adaptation.md) | [progress/011-budget-efficient-adaptation.md](progress/011-budget-efficient-adaptation.md)；两GPU会话完成，无新增解；TTT 0/2、程序输出截断未进入反馈；停止扩算并保存预算 |
| 012 GDN推理提速与程序完成性 | [plan/012-kernel-program-completion.md](plan/012-kernel-program-completion.md) | [progress/012-kernel-program-completion.md](progress/012-kernel-program-completion.md)；K不采用快速后端；P-v2四次3072 token截断，无完整程序，D/L各0/2；结束本轮，保留配额 |
| 013 奖牌目标、提交策略与27B推理研究 | [plan/013-submission-strategy.md](plan/013-submission-strategy.md) | [progress/013-submission-strategy.md](progress/013-submission-strategy.md)；4B诊断2/4，正式提交57103498公榜25.69；27B依赖CPU核对完成 |
| 014 退分归因与27B完整推理 | [plan/014-regression-and-reasoning.md](plan/014-regression-and-reasoning.md) | [progress/014-regression-and-reasoning.md](progress/014-regression-and-reasoning.md)；R v4两臂2/6、无oracle新增，结束4B扩算；V v4导入失败，先修运行与计量，T尚未开始 |
| 015 完整推理与程序反馈 | [plan/015-qwen38-reasoning-research.md](plan/015-qwen38-reasoning-research.md) | [progress/015-qwen38-reasoning-research.md](progress/015-qwen38-reasoning-research.md)；P6 两臂各 1/6，多数调用截断 |
| 016 有限思考与赛前提交 | [plan/016-budgeted-final-submission.md](plan/016-budgeted-final-submission.md) | [progress/016-budgeted-final-submission.md](progress/016-budgeted-final-submission.md)；v1 COMPLETE：C8选中、硬预算生效、B16合法6/6但无新增正确解、恢复EM@2持平、用户四项确认入§12 |
| 017 程序反馈对照与提交入口 | [plan/017-program-feedback-and-submission-entry.md](plan/017-program-feedback-and-submission-entry.md) | [progress/017-program-feedback-and-submission-entry.md](progress/017-program-feedback-and-submission-entry.md)；v1跑完：P2门未过+4未跑，F-db程序exact；原问题未答，已由018接替 |
| 018 F-291反馈对照与受约束程序生成 | [plan/018-f291-feedback-and-constrained-generation.md](plan/018-f291-feedback-and-constrained-generation.md) | [progress/018-f291-feedback-and-constrained-generation.md](progress/018-f291-feedback-and-constrained-generation.md)；A线v6中F-fix在291拿下exact、R2截断；复核更正哈希死循环与B线四项阻断缺陷，转入019 |
| 019 全量提交候选打通、首稿形成率与+4流程验证 | [plan/019-submission-and-first-draft-pipeline.md](plan/019-submission-and-first-draft-pipeline.md) | [progress/019-submission-and-first-draft-pipeline.md](progress/019-submission-and-first-draft-pipeline.md)；P1输出120task/172query，附件评分0/158；+4反馈0/4、重启1/4；并发与加载计量已更正 |
| 020 提分验证与真实提交交接 | [plan/020-score-validation-and-real-submission.md](plan/020-score-validation-and-real-submission.md) | [progress/020-score-validation-and-real-submission.md](progress/020-score-validation-and-real-submission.md)；C24 training留出17/26，正式57443913公榜12.50，未实现提分；低分复盘见§27 |
| 021 NVARC完整流程与27B互补交接 | [plan/021-nvarc-complementarity-handoff.md](plan/021-nvarc-complementarity-handoff.md) | [progress/021-nvarc-complementarity-handoff.md](progress/021-nvarc-complementarity-handoff.md)；4B加载失败未求解，27B E20版本对齐2／27；三臂配对与限时续训未运行 |
| 022 完整4B计分提交与已有27B配对 | [plan/022-full-nvarc-submission.md](plan/022-full-nvarc-submission.md) | [progress/022-full-nvarc-submission.md](progress/022-full-nvarc-submission.md)；E20 V2实际3／27、oracle5／27；正式57504054已COMPLETE／0.00，隐藏根因未确认；忠实原V1复现转023 |
| 023 恢复有效提交与候选选择交接 | [plan/023-score-recovery-and-selection.md](plan/023-score-recovery-and-selection.md) | [progress/023-score-recovery-and-selection.md](progress/023-score-recovery-and-selection.md)；正式57582282已COMPLETE／30.14并绑定361045383，与历史最高持平；原四题3／5仅公开诊断；下一步B4／R2确认未运行 |
| 024 本地清理与目录整理 | [plan/024-project-organization.md](plan/024-project-organization.md) | [progress/024-project-organization.md](progress/024-project-organization.md)；仅目录维护，研究主线仍为023 |
| 025 研究缺陷与赛前提分决策 | [plan/025-research-gap-and-next-step.md](plan/025-research-gap-and-next-step.md) | [progress/025-research-gap-and-next-step.md](progress/025-research-gap-and-next-step.md)；40task／58query确认完成，B4=R2=15/58，oracle17/58，无净增，保留B4 |
| 026 NVARC额外训练规则覆盖挑战者 | [plan/026-nvarc-extra-rule-training.md](plan/026-nvarc-extra-rule-training.md) | [progress/026-nvarc-extra-rule-training.md](progress/026-nvarc-extra-rule-training.md)；V1 COMPLETE，base=challenger=4/23，新增2／丢失2，oracle6→9；结束该配方 |
| 027 原权重TTT视图集成与提交 | [plan/027-nvarc-ttt-view-ensemble.md](plan/027-nvarc-ttt-view-ensemble.md) | [progress/027-nvarc-ttt-view-ensemble.md](progress/027-nvarc-ttt-view-ensemble.md)；确认5→6/34、净1无回归；正式57634436已接受、精确绑定362150112，分数待返回 |

## 操作与维护

- [Kaggle操作经验](kaggle-workflow.md)：历史实测的发布、挂载、日志与精确版本提交区别。
- [实验目录](../experiments/README.md)：当前主线、依赖和历史产物。
- [共享脚本](../scripts/README.md)：按用途查找实现。
- [项目协作约定](../AGENTS.md)：研究范围、手动实验和记录要求。
