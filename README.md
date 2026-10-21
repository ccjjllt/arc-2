# ARC Prize 2026 — ARC-AGI-2 研究

目标：研究可在比赛资源限制内运行的 ARC-AGI-2 专用求解方案，争取金牌与奖金。优先根据公开实现、数据和实验依据选择路线。

比赛：[Kaggle ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

**当前研究入口（2026-10-21）**：[021交接方案](docs/plan/021-nvarc-complementarity-handoff.md)与[021调研进展](docs/progress/021-nvarc-complementarity-handoff.md)。已核对官方条件、归档33.89对应的Failed in AIMO V1并保存109个新增生成器清单。NVARC V1 E20已获接受，但最近日志仍为0/27生成且输出状态未确认；三臂对照与续训尚未运行。

**最新实际成绩**：[020低分复盘](docs/progress/020-score-validation-and-real-submission.md#27-正式公榜1250与低分复盘)。正式提交 **57443913** 已COMPLETE，公榜 **12.50**，低于历史30.14，未实现提分。C24固定17/26为training来源项目留出，不能外推榜分；隐藏实际覆盖/日志尚未取得。

历史试提交：[用户指定 Notebook 版本 314617319](https://www.kaggle.com/code/ccjjllt/reproduce-nvarc-2025-results?scriptVersionId=314617319)。用户说明 30.14 分来自未修改的公开方案复制提交，不作为自主研究效果，也不限定后续方向。

**上一轮执行记录（2026-10-20）**：[020计划](docs/plan/020-score-validation-and-real-submission.md)与[020进展](docs/progress/020-score-validation-and-real-submission.md)。旧账号优先的GPU研究已完成本轮，生成采样忙碌率99.68%；zhuating仅提供CPU依赖恢复、未启用ARC2 GPU研究。正式Notebook v1普通保存300秒只做CPU产文件，隐藏41400秒运行同一冻结模型策略；C24模型产物已完整评分，普通文件不是模型评测。E20下一轮公开输入已准备、未运行。

[012实际进展](docs/progress/012-kernel-program-completion.md)记录上一轮：K短探针不采用快速后端；P-v2四次均3072 token截断，没有完整程序或实际执行，D/L各0/2，已按预定止损结束。历史配额快照不替代013的最新查询。

[011实际进展](docs/progress/011-budget-efficient-adaptation.md)记录已完成结果：F0-25 P12为0/14，多视角完整配对0/5；TTT每题4次真实更新后，A0/A4仍均0/2。程序4次均耗尽1536 token、未产出代码，repair实际0次。10月11日15:00北京时间账户快照剩15.23138配额小时、reserved=0；未来开机前重查，保留至少6小时。无新比赛分数，C6/C12确认未运行；首GPU v1的取消原因和成本保留。

**正式结果**：`57443913` / [Notebook v1](https://www.kaggle.com/code/ccjjllt/arc2-020-deep-direct-submission?scriptVersionId=360461153)，2026-10-21 17:18北京时间实查COMPLETE、**12.50**。该版本是单次长思考Direct1+窄CPU规则的替换路线，没有与历史30.14方案做过可靠融合。[得分与版本凭据](experiments/020/execution/competition-submission.json)。

**下一步**：按021使用已核实V1恢复完整4B对照，再验证27B的实际双槽净增能否抵消让预算损失。E20用于配对诊断，不能称未见集；不原样重投12.50，不把源码归档或训练loss当作提分证据。

## 研究文档

| 轮次 | 计划 | 实际进展 |
| --- | --- | --- |
| 001 基线与比赛、算力调研 | [plan/001-baseline.md](docs/plan/001-baseline.md) | [progress/001-baseline.md](docs/progress/001-baseline.md) |
| 002 基于现有方案的提分研究 | [plan/002-strategy-improvement.md](docs/plan/002-strategy-improvement.md) | 未启动；路线优先级由 003/004 更新 |
| 003 深度调研与方向选择 | [plan/003-research-direction.md](docs/plan/003-research-direction.md) | [progress/003-research-direction.md](docs/progress/003-research-direction.md) |
| 004 NVARC 网格续训对照 | [plan/004-specialist-solver.md](docs/plan/004-specialist-solver.md) | 保留对照；SOAR路线已移除；未运行 |
| 005 更强初始化模型比较 | [plan/005-stronger-backbones.md](docs/plan/005-stronger-backbones.md) | [progress/005-stronger-backbones.md](docs/progress/005-stronger-backbones.md)；已选Qwen3.8-27B |
| 006 Qwen3.8-27B与12小时方案 | [plan/006-qwen38-l4-budget.md](docs/plan/006-qwen38-l4-budget.md) | [progress/006-qwen38-l4-budget.md](docs/progress/006-qwen38-l4-budget.md)；首轮诊断已收官，源码/摘要已归档 |
| 007 正式网格微调与NVARC专用化 | [plan/007-grid-specialist-sft.md](docs/plan/007-grid-specialist-sft.md) | [progress/007-grid-specialist-sft.md](docs/progress/007-grid-specialist-sft.md)；诊断 7/8，留出 960 下 0/5、2048 下 6/20；G2 的 3 万条池已生成、未训练 |
| 008 泛化证据、数据覆盖与逐题适配 | [plan/008-generalization-adaptation.md](docs/plan/008-generalization-adaptation.md) | [progress/008-generalization-adaptation.md](docs/progress/008-generalization-adaptation.md)；研究复核完成，GPU实验未运行；先补代表性配对评价，再研究修正后的G2与TTT/TTA |
| 009 有效episode：表示、采样与消歧 | [plan/009-efficient-episodes.md](docs/plan/009-efficient-episodes.md) | [progress/009-efficient-episodes.md](docs/progress/009-efficient-episodes.md)；CPU研究完成：紧凑格式约省45% token，同预算覆盖改善，一例demo替换消歧；模型效果未运行 |
| 010 配额刷新后的GPU实验 | [plan/010-gpu-budget-experiments.md](docs/plan/010-gpu-budget-experiments.md) | [progress/010-gpu-budget-experiments.md](docs/progress/010-gpu-budget-experiments.md)；E0及F1-25/50 P12均0/14，F0-suffix-25随后在011评完0/14；记录纠错见011 |
| 011 有效预算与适配研究 | [plan/011-budget-efficient-adaptation.md](docs/plan/011-budget-efficient-adaptation.md) | [progress/011-budget-efficient-adaptation.md](docs/progress/011-budget-efficient-adaptation.md)；两GPU会话完成，无新增解；TTT 0/2、程序输出截断未进入反馈；停止扩算并保存预算 |
| 012 GDN推理提速与程序完成性 | [plan/012-kernel-program-completion.md](docs/plan/012-kernel-program-completion.md) | [progress/012-kernel-program-completion.md](docs/progress/012-kernel-program-completion.md)；K不采用快速后端；P-v2四次3072 token截断，无完整程序，D/L各0/2；结束本轮，保留配额 |
| 013 奖牌目标、提交策略与27B推理研究 | [plan/013-submission-strategy.md](docs/plan/013-submission-strategy.md) | [progress/013-submission-strategy.md](docs/progress/013-submission-strategy.md)；4B诊断2/4，正式提交57103498公榜25.69；27B依赖CPU核对完成 |
| 014 退分归因与27B完整推理 | [plan/014-regression-and-reasoning.md](docs/plan/014-regression-and-reasoning.md) | [progress/014-regression-and-reasoning.md](docs/progress/014-regression-and-reasoning.md)；R v4两臂2/6、无oracle新增，结束4B扩算；V v4导入失败，先修运行与计量，T尚未开始 |
| 015 完整推理与程序反馈 | [plan/015-qwen38-reasoning-research.md](docs/plan/015-qwen38-reasoning-research.md) | [progress/015-qwen38-reasoning-research.md](docs/progress/015-qwen38-reasoning-research.md)；P6 两臂各 1/6，多数调用截断 |
| 016 有限思考与赛前提交 | [plan/016-budgeted-final-submission.md](docs/plan/016-budgeted-final-submission.md) | [progress/016-budgeted-final-submission.md](docs/progress/016-budgeted-final-submission.md)；v1 COMPLETE：C8选中、硬预算生效、B16合法6/6但无新增正确解、恢复EM@2持平、用户四项确认入§12 |
| 017 程序反馈对照与提交入口 | [plan/017-program-feedback-and-submission-entry.md](docs/plan/017-program-feedback-and-submission-entry.md) | [progress/017-program-feedback-and-submission-entry.md](docs/progress/017-program-feedback-and-submission-entry.md)；v1跑完：P2门未过+4未跑，F-db程序exact；原问题未答，已由018接替 |
| 018 F-291反馈对照与受约束程序生成 | [plan/018-f291-feedback-and-constrained-generation.md](docs/plan/018-f291-feedback-and-constrained-generation.md) | [progress/018-f291-feedback-and-constrained-generation.md](docs/progress/018-f291-feedback-and-constrained-generation.md)；A线v6中F-fix在291拿下exact、R2截断；复核更正哈希死循环与B线四项阻断缺陷，转入019 |
| 019 全量提交候选打通、首稿形成率与+4流程验证 | [plan/019-submission-and-first-draft-pipeline.md](docs/plan/019-submission-and-first-draft-pipeline.md) | [progress/019-submission-and-first-draft-pipeline.md](docs/progress/019-submission-and-first-draft-pipeline.md)；P1输出120task/172query，附件评分0/158；+4反馈0/4、重启1/4；并发与加载计量已更正 |
| 020 提分验证与真实提交交接 | [plan/020-score-validation-and-real-submission.md](docs/plan/020-score-validation-and-real-submission.md) | [progress/020-score-validation-and-real-submission.md](docs/progress/020-score-validation-and-real-submission.md)；C24 training留出17/26，正式57443913公榜12.50，未实现提分；低分复盘见§27 |
| 021 NVARC完整流程与27B互补交接 | [plan/021-nvarc-complementarity-handoff.md](docs/plan/021-nvarc-complementarity-handoff.md) | [progress/021-nvarc-complementarity-handoff.md](docs/progress/021-nvarc-complementarity-handoff.md)；调研完成，V1和109生成器差集已归档；三臂实验与限时续训未运行 |

## Kaggle 操作链（2026-09-29 实测结论）

Notebook 发布与数据源挂载三条路只通两条，按稳定性排序：

1. **Kaggle CLI push（最稳）**：`kaggle kernels push -p <dir>`，`kernel-metadata.json` 中 `competition_sources` / `model_sources` / `dataset_sources` 声明数据源。注册与运行均验证过（A0-smoke 嵌套 worker 与 v3 扁平 worker 文件均到位）。认证用 access token（`KAGGLE_API_TOKEN`，9月29日使用 SDK 1.7.4.5 验证；10月20日当前CLI为2.2.4），不需要 `kaggle.json`。push只是保存运行；真正比赛提交须指定精确Notebook版本调用代码提交接口并取得submission ref，见020计划。
2. **官方 SDK SaveAndRunAll**：`KernelsApiClient.save_kernel`（`ApiSaveKernelRequest` 带 `competition_data_sources` + `model_data_sources`），回读元数据三项齐全；注意 worker 布局差异（扁平 vs 嵌套）会影响挂载路径判断，见 006 进展。
3. **MCP save_notebook（不通）**：数据源字段 6 种组合实测，存出的 Notebook 元数据里始终没有 sources，运行时 `/kaggle/input` 为空。只用它建/跑/查/取日志，不用它挂数据。

10月11日补记：运行中实时日志使用 `kaggle kernels logs <owner/slug> --follow`；输出列表在作业未完成时可能为空。以CLI push返回的真实slug为准，标题不匹配会生成不同slug。取消需要精确session ID（可由该版本输出下载地址的`/kf/<id>/`核对），取消后确认状态和GPU reserved释放；不要把Notebook元数据id当session ID。

## 项目原则

以研究为主，不写测试文件，不引入自动化测试或 CI 门禁。所有测试和研究实验由人工发起、查看并记录。具体协作约定见 [AGENTS.md](AGENTS.md)。

本地 M5 / 32GB 用于开发与分析，Kaggle L4×4 优先用于训练和正式环境实验，Colab v5e-1 暂作为备选。可行性判断与尚未核实的资源额度见 001 进展文档。

006已完成BF16短程LoRA与小题集推理诊断。007 的 G1 已在 4×L4 上完成 200 次更新并保存 adapter。独立加载后，诊断题 exact-match 7/8。留出题 25 个 query 已评完：960 token 下 0/5，2048 token 下 6/20。这不是竞赛分数。零步底座是 6/8。G2 的 3 万条数据池已生成但未训练。G2 的 GDN 训练、G3、TTT、Unsloth 仍未运行。

008复核发现：25个已评留出query有24个属于旧ARC1，仅1个不在ARC1；没有同题留出零步对照。2048下40个候选均合法且尺寸正确，第二个随机候选未新增命中。当前重点是补齐ARC2代表性评价、修正G2实例/query/长度覆盖，再用小实验判断TTT和多视角收益。现有`--resume`不能直接完成G1→新G2池或新增GDN，实施缺口见008计划。2026-10-01实时剩0.7874配额小时，按L4两倍消耗约23.6分钟墙钟；刷新为北京时间10月3日08:00。

008后续CPU实现已补评价清单、token统计、weights-only初始化和新base等，具体状态见其progress后续补记。009继续完成不需GPU的研究：保留原词表去空格，在相同官方完整episode上省44.98% token；4K可容纳任务694→837。离线公平采样在同800行下覆盖446→601task；11族220pair规则重算一致，并对一例欠定episode实际替换demo。上述均为CPU数据/表示证据，没有模型提分。下一轮仍先做008原格式E0，再比较009提出的F1紧凑格式、采样及示例质量候选。

010随后完成多轮GPU先导：F0底座/G1的P12均0/14，F1-25/50也均0/14；F0完整logits路径OOM，suffix路径完成25步且保存权重。011完成共享runner、TTT和程序入口，并实际运行有限先导。两次加载为5–6分钟，共享加载已生效，但无同条件A/B来归因于某个开关，四卡利用率仍低。F0-25 P12也为0/14，TTT P2无新增，程序分支尚未产出代码。成本、旧记录纠错、配额和后续止损顺序统一见011。
