# ARC Prize 2026 — ARC-AGI-2 研究

目标：研究可在比赛资源限制内运行的 ARC-AGI-2 专用求解方案，争取金牌与奖金。优先根据公开实现、数据和实验依据选择路线。

比赛：[Kaggle ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

历史试提交：[用户指定 Notebook 版本 314617319](https://www.kaggle.com/code/ccjjllt/reproduce-nvarc-2025-results?scriptVersionId=314617319)。用户说明 30.14 分来自未修改的公开方案复制提交，不作为自主研究效果，也不限定后续方向。

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
| 010 配额刷新后的GPU实验 | [plan/010-gpu-budget-experiments.md](docs/plan/010-gpu-budget-experiments.md) | [progress/010-gpu-budget-experiments.md](docs/progress/010-gpu-budget-experiments.md)；30配额小时按约15小时L4规划；同400例F0/F1数据与P12/C12清单已准备，继承/格式接口待修，GPU未运行 |

## Kaggle 操作链（2026-09-29 实测结论）

Notebook 发布与数据源挂载三条路只通两条，按稳定性排序：

1. **Kaggle CLI push（最稳）**：`kaggle kernels push -p <dir>`，`kernel-metadata.json` 中 `competition_sources` / `model_sources` / `dataset_sources` 声明数据源。注册与运行均验证过（A0-smoke 嵌套 worker 与 v3 扁平 worker 文件均到位）。认证用 access token（`KAGGLE_API_TOKEN`，本机 SDK 1.7.4.5 已验证主 API 接受 Bearer），不需要 `kaggle.json`。
2. **官方 SDK SaveAndRunAll**：`KernelsApiClient.save_kernel`（`ApiSaveKernelRequest` 带 `competition_data_sources` + `model_data_sources`），回读元数据三项齐全；注意 worker 布局差异（扁平 vs 嵌套）会影响挂载路径判断，见 006 进展。
3. **MCP save_notebook（不通）**：数据源字段 6 种组合实测，存出的 Notebook 元数据里始终没有 sources，运行时 `/kaggle/input` 为空。只用它建/跑/查/取日志，不用它挂数据。

## 项目原则

以研究为主，不写测试文件，不引入自动化测试或 CI 门禁。所有测试和研究实验由人工发起、查看并记录。具体协作约定见 [AGENTS.md](AGENTS.md)。

本地 M5 / 32GB 用于开发与分析，Kaggle L4×4 优先用于训练和正式环境实验，Colab v5e-1 暂作为备选。可行性判断与尚未核实的资源额度见 001 进展文档。

006已完成BF16短程LoRA与小题集推理诊断。007 的 G1 已在 4×L4 上完成 200 次更新并保存 adapter。独立加载后，诊断题 exact-match 7/8。留出题 25 个 query 已评完：960 token 下 0/5，2048 token 下 6/20。这不是竞赛分数。零步底座是 6/8。G2 的 3 万条数据池已生成但未训练。G2 的 GDN 训练、G3、TTT、Unsloth 仍未运行。

008复核发现：25个已评留出query有24个属于旧ARC1，仅1个不在ARC1；没有同题留出零步对照。2048下40个候选均合法且尺寸正确，第二个随机候选未新增命中。当前重点是补齐ARC2代表性评价、修正G2实例/query/长度覆盖，再用小实验判断TTT和多视角收益。现有`--resume`不能直接完成G1→新G2池或新增GDN，实施缺口见008计划。2026-10-01实时剩0.7874配额小时，按L4两倍消耗约23.6分钟墙钟；刷新为北京时间10月3日08:00。

008后续CPU实现已补评价清单、token统计、weights-only初始化和新base等，具体状态见其progress后续补记。009继续完成不需GPU的研究：保留原词表去空格，在相同官方完整episode上省44.98% token；4K可容纳任务694→837。离线公平采样在同800行下覆盖446→601task；11族220pair规则重算一致，并对一例欠定episode实际替换demo。上述均为CPU数据/表示证据，没有模型提分。下一轮仍先做008原格式E0，再比较009提出的F1紧凑格式、采样及示例质量候选。

2026-10-03的010研究已核实GPU配额刷新为30小时（L4×4两倍扣额）。新增同400例、同顺序F0/F1训练文件，冻结P12开发/C12确认和30小时上限分配。真实权重键核对发现`--init-adapter`当前会漏载G1；F1推理提示尚未透传formatter，fair的base身份与repair ID也有缺口。010计划给出最小修正与共享模型会话交接，并优先比较原生答案后缀logits以减少无监督位置的输出头开销。现有脚本未在本轮改动，GPU实验未启动。
