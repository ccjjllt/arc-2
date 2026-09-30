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
| 007 正式网格微调与NVARC专用化 | [plan/007-grid-specialist-sft.md](docs/plan/007-grid-specialist-sft.md) | [progress/007-grid-specialist-sft.md](docs/progress/007-grid-specialist-sft.md)；当前实现交接，训练未运行 |

## Kaggle 操作链（2026-09-29 实测结论）

Notebook 发布与数据源挂载三条路只通两条，按稳定性排序：

1. **Kaggle CLI push（最稳）**：`kaggle kernels push -p <dir>`，`kernel-metadata.json` 中 `competition_sources` / `model_sources` / `dataset_sources` 声明数据源。注册与运行均验证过（A0-smoke 嵌套 worker 与 v3 扁平 worker 文件均到位）。认证用 access token（`KAGGLE_API_TOKEN`，本机 SDK 1.7.4.5 已验证主 API 接受 Bearer），不需要 `kaggle.json`。
2. **官方 SDK SaveAndRunAll**：`KernelsApiClient.save_kernel`（`ApiSaveKernelRequest` 带 `competition_data_sources` + `model_data_sources`），回读元数据三项齐全；注意 worker 布局差异（扁平 vs 嵌套）会影响挂载路径判断，见 006 进展。
3. **MCP save_notebook（不通）**：数据源字段 6 种组合实测，存出的 Notebook 元数据里始终没有 sources，运行时 `/kaggle/input` 为空。只用它建/跑/查/取日志，不用它挂数据。

## 项目原则

以研究为主，不写测试文件，不引入自动化测试或 CI 门禁。所有测试和研究实验由人工发起、查看并记录。具体协作约定见 [AGENTS.md](AGENTS.md)。

本地 M5 / 32GB 用于开发与分析，Kaggle L4×4 优先用于训练和正式环境实验，Colab v5e-1 暂作为备选。可行性判断与尚未核实的资源额度见 001 进展文档。

006已完成BF16短程LoRA与小题集推理诊断。007已根据真实源码修正thinking、答案loss与结果口径，下一步是保存可复用的网格专项微调权重；尚无本项目正式SFT或新的竞赛提分结果。
