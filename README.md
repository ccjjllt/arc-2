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
| 004 专用模型与互补求解器 | [plan/004-specialist-solver.md](docs/plan/004-specialist-solver.md) | 实现与手动实验交由用户执行；未运行 |

## 项目原则

以研究为主，不写测试文件，不引入自动化测试或 CI 门禁。所有测试和研究实验由人工发起、查看并记录。具体协作约定见 [AGENTS.md](AGENTS.md)。

本地 M5 / 32GB 用于开发与分析，Kaggle L4×4 优先用于训练和正式环境实验，Colab v5e-1 暂作为备选。可行性判断与尚未核实的资源额度见 001 进展文档。

目前正在建立研究起点；文档中的实验设计不代表已经获得提分结果。
