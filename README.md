# ARC Prize 2026 — ARC-AGI-2 研究

目标：以已有约 30 分的 Kaggle 提交为起点，研究可在比赛资源限制内运行的改进方案，争取金牌。

比赛：[Kaggle ARC Prize 2026 — ARC-AGI-2](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

基线：[用户指定 Notebook 版本 314617319](https://www.kaggle.com/code/ccjjllt/reproduce-nvarc-2025-results?scriptVersionId=314617319)。分数与运行细节的核实状态以进展文档为准。

## 研究文档

| 轮次 | 计划 | 实际进展 |
| --- | --- | --- |
| 001 基线与比赛、算力调研 | [plan/001-baseline.md](docs/plan/001-baseline.md) | [progress/001-baseline.md](docs/progress/001-baseline.md) |
| 002 基于现有方案的提分研究 | [plan/002-strategy-improvement.md](docs/plan/002-strategy-improvement.md) | 尚未启动实验，开始后创建同名进展文档 |

## 项目原则

以研究为主，不写测试文件，不引入自动化测试或 CI 门禁。所有测试和研究实验由人工发起、查看并记录。具体协作约定见 [AGENTS.md](AGENTS.md)。

本地 M5 / 32GB 用于开发与分析，Kaggle L4×4 优先用于训练和正式环境实验，Colab v5e-1 暂作为备选。可行性判断与尚未核实的资源额度见 001 进展文档。

目前正在建立研究起点；文档中的实验设计不代表已经获得提分结果。
