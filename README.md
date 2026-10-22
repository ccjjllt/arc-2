# ARC Prize 2026 — ARC-AGI-2 研究

目标：在比赛资源限制内研究ARC专用求解方案，争取金牌。比赛入口：[Kaggle ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

## 当前进度

**当前唯一主线：023忠实复现Failed in AIMO V1／312063593，先取得可用正式成绩，再优化。**参照版33.89来自2026-10-21核验的外部历史公榜，不是本项目成绩，也不保证金牌。

截至**2026-10-22 15:40北京时间的已归档记录**：

- [忠实V1 Notebook](https://www.kaggle.com/code/zhuating/arc2-023-failed-in-aimo-v1-faithful?scriptVersionId=361045383)普通GPU运行已COMPLETE，精确版本 **V1／361045383**。
- 原4task／5query均完成TTT、DFS与重评分；输入匹配评分为 **3/5 query、2/4完整task**。这是公开演示集合结果，不是全量评估或比赛分数，候选oracle未确认。
- 15:34同队正式额度为0，**023尚未创建正式比赛提交**。[正式就绪参数](experiments/023/formal-v1-ready.json)已冻结；额度恢复后先查重和核实额度，再直接提交该精确V1，无需重跑普通保存。
- R2、TRM、新训练和更多27B推理后置。详细顺序见[023计划](docs/plan/023-score-recovery-and-selection.md)，实际证据见[023进展](docs/progress/023-score-recovery-and-selection.md)。本地目录整理未重新查询远端。

## 成绩与证据

| 记录 | 结果 | 限制与入口 |
| --- | --- | --- |
| 最新正式提交022／57504054 | **0.00**，COMPLETE | 隐藏执行根因未确认；[结果凭据](experiments/022/execution/submission-57504054-result.json) |
| 020／57443913 | **12.50**，COMPLETE | 27B替换路线；[结果凭据](experiments/020/execution/competition-submission.json) |
| 历史复制提交／314617319 | 用户报告 **30.14** | 未修改公开方案的试提交，不作为自主研究效果；[历史记录](docs/progress/001-baseline.md) |
| 022公开E20 | 4B最终3/27，oracle5/27 | 已开发集合；[4B与既有27B配对](experiments/022/paired-e20-v2/paired-table.md) |

022修改过预算和调度，不视为原样复现。023从冻结原V1派生，保留原求解与预算策略，只修实际运行阻塞；公开样本、代码完成或训练loss都不能代替正式成绩。

## 目录入口

```text
arc-2/
├── AGENTS.md           项目协作约定
├── README.md           当前主线与进度
├── docs/
│   ├── README.md       全轮次文档索引
│   ├── plan/           编号计划与交接
│   └── progress/       实际改动、实验与结果
├── experiments/
│   ├── README.md       当前依赖及历史产物导航
│   ├── 023/            当前忠实V1交付
│   └── 006/…022/       既有实验、冻结包和证据
└── scripts/            共享数据、训练与推理脚本
```

[研究文档索引](docs/README.md) · [实验导航](experiments/README.md) · [023产物入口](experiments/023/README.md) · [共享脚本](scripts/README.md) · [Kaggle操作经验](docs/kaggle-workflow.md)

本地用于开发与分析，主要GPU实验优先Kaggle L4×4。遵守[AGENTS.md](AGENTS.md)：手动发起并记录实验，不新增测试套件或CI。目录与脚本保持原路径以兼容冻结复现；本轮清理范围见[024维护记录](docs/progress/024-project-organization.md)。
