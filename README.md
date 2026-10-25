# ARC Prize 2026 — ARC-AGI-2 研究

目标：在比赛资源限制内，基于NVARC方案尽可能提高实际分数。比赛入口：[Kaggle ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

## 当前进度

**当前阶段：026 NVARC 4B续训已正常启动，等待完成。** 025已完成：B4与R2均15/58，oracle17/58，结束R2扩展并保留B4。026从原33.89精确V1派生，只改变初始化权重；额外training规则合成与官方training replay小规模续训，与新基线在固定16题／23query上同轮评分。未取得026结果，不宣称提分。正式最好仍为023的30.14。

2026-10-25用户确认以NVARC为唯一主线，每轮只优化一个因素。当前交接见[026计划](docs/plan/026-nvarc-extra-rule-training.md)、[实际进展](docs/progress/026-nvarc-extra-rule-training.md)；上一轮[025结果与产物](experiments/025/README.md)。

截至**2026-10-25 10:05北京时间的实查记录**：

- [忠实V1 Notebook](https://www.kaggle.com/code/zhuating/arc2-023-failed-in-aimo-v1-faithful?scriptVersionId=361045383)普通GPU运行已COMPLETE，精确版本 **V1／361045383**。
- 原4task／5query均完成TTT、DFS与重评分；输入匹配评分为 **3/5 query、2/4完整task**。这是公开演示集合结果，不是全量评估或比赛分数，候选oracle未确认。
- **023正式提交57582282已COMPLETE，公榜30.14**，GetSubmission与同队列表一致，绑定V1／361045383。与历史复制30.14持平，未实现提分；私榜未返回。[实际结果](experiments/023/execution/submission-57582282-result-20261025.json)已保存。
- 原V1同候选池B4／R2确认已完成，旧022上的净增1未在新集合重现，结束R2扩展。下一轮依据41个未生成正确候选的query分析规则缺口。最新结果见[025进展§12](docs/progress/025-research-gap-and-next-step.md#12-完整运行及同候选池评分完成)，正式起点见[023进展§9](docs/progress/023-score-recovery-and-selection.md#9-2026-10-25结果回查complete公榜3014正式起点建立)。

## 成绩与证据

| 记录 | 结果 | 限制与入口 |
| --- | --- | --- |
| 最新正式提交023／57582282 | **30.14**，COMPLETE | 精确V1／361045383；[实际结果](experiments/023/execution/submission-57582282-result-20261025.json)，与历史最好持平 |
| 上一份022／57504054 | **0.00**，COMPLETE | 隐藏执行根因未确认；[结果凭据](experiments/022/execution/submission-57504054-result.json) |
| 020／57443913 | **12.50**，COMPLETE | 27B替换路线；[结果凭据](experiments/020/execution/competition-submission.json) |
| 历史复制提交／314617319 | 用户报告 **30.14** | 未修改公开方案的试提交，不作为自主研究效果；[历史记录](docs/progress/001-baseline.md) |
| 022公开E20 | 4B最终3/27，oracle5/27 | 已开发集合；[4B与既有27B配对](experiments/022/paired-e20-v2/paired-table.md) |
| 025公开确认40task／58query | B4=R2=15/58，oracle17/58 | 本地公开诊断，非公榜；[完整结果](experiments/025/confirmation-v2-summary.json) |

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
