# ARC Prize 2026 — ARC-AGI-2 研究

目标：在比赛资源限制内，基于NVARC方案尽可能提高实际分数。比赛入口：[Kaggle ARC Prize 2026](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)。

## 当前进度

2026-10-29 **033正式57737364已COMPLETE／30.56，比此前最好30.14提高0.42**。精确[Minimal Perfpatch V2／335137615](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch?scriptVersionId=335137615)绑定通过，保留为当前正式最好；私榜与隐藏覆盖未知，不能归因单个热点。[033结果](docs/progress/033-nvarc-gpu-nll-submission.md)。

当前[034](docs/progress/034-nvarc-query-cost-queue.md)仅改033的全局任务入队顺序，按visible cells/query优先；原求解源逐字不变。普通V1已COMPLETE，两题TTT128、16批及身份核验通过，2026-10-30已正式提交 **57787438／PENDING**，zhuating／ccjjllt、精确[363524873](https://www.kaggle.com/code/zhuating/arc2-034-visible-cell-cost-per-query-queue?scriptVersionId=363524873)绑定正确；尚无新分数。CPU回放全扫尾部更慢4.64%，只有实际公榜>30.56才保留。

032原作者V1／312063593正式submission57694986已实查COMPLETE **28.89**，比02330.14低1.25；精确绑定正确，隐藏退分原因未确认，不重提原V1。023 **30.14** 保留历史对照，当前最好已为03330.56。[032结果](docs/progress/032-highest-notebook-submission.md)。

2026-10-27 **[031 V1／362740379](https://www.kaggle.com/code/zhuating/arc2-031-ttt-budget64?scriptVersionId=362740379)已COMPLETE：2/12→2/12，增0／丢0／净0，结束64步配方、不正式提交**。原128步调度保留、B确实64步停止；部署账1531.644→1409.424秒、搜索46→48批，但没有新增正确query。第二槽压力保住，预定净增条件未过。[031结果](experiments/031/v1-summary.json)、[进展](docs/progress/031-nvarc-ttt-budget.md)。保留原128步TTT／DFS0.2／B4，正式最好仍02330.14。

**2026-10-26实查：027正式成绩29.72，较023的30.14下降0.42。** submission **57634436**已COMPLETE，绑定[精确V1／362150112](https://www.kaggle.com/code/zhuating/arc2-027-ttt-view-ensemble-formal?scriptVersionId=362150112)，私榜未返回。[实际结果](experiments/027/execution/submission-57634436-result-20261026.json)。结束本融合方案的提分晋级，保留023；C23的5→6/34没有转化为公榜收益。隐藏extra覆盖及具体退分根因未确认，不重提或事后换seed。[028 DFS机制小实验](docs/progress/028-nvarc-dfs-throughput.md)已COMPLETE／362511155：稳定候选题DFS调用耗时减1.57%，另一题表面减6.63%却改变候选池；结束固定16-token配方，不晋级正式提交。 [029诊断](docs/progress/029-nvarc-numerics-cache-diagnosis.md)已COMPLETE／362548497：差异在恢复前出现，三个受检恢复点均一致；结束这轮runtime定位，尚无提分证据。

2026-10-27 **030 pilot20 V1／362694905已COMPLETE，官方快照28/28匹配，0.2与0.1最终均6/28、增2丢2净0，结束0.1配方**。oracle7→8未转化为两槽收益；覆盖25→22query，15个共同TTT task的adapter字节不同，不能纯因果归于阈值。不启动40题扩展或提交0.1。[结果汇总](experiments/030/pilot20-v1-summary.json)、[030进展](docs/progress/030-final-week-search-coverage.md)。保留原0.2／B4及023正式30.14。

截至**2026-10-25 10:05北京时间的实查记录**：

- [忠实V1 Notebook](https://www.kaggle.com/code/zhuating/arc2-023-failed-in-aimo-v1-faithful?scriptVersionId=361045383)普通GPU运行已COMPLETE，精确版本 **V1／361045383**。
- 原4task／5query均完成TTT、DFS与重评分；输入匹配评分为 **3/5 query、2/4完整task**。这是公开演示集合结果，不是全量评估或比赛分数，候选oracle未确认。
- **023正式提交57582282已COMPLETE，公榜30.14**，GetSubmission与同队列表一致，绑定V1／361045383。与历史复制30.14持平，未实现提分；私榜未返回。[实际结果](experiments/023/execution/submission-57582282-result-20261025.json)已保存。
- 原V1同候选池B4／R2确认已完成，旧022上的净增1未在新集合重现，结束R2扩展。下一轮依据41个未生成正确候选的query分析规则缺口。最新结果见[025进展§12](docs/progress/025-research-gap-and-next-step.md#12-完整运行及同候选池评分完成)，正式起点见[023进展§9](docs/progress/023-score-recovery-and-selection.md#9-2026-10-25结果回查complete公榜3014正式起点建立)。

## 成绩与证据

| 记录 | 结果 | 限制与入口 |
| --- | --- | --- |
| 正式对照023／57582282 | **30.14**，COMPLETE | 精确V1／361045383；[实际结果](experiments/023/execution/submission-57582282-result-20261025.json)，与历史最好持平 |
| 上一份022／57504054 | **0.00**，COMPLETE | 隐藏执行根因未确认；[结果凭据](experiments/022/execution/submission-57504054-result.json) |
| 020／57443913 | **12.50**，COMPLETE | 27B替换路线；[结果凭据](experiments/020/execution/competition-submission.json) |
| 历史复制提交／314617319 | 用户报告 **30.14** | 未修改公开方案的试提交，不作为自主研究效果；[历史记录](docs/progress/001-baseline.md) |
| 022公开E20 | 4B最终3/27，oracle5/27 | 已开发集合；[4B与既有27B配对](experiments/022/paired-e20-v2/paired-table.md) |
| 027正式提交57634436 | **29.72**，COMPLETE，较023低0.42 | 精确V1／362150112；[实际结果](experiments/027/execution/submission-57634436-result-20261026.json)；未提分、结束晋级 |
| 026公开开发面板16task／23query | base=challenger=4/23，新增2／丢失2，oracle6→9 | 本地配对，结束本续训配方；[完整结果](experiments/026/paired-v1-summary.json) |
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
