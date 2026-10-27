# 030 赛前一周研究审计与搜索覆盖：进展

日期：2026-10-27，北京时间。计划见[030计划](../plan/030-final-week-search-coverage.md)。

## 已完成的工作

- 三个新的只读子代理分别核验历史原始工件、实时比赛／Notebook信息、作者开源与论文；主代理复核025—029材料、原V1的剪枝／batch／重评分实现和论文Table 3。
- 直接访问Kaggle官方页面，取消公开代码列表不可访问数据源排除；目前可见最高仍是Failed in AIMO33.89，分数绑定精确V1／312063593。没有证据证明存在可复现的更高公开Notebook。
- 今日公榜第16名69.44带金牌标记，第8名75.14；最终私榜未知。不能把33.89当作临近金牌的分数。
- 只读CLI／SDK确认当前配置身份普通GPU余额0，下次刷新10月31日08:00北京时间；无新GPU、训练、正式提交、自动监控或云资源支出。
- 独立复核已存query ledger及events：025首槽14、仅第二槽1；17个oracle正确中14个首见第1批、3个首见第3批。此为既有运行工件的CPU审计，不是新模型结果。
- 建立030研究设计：原sft139／TTT／B4／runtime，只比较DFS累计路径门槛0.2与0.1；预算计入后续视角、query和重评分。首轮与扩展均需最终两槽净增，失败不扫阈值。

## 来源与实查记录

### 比赛与公开Notebook

2026-10-27约10:00—10:06北京时间的浏览器页面读取：

- [Code列表](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?competitionId=133469&sortBy=scoreDescending&excludeNonAccessedDatasources=false)：第一Failed in AIMO33.89；Learned from AIMO、Minimal Perfpatch、twin-lb33-verbatim、Reproduce NVARC、NVARC+等32.22。
- [精确V1](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593)：Version1 of2，33.89与V1绑定、sft139 BF16 V1、Apache-2.0、普通26m11s。
- [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)：Tufa Labs88.06；第8 Dawid Kopiczko75.14；第16 Lashen Learning69.44／gold；第17 ảc69.17／silver。只记今日快照，不预测最终名次。
- [Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)、[Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)、[Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)：11月3日07:59北京时间最终截止；每日1份、最终最多2份；隐藏240task；每个test output取两槽最佳exact后微平均。网页说明不是服务端代码审计。

### 额度

现成认证的Kaggle CLI／SDK只读结果摘录，**本节是工具结果转录，不是新保存的原始响应文件**：

```json
{
  "checked_at_utc": "2026-10-27T02:05:35.559478+00:00",
  "quota_refresh_time": "2026-10-31T00:00:00",
  "gpu_used_seconds": 120810.666,
  "gpu_reserved_seconds": 0.0,
  "gpu_total_seconds": 108000.0,
  "is_pay_to_scale_enabled": false
}
```

剩余按`max(0,total-used)`为0。只代表当前配置账号；未查询其他账号。TPU仍有额度，但没有为此移植当前NVARC运行时。普通会话额度与正式隐藏rerun的扣减关系本轮未核验。今日提交列表最新仍为027／57634436 COMPLETE29.72、023／57582282 COMPLETE30.14；未见pending。正式可提交次数需临操作重新读取。

### 历史结果与新审计结论

| 轮次 | 已运行结果 | 对本轮的约束 |
| --- | --- | --- |
| 023 | 正式30.14 | 仍保留；没有复现作者33.89 |
| 025 | B4=R2=15/58，raw=scored oracle17/58 | 排序最多补当前池2个；41个失败池外，但其原因不能直接归为缺训练 |
| 026 | base=challenger=4/23、增2丢2，oracle6→9 | 结束配方；oracle增加不是晋级依据 |
| 027 | C23固定融合5→6/34，正式29.72 | 结束方案；C23缺base第二槽独有正确，且隐藏extra覆盖未知 |
| 028／029 | 候选变化与数值定位；未测B4／准确率 | runtime支线结束 |

压力样本`2ba387bc_0`及晚批首次正确`d35bdbdc_1`、`7ed72f31_1`、`7b0280bc_0`来自025已存gold指纹与候选／事件匹配，不是对新方案的评分。025的40题＋027的23题＋历史排除57题覆盖当前120题；只说明没有当前账本可认证的新保留池，不能声称120题全部读过答案。

### 研究来源

本轮直接阅读[Product of Experts v2 Table3](https://arxiv.org/html/2505.07859v2#S5.T3)、[MARC原论文](https://openreview.net/pdf?id=asgBo3FNdg)及作者实现、[2026-10-09预算研究](https://arxiv.org/abs/2610.13660)、[2026-10-23候选接口研究](https://arxiv.org/abs/2610.32235)。详细适用边界见030计划§4。外部效果没有转写为本项目效果。

## 当前状态与接续

**完成：研究审计、下一轮设计与文档。未完成：030实验包装、面板与预算manifest冻结、GPU配对、扩展回归、正式挑战包及新分数。** 本轮仅文档变更，没有新增测试文件、框架或CI，也没有创建后续自动任务。

接续先按030计划完成CPU准备，临GPU执行读取实际额度；现有条件下把10月31日刷新后的窗口用于单因素配对。每个实际实验仍须记录数据版本／划分、权重／代码SHA、配置、硬件、耗时、两槽与oracle指标及产物。若无法在预算内给出正向证据，保留023，不为赶时间盲投。
