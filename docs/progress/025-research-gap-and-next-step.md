# 025 研究现状与外部资料核验

日期：2026-10-25，北京时间。对应[025评审与交接](../plan/025-research-gap-and-next-step.md)。本轮回答当前研究缺陷、争奖差距和下一步选择；没有启动GPU、训练、推理或比赛提交。

## 1 已完成的核验

- 阅读001—024计划／进展中的研究结果，重点复核020—023原始计分和提交记录。
- 确认最新正式57582282／V1 361045383为COMPLETE、公榜30.14；私榜未返回；这与历史复制30.14持平。
- 复核022 E20、023 R2回放和既有27B配对：3/27、oracle5/27、R2为4/27，27B独有正确1个。R2证据来自022候选，不能假称已经在忠实原V1上验证。
- 在浏览器直接读取Kaggle官方排行榜、按Public Score排序的Code列表、精确Failed in AIMO V1、Overview与Rules。本轮无需登录，也未操作报名、队伍、提交或发布。
- 查阅NVARC冻结训练配置、TRM、Land、TraceViT、ARC-TGI等原始资料，区分2025／2026、公评／半私有／比赛私榜及算力条件。

## 2 官方页面观察

以下是2026-10-25上午的页面读取结果，后续榜单可能变化；不是API导出的全榜。

| 来源 | 实际观察 |
| --- | --- |
| [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard) | 第1名88.06；第8名75.14；第16名68.75带gold medal；第17名68.06带silver medal。公榜约50%数据，最终使用另一半 |
| [公开Code列表](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?competitionId=133469&sortBy=scoreDescending&excludeNonAccessedDatasources=true) | 顶部Failed in AIMO 33.89；Learned from AIMO、Minimal Perfpatch、NVARC+等32.22。保留默认数据源访问筛选，不称穷尽最高 |
| [精确Failed in AIMO V1](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593) | Version 1 of 2；Public Score 33.89，链接绑定V1；L4×4；26m11s；qwen3_4b_grids15_sft139 BF16 V1；Apache 2.0 |
| [Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview) | 12小时、断网、L4×4／96GB、2倍配额消耗；最终截止11月2日23:59 UTC；入场／合队10月26日23:59 UTC |
| [Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules) | 每日1份、最终2份；排名进步奖前8；Innovation Prize按六项等权；截止后7天提交开放材料；Bonus按达85%的前五队分配 |

与旧记录相比，本轮新增了**官方当前可见金牌边界和排名奖金位置**，不能再仅以“实时金牌线未确认”避开竞争差距。当前68.75与30.14相差38.61；公榜边界仍非最终获奖门槛。

规则页称27.5万美元方案说明奖为Innovation Prize，Overview／主站仍称Grand Prize；主站Bonus叙述也与Kaggle规则不完全一致。025采用当前Kaggle规则的具体表述，不擅自推断材料开放要求的法律优先级或为第三方产物重新许可。

## 3 研究证据与纠错

1. **公开数据边界仍按实际配置解释。** NVARC生成脚本构造`arc2_evaluation6`包含公开eval答案，但冻结[sft_mg.yaml](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/ARChitects/sft_mg.yaml#L120)明确把它放在val路径；[run_sft.py](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/ARChitects/run_sft.py#L110)分开加载train和val。不能从README列表或生成脚本推出答案直接参与SFT。验证／checkpoint选择及evaluation规则派生材料仍使公开集不属于独立盲测。
2. **2025 NVARC的TRM结果依配置变化。** 不概括为TRM永远退分，也不称最终冠军必须包含TRM。025仅继承作者最终4B单独运行和组合收益不稳定的结论。
3. **新增TraceViT资料。** 2026-07论文报告18M、ARC2公开评估24.3%，使用8×A100 80GB、每题100 epoch TTT、510视角。其对象与中间状态监督可作中期研究；并不支持当前L4窗口内换架构。来源：[论文](https://arxiv.org/html/2607.29586v1)。
4. **Land系统提供候选多样性和选择机制的证据。** 历史半私有72.9%、38.99美元/题，不是本届Kaggle离线可提交成绩，也不是2026-10-25最新榜首。来源：[论文](https://arxiv.org/html/2606.31543v1)。
5. **ARC-TGI是数据资产，不是已证实提分。** 461个生成器中包含公开测试来源；任何续训使用均须保留来源和family划分。来源：[论文](https://arxiv.org/html/2603.05099v1)。

## 4 本轮产物与下一步

新增025计划与本进展，更新文档索引。求解算法、冻结Notebook、权重、现有候选和比赛状态均未改动。

当前建议仍是原V1候选上的一次B4／R2确认，同时保存完整候选池和阶段覆盖。下一轮只按证据选择候选排序、搜索覆盖、针对性续训或有限27B中的一个方向。公开确认不能代替正式隐藏评分；R2确认、新调度、续训和新27B实验均**未运行**。

本轮只做文档和只读核验，没有新的实验硬件、运行时间或提分指标；旧实验的配置、时间和产物沿用原进展文档。本次手动核对文档链接、指标算术和修改范围，不新增测试文件或自动测试体系。
