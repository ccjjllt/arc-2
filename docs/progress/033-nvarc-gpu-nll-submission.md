# 033 公开NVARC GPU NLL实现进展

2026-10-28用户要求出分后继续优化并做一次submission。已先实查032 COMPLETE／28.89、原V1／312063593绑定正确，02330.14仍保留。新方向、资源及停止条件见同编号计划。

**2026-10-29实查57737364 COMPLETE，公榜30.56，私榜未返回，精确V2／335137615绑定通过。相对02330.14提高0.42，相对03228.89提高1.67；保留033为当前正式最好，023保留历史对照。** 这是公开实现整体提交的实际结果，不是自主模型训练成果，也不能将增益归因纯速度、单个NLL热点或已证明全量覆盖。[真实结果](../../experiments/033/execution/submission-57737364-result-20261029.json)。

**新submission57737364已于2026-10-28 21:02:06北京时间受理，精确公开V2／335137615绑定通过，当前PENDING、公／私榜分数未返回。** 这是按用户要求采用固定公开计算优化实现的正式探索对比，不是自主配对已证明提分。

全新子代理分别核032身份／历史差异、资源／可执行路线、公开变体。原版已有known_scores候选重评分缓存，030／031已闭合返回候选全部合法，当前没有依据将新方案说成修复“缺去重”或大量非法输出；未完成分支的浪费仍未量化。

选定已完成公开Minimal Perfpatch V2／335137615。归档源41799字节、SHAee49384a...；source/metadata currentVersion2、44项输出版本唯一335137615。独立代理核原worker（归一模型路径后）与TTT／inference_turbo_dfs／collator／Trainer AST相同，loader／decoder／starter逐字相同；主代理核下载的四个实际runtime源与该V2源只差IPython末尾1 LF。已有submission.json23253字节、SHA62d39594...，172query两槽网格格式有效，其中原普通4task／5query非占位。源与差异记录在 `experiments/033/reference/`，来源在 `execution/`。

当前候选尚无本项目质量／速度配对或新正式成绩；V2公开分数未确认，标题／旧列表成绩不作为本次效果。本轮实际只做CPU源码及来源核对，没有新普通GPU训练或推理。正式预检今日num_allowed_now1、num_total13，公开候选仍COMPLETE；最后独立审查与正式受理将追加记录。

## 最后审查与正式受理

全新最终独立审查通过：source／实际runtime／44项输出URL身份一致，hidden全量actual test、原TTT／DFS／B4／预算保留；NLL使用完整16词表归一化，没有对12个ARC token重新归一化。未发现源码／版本身份运行阻塞。两个热点及评分cache共同变化，未证明候选逐bit相同或端到端更快；不能做单项因果归因。

通过用户指定的Kaggle CLI与zhuating认证直接提交公开原owner精确V2：

| 字段 | 实际值 |
| --- | --- |
| submission | **57737364** |
| Notebook | mikelou1/arc-agi2-lb33-89-minimal-perfpatch |
| kernel_version／scriptVersionId | **2／335137615** |
| 文件 | submission.json |
| 提交人／队伍 | zhuating／ccjjllt |
| 受理时间 | 2026-10-28 21:02:06北京时间 |
| 初始状态 | **PENDING** |
| 公榜／私榜 | **均未返回** |

受理API真实返回ref57737364，随后CLI队伍列表URL精确为 `/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch?scriptVersionId=335137615`，描述、文件及身份均正确，无error_description。今日名额已用1、remaining0，总14。没有另开普通GPU，普通额度仍以上次1846.968秒为最近实查，不断言隐藏rerun不扣额。

回执、初始绑定和接续记录在 `experiments/033/execution/` 与 `ready-v2.json`。身份核完后停止查询，等待用户完成通知，不自动监控、不重复提交。下一次按真实结果是否严格超过02330.14决定保留；当前正式最好仍30.14。
