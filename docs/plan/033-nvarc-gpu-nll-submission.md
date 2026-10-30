# 033 公开NVARC GPU NLL实现的正式探索对比

## 背景与目标

2026-10-28用户通知032出分，并要求继续优化、完成一次新submission。032原作者V1／312063593实际COMPLETE28.89，精确绑定通过，较02330.14低1.25；隐藏覆盖、数值路径与具体退分原因未确认。不能把该差值归因库版本、模型加载或任一求解参数，也不重提同源版本。

实时zhuating普通GPU余额1846.968配额秒，约15.39分钟4×L4墙钟；10月31日08:00刷新。今日正式名额1。该额度不足新方法完整配对和另一次普通验证；不扩建训练流程或启用已结束配方。按用户本次要求，采用一个已有完成普通运行的公开NVARC计算优化实现，直接做一次正式隐藏评测对比。这不是声称已通过本项目配对提分晋级。

## 固定候选与实现依据

候选：`mikelou1/arc-agi2-lb33-89-minimal-perfpatch` **V2／scriptVersion335137615**，Notebook SHA `ee49384a10a4296dc0bf0670ae67325521f798d5337ed57bee50c8a443a378f4`。以该精确源作为一个整体NLL计算实现因素，不扫描组件组合，不混入TTT、阈值、融合或排序改动。

源差异已归档：

- DFS在GPU以float32计算全词表logsumexp，再仅回传原12个ARC token的NLL；概率分母仍为完整词表，没有对12个合法token重新归一化。
- teacher-forced重评分在GPU提取目标token的log-probability，关闭本阶段未消费的KV cache。
- 模型路径指向当前固定sft139挂载；loader／decoder／starter逐字等于原V1，worker除模型路径外AST相同，原sft139、128步TTT／原scheduler／n16增强、DFS0.2、B4、原调度及预算均保留。

这是一个固定公开实现的系统对比，两个热点和cache使用没有分别隔离，不能将正式分数变化归因到某一行。它与028的16-token预计算／缓存恢复无关，不复活03164步、027融合、026权重或030阈值。

## 证据与限制

CLI current_version_number=2、状态COMPLETE；全部44个输出URL路径中的scriptVersionId唯一335137615。下载的四个实际runtime源与V2源码只差IPython末尾1 LF；已有submission.json，120task／172query格式完整，其中只有原普通4task／5query为非占位，不能称隐藏240题覆盖证明。源Notebook下载没有保存cell输出，不能凭它声称实查过全部训练日志。

本V2的公开分数未确认。历史列表32.22与标题33.89均不能赋给此V2，更不能作本队新成绩。本项目没有运行新的质量／速度配对，oracle未测。实际原权重为16词表，不采用“大Qwen词表传输”叙述夸大收益；GPU logsumexp及use_cache=False可能改变数值、候选或速度，尚未证明逐bit等价或端到端更快。

## 执行与事前决定

最终独立审查精确源、full-test rerun入口、全词表归一化和未改变核心参数；核名额与pending。直接使用用户指定Kaggle CLI、zhuating认证，固定kernel_version=2及submission.json，提交已完成公开版本，不另启动普通GPU。

受理后从队伍列表核真实ref、提交人、队伍及URL中的335137615；结果未知不重复请求。状态及分数如未返回明确待评测。实际公榜严格超过30.14才更新本项目正式最好；否则结束本公开计算配方，不以微调NLL／cache／seed救同一结果。运行错误与能力失败分开记录。无自动监控，用户通知完成后人工接续。
