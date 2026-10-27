# 029 NVARC数值与缓存恢复定位进展

日期：2026-10-26运行，2026-10-27完成复核。对应[029计划](../plan/029-nvarc-numerics-cache-diagnosis.md)。**V1／362548497已COMPLETE；差异在恢复前已出现，受检三个恢复点logits／KV完全一致。结束本轮runtime定位，不复活028；其网格替换的充分原因仍未证明，B4／准确率／提分未测量。** 下文保留历史启动状态，完整结果见文末；正式最好仍02330.14。

## 实际范围与输入

重新派出两个代理：一名负责029运行时观测模块，另一名只读审阅范围与实际包；未沿用旧代理。只取028发生候选替换的`d35bdbdc`，是有意选择的故障复现题，不称新验证集。

输入从028冻结面板逐内容提取，保留原3个query但只测首query前两批增强。query答案字段0。输入2119字节，SHA **af5e81c69f7fcd38e46866c55a3dd4348df25acc7c27e4e9b0ffb2d298583489**，见[manifest](../../experiments/029/input-v1/manifest.json)。

原sft139、TTT rank256／128步、增强seed1、Trainer／PEFT42、eval2、hash27027，单worker。三项比较共用同一次TTT的同一内存adapter，并保留原`torch.inference_mode()`及`for_inference`。固定草稿仍来自同次首批原DFS按原返回次序的第一个合法candidate。无草稿则记录无法执行，不替换题目或草稿。

## 已实现的三项观测

[numerics_diagnostic.py](../../experiments/029/numerics_diagnostic.py)只收集以下固定数据，不替换原DFS或选择新策略：

1. prompt-only两次root logits／prefix KV作为重复参照，再与prompt＋16的root／同一prefix范围KV比较，判断差异是否在任何skip／restore前已经出现。
2. 固定同16token，原逐token两次与一次full-prefill对齐17个logit位置、16token累计NLL和ARC token次序。
3. 由原逐token生成并deep clone的snapshot，在固定深度0／1／13对比原位A／B与清paged恢复A／B，预测同一下一token；同时比较恢复后内部paged前缀与snapshot、返回cache有效长度，未知layout明确为未确认。这三个深度不冒称028实际恢复位置。

报告保存实际fastdecode函数source／code SHA、inference mode、training状态、原始tokens和小logits。KV按层、按64位置分块统计；大cache只在GPU内，不保存GB级文件。主报告仅`029/diagnostic/d35bdbdc.json`，分阶段atomic落盘；cache不支持、时间不足或异常明确写error后抛出，无静默跳过或自动根因裁决。

审阅确认logit对齐、原路径snapshot与重复控制没有明显阻塞。解释边界保持：root有差异证明计算路径已引入变化，不能排除恢复问题，也不能证明它足以导致028网格替换。恢复差异需结合内容、有效长度和自身重复漂移，不直接称缓存数据丢失。C的总体KV比较包含旧prefix与新增slot，prefix另由snapshot↔paged比较提供观测，不把总体差异无条件归为prefix恢复错误。

## 包装与手工核查

[prepare_diagnostic.py](../../experiments/029/prepare_diagnostic.py)复用冻结025／027包装，原`turbo_dfs`、`inference_turbo_dfs`、`calc_scores`和训练collator AST身份均一致。跨cell顶层符号手工核对无未定义引用，`raw_rerun_flag`在setup显式定义，未重复028的变量遗漏。

第三方求解源码及Notebook仅生成`/private/tmp/arc2-029-cache-diagnostic-v1`，实际包71158字节，SHA **57613fc78197fbdf34c24ef9b2e26d343afca94e4554f20c5c18411a50771e4d**。诊断模块SHA **a4b8d78431177f8e28c4e4428039cbdb1cfc66d18699c626c9e3a486f55eacb9**，028 helper仍为 **b658d01533d599cc4074caf3f60bceb0b98cd61b5d87de3cc0e6ea8b9dce2e34**；实际求解源SHA **954e80f9e60e32d60aa564dff6bcda08dcec1b5a91f164e48217eaaa1a539661**。见[package-manifest](../../experiments/029/diagnostic-v1-package-manifest.json)。独立新代理对实际包及嵌入模块核对通过。

## 启动与预算

**23:40:59北京时间**预检可用GPU24570.67秒（6.83配额小时）、reserved0，无pending competition submission。**23:50:43—23:50:47北京时间**临近启动重新核对可用24570.67秒、reserved0，平台接受[arc2-029-nvarc-cache-diagnostic](https://www.kaggle.com/code/zhuating/arc2-029-nvarc-cache-diagnostic) **V1**，kernelId141011301。该id不是scriptVersion；实际运行版本及完整输出尚未取得。

申请Kaggle L4×4，实际只启动一个worker。会话最多2400秒、全局2220秒、进程组硬截止2340秒，含复制、加载、原TTT和收尾。不做额外离线训练，不执行B4／accuracy，不创建比赛提交。见[启动回执](../../experiments/029/execution/diagnostic-v1-launch.json)。接受回执只证明平台接单，启动健康及效果另以实查记录。

## 完成后接续

正常加载后按用户既有方式停止远端查询，无自动监控，等待用户完成通知。入口`/private/tmp/arc2-029-download-completed-v1.py`已准备，仅终态读取；它先核对当前V1与全部cell源，再下载实际输出并从URL取得真实scriptVersion。第三方源码下载到私有tmp，仓库只留数据、报告和日志。

通知完成后先检查TTT128、三阶段闭合、重复控制与budget、全部文件SHA和实际backend，再读取root／17位置／恢复差异。无论结果如何，都不把单题诊断晋级正式方案；若仍无法定位，保留未确认并结束此诊断，不继续扩大成runtime工程或参数扫描。

## 启动模型加载已确认，停止查询

**23:52:32北京时间**核对V1为RUNNING、全部cell源与冻结包SHA57613f…完全一致。真实boot确认输入1task／3query、实际测量首query、原模型config指纹、utility挂载、Python3.11.13以及4×NVIDIA L4各23034MiB，实际只用一个worker。权重复制7267271302字节、69.598秒完成。

截至**23:57:26北京时间**的有界日志流，rank0已启动并完成两片checkpoint100%加载，未见Traceback。TTT完整统计和首个诊断阶段尚未观察，不能写成整个实验完成或已定位根因。平台metadata的lastRunTime异常旧日期不用于本次计时；SSE重复消息不当作重复执行。

见[启动健康](../../experiments/029/execution/startup-health-v1.json)、[源绑定](../../experiments/029/execution/source-verification-v1.json)。现已停止所有远端状态／日志读取，无定时或后台监控；等用户通知完成，再按[接续状态](../../experiments/029/diagnostic-v1-ready.json)读取精确V1及真实输出scriptVersion。当前没有分数改善或根因结论。

## 2026-10-27完成复核：差异早于恢复，结束此轮定位

用户通知完成后，下载入口严格核对当前V1和全部冻结cell源。远端为**COMPLETE**，所有输出URL唯一真实scriptVersion **362548497**。24项数据／源码产物已下载；第三方源码保留`/private/tmp/arc2-029-diagnostic-output-v1`，仓库只留报告、输入和日志。见[终态凭据](../../experiments/029/execution/diagnostic-v1-outcome.json)。

原manifest列出的**15项工件bytes／SHA全部匹配**，实际solver、诊断模块与helper也分别与冻结包SHA一致，见[核查记录](../../experiments/029/execution/artifact-verification-v1.json)。没有读query答案、改草稿或执行新GPU。

### 完成性与实际资源

- 原sft139、实际4×NVIDIA L4各23034MiB、单worker，Torch2.8.0+cu128、CUDA12.8；`inference_mode=True`、`model.training=False`。
- TTT **128步**完成，training_loss0.0099796392；66次诊断forward和三个阶段闭合，无Traceback、OOM或预算截断。
- 实际完整运行**2026-10-26 23:52:42—23:58:25北京时间**，墙钟**342.469秒／5.71分**；诊断本身41.622秒，含首批草稿生成23.182秒。这些计量不代表生产求解吞吐或提分。
- 实际fastdecode函数为`unsloth.models.qwen3.Qwen3Attention_fast_forward_inference`，source SHA **383acf33b9c2eb80ac56461cec1e1f36b5e010974464ffc3ce24e430d22edf91**，code SHA **80ac01431b8f780b3f5f49bd3b14f77cd6176702b7583b084e7334c4e499afb1**。code hash含运行时上下文，只作本次身份记录。

### A：恢复前已经存在计算路径差异

| 比较 | root logits最大绝对差 | 同一prefix KV最大绝对差 |
| --- | ---: | ---: |
| prompt-only A／B重复 | **0** | **0** |
| prompt794／prompt＋16=810 | **0.5** | **2.0** |

prefix KV从第0层起、**36层均发生变化**。这发生在任何DFS skip或snapshot恢复之前，证明本次不同长度／计算路径已经引入数值差异；不能将所有变化归为后续恢复错误。

root四行的ARC第一名均未改变。最大logit差包含低概率token，不能单凭0.5就断言搜索决策或正确答案改变。只替换root logits仍留下36层不同prefix KV，不是已经验证成立的修复。

### B：固定16token，逐步decode与full-prefill不同

原逐token两次重复的17位置logits与全部KV逐元素相同。相对full-prefill，最大logit差**1.0**、最终KV最大差**2.0**。固定路径16token累计NLL末值如下：

| batch行 | 原逐token累计NLL | full-prefill累计NLL | full−逐token |
| --- | ---: | ---: | ---: |
| 0 | 0.040949408 | 0.040948216 | −0.000001192 |
| 1 | 1.318167806 | 1.318677306 | +0.000509501 |
| 2 | 2.682514191 | 2.682514906 | +0.000000715 |
| 3 | 1.780814171 | 1.781465054 | +0.000650883 |

所有**4×17个位置的ARC第一名均未改变**，但完整次序和概率有变化。这里强制沿相同token路径计算，并非声称原DFS会接受所有行的16步；不能把logit差直接当作候选替换的充分原因。

### C：受检三个恢复点未发现错误

原逐token产生的snapshot在深度0／1／13，原位A／B、清paged恢复A／B的**下一logits与返回KV全部0差**，恢复后内部paged前缀也与snapshot逐元素相同。

| 深度 | 原snapshot有效长度 | 下一步返回有效长度 | 已分配capacity（非有效长度） |
| --- | ---: | ---: | ---: |
| 0 | 794 | 795 | 1307 |
| 1 | 795 | 796 | 1308 |
| 13 | 807 | 808 | 1320 |

这些结果支持本次受检的原逐token snapshot清理／恢复行为正确；不证明full-prefill snapshot或所有DFS祖先分支都正确。这三个深度不是028未记录的真实三个恢复位置。

### 与028的可比性及结论

029与028目标prefix SHA相同，但使用了**新TTT adapter**。loss从028的0.0101428293变为本次0.0099796392；同规则首合法草稿也从网格SHA`41db2594…`变为`611e6dfb…`。因此本轮是同一次新adapter内部的机制配对，不是对028旧权重／旧草稿逐字重放，不能据此证明原网格替换的充分因果链。

本轮重新派出的只读审计代理独立复算三阶段数值、重复控制、有效长度和15项工件SHA，结论一致。**完成并结束029，保持028固定16-token配方关闭，保留原023求解方案／30.14。** 不再围绕此问题延伸多轮runtime工程，不以单题机制定位晋级或宣称提分。

下一步回到原NVARC候选覆盖及最终两槽的单因素、同预算配对研究；具体新配置尚未冻结、实验未运行。本轮没有新训练权重、B4／oracle／准确率评分或competition submission。完整数值、身份和边界见[diagnostic-v1-summary.json](../../experiments/029/diagnostic-v1-summary.json)，接续状态已更新为诊断完成且关闭。
