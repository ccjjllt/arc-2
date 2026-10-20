# 020 提分验证与真实提交交接

日期：**2026-10-20，北京时间**。对应[020计划](../plan/020-score-validation-and-real-submission.md)。复核代码基点 `92eb44c69ae1d3ea98b13cf26fc742cd0a593778`。

**本轮完成研究交接、公开题面划分、提交代码审阅和既有日志的CPU复算。求解代码尚未修改，D24/C24 GPU实验未运行，未新增比赛提交，没有新分数。** 用户最新要求优先使用ccjjllt、qingyu1232、jiyujia余量，zhuating完整配额留作后备；本轮未取得四号新的实时配额快照。

## 1 研究判断

019已经把公开集文件打通，但输出覆盖不等于解题效果。新路线优先比较同一个27B的“独立候选”和“有限CPU修正加选择性反馈”，并把已完成的冻结Notebook真正送入代码竞赛评分。暂停019低预算直出的原样全量扩跑，不等待旧账号刷新，不重开模型选型或大训练。

历史30.14对应用户指定的复制方案版本314617319，精确源码尚未核实；不能称当前可复现基线。013有可核实代码和正式提交57103498，实际公榜25.69。新27B策略的公开集改善与超过历史最高分需要分别验证。

## 2 已完成的证据复核

原始输入为 `experiments/019/submit-pilot/kaggle-v1/vprobe/run.json`。本地CPU通过请求发出/完成时间重算在途数量和时间积分，没有启动模型：

| 项目 | 复算结果 |
| --- | ---: |
| S1请求数 | 172 |
| 最大在途 | 8 |
| 平均在途 | 7.8768 |
| 平均单请求时间 | 172.0057秒 |
| 请求覆盖的波次时间 | 3755.964秒 |
| 原日志聚合生成吞吐 | 144.12 token/s |
| 安装加载总时间 | 662.5秒，已含安装262.5秒 |
| S2有usage的返回 | 8份，共72,253 token |
| 这8份的执行时间 | 全为0秒 |
| 最终覆盖 | 120 task / 172 query；160真实预测、12占位、0不同双槽 |

原始日志SHA、复算与用户复核来源保存在[019-manual-review.json](../../experiments/020/evidence/019-manual-review.json)。

**用户附件的0/158、8份程序CPU重放0/8、aa四邻域改八邻域的结果，本轮没有独立重新评分或重放。** 已将附件原文持久保存为[user-review-20261020.txt](../../experiments/020/evidence/user-review-20261020.txt)。目前缺少0/158逐query评分账本，不能把附件描述写成本轮独立实验。

只读源码核对确认：正式Notebook仍按排序选择challenge、S2等整波才执行、F不含原程序、按query重复生成、来源优先级压过示例证据等问题仍在。Notebook内嵌的四份源码与019目录一致，这些问题不是外部源码已修而Notebook过期。另发现Notebook起始时钟未传驱动、实际HTTP发出前缺停新请求判断、完成回调异常被吞掉；均列入020实施范围。

## 3 新划分与数据边界

新建 `experiments/020/cohorts/` 下5份JSON：

- [development-d24-manifest.json](../../experiments/020/cohorts/development-d24-manifest.json)与对应[题面](../../experiments/020/cohorts/development-d24-challenges.json)：24 task / 36 query，非ARC1，按输入规模分层；用于开发。
- [confirmation-c24-manifest.json](../../experiments/020/cohorts/confirmation-c24-manifest.json)与对应[题面](../../experiments/020/cohorts/confirmation-c24-challenges.json)：从007比赛training留出块选24 task / 26 query，其中20个非ARC1 task / 22 query和4个历史来源为旧ARC1 evaluation的task / 4 query；冻结后只确认一次。与D24来自不同官方划分，不作为隐藏榜分的无偏估计。
- [split-notes.json](../../experiments/020/cohorts/split-notes.json)：输入来源、选择规则、排除集、SHA及每query指纹。

没有读取C24 solutions；题面只有demo答案和query输入。C24与019 pilot无task交集，上游预训练覆盖未知，不称模型未见集。D24已进入019推理，不称留出；其是否全部属于附件158可评分query尚未确认，不能直接继承“0/36”。

019 pilot实际上已推理旧主C6公共evaluation的3 task / 5 query和备用3 task / 4 query，旧“确认集未动”表述已失效。新C24不沿用旧C6名称，不将公开开发题重新包装为未见验证。

## 4 资源和真实提交操作

已只读核对当前CLI为2.2.4，当前身份为 **qingyu1232**，不是zhuating。该身份当时提交额度返回 `numTotal=8 / numAllowedNow=1`；这不是zhuating快照，不能转记。

当前CLI源码中，`competitions submit -k owner/slug -v VERSION -f submission.json` 进入代码提交接口；普通 `kernels push` 仅为Save & Run。交接要求绑定精确Notebook/version，取得submission ref并回查列表后，才记“已正式提交”。没有在本轮调用提交或修改最终候选选择。

只读身份、CLI分派与提交额度证据见[account-cli-readonly.json](../../experiments/020/evidence/account-cli-readonly.json)。另核实push的`--timeout`会写入版本会话时限，是否影响隐藏重跑未知；020用Notebook内部时钟限制普通4小时，不盲目保存平台4小时限制。

普通GPU原单会话基准为D24 3小时、C24 3小时、最终Save & Run 4小时，共10小时；按2.3预算倍率约23配额小时。按最新资源指令优先分配到ccjjllt、qingyu1232、jiyujia，取消单个账号需29小时及旧账号各留6小时的条件，zhuating作为项目后备。必要的B/C分片新增安装加载和收尾另计，不增加原定task/调用上限。隐藏重跑预算独立列为11小时；其普通配额计费方式本轮未确认。

## 5 已更新文档与待执行事项

本轮新增020配对计划/进展，更新README入口及019中关于并发、加载时间和确认题暴露的结论。工作区原有 `experiments/020/evidence/competition-pages-20261020.json` 是0字节，未覆盖，也未当作有效来源。

待执行顺序：提交链局部修复 → D24同模型对照 → 固定政策后C24一次确认 → 同一冻结版本公开全量Save & Run → 正式代码竞赛submit → 记录实际榜分。若有正确解但尚不足以支持超过30.14，诚实标为首个27B计分实验；若候选无正确解或提交链仍有确定性错误，停止扩跑并记录未提交原因。

本轮没有新增测试文件、测试目录或框架，没有GPU推理、训练、发布或正式提交。上述交接不构成模型提分或奖牌证据。

交接终审：文档本地链接全部可解析，cohort产物SHA一致，D24/C24无task交集，GPU题面中query output数量为0，`git diff --check`通过。只读复核提出的非预期rerun标记问题已修进计划，CLI提交命令已对当前版本帮助及实现核对。见[人工核对记录](../../experiments/020/evidence/handoff-manual-check.json)；这些是文档和输入核对，不是新模型实验。

## 6 用户更新资源优先级

用户说明ccjjllt、qingyu1232、jiyujia目前均有配额，要求先用这些剩余配额。已同步020计划和README，zhuating从默认主账号改为后备；不把用户的余额说明写成已完成API查询。

执行时先给三个优先账号中能承载最终4小时完整运行者预留对应阶段额度，其余余量承担D24/C24。优先较早刷新、能完成有用工作块的余额；必要时按完整task拆分，保留统一冻结配置、全部query和配对调用，C24全部输出完成后才统一评分。正式Save & Run仍必须单账号、单版本完整运行，提交命令和metadata不再写死zhuating。

不为消耗额度重复已完成实验；无法承担有用工作块的尾数不空跑。启用zhuating的条件是三个优先账号都不能完成下一个必要工作块或最终完整会话，并记录当时余额依据。此前[人工核对记录](../../experiments/020/evidence/handoff-manual-check.json)中的29小时是初版单账号规划快照，现已由本节和计划第2节替代。

本次仅修改资源安排和交接文字，未切换账号、未启动GPU、未发布Notebook或提交比赛。
