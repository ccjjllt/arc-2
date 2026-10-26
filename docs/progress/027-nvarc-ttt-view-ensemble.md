# 027 原NVARC权重TTT视图集成进展

日期：2026-10-25。对应[027计划](../plan/027-nvarc-ttt-view-ensemble.md)。研究／包装已完成，确认实验待实际启动；无新提分结果或正式submission。

## 当前事实

本轮新派研究、选择、预算／包装三个代理。原始资料确认：普通多视角NLL已在NVARC中，盲目改平均长度或更多views缺乏提分证据；speculative DFS与prefix缓存是后续提速储备，未冒称作者倍率为本项目结果。旧开发候选的三个固定聚合方法仅在E20新增1、开发24题持平，停止参数扫描。

027从原权重新出发，不使用026训练权重或16题救结果。固定TTT增强seed1／2、4GPU先base后extra，第一遍拿全局预算，第二遍只用余量；不采用可能使全量覆盖减半的2＋2并跑。融合规则、前缀风险检查及晋级标准均已在读确认答案前冻结。

[确认输入](../../experiments/027/confirmation-v1/manifest.json)为23task／34query，SHA `f65c90df99dfc3f5bdf2c234342463e73082695140487d82774a21e99cce73ae`。该集合在026训练前已保留，本轮没有按新生成结果或答案重新选题。历史／上游公开接触限制仍在。

18:36:50北京时间只读[预检](../../experiments/027/execution/preflight-v1.json)：GPU available49,783.118秒（13.83配额小时），reserved0；今日num_today0、num_allowed_now1，无pending submission。计划研究上限3小时L4×4约6配额小时，实际消耗以终态为准。

研究包`/private/tmp/arc2-027-v1-confirmation`为108007字节，Notebook SHA `bb1e8320a087a0abb57948077ff15f794c1206f53abdb2b235c2969b7cca87b0`，helper SHA `ccd9d3b4e4b349e25e1f889b66e30b5f0413e60ba1df605644f6c56b1914db31`。源码AST、3项嵌入SHA、loader／decoder与成功025同源、原TTT／DFS参数和四worker队列均已手工核对。独立只读审计未发现启动阻塞，未读取确认gold。代码完成不代表提分。

## 实际启动

**18:41:40北京时间**，平台接受`zhuating/arc2-027-ttt-view-ensemble-c23`研究V1，kernelId140762645，L4×4／断网／10800秒。真实[启动回执](../../experiments/027/execution/confirmation-v1-launch.json)已保存；启动前仍有49,783.118配额秒、reserved0。上传Notebook SHA与审计包相同，未加载026权重，没有创建正式submission。随后的实际启动健康与精确运行版本另以真实日志补充。


正式准备工具`experiments/027/prepare_formal.py`已完成，尚未调用生成、上传或提交。内存构造与CPU分支核查覆盖动态hidden-test／原四题ordinary输入、43200总／42600全局截止、base先写备份和顶层submission、extra异常或finalize异常明确回退base。正式包没有C23输入。正式实际scriptVersion和submission ref均未产生，不能填预期值当事实。

原始主办方[2026计分说明](https://arcprize.org/competitions/2026/arc-agi-2)于本轮查阅，要求每个test output两次exact，最终取各test outputs平均；本轮继续以完整query分母评分，同时另报task均权／完整task。没有把NVARC自带task均权benchmark直接充作公榜。

用户通知完成后，人工下载入口已准备为`/private/tmp/arc2-027-download-completed-v1.py`，仅终态可下载，不启新GPU；先冻完整与前缀预测，再用官方固定commit评分，流程见027 README。


独立代理已静态审阅正式生成器，确认dynamic test输入、C23 payload剔除、42600截止和base异常回退关键路径；[审查记录](../../experiments/027/execution/formal-template-static-review.json)只证明静态实现。实际普通四题、隐藏运行及正式得分均未取得，不把这些写成已运行。


## 启动正常，等待用户完成通知

**18:52:09北京时间**实时SSE确认base rank0—3均启动并完成两片模型加载；rank1首题`1ae2feb7`已完成**128步／epoch1.0 TTT**，training_loss0.0141895581、train_runtime117.9196秒，日志报训练峰值13011MiB。实际4×L4、Python3.11.13、Unsloth2025.9.7／Transformers4.55.4／Torch2.8.0+cu128。模型config SHA及全部上传cell源与冻结包一致，未观察到Traceback或CUDA OOM。早期status API曾返回空对象，不能以它判作失败；真实boot与TTT日志证实已执行。平台lastRunTime异常日期仍不用于本次计时。

见[启动健康记录](../../experiments/027/execution/confirmation-v1-startup-health.json)。这只代表base正常启动，extra尚未观察，完整覆盖、确认净增和公榜收益都未取得。按用户既有等待习惯，停止所有远端状态／日志读取，无定时监控，等待用户通知完成。

下一次先执行已准备的完成下载入口，按027 README先冻结完整三策略和6／12／18任务前缀，再读固定官方gold评分。确认通过后生成正式包、普通四题核查、绑定真实scriptVersion与submission.json，再在已有授权下直接提交；不重新询问是否提交，也不在确认集改规则。公榜实际结果才决定是否超过30.14。

## 确认完成及精确数据版本评分

用户通知完成后，研究V1／**362101452为COMPLETE**。两阶段return0、无硬超时；23/23 task各完成128步TTT，34/34 query均获得搜索。base133/136批（32query×4、1×3、1×2），extra132/136（32×4、2×2），缺少的批次均为`8b7bacbf`触原1200秒task软检查，并非全局截止。各阶段事件闭合，无Traceback／OOM／非有限loss。训练／推理日志峰值分别13012／15646MiB。

实际4×L4，原固定sft139、TTT增强seed1／2、Trainer及PEFT seed42、eval增强seed2、hash27027，原TTT／DFS／B4均保留。实际时间**18:43:48—21:00:37北京时间**，墙钟**8208.981秒（136.82分）**；base3940.312秒、extra4110.561秒、收尾22.760秒。extra最后任务早于全局截止约2497秒，完整扫完23题。

### 归档与冻结

703项工件SHA／字节核查通过，含666个候选pickle。原B4从有序scored记录重放，两臂34个query均复现；执行源码仅IPython补一个末尾换行，冻结helper SHA仍为`ccd9d3b4e4b349e25e1f889b66e30b5f0413e60ba1df605644f6c56b1914db31`，无算法变更。

base raw／scored均416记录、235个query-grid，28query有候选；extra均440记录、236个query-grid，29query有候选。raw与scored网格逐query完全一致。融合对28个有真实base候选的query全部保留第一槽；仅2个base空query自然fallback extra。决策为11个首槽同解保留base第二、16个不同extra首选替换第二、4个两边空、1个extra空、2个base空；实际14个query的输出发生变化。

完整三策略于13:12:48 UTC冻结，6／12／18个extra完成任务前缀于13:12:53 UTC冻结，随后才读取query答案。独立审计复核三个前缀的task_end时间次序与集合一致，三个目标均达到。

### GitHub与Kaggle数据版本差异修复

首轮固定GitHub commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`只匹配30/34：`4a21e3da_0`、`f560132c_0`为query输入不一致，`faa9f03d_0/1`为demo不一致。它们未被计错、剔除或当作完整分母；该首轮报告单独保留为`scoring-v1-f328/`，不作为晋级最终结果。

随后直接下载本届Kaggle官方`arc-agi_evaluation_challenges.json`与`arc-agi_evaluation_solutions.json`，challenge SHA **e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8**与本次推理原始源完全相同，solutions SHA **84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1**。23题完整demo／全部query输入逐内容一致，**34/34全部匹配**；原30个已匹配gold哈希全部一致。没有改动任何预测或选择规则。正式评分采用这份精确官方比赛快照，数据留`/private/tmp/arc2-027-kaggle-scoring-snapshot/`，记录见[来源身份](../../experiments/027/execution/official-kaggle-scoring-snapshot.json)。

### 固定方案结果

| 指标 | base原B4 | extra原B4（诊断） | 预先固定ensemble |
| --- | --- | --- | --- |
| 最终两槽exact query | **5/34** | 8/34 | **6/34** |
| 完整task全query正确 | 3/23 | 6/23 | 4/23 |
| scored／raw候选oracle | 6/34 | 10/34 | 两臂union10/34 |
| 相对base新增／丢失 | — | 3／0 | **1／0，净增1** |

固定融合新增`a6f40cea_0`，没有丢失。部分补题前缀25／50／75%的最终exact分别5／5／6，净增**0／0／1**，没有回归。task均权准确率从16.67%到21.01%，作为补充诊断；不能当作正式公榜分。额外一臂8/34没有被事后改选成正式方案，仍遵守读答案前唯一固定融合规则。

预定条件全部满足：最终净增>0、34输入匹配、base覆盖完整、前缀无回归、无实现故障。**晋级原四题普通验证及正式提交流程**，不继续在C23改规则／挑seed。新增1个公开确认解并不保证公榜超过30.14，须等真正submission结果。

完整结果见[汇总](../../experiments/027/confirmation-v1-summary.json)、[晋级决定](../../experiments/027/promotion-v1.json)、[精确快照评分](../../experiments/027/scoring-v1-kaggle/score-report.json)、[逐query账本](../../experiments/027/scoring-v1-kaggle/query-ledger.jsonl)、[归档SHA](../../experiments/027/execution/artifact-verification-v1.json)与[原B4重放](../../experiments/027/execution/manual-B4-reproduction-v1.json)。

## 正式包生成与普通验证启动

实际正式包`/private/tmp/arc2-027-formal-v1`为84734字节，SHA **3c8551fbf483bd1a4755722a0b7585408c2ed789f5d8ae5b3ed618f8a4200919**。新独立代理审查了实际生成包：无C23输入、hidden动态完整test、ordinary原4题／5query、canonical SHA `f609f61bdc823802318d005eef7fd6400a2a1c6a4b9f9c93305e1de116979bc1`，原求解源与已确认研究包相同，helper未变；base先保存备份与顶层submission、extra异常回退base，未见阻塞。

21:41:48北京时间正式预检可用GPU33,305.546秒（9.25配额小时）、reserved0；今日num_today0、num_allowed_now1，无pending。研究启动前的完整预检文件在正式预检时被刷新，初值另从原工具返回与launch回执转录为明确标记的`research-preflight-v1-summary.json`；当前完整正式快照保留`formal-preflight-v1.json`，不冒称转录件是原始响应。

**21:44:30北京时间**平台接受`zhuating/arc2-027-ttt-view-ensemble-formal` **V1**，kernelId140787141，L4×4、断网。普通原四题验证的API session timeout为7200秒，模型代码内hidden预算仍43200／global42600秒；这次请求是普通执行，尚非competition submission。回执见[formal-v1-launch](../../experiments/027/execution/formal-v1-launch.json)。

21:48远端RUNNING、V1，全部cell源与正式冻结包一致。真实boot确认`original_ordinary_four`、4task／5query及canonical输入SHA正确，未带C23；模型config指纹、4×L4与原utility正常。普通完整输出、实际scriptVersionId和submission ref仍待取得，不能把kernelId当scriptVersionId。

后续用户通知普通验证完成时，先执行`/private/tmp/arc2-027-download-formal-completed-v1.py`读取真实终态、下载工件并取得实际输出scriptVersion。确认顶层submission.json、run-state的prediction SHA与base／extra完整流程、动态输入身份及固定融合一致；必要时使用同一Kaggle快照给原四题评分作诊断。再次核对源cell、名额和pending后，固定`kernel_version=1`、绑定真实scriptVersion及`submission.json`，在已有用户授权下直接`competition_submit_code`。不要改用latest、不要把普通验证当已正式提交，不再询问是否提交。


只读正式完成预检入口已准备为`/private/tmp/arc2-027-formal-completion-preflight.py`，普通完成后用于核对固定V1的cell源、全部输出URL中的实际scriptVersion、顶层submission.json以及临近提交的名额／pending／队伍。当前未执行此完成预检，未创建competition submission。仅source V1上传与普通验证已启动不等于已提交比赛。


## 普通验证加载正常，停止查询

**21:55:31北京时间**，正式普通V1的base rank0—3均完成两片模型加载，日志未见Traceback或CUDA OOM。动态输入仍为原4题／5query，canonical SHA与023参照一致；实际source cell与SHA3c8551...冻结正式包完全相同。见[普通启动健康](../../experiments/027/execution/formal-v1-startup-health.json)。此时尚未观察首题TTT完成、extra或最终输出，不能把加载正常写成普通运行完成。

按用户既有习惯，确认正常后停止所有远端查询／日志流，无定时监控，等待用户通知普通验证完成。下一次直接按[正式接续参数](../../experiments/027/formal-v1-ready.json)下载、核对真实终态和source／submission.json／实际scriptVersion，再刷新限额、pending与队伍并提交固定V1；actual scriptVersion与submission ref仍为null，不编造、不询问重复许可。

## 普通验证完成与正式提交接受

用户再次通知“跑完了你看看”后读取实际终态。**普通V1／scriptVersionId362150112为COMPLETE**，两阶段均return0，无硬超时或错误回退；输入mode为`original_ordinary_four`，4task／5query及canonical SHA `f609f61bdc823802318d005eef7fd6400a2a1c6a4b9f9c93305e1de116979bc1`与原四题一致。

- 实际4×L4，base／extra各4/4题完成128步TTT，5/5query各跑完整20/20批DFS。全阶段起止闭合，未见Traceback、OOM或非有限loss；训练／推理峰值13150／17109MiB。
- 普通运行实际区间**21:45:54—22:38:22北京时间**，总**3148.232秒（52.47分）**；base1475.436秒、extra1447.027秒。实际消费早于普通会话7200秒上限。
- `active_prediction_policy=frozen_base_first_extra_first_ensemble`。5个query两遍的首选均一致，冻结规则正确保留base第二槽，因此融合输出字节等于base；这不是错误回退。
- 顶层`submission.json`、`027/submission-ensemble.json`与base备份均为14616字节，SHA **15ead46eddffb243d17db96ffd55e409953780a3c61b10224f2c6b9d11198c04**。191项工件身份通过，包括153个候选pickle；两臂原B4及融合逐query复现，源码只差IPython补末尾换行，helper仍为冻结SHA。
- 用同一Kaggle官方公开快照匹配全部5个query后，base／extra／ensemble均**4/5，3/4完整task**。这是反复使用的原四题运行诊断；此次普通运行没有额外融合收益，正式晋级依据仍是之前预定的C23净增1，不能把4/5称为新确认或公榜80%。

初次精确源预检将版本附在slug后返回403，随后尝试`version_label='1'`返回404。已改为读取该自有Notebook的元数据并**严格断言currentVersionNumber=1、全部cell源与冻结正式包一致**，再核对全部输出URL的唯一scriptVersion为362150112；实际提交调用仍显式`kernel_version=1`，没有提交latest或重新上传版本。两次错误只是只读源请求格式问题，不是自动审批拒绝，也没有创建提交。

**23:16:27北京时间**完成预检：普通COMPLETE、源一致、顶层submission绑定362150112、今日num_today0／num_allowed_now1、无pending、队伍ccjjllt，精确版本此前未提交。独立新代理只读审计无阻塞；未改数据、权重、融合规则或再跑GPU。

**23:18:02北京时间，平台接受正式submission 57634436**：

- 账号zhuating、队伍ccjjllt、比赛`arc-prize-2026-arc-agi-2`。
- Notebook `zhuating/arc2-027-ttt-view-ensemble-formal`，**kernel V1／scriptVersion362150112**。
- 代码SHA **3c8551fbf483bd1a4755722a0b7585408c2ed789f5d8ae5b3ed618f8a4200919**，提交文件`submission.json`。
- 接受回执见[competition-submission-v1.json](../../experiments/027/execution/competition-submission-v1.json)。

23:19:08进行一次GetSubmission和同队提交列表核对，均找到57634436，列表URL精确绑定362150112、队伍和提交文件一致；**状态字段尚未返回，public／private score均未返回**。不得把空状态写成COMPLETE或把普通4/5／C23净1当作公榜提分。见[初始绑定](../../experiments/027/execution/submission-57634436-initial.json)。正式最好仍为023的30.14，027是否超过它等待真实评测结果。

按用户偏好，回执与绑定核对后停止所有远端查询，无自动化或重复提交。用户通知评测结束时，人工运行`/private/tmp/arc2-027-read-submission-57634436.py`读取实际状态／分数与精确绑定，再更新结果；不要重提同一版本、不要从普通Notebook状态推断隐藏评测终态。

本节结果入口：[普通完整汇总](../../experiments/027/formal-v1-summary.json)、[普通评分](../../experiments/027/formal-scoring-v1-kaggle/score-report.json)、[普通工件核查](../../experiments/027/execution/formal-v1-artifact-verification.json)、[完成预检](../../experiments/027/execution/formal-completion-preflight-v1.json)、[正式接续状态](../../experiments/027/formal-v1-ready.json)。
