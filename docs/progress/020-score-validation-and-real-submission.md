# 020 提分验证与真实提交交接

日期：**2026-10-20，北京时间**。对应[020计划](../plan/020-score-validation-and-real-submission.md)。复核代码基点 `92eb44c69ae1d3ea98b13cf26fc742cd0a593778`。

**当前状态：C24已完整运行并统一评分，原冻结Direct1PlusLine为17/26 query正确（65.38%）、16/24 task全部query正确，详见[§25 C24确认结果](#25-c24完整确认1726及实际成本)。正式比赛提交 `57443913` 精确绑定 `ccjjllt/arc2-020-deep-direct-submission` v1 / scriptVersionId `360461153`，15:59:07Z回查仍为PENDING，尚无新榜分。C24评分晚于正式提交，时序见[§24](#24-正式比赛提交-57443913-已受理)与§25；本地确认成绩不是超过30.14或拿牌的证据。**

**此前阶段快照（已由§18、§21–25更新）：原D24及CPU增强均0/34，短反馈F/R均0/4，等总上限深首稿0/8。CPU可把e376的demo从1/3修至3/3，但当时query仍错。此时完整F尚在运行，C24与新正式比赛提交尚未运行。** 四号配额已实查，按用户优先消耗旧账号余量；实际实施与结果见§7以后。下文§1–6保留此前交接阶段的记录。

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

## 7 用户授权后实施与D24启动

实施起点为已提交的 `d45dea4`，用户明确指令“开始做”。已在 `experiments/020/submit/` 实现task级程序生成、P0/R/F逐结果执行与原子写盘、完整原程序反馈、M0/M1候选隔离、demo证据排序、有限邻域/排序修正、严格正式输入选择及首cell共享时钟。生成器[prepare_notebook.py](../../experiments/020/prepare_notebook.py)负责内嵌源码和SHA，固定程序参数12288总token/8192思考、C8、种子5101/5103/5105。

人工核对覆盖传输滚动补位、实际HTTP前截止、静默连接中断、回调异常记录、task全部query写入、M0/M1隔离、完整反馈和输入歧义失败；没有新增测试框架或测试文件。macOS既有限制无法安装，执行器保持fail-closed，真实执行及AA邻域修正放在Kaggle同会话、模型安装前验证。见[模块核对](../../experiments/020/evidence/module-manual-check.json)、[传输核对](../../experiments/020/evidence/transport-manual-check.json)、[AA仅demo预检材料](../../experiments/020/evidence/aa-demo-preflight.json)。

19:40北京时间实时余额（均reserved=0）：ccjjllt 5.8372配额h、qingyu1232 8.11984h、jiyujia 1.84691h；zhuating随后确认30h完整未用。各号余额不能合成单号时长，最终4h会话需要约9.2配额h，三旧号均不足，因此保留zhuating承担最终完整运行的后备位置。证据见[账号配额账本](../../experiments/020/execution/account-quota-preflight.json)。

D24按每长度层2/6题拆为jiyujia的6task/10query（2850秒）和qingyu1232的18task/26query（7200秒），合计10050秒会话上限；task无交集，完整覆盖原24task/36query，每task最多三个研究调用。分片和源码在开机前固定，全部分片结束后才统一评分。见[分片清单](../../experiments/020/cohorts/shards/d24-sharding-manifest.json)。

已成功push `jiyujia/arc2-020-d24-ji` **v1**，日志已越过执行器/AA修正前置检查，安装requirements与固定vLLM wheel均返回0，正在加载模型。通过此前置检查后，成功push `qingyu1232/arc2-020-d24-qingyu` **v1**，11:54Z查询为QUEUED。二者均是研究Notebook，不是正式competition submission；当前尚无正确率结果。启动凭据保存在[d24-launches.json](../../experiments/020/execution/d24-launches.json)。

qingyu分片第一次推送被服务器明确以“source必须小于1MB”拒绝；将Notebook载荷压缩后成功，未产生失败运行版本。两分片内嵌求解源码SHA逐文件一致，压缩只影响封装；[包核对记录](../../experiments/020/evidence/package-manual-check.json)确认ji包829017字节、qi包102050字节及内嵌SHA均正确。

本地评分器[score_frozen_outputs.py](../../experiments/020/score_frozen_outputs.py)支持固定原清单分母和分片合并；先冻结所有预测SHA再读取固定官方版本的对应task，核对完整demo与query输入并保留逐query账本。当前未读取D24/C24真实query答案，未运行实际评分。

用户进一步强调研究优先、减少门控和测试、提高GPU利用率。已取消“净增2query且跨2task才确认”的硬条件；后续围绕真实正确解、候选差异和成本推进，只保留输入、落盘和截止所需核对。下一会话的调度先补发网络请求再执行CPU回调，减少执行/写盘占据请求空位；在跑D24包保持原冻结SHA。利用率与吞吐由实际生成记录及GPU采样评估，不另开利用率基准会话。

## 8 D24首片结果与短机械反馈研究

`jiyujia/arc2-020-d24-ji` v1已完成，产物在[首片目录](../../experiments/020/d24-jiyujia-v1/vprobe/)。总墙钟1963.937秒（32.73分钟），其中安装加载768.5秒。12次模型调用共116,697 completion tokens；P0波96.08 token/s、R波99.27 token/s，客户端实际6路。模型就绪后的236个GPU采样中，四卡平均忙碌率99.57%–99.58%，显存峰值均20,191 MiB。该指标是nvidia-smi忙碌时间比例，不是SM占用率或理论吞吐上限；无需为“恢复并行”另开实验。见[运行摘要](../../experiments/020/execution/d24-jiyujia-runtime-summary.json)。

AA仅demo机制在真实Linux中复现：原程序0/3，局部搜索第2个变体达到3/3，整个搜索0.729523秒。这是已知开发题上的廉价修正机制，未读取该题query答案、未计新增正确query。

首片12份程序中，6份正常返回demo网格但规则不正确，3份全部demo执行异常，2份截断，1份AST拒绝；全部示例通过为0/12。9份获准执行的程序在25个demo上16次合法、9次异常，仅1个demo exact。P0-2c将二元坐标读作三元组；P0-195混用坐标/组件，且含`for r,c in range(H)`；R-e871引用未定义`out`。P0-e871仍夹有一行未使用的`from collections import Counter`，被执行器拒绝。

原近似阈值下6题全部走R，CPU搜索与F实际均为0，故M0/M1在本片完全相同；**不能据此称反馈/局部修正效果为零**。两策略均覆盖7/10 query，其中3个query有不同双槽，3个占位。query正确率留待整个D24输出冻结后统一评分，当前未读官方query答案。

据此追加一个有明确问题、低新增token成本的研究：复用P0-2c、P0-195、P0-e871原代码，比较一次真实错误反馈F与独立短R。3task/4query/9demo在[mechanical清单](../../experiments/020/mechanical/manifest.json)冻结，选择仅依据运行/解析错误。历史首稿成本28,595 token独立保留；新F/R各最多3次请求，每次4096总token/1024思考，seed分别5103/5105，最多新增24,576 token。反馈包含原代码和真实错误；不重新生成首稿、不用query答案选case。

已push `ccjjllt/arc2-020-short-mechanical-feedback` v1，最长2700秒；这是研究Notebook，尚未正式submit。启动前ccjjllt实查可用19041.160729秒（5.2892配额h）、reserved=0；与19:40快照的差额不全部归因于本项目。记录见[启动凭据](../../experiments/020/execution/mechanical-launch.json)及[开机前余额](../../experiments/020/execution/pre-mechanical-quota.json)。这笔追加研究单列成本，不把它伪装成原D24三小时设计的免费分支。

下一会话源码还把P0回调返回的R/F追加到原队尾，在所有P0已经入队的前提下与尾部P0重叠运行，避免阶段屏障造成空闲；实际运行时按分钟打印GPU采样均值。D24原包未修改，研究方法和运行版本分别记账。

## 9 等总token上限的连续思考对照

只读检查首片reasoning，发现多份在半句话、逐格核对或尚未确定的假设处结束，final随后直接写代码，如R-8e末尾“Maybe shapes are”。尚未用tokenizer逐条计数，不声称每条刚好8192；现象支持检查思考硬上限是否在规则成形前切换。不能据此宣称更长思考会提分。

在未读取本轮D24/C24 query答案时，从原D24每个输入长度层按固定SHA规则选2题，冻结[6task/9query清单](../../experiments/020/deep-p0/manifest.json)：`3e6067c3、5961cc34、800d221b、b6f77b65、de809cff、e376de54`。新臂仅一次P0，24576总token/16384思考、seed5101；旧臂直接复用已经固定运行的P0+R，各12288/8192。两者每task最大completion allowance同为24576，实际token和时间仍须单列；深首稿只评分自身实际双槽，不合并旧臂制造四候选成绩。

最小研究开关`--p0-only`使首稿返回后不追加CPU/F/R，代码仅手工compile，未增测试文件或框架；该开关只用于有cohort的研究，不进入正式submit。保持同一题面、P0提示、seed、权重及选槽；运行调度版本差异在包SHA中保留。

已成功push `ccjjllt/arc2-020-deep-p0-equal-cap` **v1**，会话上限3600秒，初查QUEUED；Notebook SHA `ceed978dcf0d523074b82e754499e9995901e8e75cc4447fba8ef285c6ad5abd`。开机前12:46Z余额4.822267配额h、reserved0；短反馈会话已运行，实时used增加。该新臂最多147456新token，预计含装载约40分钟；新增装载与配额成本单列。见[启动记录](../../experiments/020/execution/deep-p0-launch.json)及[余额快照](../../experiments/020/execution/pre-deep-p0-quota.json)。当前未返回结果，未声称增益。

原先拟等待所有追加臂返回后再读D24答案。按用户减少门控、尽快确认效果的要求，现改为：D24原两片、短反馈和CPU v1已完成并固定预测，先对这些真实产物评分。DeepP0与CPU定点修正v2的输入、源码和远端版本在读答案前已固定且正在无网隔离运行，不能接收本地答案；它们的预测SHA将在返回后固定，明确标注发生在D24首次读答案之后，不称“所有追加臂预测在gold前已固定”。C24尚未启动，仍不读答案。

## 10 短机械反馈实际结果

`ccjjllt/arc2-020-short-mechanical-feedback` v1已COMPLETE，产物在[结果目录](../../experiments/020/mechanical-ccjjllt-v1/vprobe/)，摘要见[mechanical-runtime-summary.json](../../experiments/020/execution/mechanical-runtime-summary.json)。墙钟959.778秒，其中安装加载751.5秒（已含安装311.5秒）；6次生成207.31秒、17989 token、聚合86.77 token/s。生成窗口41组四卡采样均100%忙碌，含启动空闲的全部采样均值97.619%。本次小块研究加载占比很高，后续不重复开机验证同类机械错误。

| 臂 | 新token | 获准程序 | 全部demo可运行 | 全demo通过 | 合法query |
| --- | ---: | ---: | ---: | ---: | ---: |
| F | 7659 | 3/3 | 1/3 | 0/3 | 1/4 |
| R | 10330 | 1/3 | 1/3 | 0/3 | 2/4 |

计入28595历史首稿，M1部署逻辑成本36254 token、M0为38925；整次共享研究P0历史加新F/R共46584，不把复用首稿记作免费。query正确率仍未评分。

F确实删除了旧错误，但2c又产生未定义的`color1/color7`，195的坐标遍历仍导致IndexError；e871已能执行全部demo，却将外框错误地填成实心矩形，0/3 exact。**短反馈改善了完整交付并省token，目前没有正确性增益证据。** 不据3/3获准继续无条件追加反馈，后续重点转向首稿规则推理。

## 11 同模型近期外部线索与证据等级

10月20日20:54北京时间通过官方Kaggle API复查，新增材料见[研究更新JSON](../../experiments/020/evidence/qwen27b-research-update-20261020.json)。10月18日[Qwen3.8讨论](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/750546)中，原作者报告NVARC与27B融合后约35→45；另一回复报告整套双尝试中位70K token/task、公开集约45%，并称装不进12小时。**均为没有绑定Notebook版本的作者自报，非本项目复现成绩；未取得可核实的完整12小时高分融合实现。**

本项目首次取得[公开程序修复Notebook v3](https://www.kaggle.com/code/abisheksrivastav/arc-prize-2026-program-repair-v1/3)源码（实际10月7日已存在，非今日新成果）：按task一次通用程序、实际执行demo与全部query、携带旧程序和真实反馈；配置首稿49152总token、修复16384、C8，没有额外8K思考上限，首轮分配剩余墙钟65%。该版本榜分及成功12小时运行未核实；未执行外部代码、不更换当前运行栈。它支持研究“首稿深度和全局预算比例”，不能当作已证明的提分配方。

历史30.14的精确源码再次限时查询仍未取得：显式版本标签及精确脚本下载地址404，未拉latest冒充，也不由404断言原版本被删除。历史提交记录仍存在，当前未修改最终候选选择；SDK不返回final选择标记，不能称已独立确认哪两份被选中。

## 12 复用固定程序的纯CPU局部修正

D24两片现已全部返回。原路由均走R，不能用M0=M1判断局部修正无效。为直接回答这一问题，新增手动研究脚本[replay_local_repair.py](../../experiments/020/replay_local_repair.py)，复用同一SP/LR/SL；不走近似阈值，分别对已有P0/R代码做最多32变体、30秒的demo筛选。**这是每个初始程序的搜索上限，不是每task全部CPU路线仅30秒**；原程序demo执行和所选程序query执行分别计时。

24task/36query、48个原调用槽中有39份非空获准代码；9份无程序仍保留分母。输入、来源run SHA和原代码在[cpu-repair/input.json](../../experiments/020/cpu-repair/input.json)固定，无query答案。CPU政策保留已有原候选，再加入按demo选出的最多两份程序结果，以同一demo排序和网格去重填两个槽；最终没有四槽oracle。

已成功push `jiyujia/arc2-020-cpu-repair-replay` **v1**，明确`enable_gpu=false`，无模型、wheel、数据挂载和安装，不占GPU配额。Notebook SHA `594b4ac7e848908a68feca26b828816cf038a0e299ed6a239fee4f6714171b4a`；[启动记录](../../experiments/020/execution/cpu-repair-launch.json)保留四份源码与输入SHA。运行与评分尚未完成；它与深首稿并行，全部预测固定后才读取D24答案。

完成后补记：CPU v1实际墙钟17.087326秒，4CPU/4线程；39份代码中14份生成并尝试23个变体，涉及10task。没有任何task增加demo exact，全demo通过仍0。CPU baseline与原D24 M0合并后的36query逐槽完全相同；CPU增强只在21897d95_0增加一个不同的第二候选，不同双槽9→10，覆盖仍30/36。搜索各task墙钟之和19.877秒、全部task墙钟和64.602秒均非会话墙钟或CPU核时；新增模型调用、token和GPU使用均0。见[CPU运行摘要](../../experiments/020/execution/cpu-repair-runtime-summary.json)。

## 13 固定双槽真实评分：当前配置未产生新解

首次D24官方答案读取始于2026-10-20约13:12Z；原两片、CPU v1及短反馈所有预测均先固定SHA。来源为官方ARC-AGI-2的固定revision `f3283f727488ad98fe575ea6a5ac981e4a188e49`，完整核对demo，再按query输入匹配，不按ID/下标硬对齐。

| 固定政策 | 计划query | 匹配query | 最终EM@2 |
| --- | ---: | ---: | ---: |
| D24 M0=P0+R | 36 | 34 | **0/34** |
| D24原M1（本次全部路由R） | 36 | 34 | **0/34** |
| D24无near过滤的CPU增强 | 36 | 34 | **0/34** |
| 短机械反馈F，含同一旧首稿 | 4 | 4 | **0/4** |
| 短独立R，含同一旧首稿 | 4 | 4 | **0/4** |

完整账本见[D24报告](../../experiments/020/scoring-d24-v1/score-report.json)、[逐query记录](../../experiments/020/scoring-d24-v1/query-ledger.jsonl)和[短反馈报告](../../experiments/020/scoring-mechanical-v1/score-report.json)。D24的abc82100_0、b6f77b65_0因官方query输入不匹配单列，不计错、不虚称0/36。M0/M1候选池oracle同为0，当前没有“正确候选被选槽遗漏”的证据。以上均为公开开发集结果，不是榜分。

该结果否定原样扩跑当前8K思考策略的依据；暂不消耗完整公开运行及每日提交名额重复已知负配置。继续中的深首稿研究回答连续思考问题，定点CPU修正回答具体对象提取/单元素取值问题；它们的代码与输入在首次读gold前固定，输出返回后单独评分，C24仍未读取答案。

## 14 深首稿与语义修正：demo改善尚未转成query正确

DeepP0 v1全部完成：[运行摘要](../../experiments/020/execution/deep-p0-runtime-summary.json)。6次调用113272 token，总墙钟1942.441秒，安装加载725.7秒、生成1216.5秒、93.11 token/s，生成GPU采样均值99.585%。相同6题原TwoShort实际116823 token，新臂为其96.96%，不是实际成本增加。5份stop中4份程序获准，但3份全部demo执行失败（前沿循环不终止、三元patch被读成六元矩形、归一化收敛至0后仍除0）；596含import sys拒绝，b6耗满24K仍未交出程序。最终合法query仅1/9。

实际固定双槽[DeepP0评分](../../experiments/020/scoring-deep-p0-v1/score-report.json)为**0/8**，TwoShort同为0/8；b6 query0仍因输入不匹配单列。停止原样把程序思考从8K翻到16K扩成全量。完整代码的表示与执行错误是直接瓶颈，也仍存在反复读格、规则未收敛。

两项定点CPU假设在v3运行：e376仅延后used标记，使demo 1/3→3/3；f931仅将受单元素检查保护的next(iter(vals))换成list(vals)[0]，仍1/5。[真实query评分](../../experiments/020/scoring-demo-repair-v3/score-report.json)合计**0/3**，e376自身0/1。v2曾因封装时误替换内层base64解码而失败，未执行候选；v3只修封装，输入与候选源码SHA不变，未增加GPU。

将延后used归纳为不依赖变量名、task ID或颜色的受限AST变体后，完整D24纯CPU v4也已运行并评分：[AutoCPU报告](../../experiments/020/scoring-auto-cpu-v4/score-report.json)**0/34**。它能复现e376的demo全过，却没有新query；不能把机制正确称为提分。后续driver对M1所有有支持结构的P0/F/R运行有限CPU搜索，不再以near阈值拦截CPU；每份程序最多32变体、含query应用30秒，最多两份模型程序的CPU成本分别记录。近似阈值只负责F/R路由，尚未由C24确认。

e376的输入分析另暴露了规则欠定：demo斜线共同对齐下左端，而query同向线共同对齐上右端。现固定方向端点的程序恰好拟合demo；“从输入识别共同对齐端”的更统一候选已由demo与query输入提出，未读取官方答案网格，作为开发假设单列，不称确认结果。

## 15 复用旧P0/R，只新增完整反馈与MTP实用推理

原D24的24题全部路由R，未测试完整8K思考F；短机械F只有1K思考，不能替代原研究问题。现从原两片选择全部19份获准P0（仅按code存在，不按query是否正确），冻结[19task/27query清单](../../experiments/020/full-feedback/cohort-manifest.json)。历史P0为177458 token、旧R为185983 token，含2份没有合格代码的R仍计其消耗；不重新生成P0/R。新臂仅19次F，12288总token/8192思考、seed5103，使用真实代码和demo反馈，并继承当前有限CPU修正；CPU已全过demo时如实说明通过并探索有示例依据的另一假设，不捏造错误。

已push `ccjjllt/arc2-020-full-feedback-mtp2` **v1**，会话4200秒，Notebook SHA `2836a9c2ec3921874a70eccfdbbe634a13ca58ce386785f90f3b7de36f08629f`；[启动记录](../../experiments/020/execution/full-feedback-launch.json)。开机前13:30Z ccjjllt可用3.460543配额h、reserved0，70分钟按2.3余量约2.683h，可承载。当前RUNNING，尚无结果。

固定模型v1本身包含22个mtp权重键和约477MB的mtp.safetensors；固定vLLM commit `1f7427bc0ad31fdad1b0933c77ea5acae15da2e0`有Qwen3_5MTP实现。本轮不换模型/轮子，传`--speculative-config {method:mtp,num_speculative_tokens:2}`及`--mamba-cache-mode align`。在有用的19次反馈中记录实际吞吐、显存和失败，不另开利用率基准。**MTP是否净加速尚未实测；完整F、CPU修正和解码配置同时变化，不能把未来差异纯归因F。** 如形成候选，C24使用同一冻结配置确认，不能带研究专用frozen_initial进入正式提交。

## 16 绕开程序实现错误的长思考直出

DeepP0直接损失5/6题的合法query，其中3份是代码错误，故另用同一固定Deep6输入研究充分思考后的直接网格输出，而非继续扩同配置程序。每task一次请求，包含全部demo与所有query输入，输出完整`query0/query1/...` JSON；不执行模型代码、不伪造demo证据。完整有效JSON触及length仍保留；缺失、残缺或非法网格记失败。49,152总token、40,960思考、xhigh、seed5101、MTP2，最多新增294,912 token。

已push `qingyu1232/arc2-020-deep-direct-mtp2` **v1**，5400秒；Notebook SHA `9b9487e07f88b9822b1e834c0eaac06e9ae9f937fb03d82ed550d66be1993126`。使用qi的旧余量，发布后13:40Z实查剩6.095789配额h、reserved0，90分钟按2.3预留3.45h；[启动记录](../../experiments/020/execution/deep-direct-launch.json)与[实时余额](../../experiments/020/execution/deep-direct-launch-quota.json)。当前RUNNING、仍在装载，未出正确率或速度。该臂同时改变表示、思考预算及解码配置，是能力/成本探索，**不是等成本的格式消融**；不能与019 low/2K思考/4K总预算的0/158混为一项配置。D24是已评分开发集，题面中没有答案；C24仍未启动、未读答案。

## 17 输入不变性修正回收一个已开发query

CPU v5使用相同e376候选，仅把按方向固定端点改成：比较各平行段首/末端沿方向投影的跨度，保留更对齐的一端；仍按空间中间线确定长度。此假设来自demo与query输入的对齐结构，不读取官方query答案网格。原固定端点与新端点候选都通过3/3 demo，按同一demo证据保留两个不同query槽。

纯CPU运行墙钟2.970358秒，新模型调用为0；[固定双槽评分](../../experiments/020/scoring-input-aligned-v5/score-report.json)在2task/3query上从0/3到**1/3**，新增仅e376（1/1），f931仍0/2。这是D24已开发题上的真实新增query，**不是泛化确认、更不是超过30.14或奖牌证据**。候选尚未被实现为对任意新task自动发现的完整策略，不把手工选定单题程序冒充可直接提交方案。后续可在C24读答案前对全部task尝试这一通用线段规则、只按demo选候选，检验其互补性；不再围绕本题反复加GPU。

## 18 完整反馈无新增，MTP已有真实接受记录

FullF v1已完成：[运行摘要](../../experiments/020/execution/full-feedback-runtime-summary.json)。同19task/27query，新F共179685 token，生成1025.17秒、175.27 token/s；总1847.37秒，安装加载796.1秒（含安装316秒）。生成窗口GPU采样均值99.386%，峰8/平均6.354实际HTTP并发。17/19份F获准、9份所有demo可运行，但F全demo通过仍0；F合法query15/27。

[固定评分](../../experiments/020/scoring-full-feedback-v1/score-report.json)：M0与M1均**0/26**，abc的一个输入不匹配单列。M0覆盖25/27、M1覆盖22/27不是正确率；无新增正确query，停止原样追加程序反馈。旧同题P0+R消耗363441 token，P0+F为357143 token；整项研究含共享旧P0/R加新F共543126 token。

102条真实SpecDecoding日志累计接受105505／草拟148298 token（71.14%），mean acceptance length中位2.42。它证明MTP实际在工作；175.27并非同提示开关A/B，2.42也不是实测速率倍率。固定权重/源码依据另见[MTP证据](../../experiments/020/evidence/mtp-support.json)。

## 19 CPU窄规则覆盖与直出确认冻结

将输入对齐假设写成[通用线段规则](../../experiments/020/aligned_line_solver.py)，不含task ID或颜色常数；所有demo exact且至少一个demo非identity才保留。原样CPU遍历D24仅支持e376（1task/1query），完整公开120task/172query也只支持此题；C24的24task/26query支持0题。三个结果均保留源码、输入SHA和逐题demo记录于execution/aligned-line-*-result.json，未读取C24答案。C24零覆盖表示无法用该集合估计该窄规则的增益，不把开发题1个正确解外推。

完整程序路线均为负，现冻结单次长思考直出做一次绝对正确率和成本确认：[C24政策](../../experiments/020/execution/c24-policy-freeze.json)。冻结时Deep6直出尚未返回；不空等另一个pilot再消耗启动时间。所有24题按同一prompt、seed5101、49152总token／40960思考、xhigh、MTP2运行一次P0，两个槽重复同一预测；M0/M1完全同源，**不是增益比较**。实际服务tokenize长度决定必要的上下文裁剪，预留128 token并保留最终网格空间。固定全部分片之后才评分，不用C24答案调政策。C24来源仍是官方training／007留出，不称底座未见集。

按全部题面长度排序再index mod3切8题ccjjllt／16题qingyu1232，所有query跟随task；分别总时间盒3600／6000秒，研究启动加载另计。14:03Z余额cc2.42105、qi5.34130（Deep6仍在运行）、ji0.74273配额h。ji尾数不足以覆盖一次有用长推理与加载，不为耗尽尾数空跑。用户最新授权必要时可使用zhuating，但仍优先三旧号。

## 20 当前外部挂载故障与正式运行预算修订

C24 cc v1的发布响应把原wheelhouse列为invalid；取回run.json确认**0.498秒mount_missing、模型调用0**。源/输入政策未变，未读C24 gold。显式固定`/1`重试v2也被服务拒绝；不能把成功push或COMPLETE冒充实际推理。CC/QI读取该资产metadata与v1文件均403，而既有公开控制资产可正常读取；正在查找同字节恢复方式，不换运行栈。已经挂载原资产的Deep6直出仍在正常生成，未中断。

13:57Z独立实查zhuating剩27.22118配额h，且ARC3另有该账号的运行，不能继续以“30h满额”记账。正式普通演练不再机械保留4小时，packager新增明确`--public-seconds`（默认2700秒）与`--rerun-seconds`（默认39600秒）；普通演练初始化完整题面但仅部分实际推理，不称公开全量研究。45分钟包括安装和收尾；须按实际长请求耗时保证有真实完成。隐藏是否扣普通quota仍未确认，不能写成免费。正式包、正式保存运行和新比赛提交均尚未发生。

## 21 长思考直出与CPU互补首次得到正确信号

DeepDirect v1已完成：[实测摘要](../../experiments/020/execution/deep-direct-runtime-summary.json)。6task/9query，246358 token，总2857.084秒，安装加载968.4秒、生成1888.5秒，聚合130.45 token/s；实际峰6/平均4.952在途，GPU采样99.733%。MTP接受145556／草拟201516 token（72.23%），不是加速倍率。实际token分别为同题TwoShort的2.109倍、DeepP0的2.175倍；不是等成本格式消融。

[固定预测评分](../../experiments/020/scoring-deep-direct-v1/score-report.json)为**2/8**，同题TwoShort为0/8；新增为3e的两个query，共1task，另1query输入不匹配单列。5/6题交出网格、8/9 query实际有预测；800耗尽49152后最终通道仍写分析，没有JSON。服务确实收到了40960 thinking预算，但没有reasoning token分项，不能宣称已量证硬边界或该设置保证答案交付。

CPU规则与单次直出的互补在首次DeepDirect评分前已固定到[C24冻结修订](../../experiments/020/execution/c24-policy-freeze-v2.json)：每query最多一个CPU格与一个P0格，两槽去重后都保留。所有task统一按demo决定CPU适用，不按query正确与否选用。[实际合并评分](../../experiments/020/scoring-deep-direct-line-v1/score-report.json)为**3/8**，增加此前已开发的e376，共2task；[合并产物](../../experiments/020/deep-direct/frozen-DeepDirectPlusLine.json)不是四候选oracle。该结果仍为已开发小样本，不是C24确认、榜分或超过30.14的证据。C24此规则适用0题，不能由该集合估计其增益。

## 22 无GPU恢复固定运行依赖，C24两片已启动

不再盲推失效原数据集。原requirements原文在014已保存并核验为3900B／SHA ebb525ba…d133；恢复脚本仅从官方源下载manifest列出的相同193个wheel，逐SHA与字节核对，不重新解析版本、不安装或执行下载代码。为避免本地下载再上传约10GB，使用私有CPU carrier、internet=true、无任何比赛/model/dataset输入，输出保留原slug/wheels目录。三号均193/193、退出0：cc v1/script360450255、qi v1/script360450340、zhu v1/script360453835，恢复分别36.232／33.623／73.161秒。见[版本、源码与文件证据](../../experiments/020/execution/runtime-recovery-carriers.json)。未使用GPU、未创建公开数据集或改其他项目作业；本地备份下载已停止。

C24两份实际版本为**ccjjllt/arc2-020-c24-deep-direct v3**及**qingyu1232/arc2-020-c24-deep-direct v1**，分别挂自己carrier输出，算法、权重、运行库字节、prompt和种子相同。cc原v1/v2仅挂载失败、模型调用0；未混入预测。[启动记录](../../experiments/020/execution/c24-direct-launch.json)保存实际Notebook/source/input SHA及时间盒。14:39Z两片均RUNNING，cc首cell14:39:14、14:39:20固定vLLM wheel哈希匹配并开始安装；qi日志尚未接通，不把RUNNING直接写成模型已开始生成。全部C24预测返回并固定后再读答案；当前尚未读C24 gold，也未发起正式比赛提交。

## 23 正式Notebook普通保存运行已完成，尚未比赛提交

截至**2026-10-20 15:10:37Z（北京时间23:10:37）**，[正式保存核验](../../experiments/020/execution/final-save-verified.json)确认 **`ccjjllt/arc2-020-deep-direct-submission` v1 / scriptVersionId `360461153`** 的普通Save & Run状态为 **COMPLETE**。版本号来自远端metadata，scriptVersionId由全部输出下载地址中的同一`/kf/`版本绑定；远端全部cell源码与冻结包一致，下载源码SHA与内嵌源码一致，输入资产也与冻结包匹配。Notebook为private、internet=false，挂固定FP8模型v1、比赛输入和本账号的CPU恢复carrier。

普通分支执行`ordinary_cpu.py`，时间盒300秒，读取公开evaluation的120task/172query。实际仅通用线段CPU规则适用1task/1query，产出**1个真实CPU预测、171个占位**；这不是1/172正确率，未读gold、未评分。算法墙钟0.479352秒、会话记录0.934801秒，未执行模型推理或模型验证，也未调用GPU计算；GPU配额实际扣费未测，不能据此写成零配额。已独立核对完整题目键、query数量和提交文件格式，日志退出码为0；这次COMPLETE只确认普通分支保存了格式有效的产物。

隐藏分支已冻结为`vdriver020_submit.py`，内部会话预算 **41400秒（11.5小时）**，配置为direct、单次P0、line complement、总token上限49152／思考预算40960、MTP2。`vdriver020_submit.py`、`run_reasoning_probe.py`、`solve_program.py`、`submit_lib.py`、`local_repair.py`与`aligned_line_solver.py`这六份核心源码的SHA，均与正在运行的两个恢复依赖后的C24包一致；普通保存专用的`ordinary_cpu.py`另列。平台 **43200秒** 仅核实为[发布请求](../../experiments/020/execution/final-save-launch.json)经CLI写入`session_timeout_seconds`的请求证据，远端metadata/status接口不返回该字段，尚无独立回读，也不是隐藏运行实际持续时间。

**本节时点尚未调用比赛代码提交接口、尚无新submission ref或新榜分；C24两片仍在运行，C24 gold未读取。** 后续先完成固定C24预测与确认，再记录实际比赛提交及隐藏运行结果；普通Save & Run不替代该步骤。

前文按阶段保留：页首“最新状态”实际是完整F仍在运行时的旧快照，后续已由§18、§21–23更新；其中“C24未运行”已由§22替代。§4–6的普通4小时/隐藏11小时及§20的普通45分钟部分模型演练、隐藏39600秒，均是此前方案，本次实际冻结版本采用本节的普通CPU保存与隐藏41400秒配置；§20“正式保存尚未发生”也已由本节的COMPLETE事实替代。上述状态更新不把任何普通保存覆盖数转记为正确率，也不改变历史研究结果。

## 24 正式比赛提交 57443913 已受理

已通过`competition_submit_code`向 **`arc-prize-2026-arc-agi-2`** 发起真正的代码竞赛提交，取得 **submission ref `57443913`**；提交账号为`ccjjllt`，Notebook为 **`ccjjllt/arc2-020-deep-direct-submission` v1 / scriptVersionId `360461153`**。随后在比赛提交列表回查到同一ref，列表URL精确绑定该scriptVersionId，已满足正式提交凭据，详情见[competition-submission.json](../../experiments/020/execution/competition-submission.json)。

| 事件 | UTC时间（2026-10-20） | 实际证据 |
| --- | --- | --- |
| 发出请求 | 15:28:57.054 | 固定owner、Notebook v1及`submission.json` |
| 平台记录提交时间 | **15:28:59.223** | 北京时间23:28:59.223；直查和提交列表日期一致 |
| 收到创建响应 | 15:29:00.508 | 返回ref `57443913` |
| 回查提交及列表 | **15:30:18.743** | 均为 **PENDING**，public/private score及错误字段均为空 |

正式版本使用已冻结的 **Direct1加CPU线段补槽**，隐藏内部会话预算 **41400秒（11.5小时）**；普通保存分支及平台43200秒请求的证据边界仍见§23。PENDING表示提交已受理待处理，不证明隐藏GPU已开始、完成或取得分数，11.5小时是配置预算而非实际运行时长。

创建提交时，**C24尚未评分，qingyu1232后片仍在运行，C24 gold未读**；当时的正确信号仅来自§21的Deep6开发样本：Direct1为2/8，加已开发CPU规则为3/8，不能称未见泛化或超过历史30.14的证据。为并行推进隐藏计分与C24尾部运行，执行顺序由§23的“先完成C24确认再提交”改为：正式政策已冻结后先提交该精确版本，C24继续原政策，待全部固定输出齐备后仍完整评分。C24本身也不能证明隐藏榜分超过30.14，因此此次不将其评分完成作为提交前提。

本次命名为**首次Qwen3.8-27B正式计分研究**，尚不称提分或奖牌结果。回查确认历史30.14的submission `52072735`仍在列表；本轮**未手动修改final选择**，但未独立核实当前最终候选标记，不能声称历史提交或新提交当前已被选中。后续分别追加C24确认结果与本ref的实际榜分或失败信息。

## 25 C24完整确认17/26及实际成本

C24两个原冻结分片均已完成：**`ccjjllt/arc2-020-c24-deep-direct` v3，8task/8query**；**`qingyu1232/arc2-020-c24-deep-direct` v1，16task/18query**。两片task无交集，合并精确覆盖原清单的24task/26query；源码、权重、运行库、prompt、种子及Direct1配置一致。最后一片结束后，在**15:51:54.776Z**固定全部预测文件SHA，再读取答案并统一评分，见[预测冻结记录](../../experiments/020/scoring-c24-v1/prediction-freeze.json)、[逐query账本](../../experiments/020/scoring-c24-v1/query-ledger.jsonl)与[评分报告](../../experiments/020/scoring-c24-v1/score-report.json)。未修补无效网格或用答案重新选槽。

| 固定来源组 | Query exact / 固定分母 | EM@2 | 全部query正确的task |
| --- | ---: | ---: | ---: |
| non_ARC1 | 14/22 | 63.64% | 13/20 |
| old_ARC1_evaluation | 3/4 | 75.00% | 3/4 |
| **完整C24** | **17/26** | **65.38%** | **16/24** |

全部26个query均与固定官方版本的demo及query输入匹配，**不匹配或未解决项为0**；未因失败、占位或多query题删减分母。实际模型预测覆盖为**21/26，另5个占位**；评分器的`valid_queries=26`只表示最终文件中的网格格式有效，包含占位，不能写成26次有效模型预测。5个占位来自4task：cc的`984d8a3e`虽正常stop，但一行宽18、其余宽17；qi的`aa62e3f4`、`b7fb29bc`和双query题`c3fa4749`均耗满49152 token，最终通道继续分析而没有预测JSON。所有这些query保留在26的评分分母内。

固定CPU线段规则在全部24题上**适用0题、贡献0个query**；M0/M1文件逐片及合并后完全相同，两槽没有不同候选。因此本次是Direct1冻结政策的绝对准确率确认，**不是M0/M1增益比较，也不能由此估计CPU规则的增益**。集合来自官方training／007项目留出，其中旧ARC1 evaluation指历史来源；底座预训练及上游微调覆盖未知，不称模型未见集，不把17/26外推为隐藏榜分或超过历史30.14。

实测成本见[完整C24运行摘要](../../experiments/020/execution/c24-combined-runtime-summary.json)、[cc分片](../../experiments/020/execution/c24-ccjjllt-runtime-summary.json)和[qi分片](../../experiments/020/execution/c24-qingyu1232-runtime-summary.json)。每片使用L4×4、TP4、客户端上限C8；24task各一次模型调用，共**680338 completion tokens**，无usage缺失。

| 分片 | Completion tokens | 账号会话墙钟 | 安装加载（已包含安装） | 生成墙钟 | 聚合生成吞吐 |
| --- | ---: | ---: | ---: | ---: | ---: |
| cc v3 | 232342 | 2332.318秒 | 719.5秒 | 1612.55秒 | 144.08 token/s |
| qi v1 | 447996 | 3998.190秒 | 786.6秒 | 3211.15秒 | 139.51 token/s |

两账号累计会话墙钟为**6330.508秒**，累计安装加载1506.1秒、累计生成4823.7秒；不能将累计墙钟当并行历时。按运行记录，首片14:39:14.452Z启动至后片15:49:39.312Z结束，实际日历跨度为**4224.860秒**。以累计生成时间作分母的合并吞吐为141.04 token/s；这不是单会话端到端吞吐或配额扣费测量。CPU规则两片合计仅0.041247秒。

真实HTTP在途峰值各8，按实际发出至完成的时间积分，cc平均5.507、qi平均5.573；四卡生成期GPU忙碌采样均值分别99.664%、99.685%，合并3808个卡采样为99.678%。这些是忙碌比例，不是正确率或理论算力利用上限。MTP的482个日志区间合计接受390276／草拟579658 token，接受率**67.33%**；这是观察到的草稿接受比例，**不是加速倍率**，本轮没有同提示MTP开关对照。

评分发生在§24的**15:28:59.223Z正式提交之后**：创建提交时C24后片仍运行、未评分；后片沿原冻结政策完成，全部预测固定后才评分，未用此次答案改已提交v1。[正式提交记录](../../experiments/020/execution/competition-submission.json)的最新只读回查为**15:51:36.568Z，ref `57443913`仍PENDING、public_score为空**；该回查早于本次预测冻结与评分。C24已获得本地确认结果，比赛隐藏计分仍待返回，二者分别记录。

## 26 交付收尾与下一轮研究依据

15:59:07 UTC再次通过官方get_submission查询，`57443913`仍为**PENDING**，public/private score与error均为空。已将本次状态追加到[原始受理记录](../../experiments/020/execution/competition-submission.json)的`final_handoff_status_check`，未再次创建提交；当前不能宣布超过历史30.14或取得奖牌。研究GPU会话均已结束，正式评分排队；不把PENDING写成已运行隐藏GPU。

对DeepDirect开发失败800做了无需GPU的精确重分词：[预算证据](../../experiments/020/evidence/deep-direct-800d221b-token-budget.json)。固定Kaggle模型v1/version_id966079的tokenizer.json为12,809,320B、SHA `0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3`，本地已有tokenizers0.22.2；add_special_tokens=False时reasoning为40959、content为8192，补一个`</think>`恰为usage的49152。支持思考预算已经切换通道、模型却在final继续分析；仅凭重分词不能还原原始token流或证明强制切换边界。C24另外3个length失败也表现为最终通道分析、无可直接恢复的完整JSON，不能把失败简单归因于最终8K不足或有效网格被length丢弃。

吞吐的[TP2双副本静态分析](../../experiments/020/evidence/tp2-replica-feasibility.json)已完成、**未运行**。当前TP4每卡模型约7.25GiB、KV约11.29GiB，KV容量606556tokens；TP2估算每卡权重约14.5GiB、每副本缓存约101–109K tokens，不足以保证4路满49K长请求。减少通信是否抵消预抢占/重算尚不明确；不为“GPU利用率更高”无证据切成两个副本。

后续公开[E20题面](../../experiments/020/evaluation-e20/challenges.json)和[完整清单](../../experiments/020/evaluation-e20/manifest.json)已准备、**未运行**：20task/27query，7短/7中/6长；与本轮D24/C24及已记录开发排除项无交集，但019已整体见过输入，含2个旧ARC1来源题，不称底座未见集。下一轮优先研究最终网格交付，再用其观察真实解题/成本；不原样扩大已知负程序配置。

最新额度与启动顺序见[020计划§10](../plan/020-score-validation-and-real-submission.md#10-执行后交接与下一轮研究)。cc/ji小尾数需计安装、有效推理和收尾；qi完成后剩2.2787配额h，可优先承担一个有明确问题的小研究块。zhuating仍有ARC3用量；本轮ARC2仅在该账号准备无GPU的wheel carrier。金牌目标的[官方快照](../../experiments/020/evidence/medal-snapshot-20261020.json)与本地C24正确率是不同指标，未将65.38%当作能拿牌的榜分。
