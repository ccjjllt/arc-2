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

当前建议仍是原V1候选上的一次B4／R2确认，同时保存完整候选池和阶段覆盖。用户随后收敛到NVARC内部优化，下一轮只按证据选择候选排序、搜索覆盖、TTT／增强或针对性续训中的一个方向。公开确认不能代替正式隐藏评分；R2确认、新调度与续训均**未运行**。

本轮只做文档和只读核验，没有新的实验硬件、运行时间或提分指标；旧实验的配置、时间和产物沿用原进展文档。本次手动核对文档链接、指标算术和修改范围，不新增测试文件或自动测试体系。

## 5 用户明确以NVARC为唯一主线

2026-10-25用户进一步明确：“我们只需要在NVARC的方案上尽可能的提高我们的分数就好。”据此更新AGENTS.md、根README和025计划，固定当前目标为在比赛预算内提高NVARC的实际成绩。

下一步原V1上的B4／R2确认保持不变，随后只在NVARC内部依据损失归因优化一个因素。27B与其他架构的既有资料保留为历史证据；有限27B补题、合队及独立争奖路线不再列为活动建议。本次只更新研究约定与交接，没有改动求解代码或启动实验。

## 6 用户授权执行与实验准备

用户随后明确“直接开始”，并要求新派子代理；本轮已重新创建数据划分、研究包装、候选回放和公开Notebook版本核对四个新子代理，未沿用前轮代理。用户再次强调应在公开最高分Notebook上优化，因此固定Failed in AIMO精确V1／312063593为源，不从上游实现重新起步。

2026-10-25上午重新直接读取[按Public Score排序的官方Code列表](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?competitionId=133469&sortBy=scoreDescending&excludeNonAccessedDatasources=false)，本次请求`excludeNonAccessedDatasources=false`，首项仍Failed in AIMO 33.89，其后Learned from AIMO、Minimal Perfpatch等32.22。精确高分版本与冻结源的绑定沿用此前逐版本核验。

账号`zhuating`身份核对通过：02:39 UTC读取可用GPU额度79019.836秒，约21.95小时，reserved为0；02:42 UTC复查相同。证据见`experiments/025/execution/account-start.json`与`account-prelaunch.json`。这是额度读数，尚非本次实验实际耗时。

已冻结[40题／58query确认集合](../../experiments/025/cohort-v1/manifest.json)，由63个符合排除条件的任务分层选出。当前Kaggle公开challenge快照与008归档字节相同，源SHA为`e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8`。实验输入SHA为`38df4c7126f160f9dd045b4b0c9fb37ccb4f24dfbb51c8d32aa7b50f132ebeb9`。补排此前010已跑而旧排除记录遗漏的`1818057f`、`9aaea919`；保留历史全公开源推理与上游验证接触限制，不称盲测。选集过程未读取query答案。

新增`experiments/025/prepare_cohort.py`、`prepare_confirmation.py`与`replay_selection.py`，分别处理固定输入、最高分V1研究包装和同候选回放。回放脚本使用旧022候选作人工历史核对，重现B4 3/27、R2 4/27、新增1／丢失0；该核对没有新增模型调用，也不是40题实验结果。没有新增测试文件或测试框架。

研究包装保留原loader／decoder、单epoch rank256 TTT、原1200秒逐task软检查、DFS、增强和重评分；研究全局求解截止7080秒，父进程组硬截止7140秒，会话上限7200秒，留收尾时间。候选移入可下载目录，额外保存合法raw候选、原decoder顺序与阶段事件，控制台保留逐题日志。日志与raw导出存在少量CPU／I/O开销，不能称完全零成本；B4／R2使用同一池消除两次随机生成差异。正式023版本保持冻结。后续启动事实以真实launch回执为准。

## 7 实际启动与远端身份

**10:48:30北京时间，平台接受`zhuating/arc2-025-v1-selection-confirmation` V1，kernelId 140698356。** 请求L4×4、断网、平台7200秒。10:49读取状态RUNNING，远端metadata确认NvidiaL4、GPU开启、断网、BF16 V1权重与原Unsloth utility挂载、currentVersionNumber=1。metadata的lastRunTime日期异常，实际启动时间采用本次请求／响应记录。

上传包位于`/private/tmp/arc2-025-v1-selection-confirmation-compact`，Notebook SHA为`9a83821330e510d267ca7bdeed12a2f5aa34161ea0b0aeca22bc8e6dedd5dd08`。逐cell规范化source后与远端完全一致，原loader／decoder仍与公开V1逐字一致，训练器、DFS与重评分函数AST也与原V1相同。证据见[启动回执](../../experiments/025/execution/confirmation-v1-launch.json)、[远端身份核对](../../experiments/025/execution/remote-source-verification-v1.json)与[包清单](../../experiments/025/confirmation-v1-package-manifest.json)。

首次未压缩包2,284,457字节，被平台以400明确拒绝，诊断返回“kernel source must be less than 1 megabytes”。核对未创建目标运行、reserved仍为0后，将输入改为确定性gzip＋base64嵌入，Notebook降为113,733字节，解压后输入SHA与冻结文件一致；所有求解和输出cell源码不变，再发起上述成功请求。拒绝及诊断记录单独保留，没有把它计作一次模型实验。

当前只确认平台RUNNING与源身份；首个10:49日志回查为空，实际GPU加载、TTT和候选覆盖仍待后续日志确认。B4／R2评分、oracle与真实GPU耗时尚未取得，不能宣称提分。没有创建正式比赛submission或定时监控。人工结束后下载与评分入口已写入[025产物导航](../../experiments/025/README.md)。

## 8 V1早期依赖挂载阻塞

**10:51实查V1已ERROR，精确scriptVersionId 361989657。** 日志在In[3]报告原固定路径`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`缺失，在模型复制／加载之前停止，因此没有TTT、DFS或候选结果。已下载失败产物与完整日志，见[真实终态](../../experiments/025/execution/confirmation-v1-outcome.json)。不能继续把§7的RUNNING快照当当前状态。

只读回查上游utility仍为V2，输出分页身份272622465与013历史记录相同，暂未发现上游输出换版证据。10:52账号GPU used为29149.536秒，相比启动前增加169.372配额秒，reserved已回0；这是账号读数差，不冒称精确逐作业运行时间。

10:53启动一次最多300秒的CPU挂载探针，使用相同utility与模型数据源，只枚举目录和标记文件，不启用GPU、不导入模型。根据真实挂载报告修复路径后再恢复同一确认实验；原最高分V1求解算法与冻结40题不变。探针回执见`experiments/025/execution/utility-cpu-launch.json`，后续实际恢复状态另记。

## 9 CPU核对收紧与用户等待指令

首个CPU探针目录扫描过宽，300秒后回查为CANCEL_ACKNOWLEDGED，未取得完整挂载报告；它没有导入模型或使用GPU。11:00:54北京时间，平台接受同一CPU Notebook的**V2**，kernelId 140699090，上限180秒，改为固定层级候选目录与逐阶段落盘，避免递归扫描全部依赖树。真实回执见`experiments/025/execution/utility-cpu-fast-launch.json`。

GPU恢复V2包已准备于`/private/tmp/arc2-025-v2-mount-repair`，Notebook SHA为`9ee3ba7b180e5d4e0c2f3a55385faed46c16a0f8ee6f54b28f26446e2042d715`，总预算6900秒，为首次失败消耗留出余量。仅增加原utility／固定模型别名目录识别、023模型配置指纹校验及缩短研究全局时间；solver、TTT、DFS、decoder和输出cell与研究V1相同。恢复包与人工启动脚本`/private/tmp/arc2-025-launch-v2.py`均**未执行远程启动**。

用户随后明确“跑完我告诉你”。据此停止主动查询，不启动GPU恢复包、不设置定时任务，等待用户通知。已明确告知用户当前运行的是CPU挂载核对，GPU研究V1已早期失败。用户通知后先读CPU V2真实结果，核对挂载修复与当前配额，再恢复同一公开最高分V1实验；不要根据旧RUNNING快照或旧额度自动执行。

## 10 CPU结果确认与GPU V2恢复

用户通知“跑完了你看看”后恢复工作。**11:02:40北京时间**实查CPU核对V2为COMPLETE：原固定utility根目录`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`存在，`unsloth/__init__.py`为11452字节；报告中两条相同记录来自重复发现，按根目录去重后仅1个。模型`sorokin/.../bfloat16/1/config.json`也存在，ivan别名路径不存在。目录核对代码实际耗时约0.038秒。证据见[CPU挂载报告](../../experiments/025/execution/utility-cpu-mount-report.json)和[结论](../../experiments/025/execution/utility-mount-conclusion.json)。

这**没有证明目录改名**，GPU V1当次路径不可用的根因仍未确认。启动包装补充原utility与模型标记的最多60秒共享等待、每2秒重查，仍只做固定层级枚举，等待计入会话预算；记录`mount-diagnostics.json`。没有安装另一套依赖或修改求解算法。

**11:04:55北京时间**平台接受同一研究Notebook **V2**，kernelId仍140698356。启动前GPU可用78850.464配额秒，reserved0。V2平台上限6900秒、求解截止6780秒、进程硬截止6840秒，冻结40task／58query及TTT、DFS、增强、重评分、B4／R2规则不变。真实回执见[confirmation-v2-launch.json](../../experiments/025/execution/confirmation-v2-launch.json)。

本次实际包为`/private/tmp/arc2-025-v2-mount-ready`，Notebook SHA为`ca8c03dde65b9eca40ed5ea8e48938ef58737994b2a2e122b76edc5c671cf4ef`。11:05远端metadata显示V2、NvidiaL4、断网和正确的数据源；全部cell源与本地冻结包一致，见[远端核对](../../experiments/025/execution/remote-source-verification-v2.json)。这是平台RUNNING和源身份确认；实际模型加载、TTT与评分需看随后日志，尚无新得分。

## 11 启动核对通过并等待用户通知

用户进一步明确“你只需要确定正常就行跑完了我告诉你”。本轮仅继续核对启动，**11:10:54北京时间已从实时SSE日志确认rank0—3均完成两片权重加载**，启动流未见Traceback。实际nvidia-smi为4张L4，权重复制7,267,271,302字节、耗时66.653秒，模型配置指纹正确。证据见[启动健康记录](../../experiments/025/execution/confirmation-v2-startup-health.json)及`confirmation-v2-stream.log`。

GPU V2实际工具根目录为`/kaggle/usr/lib/pip-install-unsloth-flash-patch`，与CPU探针的嵌套目录不同；本次从sys.path发现同一工具只耗时约0.0047秒，有界等待没有实际消耗60秒。固定路径兼容修复已在GPU通过，不能把CPU结果写成所有硬件上的统一路径。V1自身实际目录未记录，不将上述观察扩大为其唯一根因证明。

已确认的是启动正常，不是完整求解或提分。按用户指令停止所有状态查询和日志流，不创建定时任务，等待用户通知完成后再下载输出、同池回放与评分。额外结果汇总脚本`experiments/025/summarize_confirmation.py`已准备，尚未对本次实验运行，不据代码完成宣称收益。

## 12 完整运行及同候选池评分完成

用户再次通知“跑完了你看看”后读取终态并下载。研究Notebook **V2／361992679为COMPLETE**，`return_code=0`、`timed_out=false`，无Traceback或CUDA OOM。代码源仍为公开33.89的Failed in AIMO精确V1／312063593；两者版本身份不能混用。

### 数据配置与实际成本

- 数据为[冻结40task／58query](../../experiments/025/cohort-v1/manifest.json)，来自保留的官方Kaggle公开evaluation challenge快照，按输入长度与query数分层；排除E20、原四题和记录的开发任务。输入SHA仍为`38df4c7126f160f9dd045b4b0c9fb37ccb4f24dfbb51c8d32aa7b50f132ebeb9`。该集合存在历史公开集推理、上游验证／规则派生训练接触，不称模型未见盲测。
- 权重为原`sft139 BF16 V1`，保留单epoch rank256逐题TTT、原1200秒task软检查、DFS、增强、重评分；生成一次后才分别导出B4／R2，选槽脚本未读取gold。研究Notebook SHA为`ca8c03dde65b9eca40ed5ea8e48938ef58737994b2a2e122b76edc5c671cf4ef`。
- 实际硬件4×NVIDIA L4、每卡23034MiB；Python3.11.13、Unsloth2025.9.7、Transformers4.55.4、Torch2.8.0+cu128。Notebook实际区间约**11:05:12—12:56:37北京时间**，墙钟**6684.238秒**；solver6570.606秒，末尾冻结22.964秒，solver结束早于全局截止118.725秒。不是由平台metadata异常日期推算的时长。
- 40/40 task均实际完成TTT：128步、epoch1.0，并有`task_end`。58/58 query都获得搜索；计划232批，实际227批均正常返回和导出，55个query跑4批、1个跑3批、2个跑2批。两个task触发原逐题软超时，但没有未开始搜索的query。
- TTT观测区间合计7484.054 rank-seconds，搜索14610.051，491次新网格重评分2222.178；这三个数为并行worker合计，不能当作Notebook墙钟。阶段起止均配对，无censored interval。

### 配对结果

两份预测与候选先冻结，随后从官方ARC-AGI-2固定commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`按完整demonstrations和query输入匹配读取答案。**58/58全部匹配，0版本不匹配**；评分输出SHA与冻结值一致。原B4回放逐query等于远端原输出，另用NumPy复算均值与排名后无选择差异。

| 指标 | B4原选择 | R2挑战者 |
| --- | --- | --- |
| 两槽exact query | **15/58，25.86%** | **15/58，25.86%** |
| 完整task全query正确 | 9/40 | 9/40 |
| 相对B4新增／丢失 | — | **0／0，净增0** |
| raw候选oracle | 17/58 | 同池17/58 |
| scored候选oracle | 17/58 | 同池17/58 |

R2更换了10个query的第二槽，但没有改变任何query的正确性。两策略共同漏选`58490d8a_0`与`d35bdbdc_1`的正确候选；这些ID只用于解释结果，不用于部署白名单或继续调本集合的排序。

raw和scored均885条记录，去重后均491个query-grid，55个query有候选；两池网格集合完全相同，无raw正确却未评分的缺口。709项归档身份全部通过字节／SHA核对，包括687个候选pickle；原decoder读取顺序与导出记录来源完全匹配，未发现候选丢失。

### 错误归因与本轮决定

最终43个未解query中，**41个raw池内没有正确候选**，仅2个正确候选未进入两槽。41个又分为38个有合法但错误的候选、3个没有合法候选；3个占位均已跑满4批DFS，不能归因于没轮到搜索。完整TTT和低训练loss也没有保证学会相应规则。

**按预定停止条件保留B4，结束R2扩展。** 不在这40题继续搜排序规则，不将公平覆盖列为本轮首要瓶颈。下一研究方向收敛到NVARC 4B的候选生成能力，优先设计单个针对性数据／续训小对照，保持原TTT、DFS与B4；尚未训练，不能承诺提分。

这次25.86%是公开诊断集合的本地结果，**不是正式公榜25.86**。正式最好仍为023的30.14；本次下载、回放和评分没有另启GPU、创建比赛submission或定时任务。

### 产物

- [完整汇总](../../experiments/025/confirmation-v2-summary.json)、[实际终态](../../experiments/025/execution/confirmation-v2-outcome.json)、[归档核对](../../experiments/025/execution/artifact-verification-v2.json)。
- [冻结两策略输出](../../experiments/025/replay-v1/prediction-freeze.json)、[B4](../../experiments/025/replay-v1/submission-B4.json)、[R2](../../experiments/025/replay-v1/submission-R2.json)。
- [评分报告](../../experiments/025/scoring-v1-f328/score-report.json)、[逐query账本](../../experiments/025/scoring-v1-f328/query-ledger.jsonl)。
- 原始runtime、候选与日志在`experiments/025/confirmation-v2-output/025/`；第三方源码单独保存于`/private/tmp/arc2-025-output-v2/`，未复制进仓库。
