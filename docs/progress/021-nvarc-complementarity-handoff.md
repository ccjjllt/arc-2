# 021 NVARC 完整流程与 27B 互补研究进展

日期：**2026-10-21，北京时间**。本轮先完成研究现状复核与执行交接，随后按用户“直接开始”进入实施；对应[021计划](../plan/021-nvarc-complementarity-handoff.md)。审阅基点`248dabac97b07afe7a4209fc19e1db5b7d4758ee`。下文“实现状态与研究效果”为调研结束快照，最新状态见后续编号段。

## 已完成的工作

1. 阅读用户方案与001—020有关记录，核对最新12.50提交、C24及E20、013/014的4B研究和现有可复用代码。确认12.50是27B替换路线的负结果，不能写成与30.14融合后的退分；013的25.69也不是完整原版NVARC能力上限。
2. 通过实时Kaggle页面核对本届官方条件：240道隐藏task、query两槽exact微平均、12小时断网运行、L4×4、北京时间11月3日07:59截止。隐藏query总数和020实际覆盖仍未确认。
3. 查阅按Public Score排序的公开Code列表，并核对Failed in AIMO：33.89对应V1／`scriptVersionId=312063593`。下载了明确V1的原Notebook；当前可见列表带数据源访问筛选，不能称全网最高公开方案。
4. 发现014本地归档实际为V2且有额外模型分支，不能用它绑定V1成绩。V1普通运行只跑4题，26分11秒不是隐藏全量耗时；V1内部评分为task加权，不能原样用作本届主指标。
5. 复核Qwen3.8-27B、CHARM、Re-ARC2、DiARC及NVARC训练配置。独立计算并保存额外109个生成器ID，确认对应generator及verifier文件均存在，未运行生成器。
6. 编写021交接方案：三臂同预算互补研究、逐query增失统计、有限27B路由、一个有日历/GPU双上限的续训挑战者、文件复用与赛前排程。更新README研究入口。
7. 复核交接中的预算公平性：E20全覆盖27B只能算候选诊断，正式B2按路由比例和总成本报告；保留020已取消固定净增题数门槛的用户决定。源码审阅未发现V1独立rescue阶段，未将其他版本的分支混入原流程。

## 形成的判断

采用用户提出的投入顺序，但把“组合更好”收紧为待验证假设。比较完整4B、让出时间后的4B、同时间4B＋27B，分别识别计算预算损失和互补收益；不把所有候选的正确并集当成可提交成绩。

27B官方42.4%属于Semi-Private、xhigh、Baseten环境，足以说明不能从12.50推断其能力上限。109个生成器是相对固定ARC-GEN的新覆盖，不能称模型未见规则。CHARM主46.7%使用公开评估demonstrations训练任务记忆；DiARC公开数据存在evaluation派生材料，都不宜当赛前即插即用升级。

再次确认NVARC的`arc2_evaluation6`在验证路径，不把README的数据列表误写成原始eval答案直接参与SFT。训练中的`nvarc_full`有evaluation规则派生材料，因此公开集仍不能称未见规则验证。

## 产物

| 产物 | 内容 |
| --- | --- |
| [执行交接](../plan/021-nvarc-complementarity-handoff.md) | 背景、假设、三臂实验、数据续训、预算、最小代码修改、排程 |
| [V1原Notebook](../../experiments/021/reference/failed-in-aimo-v1.ipynb) | 从明确版本页面下载的原文件，未执行；SHA256 `71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d` |
| [V1提取源码](../../experiments/021/reference/failed-in-aimo-v1.source.py) | 供源码阅读，含Notebook魔法命令，不是新实现的runner；SHA256 `b129bd6c316ee611722a1c75b4e757b277bd4332efb348bf7561de22462505cd` |
| [版本与文件manifest](../../experiments/021/reference/manifest.json) | URL、V1、模型源、分数证据级别、文件SHA及大小 |
| [官方与Notebook核验](../../experiments/021/evidence/official-and-notebook-review.md) | 实时页面来源、比赛口径、V1/V2区别、公开方案搜索边界 |
| [开源研究核验](../../experiments/021/evidence/open-research-review.md) | 各论文/仓库的评测条件、适用范围和数据边界 |
| [Re-ARC2差集](../../experiments/021/evidence/rearc2-generator-delta.json) | 两个固定commit、609/900/500/109计数、109个ID、核验方法 |
| [用户原方案](../../experiments/021/evidence/user-proposal.txt) | 原始附件逐字归档，保留其当时表述，不作为已独立核实的事实表 |

## 实现状态与研究效果

- **文档与证据：已完成。** 手动核对本地链接、版本/哈希、清单计数和修改范围。
- **求解代码：未修改。** 仅保存公开参照源码；旧提交、旧候选和评分产物未改。
- **GPU/模型/训练/生成器实验：未运行。** 未创建测试文件、测试框架或CI。
- **比赛提交和最终选择：未操作。** 33.89为外部V1页面成绩，本项目未复现；最新本项目正式结果仍为020的12.50，历史复制30.14保留。
- **提分效果：未确认。** 021的三臂对照、路由和续训均需由执行者按计划手动实施。

下一步先用已归档V1建立完整4B对照和实际成本记录，再决定27B分配及唯一训练挑战者。不要因本轮文档或源代码准备完成就启动大规模训练。

## 1 用户授权执行与当前配额

用户在本聊天明确要求“那你直接开始吧”，本轮据此实施021的小规模手动实验。2026-10-21 **20:25北京时间**，在各自独立进程使用既有凭据查询身份和GPU配额，均身份匹配，未输出凭据。配额刷新为10月24日08:00北京时间：

| 账号 | 可用配额小时 | GPU预留秒 | 本轮安排 |
| --- | --- | --- | --- |
| ccjjllt | 0 | 0 | 不启动GPU |
| qingyu1232 | 0.3738 | 0 | 不足以含加载完成有效研究，不启动GPU |
| jiyujia | 0.2490 | 0 | 同上 |
| zhuating | 18.4450 | 0 | 承担首轮完整4B E20观察；同时注意其他项目的共享用量 |

原始快照保存在`experiments/021/execution/quota-*-initial.json`；[查询脚本](../../experiments/021/read_account_state.py)只读取并保存脱敏身份/配额，不启动作业。zhuating近期Notebook已只读查询，reserved为0；未取消或修改其他项目。

首轮准备使用E20全部20task／27query，四卡独立worker；每task660秒含TTT/搜索/重评分，全局研究上限6000秒含安装加载和收尾。完整保留V1的单epoch/rank256与增强设置；训练实际若因截止提前停止，记录真实步数，不写成已完成原版全部TTT。6000秒是研究会话安全上限，不是证明240题全量可运行的耗时。

## 2 逐query解析CPU回放结果

基于020已经冻结的输出和输入匹配gold哈希，比较旧整task解析与完整严格JSON对象中的逐query保留，未调用模型、未下载或读取新gold网格：

| 集合 | 合法query 旧→新 | exact 旧→新 | 新增合法／新增correct |
| --- | --- | --- | --- |
| C24 | 21→21 | 17/26→17/26 | 0／0 |
| Deep6 | 8→8 | 2/8→2/8 | 0／0 |

Deep6原9个query中1个未匹配，仍另列，未强行纳入分母。旧解析与冻结记录逐项一致。失败主要是完整final仍在分析、没有JSON；另一个非法网格task只有单query，因此没有可恢复的其他成员。手动CPU回放约0.038秒，脚本、来源SHA与逐项账本见[回放报告](../../experiments/021/parse-replay/README.md)和[完整结果](../../experiments/021/parse-replay/run-v1/report.json)。

结论：此项解析改动在这两组实际产物上没有新增合法或正确预测，不为它单开GPU。后续27B候选研究沿用已核实解析，将预算用于完整4B和候选互补。

## 3 完整4B首轮已启动

2026-10-21 **20:30:54北京时间**，按实时余额在zhuating发起私有研究Notebook [arc2-021-nvarc-v1-e20](https://www.kaggle.com/code/zhuating/arc2-021-nvarc-v1-e20) **V1**，平台返回kernelId `140020539`。这是公开E20研究，未调用比赛提交接口。

- 冻结Notebook SHA `d92bbf69aeb815f407a21bc8e12258df1e5b3b36006ecf494875a9dfbdd3b346`，输入20task／27query，模型与Unsloth Utility Script、旧镜像均与原V1对应。
- 原`arc_loader.py`、`arc_decoder.py`字节相同，直接使用原`score_kgmon`；其余改动为显式输入/挂载、每task及总截止、真实TTT与候选计量、落盘和错误退出。
- 启动前可用配额66401.861秒；研究上限6000秒，向平台请求6600秒上限；每task660秒，180秒收尾。
- [启动凭据](../../experiments/021/execution/nvarc-e20-launch-v1.json)、[远端版本与挂载](../../experiments/021/execution/nvarc-e20-v1-remote-metadata.json)、[打包manifest](../../experiments/021/kaggle-e20-v1/source-manifest.json)分别保存。实时API已观察RUNNING；效果未评分。

运行日志确认原Utility Script实际挂载为`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`，首个进度事件出现在首cell时钟约329秒。此时尚无已落盘候选；不能把该时点的零覆盖写成最终正确率，也未据此确认所有worker已完成模型加载。

源码审阅发现每task660秒可能被DFS耗尽，导致其返回候选来不及重评分。已另备[V2研究包](../../experiments/021/kaggle-e20-v2/source-manifest.json)：DFS预留60秒评分，额外保存未重评分raw beams，优先检查已确认的utility路径，并在progress列出worker阶段。**V2未启动**，已运行V1保持冻结；切换与否待首轮实际证据，不在运行中改文件。

另已准备[27B E20单臂候选观察包](../../experiments/021/qwen-e20/README.md)，32K总生成、24K思考、原FP8/TP4/C8/MTP2。它是全E20候选覆盖研究，不是部署等预算的B2；尚未启动，不计作互补结果。

## 4 E20 V1 启动状态补记（2026-10-21）

- V1 E20 Notebook `zhuating/arc2-021-nvarc-v1-e20` 已被平台接受，申请6000秒研究会话；该任务不是比赛提交。启动和包信息见 `experiments/021/execution/nvarc-e20-launch-v1.json`。
- 最近归档的日志事件在elapsed 602.15秒时仍为0/27已生成query、27个占位、0个候选文件；日志轮询曾报告连接丢失并重连。输出枚举当时返回空文件列表，见 `nvarc-e20-v1-direct-live.log`、`nvarc-e20-v1-live.log` 与 `nvarc-e20-v1-output-status.json`。
- 因此截至该次取证，远端最终状态、训练/搜索是否继续、结果文件和评分均未确认；没有声称COMPLETE或模型效果。后续应先恢复只读状态与输出检查，避免重复启动或重跑。
- V2 Notebook及其时钟/进度改动已打包，但尚无推送或运行证据。V1/提取源码归档附Apache-2.0许可副本。

官方比赛挂载本身包含公开solutions文件；本次研究包仅内嵌固定challenge输入，代码没有读取这些solutions，也没有把query答案交给模型。源码注释中“no solutions are mounted”表述过宽，不能据该注释声称整个官方挂载不存在答案。

本机系统代理在执行中出现Kaggle API连接重置；进程级直连曾成功回查RUNNING及V1挂载，随后也有握手超时。没有修改系统代理、TLS验证或凭据。因无法取得新的可靠配额快照，未追加27B或V2 GPU作业；[有界只读网络诊断](../../experiments/021/execution/network-diagnosis.json)单独记录各次尝试。日志中的阶段零候选不是最终评分。V1平台6600秒与内部6000秒上限均已记录。

## 5 V1结束为加载超时与27B独立观察启动

**前节的中途状态已被本节更新。** 2026-10-21 20:54北京时间重新读到V1为ERROR，并下载本次完整运行记录。精确输出URL会话ID为`360780134`；[结果摘要与文件哈希](../../experiments/021/execution/nvarc-e20-v1-result.json)、[原运行记录](../../experiments/021/kaggle-e20-output-v1/021-run/run.json)、[worker日志](../../experiments/021/kaggle-e20-output-v1/021-run/worker.log)均保留。

- 首cell总耗时1229.14秒，`stop_reason=model_load_deadline`，900秒加载阶段结束时`loaded_workers=[]`。
- 三个worker停在`Loading checkpoint shards: 0/2`，第四个只到Unsloth import banner；GPU0—2各峰7166MiB、GPU3峰3MiB，89次采样的GPU忙碌率接近零。
- **未进入TTT或搜索，0个真实候选，27query均占位。** 不读取gold给全占位作“模型0分”，该轮无解题效果评价。
- 加载函数参数和实际库版本与013成功版一致；013曾记录约119秒完成一个worker的完整加载。现有证据支持查冷挂载I/O或宿主机资源争用，但没有充分证据确定根因。

准备V3修复包：先将同一权重完整复制一份到`/kaggle/temp`，记录每文件bytes/哈希/耗时，再按完整model-loaded顺序加载四个worker，全部ready后同时开始TTT；增加加载阶段host RAM信息。保留V2的60秒评分预留。V2不单独启动；V3启动与结果另记，不能把复制与串行加载组合成功归因于其中某一项。

20:57北京时间复核zhuating余额为17.3255配额小时、reserved为0，余额变动含平台准备或其他项目用量，不能全部归为1229秒内部运行成本。20:59:10北京时间发起独立[27B E20 Direct32K](https://www.kaggle.com/code/zhuating/arc2-021-qwen-e20-direct32k) **V1**，平台已接受，kernelId `140024619`，见[启动凭据](../../experiments/021/execution/qwen-e20-launch-v1.json)。

该27B会话沿用已运行的FP8/TP4/C8/MTP2与固定依赖，20task／27query每task一次Direct1，xhigh、24576思考／32768总token，总7200秒含安装加载，平台请求7800秒。它与4B加载修复的工作独立；只用于互补候选覆盖诊断，不能直接当作12小时B2融合成绩。尚无新评分，未作正式比赛提交。

## 6 V3修复包与运行中的资源安排

[V3研究包](../../experiments/021/kaggle-e20-v3/source-manifest.json)曾完成准备但未启动；Notebook SHA `6b83c09e04b5ffafffb5b859015bdab49bf46183227cabe0eeaaaee21bdeb67a`。设计为先向本地盘复制权重（最多600秒），再依次完整加载四个模型，全部ready后才开始求解；仍保留900秒总模型加载上限、每task660秒和6000秒总会话。模型、TTT、增强及原选2不变，保留V2的评分预留。生成源码与Notebook因上游许可未明已移除，研究设计和文件哈希仅作记录。

21:01北京时间[配额快照](../../experiments/021/execution/quota-zhuating-before-v3.json)显示27B会话运行期间reserved为86149.55秒，按`total − used − reserved`计算可用为0。这个预留不是实际已消耗时长，也不是本次7200秒内部预算的计量结果；不据此并发开V3，不把17小时起始余额重复分配给多个会话。先观察当前27B作业，结束后重查释放与实际余额，再运行加载修复。

27B实时日志已确认固定wheel SHA匹配、requirements安装返回0；截至21:03北京时间正在安装vLLM wheel，尚未取得推理结果。依赖或服务成功与模型提分分别记录。[实时日志](../../experiments/021/execution/qwen-e20-v1-live.log)持续保存，候选评分仍未运行。

## 7 27B E20执行结束与源码归档范围

22:07:34北京时间，27B E20日志记录`EVIDENCE-EXIT complete_with_evidence`且`VDRIVER_RETURN_CODE 0`。日志有若干P0生成事件，但当前仓库没有冻结的完整候选输出或评分报告；因此只确认执行流程正常结束，不报告候选覆盖率、准确率或比赛分数。未启动V3，需在配额释放后重新核对余额再决定。

复核上游[NVARC仓库](https://github.com/1ytic/NVARC)时，仓库根目录及`ARChitects`目录均未见许可证声明。为避免继续分发授权状态不明的上游实现，仓库不再保留NVARC派生的V1/V2/V3源码副本及其打包Notebook；V1实际运行日志、结果元数据、配置与源码哈希仍保留用于研究追溯。未来如需复现，应先确认上游许可或取得授权，再从有权获取的源码重新打包。V3仍是未运行的研究设计，不是已验证修复。

## 8 27B E20 完整产物、评分与决策更新

2026-10-21 **22:24—22:32北京时间**，用户通知跑完后，平台回查为`COMPLETE`、`currentVersionNumber=1`，kernelId `140024619`、输出会话ID `360788173`。下载全部预测、20份原回复、运行/GPU/server日志，先于新gold读取在22:25:20冻结33份产物；后补取实际执行的6份源码和输入，与研究包SHA全部一致，未下载的仅为`__pycache__`。本轮没有追加GPU作业或正式比赛提交。

完整[结果摘要](../../experiments/021/execution/qwen-e20-v1-result.json)、[原始输出冻结清单](../../experiments/021/execution/qwen-e20-v1-artifact-freeze.json)、[远端版本与文件清单](../../experiments/021/execution/qwen-e20-v1-remote-metadata.json)、[源码与输入核验](../../experiments/021/execution/qwen-e20-v1-source-verification.json)均已保存。主`submission.json`及M0/M1的SHA均为`3367f1cac955520dfb46a45a04296a290e1984999ffce62478df9779022a1e22`；M0/M1是同一P0的重复视图，每query两个槽也相同，不能算两个独立候选。

### 8.1 真实覆盖与官方版本对齐评分

输入仍为原E20的20task／27query，challenge SHA `ad12e367296c958b1f84a1c2bf7c2a8858cfb09db4dbc8e152d5735890b67dda`，未因失败缩小集合。沿用固定FP8权重v1、TP4/C8/MTP2、xhigh、24576思考／32768总token、P0 seed5101；每task一次Direct1，无CPU补槽或repair。E20公开输入早在019暴露过，不能称未见验证集。

| 指标 | 实际结果 |
| --- | --- |
| 模型调用／返回 | 20／20，全部HTTP 200 |
| 真实候选覆盖 | 14／20task、18／27query；其余9query为初始化占位 |
| 原固定官方版本主评分 | **2／25 query，8.00%**；另2query完整demo不匹配，原报告保留未匹配 |
| 匹配旧版本的两query补充 | **0／2**，两项均无真实候选 |
| 显式按输入版本对齐的全E20 | **2／27 query，7.41%**；完整task exact为2／20 |
| 原候选错误构成 | 18个成形预测中2正确、16错误；另9个占位 |
| 不同双槽／第二槽独有命中 | 0／0 |

主评分使用既有`score_frozen_outputs.py`，在任何gold访问前再次记录预测SHA，完整匹配官方`f3283f727488ad98fe575ea6a5ac981e4a188e49`的有序demo和query输入，见[原评分报告](../../experiments/021/scoring-qwen-e20-v1/score-report.json)及[逐query账本](../../experiments/021/scoring-qwen-e20-v1/query-ledger.jsonl)。正确query为`45a5af55_0`、`8f215267_0`。

`d8e07eb2`的两个query未匹配原因是Kaggle与该官方revision的第0个demo output有9格差异；两个query输入相同。查官方文件历史后，旧revision `bebec789c46f4be8f690a321ad852ecdb1f3f362`与本轮完整有序demo和两个query输入均一致，故只对这两项追加[独立版本补充评分](../../experiments/021/execution/qwen-e20-version-supplement-v1.json)。版本选择依据是输入完全一致，不依答案优劣选择；原主报告未改。**2／27是明确混用两个匹配官方revision的全E20诊断汇总，不能称单一f328版本成绩，也不是Kaggle公榜或私榜分数。** [评分审计](../../experiments/021/execution/qwen-e20-scoring-audit-v1.json)同时保留两种口径。

### 8.2 耗时、截断与CPU回放

实际硬件4×L4、每卡23034MiB。driver总耗时**4063.55秒（67.73分钟）**；其中安装累计261.6秒、含安装至server-ready累计711.6秒、生成3351.32秒。711.6秒已包含261.6秒，不能相加。生成551365 token，按并发墙钟计算聚合164.52 token/s；最大并发8、时间加权平均6.56，平均请求延迟1099.5秒，不能把167.6秒/task的完成间隔当单请求延迟。详见[运行审计](../../experiments/021/execution/qwen-e20-runtime-audit-v1.json)。

- **未触发全局截止。** 15个调用`stop`、5个`length`；5个length均耗满单调用32768 token，影响8query，final通道仍在继续分析，没有完整JSON。
- 另一个`stop`的`7491f3cf`有解释前缀及完整末尾JSON，原整task解析拒收。全部6个无效task共消耗189130 token，占34.30%；其中5个length占163840 token、29.72%。
- 对保存文本重新分词，20个reply的reasoning均24575 token，加结束思考标记恰24576；支持预算触发通道切换，不能还原原始token流，也不能证明再加思考预算就会解对。
- 在不读gold的独立审计中，对全20回复统一回放“唯一完整合法末尾JSON对象”规则并先冻结派生产物，仅恢复`7491f3cf`的1个query：真实覆盖18→19。[单独评分](../../experiments/021/scoring-qwen-e20-tail-json-v1/score-report.json)原/派生均2／25，新增正确为0；另外两query原占位未改，版本对齐全E20也仍为2／27。严格逐query保留恢复0。详见[解析审计](../../experiments/021/execution/qwen-e20-parse-audit-v1.json)与[派生冻结记录](../../experiments/021/parse-replay/e20-tail-json-v1/candidate-freeze.json)。原输出未覆盖，没有新增模型调用。

若240task具有本轮同样平均token成本和聚合吞吐，一次全覆盖加安装加载及300秒收尾约**11.452小时**，距现有41400秒政策仅余173秒；若全题耗满32K，约13.559小时。这是有条件算术外推，没有实测隐藏覆盖，且没有包含4B、第二次候选或融合成本，不能据此确认全量组合可行。

22:24北京时间[配额快照](../../experiments/021/execution/quota-zhuating-after-qwen-v1.json)为zhuating剩**13.6145配额小时**、reserved=0；刷新10月24日08:00北京时间。前后余额差包含平台计量及可能的共享用量，不等同于本driver的精确GPU成本。

### 8.3 本轮研究结论与下一步

本轮是负结果：当前27B Direct1 32K只答对2／27，16个成形预测仍错误，解析修复未新增正确，不能把主要瓶颈归结为JSON格式。暂停原样扩量、加长token或重投该策略；这也没有给出27B能力上限或其与4B互补收益。

维持021的顺序：先取得实际完成的完整4B对照与成本，再在同预算下衡量让出4B预算的损失、27B独有命中、最终双槽新增与丢失。当前没有可配对4B结果，不能报告oracle互补、B2净增或组合分数。4B加载修复V3仍未运行；源码归档限制沿用§7。27B模型选型不变，若后续继续，优先提出有对照的小规模完成性／任务适配假设，不把本轮2分解读成训练或其他方案已获支持；唯一限时续训挑战者亦未运行。
