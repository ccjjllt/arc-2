# 014 — 25.69退分归因与27B完整推理实验

日期：2026-10-17，北京时间。对应[实施计划](../plan/014-regression-and-reasoning.md)。本轮完成比赛状态核实、三路并行研究、源码审计、人工CPU算术和实验交接；**未启动GPU、未训练、未新增比赛提交、未修改既有求解代码**。

**10月18日复核更新**：以上及第1–5节描述10月17日交接时状态。用户随后已实现并运行R/V；R v4配对结果已独立复核，V v4在import阶段失败，实际状态见第6节。不能继续将第1–5节“未运行”当作当前状态。

## 1 正式结果与责任边界

Kaggle返回submission **57103498 COMPLETE / public_score25.69**，Notebook为`ccjjllt/arc2-013-bounded-baseline`，固定scriptVersionId357613526。较历史30.14下降**4.45个百分点**。这次是4B备用线的新配置，不是27B比赛结果；历史30.14仍为未修改公开方案复制提交。

013实施同时压缩训练/搜索和改变救援/选2，先前将其概括为有界修复不足以描述实际研究变化。4题pilot每题1200秒，没有模拟完整240题调度下的约700秒强度；小片2/4不支持提分预期。这是本轮明确纠正的实验设计缺口。

本次尝试读取比赛预测，接口返回`Permission 'submissions.download' was denied`；没有通过其他路径绕过。隐藏日志、真实候选覆盖、fallback数量、每题训练步数均未取得。普通pilot日志不能替代隐藏日志，输出文件字节数也不能推导正确率或fallback比例。现阶段可以定位退分机制，**不能精确分摊4.45个百分点的原因**。

原始返回与榜单/配额保存在[current-state.json](../../experiments/014/evidence/current-state.json)。没有私榜结果，没有改动历史提交或账户最终选择。

## 2 退分审计的实际发现

| 发现 | 已有证据 | 仍未确认 |
| --- | --- | --- |
| 隐藏规模会压低训练强度 | 源码公式在240题、启动360秒场景给695.31秒/题，TTT278.12秒；pilot允许480秒 | 隐藏各题实际预算、步数及命中损失 |
| 搜索额度远低于名义540秒 | 同场景单/双query首批DFS78.22/39.11秒，所有长度bucket共用截止 | 哪些隐藏query因此无完整候选 |
| 评分可能挤占后续query | DFS按批分，8视角评分却用整个题截止；未完整评分不落盘 | 隐藏发生频率；pilot92个完整beam均保存，未见该小片丢失 |
| 候选和第二答案策略改变 | base-only删除rescue，diverse_dual改kgmon | 各项单独公榜影响；latest v21不是历史指定源码 |
| 没找到明显格式映射错误 | loader未改，未见query错位/颜色逆变换新增错误，kgmon公式一致 | 不能因此排除所有运行或数值问题 |

pilot中0934a4d8/aa4ec2a5完整训练分别399.20/308.43秒，超过上述278秒场景；981571dc实际75/128步，说明截止确会截训练。981只用于已有运行记录，不读取答案或新增评分。

产物：[regression-audit.md](../../experiments/014/regression-audit.md)、[预算场景JSON](../../experiments/014/regression-budget-scenarios.json)。这些是CPU源码与日志证据，不是新GPU实验。确认划分沿用013纠正：主C6未运行，981从未见备用身份退役。

额外人工CPU复算允许的3task/4query，最终预测仍为2/4；本地候选原文已不在产物目录，一次普通Notebook指定标签检索未找到，故oracle及“生成对但选错”数量**未确认**，不能报零。见[pilot选择诊断](../../experiments/014/evidence/pilot-selection-diagnostic.json)。精确历史源码的一次版本标签查询也未取得；这不证明314617319已删除，不用v21补作该版本，见[检索记录](../../experiments/014/evidence/historical-314617319-retrieval.json)。

## 3 新调研如何改变决策

### 公开Notebook和推理资产

- 新Qwen27B Notebook精确版本358248812有作者约21.5分报告；同作者还报告小改后13。已读源码，主要变化是CUDA链接修补、更长思考和更大并发。没有可核验的金牌级公开27B版本，不能把论坛40+口述移给不相关代码。
- 找到`nick2187/qwen38-vllm0272-cu130-wheelhouse-v1`：固定官方commit、Python3.12/Torch2.13/TF5.15/CUDA13完整wheel集合，解决旧vLLM0.19与TF5.8的声明冲突方向。只核了小文件和manifest；L4兼容、依赖完整性、吞吐均未实跑。
- 新Merlin27B fork的主要优化限定sm86，249token/s来自3090短测；sm89资产明确gpu_tested=false。暂不以它为L4速度预算依据。
- NVARC公开改进中训练后独立算推理预算、确定性增强种子、零候选救援值得借用，但需逐项证据；不再同时改四种算法后归因。

源码和小manifest已冻结并记录SHA，详见[公开方案研究](../../experiments/014/public-solutions-research.md)、[来源索引](../../experiments/014/public-solutions-sources.json)。未下载大型权重或wheel集合，未安装新栈。

### 前沿方法和历史负结果

- 同模型官方xhigh/low差异支持优先研究调用与完成推理，不能把官方Semi-Private42.4当Kaggle保证。
- DiARC论文报告NVARC4B公开eval任务宏平均30.83→33.06，已有约2.12GB adapter资产；不是本比赛按query平均的分数，基座二进制匹配未确认，仅保留条件备选。
- Poetiq与REPL公开实现支持复用演示执行反馈；不同闭源/120B模型的收益不可移植。012没有生成完整程序，尚未真正检验反馈。
- Hourglass用更多token/多候选，不能当免费压缩；LLM+遗传搜索的6/60是局部公开集且需新DSL/GPU实现，不据此扩建框架。
- 最新作者合成训练报告使用A100数十小时仍无明确ARC2改善，结合本项目P12零增益，暂停无对照大规模续训。训练若恢复，应围绕已识别瓶颈和独立实例/可验证短程序的小问题。

详细来源、资产及可证伪设计见[方法与数据研究](../../experiments/014/methods-data-research.md)。外部成绩均保留原始评测范围和作者报告等级，没有转记成本项目效果。

## 4 最新比赛与资源快照

官方页面及API查询日为10月17日，见[competition-pages.json](../../experiments/014/evidence/competition-pages.json)：

- 2971队，账户当前排名986（已有成绩的账户排名，不是25.69的独立排名）。公榜第15名57.08、第8名67.22、第一88.06。
- 按Kaggle超过1000队的金牌公式`10+floor(0.002×teams)`，当前约15名；Progress Prize前8。只作公榜距离参照，最终私榜/参赛数/并列/资格决定获奖。
- 截止北京时间11月3日07:59；入队/合并10月27日07:59。每日最多1份提交、最多2份最终候选，无网，12小时。官方L4×4配额消耗2倍。
- GPU total108000秒、used2487.836秒、reserved0，剩**29.308934配额小时**；刷新10月24日08:00。当前used不擅自归因于本项目某个作业。本轮无新增GPU消费。
- 奖金要求开放方案；规则将writeup奖称Innovation Prize，Prizes页面仍称Grand Prize。本轮不把它当容易取得的替代奖金路线，保留来源与命名差异。

比赛成绩差距很大，现有证据不足以承诺金牌或奖金。研究优先级因此转向能带来独立新增解的27B推理/程序能力，而非只争取4B小幅回升。

## 5 已完成交接与下一步

已形成固定cohort、700秒4B两策略、27B单栈启动、xhigh1×32K对low2×16K、程序一次反馈、真实两答案净新增和主C6一次确认的实施顺序。CPU预算为本周期最多10小时会话，按2.3预留23小时配额，保留约6.31小时；实际按每次余额和阶段信号准入。

人工复核8个来源SHA全部匹配，P2/P6/P12/B4计数分别为2/2、6/6、12/14、4/6，主/备用确认交集均为空；本轮计划/进展的本地链接可解析。独立审阅后补上4B两臂顺序平衡、adapter/优化器重置与程序共享P0的三次调用记账；条件吞吐未写成实测。记录见[manual-review.json](../../experiments/014/evidence/manual-review.json)，未新增测试套件。

本轮新实验均**未运行**：4B R0/R1、vLLM新栈、27B thinking对照、程序反馈、选2互补、C6确认与正式候选。相关runner/Notebook按计划列出的修改点交由后续执行，不能称已实现或可直接开跑。

下一次首先制作并运行有限R/V窗口；不重复提交25.69、不重启大训练、不重开模型选型。若生成正确候选但筛选丢失，处理评分/选2；若完全生成不出答案，处理完成性/推理方法。只对有真实新增与预算证据的配置扩大评价并提交。

## 6 10月18日：复核用户新增R/V实施

复核起点为干净工作区、HEAD `a6d6023`；审阅用户新增`3737e40`与`a6d6023`。本次只读查询已完成Notebook、人工CPU重评分与源码复核，并补研究记录；没有启动GPU、重投比赛或修改求解器。R/V Notebook所有者为`jiyujia`，本机当前CLI身份为`ccjjllt`，运行身份、结果和配额须分别记账。

### 6.1 R v4：结果可信，按原计划结束该分支

普通Notebook `jiyujia/arc2-014-r-b4-paired-budget-diagnosis/4` 状态COMPLETE。独立复核的是`/private/tmp/rb4out4/014-r-b4-run/`，不是旧失败版本目录。实际4×L4，2456.824秒（40.95分钟），worker返回0，8次task-arm全部完成；六份冻结源码/配置SHA一致。必要的小型原始证据已复制到[持久复核目录](../../experiments/014/review-20261018/r-b4-v4/)。

| 指标 | R0：训练上限280秒 | R1：训练上限480秒 |
| --- | ---: | ---: |
| 最终query EM@2 | 2/6 | 2/6 |
| 原始完整候选oracle_any | 2/6 | 2/6 |
| 有完整候选的query | 5/6 | 4/6 |
| 原始/完成重评分候选 | 62/62 | 61/61 |

正确query两臂相同：36a08778_0和aa4ec2a5_0。仅按固定B4白名单取答案，未读C6或981答案。0934a4d8实际83→128步，aa4ec2a5实际118→128步；另外两题均128步。21897d95_0两臂都无beam；36a08778_1在R0有2个错误beam，R1无beam。没有正确raw候选被评分或最终kgmon丢失。

因此：**在这个700秒完整预算策略与固定小片上，多给训练没有净新增；没有证据启动R2评分/排序分支，也无需重跑R。** 不将结果外推为所有TTT无效，不能解释公榜全部4.45分退步。没有显式统一预热，虽已平衡顺序/同题同卡，缓存与首次JIT仍使它更适合策略取舍而非纯训练步数因果分析。

本次还发现`r-b4/supervise.py:464–466`的`worker_ok`用OR结合可能为空的`all()`，未来非零退出且缺失波次返回码时可能误记complete；本次返回码0且8个task_done齐全，不影响上述结果。该处下次复用前修正，不为此单独开GPU重跑。完整评分、来源、原始文件SHA与限制见[r-b4-v4-review.json](../../experiments/014/review-20261018/r-b4-v4-review.json)。

### 6.2 V v4：GPU环境已到位，模型推理尚未开始

远端`jiyujia/arc2-014-v-27b-service-probe/4`是COMPLETE，但实际日志在启动约256.97秒记录`EVIDENCE-EXIT status=import_vllm_failed`，随后`VDRIVER_RETURN_CODE 0`。这表示带失败证据的正常退出，**不表示vLLM服务或27B生成成功**。日志确认Python3.12.3、4×L4、driver580.178.04、模型与wheelhouse挂载、通过3.13的pip向3.12目标安装；尚未进入模型服务、hello或P2。

本地`vdriver.py`新增的site清理和错误日志增强没有重新打进Notebook：当前源码SHA `5b282acd…`，嵌入与远端boot均为`0aa6675b…`。probe与prompts的嵌入SHA一致。只改源文件再push旧Notebook不会带上改动；发布前须重打包并核三文件SHA。

**直接失败原因已取得并定位**：普通输出只有4,711,829,053字节ZIP，未下载整个环境；通过HTTP范围读取约21MB中央目录和三个小文件，提取内部`run.json`、pip/ensurepip日志，保留CRC与SHA。`import_check`实际为`ModuleNotFoundError: No module named 'vllm'`。安装命令只使用`-r requirements-qwen38-runtime.txt`；这份文件是vLLM的运行依赖，未列vLLM本体，pip的成功清单也没有vLLM。

wheelhouse manifest已列本体`wheels/vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl`，SHA256为`5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a`。下次安装必须在同一个隔离target中显式加入这一个固定wheel，并验证实际模块路径与版本。这次失败**不是已证明的CUDA13不兼容或模型能力问题**；也不需为此先更换模型/整个运行时。[原始内部状态](../../experiments/014/review-20261018/v-probe-v4/vprobe/run.json)、[窄提取记录](../../experiments/014/review-20261018/v-archive-extraction.json)。

源代码中另有下次运行前必须修正的研究计量缺口：

1. `chat_once`未请求流式usage，且先访问`choices[0]`会跳过标准usage-only尾块。人工CPU喂标准SSE已复现usage为空、吞吐None。先读usage再判断choices，并明确区分单请求与并发波次吞吐。
2. urllib timeout是socket超时，不是调用总墙钟；服务启动可独立等1500秒，driver只在阶段后检查预算。应共用绝对deadline、请求中检查余量，外层限制整个进程组，保证失败前已有证据落盘。
3. hello仅要求finish_reason非空，`length`或无final也会通过。需要区分“生成过token”“协议完成”“合法答案”，两档协议均通过才进入P2。P2解析本身会拒绝length，不能说已把截断题算正确。
4. 当前调用全部串行，C4只是服务端上限，尚未测并发吞吐；V阶段8K每档一个样本也不是T阶段32K对2×16K。
5. P2只保存截短原文，final网格未约束0–9及最大尺寸；应保存完整回复另附摘要，合法率口径须准确。

已修复的历史变量遮蔽、无pip路径不作为当前缺陷重复追责。具体行号与CPU复现见[v-code-review.json](../../experiments/014/review-20261018/v-code-review.json)，实际失败证据见[v-runtime-review.json](../../experiments/014/review-20261018/v-runtime-review.json)。本轮不因漏装主包而换模型、扩大token或继续升级整套依赖。

### 6.3 当前配额与下一步

10月18日13:20北京时间，当前CLI身份`ccjjllt`返回GPU total108000秒、used23601.973秒、reserved0，即剩**23.443896配额小时**，刷新10月24日08:00。官方SDK的timedelta转JSON使用`.seconds`、丢掉整天，曾把108000错误显示成21600；本轮以原始timedelta的`.total_seconds()`规范化，不能把序列化错误说成配额减少。见[规范化配额](../../experiments/014/review-20261018/current-quota-normalized.json)。该余额不能与记录里的`jiyujia`历史22.82小时合并或混用。

后续顺序收敛为：显式安装固定vLLM主wheel并修探针记录/预算 → CPU重打包核身份 → 在实际运行身份配额明确后，仅做一次有限V复验 → 真正生成且有有效吞吐后，执行既定T的P2两策略。有信号才进P6/P12和程序反馈。R已经结束，不再按旧文档重新制作R或开展评分臂；主C6不动，暂不新增正式提交。

按当前查询余额保留6小时后，最多可规划17.443896配额小时；以2.3系数约7.58小时L4×4会话上限，仍需执行前按实际身份重查。原10小时计划不能当未消耗的新预算再用一次。

用户随后要求单独交接，并强调专注研究、不纠结门控和测试。已整理[015研究交接](../plan/015-qwen38-reasoning-research.md)：继承R结论，以最少修正进入27B完整思考和程序反馈实验。本次只写交接，没有实施修正或启动新实验。
