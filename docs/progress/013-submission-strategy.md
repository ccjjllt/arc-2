# 013 — 提交策略与最新比赛信息：调研记录

查阅日期：2026-10-12，北京时间。对应[下一步计划](../plan/013-submission-strategy.md)。本轮只读Kaggle API、官网、公开Notebook和本地研究资料；新增文档/证据，未修改模型或求解代码，未运行GPU、训练、推理或正式提交。

## 1 比赛目标的实时参照

官方API返回2783队，账号已参赛、公榜第876。最终截止2026-11-02 23:59 UTC，即北京时间11月3日07:59；入队和合并截止为北京时间10月27日07:59。每天最多1次提交、最终可选2份，无网Notebook最多12小时。

浏览器实际读取[Kaggle奖牌规则](https://www.kaggle.com/progression/competitions)：1000队以上铜牌前10%、银牌前5%、金牌前10+0.2%，百分比向下取整。按当前队伍数只是估算：铜278、银139、金15。队伍数会变，同分排名也要计入。

| 目标 | 当前估算名次 | 本次公榜该位置分数 |
| --- | ---: | ---: |
| 铜牌 | 278 | 31.81 |
| 银牌 | 139 | 32.22 |
| 金牌 | 15 | 41.81 |
| 排名奖金末位 | 8 | 62.08 |

以上为本次[公榜](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)位置快照，不是最终奖牌分数线；第15/16、139/140、278/279位置均有同分。最终成绩由私榜及资格规则决定。不能说拿到相应数字就稳获奖。

[比赛规则](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)列排名前8共享$275,000（第8为$15,000）；另设$275,000研究writeup奖和有条件的$150,000 bonus。研究奖不是参加或整理报告即有奖金，要按准确率、普适性、进展、理论、完整性、新颖性评审，并要求开放成果。规则称Innovation Prize，Prizes页称Grand Prize，名称差异保留。开源许可具体措辞沿用001的待核对事项，不在本轮自行统一主站与Kaggle措辞。

原始摘要与页面快照：[competition-snapshot.json](../../experiments/013/competition-snapshot.json)。

## 2 账户成绩与可提交资产

Kaggle账户接口实读7次历史提交，均COMPLETE，最高30.14唯一对应submission ref52072735、2026-04-26、[指定scriptVersionId314617319](https://www.kaggle.com/code/ccjjllt/reproduce-nvarc-2025-results?scriptVersionId=314617319)。其余最高29.72。没有27B提交，也没有10月新提交。精确版本源码仍未取得并核实。

同Notebook latest v21已COMPLETE、有submission.json，使用Sorokin4B与离线依赖。实际普通运行只处理4task/5query，rerun=False，约2314秒；源码有全量隐藏读取与四worker。该版本没有绑定的新公榜分数，不能继承旧30.14，也不能拿4题运行证明全量12小时。

v21静态审阅发现：trainer.train和重评分没有充分deadline检查；base截止后两轮rescue仍可能重新加载；每题1200秒的上限不能保证240题四worker都得到预算；救援候选把beam score当score_aug，与原增强评分混合；diverse_dual只保base第一候选，不保原两次尝试的覆盖。全task/query初始化为[[0]]只保证格式，不代表实际解题覆盖。因此不建议立即原样提交v21。见[静态审阅](../../experiments/013/baseline-audit/v21-static-review.json)、[账户记录](../../experiments/013/baseline-audit/account-submissions.json)。

27B现有solve_grid/runner是研究输出payload、journal与SUMMARY；尚无已验证的正式双attempt导出和全量12小时Notebook。012的0/14、0/2及无完整程序不能当新主提交的依据。

## 3 新发现的提交排队约束

[Kaggle工作人员10月8日公告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/747515)说明，在每日1次之外，每队只能有一个pending/scoring提交。10月10日工作人员报告当时队列约33小时；这是历史队列快照，不是本次实际等待预测。不能计划每天稳定收到一个新分数，也不应把第一次有效新版本拖到截止前。

## 4 同一27B的外部新证据

[ARC Prize官方结果](https://arcprize.org/results/alibaba-qwen3-8-27b)列同模型ARC-AGI-2 Semi-Private：xhigh42.4%、low22.8%、medium13.2%、none1.5%，使用Baseten专用推理。它不等于Kaggle分数或4×L4的12小时表现，但提示关闭思考/极短输出可能显著改变能力，不能只把当前nonthinking小片表现当模型上限。

[10月10–12日讨论](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082)有参赛者自报原27B公榜26.81、公开agent Notebook约15；这些是作者自报、未复现，未确认精确版本绑定，不能当我们的效果。其价值是提供可读实现。

实际已取得源码：

- [madarshbb/arc-qwen-3-8-27b-agent](https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent)，v2：虽然标题为agent，实际配置mode=direct；FP8权重、vLLM TP4、并发10、max_num_seqs16；49152输出/40960思考预算，另有关闭思考的答案补救。代码存在不代表这些设置均被服务识别或已在我们环境跑通。
- [russcore/arcagi2-qwen38-freeform-direct](https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct)，v17：FP8、vLLM0.19、TP4、并发24；Python工具辅助后输出网格。源码注释自报旧v7约255 aggregate token/s；是作者历史注释，不是本次运行，更不能与本项目4.54单请求token/s直接作等口径倍数比较。
- [khoinguyennguyen的TIR Notebook](https://www.kaggle.com/code/khoinguyennguyen/discovery-by-acc-of-observations-arc-agi2-tir)：接口返回读取权限拒绝，未取得源码，不推断实现。

两份可读27B源码均无运行输出；都挂同一第三方FP8 repack v1，实际量化配置和来源仍待核实。russcore的metadata没有有效wheelhouse挂载，而源码要求它；默认保存运行只输出floor，不运行模型，必须显式diagnostic或hidden rerun才加载。madarshbb的MTP/FP8 KV是多级尝试与回退配置，不能称实际已启用。详细记录及只读源码见[公开Notebook审计](../../experiments/013/qwen-public/AUDIT.md)、[SHA清单](../../experiments/013/source-manifest.json)。

另读Failed in AIMO v2、Learned from AIMO v7、Minimal Perfpatch v2源码：均不能凭标题、搜索排序或历史版本分数宣称当前更高分。Failed in AIMO含程序支线，但模型挂载只列专用4B；Minimal Perfpatch主要减少CPU logit传输，不是新的解题能力证据。

决定：将“公开高吞吐推理配置审计＋足够思考预算下的小片效果”提升到下一步优先项；012的简短代码提示假设保留为小诊断，不让它独占下一周。仍固定Qwen3.8-27B，不另展开模型选型。

## 5 配额与研究状态

本轮API返回GPU total108000秒、used77838.631秒、reserved0，剩8.37816配额小时；刷新北京时间10月17日08:00。较012末快照减少0.25939小时，原因未核实，不全部归因于某个作业。本轮没有启动GPU。余额为查询时快照，后续开机前重查。

已完成：官方比赛/奖牌/奖金/队列核实；账户成绩与版本关系核对；公开Notebook静态阅读；本地提交缺口审计；013计划与本记录。

未运行：新基线修正、v21新提交、27B量化服务复现、吞吐比较、程序新提示、互补评价、C6/C12、全量12小时验证。没有新分数或获奖概率估计。

以上为调研结束时状态；下节记录用户授权后的实施，不能继续把已实施项说成未实现。

## 6 10月12日：用户授权落实近期提交

用户明确要求“那你来帮我做吧”。本次先落实可提交基线，27B只继续CPU依赖核实，不在本周期叠加第二个GPU实验窗口。

已在`experiments/013/submission-src/`建立独立候选，不覆盖历史Notebook或012代码。保持Sorokin4B、原LoRA r256、训练增强16/推理增强2、min_prob0.20、DFS上限540秒；使用base-only和统一kgmon选2，移除多轮救援与多样性奖励。它是基于已审阅latest v21的新完整配置，不是精确复现314617319，也不是已经证明提分的版本。

实际修复：

- 单个模型worker/卡，一次加载后服务队列；任务预算按剩余工作量分配。训练保留原参数，新增自然步截止，默认不超过480秒或任务预算40%；保证有真实训练步才产出候选。
- 解码、重评分按截止准入；只保存完整真实增强NLL，不用beam分数冒充重评分。逐候选原子保存，保留源SHA、训练步数和实际运行环境。
- 纯CPU输出器严格核矩形/整数/颜色，完整task/query和双attempt；初始保存合法兜底，每分钟及结束时增量重建。重建失败保留上一份合法文件，实际候选覆盖和兜底数量分开。
- 父进程覆盖安装、加载、运行与落盘；可终止整个子进程组。异常kill记失败，主动预算停止单独记原因。普通保存总45分钟留180秒，隐藏运行12小时留600秒。
- 保留原Docker image、Sorokin模型v1及Utility Script依赖（输出身份272622465）；实码区分已安装包与wheelhouse，同时支持Utility在`/kaggle/usr/lib`和`/kaggle/input`的挂载路径。

人工CPU核对完成：1000组随机候选与原kgmon选2一致；非法/损坏/临时文件、多个query与重建异常；真实子进程组清理；模拟正常退出、意外SIGKILL、预算截止、setup失败、worker错误的主流程。只使用合成数据，没有新增测试套件。CPU结果不证明CUDA/模型/解题效果。主流程记录见[诊断JSON](../../experiments/013/execution/cpu-supervisor-diagnostic.json)。

本地包v1/v2只为未发布审稿，最终冻结包为`experiments/013/kaggle-baseline-v3/`，Notebook SHA `3967414122a8b6552230b9e03b3c153972bdb149c771c523fbf12f1f436fe81f`。五源码内嵌字节和SHA与当前实现一致。

开机前重新读账户：8.378158配额小时、reserved0；45分钟按2.3预留1.725小时，剩余约6.653158小时，高于6小时储备。见[开机决定](../../experiments/013/execution/prelaunch-decision.json)。仅同v21原4道开发诊断题、全部query，不读取C6/C12答案。

CLI已成功发布[新私有Notebook](https://www.kaggle.com/code/ccjjllt/arc2-013-bounded-baseline) **远端version1**，API确认RUNNING、NvidiaL4、无网和原数据源。首次sandbox CLI在DNS阶段失败，没有发布；授权网络调用成功。见[发布记录](../../experiments/013/execution/baseline-v1-publication.json)。此时尚未收到GPU结果，尚未提交比赛。

### 启动后发现的划分交叉

在准备事后评分时，CPU交集检查发现：继承v21的4题中，`981571dc`也在011的`confirmation_C6_reserve`。启动前只沿用旧诊断题清单、没有先与本项目确认清单做交集，是本次遗漏。主C6交集为空。见[实际交集与处置](../../experiments/013/execution/pilot-split-intersection.json)。

该题已进入013开发推理；其query答案没有读取，本次不对它评分。覆盖和时间仍按全部4task/5query报告；如评价诊断准确率，只能另列其余3task/4query，不能混用分母。旧C12不能继续称为全部未运行；主C6仍保留，备用题如需替换须另行按输入元数据冻结，不能静默替换后沿用旧确认名称或成绩。此事不构成新模型效果结论。

事后评分材料只从ARC Prize官方固定revision单独取得`0934a4d8`、`36a08778`、`aa4ec2a5`三个文件；先确认它们与主/备用确认块均无交集，再核对train pairs及query inputs与本地公开题面一致。没有下载整份evaluation solutions，也没有取得981571dc答案。来源与SHA见[pilot-scoring-sources.json](../../experiments/013/execution/pilot-scoring-sources.json)。

### GPU等待期间完成的27B CPU准备

[依赖准备记录](../../experiments/013/qwen-public/dependency-readiness.md)已确认模型v1、wheelhouse v1及overlay v2可取得；官方固定revision的小配置/模板/许可证与repack逐字节一致，索引1606个tensor键一致。未下载或核对权重张量，不能据此证明数值等价。

明确发现vLLM0.19.0包元数据要求`transformers<5`，公开overlay却使用5.8.0，属于待验证的兼容性例外。已有wheel和源码不等于L4可运行；只完成小文件、锁文件及模板的CPU核对，没有安装新栈或启动第二次GPU。

## 7 验证结束：真实候选覆盖完整，诊断2/4

2026-10-12用户报告结束后，Kaggle状态核实为COMPLETE；父监督complete、workers_finished、返回码0。监督墙钟1508.280秒，即25.138分钟，低于45分钟。四worker自然退出，未见OOM或任务错误、全局截止中断。

运行使用4×NVIDIA L4，实际Torch2.8.0、Transformers4.55.4、Unsloth2025.9.7、Unsloth Zoo2025.9.9、PEFT0.17.1。五源码、输入SHA与冻结包一致。窄下载产物在`experiments/013/kaggle-baseline-verified-v1/`；首次广下载包含编译缓存，另保存在`kaggle-baseline-output-v1/`，不是第二次GPU作业。

| task | 真正训练步数 | 训练阶段秒 | 全task秒 | 实际query覆盖 |
| --- | ---: | ---: | ---: | ---: |
| 0934a4d8 | 128 | 399.201 | 565.699 | 1/1 |
| 36a08778 | 128 | 208.848 | 1143.759 | 2/2 |
| 981571dc | 75 | 482.552 | 877.974 | 1/1，答案未读取、不评分 |
| aa4ec2a5 | 128 | 308.425 | 617.843 | 1/1 |

981571dc在480秒训练限额附近自然步停止，超过边界约2.55秒，随后推理完成；这不是完整128步，也不是整个任务失败。其余三题均完成128步。四任务并行，阶段秒数不能相加当会话墙钟。

共75个候选文件、92个有效候选、无损坏或非法；5/5 query均有实际候选，选出9个真实attempt，981第二槽使用合法兜底。完整JSON有120task/172query/344attempt，其中167query是普通验证范围外兜底，不代表正式隐藏全集已经推理。隐藏评分会重新运行version1的test全集分支，不能把本地pilot JSON按静态预测上传。

仅事前明确允许评分的3task/4query，最终pass@2为**2/4**：36a08778 query0和aa4ec2a5正确，0934a4d8及36a08778 query1错误；命中均在attempt1。该小片用于开发与运行诊断，不是公榜分数，不与旧v21的task加权2.5/4直接比较，也不据此宣称提分。

[离线结果与身份](../../experiments/013/execution/baseline-v1-outcome.json)保存全部事件、训练数、覆盖和评分口径。独立只读复核确认JSON严格合法、代码/输入身份一致和2/4计数。

每卡139个约10秒采样点（setup后到结束，混合加载/训练/推理/空闲）平均GPU利用率为51.83%/36.82%/52.55%/34.25%，采样显存峰值13702/15270/18208/14870 MiB；不是decode单独利用率或张量allocated峰值。见[产物SHA及GPU摘要](../../experiments/013/execution/baseline-v1-artifact-summary.json)。

账户used77838.631→80910.275秒，前后差3071.644秒＝**0.853234配额小时**；剩**7.524924小时**、reserved0。它是账户差，不是隔离逐设备账单；6小时储备仍在。见[结束后配额](../../experiments/013/execution/post-baseline-v1-quota.json)。

## 8 已正式提交冻结version1，尚无新分数

北京时间2026-10-12 **17:56:36**，根据此前用户授权，通过Kaggle代码竞赛提交接口提交同一个未修改version1；未启动第二次开发GPU或重跑普通Notebook。

- Submission ID：**57103498**。
- Notebook：[ccjjllt/arc2-013-bounded-baseline，scriptVersionId357613526](https://www.kaggle.com/code/ccjjllt/arc2-013-bounded-baseline?scriptVersionId=357613526)。
- 文件：`submission.json`；提交方式为指定Notebook/version的隐藏重跑，不是把pilot预测静态上传。
- 创建接口返回ref，账户列表已出现该记录；查询时尚未返回status/public_score字段，不自行判断已排到机器或已开始运行。
- 保留历史30.14记录，未修改旧提交或最终两份提交选择。本次只得到受理确认，不能说比赛提分。

见[正式提交记录](../../experiments/013/execution/competition-submission-v1.json)。下一步读取这个submission的评分/运行结果，不重复提交；27B仍先完成独立依赖兼容准备。本轮没有新增确认集成绩。

## 9 10月13日上午状态复查

用户询问为何尚未完成。北京时间10月13日约11:25，官方CLI明确返回submission57103498为`SubmissionStatus.PENDING`，没有publicScore或错误描述；普通Notebook仍为COMPLETE。SDK枚举PENDING=0，是默认值，解释了MCP JSON此前省略status字段；这个状态不能进一步区分排队和实际执行。

Kaggle Staff在[北京时间10月12日21:58的回复](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/747515#3537405)表示当前容量大约支持每队隔天一次提交。这比旧33小时快照更新，但不是本作业的固定ETA。普通保存与正式提交使用不同资源池；12小时执行限制不表示提交后12小时必出分。

本次仅只读查询，没有取消、重投或新开GPU。证据见[状态记录](../../experiments/013/execution/submission-status-20261013.json)。

## 10 10月17日：正式完成，公榜25.69

本次查询submission57103498返回`COMPLETE`、`public_score=25.69`，比历史复制提交30.14低4.45个百分点。这是4B备用线新配置的结果，不是27B比赛成绩。以上“尚无新分数/PENDING”为当时历史状态，当前已由本节更新。

014审计确认本轮不止改变运行边界：训练按题预算40%截断、DFS按批均分、救援删除和选2变化均影响算法。4题pilot每题1200秒；240题、启动360秒的源码算术场景中开局约695秒/题、训练278秒，不能由pilot2/4推出隐藏提分。隐藏预测下载被接口拒绝，实际隐藏覆盖/步数/错误分布未取得，具体退分因果仍未确认。

新研究、官方榜单与配额快照见[014进展](014-regression-and-reasoning.md)、[原始状态](../../experiments/014/evidence/current-state.json)。保留历史30.14，不重复提交相同25.69版本；本次未新增GPU或改变最终提交选择。
