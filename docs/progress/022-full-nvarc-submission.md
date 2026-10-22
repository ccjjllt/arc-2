# 022 完整 NVARC 4B 计分提交与已有27B配对进展

日期：2026-10-21，北京时间。用户已明确将唯一主任务收敛为完整4B实际计分submission，随后交付已有27B配对表；对应[022计划](../plan/022-full-nvarc-submission.md)。本轮开始时Git基点`31f6f53`，工作树干净。

## 1 执行开始与资源

23:38北京时间只读查询zhuating：可用**13.092435配额小时**，reserved=0，刷新为10月24日08:00北京时间。当前没有启动新GPU作业；快照会随启动归档。余额包含其他项目共享用量，开机前重查。

正在从历史快照恢复021 V3最小加载修复到临时目录，并准备E20及正式全量参数化包。上游/派生源码不重新放回仓库；私有运行包另记SHA。原4B模型、TTT、增强、搜索、排名和最终两槽保持参照方法，必要时钟与运行修复逐项列出。

已有27B候选固定为021 E20 V1：版本对齐2／27、18个真实预测、9占位、20次调用、总4063.551秒。它仅供本轮配对，不启动27B或训练。

## 当前交付状态

- 4B E20运行、评分：**V2 COMPLETE，完整27query实际评分3／27，raw／scored oracle均5／27**。
- 完整4B正式Notebook及比赛提交：**57504054已COMPLETE，公榜0.00；失败根因未确认，可靠正式起点仍未建立**。
- 4B／已有27B配对表：**已完成**，见`experiments/022/paired-e20-v2/paired-table.md`；27B独有1query，ReplaceSecond净增1，FillMissing净增0。
- 新公榜成绩：**0.00**（57504054）；历史最好30.14不变，未实现提分。

## 2 完整4B E20 V1启动

23:47:40北京时间，平台接受`zhuating/arc2-022-nvarc-e20` **V1**，kernelId `140048696`。启动前余额47132.767秒、reserved=0；内部6000秒、固定每task660秒、180秒收尾，CLI/SDK显式请求平台6600秒和NvidiaL4。不是比赛提交。

私有Notebook SHA `2952883da2e3464dfc22a17b6cb25b1ed958cf8142f463a27ace14f5b1f0f202`。固定输入仍ad12e…、20task／27query；原loader/decoder不改，完整方法保留原r256单epoch TTT、增强、DFS与kgmon选2。先复制本地盘再依次加载四worker，全部ready后开始求解；每task/全局截止是本轮预算适配，记录实际epoch/steps完成情况，不声称字节复现上游33.89。

每批DFS先保存原tokens及合法逆变换grid，原始池不参加选槽；原重评分池独立保存。由此区分生成不足、重评分未完成和选槽损失。包清单与启动凭据见`experiments/022/e20-package-manifest-v1.json`及`execution/nvarc-e20-launch-v1.json`。此刻仅确认作业受理，训练/候选/评分尚未确认。

正式参赛账号ccjjllt只读查询显示周GPU余额为0（used108343.492／total108000秒）、reserved=0，历史成绩30.14／25.69／12.50仍在。正式代码提交额度与GPU路径正在核实；不把普通CPU或Quick Save当成隐藏GPU运行，也不切换参赛身份。

## 3 夜间自主推进与30分钟监控

用户进一步明确要求每30分钟定时检查、修复错误，运行完成后继续研究；北京时间**2026-10-22 09:00前无法回复**。已创建并回查本聊天heartbeat `arc2-4b`，状态ACTIVE、间隔30分钟，目标thread `01a15df9-571f-7223-ad68-18ebb19ab7d4`。无变化不重复通知，有意义进展/完成/失败才报告；不得因用户离线等待常规确认。两项交付完成且无必要作业后停止监控。

最新日志：模型完整本地复制已成功，**7,267,271,302 bytes，374.271秒**；不是零字节阻塞，无需启用备用I/O方案。当前仍在启动/加载阶段，真实TTT与候选尚未确认。日志持续保存到`experiments/022/execution/nvarc-e20-v1-live.log`；SDK普通`kernels_logs`中途可能返回空，不能用空串推断无日志，可用CLI `kernels logs REF --follow`。

官方双账号独立认证已确认同队：ccjjllt和zhuating各自own-team接口都返回相同9份历史提交，队名ccjjllt、同rank、已参赛、当前共享1次可提交额度，详见`experiments/022/official-submission-readiness.json`。采用zhuating自有GPU正式Notebook，不修改团队或历史最终选择。正式包在`/private/tmp/arc2-022-nvarc-work/zhuating-final-notebook/`，ref `zhuating/arc2-022-nvarc-full-submission`，Notebook SHA `bd00fcf2d9ba5e13c1b87bc7903f98df4a5032f9e6f0c8d59becb14691dfc2a1`，核心与E20一致；尚未push。

夜间接续顺序：先查当前E20，不重复启动；若成功下载完整`022-run/`和submission（注意分页），冻结后用022配对入口读取4B scored/raw池与原27B固定gold指纹评分；按实际4B wall记成本。然后GPU正式Save & Run真实4题（ordinary2700秒），内部隐藏41400秒、平台显式43200秒，核对精确版本后由zhuating为原同队正式代码提交并等待公榜。若失败只修实际阻塞，不开新27B、训练或其他分支。

SDK认证使用既有凭据**文件路径**，不读取/打印凭据正文：`KAGGLE_API_TOKEN=/Users/zhaohuaxiang/.kaggle/access_token_zhuating`，解释器`/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python`；显式authenticate并核对身份。CLI为`/Users/zhaohuaxiang/.local/bin/kaggle`。push须传显式timeout，metadata同名字段不被当前CLI读取。CreateCodeSubmission的kernel_version是vN，不是scriptVersionId；后者用输出/kf/及提交列表URL绑定。

## 4 监控改为自适应间隔

用户进一步授权按等待长度动态调度。现有heartbeat `arc2-4b`已更新并回查ACTIVE，未新建重复任务；当前启动/加载阶段设**5分钟**。以后每次唤醒按真实阶段与实测ETA更新同一任务：启动/错误/接近结束2—5分钟，稳定中等等待10—15分钟，有可靠长等待依据时30—60分钟；无可靠ETA默认10分钟、临时网络失败5分钟。配置最大时长不能当作剩余时长。检查发现运行完成时，直接在当轮执行冻结、评分、配对或提交，不等下一次定时器再做已可完成的工作。夜间自主权限及用户09:00前离线安排保持不变。

## 5 00:05夜间检查：仍在导入，缩短复查

2026-10-22约00:05北京时间实查平台RUNNING。完整模型复制成功后，worker0从23:55:58的`worker_import_start`尚无`worker_import_done`，后续worker按顺序等待；未进入TTT或搜索，不能给占位评模型分。已独立审计导入链与013成功版本差异，当前不重复启动或改运行中包。900秒worker启动上限临近，heartbeat改为2分钟；若平台输出尚不可取，待结束后取完整worker日志再据实际错误修复，不凭慢导入猜根因。

## 6 V1启动边界失败，已修复并运行V2

00:11左右V1 driver结束为failed，平台随后ERROR。已下载14项产物并冻结到`experiments/022/execution/nvarc-e20-v1-outcome.json`：总1318.467秒，checkpoint本地复制374.435秒；worker0导入792.724秒，之后权重加载6.660秒、模型及PEFT完整ready共15.024秒。可用host RAM约191GB，未见内存不足证据。只有worker0 ready；worker1开始导入约92秒后，四worker共用的900秒阶段上限触发终止。**未TTT、未搜索、0真实候选；不评分为模型0／27。** 慢导入的具体模块仍未确认。

已直接修复该边界，不改模型、复制路径、求解、TTT或选槽：首worker900秒；每出现新的model_loaded，给后继300秒；4个全部ready后关闭加载超时；6000秒全局盒优先且不变。新增Torch／默认设备／Unsloth链的分段标记和60秒堆栈转储，并转发有限worker日志尾，便于实际定位慢调用。独立源码审阅未发现阻断，未新增测试。

**00:14:48北京时间V2已接受**，同slug `zhuating/arc2-022-nvarc-e20`、v2、kernelId140048696，Notebook SHA `ffa8ae0df80c260a2a05c19447c06009da50b0aa57bb8181bcb0a7e00061307d`。启动前实查可用44289.647秒、reserved0，上一版状态ERROR。启动凭据`execution/nvarc-e20-launch-v2.json`，配置清单`e20-package-manifest-v2.json`，实时日志`execution/nvarc-e20-v2-live.log`。此刻没有新解题结果。继续2分钟检查，正式包同步为`/private/tmp/arc2-022-import-trace-work/formal-v2-notebook/`，SHA `486cbe03b1acb69604e8969ae171e2031d402a9617d4ca0b36ca70cc01388f5c`，尚未push；旧正式包不再使用。

## 7 V2实际上传身份回查

00:16—00:19检查仍为RUNNING，尚在固定权重复制阶段，无新求解结果。发现打包目录在API调用前预核与SDK再次打开文件之间被准备者修改（仅日志堆栈摘录方式），故不能沿用先前ffa8/486c作已运行身份。已只读pull远端V2并逐cell对比、解包payload核验，与runtime boot全部一致：**实际Notebook SHA为9ab9e0f478283815a60fbc93aa4d0f5484ffab726a46b1f2516debfc06c27b59，runner为90a35e364f2e50d43da8b6d902951ce33ecac5e04cca005372431c51dd2efbfc**；solver、decoder、loader、copy与V1均未改。详见`execution/nvarc-e20-v2-remote-source-verification.json`。旧预核SHA保留为历史字段，启动凭据追加更正，未取消或重复启动GPU。

实际V2及同核心正式包已单独保留在`/private/tmp/arc2-022-import-trace-work/candidate-v2b/{src,e20-notebook,formal-notebook}`，准备者停止改动。**正式包改用该formal-notebook，SHA0ec71872ba7bf3206c9640a5448f4129e3276fd36e5e3d04e51b03620ee804eb**，与实际V2六核心逐项匹配；此前旧path/486c不再用于启动。后续push前先由主执行者复制到新的私有冻结目录，再核验并提交，避免共享打包路径再次变更；这是单次运行文件冻结，不新增工程门禁或测试体系。启动阶段继续短间隔复查。

00:23检查：V2权重7,267,271,302 bytes已在321.934秒完整复制，进入worker0导入；平台RUNNING。继续2分钟检查以读取新增细分事件/堆栈，尚无TTT或解题评分。

00:25检查：新增60秒堆栈明确停在`importlib.create_module → 挂载torch/__init__.py:416 → starter.py:31 importtorch`，尚无torch_import_done；已将慢点收窄到PyTorch扩展导入，不能再写成Unsloth/模型权重阶段。当前仍RUNNING且无新异常；V1首导入最终792.7秒完成，因此不重启当前作业，等待实际进展。下次检查改5分钟，临近900秒无进展上限时再缩短；正在只读核对精确Torch源码，不改包。

00:32检查更新：PyTorch与默认设备阶段已在194.63秒后完成；后续主线程栈继续推进到挂载Transformers的import-structure扫描及Python模块读取，不能把前一个60秒样本写成持续停在torch._C。后台compile-worker读取线程的等待不是主线程故障证据。平台仍RUNNING、未TTT；首worker窗口到约00:37:06，预计接近上轮首导入完成区间，改回2分钟检查。当前不换库、不stage新runtime、不重启。

00:35检查：worker0已完整ready，加载14.985秒；V2已依新model_loaded正确给worker1刷新300秒窗口（到约00:38:47）。rank1的Torch/默认设备阶段仅10.91秒，相比首rank194.63秒更快；后续依赖仍在导入。平台RUNNING，尚无TTT/候选，继续2分钟观察暖启动是否通过，不新增修复版本。

00:37检查：worker1已在约110秒内完成暖导入与完整模型加载（load_s10.963），300秒后继窗口通过，worker2正在导入。未出现新错误，继续2分钟等待全部ready及首次真实TTT。主线程同时已将同核心正式包复制到独占冻结目录`/private/tmp/arc2-022-formal-launch-frozen-v1`，SHA仍0ec71872…804eb，解包逐项匹配实际E20核心，记录在`execution/formal-package-freeze-v1.json`；仅作必要准备，未启动或提交，后续push只用此冻结目录。

## 8 四卡完成加载，首次真实TTT与DFS已确认

00:39:16北京时间四个worker全部ready，加载窗口依设计关闭；后三个worker分别约110.58、110.28、107.18秒完成暖导入及模型加载。四卡随后各取一题进入原版TTT，未改权重、训练、搜索或选择方法。00:42检查平台仍RUNNING，rank3的136b0064已完成128／128训练步、epoch1.0，实测TTT120.634秒；rank2的16de56c4已进入DFS批次。其余两题仍TTT，尚无冻结候选或可报分数。该结果证明启动修复后已实际进入解题流程，不是提分证据。

首批真实训练／推理阶段将检查间隔调为**5分钟**，等待首批候选和完整task输出；确认稳定解题后再按实测批次耗时改10—15分钟。当前不使用6000秒上限作实际ETA，未重复启动E20或正式作业；正式包继续保持独占冻结。

00:48检查：平台仍RUNNING，真实候选已覆盖7／27query，已落盘48项增强候选文件、评分池40个逐query去重候选、raw合法池61项，当前错误列表为空（均为运行覆盖统计，不是正确数）。16de56c4实测416.565秒完成，128步／epoch1.0及全部8／8计划eval批次均完成、13个评分候选；至少有一题明确完成TTT→DFS→重评分→落盘。其他worker已开始后续题或继续多query搜索。稳定阶段改**10分钟**，依据已完成题约7分钟及20题四worker吞吐，剩余为数十分钟量级，仍未把预算上限当ETA。正式Notebook独立只读审计未发现新启动阻断，hidden全test公平预算240题首轮约660秒，与E20强度接近；1200秒是单题上限，不等于逐题实得时间。等待完整输出后冻结评分，不提前访问gold或修改选槽。

00:59检查：平台RUNNING，候选覆盖14／27query，134项增强候选文件、评分池82个去重候选、raw合法164项；错误列表为空，未读gold。四worker持续进入后续题，6ffbe589已完整TTT及4／4搜索批次后0候选（实际生成不足，不能写成未运行）；其余题仍生成／重评分。继续10分钟，当前无需要修复的运行错误，不扩新实验，不提前报分。

01:09检查：RUNNING、18／27query已产生评分候选、错误列表为空，继续10分钟。8f215267和a25697e4均已完成128步TTT，但分别只尝试2／4和2／8搜索批次，随后为评分预留时间而停止DFS；a25697e4的后续query在首次DFS前已到截止。此类缺失须在最终配对中标为预算／调度覆盖不足，不能直接归因模型学不会并据此启动续训。当前不修改运行或正式冻结包，先取得完整结果；只读审阅多query的遍历顺序，避免把RawCandidateOracle的缺口错误解释为训练需求。

## 9 E20完整冻结、实际评分与配对结果

本轮01:08触发的执行中出现长时间本地中断，恢复后实际时钟已为07:47；不能把旧RUNNING响应或旧日志当实时状态。07:48重新查询平台为COMPLETE，随后直接下载、冻结、评分并推进正式包。中断原因未确认，没有声称夜间持续监控成功。作业自身从00:16:00.782到01:18:30.556北京时间，driver总**3749.775秒**，stop_reason=workers_finished、returncode0、4×L4，全部20task均经历实际流程；d8在660秒预算内只完成113／128步TTT，不能称所有题完整单epoch或全部增强完成。

已取全3分页446项远端文件清单，只下载研究输出398项及完整Notebook日志1项，排除022-code上游源码；共**399文件逐SHA冻结**并二次逐项复核，见`execution/nvarc-e20-v2-outcome.json`。精确V2 scriptVersionId **360845520**，Notebook SHA仍9ab9e0…27b59，6份核心SHA与远端核验相符。产物在`experiments/022/e20-output-v2/`，21／27query真实候选、6占位；raw合法234项、评分池122个逐query去重候选，错误列表为空。结构27query完整保留，占位照常进入分母。

固定沿用25项官方f328主版及d8两query完整输入匹配的旧版补充，没有改旧27B的主评分或缩小分母。实际结果见[完整配对表](../../experiments/022/paired-e20-v2/paired-table.md)：

| 政策／诊断 | 正确query／27 | 相对4B new／lost／net |
| --- | --- | --- |
| 4B原最终两槽 | 3／27（11.11%） | 基线 |
| 已有27B | 2／27 | 1／2／−1 |
| FillMissing（预声明） | 3／27 | 0／0／0 |
| ReplaceSecond（预声明） | 4／27 | 1／0／+1 |
| 4B ScoredCandidateOracle | 5／27 | 仅诊断上界 |
| 4B RawCandidateOracle | 5／27 | 仅诊断上界 |
| raw／scored与27B oracle并集 | 6／27 | 仅诊断上界 |

正确候选已被评分但未选入两槽：142ca369_0、3a25b0d8_0；raw正确却未进入scored的集合为空。27B相对4B final、scored、raw均只有8f215267_0独有正确。FillMissing需逻辑补题5task却没有正确净增；ReplaceSecond逻辑需全20task，完整既有27B成本4063.551秒及551365生成token，不能把+1直接解释为12小时部署有效。**本轮均为公开E20诊断，不是未见验证，不是Kaggle公榜，也未证明超过30.14。**

多query顺序只读复核：q0非转置两批→q1非转置两批→q0转置两批→q1转置两批，整task共用截止；a25697e4的TTT273.25秒、q0两批DFS114.95+185.84秒，q1首次轮到已停止，属于调度覆盖不足。d8两query均没有DFS；6ffbe589则已尝试全部搜索但无候选。budget_exhausted=false只说明整题截止未到，不能否定提前60秒DFS截止造成覆盖损失。后续只有在实际尝试范围内解释候选缺口。

## 10 正式包GPU普通运行V1已接受，尚未比赛提交

**08:39:40北京时间**平台接受`zhuating/arc2-022-nvarc-full-submission` **V1**，kernelId **140114268**，Notebook SHA **0ec71872ba7bf3206c9640a5448f4129e3276fd36e5e3d04e51b03620ee804eb**。只使用主线程独占冻结目录`/private/tmp/arc2-022-formal-launch-frozen-v1`，与实际E20六核心相同。启动前再次实查：GPU可用36614.121秒（10.17059配额小时）、reserved0；自有slug列表仅E20，无同名正式运行；同队提交仍9份、允许1次、没有pending。启动凭据`execution/formal-save-launch-v1.json`。

实际push显式NvidiaL4、平台timeout43200，私有断网；ordinary2700秒含180秒收尾并真实求解4题，hidden41400秒含300秒收尾、全实际test、公平逐题预算上限1200秒。隐藏是否扣周配额仍未确认，不写成免费。当前平台RUNNING，权重复制推进，实时日志`execution/formal-v1-live.log`；下次检查目标改为这个正式V1，**5分钟**启动验证。E20不要重复启动。远端源码正在独立只读核验。

接下来必须等正式ordinary COMPLETE，冻结实际GPU输出、源身份和精确scriptVersionId，再立即重查同队额度／pending并用competition_submit_code指定该V1创建正式submission；尚无新submission ID或新公榜。依据本轮5→3的选槽损失，主交付完成后的下一轮只选择**候选选择**小实验，先复用现有候选做无新增GPU的离线研究；不开4B续训或27B全量推理，调度覆盖问题单列为解释限制。任何改进结果仍须新运行／比赛成绩确认。

08:42补记：正式V1远端已逐cell/config/六payload源核验，与冻结包字节一致，只有外层Notebook JSON序列化不同；见`execution/formal-v1-remote-source-verification.json`。待COMPLETE后从output URL绑定scriptVersionId，当前不据kernelId代替该值。

08:43监控外部状态变化：原automation.toml已不存在；automation_update(view)只渲染卡片，随后按相同id／原完整字段更新5分钟明确返回“Automation does not exist ... may have been deleted manually”。未另建重复任务，未手改配置；当前回合继续检查正式V1，不能再声称后续定时唤醒已保障。此事与Kaggle正式作业运行状态分开记录。

08:53配对独立审计完成，记录在`experiments/022/paired-e20-v2/audit.json`：399冻结文件SHA全部匹配；83份raw-grid批次导出与汇总池一致，raw去重后的122个grid与scored集合逐query完全一致。20／20task均start/done，19／20完整训练epoch、14／20尝试全部计划搜索批次；这是受预算约束的流程完成，不是穷尽候选能力。两项已评分正确但未选中的grid在原kgmon池分列第5／9与第3／8。此审计未新增推理，也未独立重新解码每个原始token beam；不把5／27当穷尽模型能力。

09:07左右正式ordinary平台COMPLETE，driver1552.237秒，实际4×L4、4selected task／7query，90项研究输出+完整log1项已全量下载并SHA冻结。精确Notebook V1的**scriptVersionId360953982**，见`execution/formal-v1-outcome.json`。输出保持120task／172query结构合法，其中选中4题实际生成4query候选，其他占位不能作模型全量成绩。4题均训练128步；长题d8训练797.522秒后实际搜索和重评分，ordinary不再是CPU空跑。

09:08提交前只读复核精确V1／kernelId140114268、GPU开且断网、同队9份提交／允许1次／pending0；GPU可用33454.673秒、reserved0，隐藏扣费仍未确认，见`execution/pre-code-submit-readiness-v1.json`。

用户09:07明确回复**不恢复定时任务，跑完后由用户通知**。保留原任务缺失状态，不另建；完成当前版本核验与正式提交后停止主动／定时轮询，不将用户不恢复定时器误解为取消已经授权的本次正式提交。后续score须在用户通知后实查，不提前宣称。

## 11 正式比赛提交已受理57504054；按用户要求停止轮询

**2026-10-22 09:10:48北京时间**，SDK competition_submit_code返回真实ref **57504054**。同队own-team列表随后回查绑定`zhuating/arc2-022-nvarc-full-submission` V1／**scriptVersionId360953982**，submittedBy=zhuating、teamName=ccjjllt；没有改团队、重新接受规则或改历史最终选择。提交前再次检查允许1次、没有pending、没有同精确版本已提交记录，仅创建1次。凭据`execution/competition-submission-v1.json`。

09:11:56的单次GetSubmission初始核验再次确认真实ref、账号和队伍，仍未返回status/publicScore字段；因此只能报告**正式已受理、公榜未确认**，不擅写SCORING或COMPLETE，不承诺超过历史30.14。详见`execution/initial-submission-status-v1.json`。随后按用户要求停止主动及定时轮询，等用户通知运行完成后再查真实公榜或错误。4B E20评分与已有27B配对表已经交付；正式计分结果仍待平台。下一轮候选选择实验尚未运行。

正式ordinary独立核验14／14通过，blocking_findings为空，91冻结文件及合法120task／172query输出均复核，六核心与远端审计完全一致；见`execution/formal-v1-output-verification.json`。已提交的是同一V1，ordinary占位不能当隐藏全test输出或正式成绩。

## 12 用户通知后回查：正式COMPLETE，公榜0.00，交付失败

2026-10-22 **10:38北京时间**，收到用户“跑完了你看看”后重新实查：GetSubmission与同队提交列表均返回 **57504054 COMPLETE／publicScore 0.00**，teamName=ccjjllt，精确版本仍V1／scriptVersionId360953982，正式文件totalBytes=51179，未返回errorDescription。记录`execution/submission-57504054-result.json`。这是实际零分，未超过历史最好30.14；不能把接口COMPLETE称为模型全流程成功或提分。

独立SDK审计`execution/submission-57504054-sdk-observability.json`确认：拥有者zhuating认证后的DownloadSubmission实际HTTP403，episodes为空，GetSubmission不暴露隐藏session ID、真实耗时、逐题覆盖或运行日志；未取得隐藏预测。普通Notebook输出58657 bytes，与正式51179 bytes不同，但字节大小不能证明hidden分支、全量任务或网格内容。

浏览器只读官方同队提交页亦显示Succeeded／0.00，Submission Details只提供分数、描述和最终选择框，未改描述或最终选择。官方响应给出的raw网页URL在已登录ccjjllt会话显示404。随后可见的拥有者zhuating Notebook Logs显示Successfully ran in1579.7s、boot.mode=ordinary、120task／172query、4真实query／168占位；它与已冻结普通运行一致，**不是本次隐藏运行日志**。未把该页面作为hidden完成证据；未在用户同时操作浏览器时继续导航。

额外只读核对当前官方可下载公开test：240task／259query，SHA232264c58f825ee77327dcfc9f4e5cb2f83b8d997eb69032be1bf2205bbe1a83；按本输出格式全占位为40027 bytes，也不等于正式51179，不能据大小断言全占位。见`execution/public-test-structure-result-audit.json`。10:49 GPU配额仍used74545.327秒／reserved0，与09:08提交前一致；没有配额耗尽证据，也不能单靠此断言隐藏GPU执行与否或免费计费，见`execution/zhuating-after-submission-result-account.json`。

静态复核：正式rerun flag解析、hidden读取test全任务路径及fair任务预算代码存在；ordinary GPU已验证并不能证明这些分支实际在本次评分执行。运行错误原则上会非零抛出，但全局截止允许保存当前预测后返回成功；缺少真实hidden产物时不能锁定为加载失败、路由错误、占位或模型能力下降中的任何一种。当前根因**未确认**，不凭猜测修包再烧一次GPU或提交。

**纠正当前下一步优先级**：原定的候选选择小实验暂缓，先补齐正式隐藏实际执行／输出证据，建立可用的正式起点。原3／27、oracle5／27、27B固定融合+1仍只适用于已冻结公开E20，不解释本次0.00。此前将普通四题成功视为足以保障正式交付，证据不足；完整4B的可靠正式起点尚未建立。当前未新建作业、未再次提交、未训练、未恢复监控。若现有平台权限仍取不到隐藏产物，下一次正式包须先明确其全量运行与有效预测的可观测证据，再考虑能力改进。
