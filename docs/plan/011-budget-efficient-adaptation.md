# 011 — 30小时配额下的有效适配与推理研究：实施交接

日期：2026-10-10（北京时间）。对应[实际进展](../progress/011-budget-efficient-adaptation.md)。本文件是本轮唯一执行计划；experiments/011 下两份研究备忘是依据，不另叠加其中的候选预算。固定 Qwen/Qwen3.8-27B、4×L4、BF16 与12小时提交边界。

**10月11日执行补记：用户随后明确要求由助手操作，核心入口与CPU预检已完成；S0 v2和S1 pilot v1已结束并收回结果。F0-25为0/14、多视角完整配对0/5、TTT A0/A4均0/2；程序4次均在思考阶段截断，repair未运行。当前停止GPU扩算，下一步先CPU准备吞吐和程序完成性探针，详见progress §11–14。以下保留10月10日研究设计；23小时预算及“由用户实现”的原分工已被后续授权替代，新一轮最多14配额小时、至少保留6小时。最新余额15.23138小时、reserved=0（10月11日15:00北京时间），不是必须花完的预算。C6/C12仍未运行。**

## 1. 本轮决策与要回答的问题

下一轮不默认继续同400例的50/100步离线SFT。先用已有权重完成F0-25评价，随后研究：

1. 同一初始化下，第二个变换视角能否提供真实新增解？
2. 少量题内LoRA适配能否胜过相同时间的无适配候选？
3. 原底座生成程序后，一次真实示例反馈能否胜过独立重试？
4. 在不改算法目标的情况下，共享加载及有限GDN核优化能节省多少端到端成本？

前三项以未用于本项目微调的query完整网格命中为主终点；第四项以完整工作耗时和输出变化为终点。小片结果只支持预算决策，不预测比赛分数。若全无信号，停止机械扩步，剩余配额可以保留，先做CPU数据质量研究。

“GPU利用率”在这里优先指每个配额小时得到多少完整、可解释的对照与新增解题。nvidia-smi百分比仅用于定位瓶颈；让错误解码或重复加载占满GPU不是研究收益。

## 2. 必须继承的真实起点

原始统计、来源SHA与计时口径见[成本审计](../../experiments/011/local-cost-audit.json)。

| 已有工作 | 已确认结果 | 本轮处理 |
| --- | --- | --- |
| F0底座/G1的P12评价 | 各0/14，完整配对；生成分别44.9/51.8分钟 | 直接复用，不重跑整个E0 |
| F1底座/25步/50步P12 | 均0/14；各次生成约22.9–24.3分钟 | 复用；不fresh重训25/50步 |
| F1-25 | 25更新、100例，更新计时17.4分钟 | 与F0同100例/同顺序，保留权重 |
| F0-suffix-25 | 25更新、100例，更新计时37.6分钟，无OOM | 先补P12评价，不重训 |
| S4完整日志 | 加载3308.9秒；到S4_DONE为5666秒 | 加载占58.4%，必须合并小工作到同一驻留底座 |
| 重复训练前缀 | F1-50重跑前25步、S4重跑前17步，合计41.1分钟更新计算 | 以后同阶段续跑恢复optimizer/RNG/cursor，别fresh重来 |

上述训练计时不含加载、保存和失败更新；生成计时含prefill和decode。约4.3 token/s是历史网格生成的综合有效速率，不能当新程序/长思考的速度承诺。

研究边界：

- F0/F1的25步实际100条episode ID与关键配置一致，但F0用suffix、F1用full；初始张量未逐值核对，不能叫数学上完全相同的训练。
- F1三次均在21897d95 q0触及960 token，实际生成31×30；不是简单“答案太长”。aa4ec2a5的解析失败是另一题。不得使用query真值尺寸裁剪输出。F0/F1首先比较完整配置效用，不把所有差异归因于去空格。
- suffix数值探针的实际长度为990/1212/2072；F0已完成前100例最长3844。不是全部4K场景或完整400例都已验证。
- P12的0/14不能证明模型能力上限；我们尚未验证TTT、程序反馈或充分专项训练。

### 2.1 F0权重身份：开GPU前解决

本次允许访问的Notebook为 ccjjllt/s4-f0-train-010，metadata id=137317263，current_version_number=1，输出版本目录ID=355735798，列出了010/s4-f0/adapter_step_25/{adapter_model.safetensors,adapter_config.json,formatter.json}。其日志SUMMARY与本地归档逐字段一致，25行打印指标按显示精度一致。

本地kernel-metadata却写 ccjjllt/s4-f0train-010；该对象本次返回permission denied，未继续尝试访问。**保留两个slug的区别。** 使用当前有访问权的对象，先下载小配置、train_log及step25权重，核对100条episode、参数、SHA后再登记。仅有文件列表与相同SUMMARY，不能声称远端权重哈希已确认。本轮未下载权重。

冻结模型Kaggle挂载 ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1；模板参考revision 1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0。CPU准备权重/分片身份清单，避免在每个query重新哈希约55GB。

## 3. 实际配额、预算和作业组织

2026-10-10 09:34:59查询：总额108000秒，已用2212.124秒，reserved=0，剩105787.876秒，即**29.3855配额小时**；下次刷新10月17日08:00。原始响应见[runtime-evidence.json](../../experiments/011/runtime-evidence.json)。

官方仍写L4×4两倍扣额：名义剩14.6928小时四卡机器墙钟。计划采用2.3作为保守换算系数，**不是实测稳定倍率**；E0历史账户差仍有未解释部分。30h账户配额、四卡墙钟、12h单Notebook上限是三个口径。

| 阶段 | 机器墙钟上限 | 按2.3估算配额 | 内容 |
| --- | ---: | ---: | --- |
| S0：已投工作闭环与效率探针 | 2.5h | 5.75h | 一次加载、F0-25 P12；少量状态隔离检查；有条件的GDN探针 |
| S1：多视角与题内适配P6 | 2.5h | 5.75h | O/S/T覆盖；0/4更新TTT；有条件续8更新；时间对照 |
| S2：程序反馈P6 | 2h | 4.6h | 先P2可完成性，再P6；初次程序、独立重试、一次反馈修复 |
| S3：冻结方案确认 | 2h | 4.6h | 至少C6完整配对；预估放得下才做完整C12 |
| S4：必要补缺与产物归档 | 1h | 2.3h | 只补已定义对照，不新开超参扫描 |
| 预先承诺合计 | **10h** | **23h** | 允许提前停止；并非要求花完 |
| 未承诺储备 | 约2.776h的条件算术 | **6.3855h** | 慢加载、账差、确认完整性；不预花 |

每阶段上限含其新增安装、加载、JIT、计算与保存；共享首次加载只记一次。不能在表外再加“安装时间”。储备只能在已查实际账户差后，用于保住有价值的完整对照；算法无信号时不用于盲训。

尽量1个首轮驻留会话完成S0/S1/S2，参数和分支条件在启动前写定；人工看结果后退出，再用第2个会话执行冻结确认。GPU不等待人工读论文、修依赖或讨论。单会话硬上限低于12h，建议首轮不超过7.5h、确认不超过2h；其余留保存与补缺。若某入口未准备好，跳过该支线，不让GPU空等实现。

### 3.1 截止与慢加载策略

- Notebook从首个cell开始计时；每阶段记录START/DONE和心跳。加载预算**默认20分钟**，由父进程/Notebook监督；超限则退出整个本次GPU工作并保留日志，不在同一会话循环重新加载。此上限是止损选择，可能中止本可最终完成的慢加载，需如实记录。
- 新会话首次必要加载可仅试一个候选：TF5.8的HF_DEACTIVATE_ASYNC_LOAD=1。源码默认已异步，不使用旧HF_ENABLE_PARALLEL_LOADING变量。不为加载做两次完整55GB A/B；历史冷热/I/O条件不同，只能记线索，不能宣称因果提速。若用户选择保留默认异步，照样记录与限额，算法对照不依赖这个候选成功。
- 一次超时后先CPU检查实际版本、挂载路径、读取/内存迹象，再决定下一次不同配置；不自动花储备重试。病因未确认，不把“冷存储”写成定论。
- query/candidate开始前按已测长度成本做准入，留至少5分钟保存余量。生成内部需deadline停止条件；已中断输出标incomplete，不当合法完成。optimizer只在完整update边界保存退出；真卡死才由外层硬截止终止。
- 每个候选结束立即落盘。每次作业结束核对账户used/reserved与完整墙钟；出现更高扣额，缩后续预算，不能靠固定2.3系数保证不超支。

## 4. 评价清单与0/14的处理

[research-manifest.json](../../experiments/011/research-manifest.json)本轮已在CPU生成，仅含输入侧元数据、task/query ID，无query答案；继承010，原清单未改。

| 集合 | task/query | 用途 |
| --- | --- | --- |
| P12 | 12/14 | F0闭环与开发扩展；反复使用，不能称盲测 |
| P6 | 6/6 | 继承010 TTT_P6，按S/M/L、H/E交错；所有先导看同题 |
| P2 | 2/2 | P6前两题，仅查程序分支能否在预算内产出可执行最终程序 |
| C6 | 6/8 | 从原C12每来源、每长度层取原顺序第1题；冻结方案后完整确认 |
| C6-reserve | 6/8 | 同层第2题；C6+此组才是原C12，12/16 |

C6的缩小是在没有读取其答案/结果的情况下按原输入清单决定，目的为保住完整配对。进入确认前冻结权重、formatter、prompt、TTT/程序预算、视角与选2方法；写带时间戳的配置。是否加C6-reserve由**看C6成绩前**的时间/配额预测决定，不因C6好看或难看改变。确认后的任何调参使其失去独立确认身份。

H和E分别报告，保留全部query。预算未完成单列完成率与分母；不能仅报告完成的容易题。不同输出上限、配置或候选数分开。候选oracle只事后分析，最终两个候选必须由不看query答案的规则选出。

全0时先区分：无候选覆盖、选解失败、无可执行程序、纯格式/预算失败。demo重建、留一demo命中、loss、逐格准确率仅作机制诊断，不代替query命中。必要的demo留出须从适配/反馈链完全隐藏其输出；不另挑容易题替换P12，也不占用C6做调参。

## 5. S0：先完成既有比较，再做有上限的效率研究

### 5.1 F0-25与工作格式选择

用原框架/原有效计算路径，对step25做P12 greedy、F0、输出上限2048、n_sample=0。复用原F1-25与F1-50输出，报告命中、触顶、非法尺寸、完整候选耗时。

后续网格分支只用一个工作配置，预先规定选择方式：先选P12 greedy命中最多者；同分优先成本较低的F1；同格式同分优先不增加离线训练的底座；若adapter确有新解则保留该adapter。这是开发选择，不是泛化结论。保存所选权重与理由。

如格式比较的关键差异涉及输出触顶，仅对相关case做预先标记的诊断补查，不能把补查最优结果回填原表。F1的960上限已足以表达标准30×30紧凑网格，31行重复不应靠扩大token或裁行“修正”。

### 5.2 共享加载与状态复核

从真实solve代码抽取加载、逐query推理函数。一个底座驻留，冻结adapter具名切换/禁用；一次只训练当前需要的adapter。禁止deepcopy27B、跨adapter复用KV/GDN/conv状态。A→base→A的2–3题手动复核可与本次待做query重用，确认输出、formatter和权重指纹不串。

省下的是重复加载和初始化，不声称四张卡变成四个worker。现balanced是按层切分；不要通过多线程同时generate冒充批处理，当前采样代码还有进程级torch.multinomial替换。

### 5.3 GDN性能分支：最多30分钟增量，未准备好就跳过

CPU先准备固定版本离线wheelhouse、Linux/Python3.12/torch2.10.0+cu128/C++ABI匹配材料；本地M5检查metadata不能证明CUDA兼容。不升级整个torch/TF/PEFT环境，不在GPU现场无界编译。当前可用依赖组合和wheel实物尚未完成。

固定TF5.8的qwen3_5实现，逐层记录conv/chunk/recurrent四个callable和norm类；聚合fallback警告不等于全算子都回退。用相同短/中/长输入测prefill和cached decode，JIT单列，固定128-token continuation用于速度比较，完整greedy用于行为比较。

FLA在模型构造前可见时会自动改变算子及norm选择。因此先显式固定原baseline的构造路径和norm，完成F0闭环，再临时切换指定算子。不能把“预装包后加载”自动叫原栈。baseline/fast必须保持同权重、输入、norm类与精度，明确究竟替换哪些算子。若不能在驻留模型上可靠切换并恢复，不为这个可选探针再重载整模，直接跳过GDN。数值用同路径重复噪声校准，不能沿用被噪声否定的1e-3绝对梯度阈值。性能预算内无法完成有效对照即放弃，本周继续原栈。

采纳只限已经验证的路径：推理通过不代表反向/TTT通过。端到端典型query时间至少下降10%、无系统异常、冷JIT可摊销，才值得后续复用；该门槛是预算决策，不是精度保证。改变kernel身份后不得把所有旧输出当逐值等价，至少保留少量同题桥接。

## 6. S1：先多视角，再小步TTT，不盲目扩候选

### 6.1 V：P6的三个候选

同工作checkpoint/formatter/thinking=False，取得原视角greedy O、原视角sample S、转置greedy T。O在身份完全匹配且完整时可复用；新增只需S/T。固定sample为temperature0.7、top_p0.9、top_k0、seed按task/query/view稳定派生，继续旧参照，不同时调采样超参。

比较O+S与O+T，各恰好两次答案机会，转置必须覆盖全部demo和query并逆变换输出。分别报告候选覆盖、实际时间和去重。P6无新增则不扩16视角或PoE；仅一个新增也只算探索信号。

### 6.2 A：0/4步题内LoRA

本轮主对照为A0与A4：同一个起始checkpoint、r8/alpha32、现q/k/v/o/gate/up/down目标、microbatch1、grad_accum4、常数lr2e-5、无warmup、weight_decay0.01、clip1.0。4个optimizer updates=16个microstep/题。它们是待验证的固定起点，不是最优配方；不新增GDN targets。

每task只用其demos做LOO，保留至少一个剩余context；2-demo任务合法。先完整枚举LOO base，再按固定seed循环identity/transpose一致增强，所有输入输出一起变换。只监督目标demo输出+EOS，query真值绝不进入适配。用suffix路径，完整序列上限4096；CPU先统计实际长度。不能静默截断、删demo或换更容易task；不满足上限则明确记TTT ineligible并单独报告覆盖，不能将其当A4模型错误。

如果S0选的是原底座，固定seed创建一次fresh LoRA（A正常初始化、B全零，不可同时清零），保存初始快照；若选SFT则恢复选定adapter。A0使用对应零更新状态，先确认fresh LoRA不改变底座输出。

每题恢复相同初始LoRA并新建optimizer；同题多query共享一次适配。A0/A4都生成同样的O+T，测固定候选下的适配变化。已有A0可复用，但逻辑成本表包含其成本。TTT ineligible的task在完整方案中回退到冻结A0，仍计入全部query分母；另报可适配子集机制结果，不将不适用题删掉。

先完成全部P6的A4。只有出现真实新增query、且阶段预算可容纳时，才对预定完整P6从各题A4状态续至A8；保存每题LoRA、optimizer/RNG/cursor，不能重跑前4步，也不只续已命中的题。A8结果单列为条件探索，不伪称预先固定唯一终点。主A4已使用全部demo，重建改善不能当独立demo留出，本轮不为触发A8另开这种诊断分支。若仅train loss下降、query仍全0，没有扩大到128步或全隐藏集TTT的依据。

### 6.3 等时间对照与选2

TTT还需与不更新权重、使用相同每task时间上限的候选方案比较。每题预算B定义为该题A4的适配、恢复/模式切换、全部query的O/T生成、选择及落盘逻辑耗时之和；同会话公共底座加载不重复计入，单独报告。历史复用候选也扣对应逻辑生成时间。固定无TTT队列为O→T→sample seed1→sample seed2，上限4个/每query，按多query轮转，只有剩余时间足够完成才开始；不能看query答案决定多算哪题。记录实际耗时与未用预算；队列用完但还有时间，只能称胜过这个有上限策略，不能泛化为所有等预算无TTT算法。阶段截断导致不完整配对的task单列，不宣称严格等耗时。A8若运行，按其总成本另列对照。

两候选之外的最终选2规则预先冻结：合法网格按逆变换后的整网格频次排序，同票按上述固定队列先后；不使用query真值或gold尺寸。两臂相同规则。另报oracle@已生成数量，避免把4候选覆盖当部署@2。若同时间方案存在速度差/未完成项，完整记录，不只比较optimizer步数。

## 7. S2：把持久解题状态用于程序反馈，小范围验证

先关闭网格SFT adapter，使用同一Qwen底座的完整词表。与网格分支比较属于两套完整方案；反馈净收益在下面同底座、同模板的两个程序臂之间比较。

### 7.1 提示、上下文与输出预算

- 任务级提示包含全部demos及该task全部query输入，要求一个纯Python transform(grid)返回网格。一个程序用于所有query，不为多query重复适配/生成规则。
- 通过apply_chat_template设置enable_thinking=True、reasoning_effort=low、preserve_thinking=True。固定模板实际只支持xhigh/medium/low；不能传high，更不能把这些字段直接塞入HF generate。
- 初始采样temperature1.0、top_p0.95、top_k20、presence penalty不额外加入、repetition_penalty1.0；不将官方非thinking自然语言的1.5 presence penalty盲加到重复数字网格。
- 每次最多1536个总新token，含think与最终代码；最多360秒/调用；每task三个物理调用总上限18分钟，受阶段总deadline更早截断。low是软提示，不能保证1536内结束。
- 解析明确分离think与final；没有闭合think/没有最终程序记incomplete，不能从思考片段猜代码。应用保存公开模型生成的reasoning_content、final、执行反馈；不是复用跨题KV，也没有“无限上下文”。不引入自动压缩器，当前最多一轮反馈，先保留短而完整的状态。

程序输入另设max_prompt_tokens=8192，首轮CPU计数；repair在完整渲染后再次计数，覆盖原题、思考、程序、反馈。反馈按demo原序提供通过/失败、尺寸和最多8个不匹配坐标及预测/已知值，异常文本最多400字符；总反馈最多3000字符。截短的只是预定反馈明细，保留每个demo的状态，不隐藏某个失败demo。完整提示仍超限则记该分支ineligible，不截掉demo/历史、不临时加上下文。两臂都用同一输入上限，额外反馈成本计入repair；首次需真机确认8K输入与1536续写可容纳，不借模型宣称的原生窗口推定L4显存可行。

P2先导如果两题连初次最终程序都无法在上限内产出，停止该支线，记录预算不可行；不声称模型不会编程。如果可执行，才继续固定P6，不能按题目答案或人工可写程度重新挑题。若后续repair不适用，完整方案保留P0可用预测并记录原因，不能只报告成功进入修复的子集。

### 7.2 三次调用形成两个公平方案

1. P0：初次程序，仅在已知demos执行，记录异常、预测尺寸和真实格子差异。
2. P-restart：独立第二次提案，看原任务，不看P0或其反馈。
3. P-repair：看相同P0、已知demo执行反馈与保留状态，最多修正一次。

比较P0+restart与P0+repair。P0实际只算一次以节省GPU；两臂逻辑成本都计入P0，并另报共享执行的物理总成本。后两臂用相同采样设置与上限，按task交替先后顺序；长反馈导致的prefill增量计入repair，不能称相同调用数就是相同时间。

最终候选只按demo全通过优先、再按已知demo完全匹配数量、最后固定生成顺序选两个不同query网格。禁止query金标准、人工规则提示或用结果oracle选答案。示例全通过但query错，应记过拟合，不加更多轮次掩盖。

执行器只获得demos/query输入，不挂载solutions；禁止网络、文件读取、导入、反射和动态执行入口，限制AST/可用builtins，独立子进程设CPU/内存/输出上限。每个demo最多2秒、每个程序总20秒。不要把AST检查单独称为完整安全沙箱。超时/异常是有效反馈，不能让坏程序长时间占着GPU等待。

## 8. S3：只确认有信号且成本可接受的冻结方案

优先把P6出现真实新增的一个方案扩到P12未覆盖部分；不为每个分支跑完整P12。选择依据依次为新增/丢失query、完整率、整体耗时，H/E分别看。1个新增只支持进一步确认；训练loss、demo拟合或合法率单独改善不进入确认。

S3的2小时同时覆盖必要开发扩展和确认；若预测无法完成C6配对，先不读C6答案，优先用S4补缺窗口或经核实的储备补完整。仍无法容纳则明确本周无确认，不给半片挑选结果冠以确认分数。

确认基准按问题冻结：选择TTT时使用S1的无TTT有上限等时间队列，而非退回单greedy；选择转置时与O+S比较；确认程序反馈时与P0+restart比较。若另比较程序完整方案和网格方案，明确是完整方案效用对照并报告时间。方法与基准均最多输出2个答案，不把全部历史模型答案拼在一起。C6最低6task/8query完整配对；放得下才按预定政策扩到完整C12。如果全部开发分支仍0，则不为了消耗配额读确认集，保留它。

最终12h可部署性单独估算：运行时读取真实task数/query数；题内适配/程序按task计，网格生成按query计；包含加载、长题尾部和保存。每题18分钟的先导上限显然不能原样乘240部署；有解题信号后还需基于输入/已知demo的预算准入及更快执行。不能由少量短题均速承诺全量提交。

## 9. 用户需要实现的代码：按依赖顺序交接

详细源码审计见[实现研究备忘](../../experiments/011/implementation-research.md)。以下均未实现。

| 优先级 / 文件 | 具体改动 | 手动验收标准 |
| --- | --- | --- |
| P0：solve_grid.py、run_budgeted_grid_experiment.py | 真正抽取load_session/activate_adapter/evaluate_loaded_queries；去掉假not_run_no_gpu评价路径；共享一次底座 | 同会话A→base→A结果一致；日志加载次数1；原单臂CLI仍可用 |
| P0：同上 | 修判重6参/5参不一致、tuple误当dict；身份含权重SHA、base版本、输入/模板/代码/后端/seed；逐candidate落盘 | 内存/临时JSONL手动检查同身份跳过、换权重重算、只补缺候选；不得靠吞异常 |
| P0：同上 | Notebook总deadline、加载硬截止、生成内部停止、保存余量；阶段额度分开 | 人工在两个候选之间中止，恢复只补第二个；中断不记模型错误 |
| P1：ttt_per_task.py、train_grid_sft.py | 真TTT循环；复用编码/mask/suffix/优化器；task重置；只训练active adapter；同task多query共享 | 一题A0→A4→恢复A0与初始输出相符；query标签不进入batch；2-demo可LOO |
| P1：可选独立probe文件或runner内小函数 | GDN运行路径检查和有时限数值/速度探针；原栈回退 | 逐算子身份+缓存多步数值+端到端时长齐全，失败不阻断主实验 |
| P2：拟新增solve_program.py | 单task程序生成、隔离执行、一次repair/restart、think/final分离、持久状态 | 用人工小网格检查执行/反馈；GPU先P2预算可完成性；不新增测试目录 |
| P2：runner | 接现有P12、011 P6/C6工作项与冻结配置；记录所有skip原因 | 非dry-run实际产生候选；未完成数量和全部分母可回算 |

建议接口草图，仅作实现契约：

    load_session(model_dir, runtime, load_deadline) -> session
    activate_adapter(session, adapter_ref_or_none, mode) -> weight_identity
    generate_candidate(session, job, generation_config, deadline) -> candidate
    adapt_task(session, initial_adapter, demo_episodes, updates, state) -> adapter_state
    solve_program_task(session, task_inputs, limits, journal) -> candidates
    append_candidate(journal, identity, status, raw, parsed, timings)

不重构全部训练平台：先让真实solver共享加载，再补小TTT与程序入口。现训练器freeze_non_lora会让所有lora_参数可训练，多adapter会话必须缩到active参数列表。切到eval须关闭梯度checkpointing、打开use_cache并清旧状态；训练恢复反向设置。TTT跨task不继承optimizer，same-task A4→A8才续状态。

候选identity至少含base版本、adapter权重SHA/TTT任务状态、task/query、formatter/template SHA、view、sample index/seed、generation config、代码与kernel指纹。权重同路径覆盖不能命中旧缓存；做性能探针需新run_id，不能因为已有输出就跳过计算后报告零耗时。

随机seed在运行前由固定seed20261010与task/query/arm/sample_index做SHA256派生，不用Python进程随机hash。P0只生成一次，restart使用独立预定seed。历史输出导入保留原身份、源文件SHA与original_run_id，显式记reference_reused；仅重构编排也不能伪造原代码hash。推理语义/后端变了，先做少量桥接并报告为历史参照，不宣称逐值等价或以历史时间冒充新性能。

## 10. 开GPU前的CPU交付与本周不做的工作

用户实施完毕后先在CPU准备：

1. 可核对的F0/F1权重与配置清单；S4 slug差异核验；下载字节SHA。若权重无法取得，标明闭环受阻，不默认重训。
2. 固定软件/代码版本和离线数据源；每个新入口仅--help/手工小输入预演，不新建自动测试框架。
3. P6 LOO实际token长度、监督位置、采样顺序、长例覆盖表；程序prompt与think/final处理；已有输出可复用清单。
4. 一份带阶段上限、运行顺序、条件分支、完整输出路径的手动Notebook。GPU不得用于现场整理数据、写代码或搜索依赖。
5. 若要做GDN，必须已有可用离线包；否则移除该工作项，保证原栈能执行其余阶段。

不默认运行：同池继续50/100步、完整G2 30K训练、G3裁词表、更多LoRA targets、量化/TP/vLLM大迁移、batch2重构、8专家×10轮、教师API大批生成。它们需要独立问题与预算证据。

离线SFT下一候选先做CPU：从现有训练侧新base中人工审查约20个episode，核对共享规则、歧义、目标隔离；设计“同family的新base vs 同base多增强”相同消费量对照。该比较只测实例多样性，不冒称新规则。不要整包加入含evaluation来源的Tufa SCR/ARC Mega/NVARC数据后继续称C6干净。本轮不给该SFT预分GPU预算；若适配/推理全无信号，先完成数据方案，再另编号交接。

## 11. 每次手动实验应留下的最小记录

数据来源/划分与全部task/query、权重与代码SHA、formatter/模板/思考档位、实际消费ID、LoRA/优化器、每层内核、实际设备映射、安装/加载/JIT/更新/prefill-decode/保存时间、各卡显存、Notebook完整墙钟、账户前后差；逐题新增/丢失、最终@2与oracle分开、合法/触顶/资源中断、产物路径。

标准状态：not_run / complete / invalid_output / budget_interrupted / ineligible / infrastructure_error；不能混成一个0分。旧结果只读，新产物放experiments/011对应阶段。首次确认前保存冻结配置，确认后记已使用。

## 12. 外部依据与可迁移边界

以下均于2026-10-10查阅；引用的是方法依据，不是本项目效果。详见[方法研究备忘](../../experiments/011/methods-research.md)。

- [NVARC论文与代码](https://github.com/1ytic/NVARC)：支持专项数据、题内LoRA、候选统一条件重评分；4B的大规模离线训练与27B小步实验不同，不能照搬rank/DFS缓存或其成绩。
- [MindsAI/Tufa 2025报告](https://github.com/jcole75/arc_2025_mindsai/blob/main/MindsAI_Tufa_Labs_2025_Solution.pdf)：长期预训练、待解集合已知示例适配和增强投票；主要组件消融在ARC1.5/较小模型，不能解释2026公榜88.06。上一轮回答把它概括为逐题TTT过于简化，本轮按作者实现区分。
- [ARChitects表2/3](https://arxiv.org/html/2505.07859v1)：ARC1中增加适配/视角有收益，但更深搜索的oracle收益未充分变成选2收益且成本大增；不是ARC2等预算效果。
- [Poetiq源码](https://github.com/poetiq-ai/poetiq-arc-agi-solver)：可借用生成—执行—反馈循环；API大模型的成绩不外推Qwen，也不适用本比赛无网提交。
- [Qwen3.8官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)及[固定模板](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)：提供thinking/保留状态接口；本地已CPU实测模板，模型收益未测。
- [TF5.8 Qwen3.5实现](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[加载实现](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/core_model_loading.py)、[FLA安装说明](https://github.com/fla-org/flash-linear-attention/blob/main/INSTALL.md)、[causal-conv1d](https://github.com/Dao-AILab/causal-conv1d)：支撑小范围性能调查；实际兼容性和收益仍须真机测量。
- [Kaggle代码要求](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)、[L4额度规则](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators)：本轮API重查，无网、12h、L4两倍扣额。

**研究设计与代码实现均不等于研究提分。模型效果、性能与确认结果以对应progress的实际产物为准；未运行的比较保持未运行。**
