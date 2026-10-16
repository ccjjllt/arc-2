# 012 GDN推理提速与程序完成性实际进展

日期：2026-10-11实施，2026-10-12收回最终结果（北京时间）。对应[实施交接](../plan/012-kernel-program-completion.md)。§1–4保留启动前核对记录，§5起记录执行。**012本轮已结束：K v3完成但不采用快速后端；P v2四次生成均完成调用，却全部耗尽3072 token，完整程序和实际执行均为0，D/L固定策略各0/2。会话55.53分钟，账户差1.86427配额小时，当前剩8.63755小时、reserved=0。按预定止损停止该配置扩算，未启动新GPU。** 最终证据与解读见§10。

## 1 已实现和已运行的范围

- 当前8个核心文件、本地K/P包、下载的CPU v1核心SHA一致，汇总为`2cf9a5d89b3161cafce3884a31fca003aac8a3354aefe9e91f506d59898223da`。源码包含P0-only、D/L解析及K探针函数；文件存在不能证明端到端入口可运行。
- [CPU预检Notebook](https://www.kaggle.com/code/ccjjllt/arc2-012-cpu-preflight)状态COMPLETE。实际cell只恢复代码、安装依赖、核版本、核输入task ID及无GPU；日志最后为`CPU_INPUTS_OK`。没有执行完整生成包装、解析、程序执行、journal和父监督诊断，也没有对应的完整预检报告。
- [K探针Notebook](https://www.kaggle.com/code/ccjjllt/arc2-012-kernel-probe)当前版本1，metadata id138140316，状态ERROR。已收回[原始失败日志](../../experiments/012/kaggle-kernel-probe-v1/arc2-012-kernel-probe.log)：报`Missing K wheels`，尚未进入模型加载或A/B测量，不能记为“内核无效”。
- 远端`ccjjllt/arc2-012-k-wheels`确有三个文件；Kaggle把causal wheel名称中的`+`删除。本地三个wheel SHA均匹配计划。当前文件名容错找到该文件后仍用其被清洗的名称交pip，尚未完整修复。
- P程序完成性没有本次可用实测结果。没有新解题收益、内核提速或反馈机制结论。

证据：[本次启动核对摘要](../../experiments/012/execution/launch-readiness-20261011.json)。CPU日志及下载core在`experiments/012/kaggle-cpu-preflight-v1/`；保留历史`kernel-probe-v1-ran/`原包，不覆盖。

## 2 启动前必须补齐的代码

| 缺口 | 直接后果 | 最小修正 |
| --- | --- | --- |
| 打包器`RUN_K`、`RUN_P`及当前Notebook cell10仍只打印`UNIMPLEMENTED` | 开GPU只安装依赖，不会运行研究 | 接实际命令；包内检查不得再以函数存在或打印信息代替调用 |
| `CONFIG_P/run-p.json`不是runner完整schema | 缺`challenges/out_dir/runtime/stages`；`notebook_seconds`也不是runner读取的预算键 | 构造真实四stage配置，用`time_budget_s=4500`、`load_limit_s=600`，`runtime.save_window_s=300`和固定`base_source`；先运行真实`--dry-run` |
| K worker给`load_session`传`runtime_config={}`，默认prompt文件也不存在 | 接通入口后仍会立即失败 | 传固定模型source、`deactivate_async_load=True`和正确会话截止；用实际tokenizer生成F1 prompt并显式传入路径 |
| K父进程阻塞`stdout.readline()`；阶段开始时间没有用于局部上限 | 静默加载/JIT可能不能及时止损 | 用非阻塞读或`select`轮询，按实际phase起点计算局部deadline并受全局1800秒约束；无日志时也检查超时 |
| 安装在K父监督前，基础pip timeout240秒，额外pip120秒 | 未落实安装/环境合计120秒边界 | 安装前启动父监督，全部子命令使用阶段剩余时间；超时退出，不现场追加安装 |
| P父监督只响应`load_done`，未处理已有P0-start/done事件 | 卡住的调用可能耗到整个75分钟截止 | 保存活动P0的760秒硬截止，超时终止整个worker，保存已完成记录 |
| P整组准入失败只`break`当前stage，worker继续后续stage | 剩3000秒时首臂因不足3040被拒绝，第二臂却可能因只需2280获准，破坏完整配对 | 把拒绝上报worker，停止余下P阶段并全部记`not_run`；基础设施错误也应终止整组 |
| K没有B实际shape warmup，直接进入B1正式计时 | B1包含首次JIT，时间比较失真 | A/B预热与首次JIT都放在600秒准备阶段，之后才A1/B1/B2/A2 |
| K cache末长度只计算`L+N`，没有读取真实cache；未检查NaN/Inf | 可能把无效cache或非有限logits当有效结果 | 读取实际cache长度并断言；逐trial检查shape、finite；不能让NaN阈值比较漏过；实际软件版本与EXPECTED_STACK逐项核对 |
| 被清洗的wheel名仍直接传给pip | `1.7.0cu12torch2.10...`不是合法版本名，标准wheel解析拒绝 | 先核SHA，再复制到working目录恢复原带`+`文件名，安装该规范名称；不改内容 |

主要位置：[打包器](../../experiments/012/prepare_notebooks.py)、[K探针](../../scripts/probe_gdn_inference.py)、[共享runner](../../scripts/run_budgeted_grid_experiment.py)。先修这些已知问题，不扩大研究范围或换模型。

`wheel-manifest.json`仍为`PINNED_NOT_UPLOADED/null`，与实际已上传状态不符：打包器先写旧清单、再丢掉带新slug的返回值。修正清单写出路径，保存实测SHA与实际Dataset身份，不能每次打包回写成未上传。

## 3 CPU复核与重新打包顺序

1. 按上表接通实际入口和父监督，保留原包作历史；新包用新目录和版本，不覆盖v1。
2. 在Linux CPU环境人工走真实包装路径：FakeModel生成→D/L解析→P0-only执行器→journal→汇总，确认计划四调用且无restart/repair。诊断生成内容为固定样例，不加载27B权重。
3. 专门模拟“剩3000秒不足完整P2”并确认整组零生成；模拟无日志worker挂起，确认父进程按局部截止结束。它们不是新增测试套件，只是此次人工诊断。
4. CPU核规范wheel名、文件SHA、依赖元数据与实际配置。CPU不替代CUDA launch/JIT验证；K只用一次有上限的GPU探针验证后者。
5. 保存完整CPU报告及对应新core SHA，再冻结K/P包。现CPU v1的COMPLETE不能作为新实现已经走通的凭据。

## 4 配额和下一步

本次查询GPU总额108000秒，used65974.984秒，reserved0，剩**11.67362配额小时**；刷新时间仍为北京时间2026-10-17 08:00。见核对摘要中的快照。与011末尾的账户变化不能全部归因于K v1，缺少逐作业前后快照，不编造该失败作业成本。

计划K30分钟＋P75分钟仍按2.3预留4.025配额小时，基于本次快照算结束后约7.64862小时，高于6小时储备。**目前阻断在执行入口和CPU验证，额度不是阻断。** 修复后开机前再查余额；本次没有另开GPU，也没有提交比赛结果。

## 5 助手接管修复与新CPU v3

用户明确授权“你来实现然后帮我启动”，并再次要求提高GPU利用率。开始时工作区已包含一版未提交修复、CPU v2和K v2；本次保留并补全这些改动，未覆盖历史包。

- CPU v2实际通过FakeModel四次P0、解析、Linux执行器与单stage预算拒绝；没有验证完整worker拒绝后剩余stage全部停止，也未把安装纳入GPU父监督。本次增加真实四stage worker、真实tokenizer模板和静默进程超时诊断。
- 收回已有K v2日志：wheel名称恢复已成功，随后模型加载约10分钟仍在621/851，被旧父监督终止。它不是JIT、数值或解题失败。原产物在`experiments/012/kaggle-kernel-probe-v2/`；不把其他时段账户扣额全部归给该作业。
- P补齐EOS/截断与基础设施状态；四臂不足预算时整组退出；父监督处理P0事件并硬限760秒。固定setup脚本先于模型加载，在同一受监督worker执行，保存脚本SHA；600秒加载限包括P安装/import，属于更保守的限制。
- K改为非阻塞父监督，静默/无换行输出也能按时终止进程组；安装、环境核验、prompt生成都在受监督setup内，合计120秒。补齐明确FLA函数导入、真实cache长度、A/B warmup、有限值和版本检查、逐trial证据保存。
- K可选wheel在Transformers导入前安装；核SHA后恢复规范文件名。HF加载环境变量在Notebook首次导入前设定；这个设置不能保证不会遇到慢加载。
- 两GPU Notebook均只加载一次模型；记录每10秒四卡利用率、显存与功耗。P四臂连续运行，不让GPU等待人工分析；K仍采用同token工作量A1/B1/B2/A2，P保持回退后端。

八文件冻结core SHA为`31ce366d4df03d344cfd5ef1cbc8ed747865208eba97841acbac680de30ae14f`，包位于`experiments/012/kaggle-v3/`。[实现身份记录](../../experiments/012/execution/implementation-v3.json)保存各代码/Notebook SHA和授权、预算。

本地手动CPU诊断已确认：四stage正常4调用/零repair，3000秒不足完整P2时0调用/四项not_run，基础设施失败不继续后续臂；父子进程超时、setup先执行与SHA记录；K同cache64次单token前向、每trial新cache和NaN拒绝。没有新增测试文件或CI。

[CPU预检](https://www.kaggle.com/code/ccjjllt/arc2-012-cpu-preflight)version 3已成功发布。GPU包只接受相同core SHA的完整CPU receipt；CPU通过不代表CUDA内核、27B输出或速度已验证。当前快照剩11.29231配额小时、reserved=0；开GPU前另取最新额度。

## 6 CPU通过并发布K v3

CPU v3已COMPLETE，实际日志第55.41秒生成[完整receipt](../../experiments/012/kaggle-cpu-preflight-v3/012/cpu-preflight/receipt.json)，没有加载27B权重或使用GPU。已确认：四stage实际4调用、顺序D/L/L/D、零repair；只有3000秒可工作时四stage全部not_run且零调用；Linux执行器四例和静默子进程超时通过。真实tokenizer的D/L输入长度分别为2562/2586、2578/2602，均小于8192。

CPU receipt与K/P包的core SHA均为`31ce366d4df03d344cfd5ef1cbc8ed747865208eba97841acbac680de30ae14f`。开机前重新核对代码字节及余额：used67347.676秒、reserved0、剩11.29231小时；K预留1.15配额小时后仍超过6小时储备。见[开机决定](../../experiments/012/execution/pre-k-v3-decision.json)。

[K探针](https://www.kaggle.com/code/ccjjllt/arc2-012-kernel-probe) **version 3已由CLI确认成功发布**，30分钟总上限、120秒安装/环境、600秒加载、600秒JIT准备、360秒测量、120秒保存。所有安装与模型调用均在父监督之后；各阶段超时结束整个worker，不自动重载。此刻尚无GPU数值/性能结果。

随后Kaggle API确认version 3、GPU启用、machine_shape=NvidiaL4，状态**RUNNING**，见[发布与运行记录](../../experiments/012/execution/k-v3-publication.json)。本次只启动这一GPU会话。

离线`experiments/012/summarize_results.py`同步修正真实K的timing/verdict读取、P的SUMMARY/单stage去重、每臂2题固定分母以及截断/未运行/实际执行区分。用手工内存样例及CPU v2的原始FakeModel记录核对通过；没有改动已发布GPU核心，也不形成性能或提分结论。

P75分钟包已准备并通过相同CPU核心预检，但**本次尚未发布P**。需先收回K结果并重查账户余额，再按原P2四调用方案运行；不得因为K初步测时而自动切换P后端或扩展题集。

## 7 K v3实际结果

用户报告运行结束后收回version 3全部JSON、日志和四卡CSV。Kaggle状态COMPLETE，父监督`exited:0`；从Notebook起点到结束775.764秒（12.93分钟），加载531.529秒，没有OOM或超时。原始产物位于`experiments/012/kaggle-kernel-probe-v3/012/kernel-probe/`；[结果](../../experiments/012/kaggle-kernel-probe-v3/012/kernel-probe/output/probe-result.json)、[父监督](../../experiments/012/kaggle-kernel-probe-v3/012/kernel-probe/output/probe-supervisor.json)、[离线分析](../../experiments/012/execution/k-v3-analysis.json)。

固定原P2首题F1 prompt，1392输入token，64个参考续写token；每trial实际cache从1392增至1456，比较65个next-token logits。48层B函数真实切到FLA和causal-conv1d，norm保持torch，权重未变。底座config/index SHA及分片大小与011一致；source字符串有无`kaggle:`前缀会改变身份哈希，不能误认成换权重。

| trial | prefill秒 | 64步decode秒 | 合计秒 |
| --- | ---: | ---: | ---: |
| A1 回退 | 2.088 | 13.713 | 15.801 |
| B1 快速候选 | 1.543 | 13.423 | 14.966 |
| B2 快速候选 | 1.546 | 13.424 | 14.970 |
| A2 回退 | 2.116 | 13.713 | 15.829 |

A1/A2的top1一致、相对RMS误差0。B/A全部65位置top1也一致，但两次相对RMS均为0.0237495，超过预定0.01，判定`numerical_drift`。这不是证明生成答案错误或快速算子数学错误；按预先采用条件，本轮不采用B。

两次配对的完整工作量仅节省5.28%/5.43%，也低于15%继续门槛。prefill约快26%，decode仅约快2.11%；短探针结果不能外推长输出、完整解题或训练后向。停止本轮内核扩算，不重跑K、不增加JIT预算，P继续原回退后端。

账户used从67347.676增至68946.340秒，差0.44407配额小时；剩10.84824小时，reserved=0，见[作业后快照](../../experiments/012/execution/post-k-v3-quota.json)。这仍是账户差，不是隔离逐设备账单。四卡每10秒采样、每卡78点，全会话均值1.87%/3.87%/4.28%/4.29%；加载占会话约68.5%，这些全程均值不能当decode忙碌率，也不支持“GPU已打满”的结论。

## 8 启动P程序完成性对照

用户明确要求K结束后启动下一项。重新核对P冻结包的实际内嵌源码、当前scripts和CPU-v3 receipt，八文件SHA全部一致，沿用core `31ce366d4df03d344cfd5ef1cbc8ed747865208eba97841acbac680de30ae14f`，没有重新训练或打包。

开机前剩10.84824小时、reserved=0；按2.3系数预留2.875小时后约7.97324小时，高于6小时储备。见[开机决定](../../experiments/012/execution/pre-p-v1-decision.json)。只运行原P2：db118e2a D/L，291dc1e1 L/D，共四次P0；两臂3072新token、720秒调用、760秒任务。一次底座加载后连续运行，总75分钟，安装和加载合计最多600秒，保存预留300秒，预算不足整组退出。

[P程序对照](https://www.kaggle.com/code/ccjjllt/arc2-012-program-completion) **version 1已由CLI确认成功发布**。包位于`experiments/012/kaggle-v3/program-completion/`；只挂基础wheelhouse及CPU-v3报告，不挂K新增wheel。P保持原后端、记录四卡采样，不自动启动restart/repair、TTT或竞赛提交。当前尚无P输出或评分结果。

Kaggle metadata已确认version 1、GPU启用、NvidiaL4；CLI状态明确为**QUEUED**，尚未取得实际执行日志，不把排队说成模型已开始生成。见[发布与调度记录](../../experiments/012/execution/p-v1-publication.json)。只提交一次，由Kaggle正常调度执行。

## 9 P v1加载超时与限额修正

用户报告报错后，Kaggle确认ERROR，GPU reserved=0。下载[父监督记录](../../experiments/012/kaggle-program-completion-v1/012/program-completion/output/supervisor.json)与[worker日志](../../experiments/012/kaggle-program-completion-v1/012/program-completion/output/worker.log)后确认：`budget_interrupted / load_deadline`，监督计时601.432秒，worker收到SIGTERM（-15）。Notebook末尾的`AssertionError`是`assert rc==0`对退出码2的显示，不是模型生成栈异常。

安装成功，用63.562秒；权重仍在持续加载，最后797/851（约94%），进度条模型加载用时8分53秒。没有`load_done`、`stage_start`、`text_start`或候选journal；四个P0均未运行，不能记成0/2能力结果。`SUMMARY.status=running`是被父进程终止前的中间快照，最终状态以supervisor为准。

这次合并600秒上限把必要安装也包含在内，余下真实加载时间不足；该限额设置过紧。账户used从68946.340增至70193.460秒，差0.34642配额小时，剩10.50182小时、reserved=0；记录为账户差，不能当精确隔离账单。见[失败与额度证据](../../experiments/012/execution/p-v1-failure.json)。没有模型结果可以接着推理，失败记录保留。

### 修复只改P的安装与加载上限

将P `load_limit_s`由600改为900秒；总4500秒、保存300秒、四调用3040秒、3072 token/720秒生成/760秒任务、D/L顺序与原Torch后端均不变。有效工作窗口4200秒，完整四调用允许前置耗时1160秒；900秒上限下仍保留260秒前置余量，实际准入继续使用全局剩余时间，不能为赶结果跑半组。

核心代码、输入、执行器和CPU-v3 receipt保持字节一致；仅Notebook一个代码cell中的加载参数改变，旧P v1包不覆盖。新包：`experiments/012/kaggle-p-v2/program-completion/`。打包器P默认值同步改为900，K仍600。运行真实runner配置dry-run、参数解析和包内AST均通过，没有重新消耗GPU做CPU可查问题。证据：[配置差异与SHA](../../experiments/012/execution/p-v2-config-repair.json)、[CPU配置核对日志](../../experiments/012/execution/p-v2-cpu-config-check.log)。

开机前重新查配额仍为10.50182小时、reserved=0；按2.3系数预留完整75分钟的2.875配额小时后约7.62682小时，高于6小时储备。不能把四张L4物理设备直接当作四倍账户扣额；这里沿用已核实的L4×4两倍规则及2.3保守系数，账户差另记。见[修正版开机决定](../../experiments/012/execution/pre-p-v2-decision.json)。这是根据确定原因、完成CPU核对后的单次修正版发布，不是遇错自动循环重启。

CLI已确认[P程序对照](https://www.kaggle.com/code/ccjjllt/arc2-012-program-completion) **version 2成功发布，当前RUNNING**；metadata确认GPU启用、NvidiaL4。见[修正版发布记录](../../experiments/012/execution/p-v2-publication.json)。此状态尚不证明加载或四次生成已经完成，需以最终日志和候选产物确认。

## 10 P v2完整运行结果与停止决定

用户报告结束后通过Kaggle官方CLI确认version 2为COMPLETE，并下载全部JSON/JSONL、日志和四卡CSV。MCP当时出现传输连接错误，状态和产物改用已授权CLI读取，账户额度改用同一官方SDK的`quota_view`读取；没有因此重启作业。

### 运行完整性与数据身份

父监督status为complete、exit code 0、load_completed=true，全程3331.981秒（55.53分钟），低于4500秒。安装62.696秒，纯加载556.024秒，从Notebook起点到load_done为619.201秒，验证原600秒合并限额不足；新900秒限额按预期生效。第一次生成前仍有3580.79秒有效窗口，满足完整四调用3040秒准入。

八文件manifest与冻结P-v2包、当前scripts一致；运行时报告的代码SHA逐项一致。底座仍为固定Qwen3.8-27B BF16，实际4×L4、48层均为原Torch回退，没有K快速算子残留。输入、query清单与评分文件SHA均与011继承文件匹配。只运行原开发P2，未读取C6/C12答案，也没有新竞赛提交。

四次P0均按D/L/L/D顺序运行，无缓存复用、无未准入、无时间中断、无OOM。各调用都在720秒截止之前结束，但结束原因均为`max_new_tokens`，不是EOS。`SUMMARY.work_complete=true`只表示计划工作已观察完，不能解释为程序生成成功。

### 输出与分母

| task / 模式 | 输入token | 新token | 生成秒数 | 实际输出 |
| --- | ---: | ---: | ---: | --- |
| db118e2a D 关闭thinking | 2562 | 3072 | 678.145 | 开始一个Python代码块和`transform`，随后大量注释推导；代码块未闭合 |
| db118e2a L thinking low | 2586 | 3072 | 675.744 | 持续推导，没有`</think>` |
| 291dc1e1 L thinking low | 2602 | 3072 | 675.589 | 持续推导，没有`</think>` |
| 291dc1e1 D 关闭thinking | 2578 | 3072 | 675.600 | 持续文字推导，没有Python程序块或`transform` |

**完整程序0/4，实际demo执行0次，实际query执行0次，repair调用0次。** 结果中的`demo_matches=0`及query错误项是未执行的解析失败占位，不是代码执行后全部答错。没有手工闭合代码块或从未结束的思考中提取代码来改变原实验标准。

两个策略都完整观察2题：D固定策略query EM **0/2**，L固定策略 **0/2**；完整D/L观测配对分母为2。可执行程序配对分母和执行后准确率分母均为0，不能把“执行后正确率”写成0%。两题属于已反复使用的开发集，不是未见确认集。

CPU及运行身份记录确认D/L参数和渲染prompt不同；当前结果说明**关闭thinking不足以在该提示下约束简短代码输出**，不据此判定参数没传入或接口失效。输入只有约2.6K token，未触及8192上限；此次是输出3072 token耗尽，不支持“扩大输入上下文就能解决”的结论。3072无完整程序也不证明模型在任意预算下均不能完成。

### 成本与GPU利用率

真实生成合计12288 token、2705.078秒（45.08分钟），综合约4.54 token/s。D合计1353.744秒，L合计1351.333秒；两臂都耗尽token预算，此轮实际消耗也接近。第一stage的1297.597秒日志包含前置准备和加载，自身stage实际约678.389秒，不能重复把加载再计入每臂成本。

生成占监督墙钟约81.2%，这是会话用途占比，不是GPU硬件利用率。每卡334个约10秒采样点，生成区间各271点；四卡生成区间平均GPU利用率为17.10%/28.45%/27.35%/27.01%，全会话为13.91%/23.12%/22.24%/21.96%。采样峰值显存为12834/14776/14752/15736 MiB，包含分配器保留。当前四卡按层串行、batch1仍未实现高硬件利用率，本轮也没有有效候选收益。

2026-10-12 10:31:57北京时间读取账户：used76904.838秒、reserved0，总额108000秒，剩**8.63755配额小时**。P-v2前used70193.460秒，账户差6711.378秒＝**1.86427配额小时**，不等于隔离逐设备账单。至少6小时储备仍在；本次结束后未再开GPU。

### 证据与下一步

- [完整运行结果与成本摘录](../../experiments/012/execution/p-v2-outcome.json)：身份、原始文件SHA、逐输出状态和GPU统计。
- [离线策略评分](../../experiments/012/execution/p-v2-analysis.json)：SUMMARY优先去重，保留固定2题分母及未执行口径。
- [配额快照](../../experiments/012/execution/post-p-v2-quota.json)、[父监督](../../experiments/012/kaggle-program-completion-v2/012/program-completion/output/supervisor.json)、[逐候选原文](../../experiments/012/kaggle-program-completion-v2/012/program-completion/output/candidates.jsonl)。

按012计划§8止损：**停止当前程序配置，不加到6144 token、不扩P6、不启动restart/repair、TTT补片或C6确认。** 这轮没有可执行P0，不能把反馈方法判作无效，也不证明27B能力上限。

下一步只保留CPU研究：审查并重新设计“只输出完整简短代码”的提示与输出协议，区分格式约束和规则推理能力，先写清下一次可完成性假设与成本；CPU检查只能验证协议实现，不能证明新提示对模型有效。没有新方案和明确对照前，保留剩余配额。需要后续GPU时另行冻结实验范围，不从本轮直接追加。
