# 012 GDN推理提速与程序完成性实施交接

日期：2026-10-11，北京时间。实施与GPU操作由用户完成。本轮交接只新增计划，**下述代码改动、Notebook和实验均未实现、未运行**。固定Qwen/Qwen3.8-27B、BF16、Kaggle L4×4，继承[011实际结果](../progress/011-budget-efficient-adaptation.md)。不新增测试框架或CI。

后续状态：以上为交接时记录。核心实现与CPU v3完整预检已完成，K v3实际完成但未通过快速后端采用条件。2026-10-12回收P v1的安装＋加载超时后，仅将P加载上限600→900秒，总75分钟和四调用预算不变；该执行修正优先于下文原600秒计划，详见[012实际进展§9](../progress/012-kernel-program-completion.md#9-p-v1加载超时与限额修正)。

本轮结束补记：P v2已正常完成四次调用，均在3072 token截断，没有完整程序或实际代码执行，D/L各0/2。现按§8停止当前配置扩算，剩余8.63755配额小时、reserved=0；最终结果见[012实际进展§10](../progress/012-kernel-program-completion.md#10-p-v2完整运行结果与停止决定)。

先回答两个具体问题：GDN与卷积快速算子能否降低同一工作量的推理耗时；在固定资源上限内，原两道开发题能否产出完整可执行程序。完成这两个小实验后回收结果，暂不扩大训练、TTT步数或程序题集。

## 1 实施顺序

1. 读本节至第4节，保存当前011证据；全部新配置、产物和Notebook放到`experiments/012/`。
2. 在CPU准备依赖、内核探针、P0-only入口和分模式解析；先完成第9节的手动预检，不让GPU等待改代码或处理依赖。
3. 人工运行K内核探针，最多30分钟墙钟；保存并查看结果。失败也不自动换版本重试。
4. 人工运行P程序完成性实验，最多75分钟墙钟；仅原P2、两臂各一次P0，共4次调用。P保持011的PyTorch回退后端，不因K初步测速成功就切换实现。
5. 记录实际配额、耗时、覆盖和结果，再决定后续研究。新建同主题的`docs/progress/012-kernel-program-completion.md`填写实测；不预填成功。

两个会话分别只加载一次模型。首次分开运行是为了隔离依赖和后端变化，也便于K发生JIT/导入故障时止损；接受多一次约5–6分钟的历史加载成本。不要把P的4次调用拆成4个Notebook。

## 2 必须继承的结果和身份

| 已确认事实 | 对本轮的约束 |
| --- | --- |
| F0-25 P12为0/14；多视角完整配对为0/5 | 不重训或重跑这些已完成结果 |
| TTT两题真实完成A4，A0/A4均0/2 | 不因loss或权重变化增加到A8；尚无等实际时间无TTT对照 |
| 程序P0和restart共4次均1536 token截断，没有结束思考或代码 | 实际repair调用0次；不能称反馈方法已经失败 |
| 程序输入2586/2602 token，未触及8192输入上限 | 先解决输出完成性；本轮不扩大输入上下文 |
| 011加载为310.335/352.118秒；48个GDN层使用回退算子 | 共享加载已跑通；内核提速尚未实测 |
| 4卡全会话采样平均利用率约13%–27% | 按层串行、batch1，不能把4卡当4个独立worker；不以占满GPU为目标 |

数据固定为011已有开发P2：`db118e2a/query0`（H_new）、`291dc1e1/query0`（E24）。它们已反复用于开发，不能称未见确认集。C6/C12不进入本轮，不读取其答案。

必须保留的本地依据：

- [冻结七文件core SHA清单](../../experiments/011/kaggle-s1-pilots-v1/011/pilot-code/core-manifest.json)，汇总SHA为`bbc9b83fc24f7b216878695cf7ca41fec83af15ab94bbb8af640047e2896a97e`。修改后的012文件重新计算SHA，不沿用011身份。
- [运行身份与环境摘录](../../experiments/011/execution/s1-pilots-publication.json)：含Notebook SHA、各源码SHA及原始产物SHA。底座固定`ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1`；未重新计算全部55GB底座分片SHA，身份依据是固定版本、config/index和分片大小。
- [P2输入](../../experiments/011/kaggle-s1-pilots-v1/011/pilot-code/p2-challenges.json)、[query清单](../../experiments/011/kaggle-s1-pilots-v1/011/pilot-code/p2-queries.json)、[事后评分答案](../../experiments/011/kaggle-s1-pilots-v1/011/pilot-code/p2-scoring.json)。复制后核对字节SHA；输入与评分文件分离。K不挂载评分文件，P的模型/执行器入口不读取评分文件。
- [S1原始候选](../../experiments/011/kaggle-s1-pilots-v1/011/s1-pilots/candidates.jsonl)、[离线结果分析](../../experiments/011/execution/s1-pilots-analysis.json)。保留全部截断原文，不能只保留成功输出。

真实软件栈为Python3.12.13、Torch2.10.0+cu128、Transformers5.8.0、PEFT0.19.1、Accelerate1.13.0、Triton3.6.0，Torch CUDA12.8、CXX11 ABI=true。继续`device_map="balanced"`、无CPU/disk offload、`HF_DEACTIVATE_ASYNC_LOAD=1`、无adapter。该开关与加载变快没有同条件因果对照。

## 3 GPU预算和停止规则

最后核实账户快照为2026-10-11 15:00:44北京时间：剩15.23138配额小时、reserved=0，[原始记录](../../experiments/011/execution/post-s1-pilots-quota.json)。这不是未来开机时的实时余额。

| 实验 | 全会话墙钟硬上限 | 按2.3保守预留配额 | 默认范围 |
| --- | ---: | ---: | --- |
| K 同题内核探针 | 30分钟 | 1.15小时 | 一次加载，固定输入和64步续写的A/B |
| P 程序完成性 | 75分钟 | 2.875小时 | 原P2，D/L各一次P0，共4调用 |
| 本轮合计 | 105分钟 | **4.025小时** | 不自动追加重试、反馈或TTT |

2.3是预算系数，不是精确扣额倍率。按上述快照算，预留后仍约11.20638小时；始终至少保留6配额小时。每次开机前重新查used/reserved：`可用余额－本次保守预留 >= 6小时`才安排该会话；不与其他GPU作业并跑。实际扣额以会话前后账户快照记录，不能用墙钟替代。

CPU依赖下载、文献查阅、包装Notebook和查看结果均在GPU关闭时完成。安装、导入、加载、JIT、生成、执行器和保存全部计入会话；超限保存后退出，不在原作业中等待人工决定，不自动重启。

## 4 最小代码改动清单

以下接口和新路径是**实施要求，不是现有可直接运行的功能**。只做这些局部改动，不重构训练框架。

| 文件 | 要实现的内容 |
| --- | --- |
| 新增`scripts/probe_gdn_inference.py` | 独立K探针：环境核对、一次加载、48层函数切换、相同token工作量、数值/计时输出；采用父子进程硬截止 |
| [scripts/solve_program.py](../../scripts/solve_program.py) | P0-only模式、显式thinking参数、分模式解析、资源状态优先；保留默认011反馈模式和执行器限制 |
| [scripts/run_budgeted_grid_experiment.py](../../scripts/run_budgeted_grid_experiment.py) | 加载前校验ProgramLimits/模式；透传P0-only；四调用整体准入；完成性只看计划中的proposal，不硬要求restart/repair |
| [scripts/grid_runtime.py](../../scripts/grid_runtime.py) | 优先复用`generate_text`，它已支持thinking开关与自定义token/时间；仅在缺少模式/后端身份或父进程事件时做必要补充 |
| 新增`experiments/012/prepare_notebooks.py` | 只生成本地CPU/K/P包和SHA清单，不push、不自动启动；已有输出目录拒绝覆盖 |
| 新增`experiments/012/summarize_results.py` | 人工调用的离线汇总：K配对数值/时间，P完成性、执行和query EM；不用pytest/CI |

`solve_grid.py`、`train_grid_sft.py`、`ttt_per_task.py`本轮不需要改算法。保留`experiments/011/kaggle/`及下载core的原始字节，不重新运行011打包脚本覆盖历史包。

## 5 K的离线依赖准备

精确来源、发布资产SHA和静态接口依据见[011内核备忘](../../experiments/011/gdn-followup-evidence.md)，查阅日期2026-10-11。本轮固定这些版本，不扫描组合：

| 文件 | SHA256 |
| --- | --- |
| `causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` | `8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7` |
| `fla_core-0.5.2-py3-none-any.whl` | `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761` |
| `flash_linear_attention-0.5.2-py3-none-any.whl` | `dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400` |

用Linux/Python3.12环境检查wheel元数据与现有依赖，补录实际einops版本；不要在Mac上把Linux CUDA wheel导入失败当不兼容结论。依赖若缺失，先在CPU准备固定版本wheel，写清来源和SHA。K的新增wheel放独立私有Dataset版本，发布前把**实际slug和版本**写入配置；本交接没有已上传的新Dataset，不能引用猜测的slug。

安装使用明确本地wheel路径和`--no-index --no-deps`，保持原Torch/TF/PEFT/Triton版本，不用`pip -U`、sdist、Git源码安装或GPU现场NVCC编译。CPU能核文件与部分导入，不能证明L4 launch/JIT可用；无GPU的导入限制单列，GPU侧一次核验失败即退出。

K同时挂原`ccjjllt/qwen38-wheelhouse-py312`和新增三wheel的独立Dataset；复用docker image不等于TF5.8/PEFT等依赖已经安装，011实际也需要启动后离线安装基础栈。P只挂原wheelhouse，不挂K的新增wheel，确保程序完成性研究仍使用011回退后端。

## 6 K的一次加载与同工作量对照

### 6.1 加载与后端切换

建议实现六个小函数：`check_environment`、`load_probe_session`、`set_backend`、`make_reference_tokens`、`run_trial`、`compare_trials`。沿用现有`grid_runtime.load_session`，不要另写权重加载器。

先导入并保存官方fast函数引用。加载期间临时设`transformers.models.qwen3_5.modeling_qwen3_5.FusedRMSNormGated=None`，在`try/finally`中调用`load_session`并恢复模块全局；目的是固定两臂都使用`Qwen3_5RMSNormGated`。加载后核实恰有48个GDN层，norm类型全部符合，且没有offload。

每臂开始前显式设置48层实例属性：

| 属性 | A 回退参考 | B 快速候选 |
| --- | --- | --- |
| `causal_conv1d_fn` | `None` | 官方`causal_conv1d_fn` |
| `causal_conv1d_update` | TF的`torch_causal_conv1d_update` | 官方update |
| `chunk_gated_delta_rule` | TF的`torch_chunk_gated_delta_rule` | FLA chunk |
| `recurrent_gated_delta_rule` | TF的`torch_recurrent_gated_delta_rule` | FLA fused recurrent |

不重建层、不复制模型、不调用`.to()`重新分配权重。核norm对象及其权重存储指针保持不变，每trial重新`inspect_gdn_runtime`并保存实际函数来源，不能照抄首次加载的`runtime_identity`。B只验证GDN/conv函数，**不代表已验证FusedRMSNormGated全路径或训练后向**。收尾恢复A并清缓存。

### 6.2 固定输入与缓存

使用`db118e2a/query0`、原F1完整提示、nonthinking、原底座。K不读query gold。A先greedy生成固定64个continuation token，保存token IDs和SHA，后续所有trial喂完全相同的token。若提前EOS，所有trial统一实际长度N并明确报告N，不填充、不换题；N不足16则止于诊断，不给吞吐采用结论。

每trial从`past_key_values=None`开始，使用本轮prefill返回的模型原生cache。不要构造无模型配置的空`DynamicCache()`，不要复用上一臂的状态。`solve_grid._drop_cached_decode_state`只清模型挂载的`_cache`；还必须释放探针自己持有的outputs/cache引用。

在`inference_mode()`中完成：完整prompt prefill（`use_cache=True, logits_to_keep=1`），再逐一输入N个固定token，做N次单token cached forward，默认N=64。按TF5.8该模型实际调用签名维护attention mask、cache position及返回cache，记录cache长度L→L+N。**一次喂64个token会走chunk路径，不是本实验的recurrent decode测速。**

一次trial采N+1个next-token logits（N=64时为65个）：prefill最后位置1个，加N次decode各1个。logits先保存在GPU，测量结束后才搬CPU比较，避免每步`.cpu()`改变计时。只存这些末位logits，不开完整prompt的全词表logits；每轮完成后释放，不能跨全部trial堆在GPU。早EOS时循环、cache长度、指标分母全部统一使用实际N，不能仍按64步索引。

### 6.3 计时与判定

先做小算子有限值/cache检查及A/B实际shape warmup，然后固定顺序**A1→B1→B2→A2**。计时边界同步全部四卡，分别记录prefill、decode、合计，使用`time.perf_counter`；不要每token同步或在计时区间`empty_cache()`。每trial计时前对四卡分别`reset_peak_memory_stats`，记录peak allocated、peak reserved、起始占用、功耗/利用率采样，避免把上一轮峰值带入。K不使用候选缓存作为测时结果。

下述阈值是本轮预定研究选择，不是上游对数学等价的保证：

- NaN/Inf、logits shape、cache长度或设备错误：立即停止。
- A1/A2任何位置top1不同，或整体logits相对RMS误差大于`1e-4`：记`unstable_reference`，不下性能采用结论。
- B1/B2分别与A1/A2比较：任一位置top1不同，或相对RMS误差大于`max(0.01, 5×A重复误差)`，记`numerical_drift`，保存差异，不采用B。误差统一用FP32计算`RMS(B-A)/max(RMS(A),1e-12)`，记录全N+1位置聚合值、逐位置最大值及top1分歧数。
- 数值条件通过，且`1-total_B1/total_A1`、`1-total_B2/total_A2`均≥15%，才记`worth_full_output_bridge`。否则记`insufficient_gain`，不扩大K预算。

短探针通过只支持后续完整greedy输出桥接，不自动换P后端。完整输出桥接、fused norm和TTT后向检查不挤进本轮30分钟；它们未运行，后续单独安排。

### 6.4 三十分钟硬包络

| 阶段 | 上限 |
| --- | ---: |
| 离线安装和环境核验 | 120秒 |
| 模型加载一次 | 600秒 |
| 小算子检查、所有实际shape首次JIT、A/B warmup、参考token生成 | 合计600秒 |
| 四次正式测量 | 360秒 |
| 保存和退出 | 120秒 |
| 总上限 | 1800秒 |

加载后的整模型warmup首次编译也计入600秒，不能把它排除在JIT上限外。父进程从Notebook首cell计总时间，接收阶段事件并监督局部截止；超限终止整个worker，保留已刷盘记录。CUDA导入、launch、OOM或JIT失败后不在同一会话回退重载/反复安装。30分钟是止损上限，不能保证首次JIT完成。

120秒安装上限包含原基础栈和新增wheel的准备；父监督必须在安装开始前启动。不能原样复制011安装cell中90秒卸载加240秒pip的独立timeout；每条子命令使用安装阶段的实际剩余时间，阶段超时即退出。CPU预检先核清基础栈安装步骤，避免GPU上解析新依赖。

## 7 P的P0-only实现与同额两臂

### 7.1 模式、解析和资源状态

现有`generate_text`已经支持`enable_thinking`、token和秒数；真正的限制在`solve_program.py`与runner：

1. 给`run_program_task`新增显式`policy="p0_only"`及thinking参数，默认保留旧模式。P0-only只生成P0、执行已知demo和query input并落盘，不能进入restart/repair循环，也不能生成虚假的`p0+repair`结果。
2. `ProgramLimits.validate`对P0-only允许最高3072新token和720秒调用，默认值可保持011的1536/360；输入上限仍8192，任务调用上限用760秒。旧三proposal模式继续使用原边界。本轮不放宽执行器的2秒/grid、20秒/program、512MiB、AST/builtin限制。
3. 给`split_program_response`显式传入thinking模式。L仍必须恰有一个生成的`</think>`并只解析其后的最终文本；D直接把生成正文作为final，不要求`</think>`。D若意外生成thinking标记应单列格式异常，不能默默切换模式。两臂都要求唯一闭合Python代码块和合法`transform`，不从未结束的思考中抽取代码。
4. 解析诊断不覆盖`budget_interrupted`、`infrastructure_error`等生成状态。首轮保守规定只有生成以EOS正常结束且最终代码完整、AST通过，才进入实际程序执行；截断时即使文本恰好含代码块也单列，不作为完成成功。
5. 在模型加载之前校验全部四个阶段的policy、limits、输入文件、解析模式及顺序，避免进入GPU后才触发1536/360旧上限报错。

### 7.2 冻结两臂和调用顺序

| 字段 | D 直接代码 | L 有限思考 |
| --- | --- | --- |
| `policy` | `p0_only` | `p0_only` |
| `enable_thinking` | false | true |
| `reasoning_effort` | low（模板在关闭thinking时忽略） | low |
| `preserve_thinking` | true | true |
| `max_new_tokens` | 3072 | 3072 |
| `max_call_seconds` | 720 | 720 |
| `max_task_seconds` | 760 | 760 |
| 输入上限 | 8192 | 8192 |

其余完全相同：原`SYSTEM_PROMPT`、原P2已知demo与query input、底座、PyTorch回退后端、`do_sample=True`、temperature1.0、top_p0.95、top_k20、seed基数20261011。两臂对同一task都用既有`_seed(task_id,"p0",20261011)`，不因D/L更换seed，也不重写提示增加另一变量。CPU保存实际渲染prompt和token计数；模板的thinking段差异正是实验因素。

执行顺序固定为`db118e2a D → db118e2a L → 291dc1e1 L → 291dc1e1 D`。可用现有runner的4个单task `type="program"`阶段表达；两个单task query文件从原P2清单拆出。每次都是全新P0，不把上一次答案、代码、思考或demo反馈带入下一臂。

这是相同上限的完成性对照，不是相同实际消耗的能力比较。D如果早停省时就保留真实耗时，不能补采样把预算花完。011的1536-token P0作为历史参照，不为了凑三臂重跑。

### 7.3 七十五分钟准入与截止

Notebook总上限4500秒、模型加载上限600秒、保存窗口300秒。因此有效工作截止为首cell开始后4200秒。每次P0最多720秒生成，另留20秒执行和20秒准备/落盘，共760秒；四调用合计3040秒。

首次生成前必须还剩至少3040秒有效工作窗口，否则完整P2不准入，记录原因并退出。即安装与加载合计需≤1160秒；不要再次扣除已经从4200中预留的300秒。每个pair和call开始前用剩余计划调用数×760复查，不够则标明未运行，禁止只报告已完成的容易部分。

整组剩余调用准入由runner使用**全局**`session.deadline - session.save_window_s - now`计算，不能用当前stage的局部deadline。后3个stage只有760秒，用它检查“剩余3次×760”会错误拒绝。当前这一次调用再分别受stage、task、call的局部截止约束。

**现有worker的第一个stage从Notebook起点计时，包含前置准备和加载。** 四阶段方案中首stage的`time_budget_s`用4500或省略，不可写760；后3个stage可写760，每次`run_program_task`自身仍限制760。首stage实际截止仍受全局4200控制。保存`notebook_start_monotonic`，不要为每个stage重置总计时。

720秒生成使用已有token级deadline；CPU执行器沿用子进程限制。模型调用若卡在prefill/CUDA，token停止条件可能来不及触发，父进程应从P0-start事件监督760秒任务截止，超时终止整个worker。不要只在4次调用完成后检查总时长。

### 7.4 结果身份和评分

每条生成身份包含task/query、D/L、P0-only、thinking/effort/preserve参数、完整渲染prompt SHA、源码/模板/后端身份、seed、token与时间上限。`generate_text`已把多数参数放入identity，补全mode字段即可；不能复用011截断记录为新3072预算的结果。

P0-only完成性按计划的4次P0统计，不要求不存在的restart/repair。保留`physical_generation_calls=4`预期与实际值、`repair_invoked_count=0`；“任务结束”不自动等于“代码可执行”。每个阶段及时写JSON/JSONL并flush，先保存raw输出再执行代码。

两臂分别固定以2个query为主分母，至少输出以下计数和逐题明细：

- 正常EOS、token截断、时间中断、未准入、基础设施错误。
- thinking闭合数（仅L适用）、完整代码数、AST通过数、实际demo执行次数。
- 至少一次demo执行成功的题数、全部demo执行成功的题数、全部demo网格匹配的题数；三者不能混用。
- 有效query网格覆盖、query exact-match、两臂完整配对分母，以及全计划2题的覆盖和命中。未执行不能写成“demo运行后全错”。
- 实际输出token、生成秒数、执行秒数；分别报告每个完整程序的成本和整个两题策略的成本。零完成时成本写“无完整程序”，不除零或只保留成功题。

评分只在生成完成后由独立离线入口读取`p2-scoring.json`；query答案不进入prompt、修复、裁剪或候选选择。P0各只有一候选，无需新增排序算法。P2上的query命中仍只是开发结果，不是比赛提分。

## 8 结果驱动的下一步

| 结果 | 后续动作 |
| --- | --- |
| K导入/JIT失败、参考不稳、数值漂移或节省不足15% | 保存具体原因，保留回退后端，不反复升级依赖 |
| K数值与速度均达标 | 仅获得完整输出桥接的依据；本轮P照旧后端运行 |
| D/L都没有完整程序 | 停止当前程序配置，不继续加token到6144或扩大P6 |
| 有代码但全部执行异常 | CPU审查prompt/受限执行器是否存在表达冲突，不能直接追加修复GPU |
| 至少一题有完整代码且至少一次真实demo执行成功 | 可设计下一轮反馈对restart实验；本轮不自动启动，不宣称解题已提升 |
| 某臂query EM出现新增 | 冻结该臂，另立确认计划；不在本会话打开C6或按gold调prompt |

TTT剩余P6四题、A8、等时间非TTT候选、GDN训练后向、新数据SFT均不在本轮执行包。P2的0/2没有否定TTT，但目前没有证据优先花费这些预算。CPU可继续[训练数据口径审查](../../experiments/011/data-next-study.md)，不以此为理由自动开训练。

## 9 开GPU前的人工核对

这些是在CPU Notebook或终端中人工执行、查看并记录的短诊断，不新增测试文件、测试目录或测试框架。

1. 核对新包的源码/输入/wheel SHA和固定版本；确认所有预期文件在挂载目录，输入只有query input。K没有评分文件，P的求解入口不读取它。
2. 用实际tokenizer渲染D/L两种prompt，检查仅thinking模板差异；核输入长度<8192；记录模板和prompt SHA。
3. 用固定小字符串走真实解析：D的完整代码能接受；L缺`</think>`不能执行；闭合代码但未EOS不能记完成；deadline/infrastructure状态不被格式结果覆盖。
4. 用FakeModel生成在CPU上实际走`generate_text → parser → P0-only → executor/journal → 汇总`，人工确认每stage只有1次生成、总计划4次，不会调用restart/repair。不能只检查函数是否存在或配置能解析；011首轮日志参数冲突就是因此漏掉的。
5. 用同一Linux CPU环境手动执行恒等函数、越界、死循环和禁止import四例，核执行器边界。Mac平台限制失败应单列，不能关闭限制后继续。
6. 改thinking/mode/时间或token参数时identity应变化；普通完成记录才能复用，截断旧配置不能命中新配置；K正式计时全部禁用候选缓存。
7. 演示预算不足时零生成退出，第一stage包含加载时间，保存预算只扣一次；模拟worker超时由父进程退出并留日志。K还核每阶段事件与父进程截止，不以子进程自身的时钟判断代替硬终止。

CPU报告只证明路径可调用、参数与记录正确，不声称27B性能、CUDA兼容或解题效果。所有CPU报告和012包必须对应相同新core SHA；不把011 CPU预检报告用于认证012改动。

## 10 Notebook与产物交接

打包器预计生成三个独立私有Notebook目录：`experiments/012/kaggle/cpu-preflight/`、`kernel-probe/`、`program-completion/`。CPU包不启用GPU；两个GPU包均L4×4、无互联网，复用011实际docker image及固定模型source；它们不需要S4 adapter，避免加载无关权重。K另挂已核实的新wheel Dataset版本，P不挂。

使用新编号和slug，不覆盖`arc2-011-*`版本。发布采用现有已验证的CLI链路，先确认本地CPU预检与包内SHA，再由用户手动push；以CLI返回的真实slug、版本和session为准。不要把计划slug或Notebook metadata id当作实际session id。

产物最低要求：

| 目录 | 必留内容 |
| --- | --- |
| `experiments/012/inputs/` | 原P2文件、单task query清单、来源/SHA；K包排除评分文件 |
| `experiments/012/wheel-manifest.json` | 精确文件名、版本、来源、实测SHA、目标Python/ABI及实际Dataset版本 |
| `experiments/012/kaggle/*/` | 冻结Notebook、metadata、core/source manifest、配置，不含自动push |
| K输出 | env、父子进程日志、各trial实际函数来源、prompt/token SHA、logits或可复查差异、四卡指标、阶段耗时与退出原因 |
| P输出 | 全部raw生成、逐proposal和执行记录、每stage状态、配置/身份、离线评分、四卡指标、父进程监督摘要 |
| `experiments/012/execution/` | 每次开始/结束账户快照、Notebook实际版本、session、墙钟、used差和reserved释放情况 |

创建`docs/progress/012-kernel-program-completion.md`时记录：运行日期、数据与划分、代码/权重身份、实际硬件、关键配置、完整/未完成分母、时间、配额、产物路径、问题及下一步。K与P结果分开；未运行项继续明确写“未运行”。

交回研究结果时，至少提供K的数值/耗时表和退出状态、P四次原始输出及逐题评分、两次会话的实际配额差。看到这些证据后再决定是否值得完整桥接、反馈修复或TTT补片。
