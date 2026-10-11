# 011 — 有效GPU预算与适配研究：实际进展

日期：2026-10-10研究交接；2026-10-11按用户新授权实施并完成两次有界GPU会话。对应[011计划与实施交接](../plan/011-budget-efficient-adaptation.md)。§1–7保留10月10日的研究记录，§8起记录实际执行。**S0 v2与S1 pilot v1已结束并收回产物：F0-25为0/14，多视角完整配对0/5，TTT A0/A4均0/2；程序4次均在思考阶段耗尽1536 token，没有产出代码，反馈修复尚未测到。当前停止扩算，账户剩15.23138配额小时、reserved=0（10月11日15:00北京时间快照）。实际结果和后续决策见§11–14。**

源码审阅基线为Git HEAD bfe7b46d65601c8ad5768f99e30d4a2725f8861c。10月10日改动仅文档与研究证据；10月11日实现变更和冻结代码SHA见§8。

## 1. 决策

固定Qwen3.8-27B，依次研究：既有F0-25闭环 → P6多视角与小步TTT → 有限程序反馈 → 有信号方案的独立确认。先实现一次底座加载后的真实共享评价，再补小TTT/程序入口；不能因已有同名文件就认为已可运行。

不默认再训练同池50/100步。数据独立实例/规则覆盖继续CPU研究，待对照明确后另立GPU实验。公开NVARC/Tufa训练投入远大于本项目小试验；当前0/14既未验证那些完整路线，也不能宣布27B上限。

## 2. 实时资源与远端产物

Kaggle API查询时间：2026-10-10 01:34:59 UTC，即北京时间09:34:59。

| 项目 | 返回或计算 |
| --- | --- |
| GPU总额 | 108000秒，30配额小时 |
| 已用 / reserved | 2212.124秒 / 0秒 |
| 剩余 | 105787.876秒，29.3855配额小时 |
| 下次刷新 | 2026-10-17 08:00北京时间 |
| 官方L4×4扣额 | 两倍 |
| 剩余名义四卡机器时间 | 14.6928小时，按官方2倍算术 |
| 计划承诺 | 10小时墙钟，按保守系数2.3预留23配额小时，另留6.3855小时 |

2.3仅是规划系数；历史账户差仍有未解释部分，不能称稳定实测倍率。每次作业前后核实used/reserved。规则API重查确认无网、完整Notebook≤12h、L4两倍扣额。[runtime-evidence.json](../../experiments/011/runtime-evidence.json)归档选择性原始响应，没有保存短期签名下载URL。

### F0-25产物身份

- 允许访问的 ccjjllt/s4-f0-train-010 返回metadata id137317263、版本号1、输出版本目录355735798；列表含step25配置、formatter及safetensors。
- 初次把数字1传作versionLabel返回“version label not found”；后读默认当前版本，metadata明确current_version_number=1。versionLabel与版本号是不同字段。
- 本地kernel-metadata写 ccjjllt/s4-f0train-010；访问该对象返回permission denied，未再尝试该对象的其他读取方式。不能把相似slug当同一对象。
- 允许对象完整运行日志存于[s4-accessible-notebook.log](../../experiments/011/s4-accessible-notebook.log)。SUMMARY与本地逐字段一致；25条打印update指标按显示精度一致。stdout没有episode IDs，因此未声称远端100条ID已核对。
- 本次未下载权重；用户开GPU前需从有权访问的对象取得step25权重和train_log，核对消费ID及文件SHA。相同SUMMARY不能证明权重字节一致。

## 3. CPU成本审计

手动用标准库读取JSON/JSONL，未反序列化trainer_state、未加载safetensors张量。逐query值、口径和39份来源SHA见[local-cost-audit.json](../../experiments/011/local-cost-audit.json)。人工研究及零散CPU统计耗时未单独计量，不作为性能结果。

| 项目 | 实测时间 | 口径 |
| --- | ---: | --- |
| F1-25 | 1044.5秒，17.4分钟 | 25次成功更新，100例，不含加载/保存 |
| F1-50 | 2131.5秒，35.5分钟 | fresh的50次更新，200例 |
| F0-suffix-25 | 2257.6秒，37.6分钟 | 25次更新，同100例 |
| F0底座P12 | 2694秒，44.9分钟 | 14次generate，含prefill与decode |
| G1 F0 P12 | 3106秒，51.8分钟 | 合并补缺后14次generate |
| F1 P12 | 1373–1456秒，22.9–24.3分钟 | 零步/25步/50步各一臂 |
| 重复训练前缀 | 2465.1秒，41.1分钟 | F1-50重复前25步＋S4重复前17步，不含加载 |

F0/F1的实际100条消费ID与010预定前缀一致；同seed、LR、horizon100、LoRA和软件版本。F0更新时间约F1的2.16倍，是同样例、不同格式且suffix/full不同的完整配置成本比较，不是纯格式/内核因果倍数。历史约4.3 token/s是网格生成的综合有效速率，不能预测新程序/长思考速度。

S4到S4_DONE为5666秒；基座加载3308.9秒，占58.4%，成功更新占39.8%。**减少重复加载是有直接证据的优先项。** 病因尚未拆分，不能确定是冷存储、mmap、初始化还是GPU搬运。

## 4. 更正010中不宜继承的概括

1. S3/S4的F0 suffix更新为17+25=42，最长独立前缀25步，不是93次。S4最高loss0.25856；“25步零OOM”成立，但不证明所有4K配置都无OOM。
2. suffix探针实际长度990/1212/2072。loss一致、梯度差近同路径噪声，只是有限样本证据；显存字段不是完整full/suffix分开的峰值对照。成功F0前100例最长3844，不是全400例最大4060都训过。
3. E0首载312.7秒；补缺首载6584.0秒、暖载26.2秒，中间有持续stderr进度；S4另有3308.9秒慢载。旧文“两个6600秒静默”“S3/S4未慢载”“E0五倍账差已解释”不能继承。
4. 010末尾S3扣额1.28h与§14的3.06h冲突，并遗漏F1-25轮。缺少逐作业原始账户快照，不编造统一校正或新稳定倍率。
5. F1@0/25/50的21897d95 q0均触及960，生成31×30；扩大上限未必修正。aa4ec2a5解析失败是另一题。禁止按query真值尺寸裁答案。
6. F1-25→50仅3/14生成文本改变；与零步各有6/14改变。因此adapter并非完全不影响输出，但仍没有命中增益。

旧010段落保留研究历史，上述纠错以原始产物为准。

## 5. 一手方法依据与取舍

查阅日期均为2026-10-10，详见[方法研究备忘](../../experiments/011/methods-research.md)。

- NVARC：借鉴合成规则/实例验证、题内LoRA、候选一致条件评分；不复制32×H100规模或直接移植attention-only DFS缓存。
- Tufa 2025：借鉴待解集合已知示例适配与增强投票。主要组件消融在ARC1.5/小模型，不能解释2026公榜88.06。上一轮简写为“逐题TTT”不足以表达实际集合级适配。作者15.42与官方2025最终表12.64有口径差异，本轮不混用。
- ARChitects：ARC1累计加组件不是等预算ARC2证据；更深搜索的oracle提升未充分变成最终选2，因此先问候选有无真解，再谈昂贵排序。
- Poetiq：参考程序执行反馈，与同预算独立restart比较；不外推API大模型成绩，不扩成多专家长循环。
- Qwen3.8：可以直接用原底座的thinking/保留状态接口做小程序实验，不换模型，不把上下文容量等同于推理质量。

最终采用011计划的23配额小时承诺表。备忘中的其他预算、B8或数据SFT是候选建议，未纳入本周默认GPU工作，不能叠加。

## 6. 实现审计与CPU模板实证

详见[实现研究备忘](../../experiments/011/implementation-research.md)。本轮只把具体修改交接给用户：

- runner的评价是占位，非dry-run退出；TTT GPU循环明确未实现。
- runner判重参数数目不符、tuple误当dict；solver权重指纹不含实际权重字节，候选可能直到整query才落盘。这会造成重算或误复用，需局部修正。
- balanced不是四worker；现采样临时替换进程级torch.multinomial，不能直接多线程generate。
- 固定TF5.8默认最多4线程异步加载；HF_DEACTIVATE_ASYNC_LOAD是可读源码支持的候选；旧HF_ENABLE_PARALLEL_LOADING不是该路径有效开关。
- GDN四个函数和norm分别选快慢路径；警告不足以判定全部fallback。FLA/conv离线实物与真机数值/性能仍未完成。

本地tokenizer-only渲染实际确认：默认xhigh；low加简短思考提示；medium不加额外effort文案；high报错；关闭thinking忽略effort。preserve_thinking默认保留reasoning_content；False并不删除当前问题之后工具链的全部思考。这些参数属于apply_chat_template，不是generate。

CPU环境Python3.9.6、transformers4.57.6、jinja2 3.1.6、tokenizers0.22.2，无模型加载。模板SHA为c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041。该核对只证明字符串行为，不替代Kaggle TF5.8推理效果。

## 7. 清单、复核与交付

CPU从010 manifest生成[research-manifest.json](../../experiments/011/research-manifest.json)：P6=6task/6query，P2=2/2，C6=6/8，C6-reserve=6/8；两个确认块仍是原C12=12/16。只按来源/输入长度层与原顺序选择，未读确认query答案或成绩，未改010清单。

交接稿经过独立只读复核，已补：原底座TTT的fresh LoRA定义；不适用task回退A0并保留分母；相同时间上限与实际消耗的区别；A8只由真实query新增触发；确认对照继承开发问题；GDN不能预装后默改baseline；repair输入长度与反馈上限。

交付文件：011计划/实际进展；本地成本审计；配额/规则和远端元数据证据；允许访问对象日志；研究清单；方法与实现备忘。README和010顶部更新当前指向。

交付前手动核对：三个JSON均可解析；P6与两确认块无交集；两确认块完整还原原C12；task/query计数一致；清单不含output/solution/gold字段；原010 manifest及成本审计来源SHA未变；本轮本地Markdown链接均存在；git diff --check通过；scripts目录无diff。没有新增测试文件、测试目录或框架。

仍未运行：F0-25 P12、新共享执行器、加载/内核性能改进、P6多视角/TTT/程序反馈、C6/C12确认、新数据SFT。未确认：远端权重SHA、慢加载病因、稳定扣额倍率、新方法解题收益、12h全量提交可行性。

上述为10月10日交接时状态；10月11日用户要求“你来操作”，执行责任和预算更新如下。代码实现完成不构成提分证据。

## 8. 10月11日：接管实施与预算收缩

用户授权由助手实施并操作有限GPU实验，优先提高有效配额利用率。新授权覆盖原“由用户实现”的交接分工，不扩成大规模训练、CI或竞赛提交。

- 开始时账户剩20.04532配额小时，首GPU前复查剩19.78379小时、reserved=0。期间执行的是CPU作业，不能把账户变化归因于本项目；账户还服务其他研究。
- 旧23小时承诺不再适用。新工作最多占用14配额小时，并始终按最新余额留至少6小时；首作业按实际最新余额计算可分配上限为13.78379小时。这是上限，不是必须花完的目标。
- 首会话Notebook硬上限7200秒、模型加载1200秒；超过加载上限即退出，不自动重试。L4×4按官方两倍规划；账户实际账差另记，不混同墙钟。
- 证据：[授权与初始预算](../../experiments/011/execution/authorization-and-budget.json)、[首GPU前额度](../../experiments/011/execution/pre-s0-quota.json)。

实现文件：

| 文件 | 已实现 | 当前实际证据及边界（更新至本轮结束） |
| --- | --- | --- |
| `scripts/grid_runtime.py` | 一次底座加载、具名PEFT切换/禁用、真实候选生成、逐候选持久化、权重SHA与完整配置判重、token截止 | 两次GPU共享会话通过；S0禁用adapter后6/6原底座输出与旧记录完全一致 |
| `scripts/run_budgeted_grid_experiment.py` | 一个模型worker、父进程加载/总截止、显式工作清单、P12/P6与TTT/程序阶段、事后评分与完整/未完成分母 | GPU产物已收回；S0多视角仅5/6完整，程序策略完成不等于反馈机制测到 |
| `scripts/ttt_per_task.py` | 同初态A0/A4、每题fresh optimizer、答案suffix loss、预算/异常回滚A0、LOO长度拒绝 | P2真实更新成功，但A0/A4均0/2；A8和等实际时长无TTT补偿队列未实现 |
| `scripts/solve_program.py` | P0、独立restart、带执行反馈repair；保留thinking；受限AST及Linux子进程时空限制 | P2初次及restart均未产出代码；实际repair调用0次，机制效果未确认 |

没有新增测试文件、测试目录或框架。手动CPU核对覆盖配置、断行恢复、身份变化、无标签生成入口、部分配对评分、tokenizer与受限执行器。程序执行器是受限实现，并不声称完整操作系统沙箱。

### 已有S4权重恢复

从允许访问的 `ccjjllt/s4-f0-train-010/1` 下载step25及训练元数据。SUMMARY/run_config/train_log与010本地归档逐字节相同；无需重新训练25步。

- `adapter_model.safetensors`：159451232 bytes，512 tensors（仅解析header）。
- SHA256：`9282c46cbce02923afbea3ad041614f0bc6ff3d7bd97c5174adfa3dd3e485ff9`。
- 产物：[身份记录](../../experiments/011/execution/s4-artifact-identity.json)、`experiments/011/recovered-s4-v1/010/s4-f0/`。大权重已加入忽略规则。

### Kaggle CPU预检 v1：完成

[Notebook](https://www.kaggle.com/code/ccjjllt/arc2-011-cpu-preflight)，version 1。不加载模型权重，GPU count=0，执行正文67.11秒。

真实环境：Python3.12.13、Torch2.10.0+cu128、Transformers5.8.0、PEFT0.19.1、Accelerate1.13.0。离线wheelhouse、模型版本1、S4 adapter挂载与字节SHA均通过。P12输入12task/14query，P6为6/6。thinking/nonthinking模板长度1416/1392且结构差异符合预期。

Linux受限执行器人工四例：恒等变换正确、越界报告execution_error、死循环按0.2秒超时、import被拒绝。结果见[CPU预检](../../experiments/011/kaggle-cpu-preflight-v1/011/cpu-preflight.json)。这不证明GPU路径或模型能力。

首GPU包的12份核心代码/输入SHA与此CPU预检完全一致。Notebook外围随后把P6预算由1800缩为1200秒，修正SUMMARY显示大小写，并增加每30秒nvidia-smi CSV；没有改变核心生成算法。CPU远端v1保留原配置。

### 首GPU会话 v1：主动取消，无模型结果

真实发布地址：[arc2-011-s0-shared-model](https://www.kaggle.com/code/ccjjllt/arc2-011-s0-shared-model)，version 1。CLI按标题生成了带`-model`的slug；一次查询原计划slug被拒绝后，使用CLI返回的真实地址。没有再次发布或重复启动。

- 同一个BF16 Qwen3.8-27B、balanced四卡会话，启用有源码依据的`HF_DEACTIVATE_ASYNC_LOAD=1`；不新装GDN内核，不做性能提升预断言。
- S0：F0-suffix-25，P12全部14query，original greedy，2048新token，每候选最多600秒。
- S1：原底座F1，P6 original greedy / original sample / transpose greedy，960新token，整个阶段最多1200秒。按题顺序执行，因此若预算不足，明确记部分配对，不能报成完整P6分数。
- S1是预先冻结的机制诊断，不是看S0结果选择的最优checkpoint。seed固定20261011（较原计划20261010有记录的变更）。F0已有权重不重训；TTT/程序/C6尚未进入本作业。
- 外围安装/输入校验计入Notebook预算，保存窗口300秒；逐候选落盘，父进程强制截止，无自动重试。
- 记录四卡利用率/显存/功耗，需将加载、推理阶段分开解释；balanced是分层模型并行，不能把四卡当四个独立worker。

本地v1包与代码SHA在 `experiments/011/kaggle/s0-shared/` 和[源码清单](../../experiments/011/execution/packaged-source-manifest.json)。远端产物目录 `/kaggle/working/011/s0-shared/`。

只读复核发现一个尾部状态边界：当前冻结版在资源截止后若残留网格>30，可能把`budget_interrupted`覆写为`invalid_output`。本次结果需结合finish_reason、时间和raw输出核对；后续小修应保留资源中断优先级，不能把该边界误报为模型失败或完整评分。

进一步手动CPU桩完整调用`generate_grid`，发现首候选日志有确定错误：`event(kind, **job)`和`job['kind']`重名，触发TypeError。早先预检覆盖配置与执行器，未覆盖完整候选包装路径，这是助手的预检遗漏。

发现后立即使用返回的作业输出URL确认session ID **357234485**，发送取消并确认`CANCEL_ACKNOWLEDGED`；账户reserved已回0。保存的实时流日志末尾为Notebook正文226.5秒、权重385/851，没有load_done或candidate_start。不能把被取消当成0/14模型结果。

首会话前后账户used从36778.3559275增至38728.299665秒，差1950秒=**0.54165配额小时**；剩19.24214小时。此为账户账差，不足以单独推算该作业的扣额倍率，完整记录见[取消与配额](../../experiments/011/execution/s0-v1-cancellation.json)。不隐去失败成本。

后续修复：日志形参更名、资源中断优先级、TTT事后评分与完整配对状态、TTT/program完成性、A4训练及生成联合准入、fallback复用A0、逐update持久日志。手动CPU桩通过；需要在Kaggle CPU新版预检中重复完整候选包装调用后，才发布一次修正版GPU。v1包不修改、不自动重试。新S0包预备在`experiments/011/kaggle/s0-shared-v2/`，尚未发布。

### 新版CPU预检与S0 v2

[CPU pilot preflight v1](https://www.kaggle.com/code/ccjjllt/arc2-011-cpu-pilot-preflight)实际COMPLETE，正文54.02秒、GPU=0。6次F0/F1×original greedy/sample/transpose经过真实`generate_grid → deadline wrapper → _run_one_candidate → parse → journal`，仅model.generate固定输出；另外验证缓存命中和过期准入。一个随机微型Qwen3在CPU上运行真实PEFT r8/alpha32及2 token生成。它不是27B的效果/性能验证。Linux执行器4例再次通过。

产物：[完整CPU报告](../../experiments/011/kaggle-cpu-pilot-v1/011/cpu-pilot-preflight.json)。七份核心源码汇总SHA256为`bbc9b83fc24f7b216878695cf7ca41fec83af15ab94bbb8af640047e2896a97e`。

S0 **version 2随后发布并开始加载**，阶段/权重/候选/预算沿用v1，仅使用修复后的代码和CPU报告核心SHA校验。旧v1包和取消记录保留；[v2发布记录](../../experiments/011/execution/s0-v2-publication.json)、[v2开机前余额](../../experiments/011/execution/pre-s0-v2-quota.json)。开机前剩19.01471配额小时、reserved=0；取消后到账仍有变化，因此上一笔0.54165h是当时账户差，不能冒充已结算的单次成本。

v2源包在`experiments/011/kaggle/s0-shared-v2/`，实时日志通过官方CLI `kernels logs --follow`取得（输出列表接口在运行期间可能为空）。此处保留运行中的记录；最终评分见§11。

v2实际加载 **310.335秒（5.17分钟）**，device_map仅cuda0–3，无CPU/disk offload；adapter激活后已连续完成真实候选。证据：[load_done完整事件](../../experiments/011/execution/s0-v2-load-evidence.json)。本次没有复现S4的3308.9秒慢载，不能据此声称异步加载开关单独带来10倍加速。首6条去重候选的综合速率约4.13–4.40 token/s，包含prefill/调用开销。

## 9. GPU准备期间的CPU数据复核

见[训练侧20例复核](../../experiments/011/data-next-study.md)。按预定6族抽20个呈现，实际13个独立query base；80/80呈现pair符合009既有规则重建，六组旋转均为同base变换。此为针对已知风险的目的抽样，不能外推全448族问题率。

关键发现：旧G2全局删除`target_equals_demo_output`会让只有两类固定输出的`8597cfd7`保留缺少query类别示例的episode。新base也有query=demo及常量计数捷径，因此不能把旧/新池差别都归因于独立base数量。下一轮应使用一致flag-only政策，按同family配对审查独立base/旋转；当前没有启动额外SFT。C6/C12答案仍未读取。

## 10. 吞吐瓶颈的实际证据与后续准备

48个GDN实例实际绑定PyTorch causal-conv/update、chunk/recurrent实现，norm为Qwen3_5RMSNormGated。这个结论来自实例函数身份，不是只看warning。未改变本轮底层算子。

[加速内核兼容性备忘](../../experiments/011/gdn-followup-evidence.md)补齐官方精确wheel链接、大小、SHA与静态接口审查。后续S1日志实读CXX11 ABI=true、Triton3.6.0、Torch CUDA12.8；L4导入、JIT和数值仍未验证。FLA recurrent无backward，推理通过不等于TTT后向通过。建议未来CPU备包后进行≤30分钟同题64 token推理探针，JIT最多10分钟，失败即止；本轮未安装或另起GPU编译。

四卡balanced是按层串行模型并行，不能用四卡总带宽直接估计单query速度。官方L4 300GB/s与55.563GB shard得出的0.185s/token只是全权重流读代理，既不是实际解码流量，也不是吞吐硬上限。最终解释还需四卡采样数据；独立手工分析入口为`experiments/011/summarize_session.py`，不更改冻结的GPU求解代码。

## 11. S0 v2：已完成，既有F0权重没有新增解

[Notebook](https://www.kaggle.com/code/ccjjllt/arc2-011-s0-shared-model)，version 2，session 357238142。实际硬件4×NVIDIA L4，Qwen3.8-27B BF16按层balanced，无CPU/disk offload。软件与CPU预检一致；使用§8冻结七文件core与S4 step25权重，未重训。数据来自010既定开发P12/P6，重复用于研究，不能称为盲测或比赛分数。

父进程`supervisor.status=complete`，退出码0；`SUMMARY.status=partial`是因为1200秒多视角阶段只完成5/6三候选配对。两者分别表示执行正常和计划覆盖不完整，不冲突。

| 比较 | 实际覆盖与结果 | 成本或边界 |
| --- | --- | --- |
| F0-suffix-25、original greedy、2048 token | 12 task/14 query，**0/14**；13合法完成，1非法输出 | 2791.617秒生成，11876新token；非法题为`aa4ec2a5`，不是资源中断 |
| 原底座F1、O+S | 完整配对5/6，**0/5** | O=original greedy，S=original sample；不能记0/6完整配对 |
| 原底座F1、O+T | 相同5题，**0/5** | T=transpose greedy；没有新增解 |
| 原底座F1、O单候选 | 全6题完成，**0/6** | 与010独立底座的6条文本和网格均完全一致 |

最后一题`aa4ec2a5`只完成O；S/T因剩余预算不足120秒准入余量而跳过，记录2条`budget_interrupted`，实际生成成本为0。另有`291dc1e1`的S非法。总journal32条，其中30次真实生成、2次预算跳过；28合法完成、2非法输出。缓存复用次数0。

共享加载后禁用S4 adapter的原底座桥接见[逐题复核](../../experiments/011/execution/s0-v2-base-bridge.json)。这支持切换后无明显输出污染，不是所有内部张量的数学等价证明。

会话监督计时4262.593秒（71.04分钟），其中加载310.335秒（5.17分钟），候选调用3899.586秒（65.0分钟、占91.5%）。91.5%是墙钟用途占比，**不是GPU硬件利用率**。产物含`SUMMARY.json`、`scoring.json`、两阶段JSON、逐候选journal与四卡CSV，保存在`experiments/011/kaggle-s0-v2/011/s0-shared/`；手动分析见[S0汇总](../../experiments/011/execution/s0-v2-analysis.json)。

## 12. S1 pilot v1：真实TTT完成，程序反馈未进入执行

[Notebook](https://www.kaggle.com/code/ccjjllt/arc2-011-s1-pilots)，version 1，实际COMPLETE。看完S0后按预定无SFT增益分支选择原底座F1；P2仍是事前冻结的`db118e2a`（H_new）与`291dc1e1`（E24），每题query 0，没有按成绩换题。[发布前决定](../../experiments/011/execution/pre-s1-pilots-decision.json)保留配置、配额与Notebook SHA；[最初设计](../../experiments/011/execution/s1-pilot-design.json)的`not_launched`为当时状态，实际执行以本节为准。

同一BF16权重和4×L4；Python3.12.13、Torch2.10.0+cu128、Transformers5.8.0、PEFT0.19.1、Accelerate1.13.0。下载的七份core、冻结manifest及本地core逐字节SHA相同。实际`torch_cuda=12.8`、`cxx11_abi=true`、`triton=3.6.0`。未安装FLA/conv包或切换内核。数据输入只含query input，评分答案在生成之外读取。

### A0/A4题内适配

两题均从同一fresh LoRA初态开始，每题新optimizer，r8/alpha32、lr2e-5、4次update×4 microsteps、LOO示例、4096长度上限、suffix loss；A0/A4均O+T、每候选最多960新token。总计8 updates/32 microsteps，8个候选全部合法、EOS结束、无截断、无缓存、无fallback。

| task | 训练秒数 | 梯度范数范围 | A0→A4 query命中 | 输出变化 |
| --- | ---: | --- | --- | --- |
| `db118e2a` | 190.278 | 0.265–0.312 | 0→0 | O、T均不变 |
| `291dc1e1` | 163.650 | 1.020–1.391 | 0→0 | O由26×4变24×4；gold为18×8，仍错误；T不变 |

因此完整配对A0 **0/2**、A4 **0/2**，新增0、丢失0、两臂都错2。梯度全部有限非零；共同初态SHA `76fa2c33716c10a969e909784c1883c1d86ded421bac73d9b5276dd5378b3c7e`，最终SHA分别`4cf791f74cb072adfad664984108f10faf8c678edeb5bcd6583630e9b6a5eecd`和`4641a0ab0179bef1519379128ca8ae62a7ae8970c196cf541c913c1cce660684`，说明真实更新发生。loss下降或权重变化不代表解题收益。

独立复核从原始文本重新解析、逆转置、对gold计算EM，结论一致。两题A0的O/T均与S0原底座一致，包括第二题；代码恢复初态并新建optimizer。不过恢复返回的是保存的初态SHA，没有重新计算恢复后张量SHA，也未做同题A0→A4→恢复A0回放，**不能记为完成全量张量恢复验证**。

纯训练353.928秒；A0生成172.077秒，A4生成168.618秒。加载后的TTT阶段约699.16秒；首个`stage_done.elapsed_s=1094.399`还包含前置准备和加载，不全是训练。当前只比较相同候选数，**尚无等实际时间无TTT对照**，0/2也不足以判定TTT普遍无效。

### 程序完成性与反馈分母

每题P0和独立restart各调用一次，共4次；thinking low、最多1536新token、8192输入上限、360秒/调用、1080秒/task。四次全部1536 token、`finish_reason=max_new_tokens`，没有`</think>`、Python代码块或`transform`。输入仅2586/2602 token，每次338.65–338.84秒；实际瓶颈是**输出预算耗尽**，没有触及输入上下文或时间上限。

两题repair均为`ineligible_no_complete_p0`：**repair生成0次、生成程序执行0次、反馈机制完整配对分母0**。结果里的`demo_matches=0`是未执行情况下的占位，不能称“代码执行后示例全错”。完整策略P0+restart、P0+repair各可记query EM 0/2、有效候选覆盖0/2；这只是当前固定预算策略的结果，不能称同等实际计算的反馈对重试实验，更不能据此判定反馈修复无效。

真实生成合计6144 token、1354.888秒；程序阶段1355.467秒（22.59分钟）。P0+restart逻辑成本6144 token/1354.888秒，P0+repair只有P0的3072 token/677.544秒；共同P0不能重复记入真实总成本。远端`scoring.json`仅含TTT；程序口径来自下载后手动离线评分，已在分析器中明确`policy_paired_complete_scored=2`与`feedback_mechanism_paired_complete_scored=0`。

完整产物在`experiments/011/kaggle-s1-pilots-v1/011/s1-pilots/`，包括逐update日志、program事件、原始生成文本与telemetry；[S1离线分析](../../experiments/011/execution/s1-pilots-analysis.json)和[运行身份记录](../../experiments/011/execution/s1-pilots-publication.json)提供可追溯入口。会话监督计时2453.936秒（40.90分钟），加载352.118秒（5.87分钟）。`work_complete=true`表示预定策略已结束，**不表示反馈修复实际运行**。

## 13. 实际配额与四卡利用率

| 会话 | 监督墙钟 | 账户used前→后（秒） | 账户差（配额小时） | 结束后余额 |
| --- | ---: | --- | ---: | ---: |
| S0 v2 | 71.04分钟 | 39547.0437552→48222.422 | 2.40983 | 16.60488小时 |
| S1 pilot v1 | 40.90分钟 | 48222.422→53167.042 | 1.37351 | **15.23138小时** |

证据：[S0后快照](../../experiments/011/execution/post-s0-v2-quota.json)、[S1后快照](../../experiments/011/execution/post-s1-pilots-quota.json)。后者查询于2026-10-11 07:00:44 UTC（北京时间15:00:44），reserved=0，下次刷新北京时间10月17日08:00。账户还可能被其他作业使用，以上是账户差，不是逐设备时长或精确隔离的作业账单。两成功会话账户差合计3.78333小时；失败v1和结算漂移另见§8，不遗漏、不强行归摊。

四卡每30秒采样，S0每卡141点，均值分别17.60%、25.50%、26.62%、23.71%；S1每卡81点，分别13.01%、24.46%、21.80%、24.90%。这是**全会话稀疏采样的GPU忙碌比例**，包含加载/训练/生成，不能说已经打满四卡，也不能把候选区间之外都叫空闲。NVML最大显存占用S0为20756/19678/22542/20636 MiB，S1为13452/15426/15392/16336 MiB；它包括分配器保留，不能代替逐阶段peak allocated。

一次加载服务多个研究臂已成立，且本轮加载为5–6分钟，没有复现旧S4约55分钟慢载；没有同条件A/B，不能把差异全部归因于加载开关。4卡按层串行、batch1及GDN回退仍是吞吐研究重点，当前证据没有证明GPU内核优化收益。

## 14. 本轮停止点与下一步顺序

本轮不继续启动GPU；不扩A8、程序P6、C6/C12或新SFT。C6/C12确认答案未读取、未运行，也没有新竞赛提交。F0/多视角/TTT均未见开发query新增，程序分支则尚未通过可完成性检查；这些小片结果不证明27B或公开完整方法的能力上限。

1. **先做CPU准备**：按§10备齐固定内核wheel及依赖、核SHA，写好有界同题探针；完成前不上GPU。未来单次探针建议≤30分钟墙钟（按2.3规划约1.15配额小时），JIT≤10分钟，异常即止。此探针未实现、未运行，不算当前已承诺支出。
2. **先解决程序能否完成**：不照原配置跑P6。优先在CPU设计仅原P2、仅P0的完成性对照，区分直接代码输出与延长thinking；只改一个因素并冻结模板。当前约4.53 token/s，3072 token粗估约678秒，因此仅加token而仍留360秒上限无效，需同时明确总时间准入。3072是否足够未知，不能用“无限上下文”替代有上限的预算设计。没有P0和示例执行反馈前不做restart/repair扩展。
3. **TTT保留有限补片选项**：P2的0/2不足以否定TTT；只有在性能/成本探针支持后，才考虑原P6剩余4题、同配置A0/A4的一次补片，不加步数、不调学习率、不换题。等实际时间无TTT对照尚缺，不能宣称TTT成本优势。补片如仍无新增则停止该配置扩算；本次未发布。
4. **数据研究继续留CPU**：按§9统一过滤口径、同family核查独立base与捷径。没有形成可解释数据对照前，不用剩余配额机械延长SFT。

任何后续GPU都重新读账户余额，并保留至少6配额小时；当前15.23小时不是必须花完的目标。本次只修正离线汇总的反馈分母，没有改动已运行的冻结core或Notebook包。结果收回后手动复核产物、权重/代码身份与评分，无新增测试框架。

用户随后要求由自己实现下一轮，具体代码改动、同额D/L程序对照、K30分钟与P75分钟包络已整理为[012实施交接](../plan/012-kernel-program-completion.md)。012尚未实现或运行；本节候选步骤由012的明确范围和预算进一步收缩，不叠加支出。
