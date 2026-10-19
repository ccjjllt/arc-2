# 016 — 有限思考、最终答案恢复与赛前提交

日期：2026-10-18，北京时间。对应[研究与实施计划](../plan/016-budgeted-final-submission.md)。本轮完成本地结果复核、当前比赛查询、公开源码/前沿方法研究和CPU预算计算。**没有新GPU实验、训练、求解器实现或比赛提交，也没有新的比赛分数。**

**10月19日更新**：第1–6节保留10月18日研究过程；其中旧“固定C4、high/low各8K”的实施安排已由第7节及修订计划替换，不能照旧执行。第2节榜单、额度、排队信息仍是所标日期快照，本次仅写交接，没有重新查询。

## 1 当前决策

今天有1次提交额度，但015是固定P6研究程序，没有全集输入、完整双答案导出与全量调度，不适合直接提交；013的25.69也不原样重投。保留历史30.14，先完成一次有限恢复与短预算实验；一旦有真实新增、确认结果和可运行预算，尽早提交，不必等到金牌分数才提交。

工作目标是10月19–20日得到第一轮新证据、10月21–23日争取形成第一份完整候选，10月29–30日前完成最终候选安排。实验没有支持新候选时如实记录，不为日期目标制造提分结论。

## 2 比赛和资源已重新查询

官方API查询于18:34北京；比赛页面随后再次读取。[账户原始记录](../../experiments/016/evidence/live-account.json)、[榜单下载摘要](../../experiments/016/evidence/leaderboard-summary.json)、[官方页面原文](../../experiments/016/evidence/competition-pages.json)。

| 项目 | 本轮核实 |
| --- | --- |
| 队伍 | jiyujia，成员ccjjllt/jiyujia；最佳公榜30.14，第1015名 |
| 最新正式提交 | 57103498 COMPLETE，25.69；私榜为空 |
| 当前公榜参照 | 第一88.06，第8为68.33，第16为61.81 |
| 队伍数 | leaderboard下载3020行，比赛metadata为3019；保留接口快照差异 |
| 金牌/奖金 | 按当前队数公式约16个金牌名额；排名奖金前8。公榜参照不是最终门槛 |
| 提交 | 今日0次，当前可提交1次，累计8次；规则最多2份最终候选 |
| 最终截止 | 11月3日07:59北京；入队/合并截止10月27日07:59 |
| 当前CLI身份 | ccjjllt；GPU allowed108000秒、used29307.285秒、reserved0，剩21.8591配额小时 |
| 配额刷新 | 10月24日08:00北京；不把另一身份jiyujia的12.94旧快照相加 |

本队距当前第16名31.67分、距第8名38.19分。现阶段不能基于6道开发题预计金牌/奖金概率。

通过浏览器读取[官方排队公告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/747515)：Dustin在10月16日22:13北京称当时排队约3–5小时，多数等待已是实际运行。它更新了旧33小时/隔天一次快照，但不是现在的实时ETA。每日1份之外仍有团队同时1份pending/running限制。[读取记录](../../experiments/016/evidence/queue-update.json)。

官方当前规则仍要求无网、12小时、公开外部数据可用、L4×4按2倍配额消费、完整`submission.json`。排名奖金与writeup奖不同；规则称Innovation Prize而Prizes页称Grand Prize，官网Bonus描述也有差异，后续以Kaggle正式规则核对，不混为“拿金牌自动有奖金”。

## 3 本地复核与纠错

[独立审计笔记](../../experiments/016/project-audit.md)重新检查015原始调用与实际波次成本：

- high：155733 completion token，4106.13秒，合法final1/6，已有EM@2为1/6。
- low：186047 token，2234.37秒，合法final2/12，已有EM@2为1/6。
- 总计18次调用，14次length+空final、1次stop+空final、3份合法final；涉及2个query。不是“15道题不会解”。
- 固定high→low1→low2、合法网格去重取前二，可以在既有产物保留两题命中，离线回放为2/6。不是新推理/公榜，也不证明未来三个不同候选时选2不会损失。
- `58c02a16`高臂命中只用了15941token，小于低臂16384上限。已纠正015中“更多预算换来命中”的过度归因；effort、seed、并发混杂。
- high的`db118e2a`提前stop仍无final。缺少原始stop token ID，根因未确认，不能全部归因于长度上限。

评分边界：`db118e2a`两份low输出与专用P2评分文件独立比对一致；58的命中继承015已有评分记录，合法输出及调用身份已核。没有为重复评分读取C6或其他保留集答案。

## 4 本轮新增的外部证据

### 冻结引擎已具备硬思考预算

对vLLM精确commit `1f7427bc0`读取请求schema、SamplingParams、reasoning配置、budget state、sampler及Qwen parser：请求字段`thinking_token_budget`可传到采样器，到限强制结束思考，`max_tokens`余量用于final。015未传该字段。这里的结论是**源码链路已核实，当前wheel实际生效尚未GPU验证**，无需先升级引擎。

Qwen/vLLM仓库的用户报告出现过提前EOS导致stop+空final，与本地症状相似但不能确认为本地根因。网上提及的`reasoning_eos_policy=force_end`来自未合并且已关闭PR55562，不作为可用配置。来源和固定文件blob见[方法研究](../../experiments/016/methods-research.md)。

### 公开Notebook提供恢复机制，没有核实到可直接复制的金牌27B

重新读取公开score排序前40和近期Qwen前30列表；列表不返回分数字段。又阅读dalezhong v7和mayankraaz v6源码，其标题或正文分数没有绑定准确提交证据，零output cell不能证明已在完整比赛运行。作者队伍当前最高分也不能移给旧源码。

已有russcore v17和madarshbb精确358248812有可借的final-only恢复：将自身思考尾部送入一次关闭thinking的请求。016计划先复用015已存15份失败轨迹，再从头验证硬预算。这比先重跑整套32K/16K更便宜，但其部署成本必须加回原推理。[公开方案笔记](../../experiments/016/public-solutions.md)，[持久来源与SHA](../../experiments/016/evidence/public-notebooks/manifest.json)。

### 昨日PrefARC发布，但ARC2来源是eval变体

DiARC在10月17日13:47UTC发布PrefARC（commit `fb436f2f594f8be6b54e3e1e4e84ac060bcb8776`），已有40个配置清单，014“语料未公开”表述已补更新。ARC2明确是`evaluation_variants`，配置间可能重叠，不是task-disjoint训练。只读README/manifest/入口，没有下载题目答案或adapter。不能混入训练后仍将P12/C6称未见泛化。

前沿补充读了s1、The Coupling Tax、SeVRA、Mid-Think的一手论文/实现。它们支持研究硬预算、独立回答阶段和记录二次修改损失；主要为其他Qwen版本及数学/知识任务，不能迁移其数值当作本项目预期提升。新的ARC论文若没有可用实现/权重，不据表格分数扩建框架。

## 5 实际完成的CPU计算

硬件为本地macOS、Python3.9.6，仅运行序列化计数和算术，没有模型生成。

1. 用既有tokenizer SHA `0997f410…b9f3`编码固定seed20261018的合成30×30网格：空格行围栏1803token，紧凑行933，紧凑JSON1801，默认JSON2700。支持先为严格网格协议留2K；不保证模型遵守格式。[原始计数](../../experiments/016/evidence/final-space-token-counts.json)。
2. 按015实测外推，240query现有high/low/三调用分别约45.81/25.01/70.63小时，含启动但不含额外落盘余量。240是隐藏task数，实际query可能更多。这是场景估计，不是隐藏运行。[场景计算](../../experiments/016/evidence/budget-scenarios.json)。
3. 总运行12小时中只规划10.5小时生成，假设75聚合token/s则约346个8K调用：Q240/280/340第一遍后最多余106/66/6次。65token/s只够299次。要按真实Q和长请求速度分配，不能人人两次8K。
4. 当前ccjjllt余额保留6小时、按2.3计划系数，最多约6.90小时L4×4研究会话。它不是额外获批配额，开机前重查实际运行身份。

## 6 新实施交接与未运行项

主计划冻结了三步，子研究笔记中的备选参数服从主计划：

- A：全部15空final轨迹，末16000字符，一次关闭thinking恢复，每次最多4096新token；最多61440新输出token，保留原3合法final。
- B：固定P6，high/low各一次，总8192、思考6144、相同seed4101、C4；每臂EM@1，固定两槽EM@2，最多98304输出token。部署另做一遍覆盖+选择性第二遍。
- 必要时程序分支：固定P2、共享完整P0，真实demo反馈与等预算独立重启比较，先总8K=4K思考+4K代码；没有完整程序则不称已测试反馈。

A/B建议同一加载、整会话上限2小时；有实质信号再P12、冻结后C6。大量实现与GPU执行交给用户/后续实施，本轮没有创建新测试、启动训练、发布Notebook或正式提交。

后续重点是把已有研究转成“固定两份答案+全部query覆盖+12小时内成本”的实际求解策略。若合法率提高但正确解没有增加，按负结果转向一次程序反馈，不能继续将更多token或更多训练当成默认答案。

交接后人工复核：6份当前文档的本地链接均可解析，持久保存的6份公开来源SHA均匹配，`git diff --check`通过；独立审阅未发现成本或评分口径阻断项，补充了“短请求必须触及极小思考上限才验证机制”的说明。这些仅是文档与来源核对，不是模型实验或提分证据，未创建测试文件。

## 7 2026-10-19：按用户要求修订执行交接

本次只读当前HEAD `d3084f76f713773e7a760c9f9ea4c06fed93aecb` 的源码、015原始日志与冻结输入，重新查阅vLLM官方optimization/reasoning文档，改写[016计划](../plan/016-budgeted-final-submission.md)并更新README。**未改求解代码、未开GPU、未训练、未提交、未读取C6答案。**

实际日志已确认FP8、CUDA Graph、prefix caching、chunked prefill、GDN CUDA decode均在用。旧`max_num_seqs=4`是人为配置；KV日志10.79×只是旧配置容量估算，不能称八路已测安全或能翻倍。更高并发属于此前提过但尚未执行的待办。

新的单会话安排为：

| 阶段 | 冻结设计 | 最大新增输出 |
| --- | --- | ---: |
| C：吞吐短测 | 服务上限8；同16请求、每次512，客户端C4/C8各一次，均滚动补位 | 16384 |
| R：恢复 | 全部15份空final旧轨迹，末16000字符，一次关闭thinking恢复、每次4096 | 61440 |
| B：P6预算比较 | 固定xhigh、seed4101；B8=6144思考+2048final，B16=14336思考+2048final，各6请求 | 147456 |
| 合计 | 59次主实验请求 | **225280** |

另为统一预热与预算小探针预留至多512输出token；含辅助请求至多225792。输入处理、首次JIT、加载、重算、异常都计入两小时；输出上限不是实际墙钟保证。按75token/s、旧658.3秒加载粗算约61分钟尚未计额外开销；35token/s时约118分钟，足见必须保留partial状态而非承诺全部跑完。

C8暂用标准明确为同工作量完成速度比`T4/T8≥1.20`且无OOM/持续重算，15%–20%边界默认C4，停止扩大并发搜索。先统一预热，记录prefix缓存与首次JIT影响，不让C4冷缓存/C8热缓存冒充提速。两份最长P6题面约7.7K/6.7K模板后token纳入清单；不是64K满上下文测试。

现有`chat_wave`的ThreadPoolExecutor可滚动补位，主要改动应是拆开服务/客户端并发、绕开driver外层固定chunk、完成即落盘，而非另建框架。`chat_once`只需参数化thinking预算和enable_thinking，复用usage/TTFT/deadline。

P6每预算臂只有6条，因此将共12条混合排入同一所选并发队列，以观察真实长生成C8。没有另一份完全相同长队列的C4重跑，不能把它写成严格长生成C4/C8速度对照。每臂EM@1、固定B8+B16的EM@2、各自独有命中和成本分别记录。8K不预设为部署胜者，16K首轮是否可承担要用新长生成吞吐与实际Q重新算。

程序反馈、P12/C6、32K追加、新训练和新提交均留到后续根据结果决定，不纳入本次两小时。实施者按修订计划交回吞吐、逐query正确性、恢复收益、总成本与准确产物位置。

## 8 2026-10-19：016最小代码已实施，GPU仍未运行

本节记录代码实现状态，与研究效果分离。**GPU实验、训练、提交均未运行；无新分数。**

### 实现产物

| 产物 | SHA256/状态 |
| --- | --- |
| `scripts/run_reasoning_probe.py` | `8970e7a7de10077db2b855ca5202ae60256ae8eb0af20ca32f58a2aa88c28d`；`experiments/016/crb/run_reasoning_probe.py`字节一致 |
| `experiments/016/crb/vdriver016.py` | `11830b3c42dc262c4192a9e49cd68563750d48740af522ad0c39c8c361fc6509` |
| `experiments/016/crb/recovery_tails/` | 15份各末16000字符+`manifest.json`；源为015 `P2`/`P6_extra` reasoning全文，本地截尾，GPU端不再截断即用 |
| `experiments/016/crb/notebook/arc2-016-crb.ipynb` | 582476字节；内嵌上述19文件，解包后SHA已核对一致 |
| `experiments/016/crb/notebook/kernel-metadata.json` | 沿用015挂载：比赛+wheelhouse+FP8权重，`NvidiaL4`，无网 |

改动对照计划第8节：服务`SERVICE_MAX_NUM_SEQS=8`与客户端并发分离；`chat_wave(specs, client_concurrency, on_slot_done)`整清单滚动、完成即落盘；`chat_once`新增顶层`thinking_token_budget`与`enable_thinking`，恢复请求不传effort/budget，C短测不加硬截断，B两臂分别6144/14336；016 driver只跑C→R→B，复用015安装与模板核对链路。

### 本地CPU核对（非GPU证据）

- `py_compile`通过；`--skip-serve`返回0，配置含服务8/客户端默认4。
- Mock SSE验证：xhigh+6144顶层发送且`chat_template_kwargs`含effort；恢复`enable_thinking=false`无budget/effort；短测无budget。
- Mock并发验证：`client_concurrency=4/8`分别限宽，`on_slot_done`逐条回调15次全回。
- 清单构造验证：C为六题字符降序两轮+前四共16；R为query→arm→seed固定15序且15份均可定位；B为奇题B8先/偶题B16先共12，`request_id`固定；主实验上限225280核对一致。
- 干跑验证：本地Python3.9按预期停在`env_incompatible_python_3_9`，prompt双SHA与冻结值一致，Notebook解包19文件SHA一致，`git diff --check`通过。

### 未运行与交接

Kaggle会话、两小时C/R/B、P12/C6、程序分支、新训练与提交全部未运行。C6答案未读。

用户执行前需重查实际授权身份余额与提交额度，按`session_seconds=7200`、110分钟停发新请求、115分钟收尾、120分钟进程组结束的上限推进；`kaggle kernels push -p experiments/016/crb/notebook`后取回`/kaggle/working/vprobe`，本地评分后再按计划第9节交回C/R/B/总账四表。

## 9 2026-10-19：跑前GPU利用率复核（未跑GPU）

本次只重读015冻结证据与016当前代码，不改冻结性能参数，无新GPU实验、无新分数。

### 015证据显示GPU远未触及容量上限

- 服务日志 `vllm-server.log` 共912行：KV容量707060 token（65K×10.79）；C8最坏约8×24K≈192K，占27%，安全。C8阶段KV无压力。
- 635个engine采样中Running恒等于客户端并发（1/2/4），Waiting>0仅2次且瞬时（3+1、2+2各一次），无持续排队。
- engine端生成吞吐约25（1路）/48（2路）/84–90（4路）token/s，接近线性；客户端聚合21–25/路。C8有明确上探空间，`S≥1.20`阈值合理。
- prefix命中：各题不同时0%，重复query时30.6%。016暖机（C8）+C4→C8同序已覆盖该收益。
- 不可调栈限制：L4（cc8.9）`Not enough SMs to use max_autotune_gemm`、PCIe下custom allreduce禁用、部分FP8形状缺调优配置，均属冻结wheel/硬件现状；按计划§17/§55/§101不新增开关，本轮不碰。

结论：016当前实现（滚动队列、暖机覆盖8路形状、B单波混排、阶段间仅日志落盘、110/115/120截止+partial）已是冻结约束内的最高利用；无需为利用率改动服务/批参数。

### 为省配额补的三个安全判定（不改变GPU行为）

1. 新增 `parse_running`：解析阶段内日志增量的`Running/Waiting`最大值与样本数，对应计划§82/§99的“真实八路同时生成”证据。
2. C决策：C8完整但`max_running<8`时判`C8_unsaturated_use_C4`并默认C4，排在S评估之前；`C_compare`增记`max_run4/max_run8/wait8`。
3. R空集守卫：15份全缺失时记`no_specs_all_missing`且`complete=false`，不再把空波次记成完成。

### 新产物与核对

| 产物 | SHA256/状态 |
| --- | --- |
| `experiments/016/crb/vdriver016.py` | `4703620271c64844dc8b953fcabc4d54ebf234445060a474b227ce1d9ddb3fc1` |
| `scripts/run_reasoning_probe.py` | 不变，`8970e7a7…`，副本一致 |
| `experiments/016/crb/notebook/arc2-016-crb.ipynb` | 586044字节，已用新driver重建 |

本地CPU核对：`py_compile`通过；`parse_running`在015全日志得`max4/wait2/635样本`、空输入得零、合成C8行得`max8`；Notebook解包19文件SHA与磁盘一致；`git diff --check`通过。

## 10 2026-10-19：已推送GPU，v1运行中（用户指令执行）

用户指令推送后执行，不再只是交接。推送记录见[`experiments/016/execution/push.json`](../../experiments/016/execution/push.json)。

- 推送：`kaggle kernels push -p experiments/016/crb/notebook`，约2026-10-19 03:24–03:25Z；实际slug为[`jiyujia/arc2-016-c-r-b-27b-reasoning`](https://www.kaggle.com/code/jiyujia/arc2-016-c-r-b-27b-reasoning)，推送后状态`RUNNING`。
- 事故：推送时metadata id写的是`arc2-016-crb`，与标题解析出的`arc2-016-c-r-b-27b-reasoning`不一致，Kaggle按标题建了新slug；第二次同包推送报409，未产生新版本。metadata id已纠正为实际slug，无代码改动；运行中的v1即为已核对的19文件包（driver`47036202…`），不重推打断。
- 配额门：推送前jiyujia剩11.65h（已用18.35/总额30，刷新10-24），满足2h×2.3=4.6h+6h储备=10.6h，余量仅约1h；ccjjllt未动（剩21.13h）。按账户差记账，不冒充逐设备账单。
- 下一步：等COMPLETE后取回`/kaggle/working/vprobe`，本地评分（答案不出境），按计划第9节交回C/R/B/总账四表。运行结论以取回产物为准，不以RUNNING代替成功。

## 11 2026-10-19：v1结果已取回并本地评分（P6开发集，非公榜分）

kernel `jiyujia/arc2-016-c-r-b-27b-reasoning` v1 COMPLETE，`elapsed_s=2271.4`（约38分钟），`complete_with_evidence`，无`first_error`。产物落盘`experiments/016/kaggle-v1/vprobe/`。评分用冻结本地答案（010评分文件+P2评分文件；P6已反复用于开发，不作未见验证、不预测公榜）。

机制结论先行：预算探针`effective_hint=true`（sent=8，`length`，reasoning仅30字符、final 251字符非空）——冻结wheel的顶层`thinking_token_budget`在GPU实际生效，无需升级引擎。B16六条全部`stop`且合法（6/6），相对015的14/18截断是机制成功；但正确解无新增（见表2）。

### 表1 C4/C8（同16请求×512，各8192 token）

| 项 | C4 | C8 |
| --- | --- | --- |
| 完成 | 16/16 | 16/16 |
| 整队墙钟 | 93.28秒 | 52.16秒 |
| 聚合 | 87.82 tok/s | 157.07 tok/s |
| S=T4/T8 | — | **1.788** |
| OOM/重算增量 | false/0 | false/0 |
| engine最大Running | 4（saturated） | **8（saturated）**，Waiting峰值0 |
| finish | 32/32 length（短decode负载，不计成绩） | 同左 |

全服务日志160个Running采样中96个为8路；Waiting>0全程仅6次瞬时（1/3/6/7各短暂），无持续排队。结束时prefix命中57.4%。决策：**暂用C8**（S≥1.20、无OOM/持续重算、真实八路已观察到）。KV容量704876 token（10.76×），C8最坏占用约27%，无压力。

### 表2 P6 B8/B16（固定xhigh/seed4101；B8=6144/8192，B16=14336/16384）

| query | B8（token/valid/正误） | B16（token/valid/正误） |
| --- | --- | --- |
| 291dc1e1_0 | 8192 length/非法/错 | 14521 stop/合法/错（15×6） |
| 4c416de3_0 | 6983 stop/非法/错 | 15177 stop/合法/错（19×22） |
| 58c02a16_0 | 6661 stop/合法/错 | 14853 stop/合法/错（16×16） |
| aa4ec2a5_0 | 7553 stop/合法/错 | 15745 stop/合法/错（26×27） |
| db118e2a_0 | 6599 stop/合法/**对** | 14791 stop/合法/**对** |
| fc4aaf52_0 | 6661 stop/合法/错 | 14853 stop/合法/错（16×16） |

EM@1：B8 1/6，B16 1/6；固定双槽[B8,B16] EM@2=**1/6**（仅db；291/4c缺合法B8时按规则重复B16，仍错）；B16独有0，B8独有0。B臂合计89940 token vs B8臂42649 token，差47291 token；12请求共享C8队列1074.4秒、聚合123.41 tok/s，12/12完成、无OOM/重算（preempt_total=0），所选并发在长生成上稳定。B阶段TTFT普遍0.2–1.7秒（前缀缓存命中），R阶段长prefill为5–40秒。

### 表3 R恢复（15份空final，末16000字符，一次final-only，seed3101）

15/15完成（351.74秒，36637 token，聚合104.16 tok/s），合法8/15：291×3（形状16×8/18×7/20×6各不相同，全错）、aa-high 26×27错、db-high对、fc×3（16×16各不相同，全错）；4c×3与58-low×2仍length耗尽。**新增exact仅1份**：db-high恢复网格与015 db-low已知网格逐格相同（冗余命中，无新query）。固定high→low2001→low2002去重取二，真实EM@2=**2/6**（58、db），与恢复前持平，无丢失、无新增query。计入015沉没成本后累计输出527390 token、推理7945.3秒。

### 表4 会话总账

权重/轮子/输入与冻结一致（wheel SHA`5cbeedeb…304a`，prompt双SHA匹配）；硬件4×L4（driver 580.178.04）、Python3.12.3、torch2.13.0+cu130、vLLM同wheel（import rc=0）；TP4/64K/服务8/0.9/默认KV。起止约03:24–04:03Z；安装+加载666.6秒，各阶段：暖机+探针27.9秒（512 token）、C4 93.28秒、C8 52.16秒、R 351.74秒、B 1074.4秒；未运行/中断项无（59主请求+17辅助全部完成）；新输出185610+512 token，远低于225792上限。jiyujia配额18.35→20.50已用小时（+2.15h≈墙钟×2），剩9.50h，reserved 0。

### 回答计划第9节四个问题

1. 后续用C4还是C8：**C8**（S=1.788且饱和健康）。
2. 8K/16K位置：B16把合法率从B8的4/6提到6/6，但无独立新增；16K首轮全量不可负担第二遍（Q240×约15K token在123 tok/s下仅生成约8.1小时，未计prefill/恢复/落盘）。**不宣布首轮16K**；候选为8K首轮+选择性16K/恢复。
3. final恢复：保留为廉价补充（35K token换1冗余exact+完成率信息），7/8合法恢复是错误网格，不能当主解题器；是否对失败query触发恢复，留待程序分支对照后再定。
4. 下一轮：按计划第6.3节，本轮是“完整率上升但只增加错误网格”（B16新增5个合法错解、R新增7个合法错解，正确query零新增）——**下一轮改为完整程序+一次demo反馈，不在本轮继续堆token**；P12确认暂缓。

## 12 2026-10-19：用户确认016四项判断（含两处口径修正）

用户逐项确认，修正已并入后续计划口径：

| 决策 | 用户判断 |
| --- | --- |
| 后续使用C8 | 同意。后续预算按**长生成实测123.41 tok/s**计，不用短测157.07。 |
| 暂不16K全量首轮 | 同意，但主因是**无新增正确解**（B16成本为B8的2.11倍，两臂均1/6）。8.1小时是外推；按10.5小时生成预算仍剩约2.4小时——不够全量第二遍，但可选择性补解。 |
| 恢复保留为廉价补充 | 保留实现、按需触发。不默认每个失败都恢复（15次仅1冗余exact，EM@2持平2/6）。 |
| 程序+一次demo反馈、P12暂缓 | 同意，为下一优先项。继续扩同策略验证规模收益有限。 |

下一轮（017计划）：同一场会话内先P2确认完整程序、再P6，对照“共享P0＋一次真实demo反馈”与“同预算独立重启”，看两槽新增；程序不完整则不得宣称测过反馈。同时补齐全量提交入口。
