# 009 — 有效 episode：CPU研究结果

日期：2026-10-01。对应[009计划](../plan/009-efficient-episodes.md)。本轮在008后续CPU工作基础上完成新的数据与表示研究；未使用GPU、未运行模型、未下载公共evaluation query答案、未训练或提交。用户已有的008未提交修改保持原样。

## 1. 决策更新

固定Qwen3.8-27B，新增三个有CPU证据的研究候选：

1. **F1紧凑网格格式**：保留完整词表与权重，去掉格间空格。训练侧真实token约减少45%，应在真正裁词表G3之前做小规模GPU比较。
2. **最终消费时的任务/base公平采样**：相同长度配比和近似token预算能覆盖更多任务，不能只保证原始池覆盖。
3. **基于规则见证选择demo**：用少量更有区分能力的示例替换冗余示例，而不是机械扩增base数量。

这些是数据/表示层收益，不是模型提分。008的base/G1代表性配对评价仍最先进行，原格式参照不变。

## 2. F：保留原词表就能显著缩短序列

运行 `python3 experiments/009/analyze_serialization.py`。本地macOS arm64、Python3.9.6、tokenizer-only，无PyTorch或模型权重；最终运行17.65秒。固定tokenizer revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`，套用与当前训练一致的chat template、thinking=False、答案加一个EOS。输入文件和tokenizer SHA记录于 `experiments/009/serialization-analysis.json`。

比较原空格格式、紧凑数字、游程格式。示例：一行`0 0 2 2 2`分别写成`00222`和`0:2 2:3`；均能还原同一格子序列。当前tokenizer将`0123456789`编码为10个单数字token，30个连续0也是30token；没有为了压缩而更改tokenizer。实际逐格式复核27,426次网格还原，均与原网格相同。初次手动运行因demo编号从1而不是现有0开始被参考比较发现，修正后才统计以下结果。

### 训练侧完整episode（包括答案）

| 数据与格式 | 完整token总数 | 至少一条完整episode≤4K的task | 完整episode最大token |
| --- | ---: | ---: | ---: |
| 官方训练侧892 task、全部962 test query：空格 | 2,864,908 | 694/892 | 18,124 |
| 同一批：紧凑数字 | 1,576,269（减少44.98%） | 837/892 | 9,422 |
| 新2240 base、448 family：空格 | 7,189,298 | 354/448 | 14,508 |
| 同一批：紧凑数字 | 3,932,530（减少45.30%） | 427/448 | 7,546 |

这里官方侧是所有test query、没有LOO或增强，因此不能拿694与008的D2混合索引742直接比较。两组表示对照各自保持全部原始网格不变，没有删demo或答案。紧凑格式在8K下覆盖官方891/892 task、新生成448/448 family；仍不能声称全部官方任务已装入8K。

### 公共evaluation题面（不含query答案）

实际文件含120task、**172query**（71题单query、46题双query、3题三query），E24为24task/36query。仅统计题面和已知demo；没有读取query金标准。

| 范围 | 空格prompt token | 紧凑prompt token | ≤4K的query |
| --- | ---: | ---: | --- |
| 全120task / 172query | 822,850 | 440,879，减少46.42% | 81/172 → 155/172 |
| E24 / 36query | 164,934 | 88,766，减少46.18% | 18/36 → 32/36 |

原格式全120题的prompt最小1130、中位4252、最大16324；**没有一个query的prompt≤1024**。G1所有完整训练序列≤1024。这是实际长度分布差异，不证明长度差异是全部错误的原因，但支持把完整长episode覆盖置于训练优先级前列。

F1仍调用原大lm_head，**不节省G3所裁掉的词表权重，不等于已获得45%墙钟加速**。它改变了模型看到的序列，需要GPU核实准确率与适配成本。其意义是可以把“去空格压缩”和“裁词表头部”拆成两项研究。

RLE没有统一胜出：官方完整token总数1,881,172，高于紧凑数字，最大完整序列25,868，反而高于原格式；公共eval最大prompt23,329。虽然E24稀疏题的RLE总prompt更短，也不据此选择整轮格式。当前优先F1，不同时研究更多编码。

## 3. S：800条里实际看到哪些任务

本轮读取现有3000行D2索引和008真实token缓存，没有把尚未物化的2240新base算进旧池。训练器需要`messages/target_grid_text`，D2索引只是元数据；raw新base也不是现成训练行。首行调用编码入口得到`bad_messages`，明确区分“素材准备完成”和“可以直接传给训练器”。

采样分析使用本地arm64 CPU、Python3.9.6，脚本与输入SHA记录在JSON；该分析墙钟未单独记录，不作为训练速度实测。

旧索引含53条仅一个剩余demo的LOO，与当前`min_loo_demos=2`不符。本轮另建2947行统计子集，不修改原池。其≤4K实际可达730/892task；55个无V2的非ARC1任务可达35个。剩余20个在**此索引、此上限**下不可达，不说明穷尽所有query/LOO后它们仍必须超4K。

seed20260930，同一2947行子集；前两行800样本固定长度配额160/320/320，第三行按同full-token上限截取候选前缀：

| 选择方式 | 消费行 | task | 独立base | 可达保护题 | full token | 答案+EOS token |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 当前长度桶随机 | 800 | 446 | 669 | 14/35 | 1,565,060 | 179,562 |
| 来源/task/base公平＋保护 | 800 | 601 | 725 | 35/35 | 1,580,362 | 177,190 |
| 公平＋保护，同full-token上限 | 792 | 599 | 721 | 35/35 | 1,563,562 | 175,214 |

固定800行时，task增加155、full token只增加0.98%；重复base的增强行由131降至75。原构造顺序不能保证这些性质，因为训练器在长度桶内按episode重新打乱。800行候选在每桶内约束40%官方/60%生成，再优先已选次数少的task和base，并照顾无生成器非ARC1任务；不是全局最优求解器。792行前缀的实际长度配额为159/317/316，官方/生成为318/474，不再宣称精确160/320/320或40/60。

预定20个连续seed20260930..20260949，随机方法平均454.7task，公平＋保护平均608.5task，35个可达保护题均保留。没有按最漂亮seed选择效果。完整选中ID、逐行token和方法保存于 `experiments/009/sampling-analysis.json`。

这不能证明广覆盖比重复练习更有利于27B。相同full token也不等于相同时间：候选的ΣL²代理仍约高1.89%，混合GDN、attention、跨卡搬运和checkpointing成本需要GPU实测。新800行比G1旧800行的full token约多3倍，不能照抄旧更新用时。

发现的接线问题作为交接保留：`_make_episode`仍拒绝部分同输出LOO，与上层标记保留政策不一致；统计辅助函数混淆full/supervised token以及task/base计数。本轮使用独立口径，不改变用户正在修改的脚本，也不把本轮模拟当成已验证的GPU数据入口。

## 4. Q：独立随机不等于规则不一致

数据：008新2240base、448族、8960个pair；ARC-GEN commit `a15cbdb44c776610aeeb9f487a06af875d3d0878`。本地CPU全池统计加六族初次复算约0.46秒；扩展11族220pair逐输入重建后，220/220与保存输出一致。人工读源码用时未单独计量。没有调用通用模型或读取留出query答案。

全池观察到同族相同输入却不同输出0例，唯一族＋输入8884种。它只排除本批可见的直接矛盾，不是448族全部正确或可学的证明。

对008原6存疑族的判断应修正：

| 家族 | 本轮源码与数据证据 | 研究处理 |
| --- | --- | --- |
| `0b17323b` | spacing/start由输入点直接可读，20pair均符合延伸规则；仅5种唯一输入 | 不因参数独立删掉；3个query重复demo需要重配 |
| `2072aba6` | 是逐格局部模板替换；随机分支只激活3×3前5个位置，后4位恒0 | 标签一致，问题是生成支持范围窄 |
| `00dbd492` | 厚度1/2/3对应输出色8/4/3；instance102的demo未展示厚度2 | 有具体任意映射欠定见证，优先补一个区分性demo |
| `3b4c2228` | 计数规则正确；小样本只观察计数1/2，没有3 | 改善计数和干扰覆盖，不能因输出重复而删 |
| `8597cfd7` | 5个base的demo实际都展示了两种胜者 | 撤回“两个分支未覆盖”的概括；无充分证据重配 |
| `c8b7cc0f` | query前景色来自自身输入，3/5 query颜色未出现在demo仍可复制 | 不以新颜色为理由删样本；关注计数与框内外关系 |

进一步得到10条有具体简单替代规则见证的episode：源标签正确，但query使用的任意颜色映射或交互未在demo体现。另有7条query输入直接重复demo；42条仅输出重复。两个后者现象不能混为泄漏。逐行清单、源码链接和同族补配方案见 `experiments/009/episode-semantics.json`。

这里“欠定”只相对于已说明的合理短规则类；模型仍可能依靠先验知道原规则。不能据此宣称题绝对不可解，更不能把有限示例总能拟合无穷函数的事实当作删除全池的理由。其余437族尚未逐族审阅。

### 已实际完成一个demo修复试件

保留`00dbd492 instance102`的query和target、保留3个demo，把全空白demo替换为同族instance104/demo0。选择依据是query输入可见的厚度2，未用query答案选择。

在预声明的336种“5/7/9框→三个互异填充色”短规则中，只用demo筛选：

| 观察 | 原episode | 修复episode |
| --- | ---: | ---: |
| demonstration数量 | 3 | 3 |
| demonstration输入＋输出格子数 | 388 | 418 |
| 仍拟合全部demo的候选规则 | 6 | 1 |
| 对原query给出的不同答案 | 6 | 1 |

候选消歧统计耗时约0.12秒。这是**在明确假设类内减少歧义**，不是模型exact-match提高。修复没有增加独立pair，只重新组合已有训练pair。

已保存 `experiments/009/episode-repair-pilot.jsonl` 的原/修复配对、完整lineage与不变性记录；`demo-replacement-analysis.json`保存规则集合。原2240base文件没有被改写。10条都已有同族补配建议，本轮只物化这一例，未声称完成全池修复或语义认证。

## 5. 前沿研究带来的方法与限制

查阅日期均为2026-10-01。采用一手论文与作者源码，不把外部成绩换算成本项目分数。

- [ARC-TGI论文](https://arxiv.org/html/2603.05099v1)区分任务级变量与单个网格的可观察变化，并研究episode约束；其Qwen3-8B微调没有展示稳定泛化收益，不能用“人审数据”推断有效。具体[0b17323b源码](https://github.com/Omega-Reasoning/ARC-Task-Generators-Inventory/blob/a614132ff5b2cb3628063d541e7cbd74a2cd2edb/Generators/ARC-AGI-2/Train/task0b17323b.py)主动让间隔变化并从input恢复间隔。借用这种变量区分，不整包导入含Eval目录的数据。
- [CausalARC](https://arxiv.org/html/2509.03636v1)，2025-09，使用显式因果规则构造干预与对比，静态50个自建任务不是ARC2官方成绩；反事实示例收益并不稳定。适合借作受控诊断：一次只改一个关键因素并重新计算正确输出。暂不导入许可未核实资产或扩成大规模SFT。
- [ARGA](https://arxiv.org/html/2210.09880v2)的对象图/约束搜索可用于CPU候选与分歧审计，但57/160来自选择过的ARC1子集，非ARC2。[作者代码](https://github.com/khalil-research/ARGA-AAAI23)默认每题1800秒，原读取入口要求test output，不适合直接移植为提交。只保留有界符号互补的参考，未运行ARGA。
- [ARC-GEN固定源码](https://github.com/google/ARC-GEN/tree/a15cbdb44c776610aeeb9f487a06af875d3d0878)的任务`validate()`返回固定参数的示例；[CLI](https://github.com/google/ARC-GEN/blob/a15cbdb44c776610aeeb9f487a06af875d3d0878/arc_gen.py#L25-L37)再比对原题。008“60/60可构造”不是对任意随机pair的正确性证明。本轮独立输入→输出重建解决的是这个更窄的问题。

这些证据支持“让每个episode提供更多可推断信息”，不支持继续盲目扩池或重建大框架。新的数据质量指标最终必须与GPU上的真实新增/丢失解题一起报告。

## 6. 比赛约束带来的新预算提醒

本轮重新读取Kaggle元数据及官方data/evaluation/code要求，记录于 `experiments/009/competition-context.json`；队伍数2295，截止仍为UTC11月2日23:59。没有重新查询公榜，008的榜单仍只是其查询时快照。

官方明确240个隐藏**task**、每个test output两次尝试、完整Notebook12小时。[数据说明](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data) · [代码要求](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)。本地公开eval实际120task/172query，进一步证明不能拿240task当240query。

条件算术：如果隐藏集也恰好按172/120的query倍率，240task就有344query；按G1旧20个query两候选均值137.44秒/query，生成已约13.13小时，尚未加载和TTT。**344不是隐藏query数预测，旧题速度也不是长题速度预测**；它只是展示多query会怎样吃掉008按240query估算的余量。提交预算应从实际题面读取query数，TTT按task共享、生成按query计费，不能乘以四张卡假装四个worker。

全public eval prompt中位4252也说明旧短题时速不足以支持正式长题预算。F1的长度压缩与候选效率因此都值得研究，但只有GPU实测才能替换保守预算。

## 7. 本轮交付与状态

完成：009计划/进展、表示分析脚本及真实token统计、固定预算采样对照、全池直接矛盾/重复探针、11族语义复算、10条欠定见证及一条实际修复配对、有限假设消歧计量、比赛约束复核。JSON保存来源与关键配置；研究读取与人工源码推导不计作GPU实验。

语义审计的四份手动执行源码快照保存在 `experiments/009/analysis-source/`，只将临时脚本依赖路径改为同目录路径，原执行稿及归档稿SHA均有记录。它们是本轮CPU分析依据，不是自动测试套件或竞赛求解器。

未完成且未声称完成：F1模型推理/训练、全新合并池的采样结果、10条全部修复、全部448族规则验证、CPU通用符号求解器、G2/TTT/G3/Unsloth实验、任何新增公榜/私榜成绩。也没有修改008已有评价清单或把公共eval query答案引入研究。

下一轮优先级：先按008完成原格式E0；据配对结果选初始化，再用相同数据的小对照研究F1及任务/base公平采样；示例消歧作为小数据质量分支。没有理由在GPU刷新前空等，也没有依据现在宣布这些CPU收益会带来金牌。

## 8. 2026-10-01 CPU衔接补记（§4交接表实现，不含GPU）

- `arc_episode_data.py`（+562行，`py_compile`+`--help`+dry-run已验）：`FORMATTER_F0/F1`、`serialize_grid/parse_grid_text`（F1逐字符成整数格，含空格返回None；已验`002`↔`[[0,0,2]]`双向还原）、`build_messages(...,formatter)`（F1换system+紧凑串，旧两参调用兼容走F0）；`_make_episode`默认`drop_suspect=False`标记保留（与上层一致），`build/build_g2`显式传True保复现；`materialize_episodes`（`--materialize --index/--newbases/--repair --formatter --materialize-tasks/--materialize-max-rows`）将D2索引、新base（按planned ident/rot90展开）、修复配对物化为训练器完整episode格式，53条`loo n_demos<2`旧行在新物化中排除计数（原文件只读），拒写已存在文件与`experiments/007`内。
- `train_grid_sft.py`（`--sample-policy {legacy,fair_task_base}`，默认legacy行为不变；fair须联用`--length-mixture`，配额复用`apportion_mixture_targets`，桶内来源→task→base已选少者优先、无生成器非ARC1用official-only代理，40/60仅软目标并报告实际；`bucket_consumption`扩展`sources/independent_bases_consumed/tasks_covered`不删旧键；`--resume`精确恢复与互斥语义不动）。`--length-mixture`桶界与D2同界。
- `solve_grid.py`（`--formatter {auto,f0_space,f1_compact}`，默认auto按adapter_config→run_config→config.json读formatter字段，无字段回退F0并记`formatter_fallback=true`，未知值`UsageError`；F1按字符解析，失配`FormatterMismatchError`记`parse=FAIL+parse_error`，不回退不混用；对外仍标准submission JSON网格）。
- 缺口诚实记录于`code_gaps_followup.json`：fair消费与长度混合标“已实现（CPU，待GPU验证）”；`f1-train-consume`未实现（train侧默认仍空格，F1比较前需显式formatter透传）；GDN/DFS/旋转/TTT/程序仍未实现。
- 合规：只改3脚本+followup；007池sha不变；e0清单未动；raw-eval仅题面；无测试文件/CI；无GPU/推理/提交。GPU刷新后仍先跑008原格式E0，再小对照验F1/公平采样/消歧。
