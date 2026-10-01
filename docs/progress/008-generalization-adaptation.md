# 008 — 泛化证据、数据覆盖与逐题适配：研究结果

日期：2026-10-01。对应计划：[008-generalization-adaptation.md](../plan/008-generalization-adaptation.md)。本轮完成文献调研、比赛实时查询、代码阅读和现有产物的 CPU 统计；没有启动 GPU、训练、模型推理或竞赛提交。

## 1. 研究决定

继续使用用户选定的 **Qwen/Qwen3.8-27B**。下一方向是 **有代表性的配对评价 → 修正数据与长度覆盖的 G2 → 小规模 TTT / 多视角候选对照**。保留完整词表，让同一个底座的程序生成能力成为后续互补选项。GDN、G3 和 Unsloth 不与第一次 G2 数据实验同时改变。

当前没有证据证明 G1 提高了 ARC2 公共 evaluation 或比赛成绩，也没有证据证明这个模型已到能力上限。不能把“完成 G2/G3”本身当作研究目标；每轮应回答一个会改变预算分配的问题。

本轮两项关键更新：

1. 已评的 25 个留出 query 中，**24 个来自旧 ARC1，只有 1 个不在 ARC1**。当前评价几乎没有覆盖 ARC2 新增任务，必须补充。
2. 将 007 中“高分 checkpoint 才研究 TTT”的顺序调整为：**允许低预算、可比较的 TTT 实验，部署仍须证明收益与时间可行性**。6/20 不能独立作为否定 TTT 的依据。

## 2. G1 实际训练量与能够支持的结论

原始来源：`experiments/007/g1/kaggle-v1/007-g1/{SUMMARY.json,train_log.jsonl,zero_step.json,run_config.json}`，三次独立评价的 `solve.json`。本轮统计保存在 `experiments/008/local-audit.json`。

| 项目 | 复核结果 | 含义 |
| --- | --- | --- |
| 更新与训练量 | 200 次更新、800 episodes、264 个 task_id | 10K 是池大小，不能说完成了 10K 样本训练 |
| 长度筛选 | 10K 中 2514 条完整序列 ≤1024；没有加入 1024–2048 桶 | `overflow` 是短池不足时补足，并非长度课程 |
| 实际 token | prompt 458,556；target 含 EOS 62,808；合计 521,364 | 监督规模仍小，低平均 loss 不等于学会新任务 |
| 完整长度 | 中位数 648、最大 1024 | 训练集中于短 episode |
| 实际 query 类型 | LOO 426、官方 test 374 | G1 确实含示例留一监督 |
| 实际几何增强 | ident 291、rot90 276、rot180 231、flipud 2 | 不是均匀覆盖全部几何变换 |
| 最后学习率 | 第 200 次为 0 | 保留“200 次更新”的日志口径；最后一步并不继续改变权重 |
| 独立加载 | adapter 可保存、恢复用于推理；诊断 6/8→7/8 | 证明实现闭环及小诊断增益，不证明 ARC2 泛化提分 |

训练日志 `seconds` 合计 4567.7752 秒，约 1.269 小时；Notebook 日志时钟约 5045 秒。007 记录的训练前后**账户配额差**为 6.166 小时。它与日志墙钟按官方 2 倍扣额并不完全对应，本轮没有证据解释差额。以后同时记录 Notebook 完整墙钟、纯训练时间、账户配额差及并行会话，不能用 6.17 小时直接当单次训练墙钟或稳定扣额倍率。

`family_id=task_id` 只证明任务 ID 隔离；没有完成跨题语义规则聚类，也不知道底座预训练是否看过这些公开任务。因此准确名称是“本项目未用于微调的任务留出”，不是已证明的未见规则或无污染评估。

## 3. 新增题源核对：现有留出结果偏向旧 ARC1

用官方 [ARC1 仓库](https://github.com/fchollet/ARC-AGI) 的完整文件树对照任务 ID，revision `399030444e0ab0cc8b4e199870fb20b863846f34`；training、evaluation 各 400 个 ID。清单随审计 JSON 归档。这个对照识别公开题的历史来源，**不证明底座记忆或污染**。

| 集合 | 旧 ARC1 training | 旧 ARC1 evaluation | 不在 ARC1 两个公开集合 |
| --- | ---: | ---: | ---: |
| 当前官方 training，1000 task | 391 | 376 | 233 |
| 项目 holdout，100 task | 39 | 35 | 26 |
| G1 实际消费，800 episodes | 440 | 223 | 137 |
| G1 实际涉及，264 task | 146 | 68 | 50 |
| 960 上限留出，5 query | 0 | 5，命中 0 | 0 |
| 2048 上限留出，20 query | 15，命中 6 | 4，命中 0 | 1，命中 0 |

唯一不在 ARC1 的已评留出题是 `342dd610`。2048 的 6 个命中仍是 `1c786137`、`22eb0ac0`、`3c9b0459`、`44f52bb0` 两个 query、`50cb2852`；均属于旧 ARC1 training。

这里仍按两个输出上限分别计分。0/5 和 6/20 不合成统一分数，也不能推出“新 ARC2 的准确率为 0%”：新增任务只评了一题。字典序前 24 task 的样本构成不能代表官方 120 题公共 evaluation。

另一个需要避免的混淆：**ARC-GEN V2 是生成器版本标签，不等于“所有规则都首次出现在 ARC2”**。本项目 G2 的 448 个可用 V2 family 中，297 个来自旧 ARC1 evaluation，151 个不在 ARC1。003 关于 V2 全部对应当前 ARC2 training 的结论仍成立；但该结论不能用来证明它们全部是新规则。

## 4. 错误、第二候选与速度告诉我们什么

仅对输出上限 2048 的 20 个 query 汇总：

| 项目 | 实测结果 |
| --- | --- |
| 合法、完整、尺寸正确 | 两候选合计 40/40；没有截断 |
| greedy / sample 单独命中 | 6/20 / 4/20 |
| 两候选任一命中 | 6/20，第二候选新增 0 |
| 两候选文本完全相同 | 9/20；题内去重后合计 31 个候选 |
| greedy 生成总时长 | 1374.7179 秒 |
| sample 生成总时长 | 1374.0528 秒，未新增命中 |
| greedy tokens / 时间 | 5828 / 1374.7179，约 4.24 tokens/s，包含 prefill |
| 单次 greedy 用时 | 中位数 61.79 秒；最大 148.14 秒 |

当前瓶颈主要体现为格子内容和规则执行错误，不能以“合法率不足”为首要理由推进 G3。尺寸正确不证明内部规则正确。`2204b7a8` 只错 1 格；其余 greedy 错题错 7–122 格。逐格正确率 88.78% 只适合定位错误，不能代替 exact-match，也不能据此盲改一格或按金标准指导推理。

按这些 query 的均值机械放大：单 greedy 约 68.74 秒/query，240 query 约 4.58 小时；加当前 sample 约 9.16 小时。这只是**同速算术**，不是隐藏集预测：正式集合是 240 task、query 数可能更大，长题分布也不同，还要加加载、TTT、重评分等。因此应先比较“同视角 sample”与“第二个变换视角”的每秒新增命中，再投入大量搜索。

## 5. G2 数据池尚不足以直接决定训练配方

G2 原文件和 G1 文件未修改。30K 池已生成、未训练。

| 来源 | 行数 | 独立 base episode 键 | 任务数 | 当前问题 |
| --- | ---: | ---: | ---: | --- |
| 官方 | 12000 | 892 | 892 | 每题仅 test[0]，无 LOO、无其他 query；按增强前缀截到限额 |
| ARC-GEN V2 | 18000 | 750 | 448 | 每个 base 固定展开 24 种增强；302 族有 2 个 base、146 族仅 1 个 |

ID 没有重复，不等于内容没有重复：官方 `messages+target` 唯一内容为 11,873 条，生成侧为 17,800 条。它们多为对称或增强等价，不能直接称为错误数据；应把新实例数与增强副本数分开记录。

全池按历史来源计：旧 ARC1 training 4667 行、旧 evaluation 16,297 行、不在 ARC1 9036 行。训练划分中的 206 个“非 ARC1”任务中，只有 151 个有当前可用 V2 生成器，余下 **55 个需要靠官方完整 episode/LOO 保持覆盖**。先不让生成器缺失变成任务永久缺失。

下一版数据优先改为：来源 → task → 独立实例 → query → 随机一致增强的采样顺序；固定的 40/60 来源比例只是起点。让新增实例先于重复同一 base 的 24 个增强。对官方所有 test query 和合法 LOO 明确分配采样机会，不能期待 round-robin 截断自动公平。

当前把“目标网格恰好等于某个 demo 输出”视为 LOO/生成 episode 泄漏而丢弃，过于宽泛。不同输入产生相同答案本来可以合法。应检查目标 pair 是否被同时作为 demo、同一输入是否对应矛盾输出；输出值相同单独标记、抽样审查，不一律删除。

生成器输出矩形、数值合法也不保证整个 episode 能从示例推断。当前四次独立 `generate()` 抽样没有验证示例是否暴露了 query 需要的变化。先人工检查小批生成 episode 的共享变量、颜色语义、退化与歧义，再决定是否需要生成器专属约束。ARC-GEN 的 `validate()` 未运行；也不能没读其语义就当它是任意新样本的规则 oracle。

## 6. 继续研究前必须理解的代码状态

`scripts/train_grid_sft.py` 的参数不是现成的 G1→G2 阶段切换实现：

- `--resume` 在 1515–1547 行按旧 `pool_episode_ids` 恢复池和 cursor，不重新构建长度桶。G2 不含 G1 的 LOO episode，直接换文件会缺旧样本。
- 1605–1610 行加载旧 adapter；1674–1686 行拒绝目标模块变化。`--resume --extra-targets ...` 不会给 G1 加上 GDN LoRA。
- 不传 `--resume` 会创建新的 LoRA，从原底座开始，不会自动继承 G1。
- 原 `--optimizer-updates 200` 配合完成 200 步的状态会跳过训练。调大总步数只会继续旧池与游标。
- 定期保存覆盖同一 adapter 目录，不能假定已有第 25/50/100 步独立 checkpoint。

因此交接需要一个“仅继承 adapter 权重、开始新数据阶段”的入口，与原配置中断恢复分开。扩 GDN 时保留旧模块、新 B 矩阵置零并重建 optimizer。具体实施见计划；本轮没有改训练器。

## 7. 前沿研究：哪些证据值得迁移

以下都是外部资料，不是本项目结果；查阅日期均为 2026-10-01。

| 来源与发表时间 | 核实到的证据 | 本项目取舍 |
| --- | --- | --- |
| [NVARC 作者 writeup](https://www.kaggle.com/c/arc-prize-2025/writeups/nvarc)，2025-12 | 最终训练数据集含约325万增强样本、104,539个puzzle；作者报告4B全参SFT使用32×H100、27小时，实际消费样本数本轮未核实。结合逐题LoRA、增强搜索、排序，2025 private约24% | 迁移离线训练与在线适配分工；不能把其成功简化为裁词表或800条SFT，也不能照搬算力 |
| [Matter of Perspective / PoE](https://arxiv.org/html/2505.07859v1)，2025-05 | ARC1 消融把 TTT、增强候选、PoE、DFS 分开；8B 从无适配18.3到完整组合71.6 | 支持分解实验；这些百分比不能外推到 ARC2，低起点也不足以排除 TTT |
| [ARC-TGI](https://arxiv.org/html/2603.05099v1)，2026-03 | 强調 episode 级约束、可执行变换与示例消歧；含 ARC1/2 的 train/eval 来源 | 借用数据质量思想，暂不整包混入，先排除评价来源；不宣称已提供新的干净规则全集 |
| [ABPR](https://arxiv.org/html/2603.20334v1)，2026-03 | Prolog 执行轨迹辅助修复；Gemini3Flash 作者自报 ARC2 public pass@2 56.67%；8线程、每线程最多10次修复；Qwen3-Max改善很小 | 只借执行反馈定位错误，用同一27B做小规模程序互补。Python试验是我们的设计，不是论文原实现 |
| [Allspark](https://arxiv.org/html/2609.32913v1)，2026-09-26 | ARC2 training 拆成904/96；96开发题mean pass@1，Inkling78.1→81.6；276B教师和975B学生，长CoT与远程服务 | 不是竞赛或120题public eval；没有27B ARC验证，双模型成本不适合直接迁移 |
| [Thinking with Looped Flows](https://arxiv.org/html/2609.11801v1)，2026-09-10 | TRM系7M，ARC2 pass@2 12.2±1.9；ARC1另58.8%；机构GPU上的ARC训练1–2天 | 新架构不等于现成27B适配方案，当前不从零重训 |
| [Multi-Perspective Transformers](https://arxiv.org/html/2605.01154v1)，2026-05 | 摘要21.7%是预训练TinyLM的公开eval；表2的TTT/PoE配置结果和合法率均为0 | 不能用摘要证明其TTT有效，也不能把输出失效一概解释为TTT能力不行 |

两项高分声称未用于决策：[Multi-Stage Rule-Chaining](https://arxiv.org/html/2609.10654v1) 的“230/240、95.4%”算术和隐藏标签来源未形成可核实提交证据；[ARCANA](https://arxiv.org/html/2607.09059v1) 对 NVARC 架构及离线比较设置的描述与作者材料不一致。本轮不据此推荐重建新框架。003 已指出的 LLM＋遗传搜索论文题号问题仍保留。

研究推断：最可能值得本项目花费下一批算力的，是完整 episode 与真实实例覆盖、逐题适配以及候选互补；目前没有一手证据证明立即更换模型、堆随机采样或移植大型闭源 agent 更划算。

## 8. 实时比赛情况与资源边界

Kaggle MCP 查询记录时间 `2026-10-01T12:56:13Z`，北京时间 20:56。API 本身未返回 currentTime；记录来自查询时钟。原始选择性响应归档到 `experiments/008/competition-evidence.json`。以下不是网页缓存排名。

| 项目 | 查询结果 |
| --- | --- |
| 队伍数 | 2289 |
| 公榜前四 | 83.06 / 79.72 / 75.42 / 55.14 |
| 第8个条目 | 36.94 |
| 第14、15个条目 | 均显示33.89 |
| 金牌名额规则计算 | 10 + floor(2289×0.002) = 14；最终以实际资格、队数和排名为准 |
| 排名奖金 | 前8名合计275,000美元；金牌不自动等于奖金 |
| 截止 | UTC 11月2日23:59；北京时间11月3日07:59 |
| 报名/合队截止 | 北京时间10月27日07:59；最多5人 |
| 提交 | 每天1次，最终最多选2份；Notebook无网、最多12小时 |

来源：[榜单](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)、[奖牌规则](https://www.kaggle.com/progression/competitions)、[比赛规则](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)、[时间表](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/timeline)。榜单 API 没有显式 rank/medal 和未舍入分数字段，同分规则优先较早提交。33.89 和36.94均不是最终私榜门槛；不能把 G1 的30%与它们相减。

官方数据与评分：公开 training 1000 task，本地实数1076 query；公共 evaluation 120 task。正式 rerun 为240隐藏task，可能每题多个query，每个query取两个答案的任一 exact-match。不能按恰好240个输出做硬预算，也不能把占位test运行当私榜结果。[数据页](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)、[官方数据结构](https://arcprize.org/arc-agi/2)。

实时配额 `108000 - 105165.421302300 = 2834.5786977` 秒，即 **0.7874配额小时**，reserved=0。按当前官方L4两倍消耗，最多约 **23.6分钟墙钟**；不是47分钟。刷新为 **10月3日08:00北京时间**。每周30配额小时理论对应15小时L4墙钟，实际以账户前后差和平台政策为准。[升级算力说明](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators)。

奖金另有研究/writeup奖275,000美元，Kaggle称Innovation、主站称Grand Prize，按六项研究标准评审，并要求期限内开源材料。它不是榜首自动附加奖金。85% bonus分法和作者代码许可在[Kaggle规则](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)与[主站](https://arcprize.org/competitions/2026/arc-agi-2)存在措辞差异，本轮不自行解释消除冲突。到最终交付按具体产物核对当时规则。当前研究应保留数据、权重、完整方法和消融证据；不为此建设CI流程。

## 9. 实际产物与尚未完成

本轮新增008计划、008研究结果、两份证据JSON，更新README和007的后续研究指向。没有修改G1/G2原数据、adapter或算法脚本，没有新模型成绩。

仍未运行：配对底座留出评价、代表性公共evaluation、G2训练、GDN新增模块训练、G3真实转换和训练、TTT、TTA/PoE新对照、程序互补、Unsloth对照、使用本轮权重的竞赛提交。历史30.14保持为未修改公开Notebook的复制试提交。

## 10. 2026-10-01 CPU执行补记（E0清单+D2索引+代码缺口，不含GPU）

本节记录008计划§3/§4/§5-6交接后的本地CPU执行；没有启动GPU训练/推理/提交，没有新模型成绩，没有修改G1/G2原池与adapter权重。

- E0清单（`experiments/008/e0_*.json`，seed统一`20260808`，输入侧分层）：
  - `e0_h25_bridge.json`：25条/24task重建，口径不合并（cap960 0/5，cap2048 greedy 6/sample 4/oracle 6），seed=null注明重建无抽样。
  - `e0_hnew_12.json`：26选12（S/M/L各4）+reserve14，只用输入格元数据；`342dd610`命中选中组并标`seen_before=true`，其余false；不得加入训练/生成器。
  - `e0_e24_24.json`+`e0_e24_STATUS.json`：官方120题题面（`raw-eval/arc-agi_evaluation_challenges.json`，sha `e7c62a4b…`）按输入格分层S/M/L各40抽8，得24task/36query+reserve96；solutions从未下载查看；本轮`NOT SCORED`。
- D2索引（`experiments/008/data/`，007原池sha不变）：
  - `d2_manifest.json`（seed 20260930，892/100/8继承，52排除，来源→task→base→query→增强，40/60起点，suspect标记不删）。
  - `d2_sample_index.jsonl` 3000行（官方1200：turn0全892+turn1前308；生成1800：复用G2已保留750base，pass1 ident750+pass2 rot90750+pass3 rot180前300，每base至多3行；未新造格子）。
  - `d2_coverage_report.json`：本地无27B tokenizer，真token五桶均为`unconfirmed`，>4096明确未覆盖；206非ARC1官方覆盖206/206，55无V2靠官方55/55、生成侧0/55；多query仅首query进首轮，test1等记为未覆盖。
  - `d2_audit20.md`：20 base人工初查（长7/非ARC1 6/多色9/重复5/LOO3/生成6）；生成侧4 slots独立种子、无跨slot共享、`validate()`未运行，6例均为存疑/未确认；11/12两LOO在旧规则下会被删，D2已改标记保留。
  - 偏差说明：首轮每base至多3行，略宽于计划“1–2个一致增强”，远少于G2的24；3000为CPU索引目标，非GPU训完承诺。
- 代码缺口最小实现（仅`scripts/`，`py_compile`与`--help`已验，未跑模型）：
  - `train_grid_sft.py`新增`--init-adapter`（与`--resume`互斥；新建LoRA仅复制交集shape一致的`lora_*`，新模块零初始化；池/cursor/optimizer/scheduler/RNG全重启；`run_config.init_adapter/stage`记旧新训练量；`adapter_step_{N}`immutable快照，`adapter/`仅作最新指针）；`--resume`原精确恢复语义未动。
  - `solve_grid.py`新增`--compare-mode base_vs_adapter`（单次加载，先base greedy经`disable_adapter`确认再adapter greedy，切换清cache；要求`--adapter`+2048+`--n-sample 0`）、`--task-view-seed`（sha256(task|query|view|sample)，默认关闭时沿用旧`sample_seed`）、`--second-view transpose`（全demo+query转置、逆变换重评、去重/oracle统计）；逐query四类+合法/尺寸/截断/EM/时间/tokens。
  - 未实现记入`experiments/008/code_gaps_followup.json`：GDN独立分支、混合cache DFS移植、旋转/换色视角、TTT入口、程序分支、显式长度混合CLI。
- 合规：无测试文件/CI新增；G1/G2 episodes sha复核不变；raw-eval仅题面；GPU配额未动用（仍0.7874h，刷新10-03 08:00北京）。下一步仍按008：先跑E0配对评价，再定G2初始化与小规模TTT/TTA。

## 11. 2026-10-01 CPU第二批补记（D2脚本+真token+新base+长度混合，不含GPU）

- `arc_episode_data.py`（1363→1757行，`py_compile`+`--help`+前3 task dry-run已验）：`collect_bases`枚举全test+合法LOO（demos≥2，`loo_demos_lt2`跳过），`origin/base_instance_id/suspect_same_as_demo/target_isolated`落字段，旧删除仅`--drop-suspect`显式传时走；`d2_first_k_augmentations(K=2)`、`d2_order_episodes(official_frac=0.4)`固定来源→task→base→query→增强；`d2_token_buckets/d2_coverage_report`外部tokenizer口径；`--gen-cross-slot-check`仅留锚点。dry-run：bases13（test3+loo10）、episode10、代理桶6/2/2、lost空。
- 真token（tokenizer-only，`~/.cache/qwen38-tokenizer`，revision `1d4bf0f2`，vocab 248044/len 248077/added 33，EOS 248046，未下模型权重，未进git）：口径与训练编码一致，已消费800 prompt/target 458556/62808与Kaggle日志一致。D2 3000真桶（行/任务/base/监督token）：≤1024 589/226/395/43518；1025–2048 802/329/546/140702；2049–4096 853/326/557/290424；4097–8192 598/198/358/389826；>8192 158/55/94/209008；覆盖全部892任务，最大18124。G1全量10000真桶2514/2996/2722/1409/359（已消费800全≤1024）。丢失：G1在1K下606任务、2K下333个；D2在4K下150任务、8K下31任务。结论：>4096在D2明确覆盖（756行/253任务）；4K训覆盖742/892、8K覆盖861/892，全覆盖需>18K上下文，不做速度承诺。`d2_coverage_report.json`已刷新（旧unconfirmed/proxy保留，新增confirmed_by_tokenizer，含2条手工复算）。
- 新base（ARC-GEN可用，`/tmp/arc007-arcgen` HEAD=`a15cbdb`）：`data/gen_newbases.jsonl` 2240行（instance 100–104×448族，与G2已用0..6不交叠），+G2的750=2990≈3000；每base存原文与耗时（均值490ms/p95 452ms/最大47s）；规划一致增强2个未展开为episode行。`validate()`旧30+新30族60/60通过，但其固定参数绕过随机分支与draw→None路径，只能证手选示例可构造，不能当规则oracle。6存疑族均为存疑/不足（行号依据见`d2_audit20.md`附录A），跨分片全量attempt计数未保留已如实记录；`d2_manifest.json`只加newbases段，已确认字段未改。
- `--length-mixture`（`train_grid_sft.py`，默认None走传统路径完全兼容，推荐`'20,40,40,0,0'`，>4096默认未覆盖；显式如`'20,30,30,10,10'`可含4K/8K）：五桶界与D2同界；最大余数法配额（如800步→160/320/320/0/0）；桶内`(family,episode)`排序+`seed+桶序`shuffle、缺桶记`mixture_gaps_by_bucket`、桶间交错；`train_log`每update记`length_bucket`，save点写`run_config.bucket_consumption`；与`--resume`联用拒绝（新配比走`--init-adapter`新阶段）。`code_gaps_followup.json`中该条已改为“已实现（CPU，待GPU验证）”，其余GDN/DFS/旋转/TTT/程序仍未实现。
- 合规：无测试文件/CI；007池sha不变；tokenizer权重未进git；无GPU/推理/提交。文档CPU任务至此收完；剩余均为等配额GPU项（E0配对、G2a短训、A1对照、GDN/G3/Unsloth/提交）。
