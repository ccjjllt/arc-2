# 003 — 深度调研与方向选择：结果

日期：2026-09-29。对应计划：`docs/plan/003-research-direction.md`。实现交接：`docs/plan/004-specialist-solver.md`。

后续更新：用户明确排除 SOAR 7B，并要求优先研究比 Qwen3 更强的初始化模型。本文保留当时的文献记录；SOAR 次线推荐已撤回，当前优先级与模型比较见 `005-stronger-backbones.md`。没有证据证明 NVARC 的 Qwen3 已达容量上限。

## 本轮决策

主线选择 **ARC2 新规则数据 + 已有 4B 网格专用模型继续微调 + 逐题适配**；次线选择 **独立的 7B 程序模型 + 示例执行反馈与修复**。先回答新数据是否带来新解题能力，再研究两分支互补。

这不是选择“继续复制原 Notebook”。公开模型与 Notebook 只是降低启动成本的工具，研究贡献应来自数据分布、训练目标或互补求解方法。30.14 分已按用户澄清记作未改动的复制试提交，不作为本项目能力证明。

暂不优先：从零训练新递归架构、完整重做冠军预训练、迁移到单芯片 TPU、把更大的通用模型硬塞进四卡、长期围绕解码阈值调参。它们仍可作为后续方向，但当前证据与资源适配不占优。

## 实际完成与未完成

- 已经通过 Kaggle MCP 读取公开赛事元数据、完整规则页、数据说明、文件概要、公开 Notebook 源码及作者讨论。
- 已读取模型卡、论文、GitHub 源码和数据生成器目录；进行了官方任务 ID 与生成器/论文题号的集合核对。
- 已更新 AGENTS、README、001 的用户澄清、002 的历史状态，建立本轮计划与研究结果，以及 004 实现交接。
- 未启动训练、模型推理、付费教师调用、比赛提交；没有本项目新增分数。
- 未创建测试文件、CI 或工程门禁。大量代码与实验由用户按 004 接手。

## 1. 通过 MCP 重新核实的比赛事实

上轮关于“当前 MCP 读取都受认证阻塞”的概括需要收窄：部分账户接口仍需要认证，但本轮公开接口确实成功返回内容。已使用 `get_competition`、`list_competition_pages`、`get_competition_data_files_summary`、`get_forum_topic`、公开 `get_notebook_info`。不要把公开读取成功写成已取得账户权限。

| 事实 | 直接读取结果及研究含义 |
| --- | --- |
| 队伍数 | 本轮 MCP 返回 2,237；按常规奖牌公式约前 14 名金牌，不是固定分数线 |
| 评分 | 对每个 test output 取两次尝试的最高 exact-match 得分，再对所有 test outputs 求均值；不是同题全部输出都对才计分 |
| 运行 | CPU/GPU 都不超过 12 小时；无网；`submission.json`；每天 1 次、最终选 2 次 |
| L4 | 关联本赛事才能使用；开发会话同样禁网；配额消耗按 T4×2/P100 的两倍，账户具体可用小时未核实 |
| 数据概况 | 6 个 JSON 文件，总字节 6,912,541；能读取文件概要不等于读取了全部文件内容 |
| 文件结构 | training/evaluation 各有 challenges 和 solutions；test challenges 与 sample submission；正式 rerun 替换 test 文件 |
| 官方数据文案 | 称正式隐藏集合 240 tasks；文件名在文案里混有连字符/下划线，实际读取必须按挂载目录确定，不能硬抄文案 |
| 奖金 | 排名前 8 的 Progress Prizes 合计 $275,000；Rules 称另一个 $275,000 为 Innovation Prize，Overview 称 Grand Prize；按六项研究指标评 Solution Writeup；$150,000 为 85% bonus |

来源：[Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)、[Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)、[Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)、[奖牌规则](https://www.kaggle.com/progression/competitions)。具体奖金以最新比赛条款为准。本轮没有取得实时金牌/第八名的分数线，001 的缓存榜单不能替代它。

当前可直接核对的 **官方 GitHub** ARC2 包为 1,000 个 training task、120 个 evaluation task；这是独立来源的实际文件统计，不与尚未读取内容的 Kaggle 挂载文件强行等同。[官方数据仓库](https://github.com/arcprize/ARC-AGI-2)

研究含义：训练可以在赛前其他平台进行，提交时必须自包含；不能在评测过程中远程调用 Astra 或 Colab。金牌与排名奖金相关，研究 writeup 奖另考虑方法与证据的原创性，因此只复制提交不够支持后一类成果。

## 2. Astra 的 95%：属实，但来源条件不同

[ARC Prize 官方结果](https://arcprize.org/results/openai-gpt-6-astra)列出 ARC-AGI-2 Max **95.0%**、XHigh **93.3%**。用户提到的 90%+ 有依据。另一个常见数字 ARC3 99.95% 是 Provider Adapter 配置；ARC3 Standard 的最佳值为 62.71%，不能互换。

[官方评测政策](https://arcprize.org/policy)区分 frontier 模型的 Semi-Private 和 Kaggle 的 Private；ARC1/2 模型验证走直接输入到网格输出，不使用客户端工具或 agent harness。因此不能说“Astra 的 95% 已证明是程序搜索带来的”，也不能套用 ARC3 的费用来估算 ARC2。

[OpenAI 官方模型页](https://developers.openai.com/api/docs/models/gpt-6-astra)当前标明不支持 fine-tuning。本轮未找到可下载的 Astra 权重或复刻训练代码。可用它辅助研究规则、程序与生成器，但“用 L4 微调 Astra”不是当前可执行路线。

本轮不启动教师 API。先利用已有公开程序和生成器；只有后续确有数据缺口时，才考虑用可获得的教师产生新的、可执行验证的规则与程序。教师能力到小模型效果的转移仍需实验，不把 95% 当作学生可达分数。

## 3. 高分 Notebook 与实码检查

| 资源 | 核实到的内容 | 如何使用 |
| --- | --- | --- |
| [sorenravn/arc2-vanilla-exact](https://www.kaggle.com/code/sorenravn/arc2-vanilla-exact) | 官方页面 V1 31.39，使用 Sorokin 4B 权重 | 可作为可读参考实现；页面普通运行时长不是隐藏评分耗时 |
| [christopherdaleman/arc-2026-nvarc-trm-aggressive-cost-order](https://www.kaggle.com/code/christopherdaleman/arc-2026-nvarc-trm-aggressive-cost-order) | MCP 实际取得公开 V1 源码，last_run_time=2026-09-26；关联 Sorokin 4B、TRM 数据与离线安装依赖；页面 V1 31.11 | 可分离网格和 TRM 路径，理解已有 TTT、候选生成、重排和按题 GPU 分工 |
| [koushikrudra/failed-in-aimo](https://www.kaggle.com/code/koushikrudra/failed-in-aimo) | 网页的 33.89 对应 V1；MCP latest 已是 V2，加入 coder/DSL 分支 | 不能把 V1 分数归给 V2 的新分支；所需 coder 权重未在返回 metadata 中列出，是否实际运行未确认 |
| [Sorokin 专用 4B 模型](https://www.kaggle.com/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16) | V1，约 7.27GB，Apache-2.0，标注可微调 | 作为网格学生初始化；“关联最佳分数”不是模型独立成绩 |

实码层面，已读到候选 `score_kgmon`、TTT Trainer、每卡任务进程及最终选解逻辑。说明加入“微调/重排”本身并非新方法。公开 Notebook 里的算法注释也不能当作正确性依据：只因程序拟合了全部示例就替换一个既有候选，仍可能丢掉真解，不能保证单调提分。

2026-07-26 的[参赛者实验复盘](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/729743)报告：多种阈值、更多解码、延长 TTT 与 TRM 集成没有显著公榜进步；其后转向合成数据续训与蒸馏。这是作者定性经验，没有完整对照数表，支持研究优先级判断，但不证明我们续训一定有效。

2026-09-19 的[随机性讨论](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742027)报告同代码提交 29.86/30.14，并指出评分增强 seed 中使用 Python `hash` 的问题。这个证据使“单次公榜涨几分”的解释更谨慎；研究时应查看同一题的候选覆盖和新增/丢失解题，而不是只盯一个汇总数字。

[高分团队讨论](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/724647)中 CPMP 表示截止前不会分享更多。未取得榜首可完整复现的公开方案；不能把排行榜位置写成“代码和权重已可用”。

## 4. 最有价值的数据资产

| 数据 | 一手证据与规模 | 本项目用法 |
| --- | --- | --- |
| [Google ARC-GEN](https://github.com/google/ARC-GEN) | 2026-03-25 新增 500 个 ARC2 任务生成器，加旧 400 共 900；Apache-2.0 | **主数据来源**。优先新增 ARC2 规则，而不是只做旧题颜色/旋转增强 |
| [arc-steps](https://huggingface.co/datasets/lbn32/arc-steps) | `arcgen_v2` 有 500 个 ARC2 training task、487,091 条记录；其中 486,091 条有步骤；CC-BY-4.0 | 最快先使用 input/output 并按 task_id 组成 few-shot episode；中间网格留给后续单独消融 |
| [NVARC Synthetic](https://www.kaggle.com/datasets/sorokin/nvarc-synthetic-puzzles) / [Augmented](https://www.kaggle.com/datasets/sorokin/nvarc-augmented-puzzles) / [Artifacts](https://www.kaggle.com/datasets/sorokin/nvarc-artifacts-puzzles) | 作者仓库说明分别约 103K 合成题、3.2M 增强题及生成描述/程序材料 | 保留一部分旧分布防止遗忘；生成材料用于研究新规则组合，不必全量重训 |
| [RE-ARC](https://github.com/michaelhodel/re-arc) | 400 个 ARC1 任务的生成器与 verifier，MIT | 可执行程序来源和旧能力保留；不代表 400 种新 ARC2 规则 |
| [ARC-TGI](https://arxiv.org/abs/2603.05099) | 461 个任务族生成器；其中 ARC2 66 个，含 55 train 和 11 test；提供规则链模板 | 有用的补充，明确排除所选留出任务及其衍生数据，不能按样本随机切分 |
| [SOAR 程序数据](https://huggingface.co/datasets/julien31/soar_arc_train_5M) | 作者名称为 5M；查看器与索引行数口径不同，未全量核实 | 程序学生用；按 task_id、成功程序和长度抽取，避免直接吞长轨迹 |
| [Nemotron SFT ARC](https://huggingface.co/datasets/nvidia/Nemotron-SFT-ARC-AGI-v1) | 2026 年公开的 single/multi agentic traces；长输出且许可 metadata 与正文存在不一致 | 备选程序资料，暂不纳入第一轮必需训练数据 |

### 实际任务集合核对

以 ARC-GEN commit `a15cbdb44c776610aeeb9f487a06af875d3d0878` 和 ARC2 commit `f3283f727488ad98fe575ea6a5ac981e4a188e49` 的目录比较：

- 新增 500 个 V2 生成器全部对应 ARC2 training，0 个对应官方 evaluation。
- 全部 900 个生成器中，891 个对应 ARC2 training，9 个是未包含在 ARC2 中的旧 ARC1 任务，0 个对应 ARC2 evaluation。
- 同一 generator 的大量随机样本仍属于同一种基础规则；数量增加不等于规则多样性增加。

“新增500生成器”指2026新增的程序化采样能力，不代表底座没见过这500种规则：NVARC已用过官方ARC2训练题。首轮研究的是增加每种规则的变体覆盖、反例与组合难度，真正未见规则覆盖需要额外设计，不能用生成器数量直接证明。

这些是本轮实际文件/目录核对结果，不是模型实验成绩。

### 训练覆盖的实际影响

[NVARC 训练配方](https://github.com/1ytic/NVARC/tree/main/ARChitects)包含 `arc2_evaluation6`；其数据脚本把任务的 train/test pairs 合并构造训练材料。因脚本有过滤逻辑，未逐题证明全部 120 个都实际进入最终 checkpoint，但已经足以说明不能默认公开 evaluation 是干净留出集。

因此要区分：公开题诊断、已知任务族的新采样、新组合规则与未知私榜。即使把 ARC-GEN 的 task_id 分组留出，也只能保证本轮续训没有使用这些 family，不能抹去初始化 checkpoint 可能已经看过原始规则的历史。

## 5. 方法比较及选择理由

| 方法 | 证据 | 决策 |
| --- | --- | --- |
| 4B 网格模型继续训练 | NVARC 有模型、数据和训练实现，2025 private 24.03%；2026 多个公开 Notebook 已能运行 | **主线**：复用底座，把资源投入新增 ARC2 规则和组合难度；不把历史分数当本项目结果 |
| SOAR 小模型程序搜索 | [代码](https://github.com/flowersteam/SOAR)、[7B权重](https://huggingface.co/julien31/Soar-qwen-7b)、[论文v2](https://arxiv.org/html/2507.14172v2)齐备；52%是 ARC1 大搜索集成结果 | **推荐已撤回**：用户明确排除；仅保留历史文献事实，不再安排实验 |
| Poetiq / ArcMemo | [Poetiq](https://github.com/poetiq-ai/poetiq-arc-agi-solver)依赖 API；[ArcMemo](https://arxiv.org/html/2509.04439v3)59.33%来自 ARC1 的100题子集 | 借鉴执行反馈、修复、不同输出保留；先不复刻多 API agent 系统 |
| LLaDA 递归扩散 | [ARChitects 报告](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)有 2D 位置编码和递归解码；完整离线训练8×H100约39h | 有价值但工程与训练投入较大，本轮不做主线 |
| TRM / DRM | [TRM](https://github.com/SamsungSAILMontreal/TinyRecursiveModels)官方ARC2训练示例为4×H100约3天；[DRM论文](https://arxiv.org/abs/2604.18839)的14M模型public pass@2为24.9%，[代码](https://github.com/wwwwwwwwz/DenoisingRecursionModels)公开但本轮未见checkpoint | 可直接复用的TRM权重先作互补候选；DRM需单独训练研究，暂不从零重造架构 |
| TraceViT | [论文](https://arxiv.org/abs/2607.29586)报告18M模型ARC2 public pass@2 24.3%；8×A100 80GB训练；[仓库](https://github.com/LiuBinnan/TraceViT)当前未见完整训练代码和权重 | 优先取其已公开数据，不把论文24.3%写成可下载模型效果 |
| Inkling-Small | [官方ARC2成绩40.1%](https://arcprize.org/results/thinky-inkling-small)；[作者规格](https://thinkingmachines.ai/news/inkling-small/)276B总参数、12B激活 | 仅按4bit权重下界约138GB，已大于四L4物理96GB；不作为当前全GPU微调底座 |

### 两种学生必须分开

NVARC 原作者明确将 Qwen3 缩为 **16 个 embedding/token**。它面向数字网格，不是可直接输出自然语言或 Python 的普通聊天模型。把 SOAR 程序、教师自然语言解释或 arc-steps 的 `rule` 文本混入这个 tokenizer，是错误的训练设计。

- 网格学生学习示例→最终网格；中间网格监督需额外设计序列格式、边界和推理过程。
- 程序学生使用完整 coder tokenizer，学习题目→Python，以及失败程序＋示例差异→修复。
- 两分支共用任务与候选网格格式，可以比较输出；不共用文本序列化。

SOAR公开checkpoint的模型卡未明确它对应离线训练还是包含后续公开题适配的阶段，训练覆盖也记为未确认；本轮不追加训练不等于该模型没见过公开题。

TraceViT 的消融也不支持“多加步骤天然更好”：其 ARC1 trace-only 对照曾退步，加入 task reference/object workspace 后才改善。因此第一轮先做新数据的 final-grid 训练，再单独研究步骤监督。

## 6. 一篇最新论文的题号核查

[2026-09-23 的 LLM＋遗传搜索论文](https://arxiv.org/html/2609.27242v1)称在前60个 ARC2 公开评估题上从2题增到6题。我们将表4六个成功题号与官方 ARC2 包直接比对：

`08ed6ac7`、`0b17323b`、`017c7c7b`、`0c786b71`、`0c9aba6e`、`00576224` **全部位于 training，全部不在 evaluation**。

因此不采用它的“10% ARC2 泛化”作为选型证据。其 LLM 提供搜索起点的思想可以研究，但论文标题和数字不能替代数据口径核实。这里只报告可证实的不一致，不推断作者动机。

## 7. 下一轮与效果边界

先执行 004 的第一轮：固定一个公开网格底座，形成 ARC2 新数据混合，做一个短程 LoRA 对照；并行准备 SOAR7B 的小规模离线候选实验。第一轮无需教师付费、无需新架构、无需复杂记忆系统。

本轮的新成果是可执行的研究判断与证据整理。继续微调是否提分、程序分支是否互补、是否足够金牌，都还没有本项目实验结果。

资料访问方式：Kaggle MCP 公开工具、官方网页/模型卡、GitHub MCP、少量公开文本和 ARC2 数据归档只读下载；没有下载大模型权重或执行外部训练代码。所有网络资料查阅日期：2026-09-29。
