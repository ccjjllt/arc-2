# 016 公开方案与可复用资产：先把现有推理转成实际答案

查阅日期：2026-10-18，北京时间。本轮读取了当前 Kaggle 公开 Notebook 列表、两份新 Notebook 源码、Qwen/vLLM 官方资料、DiARC 最新发布及本项目 013–015 记录；没有启动 GPU、训练、提交比赛或下载大型权重。本文的下一步实验均未运行。

## 1. 决策

**不能依据现有公开资料认为“复制一个最新高分 Notebook”就能接近金牌。最短研究路径是复用 015 已经保存的长推理，做一次有限的最终答案恢复，然后验证从头运行的思考预算控制。**

015 的两个策略各为开发集 query EM@2 1/6，18 次请求中 14 次 length 且无最终内容、另有 1 次 stop 无最终内容。请求未设置硬思考预算是已核实代码事实；这支持先研究完成率，不证明新增正确答案必然出现。恢复程序即使让合法网格率提高，仍须按全分母评分。不能直接拿被截断的 015 版本作为完整的新比赛求解器。

## 2. 公开“高分”Notebook 的证据级别

本轮用官方 CLI 实时读取本比赛 `scoreDescending` 前 40 项和含 qwen 的 `dateRun` 前 30 项。**列表实际仅返回 ref/title/author/lastRunTime/totalVotes，没有分数字段。排序、标题和点赞不能建立最新版本与分数的对应。** 前列仍以 NVARC 4B 分支为主，没有在本次可读材料中找到可核验的金牌级 Qwen3.8-27B 源码及提交绑定；这不是对所有公开材料的穷尽性断言。

| 来源 | 精确可读身份 | 成绩证据 | 可借用部分与限制 |
| --- | --- | --- | --- |
| [madarshbb direct b98k](https://www.kaggle.com/code/madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9?scriptVersionId=358248812) | 014 已冻结 v1，作者显式链接 scriptVersionId 358248812 | [作者论坛报告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082#3538430)约 21.5；另称小改跌到 13。非本项目复现，未读其私有提交记录 | 思考硬预算、答案保留空间、无思考恢复、第一遍覆盖优先。不能把 context131072/C16/98K 思考照抄为 L4 已验证配置 |
| [russcore freeform](https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct) | v17，本轮 API 仍为 v17；013 已有冻结源码 | 未确认对应分数；旧源码中的吞吐注释不等于当前实测 | 截断推理尾部恢复、Python 反馈。挂载元数据仍有空 dataset source，整份不可直接当已可运行资产 |
| [dalezhong NVARC Plus Ckpt](https://www.kaggle.com/code/dalezhong/arc-agi2-nvarc-plus-ckpt) | 本轮 `GetKernel` currentVersionNumber=7，零 output cell | 源码叙述 31.94/27.64；没有精确提交记录。其“private set”用语不作为实际私榜结果 | 三轮种子池、按剩余预算选择继续搜索/重训/评分；新源码过于复杂，不应替换当前研究主线 |
| [mayankraaz Dual-Pass 35.9](https://www.kaggle.com/code/mayankraaz/arc-agi-2-dual-pass-nvarc-engine-35-9-lb) | 本轮 `GetKernel` currentVersionNumber=6，零 output cell | 标题 35.9，正文 33.89–35.97，未找到绑定该版本的运行日志/提交成绩 | 两轮 NVARC 池及 CPU LOO 原语搜索可参考；其“100% precision”没有可核验证据，LOO 通过并不保证 query 正确 |
| [mikelou1 Minimal Perfpatch](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch) | 014 冻结 v2 | 33.89 为标题/作者历史关联；论坛另一个约 40 的方案没有可读源码绑定 | GPU 端 teacher-forced NLL 可参考；不得将未公开新方案成绩赋给旧 Notebook |

新读取的两个 NVARC 文件都挂 `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1` 及原 Unsloth utility。不是新 27B 权重或已证明更强的底座。

本轮新 NVARC 代码还有两个具体提醒：

- dalezhong v7 首格手动设 `KAGGLE_IS_COMPETITION_RERUN=1` 并断言有 240 道 test task。普通运行可见的比赛 test 文件不等于隐藏评测全集，因此不能照搬这段作为全量验证；该源码没有输出可证明断言成功。官方 list 与 GetKernel 的 lastRunTime 字段存在不一致，本轮不凭它判断真实运行日期。
- mayankraaz v6 的 DSL 分支对 demo 少于 2 的情形直接保留 `loo_ok=True`，随后用整个任务层面的通过标识提供候选；其最终注入实际上只补空槽/重复槽，不是正文声称的全局最高优先替换。可读实现比营销文案更可靠，仍不能从 LOO 通过推出最终网格必对。

## 3. 最便宜的有效研究：复用 015 的推理尾部恢复

013 冻结的 russcore v17 `recover_grid()` 有直接可读实现：原题 + assistant 截断推理末 8000 字符 + 要求只给最终网格的 user 消息，`enable_thinking=False`，最多 4096 新 token。014 冻结的 madarshbb `answer()` 则把末 16000 字符放在恢复提示中，最多 8192 新 token；有至少 120 秒余量才启动。两者都是额外一次生成，不是从已有文本中随意摘矩形。

**建议先对 015 已保存且最终内容为空的固定请求做一次恢复，不重算已有长推理。** 所有失败请求按统一规则处理，不能只挑看起来接近答案的题；成功请求原样保留，按原 high/low 两臂分别报告。先用末 8000 字符/4096 新 token 的有限设置即可，切勿同时比较多组尾长。必须记录新调用 token/时间及累计原始成本，恢复后全 P6 query EM@2、合法率、净新增与丢失分开。

这个小实验能回答“此前已经想出的信息是否足以写出正确网格”；不能回答“完全从头给更少预算是否同样有效”。如果恢复无新增但全部完整，应转程序执行反馈或不同可检验规则；不要只不断扩大思考长度。如果有新增，随后用从头生成的硬思考预算方案确认可复现部署，才进入 P12 和完整比赛预算估算。

公开恢复代码有宽松解析、超时处理及第二槽选择的自身限制；建议提取这一个机制接入已有 runner，不复制整个数千行调度框架。继续保留 015 对 closed grid、合法颜色、尺寸及 finish 的严格计量口径。

## 4. 思考预算、最终答案与预算匹配

[Qwen 官方模型卡](https://huggingface.co/Qwen/Qwen3.8-27B)提供 `xhigh/medium/low` effort 以及 `enable_thinking`。effort 是模型行为引导；它不为最终答案预留固定 token。官方还提示低 effort 在多轮任务上未必节省整体时间。

[vLLM 官方文档](https://docs.vllm.ai/en/stable/features/reasoning_outputs/#thinking-budget-control)说明 `thinking_token_budget` 控制思考长度，达到阈值时注入 reasoning end；`max_tokens` 仍控制总生成。只设后者不保证退出思考。这条机制的本项目固定 commit `1f7427bc0ad31fdad1b0933c77ea5acae15da2e0` 后端链路由本轮另一份方法研究审阅；本文件不把最新 stable 文档单独当作固定 wheel 运行成功的证据。

建议从头确认时先保持已有 FP8/TP4/context65536/并发4/无MTP栈，只变更思考上限及答案保留空间。例如 high 总 32768，其中思考 24576；low 每次总 16384，其中思考 12288。数值是待验证实验配置，不是已知最优。若触顶后仍无最终输出，分别记录预算未生效、提前 EOS、最终内容不合法等情形。不能以“stop”一项就认为模型交回答案。

015 长请求实测高臂聚合 31–46 tok/s、低臂 75–87 tok/s。即便仅假设每题一个 query，240×32768≈786 万输出 token；按 75–87 tok/s，单输出理论就约 25.1–29.1 小时，尚未加加载/prefill。**在当前实测速度下，全量每题32K最大预算不能当12小时策略。**实际题数/query数、早停率和吞吐均影响运行；应先全题较短首轮，再给有证据值得继续的题追加，不能将两个各跑12小时的解题器离线并集当可提交集成。

## 5. DiARC 昨日的新发布改变了什么

作者 2026-10-17 13:47 UTC 的 [commit fb436f2](https://github.com/szu-tera/DiARC/commit/fb436f2f594f8be6b54e3e1e4e84ac060bcb8776) 发布 PrefARC 并整理训练/评价入口。因此 014 中“完整原始DPO语料未发布”的表述已过时，当前已有 40 个配置文件清单，不能继续将其称为只有构造代码。

但新 [PrefARC README](https://github.com/szu-tera/DiARC/blob/fb436f2f594f8be6b54e3e1e4e84ac060bcb8776/data/PrefARC/README.md)明确：ARC2 数据的来源是 `evaluation_variants`，由公开 evaluation 题衍生，**不是 task-disjoint 的训练划分**。这比仅看论文成绩更影响本项目决策。

[固定 manifest](https://github.com/szu-tera/DiARC/blob/fb436f2f594f8be6b54e3e1e4e84ac060bcb8776/data/PrefARC/manifest.json)列出以下 ARC2 配置：

| 负例构造配置 | preference pairs | 数据字节 |
| --- | ---: | ---: |
| grid block / eval10 | 2708 | 15,177,809 |
| morphology / eval10 | 4004 | 22,420,929 |
| random perturb / eval10 | 2302 | 12,617,436 |
| rigid shift / eval10 | 4617 | 25,256,928 |
| rigid shift / eval14 | 6392 | 35,054,440 |

配置间可重叠，不能把行数相加称独立训练样本数。manifest 另列两个 ARC1 task-specific 文件分别有 6/200、70/795 行 chosen==rejected；若今后使用，按记录检查过滤，而不是把发布即视为无噪声数据。此轮只读 README/manifest/代码，没有下载这些题目及答案。

这支持的动作是复用负例构造机制、保留来源和 task ID，在本项目允许的训练来源上先做小样本。**不建议现在把公开 eval 衍生 PrefARC 混入 27B 训练后再以 P12/C6 成绩宣称泛化。** 也不建议将既有 4B adapter 直接加载到 27B。

DiARC 当前 [DPO 入口](https://github.com/szu-tera/DiARC/blob/fb436f2f594f8be6b54e3e1e4e84ac060bcb8776/src/diarc/train_dpo_qwen.py)仍为普通 q/k/v/o、FFN、embedding/lm_head 的 r256 LoRA，lr1e-6、1 epoch、batch2×accum4、max_length4096、max_prompt3584。它没有为 Qwen3.8 混合 DeltaNet 架构提供已经验证的27B训练适配。现成约2.12GB `qwen3-4b/arc-agi-2` adapter 的基座匹配仍继承014的未确认状态。

## 6. NVARC 与许可：可借资产，不重启重训练

[NVARC 作者仓库](https://github.com/1ytic/NVARC)仍提供 103K synthetic、3.2M augmented 数据及 SFT/TTT/搜索实现。继承014结论：它值得借的是独立规则、实例构造与示例一致性，不是现在复刻32×H100的训练量。014 的4B预算对照已经结束，本轮没有新证据支持重开另一轮4B训练步数搜索。

许可按资产分开：DiARC 仓库当前是 Apache-2.0，PrefARC README 明确第三方 benchmark/generator 仍按原许可；Qwen 模型与 vLLM 各自许可不能替代 Kaggle repack/wheelhouse 的来源记录。NVARC 根目录本轮未见统一 LICENSE，不能将整库自动视为 Apache；已有 ARChitects 文件许可头应逐文件保留。新两份公开 Notebook 的具体 license 字段未由本轮 CLI metadata 返回，记为未确认；不因“公开可读”就宣称可任意再授权。

## 7. 本轮读取身份与未运行边界

当前公开列表与新 Notebook 的临时读取文件 SHA256：

| 文件 | SHA256 |
| --- | --- |
| scoreDescending 前40列表，8149字节 | `9e9209a3bfd41083b1127309fad2b67c04b05c36cad49fe69e20d63237318b2f` |
| qwen/dateRun 前30列表，6147字节 | `9ae8c4c0b7181de5d95d5adbf033208a0eb488ce76b0cd301a3a84956fda7d4d` |
| dalezhong v7 ipynb，101798字节 | `40fd6a5000e09f1357cef80f8658fb133dd2c191dca1f1f004d6b4c33f2d914e` |
| mayankraaz v6 ipynb，133875字节 | `9318224b200c6717b8362c679546b9f9441d5295f3e5a59a22948a0fe75b92be` |

原临时读取路径分别在 `/private/tmp/arc2-016-public-kernels.json`、`arc2-016-qwen-kernels.json`、`arc2-016-dale/`、`arc2-016-mayank/`。主研究已将两份列表、两个准确版本Notebook及metadata复制到[持久证据目录](evidence/public-notebooks/manifest.json)，保留SHA，不执行其代码。旧27B冻结源码继续使用 `experiments/013/qwen-public/` 与 `experiments/014/public-sources/` 的现有身份。

**执行参数以[016主计划](../../docs/plan/016-budgeted-final-submission.md)为准**：恢复固定末16000字符/4096新token；前向优先比较同seed、C4的两臂8K总预算（6K思考+2K答案）。本文列出的8K字符尾、32K/16K预算仅为公开代码先例和早期候选，不与主计划同时展开。

本轮未运行：final恢复、硬思考预算生成、程序反馈、DiARC adapter推理、新DPO、互补选2、C6及新比赛提交。没有新的公榜分数，没有依据宣称金牌或奖金概率上升。
