# 007 — Qwen3.8-27B 正式网格微调与 NVARC 专用化

日期：2026-09-30。对应进展：`docs/progress/007-grid-specialist-sft.md`。

后续研究顺序由[008计划](008-generalization-adaptation.md)更新：先补代表性底座/adapter配对评价，修正G2数据与新阶段初始化，再以小实验判断TTT/TTA收益。本文保留007原方案与历史状态，不把“6/20不值得部署”解释为不允许研究TTT。

状态：G1 训练和计划中的诊断加 24 题留出评价已完成。诊断 7/8。留出 25 个 query 已全部评分：960 token 下 0/5，2048 token 下 6/20。这些都不是竞赛分数。G2 的 30K 池已生成，官方侧 12000 全是 test[0]，ARC-GEN 侧 18000 在 448 个训练规则上均匀增强；该池未训练。G2 的 GDN 训练、G3 权重转换、Unsloth 对照、TTT 和竞赛提分仍未运行。不新增测试文件或 CI。

## 1. 本轮要交付什么

主要产物是**能够保存、恢复、独立加载进行 ARC 推理的专项微调权重**，而不是再次做完几个梯度步就丢弃 adapter。第一版以原 Qwen3.8-27B 的 BF16 主干、现有 HF/PEFT 路径为起点，训练直接输出答案网格。后续在相同数据与时间预算下研究 NVARC 式小词表及候选搜索。

本轮分为相连的三个研究阶段：

1. **G1：正式直接网格 SFT**。修正 thinking 调用、输出损失、数据划分和保存逻辑；产出第一个可复用 checkpoint。
2. **G2：扩大规则覆盖与适配范围**。加入 ARC-GEN 新实例、较长完整 episodes，研究 GDN LoRA；继续保存并恢复训练。
3. **G3：NVARC 小词表专用化**。保留主干，裁剪并重新训练输入／输出词表模块，比较直接解题与受控候选生成。与 G1 分支同时保留，避免丢失已获得的参照。

单次训练会话内可以先保存零步输出，再训练并保存权重；关闭 thinking 的修正不用另起一轮长期可行性研究。真正 TP、量化多副本和复杂 DFS 可后续独立研究，不阻止 G1 产出。

## 2. 从 006 继承的事实与边界

本次通过 Kaggle MCP 回读 R0L v5、R1 v3、R2 v1、R2b v1 指定版本的源码和日志。原始 Notebook、可读代码、打印的完整统计摘要与元数据归档到 `experiments/006/source/`；未重新运行。

- BF16 冻结底座＋LoRA 的真实梯度更新已跑通。R0L 共 39,845,888 个可训练参数，6 个短 episode、6 次更新，step 3.8–8.5s。没有保存 adapter。
- 实际是**单进程 `device_map="balanced"` 按层分布到四卡**，不是 FSDP2、TP4、DDP 或四个独立求解 worker。保持实际名称，不能把计划中的并行方式写成已实测。
- R0 的 1.81s forward 只含 7 个 token，不能外推到 2K/4K 的训练或 prefill。
- R1 的 4/8 是官方训练集里按最短 prompt 选出的 8 题、各自 `test[0]` 的两候选诊断结果，不是 ARC2 公共 evaluation 或竞赛分数。
- R2b 在四个定向选取短题上，相对历史 R1 的 T0 多命中一题；T8 与 T16 命中集合相同。保留为 TTT 正面线索，不能认定 8 步已普遍最优。
- 006 仍走默认 thinking；GDN 日志明确出现 fast path 缺失和 Torch fallback。两者是可具体改善的运行因素。

更正细节与原始文件位置见同名 progress，不把复核发现变成工程门禁。

## 3. G1：将目标改成直接预测答案网格

### 3.1 统一训练和推理模板

固定模型 revision：`1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`；挂载来源沿用 `ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1`，运行时记录实际 revision 与 tokenizer 配置。

直接向模板传入参数，不再查函数签名中是否显式包含该名字：

```python
prompt_text = tokenizer.apply_chat_template(
    messages,  # system + user；user 含 demonstrations 与 query input
    tokenize=False,
    add_generation_prompt=True,
    enable_thinking=False,
)
```

官方模板经 `**kwargs` 接收开关；False 会把空 `<think>\n\n</think>` 作为已知 assistant 前缀。首次人工查看渲染后最后一小段文本即可确认分支。不要把空 thinking 块再写一遍进训练答案。`reasoning_effort="low"` 只是文本指令，不是硬思考上限。

首版保持 006 的空格分隔数字＋换行表示，以便比较；紧凑无空格格式作为后续单独变量。BPE 可能合并连续数字，不能根据字符数判断 token 数。

### 3.2 答案损失

每条训练样本分为 `prompt_ids` 和 `target_ids`：

- prompt：已知演示、query 输入、assistant 前缀；标签均为 `-100`。
- target：真实答案网格＋正确的 `<|im_end|>`；只对这些 token 算交叉熵。
- padding：attention mask 为 0，labels 为 `-100`。

HF 自带 causal loss 会内部平移一位；不要再手动平移两次。若自行做 chunked CE，则明确 `logits[t]` 预测 `labels[t+1]`。模板前缀和 target 拼接边界需要人工查看一次编码、解码和被监督的位置；不要仅按字符串字符长度生成 mask。

006 的 `labels=input_ids` 会同时学习抄写输入、示例、标签文字；G1 不继承该目标。**mask 掉 prompt 只改变监督范围，默认 forward 仍可能生成整段大词表 logits，不能把它当作已经节省 logits 显存。** 优先复用受支持的 fused/chunked CE；若需要改模型 forward，保留一个未改 loss 的小样本数值参照，通过手动运行查看差异。

NVARC 的原作者 TTT collator 监督所有 assistant 网格与 EOS。G1 先采用“演示放在 user、只监督 query 答案”的清晰格式；若改成多轮并监督所有 assistant 输出，需要作为新的 formatter/loss 配方记录，不把两者称为完全同一目标。

### 3.3 数据与划分

先在本地 CPU 生成数据，随后挂载到离线 L4 会话。

1. 对官方 training 的 1000 个 task 按规则族／task_id 划分，建议约 900 个用于训练、100 个项目留出；划分在增强之前完成，固定 seed 和清单。006 已看过的 8 题留作诊断，不放进声称全新选择的留出组。
2. 官方题的 `test` 输出从 `training_solutions[task_id][query_index]` 正确 join，覆盖全部 query。训练集里的公开答案可以用于赛前训练；项目留出题和公共 evaluation 的 query 答案仅用于评分。
3. 首版准备 **10,000 条候选训练 episodes**。可以让已知演示轮流成为伪 query，也可使用训练 task 的公开 query；同一条样本的目标不能同时作为其 demonstration 泄露。
4. G2 扩到 **30,000 条数据池**：初始采样比例建议官方题派生 40%、ARC-GEN 60%，先按 task/family 均匀抽取，再抽实例。比例是假设，不是验证过的最优配方。
5. ARC-GEN 使用已记录的固定 commit，优先对应 ARC2 training 的 V2 生成器；G1 留出 task 的生成器及其变体一并排除。500 个 V2 生成器提供的是对应训练规则的新实例，不是 500 个未见的新规则。
6. 同一 episode 中统一几何变换与颜色置换。颜色用 lookup 一次性替换，不能循环原地覆盖 0→1、1→0。对生成器输出按相同颜色语义组装 demonstration 与 query，特别注意生成器是否已经分别换过颜色。
7. 每条保存 `task_id/family_id/source/revision/seed/augmentation/query_index/prompt_tokens/target_tokens`。长度超限时移到较长组或重组完整示例，不静默截断网格；记录短序列筛选后实际剩余多少规则族。

公共 evaluation 的 120 题可作为更贴近 ARC2 的项目外部评估，必须说“未用于本项目微调”；原基座预训练是否见过它们未知。先用固定、兼顾长度和输出大小的 24–32 题做开发读数，定型 checkpoint 再人工发起更完整评估。单独报告训练规则新实例、项目留出规则、公共 evaluation；不能混成一个分数。

G1/G2 不先混入来源不明的长 CoT、规则描述或 `nvarc_full`，避免同时改变答案格式和评价口径。

### 3.4 初始训练配方与真正的训练量

| 项目 | G1 起点 | 说明 |
| --- | --- | --- |
| 底座与精度 | Qwen3.8-27B BF16，主干冻结 | 006 没有 FP8/INT4 实测；先保留可用参照 |
| 框架 | Transformers 5.8.0＋PEFT，单进程四卡按层放置 | 沿用已通 wheelhouse；记录实际包版本，不把它写成 FSDP/TP |
| LoRA | r8 / alpha32 / dropout0 / bias none | 继承现有 q/k/v/o＋MLP 目标；G2 单独扩展 GDN |
| 学习率 | 2e-5 起点，短 warmup 后衰减 | 是保守起点；有稳定 checkpoint 后再比较 5e-5，不能单凭 loss 选模型 |
| 优化器 | AdamW，仅 adapter，clip norm 1.0 | 显式记录 weight decay 等完整参数 |
| batch | microbatch1、梯度累积4 | 当前是一份跨四卡模型，effective batch=4，**不能再乘4张卡** |
| 序列 | 第一段完整 episode ≤1024，随后 2048；4096 单独分组推进 | 006 只实测短序列；按实测显存调整，不硬加 batch |
| 初始保存目标 | 200 optimizer updates，之后向累计1000–3000推进 | 200×4=800 episodes；1000–3000×4=4000–12000 episodes，尚非完整遍历30K |
| 保存 | 每25–50次 optimizer update＋会话退出前 | adapter、优化器、scheduler、RNG、数据进度、formatter 与环境一起保存 |

**10K/30K 是准备的数据池，不是运行100–300步就已训练这些样本。** 日志同时写 optimizer steps、microsteps、实际 episodes、输入／目标 token 数和 family 覆盖，恢复训练后累计统计。

若暂按 006 的 6s/microstep 估算，200 optimizer updates×4累积约4800s（80分钟），300 updates约2小时；这些都未加加载、评估、保存，也未考虑2K/4K长度增长。以前“100–300步10–30分钟”仅对应累积1、原短序列的算术，不能用于本配方承诺。

2026-09-30 只读查询：GPU quota 剩余 36,561.84s＝10.16计费小时，reserved=0，2026-10-03 00:00 UTC 刷新（北京时间08:00）。若当前 L4 仍按官方2倍扣额，粗略相当于5.08小时会话墙钟；实际以每次前后 quota 差为准。首会话建议先投入约2小时墙钟、持续落盘，保留重新加载与验证预算；后续在额度刷新后续训，不承诺一次会话完成全部训练。

checkpoint 首先保存小 adapter；在后续独立推理会话加载它，实际解题后再考虑赛前合并55GB底座。持久化到私有 Kaggle 输出／数据集，提交模型应包含 tokenizer、format 配置和精确底座来源。只保存训练后内存对象不算本轮交付。

## 4. G2：增加规则覆盖、GDN 适配和有效算力

006 adapter 只覆盖普通 attention 的 q/k/v/o 与 MLP，**48个 GDN block 的专用投影没有 LoRA**。MLP 在这些 block 中仍可训练，因此不能说这些层完全没适配。

G2 优先新增受支持的 `in_proj_qkv`、`in_proj_z`、`out_proj`；按实际 `named_modules()` 的完整路径限定 text GDN，避免误匹配其他模块。`in_proj_a`、`in_proj_b`、conv、norm、递归状态参数先保留，逐项增加有助于解释结果。r16/32 是后续容量选项，不照搬 NVARC r256。

扩大目标模块时保存为新分支：已有 adapter 权重保留，新 LoRA B 零初始化，重新建立匹配参数组的优化器；不要无说明复用旧 optimizer state。若将来切换推理引擎，需确认它支持这些 GDN adapter 模块；可以赛前合并解决固定权重部署，动态 TTT 仍须另行确认。

真实日志已有 GDN fast-path 缺失提示。可先在原 HF 栈补匹配 Linux x86_64 / Python3.12 / Torch2.10.0+cu128 / L4 的 `flash-linear-attention` 与 `causal-conv1d` 依赖，离线 wheelhouse 准备好后再挂载。TF5.8 实现分别使用 chunk delta-rule 处理训练/prefill、fused recurrent delta-rule 处理单token缓存解码；只装 `flash-attn` 不等于补齐 GDN 加速。

不能据一条警告断定所有算子均走最慢实现；记录实际导入的四个算子与 active path。在同一小批样本和相同 dtype/adapter 下看速度、显存、loss/输出是否异常，再继续同一个正式训练任务。编译失败时保留原可用栈和 checkpoint，避免把内核安装变成整轮研究停摆的前置条件。没有本项目提速倍数的实测。

## 5. G3：能否转成 NVARC 网格模型

可以，定义为**条件网格序列模型＋外部候选搜索**：给若干输入／输出例子和新输入，预测答案的颜色 token、行分隔和结束符；网络给出条件概率，搜索程序沿概率探索不同答案。修改 logits 或词表不会自动产生已学会的 ARC 规则，仍需要专项 SFT。

### 5.1 改造分三级

| 分支 | 改什么 | 收益假设与代价 |
| --- | --- | --- |
| A：完整词表直接答案（G1） | thinking开关、目标mask、数据和LoRA | 尽快得到专项权重；保留原模板/通用词表 |
| B：网格语法约束 | 生成时限制数字、空格/换行与结束符，并追踪合法行宽/行数 | 减少解释和非法答案；普通logits mask仍会计算完整lm_head，不能宣称已节省头部计算 |
| C：输入＋输出小词表 | 从旧权重选行初始化16–32个网格/控制token，替换tokenizer、embedding、lm_head | 更小logits和参数；需要新格式专训，通用自然语言输出能力不再保留 |

首版不替换64层主干，不把48个 GDN 硬换成全attention，不随机重建大块网络；这些会改变预训练函数，当前数据和算力下缺少支持。用户要的专用化可先通过 C 实现，不需要先做整网重构。

NVARC 的16项是：10数字、换行、`user`、`assistant`、`<|endoftext|>`、`<|im_start|>`、`<|im_end|>`。新模型必须从自己的 tokenizer 查旧ID，不硬编码4B ID。每个格子直接映射一个颜色token，防止整串数字被BPE重新合并。新模板只包含保留符号，不再调用会注入think或自然语言的完整聊天模板。

Qwen3.8-27B 的 `tie_word_embeddings=false`：分别从输入 embedding 与输出 lm_head 复制所选旧token的行，**不能只裁 embedding，也不能强行把两张矩阵共享**。保留 old→new/new→old 映射，更新 text/model/generation config、EOS/PAD与词表大小。若选中的符号不能由一个原token表示，先调整符号，不能声称任意平均初始化等价。

建议 C 从同一原始底座建立独立训练分支，沿用相同数据清单与任务划分，并用相同训练token/墙钟预算与 A 比较。先仅训练 LoRA＋很小的 embedding/head，保存完整小词表模块；后续再研究把 G1 合并后转换能否更快收敛。转换后不直接套用旧adapter、旧token ID、旧EOS和旧概率阈值。

按 V=248320、d=5120、非共享矩阵计算：原输入＋输出词表为2,542,796,800参数；16项仅163,840参数，BF16约减少 **5.085GB（4.736GiB）**。4096×V 的BF16 logits约2.034GB，4096×16只有128KiB。节省很有价值，但24B左右主干仍在，不能按248320/16宣称整体速度增长。

中间也可研究“保留输入词表、只计算小输出头”，但生成token需映射回原输入ID、loss需映射新label；不是把config.vocab_size改小就完成。初轮选A/C两个清晰分支，避免同时开发大量变体。

### 5.2 网格长度与搜索

- 一格一token时，30×30答案为900数字＋29换行＋1EOS＝930 token；可预留960。保留空格与原词表时先用2048左右输出上限，并用实际tokenizer核对最大格式长度。**预算不能利用待答题的真实输出尺寸**；EOS正常早停，不能继续统一512。
- 第一版生成 greedy＋1个独立采样候选。候选只使用自身cache；不惩罚重复颜色，起点 repetition penalty=1、presence penalty=0、无no-repeat-ngram限制。
- 增强候选从少量视角开始，同一题所有grid一致旋转/换色，输出逆变换后去重。候选多于2个时按仅依赖题面和候选的评分选2个；全部候选oracle与最终选2命中分别记录。
- NVARC DFS以累计NLL小于约1.609剪枝，该阈值依赖其网格SFT与TTT后的概率分布。不能原样用在27B或重归一化后的受限词表上。
- Qwen3.8有KV、GDN卷积状态、递归状态；TF5.8的线性注意力cache `crop()` 不回滚这些状态。直接复制旧DFS的cache复用/position回退会污染分支。先独立候选，必要时用正确重排全部状态的受支持beam；显式快照/重算前缀DFS留作后续研究。
- 不为930深度的每个节点保存完整混合cache。仅GDN递归状态已可约72MiB/BF16或144MiB/FP32每序列，另有KV；深层快照会迅速耗尽显存。

## 6. 专项SFT之后重新研究TTT

对新checkpoint比较0与8步；16步仅在8步之后仍有有价值变化时继续。每题重置到同一份**已经完成离线SFT**的参数，不从上一题的adapter继续训练。若赛前adapter和在线adapter并存，明确哪份冻结、哪份更新，不能无意卸载赛前学习成果。

训练材料只用该题已知demonstrations：留一个已知pair作伪query，其余为上下文；几何/颜色变换对该episode一致；仅监督伪query输出。一个demo的任务可退为该已知input→output，不把待答query答案加入训练。逐题不重新加载55GB底座，不在改权重后复用旧cache。

006 TTT把整段demo文本重复训练8/16次；G1之后的此目标已改变，所以其最佳步数需要新的实际依据。0/8在同一运行里使用相同候选配置、固定seed并分别记时间；比较固定步骤和相同墙钟预算下的最终选2命中。

## 7. Unsloth 决策

NVARC 的**离线SFT入口用 NeMo-RL/Megatron**，Kaggle逐题LoRA/推理用Unsloth；不能概括成“整套训练都因为Unsloth才成功”。

Unsloth 支持优化内核和多卡加载，可能有帮助。但截至本次源码查阅，其HF依赖声明仍约束 Transformers<=5.5.0，006已通的是5.8.0。公开精确Qwen3.8 QLoRA用户报告不是4L4 BF16实测；普通DDP会每卡复制整份模型，单卡24GB装不下55GB BF16底座。

本轮默认沿用HF/PEFT先产出G1权重。Unsloth作为**独立性能对照**：单独wheelhouse，固定同一数据、精度、rank、targets、长度、累积和有效样本量；确认其支持当前分层布局或实际FSDP配方，再看tokens/s、每卡allocated/reserved和保存后推理结果。只有收益实际可见时才迁移主训练；不引用通用宣传中的倍数作为本项目预算。

## 8. 最小代码交接与手动实施顺序

以下文件已经写在仓库里。G1 会话使用其中的训练器、数据脚本和 `experiments/007/g1/episodes.jsonl`。G2 的 30K 池在 `experiments/007/g2/episodes.jsonl`，尚未训练。G3 的真实 27B 转换和 TTT 仍未运行：

| 文件/产物建议 | 责任 |
| --- | --- |
| `scripts/arc_episode_data.py` | task/family划分、solutions join、ARC-GEN组episode、统一增强、token/target mask |
| `scripts/train_grid_sft.py` | BF16 LoRA、梯度累积、答案loss、分段长度、保存/恢复；以已归档R0L真实代码为起点 |
| `scripts/solve_grid.py` | 正确thinking开关、完整最终grid解析、全query、独立候选cache、时间与原始输出记录 |
| `scripts/convert_grid_vocab.py` | G3才实现：token映射、两张矩阵选行、专用模板与配置、独立产物 |
| `experiments/007/<run>/` | split/data/config、checkpoint位置、训练消费量、全文候选与scores、实际环境/运行时间 |

手动顺序：

1. 在本地准备划分和初始10K池；从 `experiments/006/source/r0b-v5.py`、`r1-v3.py` 迁移入口。仓库原 `scripts/r0_qwen38_lora_feasibility.py` 只是旧草稿，不能当成已跑通训练器。
2. 同一次正式训练会话加载底座，渲染False模板，保存零步基线输出；随后直接进行G1，定期保存并退出，不重新开一整轮006。
3. 另一次手动推理加载保存的adapter，在相同诊断/留出清单上评价。记录完整生成文本和parsed grid；不要只保留前300字符。
4. 继续G2的ARC-GEN、长度与GDN分支。达到值得部署的checkpoint后才做TTT与量化预算整合。
5. G3独立转换并训练，比较相同数据预算下的合法输出率、最终选2 exact-match和总时间，再决定是否作为主提交模型。

指标围绕研究问题：最终网格合法完整率、截断率、exact-match@1/@2（按每个test output）、训练已见与留出分开、生成token和墙钟、TTT新增/丢失、候选oracle与实际选2。旧 `prefill_s` 只测tokenization/搬运，重命名为prepare_s；未实际拆分时统一记录generate_s。完整Notebook时间还要加安装/加载/保存，不能仅看主cell计时。

## 9. 一手资料

查阅日期均为2026-09-30；移动分支内容可能变化，实施时记录实际revision。

- Qwen3.8 [模型与配置](https://huggingface.co/Qwen/Qwen3.8-27B)、[固定版本模板](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)。
- NVARC [16-token转换](https://github.com/1ytic/NVARC/blob/main/ARChitects/cut_tokenizer.ipynb)、[vocab](https://github.com/1ytic/NVARC/blob/main/ARChitects/qwen3_configs/vocab.json)、[SFT运行脚本](https://github.com/1ytic/NVARC/blob/main/ARChitects/run_sft_4b.sh)、[SFT配方](https://github.com/1ytic/NVARC/blob/main/ARChitects/sft_mg.yaml)。
- NVARC [原作者Notebook](https://www.kaggle.com/code/sorokin/arc2-qwen3-unsloth-flash-lora-batch4-queue)，本次MCP回读current_version_number=10；其TTT、collator、turbo_dfs和排序实现是一手代码依据。
- Transformers 5.8.0 [Qwen3.5/GDN实现](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[混合cache](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/cache_utils.py)。
- [ARC-GEN](https://github.com/google/ARC-GEN)，V2有500个ARC2规则生成器；沿用003记录的ID对应关系并在实际数据manifest固定版本。
- Unsloth [多卡说明](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)、[DDP说明](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth/ddp)、[依赖源码](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)、[精确模型用户运行报告](https://github.com/unslothai/unsloth/issues/9650)。
- [NVIDIA精确Qwen3.8 FSDP2配方](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/qwen3-8-27b)，其公开硬件为8×H100，不能冒充本项目4×L4结果。
