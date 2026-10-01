# 007 — Qwen3.8-27B 正式网格微调与 NVARC 专用化

日期：2026-09-30。对应计划：`docs/plan/007-grid-specialist-sft.md`。

后续研究见[008结果](008-generalization-adaptation.md)：25个已评留出query中24个来自旧ARC1；现有证据不足以判断ARC2新任务泛化。008更新下一阶段评价、G2初始化和小规模TTT/TTA研究顺序，以下保留007当时的执行记录与判断。

## 本轮结论

可以开始正式网格SFT。006已经提供短序列BF16 LoRA可运行的实测起点；下一步应该留下可复用权重，训练直接答案，再研究小词表与搜索。源码复核、G1 训练和计划中的诊断加 24 题留出评价已经完成。`ccjjllt/g1-grid-sft-007` 版本 1 为 `COMPLETE`。诊断集 exact-match 7/8。留出集 25 个 query 都已评分，但分属两种输出上限：960 token 下 0/5，2048 token 下 6/20。**这些都不是竞赛分数。** G2 的 30K 池已在本地生成并核对，尚未训练。G2 的 GDN 训练、G3 的 27B 转换、Unsloth、TTT 和竞赛提分仍未运行。

优先级：正确关闭thinking＋答案mask＋正式SFT保存权重 → ARC-GEN/长度/GDN适配 → 小词表网格专用分支 → 新checkpoint上的有限TTT/候选搜索。64层混合主干先保留，Unsloth不是开训前提。

## 实际完成的工作

- 通过已认证Kaggle MCP只读获取四个实验指定版本的源码及运行日志，没有启动新Notebook。
- 将Notebook原文、提取的可读`.py`、元数据和源码SHA256写入 `experiments/006/source/manifest.json` 及同目录文件。
- 从服务端日志提取 `R0L_SUMMARY`、`R1_SUMMARY`、`R2_SUMMARY`、`R2B_SUMMARY`，保存为 `*-result.json`。这是原运行打印的完整**统计摘要**，其 `text_head` 仍只是生成前缀（R2b为前300字符、R1为前500字符），不能冒称完整候选答案。
- 逐段阅读实际训练循环、模板、LoRA targets、逐题重置、解析器和计时逻辑；旧 `scripts/r0_qwen38_lora_feasibility.py` 未改，它只是早期探针草稿。
- 读取NVARC原作者Notebook v10及词表转换/SFT源码，核对Qwen3.8官方模板、TF5.8混合cache和Unsloth多卡/依赖说明。
- 新增配对007计划与本进展，更新README及006现状说明；没有新增测试文件、训练器实现或GPU实验。

## 006数据和代码复核

| 项目 | 核实到的事实 | 应如何解释 |
| --- | --- | --- |
| R0L训练 | 6次真实forward/loss/backward/optimizer.step，LoRA可训练39,845,888参数 | 梯度训练链路可用；未保存权重，未证明专项泛化能力 |
| R0L loss | 0.4821、0.3400、0.5502、0.2064、0.2512、0.1260 | **非单调**，6步分别取6个不同episode，不能把首末差当作同一任务收敛曲线 |
| R0 forward | 旧R0b v2记录1.81s、logits `[1,7,248320]` | 只有7-token forward，不代表完整ARC长输入耗时 |
| 精度 | BF16，未配置FP8/INT4/QLoRA | BF16是16位浮点，不应称FP32全精度；对比量化时可称未量化BF16 |
| 多卡布局 | `device_map="balanced"` | 单进程按层放置；没有实测FSDP2、TP4或4副本并行 |
| R1集合 | 官方training里最短prompt的8题、每题仅`test[0]`；两候选命中4题 | 属于选定训练题诊断，不是ARC2 evaluation50%或比赛50分 |
| R2v1 | T0跑8题、T8跑7题、T16跳过；共同完成部分无新增 | 不写成三个组都完成8题 |
| R2b | 4题，T8和T16各命中3题；T0沿用R1历史2题 | 8步新增`5582e5ca`是有价值线索；不能外推为稳定增益或通用最优步数 |
| 输出截断 | R2b T8、T16各8候选中3个达512，共6/16 | `groups.truncated=false`只表示整组未被时间保护中止 |

R2b对T8/T16使用相同题目、相同配置、每题fresh adapter，比与历史T0的比较更直接。但R1没有固定采样seed，R2b固定seed且题目顺序不同；新checkpoint的0/8对照应在同次运行重新生成。

### 发现1：关闭thinking的分支没有执行

实际代码先检查 `inspect.signature(tok.apply_chat_template).parameters` 是否包含 `enable_thinking`；真实签名里是 `**kwargs`。于是False没有传入，三次推理记录均为 `prompt_mode_probe="chat_default"`。

[固定revision官方模板](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/chat_template.jinja)明确支持False；默认未传时还会选择xhigh reasoning提示。已找到明确调用问题，**尚未运行修正后的精度/速度对照**。不能把006现象称为“关闭思考后27B仍然只讲道理”。

### 发现2：006训练目标还不是NVARC式答案监督

R0L把演示、query输入、公开query答案全部拼接，`labels=input_ids`；R2则把已知demo文本全序列重复训练，同样没有答案mask、留一示例或一致增强。原作者NVARC TTT collator将user/padding置`-100`，只学习assistant网格及EOS。

G1需统一训练/推理表示并监督答案。现有代码确实运行了LoRA更新；但loss可能主要反映高可预测的输入/格式，不能据此认定ARC规则学习已充分发生。

### 发现3：已确认存在GDN慢速回退

四份原日志都出现：`The fast path is not available ... Falling back to torch implementation.` 并指向FLA和causal-conv1d。各结果归档保留了此提示。

TF5.8源码按算子可用性分别选择chunk/recurrent/conv实现。需要记录实际加载到的算子，不能仅靠警告计算提速倍数。优先研究在原HF栈补GDN内核；Unsloth整栈迁移另设对照。

### 发现4：解析与计时影响结论口径

- `parse_grid`从整个生成文本取最后一个矩形数字块，可把thinking里的demo/query解析成grid；`parse=OK`不代表完成最终答案。007只接受final段的完整合法grid，记录原文及parsed grid。
- `prefill_s`实际只测tokenization和输入搬运；真正模型prefill包含在旧`decode_s`的generate调用内。本次不报告纯decode速度。
- `total_s`从主实验cell开始，不包括安装cell等；正式12h须按Notebook整体计算。
- tokenizer的248044与模型输出词表248320口径不同，不能自动推断模型revision不一致。007分别记基础vocab、`len(tokenizer)`、added tokens及padded embedding/head行数。
- `max_reserved=23.32GB`不等于活跃张量已占23.32GB；`max_allocated=18.06GB`是另一个峰值，两者还可能不在同一时刻。较长序列的可用空间需实测，不能仅凭reserved推断只剩0.4GB。

## 从原始数字得到的预算含义

| R2b组 | 每题TTT | 每题generate | 每题组内全流程 | 同速线性扩到240题 |
| --- | ---: | ---: | ---: | ---: |
| T8 | 29.15s | 165.875s | 196.05s | 13.07h |
| T16 | 57.30s | 154.475s | 212.80s | 14.19h |

未计加载376.6s与其他初始化。这只是四个短题的算术，不能当作真实隐藏集预测；它说明**当前HF串行、默认thinking的原样流程没有12h余量**，不说明27B或L4的性能上限已到。

T8生成2970 tokens/663.397s，T16为2767/617.870s，均约4.48输出tokens/s，包含prefill。关闭thinking、GDN内核、批处理、TP/量化都可能改变数字，目前没有本项目实测收益。

2026-09-30 MCP quota：GPU used=71438.1583023s，total=108000s，reserved=0；剩10.156计费小时，刷新2026-10-03T00:00Z。GPU类型扣额倍率不包含在该响应里；计划依据官方L4倍率给出约5.08h会话预算的条件估算，实际前后差仍需记录。

## 对结构改造与Unsloth的研究决定

1. **直接网格SFT可立即推进**。全词表、正确关闭thinking、query答案mask和稳定保存权重构成G1；不用先动主干。
2. **NVARC式小词表值得正式比较**。27B的输入/输出矩阵不共享，16项选行可算术减少约5.085GB BF16权重及大量logits内存；颜色序列仍可表达全部合法网格。不承诺精度不降或整体倍数加速。
3. **搜索程序独立于词表转换**。先独立候选、增强与选2；Qwen3.8的GDN递归状态不能用旧纯attention DFS的KV回退恢复。没有实现新的DFS。
4. **Unsloth是可选性能路线**。已通环境TF5.8与其当前HF依赖<=5.5.0存在差异；普通DDP会复制整模，不解决24GB卡上的BF16容量。先在HF/PEFT产出模型，明确看到兼容与性能收益后再迁移。

完整参数、数据量与optimizer steps的换算、GDN targets、转换双矩阵的步骤、候选长度预算及实施文件分工见同名plan。没有声称本轮已训练出更高分模型。

## 一项历史资料更正

复查NVARC `sft_mg.yaml`：`arc2_evaluation6`属于`val_dataset_path`，不能因README列出该子集就说真实evaluation答案直接进入SFT训练。其`nvarc_full`使用evaluation规则描述生成数据，仍意味着公开evaluation不应默认当作未见规则评价。003进展已按此修正，保留不把作者模型公共eval分数直接当干净泛化的原则。

## 接手后的第一项工作

按007计划实现数据入口与正式SFT训练器，从已归档R0L代码出发，修正模板和答案mask，准备10K数据池；手动启动首次约200 optimizer updates的保存式训练。初始800条实际消费与10K池分别记录，随后独立加载adapter评价，再继续G2/G3。**本轮没有新训练结果；下一次progress追加真实产物路径和分数。**

## 2026-09-30 实现与 G1 启动

代码和数据已经落地，G1 会话已推送。下面的损失、合法率、exact-match 和配额差都还没有从这次运行返回。

### 本地已核对

- `scripts/arc_episode_data.py`、`scripts/train_grid_sft.py`、`scripts/solve_grid.py`、`scripts/convert_grid_vocab.py` 已通过 `python3 -m py_compile`。`train_grid_sft.py --help` 在导入 torch 之前退出。`convert_grid_vocab.py --self-check` 打印 `SELF_CHECK_OK`，没有加载 27B。
- 10K 池：`experiments/007/g1/episodes.jsonl`，10000 行，sha256 `42765f3b078484a85e8c7d256e51d5a88c404da83b9e4fe6995d9eefa40568e3`。训练任务 892，留出 100，诊断 8，三者不相交。`query_kind` test 5378 / loo 4622。几何 ident 3558、rot90 3558、rot180 2858、rot270 10、flipud 10、fliplr 6。颜色 ident 5352、perm1 4648。`family_id` 等于 `task_id`。系统提示与计划中的句子一致，用户文本以 `[query] out:` 结束。
- 输入文件 sha256：challenges `779eaba89790ebad9af02514a7efc0aefaf2cf8236f046a31bbf8b9ec48f20f5`，solutions `9f07a38bd25af5e83aa5bf85c5cb1a1fefdb30f6a755256fa65429e697ca97f9`。Python 3.9.6。ARC-GEN 池字段为 `not_generated`。
- 推送前修正了零步评测：它现在用官方 train 对作为 demo、`test[0].input` 作为 query，调用 `build_messages(demos, query_input)`。答案不写在 `[query] out:` 后面。八个诊断题在本地 challenges 上都能组出这种 prompt。
- `solve_grid.py` 对已经渲染好的聊天文本使用 `add_special_tokens=False`，与训练编码一致。这个脚本尚未在 GPU 上运行。
- 训练器复核：`enable_thinking=False` 直接传入，不查函数签名。标签只监督答案网格和一次 EOS。LoRA 默认目标是 q/k/v/o 与 gate/up/down。AdamW，weight decay 0.01，microbatch 1，梯度累积 4。输出上限 960。CUDA OOM 不执行该次 optimizer step，累计 8 次后以 `FAILED_OOM` 退出，并仍保存当时的 adapter。

### 已推送、尚未返回结果

- 私有数据集 `ccjjllt/arc2-007-g1-episodes`，状态 ready。内容是 `train_grid_sft.py`、`arc_episode_data.py`、`episodes.jsonl`。`kaggle kernels push` 只上传 notebook 正文，所以脚本和 JSONL 走数据集挂载。
- Notebook `ccjjllt/g1-grid-sft-007` 版本 1：https://www.kaggle.com/code/ccjjllt/g1-grid-sft-007 。推送后 CLI 状态为 `KernelWorkerStatus.QUEUED`。MCP `list_notebook_session_output` 当时日志为空。
- 配方：200 optimizer updates，grad_accum 4，microbatch 1，max_len 1024，overflow 2048，save_every 25，lr 2e-5，weight_decay 0.01，warmup 10，clip 1.0，seed 20260930，zero-step max_new 960。机器 `NvidiaL4`，挂竞赛 `arc-prize-2026-arc-agi-2`、模型 `ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1`、wheelhouse `ccjjllt/qwen38-wheelhouse-py312`。Internet 关闭。
- 推送前配额：GPU used 71455.357302300s / 108000s，reserved 0，刷新 2026-10-03T00:00:00Z。推送后的 used 还没再查，不能把这次会话的计费时长写成已测值。

### 仍未运行

- G1 的训练损失、零步输出、adapter 是否成功写入 `/kaggle/working/007-g1`、独立加载评价。
- G2 的 ARC-GEN 30K 池、更长序列和 GDN LoRA 训练。
- G3 在真实 Qwen3.8-27B 上的词表转换和训练。本地 self-check 不代替这一步。
- TTT、Unsloth 对照、竞赛提交。没有新的公榜或私榜分数。

## 2026-09-30 G1 训练结果

Notebook `ccjjllt/g1-grid-sft-007` 版本 1，状态 `COMPLETE`。日志用 Kaggle MCP `list_notebook_session_output` 回读。产物在 Kaggle `/kaggle/working/007-g1/`，已下载到 `experiments/007/g1/kaggle-v1/007-g1/`。`G1_RESULT=OK`，`G1_DONE`。没有 OOM，没有 `SKIP_`。

### 运行条件

- 硬件：4×L4，`device_map=balanced`。日志 `GPUs: 4`。模型目录 `/kaggle/input/models/ugvfpdcuwfnh/qwen3-8-27b/transformers/transformers-bf16/1`。
- 环境：先卸载 torchao，再从 `ccjjllt/qwen38-wheelhouse-py312` 离线安装。日志 `TF_NOW: 5.8.0`。GDN fast path 缺失，回退 PyTorch。
- 数据：`ccjjllt/arc2-007-g1-episodes`，10000 行。长度筛选后 `kept_le_1024=2514`，`added_le_2048=0`，`longer_than_2048=4490`，池中规则族 286，实际走入更新的规则族 264。800 条 microstep，不是 10K 全部训完。
- LoRA：r=8，alpha=32，dropout=0，目标只有 `q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj`。可训练参数 39,845,888。配置里 `peft_version` 为 0.19.1。词表：config 248320，tokenizer `__len__` 248077，`vocab_size` 248044。
- 优化：AdamW，lr 2e-5，weight decay 0.01，warmup 10，200 次更新，microbatch 1，grad accum 4。第 200 次更新的学习率是 0。
- 模板探针：`assert_applied=True`，`assert_ok=True`，空 `<think>\n\n</think>`。监督文本等于答案网格加 `<|im_end|>`，prompt 标签全是 -100。样本 `beb8660c|loo|0|rot180|ident`，prompt 464 token，答案 56 token。
- 会话日志时钟最大 5045 秒。配额：推送前 GPU used 71455.357s，结束后 93653.787s，差 22198.430s。总额 108000s，剩余约 14346s，刷新 2026-10-03T00:00:00Z。reserved 为 0。
- 训练阶段峰值 allocated GB（10^9 字节）：gpu0 13.644，gpu1 15.294，gpu2 15.265，gpu3 18.127。reserved 最高 gpu3 22.530。

### 损失

200 行 `train_log.jsonl`。mean loss 最小 0.00617，最大 1.653，第 1 次 0.1215，第 200 次 0.08816。不是单调下降，不能当成泛化曲线。

### 零步输出

这是更新开始前的底座，在同一次会话里对 8 个诊断题的 `test[0]` 做 greedy。`enable_thinking=False` 已经生效：8 题都生成合法网格并以 `<|im_end|>` 结束，没有解释文字，没有截断。parse 8/8，exact match 6/8，用时 49.4 秒。命中 `833966f4, 6150a2bd, e9afcf9a, 5582e5ca, d037b0a7, 66e6c45b`。未中 `d631b094, 00576224`。这些题来自官方训练集，而且是 006 已经用过的诊断题，不是未见规则，也不是训完之后的 adapter。

### 权重

`adapter_model.safetensors` 159,451,232 字节，sha256 前缀 `37fde109458c94b5`，512 个 LoRA 张量。`trainer_state.pt` 已下载到 `experiments/007/g1/kaggle-v1/007-g1/trainer_state.pt`，319,386,933 字节，文件头是 zip（`PK`）。没有在本地用 torch 打开它，也没有用它续训。

### 接着做、以及做不了的

剩余配额大约 4.0 个计费小时。这次 G1 会话用了 6.17 个计费小时，所以同样规模的 G2 训练放不进这次刷新周期。下一步是把已保存的 adapter 单独加载，在 8 个诊断题加留出集按字典序前 24 题的全部 test query 上评分。时间预算设为 2000 秒，避免会话在写结果前被配额掐掉。G2、G3、TTT、Unsloth 和竞赛提交仍然未运行。

## 2026-09-30 独立评价已推送

私人数据集 `ccjjllt/arc2-007-g1-adapter` 状态 `ready`。扁平文件：`adapter_model.safetensors` 159,451,232 字节，`adapter_config.json`，`adapter_README.md`，`solve_grid.py`，`arc_episode_data.py`，`split.json`。没有再上传 31MB 的 `episodes.jsonl`。

Notebook `ccjjllt/g1-grid-eval-007` 版本 1 已用 CLI 推送：https://www.kaggle.com/code/ccjjllt/g1-grid-eval-007 。挂载 wheelhouse、比赛数据、`ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1` 和上述 adapter 数据集。离线、GPU、`NvidiaL4`、与 G1 相同的 docker pin。推送后 CLI 与 MCP 都是 `RUNNING`。日志当时还是空的，分数未返回。

评测命令：`--time-budget 2000`，`--max-new-tokens 960`，`--n-sample 1`，`--seed 20260930`，`--split-file`。集合是 8 个诊断题加上留出集按字典序前 24 题，共 32 题、33 个 test query、66 个候选（greedy 加 1 个 sample）。诊断题和留出题分开计分。诊断题仍是官方训练分布上的诊断，不是竞赛分数。

推送时 GPU 配额：used 26.02h，remaining 3.98h，total 30.00h，刷新 `2026-10-03T00:00:00Z`。G2 全量训练、G3 的 27B 转换、TTT、Unsloth 和竞赛提交仍然未运行。

## 2026-10-01 独立评价结果

Notebook `ccjjllt/g1-grid-eval-007` 版本 1，CLI 状态 `COMPLETE`，`SOLVE_EXIT 0`，`SOLVE_DONE`。没有 traceback，没有 `SKIP_`。产物 `experiments/007/g1/kaggle-eval-v1/007-g1-eval/solve.json`。日志时钟约 2048 秒。结束后 GPU used 27.15h，remaining 2.85h，与推送时的 26.02h / 3.98h 相差 1.13 个计费小时。

这是训完后的 adapter 单独加载，不是零步底座，也不是竞赛提交。`base_merged` 为 false。`enable_thinking=False` 的空 `<think>` 块在 `prompt_tail` 里。环境：Python 3.12.13，torch 2.10.0+cu128，transformers 5.8.0，peft 0.19.1（镜像里的版本，wheelhouse 里的 0.21.1 没有装上去），4 GPU，模型加载 611.1 秒。`presence_penalty` 没有传入。生成显式传了 `top_k=0`、`no_repeat_ngram_size=0`。greedy 加 1 个 sample（temperature 0.7，top_p 0.9，seed 20260930）。`max_new_tokens` 960。

`time_guard=true`，`elapsed_s=2002.441`。计划中的 32 题、33 个 query 只完成了 13 个 query：诊断 8/8，留出 5/25。下一条未开始的题是 `1c786137`。已完成的留出题是 `12eac192, 14754a24, 17b80ad2, 18419cfa, 1acc24af`。

| 集合 | query | 合法候选 | 截断候选 | exact-match@1 | exact-match@2 |
| --- | --- | --- | --- | --- | --- |
| 诊断，官方训练题，不是未见规则 | 8 | 8 个 query 的 greedy 都合法 | 0 | 7 | 7 |
| 留出，未进 G1 训练池，只评到 5 题 | 5 | greedy 合法 4 | 2（同一题的两个候选） | 0 | 0 |

诊断题 greedy 与 sample 的文本相同。相对零步底座的 6/8，adapter 多对了 `00576224`，`d631b094` 仍然生成 `4 4 4 4`，和零步相同，不是截断。其余六题零步已经对，adapter 也对。sample 没有把 @1 提到 @2。

五道留出题的 greedy 都不是答案。`18419cfa` 的答案是 28×26，两个候选都在 960 token 处截断，解析失败。另外四题解析成功但格子不对；`14754a24` 和 `1acc24af` 的 sample 与 greedy 不同，仍然不对。这五题的 prompt 长度是 960、5046、3306、5732、2716，都已经达到或超过 G1 的 1024 训练上限，其中四题也超过 2048。它们不在那 800 条已训练 episode 的长度范围内。

G1 那次训练用了 6.17 个计费小时。这次评价之后还剩 2.85 小时，放不下同样规模的 G2 训练，也放不下 G3 的 27B 转换加训练。留出集还没评完的 19 题另开一次加载，不再重跑已经记分的 8 道诊断题和 5 道留出题。TTT 要等值得部署的 checkpoint；目前已评留出题 0/5，不做 TTT。Unsloth 对照仍未运行。

续评 notebook `ccjjllt/g1-grid-eval-007-rest` 版本 1 已完成：https://www.kaggle.com/code/ccjjllt/g1-grid-eval-007-rest 。`SOLVE_EXIT 0`，没有 traceback。产物 `experiments/007/g1/kaggle-eval-rest-v1/007-g1-eval-rest/solve.json`。同一 adapter，`--task-ids` 是字典序留出集里当时还没评的 19 个 id。`max_new_tokens` 2048，`--time-budget` 2000。`time_guard=true`，`elapsed_s=2003.459`，模型加载 535.2 秒。环境与上一轮相同：transformers 5.8.0，peft 0.19.1，4 GPU。日志时钟约 2042 秒。配额从 used 27.15h / remaining 2.85h 变为 used 28.29h / remaining 1.71h，这次约 1.14 个计费小时。

这一轮完成 9 个 query，全部标成 holdout，greedy 都解析成功，没有截断。exact-match@1 和 @2 都是 2。命中 `1c786137`、`22eb0ac0`，greedy 与 sample 相同。未中 `2204b7a8, 29623171, 2b01abd0, 31aa019c, 3345333e, 342dd610, 36fdfd69`。这 9 题的 prompt 长度是 1508 到 3744，都超过 G1 的 1024 训练上限。未开始的 10 个 id 是 `3c9b0459, 3d31c5b3, 3f23242b, 423a55dc, 44f52bb0, 4522001f, 47c1f68c, 4938f0c2, 4c5c2cf0, 50cb2852`。

两轮留出评测的输出上限不同，不合成一个分数。960 token 那轮是 0/5，2048 token 这轮是 2/9。加起来是 14 个留出 query 里 2 个 exact-match，还有 10 个留出 task 没评。诊断集仍是上一轮的 7/8，没有重跑。

## 2026-10-01 G2 数据池

`experiments/007/g2/episodes.jsonl` 有 30000 行、114,264,283 字节，sha256 `d09dd0dec76adf9604394d893ef42ceb32bf8eead1b3b63b46175201bba3ca74`。独立重数与 manifest 一致：官方训练侧 12000，ARC-GEN V2 侧 18000，重复 `episode_id` 为 0，诊断集和留出集 id 泄漏为 0。`prompt_tokens` 和 `target_tokens` 都是 null。`trained` 为 false。G1 的 `episodes.jsonl` sha256 仍是 `42765f3b078484a85e8c7d256e51d5a88c404da83b9e4fe6995d9eefa40568e3`。生成用 Python 3.9.6，耗时 387 秒。ARC-GEN 目录是 `/tmp/arc007-arcgen`，commit `a15cbdb44c776610aeeb9f487a06af875d3d0878`。没有调用 `validate()`，只调用 `generate()`。

500 个 V2 生成器里，`exclude_arcgen_families` 去掉 52 个（诊断 3、留出 49），留下 448 个。这 448 个 family 全部落在官方 train 的 892 个 id 里面，所以全池 unique `family_id` 仍是 892。ARC-GEN 增加的是训练规则的新实例，不是新的未见规则。448 个 family 都至少有一条保留下来的 base。750 条 base 各展开 24 种增强，正好 18000 行。ARC-GEN 侧六种几何各 3000，四种颜色各 4500。

官方 12000 行全部是 `test|0`，没有 loo。24 步增强循环在 12000 的上限处停住，所以官方侧 perm3 是 0，perm2 只有 1296，六种几何都有但 ident 2676、rot90 2188、其余各 1784。72 条 loo base 因答案等于剩余 demo 输出被跳过，和 G1 同一条规则。187 条官方 test 行的答案文本等于某个 demo 输出，按规则保留。ARC-GEN 侧有 19 个 instance 因变换后的答案等于某个 demo 输出而整段丢弃，文件里的 ARC-GEN 行没有这种重合。生成失败被重试或跳过，没有编造格子：超时 27、ValueError 3、IndexError 2、非矩形 2。

这个池还没有拿去训练。入口是 `arc_episode_data.build_g2(...)`，默认的 `main()` 仍是 G1 的 10K 池。

## 仍未运行

G2 的 GDN LoRA 训练、G3 的 27B 词表转换和训练、TTT、Unsloth、竞赛提交都未运行。GPU 剩余 0.79 小时，刷新 `2026-10-03T00:00:00Z`。G1 同样规模的训练用了 6.17 小时，这次额度不够再开训练，也不够再加载 27B 做转换。2048 token 下留出 exact-match 是 6/20，还不是值得部署的 checkpoint，所以不做 TTT。

最后 10 个留出 task 的续评 `ccjjllt/g1-grid-eval-007-rest2` 版本 1 已完成：https://www.kaggle.com/code/ccjjllt/g1-grid-eval-007-rest2 。`SOLVE_EXIT 0`，没有 traceback，没有 `SKIP_`。产物 `experiments/007/g1/kaggle-eval-rest2-v1/007-g1-eval-rest2/solve.json`。`max_new_tokens` 2048，`--time-budget` 2200。`time_guard=false`，`elapsed_s=1611.743`，模型加载 291.3 秒。peft 0.19.1，4 GPU。10 个 task、11 个 query 全部评完，没有截断，greedy 全部解析成功。exact-match@1 是 greedy，exact-match@2 是任一候选；两者都是 4，sample 没有多救回任何 greedy 错过的 query。

greedy 命中：`3c9b0459`，`44f52bb0` 的 query 0 和 query 1，`50cb2852`。`44f52bb0` 的 query 0 以及 `50cb2852` 的 sample 是错的，所以 @2 没有高于 @1。未中：`3d31c5b3, 3f23242b, 423a55dc, 4522001f, 47c1f68c, 4938f0c2, 4c5c2cf0`。`3c9b0459` 和 `44f52bb0` 的 prompt 只有 286 和 294 token，在 G1 的 1024 训练长度内。`50cb2852` 的 prompt 有 2404 token，仍然 greedy 命中。配额从 used 28.29h / remaining 1.71h 变为 used 29.21h / remaining 0.79h，这次约 0.92 个计费小时。刷新仍是 `2026-10-03T00:00:00Z`。

计划里的 8 个诊断题和留出集字典序前 24 题现在都有记录。25 个留出 query 的分布是：960 token 那一轮 5 个，0 个 exact-match，其中 `18419cfa` 被截断；2048 token 的后两轮合计 20 个 query，exact-match@1 和 @2 都是 6，没有截断。2048 的 6 个命中是 `1c786137, 22eb0ac0, 3c9b0459, 44f52bb0` 的两个 query，以及 `50cb2852`。两种输出上限不合成一个分数。诊断集仍是 7/8，没有重跑。

剩余 0.79 个计费小时放不下 G2 训练，也放不下再加载一次 27B 做 G3 转换。TTT 仍不做：2048 token 下留出 exact-match 是 6/20，还不是值得部署的 checkpoint。Unsloth 和竞赛提交未运行。
