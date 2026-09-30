# 007 — Qwen3.8-27B 正式网格微调与 NVARC 专用化

日期：2026-09-30。对应计划：`docs/plan/007-grid-specialist-sft.md`。

## 本轮结论

可以开始正式网格SFT。006已经提供短序列BF16 LoRA可运行的实测起点；下一步应该留下可复用权重，训练直接答案，再研究小词表与搜索。**本轮完成的是源码复核、证据归档和007实现交接；007训练、Unsloth对照、词表转换及竞赛提分均未运行。**

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
