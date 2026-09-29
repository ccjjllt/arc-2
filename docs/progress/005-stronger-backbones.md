# 005 — 更强初始化模型：调研结果与路线调整

日期：2026-09-29。对应计划：`docs/plan/005-stronger-backbones.md`。

## 结论

可以微调其他更强的开放模型，而且应把这条研究提前。当前首选能力候选为 **Qwen3.8-27B**，成本对照为 **Qwen3.5-9B**，非Qwen对照为 **Gemma 4-31B-it**。这些是值得实验的候选，不是已证实在本赛超过NVARC的配方。

没有证据证明 NVARC Qwen3 已到上限。已有材料主要说明某份公开权重与推理配方出现局部收益平台；不能上升为4B参数规模或整个Qwen家族的能力上限。

按用户决定，**SOAR 7B已从执行计划移除**。ARC1成绩不足以支持本项目中的优先推荐；不再安排它的训练、推理或集成。

## 1. “上限”需要区分什么

| 说法 | 证据判断 |
| --- | --- |
| 同一checkpoint继续加TTT步数/改阈值收益小 | 有2026参赛者经验报告支持，但不是所有任务与配置的定论 |
| 同一4B架构换数据/重新训练也不能进步 | 未证明 |
| Qwen系列或小模型在ARC2有约30分上限 | 没有依据 |
| 更大的通用模型直接运行必定超过专用4B | 没有依据；需要ARC2适配和时间预算比较 |

[NVARC原论文](https://github.com/1ytic/NVARC/blob/main/nvarc_2025.pdf)表2报告：用同一3.2M数据集合，2B模型公榜22.22、耗时6小时10分；4B为29.72、耗时12小时。这是论文中的赛后最佳运行，不能与当届官方私榜成绩混用。两者初始化、规模和推理预算不同，不能解释成严格控制后的纯参数量收益，但也没有显示“4B能力已封顶”。作者还提到同Notebook重跑可能浮动1–2分。

原文数据混合及训练步数变化也带来明显成绩差异；这是训练配方仍重要的证据，不是单独某种数据的净因果增益。[原作者实现](https://github.com/1ytic/NVARC)、[2026参赛者实验复盘](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/729743)

NVARC上游[4B训练脚本](https://github.com/1ytic/NVARC/blob/main/ARChitects/run_sft_4b.sh)的起点来自 **Qwen3-4B-Thinking-2507**，随后收缩网格词表并做大量ARC专用训练；不能把它等同未经适配的普通4B模型。

16-token词表可以完整表达ARC网格，**本身不造成某个准确率上限**。它关闭了直接输出自然语言和Python的通道，但不等于删掉全部Transformer内部预训练知识。对更强推理模型，首轮保留完整词表更有利于研究其已有能力，再另行讨论专用压缩。

## 2. 候选模型的直接证据

| 候选 | 一手事实 | 本项目定位 |
| --- | --- | --- |
| [Qwen/Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) | 27B稠密主干；后训练权重；Apache-2.0；混合Gated DeltaNet/全注意力；官方通用评测较3.6提升 | 能力主候选，保留推理与完整tokenizer |
| [Qwen/Qwen3.5-9B](https://huggingface.co/Qwen/Qwen3.5-9B) | 9B稠密、后训练、Apache-2.0；另有真实的[9B-Base](https://huggingface.co/Qwen/Qwen3.5-9B-Base)预训练版本 | 较低成本的ARC2数据与适配配方迭代；Base与后训练版不是同一实验对象 |
| [google/gemma-4-31B-it](https://huggingface.co/google/gemma-4-31B-it) | 官方卡约30.7B稠密、局部/全局注意力、Apache-2.0；另有预训练版本 | 不同家族对照，有现成QLoRA路径 |
| [Qwen/Qwen3.6-35B-A3B](https://huggingface.co/Qwen/Qwen3.6-35B-A3B) | 总35B、激活3B、后训练、Apache-2.0 | 吞吐备选；训练复杂度与总权重内存不因激活3B而消失 |

本轮未找到前三者在同一ARC2训练数据、相同适配和相同推理预算下超过NVARC的受控结果。数学/编程通用榜单帮助筛选候选，不能代替ARC2证据。`ARC-Challenge`也不是`ARC-AGI-2`。

## 3. L4×4能否微调：具体边界

| 方案 | 已核实材料 | 可得结论 |
| --- | --- | --- |
| Qwen3.8-27B BF16 LoRA | [NVIDIA NeMo官方配方](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/qwen3-8-27b)提供LoRA与FSDP2实现；其公开100步记录为8×H100 80GB | 训练有实现可参照；**不代表4×L4已验证**。约54–56GB底座需要分片，DDP全复制不可行 |
| Qwen3.5-9B BF16 LoRA | [Unsloth训练文档](https://unsloth.ai/docs/models/qwen3.5/fine-tune)示例约22GB | 单L4短序列研究可考虑，长ARC上下文另计 |
| Gemma 4-31B QLoRA | [Unsloth官方训练指南](https://unsloth.ai/docs/models/gemma-4/train)提供Kaggle示例，示例显存约22GB | 有较具体的低显存起点，但不是所有序列、rank、batch均22GB |
| Qwen3.5系27B QLoRA | [Axolotl文档](https://docs.axolotl.ai/docs/models/qwen3.5.html)示例约47GiB并提供分片配方 | 不能用裸4bit权重约13.5GB推出单卡24GB必能训练 |

Unsloth的Qwen3.5文档正文对4bit训练的量化误差有不推荐说明，示例注释却说dense27B可以，存在措辞不一致。故不把QLoRA作为所有新Qwen模型的默认无损方案。量化后的ARC完整网格效果要另记，不能只看能否加载。

Qwen3.5/3.8混合注意力还要求正确覆盖LoRA模块；旧Qwen3的q/k/v列表、DFS缓存处理和分支重用不能直接当作兼容实现。先使用官方支持的模型生成与训练入口，避免把模型比较变成重写推理引擎。

## 4. 现有ARC2实证能说明什么

强开放模型的正证据也存在：[ARC Prize官方Dots3-Note Preview结果](https://arcprize.org/results/dots-studio-dots3-note-prev-max)报告76.8%，评测使用8×H100。其[作者仓库](https://github.com/studio-dots-ai/dots3-note-prev)明确280B总参数、16B激活、Apache-2.0；仅按4bit裸权重下界约140GB就超过四L4物理显存。它说明更高开放模型能力有证据，但不能直接成为当前常规全GPU微调起点。不是仅凭评测使用8×H100就推断部署下限。

- [2026-09的agent讨论](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/742796)里，有参赛者报告Qwen3.8 agent公榜13.75；未公开尺寸、完整配置和代码。这不构成27B上限或训练后能力证据。
- [Nex-N2-mini/Aquila-mini对照](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/732723)报告两种Qwen3.5系MoE衍生模型在其64个ARC2 evaluation输出上均0命中；未做TTT。它说明新模型直接套提示词也可能很弱，不能否定专项微调潜力。
- Kaggle上有Qwen3.5-4B ARC2026模型上传，但本轮没有取得对应有效高分结果。体积小的产物可能是adapter，在读取文件前不作确定判断。
- 前轮发现的遗传搜索论文把训练题写作评估题，继续不把其“10%”当作ARC2泛化证据。

以上参赛者报告都不等同受控、独立复现；当前证据支持开展新基座研究，尚不足以预先选出ARC2优胜模型。

## 5. 公平比较与实用选型

NVARC已接受约3.2M增强样本的专用训练。如果新通用模型仅追加少量LoRA后暂时输给NVARC，可能是适配不足，而非底座能力更差。

本项目先比较**相同额外算力预算下哪个完整方案更有用**；若要归因于底座，则补充原始Qwen3-4B-Thinking-2507与新模型的共同训练实验。跨tokenizer不能只凭step数或生成token数宣称完全相同计算量，应同时记录样本暴露、实际tokens和GPU时间。

因此新的优先安排是：27B摸能力上限方向、9B快速迭代数据和训练配方、Gemma作家族对照；NVARC保留为已有专用化参照。无需证明旧模型已达上限才启动这些实验。

## 6. 本轮实际修改与效果

- 新增本轮计划/进展文档。
- AGENTS记录排除SOAR和更强初始化模型研究优先。
- 004移除SOAR、P0/P1及程序分支实现要求，保留网格续训对照。
- 003保留历史文献但标注次线推荐撤回；README更新当前入口。

本轮未下载权重、未训练、未运行模型推理或提交比赛。没有新的ARC2分数，也没有4×L4训练吞吐实测。所有建议和初始配置均为后续手动研究设计。

资料查阅日期：2026-09-29；来源包括官方模型卡、训练框架文档、原作者论文及Kaggle公开MCP/网页。
