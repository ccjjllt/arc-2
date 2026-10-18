# 016 方法研究：从长思考空答案转向有答案的有限预算实验

查阅日期：2026-10-18，北京时间。承接 [014 方法资料](../014/methods-data-research.md) 与 [015 实验结果](../../docs/progress/015-qwen38-reasoning-research.md)。本文件仅记录资料核实、源码阅读与新实验设计；**没有运行新 GPU 推理、训练或比赛提交，以下效果全部未确认**。不更换用户选定的 Qwen3.8-27B，不启动新框架或测试体系。

## 1. 新证据改变了什么

015 已证明现有 27B 能在部分开发题产生正确答案，但尚未形成适合全题运行的方案：P6 两臂各命中 1/6，18 次调用只有 3 次完整 final；14 次触及长度上限，另 1 次提前 stop 但 final 为空。约 341,780 个 completion token 消耗在这些调用中。不能把目前的主要损失全解释成模型不会解，也不能把补上 final 等同于解对。

这次找到直接适用的、**冻结版本已存在**的机制：`thinking_token_budget` 可在达到推理预算时强制输出结束思考标记，让 `max_tokens` 的余量用于 final。015 的请求仅有总 `max_tokens` 和模板中的 `reasoning_effort`，没有传这个硬预算。后者只是模板指令强度，不保证预留答案 token。无需先升级 vLLM。

还需区分两个故障：`length + 空 final` 可由硬预算直接针对；`stop + 空 final` 可能是在思考中提前生成 EOS，硬预算并不必然阻止预算之前的 EOS。015 没有保存原始 token ID，不能仅凭症状把提前 EOS 写成本项目已确认根因。

## 2. 冻结源码核实：不是只查最新版文档

本项目实际 wheel 版本为 `0.27.2rc1.dev91+g1f7427bc0`。下表通过 GitHub connector 读取该精确 ref，不是 `main`。Git blob SHA 标识所读文件，不能替代 wheel SHA 或证明 GPU 实际路径已经执行。

| 文件 | 读到的机制 | Git blob SHA |
| --- | --- | --- |
| [chat_completion/protocol.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/entrypoints/openai/chat_completion/protocol.py) | `ChatCompletionRequest` 接受顶层 `thinking_token_budget`，`to_sampling_params` 将其传入 `SamplingParams` | 本次核实内容与固定 ref，未另记录 blob |
| [sampling_params.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/sampling_params.py) | 参数校验为非负整数；-1 转成无限制；独立于总 `max_tokens` | 本次核实内容与固定 ref，未另记录 blob |
| [config/reasoning.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/config/reasoning.py) | `initialize_token_ids` 可以从 parser 自动取得开始/结束标记并编码；显式结束字符串也支持过渡语句 | 本次核实内容与固定 ref，未另记录 blob |
| [thinking_budget_state.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/v1/sample/thinking_budget_state.py) | 从 prompt 最后一个开始/结束标记判断是否已在思考；累计 token，到限后逐 token 强制结束序列 | `efac0111779e3b19e846d73f1e6bea835afbb9bc` |
| [sampler.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/v1/sample/sampler.py) | 约404行起，在 penalties 之后调用 holder 的 `update_state` 和 `apply_to_logits`；强制值作用于采样前 logits | `ac8512f22c8f632efc0bdba9656b5b9120d8b309` |
| [parser/qwen3.py](https://github.com/vllm-project/vllm/blob/1f7427bc0/vllm/parser/qwen3.py) | Qwen3 parser 的边界为 `<think>` 与 `</think>`；注册路径已迁移到 engine adapter | `0041a46eaaea810e20aaba03aaaf717da2dd0732` |

因此可从现有 HTTP 请求做最小变更：保持权重、TP4、context65536、C4、采样参数及模板传参不变，新增顶层 `thinking_token_budget=6144`，总 `max_tokens=8192`。如使用 OpenAI SDK，非标准字段放 `extra_body`；当前脚本直接构造 JSON，放顶层。不要把它误塞进 `chat_template_kwargs`。官方 [Thinking Budget Control 文档](https://docs.vllm.ai/en/stable/features/reasoning_outputs/#thinking-budget-control) 描述了相同用法。

**源码存在不等于运行成功。** 下一次同会话开始，用一次短问题、极小硬预算记录实际 reasoning→final 切换；然后进入已冻结研究题。不要为了这个机制另建测试套件。当前没有启用 MTP，不为本轮同时打开 MTP：公开 [issue 54906](https://github.com/vllm-project/vllm/issues/54906) 报告的是另一组 NVFP4 / Model Runner V2 / MTP 组合中的预算失效，不能当作本项目已发生的错误，但说明不应同时引入这些变量。

### 提前 stop 的相关报告与不可用修复

Qwen 官方仓库的 [issue 216](https://github.com/QwenLM/Qwen3.8/issues/216) 与 vLLM [issue 55420](https://github.com/vllm-project/vllm/issues/55420) 有用户实测报告：Qwen3.8 在思考内发出 EOS，返回 `finish_reason=stop` 且 content 为空。这是社区在项目仓库提交的一手实验记录，**不是官方已确认结论或 ARC2 对照结果**；其报告的失败率、量化栈、提示和改采样效果均不能移植。

建议的 `reasoning_eos_policy=force_end` 不是现有可用参数。[PR 55562](https://github.com/vllm-project/vllm/pull/55562) 本次经 GitHub API 核实：`state=closed`、`merged=false`、`closed_at=2026-09-13T11:17:09Z`；作者也注明没有跑 Qwen3.8 GPU 评测。不临时 cherry-pick 该 PR。全局 `ignore_eos` 或盲加 repetition/frequency penalty 也不是本轮首选：它们会改变合法终止或重复格子的生成分布。先对空 final 做一次有限的 final-only 恢复，并保存其额外成本。

## 3. 前沿研究只取能验证的部分

| 资料与公开资产 | 实际证据范围 | 本项目采用/不采用 |
| --- | --- | --- |
| [s1 论文与作者代码](https://github.com/simplescaling/s1) | s1-32B、数学/知识推理；用强制结束思考与继续思考控制预算。作者代码展示显式终止与 final阶段 | 借用阶段边界机制；不据其数学成绩宣称 ARC2 收益，不照抄无限追加 Wait |
| [The Coupling Tax，2026-05](https://arxiv.org/html/2605.07686v1) | Qwen3-8B、Qwen3.5-27B 等，GSM8K/MATH-500/BBH；把截断轨迹送入独立回答阶段可回收部分损失。不是本项目 Qwen3.8-27B，也不是 ARC2 | 支持重用015原轨迹做一次 final-only诊断；第二次调用的 prefill、token 与墙钟不能免费 |
| [SeVRA，2026-06论文](https://arxiv.org/html/2606.19808v1)、[作者代码](https://github.com/Sajib-006/SEVRA) | 冻结 Qwen3-4B，MATH500/GSM8K/CommonsenseQA；作者明确报告某些二次验证会把正确答案改错，且更长首轮在数学上更省总体token | 先调首轮预算；只对可观测失败触发一次恢复，并记录新增/丢失。无需在6题上训练路由器 |
| [Mid-Think，2026-01论文](https://arxiv.org/html/2601.07036v1)、[作者代码](https://github.com/uservan/Mid-Think) | Qwen3-8B/14B/32B与DeepSeek蒸馏7B，MATH500/AIME/GPQA；token前缀可诱导中等长度推理。其预算曲线含先生成完整轨迹再截取的操作 | 暂不展开第三套提示搜索。它不保证硬上限，也没有 Qwen3.8 ARC2证据 |

程序反馈与数据方面，014 已审过 Poetiq、agentic coding、NVARC、ARC-TGI、DiARC、Hourglass，不重复把它们包装成“新路线”。本轮新增判断是：**先产出可执行答案，再让可验证反馈影响第二个候选**。已有 `scripts/solve_program.py` 足够承载小试验；当前真正执行过的27B反馈轮数仍为0。

另检索到 [ARCANA，2026-07](https://arxiv.org/html/2607.09059v1)，声称4×L4/12h的多agent结果，但本次未发现论文提供可下载实现/权重，主表对已有方案的参数标注也需另外核对。它不能作为目前可复用的高分产物，不据其表格数字改变计划。论文新、标题贴近比赛不等于可直接部署。

## 4. 两项主实验与一个便宜的前置诊断

### D0：原15份空final的有限恢复——先确认长轨迹有没有可兑现的信息

**问题**：015已花掉的思考里是否包含足够的规则信息，使独立回答阶段产生正确网格？

- 固定所有15份空final轨迹：14个length加1个stop，不按事后认为“快解出”挑选。原来3份合法final保留，不能用恢复覆盖它们。
- 输入原题面与原轨迹，明确这是尚未确认正确的草稿；仅请求最终网格，`enable_thinking=false`，每份最多4096输出token。一次恢复，不重启多轮思考。
- 不向模型提供query答案或得分。模型请求冻结、全部产物落盘后再在本地评分。评分表仍保留原18次调用及固定P6分母。
- 现有落盘是解码后的reasoning文本，不是原始token序列；重编码/重新prefill属于新的finalizer条件，不能称为原请求KV缓存上的无损续写。避免把原思考直接当普通assistant历史后被模板自动剥掉：实施前输出渲染prompt的尾部和token数，确认完整草稿实际进入请求。
- 最多新增 `15×4096=61,440` 输出token。按015低臂75–87 aggregate tok/s仅作算术约12–14分钟纯decode；长前缀prefill、安装加载与最终阶段实际速度另计，**不是承诺完成时长**。部署若使用同策略，必须把原341,780token与恢复成本一起算。

结果解释：恢复了合法答案但仍无新增正确query，说明“空final”只是表现的一部分；新增正确query则说明预算分配有可回收损失。无论哪种结果，D0都不是8K前向策略的效果，也不是“只花4K解题”。它只消化已有研究成本，后续不能依赖预先知道哪个query对错来决定触发。

### E1：真正适合12小时讨论的前向8K策略——主优先级

**假设**：比起32K/16K持续思考到耗尽，6K思考加2K最终输出能用更少总资源保留一部分真实解题能力，并形成实际两个答案。

建议固定开发P6，同015权重与采样参数，C4：每query xhigh一次（seed1001，仍沿用T-high标签）、low一次（seed2001），**每次8192总token，其中thinking最多6144**。每个query总上限16384，12次调用合计98304；分别报告high/low的EM@1，再用预先固定的high+low两槽报告真实EM@2。不要把两臂各新增第二次采样悄悄算成同样预算。最终以主计划冻结调用表为准。

输出保留空格数字行和闭合fence。主代理CPU测了合成30×30：该协议1803token、紧凑行933、紧凑JSON1801、默认带空格JSON2700，见 [容量记录](evidence/final-space-token-counts.json)。因此2048只在严格网格协议下具有容量依据；若模型在final继续写解释仍可能不够。这个计数是单个合成网格的容量检查，不能证明所有响应都按协议输出。

为了保持与015可解释的对照，不同时换量化、模板、采样惩罚、表示、TTT或LoRA；显式硬结束标记首轮用原生 `</think>`。若要试官方推荐过渡短语，应是后续单一变量，不能在第一个8K结果不佳时边跑边改题。

**判别结果**：合法率增加只是机制成功；至少看固定两个槽的正确query、新增、原已知命中的丢失、每query实际token与总墙钟。015两臂32K最大配置成本不同，只作历史参照，不冒称等预算A/B。即使8K无新增，也有价值：它给出“当前12h成本级别的27B直出没有实质增益”的具体负结果，阻止继续盲目扩大。

有信号才固定完整策略到P12与一次确认集；没有信号转E2，不把P6不断改提示直到高分。模型公开evaluation训练来源仍未确认，确认只表示“本项目未用于开发”。

### E2：有限程序执行反馈——E1之后的唯一新增求解分支

**假设**：完整程序让demo执行提供真实错误信号，能解出纯文本思考尚未解出的query。

- 先沿用固定P2，接口仍为`transform(grid)`，复用现有受限执行器。P0每题一次，建议总12288、思考最多8192、保留4096代码。这里代码目标更长，不能机械套用1803token的网格上限。
- 完整P0才执行demo；P0残缺时不追加假反馈，也不称反馈已被检验。P0之后共享同一原始程序分叉：一次真实demo差异修复，与一次独立restart，同样的每次调用上限。
- 全部实现不看query答案；demo不完全通过时不因为“很接近”替换可靠候选。仅看demo得分不能确认query正确。
- 两题共享P0对照最多6次调用、73728输出token；这是含两个比较分支的研究成本。实际部署只能选一个冻结分支，不能拿所有分支的oracle并集当EM@2。反馈版实际P0+repair每题最多24576token，仍须实测是否能与主提交方案共处12小时。
- 主要看原策略两个槽、程序两个槽和预定混合两个槽的净新增：丢掉原第二槽独有正确解须扣除。全demo一致但query错误时，记录为规则消歧失败，不继续堆自信评分器。

本阶段不新增大规模代码。后续如需接入vLLM/完整代码落盘，只改现有请求与执行连接，用户人工发起实验并回交原始结果。程序反馈出现正信号后才考虑小型可验证程序SFT；否则不同时消耗余量训练。

## 5. 数据与微调的当前决策

保留014的training-only独立base、episode可辨识、短程序目标方向，**暂不GPU扩训**。目前需要先区分未完成答案与错误规则；对这种不明瓶颈同时换训练数据会失去归因。既有网格小池25→50/100步没有新增解，loss下降也不是训练方向正确的证据。

CPU准备可继续，但只有一个目的：若E2表明模型能写有用程序却频繁犯语法/局部实现错，为下一轮少量短程序监督准备200条左右经执行验证的独立实例。复用现成training-only生成器，保留family/base/seed与程序来源，不从本轮P6推理轨迹或公开eval query答案构造训练目标。无需下载NVARC全增强池或重建其集群训练。

当前顺序是D0（有限消化旧轨迹）→E1（实际成本级别的前向策略）→有必要才E2（真实执行反馈）。这是为了在剩余约两周多内尽快得到可证伪的解题证据；不以任何外部论文、程序完成率或本地6题结果保证金牌、奖金或公榜提升。

**实施以[016主计划](../../docs/plan/016-budgeted-final-submission.md)冻结值为准**：恢复用末16000字符；E1两臂统一seed4101，避免同时更改seed与effort；E2先从总8K=4K思考+4K代码开始。本文12K程序预算只是后续单变量备选，未运行，不与主计划叠加开销。
