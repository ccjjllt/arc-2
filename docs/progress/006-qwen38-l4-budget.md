# 006 — Qwen3.8-27B：训练、推理与12小时预算调研

日期：2026-09-29。对应计划：`docs/plan/006-qwen38-l4-budget.md`。

> 最新状态（2026-09-30）：006首轮诊断已收官，真实BF16短程LoRA与推理均已运行；下文按历史阶段保留。007已回读指定Notebook版本源码和原始日志，归档于`experiments/006/source/`，详细复核见`docs/progress/007-grid-specialist-sft.md`。关键修正：关闭thinking的分支未实际执行；R2b跑完4题但6/16候选仍截断；R0L六个不同episode的loss非单调；T8新增一题属于小样本线索，未确立最优步数。正式保存式SFT按`docs/plan/007-grid-specialist-sft.md`继续。

## 当前判断

Qwen3.8-27B在L4×4上有值得推进的LoRA训练及推理实现路径。**显存有规划空间，并不等于已经实测可运行，更不等于12小时内足够高分。** 当前合理方案是赛前ARC2专用LoRA，提交时固定权重或有限预算TTT，配合受控候选生成。

用户已选定模型；不再展开其他底座选型。NVARC提供两阶段适配和候选聚合的研究思想，其4B代码、词表、训练量与耗时不能原样搬到27B。

## 已查证事实

- [官方模型配置](https://huggingface.co/Qwen/Qwen3.8-27B/blob/main/config.json)：64层、24个注意力头、4个KV头；全注意力与Gated DeltaNet混合；词表248,320；模型类型沿用qwen3_5。
- [NVIDIA官方训练配方](https://docs.nvidia.com/nemo/automodel/recipes-e2e-examples/qwen3-8-27b)：有BF16、FSDP2、activation checkpointing和rank8 LoRA实现；公开结果为8×H100 80GB上100步医学VQA训练。没有公布本项目4×L4、ARC2数据的训练显存/吞吐。
- [Axolotl同架构配方](https://docs.axolotl.ai/docs/models/qwen3.5.html)：有QLoRA和FSDP2路径；其27B QLoRA示例约47GiB，说明4bit底座能存下不代表训练总显存能存下。
- [Red Hat准确型号INT4实现](https://huggingface.co/RedHatAI/Qwen3.8-27B-INT4)：W4A16 GPTQ/AWQ smoothing、vLLM部署；保留embedding/lm_head及部分模块高精度。可借其配方量化我们合并后的SFT权重，不代表其通用评测能保证ARC2无损。
- [Kaggle要求](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)：每次评分Notebook总运行不超过12小时，无网。赛前训练与评分运行分开；赛前仍受平台会话和账户额度约束。

## 显存估算：哪些是算术，哪些尚未实测

以约27B参数估算，单位为十进制GB：

| 项目 | 估计 | 限制 |
| --- | --- | --- |
| BF16权重 | 约54GB，实际完整产物约55–56GB | 四卡分片平均约14GB/卡只是常驻权重部分 |
| 8bit裸权重 | 约27GB | 未含scale、未量化层、运行时与缓存 |
| 4bit裸权重 | 约13.5GB | 不能当作训练或推理全部显存 |
| 常规混精Adam全参数状态 | 若按16bytes/parameter约432GB | 未含激活；具体优化器实现不同。不作为96GB四卡主训练路线 |
| 4096×248320的BF16 logits | 单序列约2.03GB | FP32约4.07GB，还可能有临时共存；NVARC的16-token loss成本不能套用 |

训练峰值还包含：激活、LoRA参数/梯度/优化器、临时参数收集、loss logits以及CUDA工作区。默认官方LoRA配方的MaskedCrossEntropy会生成并处理大词表logits；是否使用受支持的chunked/fused linear CE对L4显存很重要，不假定修改一个参数就完成适配。

补充：官方配置的input embedding与lm_head不共享，合计约2.54B参数；若它们保持BF16，仅这一项就比“所有参数全4bit”的算法多约3.81GB。因此13.5GB只是编码下界，不是现实部署模型体积。[Masked CE源码](https://github.com/NVIDIA-NeMo/Automodel/blob/main/nemo_automodel/components/loss/masked_ce.py)、[Axolotl Cut CE分片配方](https://github.com/axolotl-ai-cloud/axolotl/blob/main/examples/qwen3.5/27b-qlora-fsdp.yaml)

这些估算支持首先研究冻结底座LoRA与分片，不支持承诺全参数训练或单卡QLoRA必然可行。

推理侧优先比较BF16 TP4与INT4 TP2×2，单卡四副本待显存实测。官方24/4注意力头及16/48 GDN头均可被TP2/TP4整除；完整注意力BF16 KV理论约64KiB/token/序列，8K约512MiB，额外GDN状态和运行时仍需计入。[vLLM模型支持](https://docs.vllm.ai/en/latest/models/supported_models/)、[混合缓存说明](https://docs.vllm.ai/en/latest/design/hybrid_kv_cache_manager/)

## 12小时规划结果

本轮做了预算算术，没有测量真实token/s。

采用240任务、固定预留1小时的示例，可用39,600秒：

- 1个四卡模型副本：平均每题165秒。
- 2个两卡独立副本：平均每题330秒/worker。
- 4个单卡独立副本：平均每题660秒/worker。

多个副本时单副本速度与精度可能变化，不能把数字当作无代价加速。实际隐藏输入必须读取任务数及每题query数，240只是官方说明中的规划假设。

若每题1个query、2候选×4096输出token，共需1,966,080 token，解码吞吐下限约49.65 token/s；8候选则约198.59 token/s。此为全系统聚合、忽略prefill/TTT/排序后的下限；更复杂任务需要额外预算。

因此第一版优先控制thinking和生成预算，赛前承担尽量多的任务学习；TTT需按真正新增解题能力决定投入。不能把每个任务都进行原NVARC训练量作为默认配置。

## NVARC哪些方法保留

可迁移：ARC任务式SFT、几何/颜色增强、每题从固定底座重新适配、根据已知示例训练、多候选和增强一致性、任务间时间分配。

需要重做：完整词表的序列化和loss、混合注意力LoRA目标、缓存/搜索实现、PoE概率尺度、每题TTT步数、候选与视角数量。原NVARC使用的Qwen3-4B与27B的成本不同；参数倍数也不能直接当速度倍数。

当前确定的研究顺序为：**赛前LoRA → 固定权重受控推理 → 小步数TTT净收益研究 → 量化/副本布局与候选预算优化**。具体交接见同名plan。

## 实际产出与尚未完成

- AGENTS记录用户已选定Qwen3.8-27B；005标记选型结束；README转到006。
- 新增006计划及本进展，写明赛前/提交两阶段、训练路径、显存估算、12小时算术与三组手动研究。
- 没有实现训练/推理代码，没有下载模型权重，没有启动Kaggle GPU、付费API或比赛提交。
- 未测实际峰值显存、训练速度、推理吞吐、完整提交时间或ARC2得分；这些由接手后的手动实验补齐。
- 无测试文件、测试框架或工程门禁。

资料查阅日期：2026-09-29。

## 2026-09-29 MCP 通道核实与 R0 交接（本轮新增）

- `get_competition` 成功：2237 队，决赛截止 2026-11-02 23:59 UTC；`get_competition_data_files_summary` 确认 6 个 JSON 共 6,912,541 字节；tree 列表与 003 记录一致（training/evaluation/test challenges + solutions + sample submission）。
- 规则页已重读：Code Requirements 确认 CPU/GPU 均 ≤12 小时、无网、`submission.json`；Evaluation 确认按 test output 取两次尝试最高 exact-match 再平均。
- `list_models(search=Qwen3.8-27B)` 找到可离线挂载的 BF16 原始权重 `ugvfpdcuwfnh/qwen3-8-27b`（Transformers/transformers-bf16/v1）；`get_model` 确认上游 revision `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`、约 55.56GB、18 shards、Apache-2.0，另有完整性校验 Notebook 记录。另有官方 FP8 镜像（`mikedan7/qwen3-8-27b-fp8-official` 等）可作推理对照，不作为已微调产物。
- `get_accelerator_quota`（2026-09-29 会话 token 实测）：GPU 已用 6618.79s / 108000s，剩余约 28.16h，刷新于 2026-10-03T00:00Z；TPU 0 / 72000s。R0 及后续短程训练按此剩余额度安排，不假定无限。
- 同日直连诊断：本地 MCP 配置为 remote `https://www.kaggle.com/mcp` + `Bearer {env:KAGGLE_MCP_TOKEN}`（配置结构正常，密钥值未写入文档）。实测矩阵——成功：`get_competition`、`get_competition_data_files_summary`、tree files、`list_competition_pages`、`list_models`、`get_model`、`get_user_profile`（含 ccjjllt 公开画像）；失败：`get_accelerator_quota`、`get_competition_leaderboard`、`search_competition_submissions`（均为 `Unauthenticated`），`get_notebook_info`（`Permission 'kernels.get' was denied`）。结论：服务端返回的凭证无账户级权限，需用户刷新 MCP token（补 kernels/competitions 相关 scope）并重启会话，本机进程内无法代为修复。
- 新增 `scripts/r0_qwen38_lora_feasibility.py`（草稿，未运行）：只读探针 + R0 配置打印，不启动全量训练；本地仅做 `py_compile` 通过，L4 实测待用户按文件头三步手动执行后回填每卡峰值、step time、tokens/episode。
- R1（无 TTT 推理基线）与 R2（0/8/16 步 TTT 净收益）未运行；未下载权重、未训练、未提交。

## 2026-09-29 MCP 直连操作链验证（本轮新增，用户会话 token，经用户同意开 CPU 会话）

- 通：MCP initialize + tools/list（71 工具）、`save_notebook` 建私有没有网 CPU Notebook 并 SaveAndRunAll、`get_notebook_session_status` 轮询、`list_notebook_session_output` 取日志。探针 Notebook `ccjjllt/r0-cpu-chain-probe-mcp-operated`（kernel 136389170，v1/v2）与 `r0-cpu-chain-probe-v3-setter-at-create`（kernel 136390656，v1）均 COMPLETE。
- 不通：`save_notebook` 的比赛数据源挂载。先后试了创建时 `competitionDataSources`+has 标记、更新时 `competitionDataSourcesSetter`、新建时 `competitionDataSourcesSetter`（值均为 `arc-prize-2026-arc-agi-2`，与用户可用 Notebook 的元数据字符串一致），三轮元数据 `competition_data_sources` 均为 None，`/kaggle/input` 为空。结论：当前 MCP 服务端忽略该字段，不是本会话用法问题。
- 后果：R0 的数据/模型挂载这一步需用户在 UI 点一次（Add data → 本比赛 + BF16 模型），之后探针与 R0 均可由 MCP 代跑。两份探针 Notebook 为私有无网，不影响配额大头；用户可在 UI 删除或留作 R0 底座。

## 2026-09-29 挂载链打通（本轮新增）

用户在 UI 将比赛数据与 BF16 模型挂到 `r0-cpu-chain-probe-v3-setter-at-create`（v2，kernel 136390656）后，MCP 只读拿到 COMPLETE：`/kaggle/input/competitions` + `/kaggle/input/models` 均已挂载，`arc-agi_training_challenges.json` 含 1000 个训练任务，示例 episode 组装成功（`PROBE_RESULT=OK`，样例 233 字符）。MCP 代操作链（建/跑/查/取日志）至此全通；下一步开 L4 跑 R0 时模型文件明细一并核对。

## 2026-09-29 SDK 直推打通数据源挂载（本轮新增，替代 MCP save_notebook）

结论：用户说得对，access token 就够，不需要 `kaggle.json`。实测：`KAGGLE_API_TOKEN=<会话 token>` 经新版 SDK（`KaggleHttpClient` Bearer）被主 API 接受（`/api/v1/competitions/list` 200）；`KernelsApiClient.save_kernel` 带 `competition_data_sources` + `model_data_sources` 推 v3 成功，元数据确认双挂载，运行 COMPLETE 且 `PROBE_RESULT=OK`。MCP `save_notebook` 忽略数据源字段的问题就此绕过；后续 R0/R1/R2 的 Notebook 发布改走 SDK，执行状态与日志继续走 MCP 只读接口。

## 2026-09-29 R0a v4 实测（SDK 推 GPU 版，subagent 审查后修复 3 blocker + 5 major）

- SDK 无 `machine_shape` 字段，推送拿到的是**默认 GPU：2×T4（31.3GB），不是 L4×4**。L4×4 仍需 UI 切换或 CLI `machine_shape` 指定；本次 Ras probe 按设计走 SKIP 分支，无失败。
- 镜像预装：torch 2.10.0+cu128、transformers 5.0.0、accelerate 1.13.0、peft 0.19.1。**transformers 5.0.0 无 `Qwen3_5ForCausalLM` 且无 `models.qwen3_5` 模块**：27B 加载必须走 `trust_remote_code=True`（上游自带建模代码）或先升级 transformers（离线需 wheelhouse，见 README 的 003 模式）。
- 模型选择回退：config 的 `model_type=qwen3_5` 能读到，但层数字段名与预期不一致（读到 None），校验未命中走 fallback；后续按实际 config 键名收紧选择器。
- tokenizer 可用，实测 vocab 248044（与 HF 页面 248320 不一致，按挂载 revision 的 tokenizer 为准）。12 个诊断 episode token 长度 212–12660，短/中/长分布齐备。
- 27B 加载探针：`SKIP_SMALL_GPU`（31.3GB < 60GB 阈值），符合预期；`R0A_RESULT=SKIP_SMALL_GPU`，Notebook COMPLETE 无 ERROR。
- R0a 结论：默认 GPU 跑不了 27B 是算术必然；真正的 L4×4 加载/forward/显存数字待 L4×4 会话补测（R0b）。配额消耗仅一次短 CPU/GPU 会话，可忽略。

## 2026-09-29 R0b 实测：4×L4 就位，27B 加载被 transformers 版本挡住

- L4×4 配方确认：CLI push + `kernel-metadata.json`（`machine_shape: NvidiaL4` + `competition_sources` 挂本比赛）→ 4×L4、每卡 23.7GB、合计 94.6GB。无比赛挂载时同字符串只给 2×T4。Notebook：`ccjjllt/r0b-27b-l4-feasibility` v1，COMPLETE。
- R0b 结果 **LOAD_FAILED（有意义的阴性结果，不是链路失败）**：镜像 transformers 5.0.0 无 `Qwen3_5ForCausalLM`、无 `models.qwen3_5` 模块；`trust_remote_code=False` 报 architecture 不识别，重试 `True` 同样失败。tokenizer 与 12 个 episode（212–12660 tokens）正常，说明权重文件本身可读，卡在建模代码。
- 006 未完成：R0a ✓（环境/tokenizer/数据）、R0b ✓（拿到阴性结论）、R1/R2 未启动（被加载问题阻塞）。
- 下一步（三选一）：① 查支持 qwen3_5 的 transformers 最低版本，打 wheelhouse 数据集离线挂载（003 项目同款模式）；② 先用 FP8 官方镜像（`mikedan7/qwen3-8-27b-fp8-official`）验证推理链路，BF16 训练问题并行解决；③ 读上游 modeling 代码确认其 transformers 下限。推荐 ①+② 并行，配额充足（剩余约 28h）。

## 2026-09-29 R0b v2 实测：wheelhouse 生效，27B 在 4×L4 跑通（R0A_RESULT=OK）

- wheelhouse 数据集 `ccjjllt/qwen38-wheelhouse-py312`（私有，15 包约 18MB：transformers==5.8.0、tokenizers 0.22.2、safetensors、hub 1.33.0、accelerate 1.15.0、peft 0.21.1 等；tokenizers 必须 ≤0.23.0，hub 必须 <2.0，torch/numpy 等大包沿用镜像）。Notebook 首 cell `pip install --no-index --find-links` 成功把 5.0.0 换成 5.8.0（`has_Qwen3_5=True`）。
- Notebook：`ccjjllt/r0b-27b-l4-feasibility` v2（CLI push，数据源=比赛+BF16 模型+wheelhouse，`machine_shape NvidiaL4`），COMPLETE。
- 加载：BF16 + `device_map=balanced` + `trust_remote_code=False` + `dtype` 成功；加载后峰值 gpu0 12.03 / gpu1 14.07 / gpu2 14.04 / gpu3 19.81 GB，每卡剩余约 4–11GB。
- 单 forward（7 token）：1.81s，logits (1,7,248320)；forward 后峰值 11.73/13.77/13.75/14.78 GB。

## 2026-09-29 R0L v5：短程 LoRA 6 步跑通

- 解法：subagent 查明 torchao 门限来自 peft（>=0.19.0 要求 torchao>=0.16，缺失则放行，只有残留旧版才 raise）；plain BF16 LoRA 直接 `pip uninstall -y torchao` 绕过，无需新轮子。
- Notebook `ccjjllt/r0b-27b-l4-feasibility` v5：6 步 loss 0.4821/0.34/0.5502/0.2064/0.2512/0.126（有学习信号），step 时间 3.8–8.5s；训练峰值 alloc 13.61/15.24/15.22/18.06 GB，reserved 最高 gpu3 23.32GB（距 23.7 上限仅 0.4GB，batch/序列放大前必须先处理碎片与分片均衡）。
- R0 完成：环境、tokenizer、数据、加载、forward、短程训练吞吐与峰值全部落数。rank8/alpha32冻结底座LoRA在4×L4可跑；100–300步的G1/G2级训练按~6s/step估算约10–30分钟/GPU阶段（未含数据准备与多卡通信波动）。

## 2026-09-30 006 第一轮收官：R0/R1/R2 全部落数

- R0：环境/tokenizer/数据/加载/forward/短程训练全部完成（见上）。rank8/alpha32冻结底座LoRA在4×L4可跑，step约4–9s；gpu3 reserved余量0.4GB是放大前必须处理项。
- R1（`r1-27b-infer-baseline` v3）：8题oracle 4/8，总耗时1842s；未微调27B解`833966f4/6150a2bd/e9afcf9a/66e6c45b`，另4题因512截断全败（thinking吞预算）。
- R2v1（`r2-ttt-0-8-16` v1）：T0=4/T8=4零增减，T16被TIME_GUARD整组跳过，耗时3406.8s。
- R2b（`r2b-ttt-8-16-focused` v1）：4题全部完成，没有组级时间中止；仍有6/16候选触及512输出上限。总耗时2014s；相对历史R1的T0，T8新增`5582e5ca`、零丢失；T16与T8命中集合相同；TTT耗时T8合计116.6s/T16合计229.2s，生成占大头。结论：8步TTT提供值得继续研究的正面线索，16步在这4题中没有新增命中；尚无普遍最优步数结论。
- 产物：`experiments/006/runs.json`（关键指标）+ `r2b-summary.json`（R2b统计摘要，生成文本仅保存前缀）；日志中的统计摘要、相关运行提示及指定版本源码另于007复核时归档到`experiments/006/source/`，完整日志仍在对应Kaggle Notebook输出中。
- 配额：GPU已用62416.81s/108000s（剩约12.7h，10-03刷新）。后续G1/G2级100–300步训练（约10–30分钟/阶段估计）与12小时提交整合是下一轮（007）内容，本轮问题已全部回答，无遗留实验。

## 2026-09-29 R1 完成：无TTT基线 oracle 4/8（`r1-27b-infer-baseline` v3）

- 8 短任务×2候选（greedy+temperature采样，≤512 token），总耗时 1842s。命中：`833966f4`、`6150a2bd`、`e9afcf9a`、`66e6c45b`（双候选全对）；另 4 题双双截断——模型长篇"讲道理"耗尽 512 token，没输出答案。
- 关键发现：未微调 27B 已能解 4/8 简单题，但 thinking 不受控时吞掉全部生成预算。后续必须设 thinking 预算/直接输出对照，否则 12 小时会被推理 token 吃光。

## 2026-09-29 R2v1 完成（部分）：8 步 TTT 零增益，16 步无数据

- Notebook `ccjjllt/r2-ttt-0-8-16` v1，`R2_RESULT=OK T0=4 T8=4`，T8 新增/丢失均为 0（7 题完成，第 8 题被 TIME_GUARD 跳过）；T16 整组跳过（`t16_trunc=True`），耗时 3406.8s 触顶。
- 结论：8 步 TTT 在该集合上相对固定权重无增减；16 步是否有益未知，需小规模 R2b 补测（3–4 题 × T8/T16，T0 沿用 R1 同 tid 分数）。
- 监控链修复：轮询改以输出 `R2_COMPLETE` 标记为准、60s 间隔、命中即停；本次首次 COMPLETE 即命中，不再靠人工通知。

## 2026-09-29 R0b v3/v4：episode join 修好，撞上 torchao 版本墙

- v3 根因（已定位）：R0L cell 直接读 `training_challenges` 的 `test[i].output`，而该文件 test 项只有 input（输出在 `training_solutions.json`），1000 题全部 KeyError 跳过 → `only 0 short episodes`。v2 之前 55 分钟基本烧在重复加载上，无有效训练数据。
- v4（install + 修复版 R0L，solutions 按 task_id/test 序号 join，skipped 打印）：`kept=6 lens=[283,419,299,843,779,1015] skipped=0`，join 正确。
- v4 新失败（COMPLETE，明确错误）：`Found an incompatible version of torchao. Found version 0.10.0, but only versions above 0.16.0 are supported`。transformers 5.8.0（或 accelerate/peft 链）硬要 torchao>=0.16，镜像自带 0.10.0。下一步：wheelhouse 追加 torchao>=0.16 的 cu128/cp312 轮子（subagent 查准确规格），数据集增量更新后重跑 R0L。
