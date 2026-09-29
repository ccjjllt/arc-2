# 006 — Qwen3.8-27B：训练、推理与12小时预算调研

日期：2026-09-29。对应计划：`docs/plan/006-qwen38-l4-budget.md`。

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
