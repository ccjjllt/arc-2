# 021 外部研究核验
查阅日期：2026-10-21（Asia/Shanghai）。只做网页/源码/清单核验，未运行模型、训练或生成器。

## 核验结论

1. **Qwen3.8-27B 42.4%属实，但集合需要纠正。** ARC Prize 官方结果页明确写 ARC-AGI-2 **Semi-Private**，xhigh，Baseten 专用推理。medium 13.2%、low 22.8%、none 1.5%。网页还说低与xhigh档由聊天模板加额外指令，medium没有；effort不只是token预算旋钮。官方费用是记录token按云定价折算，不是Baseten实付或Kaggle耗时。42.4不能转成4×L4 FP8 12小时预期。该页下方另列120道public eval逐题结果，不能把表与Semi-Private总分混同。来源：https://arcprize.org/results/alibaba-qwen3-8-27b

2. **109个额外训练ID已独立复算确认。** ARC-GEN固定commit a15cbdb44c776610aeeb9f487a06af875d3d0878 中tasks/task_<id>.py有900个。synth-rearc固定commit d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0 的arc2_training_minus_arc1_task_ids.txt有609个，交集500、差集109。109个差集均有generator.py和verifier.py。完整ID、方法、来源和计数见 experiments/021/evidence/rearc2-generator-delta.json。它证明新增可用生成器资产，不证明109种模型未见规则，也不证明109个生成器质量或微调效果。固定树里有synth_rearc2_training.zip / synth_rearc2_training_v2.zip（各约85MB），本轮未下载核对其内容。该仓库同时含eval任务清单与生成器，因此要按官方training白名单接入，不能递归吞整个repo。

3. **CHARM 46.7%不是未见任务即插即用能力。** 主实验长期训练中使用public eval demonstrations学习任务记忆，不使用其held-out输出；46.7使用2.38M步延长训练。标准1M步配置的8×H100约127.3小时是论文测量推算，不是生产run实际总耗时。对隐藏题更有参考性的ARC1 checkpoint→未见ARC2适配实验，CoSE r=32记忆适配public eval pass@2为12.19±0.90，全模型逐题适配15.20±0.88；这不是ARC2预训练checkpoint适配的上限，也不能同主46.7直接对比。优先复用数据，不安排本届重训新架构。来源：https://arxiv.org/html/2609.33270v1 、https://github.com/water-vapor/charm

4. **Re-ARC2并非无需审查的完美真值。** 论文描述人工检查、跨模型审计、规则修复和部分原官方例子的错误修正。对首轮续训，建议抽109族中少量代表核对规则、输出尺寸与非同一输入/输出、演示间规则一致性；原版官方文件保留，作者修正单列，不静默替换评估真值。生成器和verifier互相一致不能独立证明规则正确。来源同CHARM论文附录D及固定synth-rearc。

5. **DiARC有真实论文结果，但不构成本届增分保证。** 论文Qwen3-4B ARC2表中30.83→33.06（+2.23），为作者在其public eval协议下报告；论文注明单张L40训练，不能换算成本地L4预算。公开PrefARC README明确ARC2来源为evaluation_variants，并保留chosen=rejected的异常记录（manifest注明）；因此不能直接导入并声称未见验证。可以保留“从本轮training白名单生成近错负例”的方法备选，不并行再开DPO线。来源：https://arxiv.org/html/2606.26530v1 、https://github.com/szu-tera/DiARC/blob/main/data/PrefARC/README.md

6. **NVARC数据边界必须准确。** ARChitects README罗列arc2_evaluation6，不等于它是训练数据。独立读取sft_mg.yaml确认它在val_dataset_path；train_dataset_path是arc2_training、mini、concept、rearc、nvarc_training、nvarc_full。run_sft_4b.sh的EXP_NAME是qwen3_4b_grids15_sft139。build_datasets.py将eval train/test拼起来生成6种变体是验证资产构造，不能写成真实eval test答案已直接SFT。项目003已有nvarc_full来自eval规则描述的核验，所以公开eval不能宣称未见规则；精确权重覆盖范围仍需权重/数据版本绑定。来源：https://github.com/1ytic/NVARC/blob/main/ARChitects/sft_mg.yaml 、https://github.com/1ytic/NVARC/blob/main/ARChitects/run_sft_4b.sh 、https://github.com/1ytic/NVARC/blob/main/SDG/scripts/build_datasets.py

## 对交接的研究建议（推断与设计，未验证）

- 同意“成熟NVARC完整流程→同预算27B互补→限时4B续训挑战者”的投入排序。但互补必须由两个最终提交槽的**净得分变化**判断。NVARC所有候选与27B所有候选的正确并集仅是oracle上界，不能当实际策略成绩。
- 27B路由不要只拿4B最低置信度题。第一轮按4B置信度/候选一致性分层，覆盖中等置信度与部分高置信度对照，避免把27B预算全投到两者都不会的题。已有本地27B历史解题记录可用于开发诊断，不能以已知答案选择“隐藏题路由”。
- 先保存冻结NVARC候选，再跑一次27B候选生成；离线重放三种朴素策略：保留NVARC top2、保留top1+符合训练示例程序输出、仅在NVARC重复槽或缺候选时插入27B。选择规则必须在看留出答案前冻结。后两种再用完整同预算实跑验证，因为离线重放不含被挤掉的4B计算。
- 对程序候选，demo全过是必要信号但不是正确证明；优先收集“多个不同程序产出同一答案”“支持集留一验证”“旋转/颜色等变一致”信号，按实际成本使用。不要为此建复杂学习排名器。
- 若用4B likelihood重排27B答案，必须采用同一套增强、长度处理和同一评分程序，单独记“新正确被拒、错误被提升”；这只是候选选择假设，可能压制真正互补。
- 48小时应定义为工作期限并附GPU配额上限，而不是无限并发的训练时间。先做一次小规模接入和吞吐测量，再固定唯一checkpoint、数据配比及训练上限。只训练一个SFT挑战者，DPO/蒸馏/新架构不并行。
- “新增109族”优先级应依据实际失败与难度分布，不能109族一律加倍。留出按task/family；新合成留出只能叫“本轮未用于续训的family/样本”，不叫初始化模型未见题。
- 4B续训后必须在原TTT、增强、搜索、选择下比较。若只是无TTT改善而完整TTT无净收益，不能晋级。adapter用于离线SFT后再逐题TTT时，应明确合并或分层方式并确认逐题重置，避免把跨题适配累积误当改进。

## 保留但不展开的额外文献
ARCANA（https://arxiv.org/html/2607.09059v1）有自报较高ARC2结果及复杂agent架构，但本轮正文未找到代码/权重链接，不能据其宣称替换现有可运行基线。不进入当前实现计划。

