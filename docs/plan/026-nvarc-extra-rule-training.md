# 026 NVARC额外训练规则覆盖挑战者

日期：2026-10-25。承接[025实际结果](../progress/025-research-gap-and-next-step.md#12-完整运行及同候选池评分完成)。用户授权继续尽可能提高NVARC分数。本轮只推进一个小规模4B续训对照。

**实际结果：已完成。** 两臂最终均4/23，新增2／丢失2，净增0；oracle6→9。按本计划结束该配方，不扩验23题、不提交此权重。详见[026实际进展](../progress/026-nvarc-extra-rule-training.md)。

## 背景与问题

025原B4为15/58，R2也为15/58；raw oracle为17/58。43个失败query中41个候选池缺少正确解，因此结束R2排序扩展，先检验生成能力。历史007—011训练多属27B，不能当作这次16-token NVARC 4B续训的结果。

当前没有可靠的失败题到训练生成器规则映射。本实验称“额外training规则覆盖挑战者”，不声称已定位或补齐41类失败规则。109为相对ARC-GEN生成器资产差集，也不代表初始化权重未见这些规则。

## 假设与数据

假设：固定sft139 BF16 V1在少量额外ARC2 training生成规则上续训，同时保留官方training replay，可使原TTT／DFS生成更多正确候选，并最终提高B4两槽命中。

生成器来源固定`water-vapor/synth-rearc` commit `d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0`，只使用training来源，不读取evaluation query答案。109个族先与官方training demonstrations核对；3个不兼容族整族隔离。剩余106个族按固定seed分为90训练／16族隔离，隔离族也不得从官方replay进入训练。合成episode包含3个demo与1个query，用完整episode进行同一颜色和几何增强。

先做16个可读规则族的CPU语义核查，再扩到合格族；全量的generator/verifier一致并不等于独立语义证明，须记录核查等级、任意cue和歧义。最终输入固定平衡的合成／官方training replay，实际训练最多1024 episode。独立基础episode数量与增强数量分开记录，不靠重复颜色排列夸大数据规模。

## 固定训练与评价

- 原模型：`sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1`，保留16-token Qwen网格格式。离线SFT仅监督末query答案与EOS；逐token核对真实tokenizer的prompt前缀和mask。不能借用27B模板或让原TTT collator覆盖query-only标签。
- 单GPU LoRA rank32、alpha32、RS-LoRA、lr1e-5、microbatch1、accum4、max sequence8192；默认最多256 optimizer steps／1024 episode。来源最外层严格交替，来源内按family与长度桶预定次序，只走训练池一遍；离线SFT启用Unsloth梯度检查点，原逐题TTT不变。32步仅记录pilot快照，同一优化器继续，不择优checkpoint。
- 一个最终checkpoint；按预定步数／训练时间结束。训练进程含加载和合并最多3600秒，训练本身约2700秒。先落盘合并BF16普通checkpoint，训练进程退出后再重新加载，为每题创建原rank256 TTT adapter，避免离线adapter叠加或被逐题reset擦除。
- 同一个4×L4会话内，base与challenger各2个worker，独立相同任务队列；固定`PYTHONHASHSEED=26026`、原增强seed、原TTT／DFS／B4和1200秒task软检查。两臂共享评价截止，总会话约3小时，收尾留缓冲，实际预算见package manifest。
- 训练前冻结[16题／23query开发面板](../../experiments/026/panel-v1/manifest.json)：025任务结果4全对／1部分对／11全错，输入层次L_multi3/L_single3/M_multi2/M_single3/S_multi2/S_single3。种子在可行集合中确定选择，不按生成器可得性或新模型结果挑题；所有query保留分母。
- 025的哈希种子未固定，不能把新模型单跑相对旧15/58的±1解释为训练收益。此处使用新配对基线；同时记录搜索覆盖、raw/scored oracle、两槽新增／丢失与完整task。

## 晋级与停止

只有新配对结果最终两槽query净增严格为正，才固定这一个checkpoint进入[预留23题](../../experiments/026/panel-v1/confirmation-reservation.json)的一次确认。零或负结束本训练配方，不在16题上调学习率、步数或排序救结果。oracle增长而两槽持平不算比赛提分。

16题为按历史正确性分层的开发集合，不能推算全量公榜；23题仅未用于025策略选择，仍有历史公开推理／上游接触，均不称盲测。正式最好仍为30.14，33.89为外部精确V1历史公开成绩；本轮不承诺金牌或奖金。

## 实现和运行步骤

1. 固定面板、生成器源码与数据身份，手动核查CPU数据质量和query mask。
2. 新增`experiments/026/prepare_data.py`、`train_specialist.py`、`prepare_panel.py`、`prepare_paired.py`；生成第三方Notebook仅放`/private/tmp`，保留原V1来源和SHA。
3. 核对当前账号配额、精确上传源、runtime、模型与utility挂载，启动一次有界小实验；不启动全量训练或新架构。
4. 确认正常训练启动后停止查询，由用户通知完成；不设定时任务。完成后下载、冻结预测，再用固定官方参考评分。
5. 按实测晋级或结束，不以代码完成或loss下降宣称提分。

外部来源查阅日期2026-10-25：[原生成器](https://github.com/water-vapor/synth-rearc/tree/d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0)、[公开高分精确V1](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593)。原源SHA `71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d`。
