# 028 NVARC DFS提速与同预算搜索核查

日期：2026-10-26。承接[027正式退分复盘](../progress/027-nvarc-ttt-view-ensemble.md)。用户进一步授权继续优化后，机制proxy、面板与手动配对已完成，V2／362511155已COMPLETE。**一题候选稳定且DFS调用耗时减1.57%，一题表面减6.63%但候选池变化，结束固定16-token配方，不晋级同预算或正式提交。** B4、准确率与提分未测量。见[实际进展](../progress/028-nvarc-dfs-throughput.md)。下文保留原研究目标，实际pilot范围以同号progress为准。

## 研究背景与唯一方向

027正式submission57634436已COMPLETE／公榜29.72，相比023的30.14下降0.42，结束其作为当前提分方案的晋级。公开C23的5→6/34只新增1个query，且base第二槽命中为0，未检验出替换正确第二槽的风险。正式隐藏运行的base备份、extra覆盖和决策记录不可得，不能把退分指定归因于融合或余量不足。继续保留023作为正式对照，不在C23挑seed／改融合救结果。

手工汇总025与027已有events，DFS占TTT／DFS／重评分三个阶段累计worker工作量约60%，重评分包络只占7%—9%。这是host事件区间求和，不是墙钟关键路径或CUDA精确profile；但它支持先研究DFS，而不是假定重评分缓存能解决双遍预算。数据、源文件SHA和口径见[复盘JSON](../../experiments/027/postmortem-20261026.json)。

下一轮只研究原sft139的**已知候选路径复用与分块验证**。先保持原搜索树和预算，检验真实吞吐及输出变化；不同时降DFS概率阈值、不加TTT seed、不改B4、不换权重。不宣称提速必然提分；如果原搜索已完成，提速可能只省时间。

## 原始依据与可读实现

2026-10-26复查[ARChitects 2025技术报告](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)：其自回归方案报告16—32 token speculative DFS与prefix缓存提速。报告倍率不等于NVARC4B实测；目前未在该报告仓库定位到对应2025批量DFS实现。

可读参考为[2024 inference_tools.py固定commit](https://github.com/da-fr/arc-prize-2024/blob/d3ac3f6ebf6fb609bfdc782561ee99977ca35d95/training_code/inference_tools.py)，`explore`／`dfs`利用已知guess一次前向的logits，偏离该路径时恢复逐token前向，并裁剪KV。文件头为Apache-2.0。它明确不支持batch，不能原样换入NVARC4view batch；参考机制与移植改动必须注明来源。

本项目原始位置为`experiments/021/reference/failed-in-aimo-v1.source.py`：`turbo_dfs`约496行、`inference_turbo_dfs`约558行、worker解码约818行。该文件保持冻结。第三方修改源码只生成到`/private/tmp/arc2-028-dfs-pilot/`，不向仓库复制一套求解器。

## 拟修改与实现步骤

1. 从已核对的原V1派生一次base-only小实验。固定sft139模型、原utility和已成功运行的环境；TTT rank256／128步、训练增强seed1、Trainer／PEFT seed42、eval增强seed2、原B4均保留。两种DFS共享**同一次TTT的同一内存adapter**、完全相同的实际prompt tokens和四view batch。实验两臂统一`PYTHONHASHSEED=27027`，只为控制本轮配对；不能称其与023未固定hash的历史输出严格相同。
2. 先记录真实cache类型、有效长度与每次回溯前后长度、forward次数／token数，使用CUDA同步和events补测prefill、decode、CPU拷贝及分支控制。同步测量单独一遍，不把新增同步开销混进速度主比较。保留原版重复运行漂移作为参照。
3. 唯一草稿规则：每个query取**更早已完成DFS批次按原返回顺序的第一个合法candidate**，保存原坐标grid；用`ArcDataset.forward_mod(grid, subkey)`映射到当前view，经原formatter／tokenizer得到路径。没有此前候选时直接走原DFS。不根据gold、候选分数或task ID选草稿，不额外调用模型生成草稿。
4. 每次最多预计算该路径后续16 token。第一token概率来自当前prefix最后logit；后续来自前一token位置的target logits。只要当前实际前缀偏离草稿，该行预计算值立即失效。草稿只是减少相同路径上的forward，**每个仍满足原累计NLL阈值的替代分支都必须枚举**。
5. 保持原每行按累计NLL排序的遍历顺序；不能为了利用草稿而优先访问原本更靠后的分支。保持`score < -log(0.2)`、原ARC_TOKENS、EOS15、最大输出token与每次540秒DFS。机制ABBA多次重复测量受本次5400秒会话约束，不作为原1200秒task预算实证；后续端到端评价保留1200秒task／42600秒全局检查。当前pilot不启用新阈值。
6. 明确处理四行batch的不同接受长度、EOS、无候选PAD行及RoPE position。离开预计算路径或回溯时恢复**实际Unsloth cache的全部可变状态**，不能假定它是普通tuple或仅裁剪一个可见长度字段。无法可靠恢复则停止该实现，不默认为等价。

只新增必要的包装、观测与手动实验脚本，不新增测试文件、测试目录或框架。

## 最小手动实验

先从`experiments/027/development-replay.json`的`025_development24_excluding026`固定开发集合取题，排除026十六题和C23。只按既有query输入长度与DFS时长选择短／中／长各一题；必须在新GPU运行前把ID、输入SHA、batch和选择口径写入manifest，不按答案、oracle或新方案得分选题。这是开发机制面板，不称未见验证集。

每题先做一次原128步TTT，原版与改版在同一adapter上交错运行相同batch。最早一批原DFS生成的候选可作为后续批次的草稿；不把此前独立TTT产生的候选伪装为当前进程已有候选。无草稿的fallback也记录。优先比较少量不会触540秒的batch，以便比较完整树；会触截止的batch单独记录候选先后次序和覆盖，不能声称集合完全一致。

记录实际GPU型号／显存、环境版本、权重与源码SHA、同步测量开关、冷启动／编译时间、adapter身份、batch tokens SHA、草稿来源批次、forward次数、target验证token、接受长度、NLL差异、合法候选序列与网格、B4两槽、真实耗时及峰值显存。原版／改版交错顺序在运行前固定，预热排除规则同时冻结。pilot人工设置不超过2小时会话上限，实际启动前读取可用配额；不以旧配额快照保证可执行。

## 晋级与停止

- cache回溯、位置或分支遗漏有任何未解释问题，停止；不以“跑得快”代替求解正确性。
- 完整树批次应逐序列核对候选与原累计NLL；batch数值变化必须同原版重复漂移一起报告。若出现正确性／排序行为变化，按新算法记录，不能称纯等价提速。
- 唯一改版在冻结面板没有稳定实测吞吐收益时结束，不扫描草稿长度／阈值救结果。32token、更多views和重评分缓存均后置。
- 机制通过后才设计一次固定同预算、同hash、base-only配对。需看所有计划query的覆盖、raw／scored oracle、最终两槽新增／丢失与截止前缀；公开诊断与正式成绩分开。C23已用过，不再作为新确认集。
- 之后如研究更深阈值，另编号冻结单一因素；不把提速和阈值同时改动后无法归因。正式提交仍以实际配对证据和可部署预算为前提，不直接重提027或改用其extra单臂。

## 当前交接状态

已完成：027实际结果归档、已有阶段事件CPU统计、C23第二槽风险覆盖复盘、三题面板冻结、16-token已知路径proxy及手动同adapter ABBA包装。实际Unsloth不支持multi-token withpast，本轮改用无past prompt+16token prefill、不可变snapshot及原DFS callback；尚未实现后续连续分块。V1仅封装日志变量错误、未进入TTT；V2完整配对已完成。稳定候选题调用耗时减1.57%，d35候选变化不能称等价提速，按预定条件结束固定16-token配方。不扩长度／seed／阈值救结果，不进入同预算或正式提交。具体版本和SHA见同号progress；cache恢复与数值形状偏差的因果定位仍未完成，B4及真实提分未测量。
