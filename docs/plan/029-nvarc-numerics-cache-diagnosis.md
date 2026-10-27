# 029 NVARC数值计算与缓存恢复定位

日期：2026-10-26。用户确认继续后启动这一轮有限诊断，承接[028实际结果](../progress/028-nvarc-dfs-throughput.md)。**V1／362548497已COMPLETE；2026-10-27复核确认差异在恢复前已出现，受检0／1／13点恢复结果均一致。结束本轮定位，不复活028，也不作为提分方案；028网格替换的充分原因仍未证明。**

## 为什么只做这一项

028固定16-token proxy已经结束，不重新挑长度、seed或阈值。其d35原A／B序列与NLL完全一致，proxy却稳定替换1个网格；长题候选相同时只省1.57%调用耗时，没有晋级同预算或正式提交。原023／30.14保持正式对照。

现有记录缺少逐位置logits与KV，无法区分：prompt从794变810后、尚未执行DFS就产生的数值变化；prefill和单token fastdecode之间的变化；以及清paged后snapshot恢复的变化。只读代码没有找到可证明的索引／回溯错误，不能先认定某一种原因。源码依据仍为028已冻结的原模型、utility、实际attention源码及[运行时证据](../../experiments/028/cache-source-evidence.json)，不换backend或权重。

## 固定输入与控制

只取发生差异的历史开发task **d35bdbdc**。这是有意选择的故障复现题，不称新确认集。源文件保留3个query，实际只测首query的前两批原增强。输入2119字节、SHA **af5e81c69f7fcd38e46866c55a3dd4348df25acc7c27e4e9b0ffb2d298583489**；全部query只有input，known demos仍有output。见[输入manifest](../../experiments/029/input-v1/manifest.json)。

一次原sft139 TTT rank256／128步、增强seed1、Trainer／PEFT seed42、eval seed2、hash27027，所有比较共用同一内存adapter，保留原`torch.inference_mode()`及`for_inference`状态。新的TTT不保证逐字复现上次GPU浮点轨迹，但本轮各条件始终配对，不以历史adapter替代当前参考。

首批原DFS产生草稿，按原返回顺序选首个合法candidate，映射到第二批各view；与028规则相同。不使用正确答案、NLL排名或ID白名单来挑草稿。若本次无草稿，明确记录该题无法继续固定token诊断，不另挑题救结果。

## 三项诊断，运行前固定

1. **初始计算。** prompt-only无past前向重复两次，建立root logits／prefix KV重复漂移参照；再比较prompt＋最多16个固定draft token的无past前向。比较同一prefix长度的各层K/V和root最后logit，记录差异是否在任何skip／restore之前出现。
2. **固定路径计算。** 固定相同16token，原逐tokendecode重复两次，与一次完整prefill的逐位置结果对齐。root logit预测token0，消费第d个token后预测d+1，保留17个logit位置和16token累计NLL、ARC token次序；不沿新logits重新选路径。
3. **恢复。** snapshot必须由**原逐tokendecode**产生并deep clone，关闭skip；固定深度0／1／13，比原位继续与清paged恢复后预测相同下一token，双方均有重复参照。先核对snapshot与恢复后内部paged的prefix内容及有效长度，再比较下一logit和KV。这三个深度是预先声明的机制位置，不冒称028真实三次祖先恢复深度。

保留root／逐位置小logits和tokens，KV按层、按块统计，避免一次创建全部FP32差值或保存GB级cache。每个阶段及时原子写报告，未知cache类型、状态不符或预算不足时明确失败并保留诊断，不静默跳过。

## 实现与预算

- 自有`experiments/029/numerics_diagnostic.py`提供运行时观测，不实现新求解器；`prepare_diagnostic.py`从冻结025／027来源包装。
- 第三方Notebook和求解源码仅生成到`/private/tmp/arc2-029-cache-diagnostic-v1`。原TTT、DFS函数保留，只有TTT后调用本诊断入口。
- 单worker，Kaggle L4×4会话最多**2400秒／40分钟**，全局2220秒、进程组硬截止2340秒，包含复制、加载、TTT及收尾。启动前读取实时配额和reserved，不能用旧快照推断可运行。
- 不训练新离线权重、不执行B4、不读取query gold、不计准确率、不创建competition submission。正常加载后停止远端查询，按既有方式等待用户完成通知。

## 解释与停止边界

如root／prefix KV已不同，只能证明不同计算路径在DFS前引入差异，不能排除恢复还存在问题，更不能单凭它证明028最终网格替换的充分原因。恢复结果不同也需结合paged内容与双方重复漂移，不能直接称数据丢失。

本轮完成后只给阶段定位结论。单题诊断不产生排序、oracle或得分晋级；无论结果如何，都不恢复028为正式方案。若三项仍无法定位，记录未确认并结束这一诊断，不扩成长周期runtime工程。后续候选生成方案另行冻结单因素和正式预算，以实际得分证据评判。
