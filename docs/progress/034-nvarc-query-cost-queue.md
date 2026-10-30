# 034 原NVARC query成本队列进展

2026-10-29用户要求继续优化并提交。已先确认033正式30.56（+0.42），保留为当前基线。新公开分支审计没有找到合格的已完成新单因素；具体排除记录另归档。本轮自主改动仅starter全局任务队列，原033三个求解源逐字保持。

公式在CPU回放前冻结：visible cells（train input/output＋test input）／query数，Fraction升序、task_id破平。仅读025输入与闭合成本事件，1300／2600／5200秒获搜索query14→22、26→34、48→54，但完整makespan增加4.64%；未读gold、没有准确率或隐藏提分结论。全部回放包括风险见 `experiments/034/queue-cost-replay.md`／JSON。

实际候选包 `/private/tmp/arc2-034-queue-candidate-v1`，50190字节，SHA `d748c29ea86260a2827ca91a9839e96f55910b579d9e1087fdbd163ca80091af`。手工AST、120task成员exactly-once、ordinary2题4query顺序d35→2ba通过；普通不读gold，隐测全部actual test／42600秒。CPU检查不是GPU执行或提分证明。此时尚未上传、未启动GPU或正式提交；独立包审查后才启动普通验证。

## 独立审查与普通V1上传

全新只读包审查通过：actual Notebook SHA一致，全部源码AST通过，033三个求解源逐字保持；实际QUEUE AST对02540题58query成员无缺失重复，ordinary顺序d35→2ba，2task／4query。launcher正则对真实031日志匹配8题TTT128、8finished、46batch行；它检查的是批启动而非搜索树穷尽，不将smoke当准确率／速度证据。

最后quota实查zhuating used106153.032、reserved0、available1846.968秒，足够840秒4L4上限（规划1680配额秒）。CLI成功上传V1，但平台警告title/id不一致并实际创建 **zhuating/arc2-034-visible-cell-cost-per-query-queue**，后续以真实ref为准，不能沿用请求的arc2-034-queue-query-cost。请求metadata仍留作历史，未重命名／重跑版本。

随后CLI current_version_number=1、全部cell源与冻结包逐字匹配，status为QUEUED。actual scriptVersionId尚未取得，正式submission仍无。启动后真实硬件、训练步数、16批与完成结果待核，最近日志读取尚未见worker输出；不能把QUEUED写成训练成功。

后续一次状态已为RUNNING。启动日志实际打印queue_priority_manifest：源为官方challenge SHAe7c62a4b...，2task／4query、d35→2ba，visible-cell成本900／2756、query数3／1、比值300／2756，均与冻结策略匹配；未读query outputs。rank0已实际启动，日志确认为NVIDIA L4、Unsloth2025.9.7／Transformers4.55.4／Torch2.8.0+cu128，未见Traceback／OOM。尚未证实其他worker全部加载、两题128步及最终完成，不能把初始健康写成普通验证通过。

启动核对后停止远端日志与查询，等待用户结束通知，不自动监控。全部实际接续和直接正式提交命令已写034 README，使用真实title-derived slug；当前没有034比赛submission，今日名额仍保留，正式最好03330.56。

## 2026-10-30 普通完成与正式提交

用户通知跑完后，CLI实查V1为COMPLETE。下载currentVersion1全部9个cell与冻结Notebook逐字一致；48项输出URL全部绑定实际scriptVersion **363524873**。四份实际runtime源均为冻结writefile body加一个IPython终止LF，solver／loader／decoder保持033实现。原输入SHA、queue-manifest和prediction SHA全部闭合；关键下载已保存于 `experiments/034/ordinary-v1-source/`、`ordinary-v1-output/`，版本响应与核验记录保存在 `execution/`。

实际4个worker均为L4并正常结束。两题TTT均完成128步，2ba训练249.0453秒、d35训练84.0787秒；4个query各启动4批，合计16批，两题均finished（371.2／402.7秒），没有timeout、Traceback或OOM。launcher返回0、运行650.576秒，求解于Notebook开始后693.280秒结束，早于720秒截止26.720秒；B4导出完成于727.621秒，须与求解截止区分。输出schema为120task／172query，仅本轮4个query有非占位预测，所有两槽网格格式有效；未读gold、未测准确率或提速。

队列入队d35→2ba、成本比300→2756，与冻结公式一致。rank1处理d35、rank0处理2ba，worker异步就绪使rank或训练日志先后不等于入队顺序，不据此宣称队列反转或完成顺序受控。全新子代理 `r034_completion_review_new` 独立复核通过，无实现阻塞。普通实耗1530.456配额秒；zhuating剩316.512配额秒、reserved0，10月31日08:00北京时间刷新，未新开其他GPU。

提交前CLI确认已有14次正式提交、今日剩1、无pending。**2026-10-30 08:46:34北京时间已正式提交同一V1**：ref **57787438**，账号 **zhuating**、队伍 **ccjjllt**，文件submission.json；服务端URL精确绑定363524873。已受理、当前PENDING，公／私榜尚未返回，今日剩0、总15。回执见 `execution/formal-v1-receipt.json`，精确绑定见 `formal-v1-list-binding.json`／`formal-v1-summary.json`。

当前正式最好仍033的30.56。034只有实际公榜严格超过30.56才晋级，否则结束固定队列配方；不根据smoke或成本回放宣称提高分数，不重复提交或调公式救结果。停止远端查询，等待用户分数完成通知后仅读一次CLI终态及版本绑定；不创建自动监控。
