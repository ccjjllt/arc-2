# 034 输入成本／query任务队列

从033正式30.56的精确源派生，唯一hidden改动是starter入队按visible cells/query升序、Fraction与task_id稳定破平。原solver／loader／decoder逐字不变；128TTT、DFS0.2、B4、4worker、hidden42600秒和task1200soft保持。

**正式submission 57787438已受理，当前PENDING。** 2026-10-30 08:46:34北京时间由 **zhuating／ccjjllt** 提交原候选V1，文件submission.json；服务端精确绑定[scriptVersion363524873](https://www.kaggle.com/code/zhuating/arc2-034-visible-cell-cost-per-query-queue?scriptVersionId=363524873)。公／私榜未返回，当前正式最好仍033的30.56。今日已用1、剩0，总15。

普通V1已COMPLETE，全新子代理独立复核通过：两题各TTT128、4query各4批启动、2finished，4个L4 worker正常结束，无timeout／Traceback／OOM。求解在Notebook开始后693.280秒完成，早于720秒截止；B4导出完成727.621秒。全部9个cell与冻结包一致，四个实际runtime源仅多IPython末尾一个LF；48项输出URL均为363524873。schema120题／172query中仅本轮4query非占位，未读gold、没有新准确率或提速证明。

实际入队d35→2ba，visible-cell成本900／2756、query3／1、优先值300／2756。worker异步就绪，rank和训练日志先后不能用于证明入队顺序。CPU历史成本回放部分预算覆盖提高，但完整makespan+4.64%；正式分数严格超过30.56才保留，否则结束固定队列配方，不调公式／seed／阈值救结果。

平台按title创建实际slug **zhuating/arc2-034-visible-cell-cost-per-query-queue**；不要沿用原metadata请求的arc2-034-queue-query-cost，也不要重跑、重命名或重复提交V1。

证据与交接：

- [ready-v1.json](ready-v1.json)、[计划](../../docs/plan/034-nvarc-query-cost-queue.md)、[进展](../../docs/progress/034-nvarc-query-cost-queue.md)
- [普通实际源](ordinary-v1-source/)、[普通实际输出](ordinary-v1-output/)
- [手工完成核验](execution/v1-manual-completion-check.json)、[独立复核](execution/v1-independent-review.json)
- [正式回执](execution/formal-v1-receipt.json)、[精确绑定](execution/formal-v1-list-binding.json)、[受理摘要](execution/formal-v1-summary.json)
- [CPU回放](queue-cost-replay.md)、[公开分支审计](public-audit-summary.json)

本次普通实耗1530.456配额秒，zhuating剩316.512配额秒、reserved0；10月31日08:00北京时间刷新。未启动其他GPU或自动监控。

## 用户通知正式分数完成后的人工接续

只读一次正式列表；不重复提交、不主动轮询：

```bash
/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python experiments/032/capture_submission_cli.py experiments/034/execution/formal-v1-result-read.json competitions submissions arc-prize-2026-arc-agi-2 --page-size 20 --format json
```

选ref57787438，核账号zhuating／队伍ccjjllt、submission.json、URL精确363524873以及实际终态／错误／公私榜。公榜严格>30.56才更新正式最好；否则结束该固定配方。私榜空值是未返回，不是0。032／033隐藏产物下载曾明确Permission submissions.download denied，不换接口绕过，缺失的隐藏覆盖不能推断。
