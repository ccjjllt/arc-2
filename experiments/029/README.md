# 029 NVARC数值与缓存恢复诊断

**2026-10-27复核：V1／362548497已COMPLETE，三阶段完整。差异在恢复前已出现；受检三个恢复点完全一致。结束此诊断，不复活028或宣称提分。** [完整汇总](diagnostic-v1-summary.json) · [工件SHA核查](execution/artifact-verification-v1.json)

[计划](../../docs/plan/029-nvarc-numerics-cache-diagnosis.md) · [进展](../../docs/progress/029-nvarc-numerics-cache-diagnosis.md)

2026-10-26 23:50北京时间平台接受`zhuating/arc2-029-nvarc-cache-diagnostic` **V1**，最长40分钟、一个worker。实际包SHA `57613fc78197fbdf34c24ef9b2e26d343afca94e4554f20c5c18411a50771e4d`，见[manifest](diagnostic-v1-package-manifest.json)与[回执](execution/diagnostic-v1-launch.json)。完整结果／实际scriptVersion仍待取得。

只对028发生变化的历史题d35bdbdc，在同一原TTT adapter上比较三项：prompt-only与＋16的root／prefix KV；固定16token逐步decode与full-prefill；原逐token snapshot在深度0／1／13的原位／清paged恢复。每类有重复控制，不换seed、草稿或阈值，不读gold、不执行B4／准确率或比赛提交。

输入保留3query、实际只测首query，故障题的选择不称新验证。KV按层分块统计，不落大tensor；tokens／logits和指标保存单个主报告。本次定位已完成，只能确认受检计算路径差异，不证明028候选替换的充分根因或提分，028固定配方保持结束，正式最好仍30.14。

正常启动后停止查询，用户通知完成时人工执行`/private/tmp/arc2-029-download-completed-v1.py`，先核对真实V1及产物，再解释三阶段差异。全部状态入口在`execution/`，不要用kernelId代替scriptVersion，也不要从诊断数值直接推断正确率。

23:57:26北京时间已确认V1全部cell源一致、原模型指纹和挂载正确、单worker两片checkpoint加载完成，无Traceback；TTT完成及诊断阶段尚未观察。见[启动健康](execution/startup-health-v1.json)与[接续参数](diagnostic-v1-ready.json)。已停止远端查询，无后台监控。

用户通知后已核对真实版本362548497、15项工件和66次诊断forward。结束runtime支线，下一步回到原NVARC候选覆盖／最终两槽配对研究；当前没有启动新GPU或比赛提交。
