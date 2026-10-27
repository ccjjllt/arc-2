# 032 直接提交最高分公开Notebook

2026-10-27用户明确要求直接使用分数最高的Notebook做一次submission。本次是直接提交授权，覆盖先前“不重提未改基线”的研究安排；不复活03164步或其他失败配方。

本次Kaggle CLI实时查询competition代码按scoreDescending排序，第一为 `koushikrudra/failed-in-aimo`。已核验高分为精确V1／scriptVersion312063593的外部33.89；冻结源码SHA `71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d`。不把标题、latestV2或作者队伍分数代替该版本，不承诺重提取得同分。

先核今日名额、近期pending及版本身份，然后用用户指定的Kaggle CLI与zhuating认证，尝试直接提交原owner／slug的V1和 `submission.json`。客户端没有owner限制，服务端是否允许尚未确认；不得把请求发出当作受理成功。固定V1，不重复尝试未知结果。

若服务端明确拒绝非owner，则使用已完成、保留该原高分求解逻辑且仅修runtime阻塞的自有023精确V1／361045383正式提交，无需新普通运行。应明确该副本历史真实成绩30.14，不能称原样源码或已复现33.89。若原请求结果未知，先查submission列表恢复身份，不能盲目再次创建。

保存真实受理ref、精确Notebook／scriptVersion绑定和初始状态；分数未返回则明确待评测。只做一次用户要求的正式提交，不建自动监控、不额外启动GPU研究。
