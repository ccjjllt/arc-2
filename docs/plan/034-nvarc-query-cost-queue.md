# 034 原NVARC全局任务队列的query成本优先

## 当前依据与目标

2026-10-29用户要求继续submission提高分数。先核实033正式57737364 COMPLETE30.56，精确V2／335137615绑定通过，成为当前已确认最好（比023高0.42）。本轮从033原源派生，不撤回其GPU NLL、不复活已结束的TTT64、融合、阈值或cache配方。

新检查的公开分支里，两个与033逐字相同，其余包含错误／不可访问版本、半步TTT、融合、独立DSL或多项运行变更，不能直接作为合格单因素。读取同队hidden产物的只读工具明确拒绝submissions.download，未绕过；当前仍无隐藏逐题覆盖证据。

## 单一改动与可否证假设

仅改starter全局任务入队顺序：计算每task所有可见train输入／输出格子及test输入格子总数，除以test query数，以精确Fraction升序排序，task_id破平。读取demo output属于题目可见示例；绝不读取test output。全部实际test任务保留exactly once。

该公式在派发CPU回放前固定，随后写本计划，没有据回放结果调系数。假设：同一128步TTT成本可被多query分摊，输入成本较小的query优先可能提高全局截止前的有效搜索覆盖。它不是隐藏准确率结论，也没有按答案或已知得失挑task优先级。

033的arc_solver／arc_loader／arc_decoder逐字保持；原4worker、sft139、128步TTT／n16／scheduler／seed、DFS0.2、B4、每task1200秒软检查及hidden全局42600秒不变。这不是022的公平硬预算改写，也不改变每个task内部的query／view顺序。

## 已做CPU回放与局限

仅读025 input-only40题／58query与原四worker闭合事件，用原task服务时间不变的反事实回放，未读答案或预测。预定1300／2600／5200秒三个预算，原队列→新队列得到搜索query分别14→22、26→34、48→54。全部工作makespan反而6451.351→6750.587秒（+4.64%）；不是全预算支配。如果隐藏运行已接近全扫，长任务后置可能有害。

输入cells/query与观测seconds/query Spearman0.7345；cells与TTT秒0.9334。基线worker分配复现，时间误差最大0.00496秒。这里只是025原NLL实现／历史成本模拟，不是033同预算实测，不包含模型加载成本，也没有检验新排序下的TTT／数值或正确数。全部结果和来源SHA记录在 `experiments/034/queue-cost-replay.json`，不扫描公式或挑预算。

## 普通验证与正式交付

最近精确zhuating余额1846.968配额秒，约923秒4L4墙钟，刷新10月31日08:00。准备一个普通运行与hidden共用的V1，CLI session最多840秒（规划1680配额秒）。ordinary只跑已曝光的2ba387bc与d35bdbdc、共2task／4query，global为Notebook开始后720秒；hidden读取全部actual test、global仍42600秒。普通队列实际d35→2ba，成本比300→2756。ordinary不读gold、不评分，不能据此宣称提分。

普通完成后需核：平台COMPLETE、固定V1及实际scriptVersion、各2题真实TTT128、每query4批启动／共16批、两题finished、无timeout／错误，求解结束不晚于720秒，原B4输出和源码身份有效。任一执行条件失败，不盲提交。CPU成员保持和普通运行仅是实现证据，正式submission作为该固定单因素策略的一次探索对比；实际公榜严格超过30.56才更新最好，否则结束该队列配方，不在同材料调公式／阈值／seed救结果。

通过后在用户已有授权下直接提交同V1，不另跑四题、不重新询问许可；记录真实ref、版本及分数。健康确认后按用户既有习惯停止日志与远端轮询，等完成通知，不创建自动监控。
