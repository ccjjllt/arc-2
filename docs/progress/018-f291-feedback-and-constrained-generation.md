# 018 — F-291反馈对照与受约束程序生成

日期：2026-10-19，北京时间。对应[018计划](../plan/018-f291-feedback-and-constrained-generation.md)。

**当前状态：A线GPU会话已推送运行中（v1 RUNNING，结果未取回）；B线CPU实现＋干跑完成，B-verify GPU会话待批。** 无训练、无提交、C6/981未动。

## 1 A线：冻结F-291一次反馈对照（GPU进行中）

- 冻结物（`experiments/018/frozen/`，SHA见`execution/push-a.json`）：v1原文`final.txt`(16e3c18e…)、`code.py`(cc808fdb…，本地`ast.parse`通过、不执行)、干净291题面(`task-291-clean.json`, 4demo/1query)。
- 驱动`experiments/018/aq/vdriver018a.py`（SHA`3e3d4545…`）：沿用017安装/服务8/C8/解析/执行/落盘链路，A线**零改动**（无约束补丁）。流程：执行器预检→GPU重跑冻结程序于4demo→`demo_feedback`构造反馈（断言4条、无人工补丁）→F-fix与R2两调用并发（同预算12288/thinking≤8192/seed 5103，R2避开v1的5102防复刻）→执行各自输出落盘。新增调用恰2次，预算上限24576；沉没成本P0-291 12288＋F-291 9842＝22130记入ledger。
- 推送：`ccjjllt/arc2-018a-f291-feedback-vs-restart` v1，会话2700秒；推送前ccjjllt剩20.11h（门槛约7.73h）。标题派生slug再次多出`-vs-restart`后缀（同016教训），以notebook URL slug为准，已记录。
- 待取回后本地评分：F-fix vs R2四元组＋EM；若F-fix query exact（旧组合没有）→开+4，否则暂停这条线。度量：每GPU小时新增正确题。

## 2 B线：受约束生成＋可靠原语（本地完成，GPU待验）

- `experiments/018/congen/opspec.py`（SHA`753ceb0c…`，stdlib only）：6操作（extract/filter/sort/transform/place/output），对象保留真实格点、bbox只派生；`validate_template`逐字拒绝未知op/多余键/自由文本/超长/超候选；`guided_schema()`同源生成（server尽力、CPU为准）；`search_template`上限32候选/10秒CPU、同一配置必须全过所有demo。
- 干跑（证据`congen/dryrun-report.json`，手造网格）：同色/多色提取、4/8连通、旋转约定、越界拒绝、shift搜索一次命中（dr=-2）、64候选拒绝/32候选放行、非法计划逐条拒绝。只证接线/格式。
- 外部依据：michaelhodel/arc-dsl README＋示例（2026-10-19查阅）：`objects(grid,univalued,diagonal,without_bg)`返回保留位置颜色的格点集，paint/fill/cover作用其上；本轮只移植子集，背景取众数色记为第一版局限。
- B-verify GPU会话（schema＋约束＋模板提示，1–2题验"可执行结构率"）待本地结论稳定后另批，不与A合并。`chat_once`的`extra_body`补丁只出现在B-verify vendored副本，A线副本与017一致（SHA`8970e7a7…`可直接复用，但独立存放待推送时核对）。

## 3 第3项（有界编辑）

本轮只留设计（见计划§5），排在B1/B2之后，不实现。
