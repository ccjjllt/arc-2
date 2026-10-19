# 018 — F-291反馈对照与受约束程序生成

日期：2026-10-19，北京时间。对应[018计划](../plan/018-f291-feedback-and-constrained-generation.md)。

**当前状态：A线v1–v3均在模型调用前以`signal_15`结束；v4已改由teammateb身份推送，当前记录为RUNNING，结果未取回。** B线CPU实现＋干跑完成，B-verify驱动已就绪但未推送。无训练、无正式比赛提交、C6/981未动。

## 1 A线：冻结F-291一次反馈对照（GPU进行中）

- 冻结物（`experiments/018/frozen/`，SHA见`execution/push-a.json`）：v1原文`final.txt`(16e3c18e…)、`code.py`(cc808fdb…，本地`ast.parse`通过、不执行)、干净291题面(`task-291-clean.json`, 4demo/1query)。
- 驱动`experiments/018/aq/vdriver018a.py`（SHA`3e3d4545…`）：沿用017安装/服务8/C8/解析/执行/落盘链路，A线**零改动**（无约束补丁）。流程：执行器预检→GPU重跑冻结程序于4demo→`demo_feedback`构造反馈（断言4条、无人工补丁）→F-fix与R2两调用并发（同预算12288/thinking≤8192/seed 5103，R2避开v1的5102防复刻）→执行各自输出落盘。新增调用恰2次，预算上限24576；沉没成本P0-291 12288＋F-291 9842＝22130记入ledger。
- 推送：`ccjjllt/arc2-018a-f291-feedback-vs-restart` v1，会话2700秒；推送前ccjjllt剩20.11h（门槛约7.73h）。标题派生slug再次多出`-vs-restart`后缀（同016教训），以notebook URL slug为准，已记录。
- **v1失败**：notebook 2640秒截止时`signal_15`，run.json仅elapsed 2639.7s、零调用、零驱动stdout。无install/server日志留存，定位为首行驱动日志之前的静默stall（预安装阶段），加固：驱动启动/分阶段日志、vendored probe的nvidia-smi加30秒超时、会话延至3600秒。教训：驱动首行必须先打日志再做任何事。
- **v2失败**：同样`signal_15`（notebook 3540秒截止，elapsed 3539.8s），零调用。但这次有行日志：**同一cell内的驱动被执行了两次**（"driver start"+"versions recorded"各出现两次，相隔0.2秒），之后双双沉默。两个驱动并发跑`pip install --target`到同一目录、抢写同一批文件，是最可能的stall原因；v1零日志则仍无法定位（无stdout文件留存）。加固（v3）：`driver.lock`排他锁（第二个实例安静退出rc=0）、全部`log()`镜像到`driver-stdout.log`文件、每安装阶段后落盘run.json、会话延至5400秒（与跑通的017同规格）。
- **v3失败**：同样`signal_15`（elapsed 5339.5s），零调用。**定位精确**：单"driver start"+单"versions recorded"（无双执行、无lock busy），之后89分钟沉默。`driver-stdout.log`镜像与run.json一致——hang点在`record_versions`返回之后、冻结SHA循环之内，即3个5KB本地文件的stat+read。直线上无循环、无网络调用，89分钟零进展而SIGTERM处理正常，最可能的是宿主/存储卷级stall（可中断睡眠，故信号能响应；结束时save()正常说明卷已恢复），非代码死循环。v1（无启动日志）与v2（双执行+旧probe无nvidia-smi超时）症状均与同一位置兼容。3次失败共耗约2h配额（qingyu1232 used 2.04→12.69h含用户SFT约10.6h，剩17.31h）。
- **v4加固**：驱动内置60秒看门狗（`driver-watch.log`记phase+`faulthandler.dump_traceback`），phase标记打到pip/import/template/serve/preflight/calls；锁+镜像+分阶段落盘保留。驱动SHA`b6ea61e4…`。另：旧内核SaveKernel持续409，已改推新内核；推送期间API抖动（SSL EOF、list Not found）+`Maximum batch GPU session count of 2`，待槽位释放重推。
- **B-verify驱动就绪（待推）**：`experiments/018/bv/vdriver018b.py`——291/db各1次guided调用（4096上限/thinking 8192/seed 5201/5202），final走JSON解析→`validate_template`→`search_template`（≤32/10秒）→同配置跑test得query；服务端拒收记`server_rejected`停止。`enable_in_reasoning=False`放顶层（按用户核对的固定版本源码；若字段名有误，服务端400会如实记拒收、不静默重跑）。请求链已本地验（假SSE劫持urlopen）：payload含guided schema＋`enable_in_reasoning:False`＋thinking预算，非法extra_body本地拒收，模板prompt约3.4K字符。
- **v4已推送（teammateb身份，RUNNING）**：探针`arc2-018a-probe-sanity` v1秒级COMPLETE（L4健康、文件落盘），平台链路正常，遂推v4为`fix-restart-v3` version 2（驱动`b6ea61e4…`，5400秒）。推送前该身份剩23.85h（门槛9.45h）。见`execution/push-a.json` v4条目。
- **诊断探针**：`experiments/018/probe/`（hello+nvidia-smi+写文件）已COMPLETE，约14秒，L4与文件落盘正常；排除平台基础设施故障后推送v4。
- 待取回后本地评分：F-fix vs R2四元组＋EM；若F-fix query exact（旧组合没有）→开+4，否则暂停这条线。度量：每GPU小时新增正确题。

## 2 B线：受约束生成＋可靠原语（本地完成，GPU待验）

- `experiments/018/congen/opspec.py`（SHA`753ceb0c…`，stdlib only）：6操作（extract/filter/sort/transform/place/output），对象保留真实格点、bbox只派生；`validate_template`逐字拒绝未知op/多余键/自由文本/超长/超候选；`guided_schema()`同源生成（server尽力、CPU为准）；`search_template`上限32候选/10秒CPU、同一配置必须全过所有demo。
- 干跑（证据`congen/dryrun-report.json`，手造网格）：同色/多色提取、4/8连通、旋转约定、越界拒绝、shift搜索一次命中（dr=-2）、64候选拒绝/32候选放行、非法计划逐条拒绝。只证接线/格式。
- 外部依据：michaelhodel/arc-dsl README＋示例（2026-10-19查阅）：`objects(grid,univalued,diagonal,without_bg)`返回保留位置颜色的格点集，paint/fill/cover作用其上；本轮只移植子集，背景取众数色记为第一版局限。
- B-verify GPU会话（schema＋约束＋模板提示，1–2题验"可执行结构率"）待本地结论稳定后另批，不与A合并。`chat_once`的`extra_body`补丁只出现在B-verify vendored副本，A线副本与017一致（SHA`8970e7a7…`可直接复用，但独立存放待推送时核对）。

## 3 第3项（有界编辑）

本轮只留设计（见计划§5），排在B1/B2之后，不实现。
