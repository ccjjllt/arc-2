# 018 — F-291反馈对照与受约束程序生成

日期：2026-10-20，北京时间。对应[018计划](../plan/018-f291-feedback-and-constrained-generation.md)。

**当前状态：A线v6完成，F-fix在291得到exact，R2截断；按门控规则触发+4后续研究。** B线CPU实现＋干跑完成，B-verify驱动已就绪但未推送。无训练、无正式比赛提交、C6/981未动。

## 1 A线：冻结F-291一次反馈对照（GPU已完成）

- 冻结物（`experiments/018/frozen/`，SHA见`execution/push-a.json`）：v1原文`final.txt`(16e3c18e…)、`code.py`(cc808fdb…，本地`ast.parse`通过、不执行)、干净291题面(`task-291-clean.json`, 4demo/1query)。
- 驱动`experiments/018/aq/vdriver018a.py`（SHA`3e3d4545…`）：沿用017安装/服务8/C8/解析/执行/落盘链路，A线**零改动**（无约束补丁）。流程：执行器预检→GPU重跑冻结程序于4demo→`demo_feedback`构造反馈（断言4条、无人工补丁）→F-fix与R2两调用并发（同预算12288/thinking≤8192/seed 5103，R2避开v1的5102防复刻）→执行各自输出落盘。新增调用恰2次，预算上限24576；沉没成本P0-291 12288＋F-291 9842＝22130记入ledger。
- 推送：`ccjjllt/arc2-018a-f291-feedback-vs-restart` v1，会话2700秒；推送前ccjjllt剩20.11h（门槛约7.73h）。标题派生slug再次多出`-vs-restart`后缀（同016教训），以notebook URL slug为准，已记录。
- **v1失败**：notebook 2640秒截止时`signal_15`，run.json仅elapsed 2639.7s、零调用、零驱动stdout。无install/server日志留存，定位为首行驱动日志之前的静默stall（预安装阶段），加固：驱动启动/分阶段日志、vendored probe的nvidia-smi加30秒超时、会话延至3600秒。教训：驱动首行必须先打日志再做任何事。
- **v2失败**：同样`signal_15`（notebook 3540秒截止，elapsed 3539.8s），零调用。但这次有行日志：**同一cell内的驱动被执行了两次**（"driver start"+"versions recorded"各出现两次，相隔0.2秒），之后双双沉默。两个驱动并发跑`pip install --target`到同一目录、抢写同一批文件，是最可能的stall原因；v1零日志则仍无法定位（无stdout文件留存）。加固（v3）：`driver.lock`排他锁（第二个实例安静退出rc=0）、全部`log()`镜像到`driver-stdout.log`文件、每安装阶段后落盘run.json、会话延至5400秒（与跑通的017同规格）。
- **v3失败**：同样`signal_15`（elapsed 5339.5s），零调用。**定位精确**：单"driver start"+单"versions recorded"（无双执行、无lock busy），之后89分钟沉默。`driver-stdout.log`镜像与run.json一致——hang点在`record_versions`返回之后、冻结SHA循环之内。（**2026-10-20复核更正**：原归因"宿主/存储卷级stall、非代码死循环"不成立；根本原因是`sha256_file`中写为`for ch in iter(lambda: h.read(1 << 20), 1):`，文件读完返回`b""`永不等于整数`1`，造成确定的无限空读死循环；且驱动中`VLLM_WHEEL_SHA256`被截断为62位，修完循环后仍会匹配失败。v6仅因将该哈希放入后台线程并在300秒超时后跳过才得以继续运行，`driver-watch.log`显示该后台线程在模型调用期间仍停留在同一循环中。）3次失败共耗约2h配额（qingyu1232 used 2.04→12.69h含用户SFT约10.6h，剩17.31h）。
- **v4加固**：驱动内置60秒看门狗（`driver-watch.log`记phase+`faulthandler.dump_traceback`），phase标记打到pip/import/template/serve/preflight/calls；锁+镜像+分阶段落盘保留。驱动SHA`b6ea61e4…`。另：旧内核SaveKernel持续409，已改推新内核；推送期间API抖动（SSL EOF、list Not found）+`Maximum batch GPU session count of 2`，待槽位释放重推。
- **B-verify驱动已编写（存在阻断缺陷，严禁原样推送，转入019修复）**：`experiments/018/bv/vdriver018b.py`原设计为291/db各1次guided调用（4096上限/thinking 8192/seed 5201/5202）。**2026-10-20复核发现四重阻断**：① `sha256_file`同样存在哨兵`1`死循环与62位wheel哈希截断，会在读题面时直接卡死；② 请求体使用嵌套`extra_body.guided_json`与请求级`enable_in_reasoning`，而固定版本vLLM实际协议字段为顶层`structured_outputs: {"json": schema}`，未知字段会被静默忽略；③ 总输出上限4096小于thinking上限8192，长思考会先撞总上限；④ `opspec.py`的`place_objects`固定创建输入尺寸`H×W`画布，无法表达291和db的变尺寸输出，且`guided_schema()`仅允许标量、未放行`choices`。必须按[019计划](../plan/019-submission-and-first-draft-pipeline.md)先完成CPU修复。
- **v6成功：F-fix在291拿下旧组合没有的exact**（`complete_with_evidence`，elapsed 1706.8s，其中安装加载1193.8s占约70%，生成波次约507s；无preempt，执行器预检过，wheel哈希因后台线程300秒超时跳过而未阻塞主线程）。冻结重跑3/4（与v1一致），反馈459字符纯执行结果、无人工补丁。F-fix：stop/program_ready/4demo全过/query合法，9596 token；R2：12288截断、final通道输出12260字符分析文字而无代码围栏、无程序。本地比对官方291答案确认逐格exact；程序143行通用写法（找0角定框方向、内区提取），修的正是F-291认错框的问题。成本分三笔独立记账：本次修复本身**9,596 token**；加回历史起始程序（P0-291 12288＋F-291 9842＝22130）的完整修复路线为**31,726 token**；再加实验对照R2（12288）的整次研究账本为**44,014 token**。
- **门控触发与019交接**：F-fix query exact满足"补其余四题（+4）"门控，后续统一转入[019计划](../plan/019-submission-and-first-draft-pipeline.md)，按"①全量可提交Notebook端到端交付 → ②其余四题短首稿+一次真实反馈全流程验证 → ③B线CPU修复与同会话微探针"顺序推进。
- **v6已就绪**：驱动`9292c23d…`——wheel SHA改工作线程300秒超时，超时则跳过（pip install本身是完整性门）、绝不等；其余链路不变。notebook已同步。
- **推送身份对不上**：当前active令牌09:18被换成ccjjllt（`18c5`），推qingyu1232内核理所当然403。token↔账户映射已钉死（push-a.json token_map）。推前必须先对身份，不再盲推。
- **v5出推前检查清单（已做）**：驱动通读无逻辑问题（F-fix/R2同seed 5103、同预算12288、ledger、落盘链完整）；嵌入三SHA内存校验通过；notebook重建stale NONE（202092字节）；配额门：2700秒worst-case 3h＜剩余4.53h。恢复推送后顺序：探针先行（测SaveKernel＋槽位，约0.1h）→探针COMPLETE才推v5。
- 配额速记（teammateb身份）：23.85→4.59h；v4空转约耗6h（90分钟×4卡），其余为用户自有作业消耗。后续任何90分钟会话先看剩余额度，不再默认能推。
- **诊断探针**：`experiments/018/probe/`（hello+nvidia-smi+写文件）已COMPLETE，约14秒，L4与文件落盘正常；排除平台基础设施故障后推送v4。
- **本轮评分完成**：F-fix query exact、R2未生成可执行程序；触发+4后续研究。后续度量仍为每GPU小时新增正确题。

## 2 B线：受约束生成＋可靠原语（本地完成，GPU待验）

- `experiments/018/congen/opspec.py`（SHA`753ceb0c…`，stdlib only）：6操作（extract/filter/sort/transform/place/output），对象保留真实格点、bbox只派生；`validate_template`逐字拒绝未知op/多余键/自由文本/超长/超候选；`guided_schema()`同源生成（server尽力、CPU为准）；`search_template`上限32候选/10秒CPU、同一配置必须全过所有demo。
- 干跑（证据`congen/dryrun-report.json`，手造网格）：同色/多色提取、4/8连通、旋转约定、越界拒绝、shift搜索一次命中（dr=-2）、64候选拒绝/32候选放行、非法计划逐条拒绝。只证接线/格式。
- 外部依据：michaelhodel/arc-dsl README＋示例（2026-10-19查阅）：`objects(grid,univalued,diagonal,without_bg)`返回保留位置颜色的格点集，paint/fill/cover作用其上；本轮只移植子集，背景取众数色记为第一版局限。
- B-verify GPU会话（schema＋约束＋模板提示，1–2题验"可执行结构率"）待本地结论稳定后另批，不与A合并。`chat_once`的`extra_body`补丁只出现在B-verify vendored副本，A线副本与017一致（SHA`8970e7a7…`可直接复用，但独立存放待推送时核对）。

## 3 第3项（有界编辑）

本轮只留设计（见计划§5），排在B1/B2之后，不实现。
