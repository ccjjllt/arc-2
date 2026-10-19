# 017 — 程序反馈研究复核与交接修订

日期：2026-10-19，北京时间。对应[017修订计划](../plan/017-program-feedback-and-submission-entry.md)。复核起点为`7e978e9`，工作区起始干净。

**当前状态：已完成研究/文档复核；017求解代码、GPU程序实验、提交入口尚未实施/运行。** 本轮没有训练、发布Notebook、正式提交或读取C6/981答案，没有新比赛分数。只改研究文档、README并保存三份证据笔记。

## 1 总体判断

016核心结果经独立重算成立：C8短测有收益、B16无独立新解、R恢复无新增query，继续“完整程序+真实demo反馈”的方向合理。原017仍有会妨碍执行或答错研究问题的设计和接口缺口；现已修订，不需推翻主线、换模型或重跑016。

## 2 已修正的主要问题

| 原问题 | 实际影响 | 文档修正 |
| --- | --- | --- |
| 两槽设为F/R，却声称比较共享P0后的反馈vs重启 | 混合方案成绩无法回答反馈是否优于重启，也丢掉了P0候选 | 分别评价A_F=[P0,F]、A_R=[P0,R]，F/R同seed5102，P0=5101；P0也执行query |
| P2先跑后再完整重跑P6 | P2是P6子集，无必要重复6调用，还可能混选不同重跑结果 | 复用P2，只补4题；完整P6共18次/221184最大输出，较原24次/294912减少25% |
| 原解析器直接接vLLM final | 完整程序会被误判incomplete_thinking或incomplete_no_eos | driver先检查stop且无错误，再direct解析独立content；不伪造EOS token；模型仍thinking |
| “原执行器原样import”未交代文件/旧入口契约 | 直接exec可能破坏__file__自启动；旧ProgramLimits拒绝12288；旧runner走HF Session | 独立物化solve_program.py，只复用CPU helper，生成和调度留在016/vLLM链路 |
| 程序完整、全部demo可运行、demo全对混为一谈；停止规则前后不一致 | 反馈已执行也可能误记未测，或因一题失败错停全部研究 | 区分解析恢复/运行异常/网格差异反馈；P2至少一个完整P0有一个合法demo返回时才继续，其他情况如实记录保守止损 |
| 后续预算直接采用123.41tok/s，假设C8一直填满 | P2依赖与长修复输入会降低并发/增加prefill | 用P2实际阶段成本估算；首次建议90分钟总上限，80/85/90收尾，运行前重查余额 |
| “只读test字段”、全作业预算不足即拒绝 | 可能丢train示例，或未生成submission就退出 | 保留train输入/输出、test仅输入；先全Q合法占位，逐结果更新，停止超预算追加而不丢文件 |
| 013提交构建器被笼统视为可直接复用 | 其pickle/kgmon入口不识别程序JSON，还要求增强NLL | 只借格式、网格检查、初始占位与原子写盘，显式映射程序候选，不伪造评分 |

P12不再由“任意非db命中”自动触发：58在015已命中，只有R重启新增也不能证明F反馈有效。修订要求分别看相对B、既有27B组合和A_R的增益，按照真正有收益的策略继续研究。

## 3 对016研究记录的复核结果

详见[016证据笔记](../../experiments/017/review-016-evidence.md)，对应016进展§13。独立评分只读取P6允许键，未输出答案网格。

| 指标 | 复核值 |
| --- | --- |
| C4/C8短测 | 各8192token，93.28/52.16秒，速度比1.788；有Running8观测 |
| B8/B16 | 合法4/6、6/6；各EM@1=1/6，固定EM@2=1/6，仅db |
| R恢复 | 合法8/15，新增一份db冗余exact；固定混合仍2/6，无新增/丢失query |
| 016全部输出 | 主185610+辅助512=186122，76次usage齐全 |
| R策略累计 | 015原任务341780+R36637=378417token，6340.5+351.74=6692.24秒生成阶段，加载另列 |

原527390混入016全部主实验，不是R策略成本；原“2.15配额小时≈38分钟×2”算术不成立，实际×2约1.262小时，且缺事后原始配额响应。重算字段只是日志关键词计数；123.41为暖缓存混合队列，不保证程序阶段达到；B16新增合法query是2个而非5个。均已在016记录中就地纠正，核心EM不变。

## 4 本次人工CPU核对

- 导入现有`solve_program.py`未加载torch。对自行写的恒等函数围栏文本，默认解析为`incomplete_thinking`；direct解析为`program_ready`；direct加仅含vLLM finish_reason的旧generation结构为`incomplete_no_eos`。这定位了接口问题，没有执行模型生成程序。
- `ProgramLimits(max_new_tokens=12288).validate()`按现有实现抛ValueError，证明不能直接复用旧生成入口。
- 核对执行器、题面、两份评分来源SHA，与原017声明一致；P6全部在题面/评分中，P2两键答案在两来源相同。后续可用010为主评分来源、P2文件为交叉身份核对。
- 013 helper在内存内构造2task/3query得到完整6attempt，确认格式小函数可复用；不代表隐藏输入或运行已验证。
- 18×12288=221184，单策略6×2×12288=147456；建议90分钟对应3.45规划配额小时+6小时储备=9.45小时。旧余额不是当前可用额度，未自动切换账号或开机。

以上均为人工一次性检查，不新增测试文件/框架，不把CPU核对称为研究提分。

## 5 产物与下一步

- [修订后的017计划](../plan/017-program-feedback-and-submission-entry.md)：唯一实施入口。
- [016原始证据复核](../../experiments/017/review-016-evidence.md)：成绩、成本、吞吐与配额口径。
- [设计复核](../../experiments/017/review-design.md)：两槽对照、重复调用与反馈语义。
- [实现复核](../../experiments/017/review-implementation.md)：解析器、执行器、题面和提交连接。

修订后再次独立阅读，未发现两槽、调用预算、反馈范围、90分钟截止或最终文件保存方面的阻断问题。后续实施只需局部适配与手动小实验；成果仍取决于真实程序生成/执行，未确认前不扩P12、C6或宣称可正式提交。

## 6 2026-10-19：GPU驱动已实现并推送（ccjjllt v1运行中）

按修订计划实施，未启动部分为GPU运行结果分析（待COMPLETE后取回）。

### 实现产物

| 产物 | SHA256 |
| --- | --- |
| `experiments/017/pq/vdriver017.py` | `c06caeb8673ace84724071d6cb38cc18c124408415232e0d37e531c7b034e191` |
| `experiments/017/pq/run_reasoning_probe.py` | `8970e7a7…`（与016一致，未改） |
| `experiments/017/pq/solve_program.py` | `03aaa381…`（与脚本一致，未改；import用，不exec） |
| `experiments/017/pq/p6-challenges.json` | `03c5c50b…`（P6六题，train全留、test仅input，由`task_inputs_only`重建） |
| Notebook | 204866字节，内嵌上4文件，解包SHA一致 |

要点对照计划§6–§7：复用016安装/模板/服务8/C8滚动/落盘/截止链路；`parse_generation`先查stop+无错再direct解析（`generation=None`）；沿用解析器python/py/无语言标签范围；`execute_grid` 2秒/512MB、程序累计20秒；反馈`demo_feedback`≤3000字符且先断言条数对齐；P0 query预测落盘复用不覆盖；F用P0 final文本（reasoning另存）；门控“两题各≥1完整P0且≥1合法demo输出＋时间够”否则`p2_gate_closed`收尾；两槽A_F=[P0,F]、A_R=[P0,R]缺一重复、皆缺记空；P0波→F/R波顺序组装，奇偶query交替先F/先R。

本地CPU核对（非GPU证据）：`py_compile`通过；prompt构造冒烟（291约3.6K、db约3.5K字符，均远小于64K）；direct解析示例程序得`program_ready`；F/R交替顺序与request_id唯一性已核；Notebook解包4文件SHA一致；`git diff --check`通过。macOS执行器返回`executor_unavailable`属预期（资源限制装不上），已按计划交由GPU会话内预检判定，不冒充模型错误。

### 推送记录

[`experiments/017/execution/push.json`](../../experiments/017/execution/push.json)：`ccjjllt/arc2-017-p0-f-r-program-feedback` v1，`RUNNING`，会话5400秒。推送前ccjjllt剩21.13h（满足9.45h门槛）；标题即slug，无016式id错位。用户明确指示使用ccjjllt配额。

## 7 2026-10-19：B提交入口已实现并CPU干跑

`experiments/017/submit_entry/submit_lib.py`（SHA`0f0f72e2…`）：逐字复用013的`atomic_json/initial_submission/valid_grid/existing_submission_matches`（kgmon/NLL/beam/pickle路径未借用），新增`count_queries/estimate_and_guard/update_slots/validate_submission`。

人工CPU干跑（证据`dryrun-report.json`，非测试套件）：公开多query输入2task/4query占位→校验通过；P6已记录合法网格部分更新（单候选重复填两槽已断言）→未知键/锯齿/bool/越界更新被拒并记错；更新后文件仍合法可读；调度门对“240query×15K单遍”放行（约9.02h）、“两遍等价”拒绝并报23842秒缺口。干跑只证明接线/格式，不证明隐藏运行、12小时可完成或提分。

## 8 2026-10-19：v1结果（P2门未过，+4未跑；F-db程序exact）

会话`complete_with_evidence`，rc=0，耗时1806秒；server无preempt/OOM记录；执行器预检通过；ccjjllt配额8.93→9.88h（约0.95h）。6次调用全部有usage。

### P2逐调用（P0 seed 5101，F/R seed 5102，各12288/思考≤8192）

| 调用 | finish | parse | demo命中 | query合法 | ctok |
| --- | --- | --- | --- | --- | --- |
| P0-291 | length | generation_not_ok | 0 | 否 | 12288 |
| P0-db | length | generation_not_ok | 0 | 否 | 12288 |
| F-291 | stop | program_ready | 3/4 | 是 | 9842 |
| R-291 | stop | program_ready | 0/4 | 是 | 9965 |
| F-db | stop | program_ready | 4/4 | 是 | 9701 |
| R-db | length | generation_not_ok | 0 | 否 | 12288 |

门控：`p0_both_ok=false`（两题P0均被12288截断、无完整程序），`demo_legal_any=true`，`time_ok=true`（P2用1065秒，+4预估3029秒、剩余3299秒）。按计划保守止损：`p2_gate_closed`，+4四个题未跑，`slots.json`中+4槽位全空。

### 本地评分（GPU只做执行，答案比对全在本地；P6为反复用开发集，不作公榜/泛化声明）

- F-db query网格与`experiments/010/kaggle-e0-v1/010/scoring.json`中db答案字节一致：**exact**；且同程序4个demo全过。程序是通用写法（找外框颜色、inner归一化到3×3、下采样兜底、15×15铺两块），非硬编码答案。
- F-291、R-291 query均为形状错失（26×6、28×8 vs 答案18×8）。
- 两槽装配（P0缺席，实际退化为单臂重复）：A_F=[F,F]在P2为1/2（db），A_R在291为[R,R]错失、db为空槽 → 0/2。

### 研究结论（诚实口径）

1. **原问题未被回答**：P0两题均无完整程序，共享P0锚点不存在，实际比较的是[失败P0＋错误上下文重试] vs [独立重启]，不是计划要的[P0,F] vs [P0,R]。F-db的exact不能记为“反馈修复优于重启”。
2. **程序可行性信号**：同seed下F臂2/2完整程序、R臂1/2；F-db一次拿到4demo全过＋exact，说明程序路线在db题上走得通，反馈文本即使基于失败P0也有用（重试效应）。但n=2，不作胜负断言。
3. **预算教训**：6次调用3次撞12288上限（含两个P0），P2波agg仅约22tok/s（长程序生成远慢于016短测123）。程序任务thinking 8192＋12288总量不够P0一次写完；下一轮要么加总量、要么降thinking预算、要么P0给更短的脚手架提示。按当前速度，+4（12次）约需3000秒以上，90分钟会话装得下但余量小。
4. B提交入口不受影响：`submit_lib`干跑结论仍成立，待真实程序+4或全量时再接入。

### 下一步（需用户定）

- 选项a：调预算重跑P2门（P0加到16384或thinking降到4096），再进+4。成本约1h配额。
- 选项b：放弃共享P0锚点，改测更便宜的问题（F/R都从失败P0重试的胜率对照），但这已偏离017原问题。
- P12仍暂缓；C6/981未动；无提交。
