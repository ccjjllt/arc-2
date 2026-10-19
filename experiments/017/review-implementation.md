# 017 实施可行性与提交入口只读复核

日期：2026-10-19，北京时间。检查对象为 `docs/plan/017-program-feedback-and-submission-entry.md`、现有 `scripts/solve_program.py`、016 vLLM调用接口、013提交构建器及P6范围输入。只做源码读取、SHA计算、合成文本解析及内存内格式检查；没有GPU、训练、提交、依赖安装或求解代码修改。本次仅新增这份笔记。

## 结论

017可用现有代码作小范围实施，但不能将“原执行器一字不改导入”解释成“旧整个程序runner直接调用”。**需先明确vLLM final解析适配、独立执行器文件落地、train示例保留及预算不足时仍持久化全量submission。** 这些是直接影响实验能否跑通的衔接问题，不需要新增框架。

## 1. 输入身份与P6覆盖：通过

文件字节SHA与017一致：

| 实际路径 | SHA256 |
| --- | --- |
| `scripts/solve_program.py` | `03aaa3818a8f3beafee019140bae823f968e913af72ef297fa7531f24816a37a` |
| `experiments/010/e0-p12-challenges.json` | `792ca71c069abb420368efd1b0a29c04adaaed9761ba9225528db735cfe44a40` |
| `experiments/010/kaggle-e0-v1/010/scoring.json` | `fbe21ed91e38259e70aec5864994944759e2ec83dd0bf2c9093a7c549c562459` |
| `experiments/012/inputs/p2-scoring.json` | `2efbc9bf7437fd332521571aa582fc2126b30b9ff8658280a8d31c331ed08daf` |

010另一份 `experiments/010/kaggle-e0fill-v1/010/scoring.json` 与上表评分文件SHA相同。计划仅写“010评分文件/P2评分文件”，实施交接应补准确路径，避免执行者重新搜整个公开evaluation答案。

题面中的P6范围如下，每题均一个query，test项均只有 `input`：

| task | demo数 | query数 |
| --- | ---: | ---: |
| 291dc1e1 | 4 | 1 |
| 4c416de3 | 3 | 1 |
| 58c02a16 | 3 | 1 |
| aa4ec2a5 | 3 | 1 |
| db118e2a | 4 | 1 |
| fc4aaf52 | 2 | 1 |

010评分文件已覆盖全部P6；专用P2文件中的291/db答案与010对应值一致。可将010作为P6唯一评分源，P2文件用作身份交叉核对，避免出现合并覆盖规则。只筛选并检查了上述P6键，没有显示任何答案格子；没有读取C6或其他保留集答案。上传包应仅物化P6六题的 `train` 和 input-only `test`，不上传整个评分文件，也不因原文件覆盖P12就让GPU端扩大范围。

## 2. `solve_program.py`实际接口与stdlib边界

人工CPU导入确认没有加载torch、transformers、numpy或vLLM。可复用的CPU接口为：

```python
validate_code(code)
split_program_response(raw_text, thinking_mode="thinking", generation=None)
execute_grid(code, grid, *, wall_seconds=2.0, memory_mb=512)
demo_feedback(demos, results, max_chars=3000)
task_inputs_only(task)
```

**不要复用 `run_program_task()` 作为017整个调度入口。** 该函数内部导入 `grid_runtime`，走旧Session生成；`ProgramLimits.validate()` 对旧三调用策略要求每次≤1536 token，P0-only也仅≤3072，设置12288会直接抛ValueError。旧repair还会跳过没有完整P0的情形，种子经task/arm哈希，输出策略为P0+alternative，均与017的新计划不同。可保持原文件字节不变，仅调用上面的CPU helper，生成与队列继续在016 driver侧编排。

`execute_grid`的2秒包括解释器启动、输入输出和实际运行，不能在计划里承诺“CPU毫秒级”；本轮未运行真实生成程序。它内部CPU时间另限制≤2秒、内存≤512MB，subprocess不是普通函数内直接exec。资源限制不能安装时返回 `executor_unavailable`，这与模型程序错误应分开。

### 2.1 内嵌方式须保留独立文件

`execute_grid`用 `[sys.executable, "-I", os.path.abspath(__file__), "--execute-worker"]` 启动自身。Notebook应把冻结字节写成独立 `solve_program.py`，再import该文件。若将源码直接 `exec()` 到driver的globals，`__file__`可能指向driver，子进程会错误启动driver并传入其不支持的参数。这不是重写执行器，而是保留现有自启动设计所需的文件身份。

### 2.2 vLLM final不能直接交默认thinking解析

016 `chat_once()`已经返回分开的 `content`、`reasoning`、`finish`；它没有返回带 `</think>` 的完整原token串，也没有 `ended_eos` 标记。

合成完整程序的人工解析结果：

| 调法 | 实际状态 |
| --- | --- |
| `split_program_response(final)` | `incomplete_thinking`，因为默认要求原文恰好一个 `</think>` |
| `split_program_response(final, "direct")` | `program_ready` |
| direct加 `generation={"status":"complete","finish_reason":"stop"}` | `incomplete_no_eos` |
| direct加旧协议 `status=complete, ended_eos=True` | `program_ready` |

最小适配是：driver先根据vLLM实际 `finish_reason=stop`、无client_error/取消等判定本次生成正常结束；再仅对final调用direct语法解析，`generation=None`。记录真实vLLM完成证据，不伪造观察到的EOS token。只有两层均成立才承认程序可执行；被截断但有闭合块的内容仍按计划记未完成。这里direct只表示“解析已经拆分后的final”，并不表示本次模型关闭思考。

另一个协议细节：现有正则允许 `python`、`py`以及无语言围栏，范围比017声明的“恰好一个闭合python块”宽。若坚持017精确协议，在driver加一个窄格式判断；或在计划明确沿用现有解析器范围。不要一边声称只接受python、一边实际接受更多格式。

### 2.3 demo反馈需长度对齐

`demo_feedback(demos, results, max_chars=3000)`返回JSON字符串，逐项携带执行状态、尺寸、至多8处格子差异及截短错误；它不会执行代码，也不自动生成parse错误。函数内部用 `zip()`，结果条数不足会静默少报demo，因此调用前应要求 `len(results)==len(demos)`。未通过解析的P0不能执行，但可以为每个demo生成对应的未执行/解析错误状态，并将该分支记为语法修复而非真实执行反馈。

3000字符不是简单 `feedback[:3000]`：现有helper先保留全部demo摘要，再按预算加入细节；摘要本身超限会抛ValueError。P6最多4个demo，通常不触发，但应沿用helper而不是截断JSON。完整程序也应检查结果条数齐全、每个status为ok，避免 `all([])` 被当成功。AST验证返回的 `code` 字段可能在 `incomplete_no_eos` 状态仍存在，不能只凭非空code就运行。

## 3. 013 builder可复用什么

`experiments/013/submission-src/build_submission.py`可直接借用的部分：

- `initial_submission(challenges)`：按实际task/query生成完整双attempt及映射，缺失槽为 `[[0]]`。
- `valid_grid(value)`：矩形1–30、整数0–9，拒绝bool/浮点。
- `atomic_json(path, value)`：临时文件写入后替换，保留上次完整JSON。
- `existing_submission_matches(path, reference)`：检查task集合、query数量、两个attempt键与每个网格。

人工CPU合成2task/3query得到完整6attempt，全部网格合法，未导入GPU栈。该检查仅验证结构，不能证明隐藏题解答或完整运行。

**不能直接照搬整个builder main。** `populate_submission`扫描本次可信 `.base` bz2 pickle，要求每个候选有 `solution/score_aug/beam_score`，并执行4B kgmon。017程序结果是JSON和真实执行网格，无增强NLL。直接丢JSON进这个目录会被忽略；伪造NLL凑格式会改变研究口径。只需一层P0/F/R结果到显式attempt的映射，不需要新增统一评分框架。

提交JSON装的是整数二维数组，不存在围栏/字符间空格要求。那些是模型文本解析协议，应留在读取模型输出阶段；程序 `execute_grid` 已返回合法二维数组，不应重新包成代码围栏再解析。

## 4. 必须修正的提交入口描述

### 4.1 不能只保留test字段

017 §5.1“只读test inputs（test字段）”若按字面实现，会丢掉归纳规则需要的train示例。正确边界是读取challenge文件全部task，保留 **train.input + train.output**，test只保留input，永不读取test.output/独立solutions文件。已有 `task_inputs_only()` 正好执行这项重建且要求非空demos/queries，可复用。

013 supervisor的隐藏路径选择可参考：根据真实 `KAGGLE_IS_COMPETITION_RERUN`读取 `arc-agi_test_challenges.json`，按两种已知挂载路径查找；不要手动把普通Notebook标成hidden，也不要假定普通运行可见隐藏240题。Q按每题test长度求和正确，应在每次运行从challenge实读，而非写死240。

### 4.2 超预算拒绝与完整输出应分层

全作业若因估算超预算、且在写初始submission之前退出，便违反“完整submission始终存在”的目标。建议最小顺序：

1. 读取challenge后立即初始化所有Q的两个合法兜底并原子落盘，记录真实覆盖为0。
2. 估算首轮/第二轮的额度，拒绝超额配置或额外请求；这个拒绝不能删掉已经有的文件或候选。
3. 每个有效结果落盘后更新对应槽，定期及结束前原子导出；到期停止新请求，保留已完成解和明确未解的兜底。
4. 分别报告结构覆盖Q/Q、真实候选覆盖、兜底数；有合法JSON不等于全题都完成模型推理。

本地CPU工具可以拒绝一个不可能完成的部署配置，并返回缺口；真正隐藏运行的父进程仍需维护最终文件。若目标还要求所有query至少一次真实调用，就必须在部署前选一个能承受的首轮配置，不能靠异常退出或把兜底算成覆盖实现。

## 5. 调度和成本仍需计划说清的边界

- P2是P6的子集且同配置同seed。当前§6按P2 6次+完整P6 18次记24次，会再生成P2；若没有明确重跑目的，最省预算的是复用P2，新增四题12次，共18次/221184输出token。若保留24次，应明确第一次P2只用于门控、第二次P2进入主分数，以及重复样本身份，不能混选更好的结果。
- 123.41 tok/s来自016的特定C8长网格工作量。017有F依赖P0完成与CPU反馈，P2只有两题，不能维持8路并发；程序提示、repair长前缀也改变prefill成本。因此52分钟是条件估计，不是已有运行证据。所有调用仍共用绝对截止，不能为了凑计划完整度自动延长会话。
- F依赖对应P0，不能一次把三调用全部作为独立spec投入旧 `chat_wave()`。可用一个小队列：先准备P0与restart任务，在P0完成并反馈落盘后追加repair；用已有C8上限，无需重写通用agent框架。P2自然不足8路不属于配置违约。
- 017目前只执行F/R的query，若要报“F相对P0保住/改错/修好”，必须也执行P0的query并本地离线评分；此操作不增加模型生成且不进入反馈。共享P0的两策略应另外报告P0+F与P0+R真实两槽，否则F+R合并EM@2不能单独回答反馈是否优于重启。主研究设计如何选择以父审阅为准。

以上均是未实施的修正建议。最关键的几个衔接点写清后即可按现有runner推进；本轮没有新增自动测试文件、测试框架或CI。
