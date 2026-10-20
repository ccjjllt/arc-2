# 019 — 全量可提交候选打通、首稿形成率与+4完整流程验证

**后续入口已更新为[020提分验证与真实提交](020-score-validation-and-real-submission.md)。** 本计划保留019实施时的历史设计；P1/P2现已运行，结果见[019进展](../progress/019-submission-and-first-draft-pipeline.md)，不再按下面的初始“尚未运行”状态或低预算直出优先级继续执行。

修订日期：**2026-10-20，北京时间**。复核起点为`efca1ba`（2026-10-20 10:07提交）。承接[018实际进展](../progress/018-f291-feedback-and-constrained-generation.md)与独立复核结论。**当前状态：已完成018结果与代码缺陷复核，本文件为019唯一实施入口；代码修改与GPU实验尚未运行。**

本轮核心主线：**先交付可完整运行的27B全量提交候选，再攻克“固定总预算内稳定生成首个完整程序，并做一次真实执行反馈修复”的完整流程。** 当前B线（受约束程序生成）存在确定性代码与协议缺陷，严禁原样推送，必须先完成本地CPU最小修复并在已加载模型的会话中做极小探针验证。

## 1 018复核结论与本轮继承事实

### 1.1 `291dc1e1`新增正确解成立，但须区分三笔成本账本

在018 A线v6（[`experiments/018/kaggle-v6/vprobe/run.json`](../../experiments/018/kaggle-v6/vprobe/run.json)）中，冻结程序[`F-291dc1e1_0.code.py`](../../experiments/018/frozen/F-291dc1e1_0.code.py)原本通过3/4个demo，经GPU端真实执行产生459字符差异反馈（[`frozen-feedback.json`](../../experiments/018/kaggle-v6/vprobe/frozen-feedback.json)，无人工补丁），对比同预算独立重启R2结果如下：

| 项目 | F-fix：已有程序＋真实反馈 | R2：同预算独立重启 |
| --- | ---: | ---: |
| 实际生成 token | 9,596 | 12,288 |
| 结束原因（`finish_reason`） | `stop`（正常结束） | `length`（达到长度上限） |
| 完整可执行程序 | 有（143行通用Python） | 无（final通道12,260字符分析文字，无`def transform`与代码围栏） |
| 示例通过情况 | **4/4** | 无程序可评（0/4） |
| Query 预测结果 | **与官方答案完全一致（exact）** | 无有效候选 |

這证明了**“已有接近正确的完整程序时，一次真实反馈修复能够带来此前未解出的新增正确题”**，满足018既定的“补P6其余四题（+4）”门控条件。但成本核算必须严格区分三个口径，不得混用：

1. **单次反馈修复本身**：`9,596` completion tokens（`F-fix-291dc1e1_0`）。
2. **实际修复路线完整生成成本（含历史起始程序）**：`9,596 + 22,130（017 v1的P0-291 12,288 + F-291 9,842）= 31,726` completion tokens。不能拿9,596估计从零开始的部署成本。
3. **整次研究对照总账本（含实验对照R2）**：`31,726 + 12,288（R2）= 44,014` completion tokens。研究对照R2不必计入实际修复部署路线的必需成本。

### 1.2 已定位的四项阻断缺陷（本轮必须优先清零）

| 编号 | 缺陷模块与位置 | 复核定位事实 | 本轮处理原则 |
| --- | --- | --- | --- |
| **缺陷1** | 启动哈希死循环与62位截断常量：[`vdriver018a.py`](../../experiments/018/aq/vdriver018a.py)、[`vdriver018b.py`](../../experiments/018/bv/vdriver018b.py) | `sha256_file`写为`for ch in iter(lambda: h.read(1 << 20), 1):`，EOF返回`b""`永不等于整数`1`，导致无限空读；018进展中“存储卷stall、非代码死循环”归因错误。v6靠后台线程300秒超时跳过才跑通，[`driver-watch.log`](../../experiments/018/kaggle-v6/vprobe/driver-watch.log)证实该线程在模型调用期间仍停留在`sha256_file`第97行。同时两驱动`VLLM_WHEEL_SHA256`仅有**62位**，修完死循环后会立即因哈希不匹配失败。 | 哨兵改为`b""`；恢复017已校验的64位wheel SHA256；同步更新外部`.py`与所有提交/实验Notebook内嵌源码。 |
| **缺陷2** | 思考预算封顶后final通道仍泄漏长篇散文：[`R2-291dc1e1_0.final.txt`](../../experiments/018/kaggle-v6/vprobe/A291/R2-291dc1e1_0.final.txt) | 限制`thinking_token_budget=8192`后，R2在`final`通道继续写了12,260字符分析文字而未写出代码围栏，耗尽12,288总预算。没有首稿程序，反馈机制就无法介入。 | 首要研究指标收敛为**“固定总预算内，有多少题能交出首个完整、可执行的程序”**；收紧首稿生成协议与提示模板，不靠盲目加长上限或机械重跑4次长生成。 |
| **缺陷3** | B线约束字段无效、预算倒挂、表达力不足与schema不一致：[`run_reasoning_probe.py`](../../experiments/018/bv/run_reasoning_probe.py)、[`opspec.py`](../../experiments/018/bv/opspec.py)、[`vdriver018b.py`](../../experiments/018/bv/vdriver018b.py) | ① 原始HTTP请求传嵌套`extra_body.guided_json`，固定版本vLLM（`0.27.2rc1.dev91+g1f7427bc0`）顶层协议字段实为`structured_outputs: {"json": schema}`，且未知字段默认允许存在并被静默忽略，“不支持必返400”假设不成立；② `GEN_MAX=4096 < THINK_BUDGET=8192`，思考未完先撞总上限；③ `place_objects`固定创建输入尺寸`H×W`画布，而291（4个demo均改变尺寸）和db（输出固定15×15）都改变输出尺寸，当前语言根本无法表达这两题正确解；④ CPU搜索支持`{"choices": [...]}`，但`guided_schema()`仅允许标量。 | 先在CPU端修齐传输协议、预算配比、变尺寸/裁剪/排列原语与`choices` schema；随后在已加载模型的会话中用极小预算请求验真约束生效，再跑任务。 |
| **缺陷4** | 提交入口未完成全量连接且`update_slots`存在双槽覆盖陷阱：[`submit_lib.py`](../../experiments/017/submit_entry/submit_lib.py)、[`dryrun-report.json`](../../experiments/017/submit_entry/dryrun-report.json) | 017干跑仅2个task/4个query；`update_slots`收到单个新候选时会执行`if len(ok) == 1: ok = [ok[0], ok[0]]`，将两个槽同时覆写为该候选，若逐候选直接调用会抹掉此前已有的不同有效候选。同时格式校验通过不代表有真实预测（全`[[0]]`占位也能过检）。 | 建立每题候选池（candidate pool），按固定规则去重并选出两份最优候选后再调`update_slots`；分别统计格式合法性与真实非占位预测覆盖率。 |

### 1.3 会话加载开销与批处理原则

018 v6总耗时`1706.8`秒（28.45分钟），其中离线安装与模型加载`install_load_s = 1193.8`秒（19.90分钟，**占总时长约70%**），两条长请求的生成波次仅约8.45分钟。后续GPU实验必须**把多个任务/验证步骤合并在同一次模型加载之后批量执行**，不再为单题或单个字段检查单独消耗20分钟启动开销。

## 2 验收顺序与本轮总体安排

本轮严格按以下三个优先级依次验收：

1. **第一优先级（P1，全量可提交产物交付）**：今天先完成一个可以完整运行的27B全量提交候选（Notebook + 完整`submission.json` + 真实预测覆盖数 + 占位数 + 总耗时），不等待B线研究完成。
2. **第二优先级（P2，+4完整流程验证）**：在P6其余四题（`4c416de3_0`、`58c02a16_0`、`aa4ec2a5_0`、`fc4aaf52_0`）上一次性验证**“短而可执行的首稿 + 一次真实反馈，能否以可接受的总成本在其他题上重复获得新增解”**，先看首稿完整率，再看示例全过率与query exact。
3. **第三优先级（P3，B线最小CPU修复与同会话微探针验证）**：本地修完哈希、顶层`structured_outputs`、变尺寸表达能力与`choices` schema一致性，在已加载模型的GPU会话中附带一个极小约束探针确认服务端执行约束后，再运行受约束解题验证。

## 3 阶段0：公共阻断缺陷修复规范（本地CPU先行）

在打包任何Notebook或启动GPU会话前，必须先完成以下本地代码修正并人工核对：

### 3.1 修复`sha256_file`死循环与64位wheel哈希常量

1. 将所有驱动（含[`experiments/018/aq/vdriver018a.py`](../../experiments/018/aq/vdriver018a.py)、[`experiments/018/bv/vdriver018b.py`](../../experiments/018/bv/vdriver018b.py)及019新建驱动与Notebook内嵌脚本）中的文件哈希函数统一修正为：
   ```python
   def sha256_file(path: Path) -> str:
       d = hashlib.sha256()
       with path.open("rb") as h:
           for ch in iter(lambda: h.read(1 << 20), b""):
               d.update(ch)
       return d.hexdigest()
   ```
2. 移除v6为绕过死循环而引入的300秒后台线程超时跳过逻辑（`_hash_wheel`线程包装），直接同步快速完成校验。
3. 将`VLLM_WHEEL_SHA256`统一恢复为017已实测通过的64位常量：
   ```python
   VLLM_WHEEL_SHA256 = "5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a"
   ```
4. 打包Notebook后，必须从生成的`.ipynb`中解包或扫描内嵌源码，确认不存在`iter(lambda: h.read(1 << 20), 1)`与62位哈希残留。

### 3.2 修复`submit_lib.py`双槽候选池选择与覆盖率统计

在019提交模块（继承[`experiments/017/submit_entry/submit_lib.py`](../../experiments/017/submit_entry/submit_lib.py)）中补充候选池管理与覆盖率统计函数，避免单候选覆写双槽：

1. **每题候选池结构**：维护`pool: dict[str, list[dict]]`（键为`f"{task_id}_{index}"`），每条候选记录：
   - `grid`：通过`valid_grid`校验的归一化整数网格；
   - `source`：`'program_fix'` | `'program_init'` | `'direct_grid'`；
   - `all_demos_pass`：`bool`（程序候选在全部`train`示例上是否逐格全过；直出网格记`False`）；
   - `demo_matches`：`int`（命中的`train`示例数）；
   - `order`：生成先后序号（用于稳定同分排序）。
2. **固定去重与双槽选取规则（绝不使用query真值）**：
   - 先按优先级排序：`all_demos_pass == True`优先 → `demo_matches`降序 → 来源优先级（`program_fix` > `program_init` > `direct_grid`） → `order`升序；
   - 按网格内容逐格去重（保留首次出现的最高优先级候选）；
   - 若去重后有`>= 2`个不同合法网格，取前2个分别填入`attempt_1`和`attempt_2`；若仅有`1`个合法网格，才将该网格重复填入两槽；若为`0`个，保留初始`[[0]]`占位。
3. **分离格式校验与真实预测覆盖统计**：
   - 新增`coverage_report(submission, query_map, updated_keys)`：分别统计总task数、总query数$Q$、格式合法query数、**至少1槽为真实非占位预测的query数**、**两槽均为相异真实预测的query数**、**仍停留在初始`[[0]]`占位的query数**。

## 4 第一优先级（P1）：27B全量可提交候选端到端打通

### 4.1 端到端执行链路

直接复用已验证的`Qwen/Qwen3.8-27B` FP8 + 固定vLLM wheel + TP4 + 服务容量8 + C8滚动队列，在`experiments/019/submit/`下实现完整提交驱动与可提交Notebook：

```text
读取真实 challenge 输入 → 保留全部 train demos + 统计全部 test queries (Q)
  → 立即落盘全量 [[0]] 占位 submission.json
  → 阶段1（广覆盖保底）：C8 滚动运行第一遍预算受控生成，解析合法候选入池并持续原子写盘
  → 阶段2（选择性程序+一次反馈补充）：在剩余共享总墙钟内，对尚未获得 demo 全过或双槽相异候选的题目运行短程序首稿与最多一次 demo 执行反馈修复
  → 全局截止前收尾：停止新请求、排空在途生成、最终校验与原子写盘、安全关闭服务
```

### 4.2 全局共享时间预算与收尾门控

- **单一总时钟**：从Notebook进程启动刻（`t0_mono = time.monotonic()`）起算全局唯一截止时间`session_deadline = t0_mono + session_seconds`。离线安装、模型加载、阶段1直出、阶段2程序首稿、CPU示例执行、阶段2反馈修复、写盘收尾**全部共享这一个总预算**，严禁各阶段各自重置或独立计算额度。
- **动态余量调度**：
  - `T - 600s`（截止前10分钟）：停止向队列追加任何新阶段或新请求；
  - `T - 300s`（截止前5分钟）：硬截止在途请求，完成最后一次候选池汇总、`atomic_json`写盘与`validate_submission` + `coverage_report`记录，随后清理进程组退出。
- **候选池持续写盘**：在`chat_wave`的`on_slot_done`回调中，每完成一条请求并解析/执行出合法网格后，立即更新该`query_key`的候选池并调用`atomic_json`刷新`/kaggle/working/submission.json`，确保任意时刻中断都留有当前最高覆盖的完整合法提交文件。

### 4.3 公开输入端到端演练交付物

用公开挑战集输入（先在CPU做完整多题调度与故障注入演练，再在Kaggle L4×4完成真实端到端演练）交付以下四项产物：

1. **可提交Notebook**：`experiments/019/submit/notebook/`（内嵌修复后的全部源码，无外部未挂载依赖）。
2. **完整输出文件**：`submission.json`（通过`validate_submission`全部task/query键与网格合法性检查）。
3. **覆盖率与占位账本**：明确列出总task数、总query数$Q$、真实预测覆盖query数（单槽覆盖与双槽相异覆盖分开报）、占位数（未覆盖query数）。
4. **分阶段耗时与吞吐记录**：安装加载耗时、阶段1耗时与完成请求数、阶段2耗时与完成请求数、总耗时。

## 5 第二优先级（P2）：P6其余四题（+4）的完整流程验证

### 5.1 核心研究问题与实验范围

本支线只回答一个问题：

> **短而可执行的首稿（P0）＋一次真实执行反馈（F），能否以可接受的总成本，在P6其余四题上重复获得新增正确解？**

- **测试题集（固定分母=4）**：`4c416de3_0`、`58c02a16_0`、`aa4ec2a5_0`、`fc4aaf52_0`（题面来自[`p6-challenges.json`](../../experiments/017/pq/p6-challenges.json)，评分仅在取回后于本地对照`experiments/010/kaggle-e0-v1/010/scoring.json`进行）。
- **不重跑`291dc1e1_0`与`db118e2a_0`**：017/018已有完整记录，本轮不再围绕这两题消耗GPU。
- **单次模型加载批量执行**：四题的P0生成、CPU示例执行、F修复与R重启全部放在同一次vLLM服务启动之后批量完成，摊薄约20分钟的固定安装加载成本。

### 5.2 提升首稿（P0）完整可执行率的提示与预算设计

针对017 P0两题截断及018 R2在`final`通道写满12,260字符分析文字的教训，首稿P0采取**“预算封顶 + final通道禁散文 + 紧凑通用函数模板”**：

1. **生成预算与采样**：保持`max_tokens = 12288`、`thinking_token_budget = 8192`、`effort = "xhigh"`、`seed = 5101`。
2. **final通道强契约提示**：在System与User提示末尾明确要求：
   - 推理分析必须在`<think>`阶段内完成；
   - 进入最终回复（`final`通道）后，**首行必须直接是` ```python `，禁止在代码块前后输出任何分析段落、步骤复述或解释文字**；
   - 提供简短的通用纯函数编写规范（直接复用网格尺寸、连通块/外接框提取与坐标映射的紧凑写法，控制在150行以内返回`out`网格）。

### 5.3 一次真实反馈（F）与同预算独立重启（R）对照协议

对四题一次性批量执行以下三步流程：

1. **Step 1 — 批量运行四题首稿 P0（4次调用，C8并发，seed 5101）**：
   - 解析每题`final`输出，检查是否正常结束（`finish_reason == "stop"`）且包含唯一闭合代码块并通过`validate_code`。
   - **门控规则（按题独立进入反馈，如实计分母）**：
     - 若某题P0产出了完整程序，则在全部`train`示例与`test`输入上执行（单grid限时2秒、512MB，单程序总计20秒），记录P0的demo命中数与query预测网格，并用`demo_feedback`构造真实执行反馈（运行异常或预测网格差异，上限3000字符，零人工干预）；
     - 若某题P0未产出完整程序（如截断或语法非法），**不再向其发送伪造或残片恢复请求充当“执行反馈”**，该题在反馈机制上直接记为**首稿失败（整条反馈流程失败）**，但仍始终保留在4题总分母中。
2. **Step 2 — 批量运行 F（一次修复）与 R（独立重启）（seed 5103，12288/8192）**：
   - 对P0完整的题，并发发出1次**F（携带题面 + P0完整代码 + 真实`demo_feedback`）**与1次**R（仅原始题面，不见P0与反馈，seed 5103）**；
   - 对P0未完整的题，仅发出1次同预算独立重启**R（seed 5103）**以完成双槽重启策略`A_R = [P0, R]`的公平对照，而`A_F = [P0, F]`在该题上无F候选（维持空槽或P0结果）。
3. **Step 3 — 保留起始候选与双槽评价**：
   - 严格保留P0的预测候选：反馈臂双槽为`A_F = [P0, F]`（当F产出全过demo或合法网格时，与P0共同构成两槽；若P0已全过demo而F改坏，P0仍在槽中保底），重启臂双槽为`A_R = [P0, R]`。

### 5.4 必须完整报告的五项指标（分母固定为4）

取回产物后在本地完成评分，必须同时报告以下完整漏斗与成本账本：

1. **首稿完整率（First-Draft Completeness）**：`P0_complete / 4`（及P0可运行率`P0_runnable / 4`）。
2. **示例全过率（All-Demos Pass Rate）**：`P0`、`F`、`R`各自通过全部`train`示例的题数（`/ 4`）。
3. **Query Exact 命中率**：`P0`、`F`、`R`的单臂EM@1，以及`A_F = [P0, F]`对比`A_R = [P0, R]`的双槽EM@2（`/ 4`），并汇总P6全集（合并017/018已跑的291与db，分母为6）的整体账本。
4. **全流程真实成本**：总耗时（安装加载墙钟 + P0波墙钟 + CPU执行耗时 + F/R波墙钟）、四题全部请求的输入/输出token总和（包含首稿失败题消耗的token）。
5. **留出集C6确认规则**：P6属于反复使用的开发集，任何新增解均不得直接宣称为泛化提分；仅当本轮固定下来的“短首稿 + 一次反馈”策略在P6上展现出相对直出基线与独立重启的净新增解且单题时间/token成本可放入12小时预算后，下一轮才在未动过的留出集C6上做**唯一一次独立确认**。

## 6 第三优先级（P3）：B线受约束生成的最小CPU修复与同会话微探针

在推送任何B线GPU请求前，必须在本地完成以下四项修复并通过干跑校验：

### 6.1 修复vLLM约束请求协议与生成预算

1. **改为顶层`structured_outputs`字段**：根据固定版本vLLM（`0.27.2rc1.dev91+g1f7427bc0`）的`ChatCompletionRequest`协议定义，修改[`run_reasoning_probe.py`](../../experiments/018/bv/run_reasoning_probe.py)的`chat_once`：
   - 移除`payload["extra_body"] = {"guided_json": ...}`及请求体顶层的`enable_in_reasoning`；
   - 改为在请求体顶层传入`payload["structured_outputs"] = {"json": schema}`；
   - 将推理阶段免除结构化约束的配置放入`start_server`的服务端启动参数（或通过微探针实测验证固定版本下`<think>`与`structured_outputs`的兼容行为）。
2. **对齐解题预算与微探针预算**：
   - 极小接线微探针（验证服务端是否真正执行JSON Schema约束）：使用`max_tokens = 128`、`enable_thinking = False`，发送一个仅允许返回`{"steps": [{"op": "output", "params": {}}]}`的请求，检查返回内容是否为纯JSON且对非法schema是否真实拒收/约束；
   - 正式解题调用（`291dc1e1_0`、`db118e2a_0`）：将`GEN_MAX`从`4096`恢复为`12288`（配`THINK_BUDGET = 8192`），确保8192思考完成后仍有4096 token余量输出受约束的JSON操作序列。

### 6.2 统一`choices` Schema并补齐变尺寸画布原语

修改[`experiments/018/bv/opspec.py`](../../experiments/018/bv/opspec.py)（或在`experiments/019/congen/opspec.py`中演进），解决表达能力与协议脱节问题：

1. **统一`guided_schema()`与`validate_template()`的`choices`协议**：
   - 在`guided_schema()`生成每个参数属性时，将原本仅允许标量的定义包装为：
     ```python
     scalar_schema = p(k, *v)
     param_schema = {
         "anyOf": [
             scalar_schema,
             {
                 "type": "object",
                 "properties": {
                     "choices": {
                         "type": "array",
                         "items": scalar_schema,
                         "minItems": 1,
                         "maxItems": MAX_CHOICES_PER_PARAM,
                     }
                 },
                 "required": ["choices"],
                 "additionalProperties": False,
             },
         ]
     }
     ```
   - 确保模型在解码约束下能够合法输出`{"choices": [...]}`供CPU端的`search_template`搜索。
2. **补齐变尺寸输出、裁剪与对象排列/拼接原语**：
   - 针对`291dc1e1`（提取框内多色矩形对象、按长边方向排序、竖向对象逆时针旋转90°后居中纵向拼接为变尺寸网格）与`db118e2a`（提取并归一化到固定尺寸画布铺放），扩展操作原语：
     - `crop_region` / `extract_objects`：支持框内子区域提取及保留多色矩形块；
     - `normalize_objects`：支持“若`height > width`则逆时针旋转90°”的条件朝向归一化与局部坐标归零；
     - `pack_objects` / `place_objects`：支持指定目标画布尺寸（`height`、`width`）或按选中对象序列自动纵向/横向拼接（`layout: "stack_v" | "stack_h"`，对齐方式`align: "center" | "left" | "right"`，填充背景色`background`）。
   - **本地真实题面干跑门控**：不用手造玩具网格充当充分条件，必须在本地CPU上用新`opspec.py`构造出能**4/4通过`291dc1e1`全部train示例**的操作模板，确认语言表达能力确实覆盖目标题后，才允许进入GPU模型生成测试。

### 6.3 与P2会话合并验证，不单独开长GPU会话

待上述CPU修复与291真实demo干跑通过后，将B线的“1条128-token约束生效微探针 + 2题受约束生成”挂接在P2（或P1演练）已启动的vLLM服务末尾顺带执行，省去单独一次20分钟的模型加载开销。

## 7 拟新增与修改文件清单

| 阶段 | 文件路径 | 操作类型与说明 |
| --- | --- | --- |
| 文档交接 | `docs/plan/019-submission-and-first-draft-pipeline.md` | 新建：本计划文档（019唯一实施入口） |
| 文档交接 | `docs/progress/019-submission-and-first-draft-pipeline.md` | 新建：记录018复核结论、四项阻断缺陷定位与019分阶段进展 |
| 文档纠偏 | `docs/progress/018-f291-feedback-and-constrained-generation.md` | 修改：更正v1–v5“存储卷stall”归因为`sha256_file`哨兵`1`死循环与62位哈希常量，补齐三笔token成本账本 |
| 阶段0修复 | `experiments/018/aq/vdriver018a.py`、`experiments/018/bv/vdriver018b.py` | 修改：修复`sha256_file`哨兵为`b""`，恢复64位`VLLM_WHEEL_SHA256` |
| 阶段0 + P1 | `experiments/019/submit/submit_lib.py` | 新建（继承017）：新增候选池去重/排序选槽`merge_candidate_pool`与覆盖率报告`coverage_report` |
| P1全量提交 | `experiments/019/submit/vdriver019_submit.py` 及 `notebook/` | 新建：全量challenge输入→C8直出保底→选择性程序/反馈补充→持续原子写盘`submission.json` |
| P2 (+4验证) | `experiments/019/p4/vdriver019_p4.py` 及 `notebook/` | 新建：P6其余四题批量P0（强约束final禁散文）→真实demo执行反馈→F vs R同预算对照 |
| P3 (B线修复) | `experiments/019/congen/opspec.py`、`run_reasoning_probe.py` | 新建/修改：顶层`structured_outputs`、`choices` schema对齐、变尺寸拼接/画布原语及291真实demo干跑 |

## 8 执行步骤与止损门控

1. **Step 1（本地CPU）**：完成阶段0全部确定性修复（`sha256_file`、64位wheel哈希、`submit_lib.py`候选池与覆盖率统计），用有界读取和合成多候选输入人工验证通过。
2. **Step 2（P1提交候选交付）**：完成`vdriver019_submit.py`与提交Notebook打包，核对内嵌代码SHA及无残留缺陷；在核查当前活跃Kaggle账号剩余GPU配额（满足会话上限×2.3 + 6h安全余量）后，用公开challenge输入完成端到端运行，交付完整`submission.json`、真实预测覆盖数、占位数与总耗时。
3. **Step 3（本地CPU B线修复）**：完成`opspec.py`变尺寸原语、`choices` schema与顶层`structured_outputs`修改，在本地跑通`291dc1e1`四个demo的模板搜索验证。
4. **Step 4（P2 +4完整流程与P3同会话验证）**：在单次加载的GPU会话中，先跑1条128-token微探针验真`structured_outputs`，随后批量运行P6其余四题的`P0 → 真实demo反馈 → F vs R`（及B线2题受约束验证），取回后在本地完成四题全分母评分与成本核算。
5. **Step 5（C6独立确认门控）**：若Step 4证明“短首稿 + 一次真实反馈”在其余四题上有明确净新增解且总成本可控，锁定该配置在留出集C6运行一次独立确认并合入最终提交Notebook；若首稿完整率仍低或反馈无净新增解，则提交Notebook保持以直出高覆盖为主、仅在极少数高置信度完整程序上补充槽位。
