# 018 — F-291冻结程序反馈对照与受约束程序生成

修订日期：**2026-10-19，北京时间**。承接[017实际结果](../progress/017-program-feedback-and-submission-entry.md)§8与用户三项决策：保留C8、暂缓16K扩量和P12；A线复用已冻结的F-291完整程序做一次反馈vs重启（只新增2次调用）；B线把"受约束的程序生成＋可靠的对象操作"作为主任务；修复改小编辑（第3项）优先级排后、本轮只留设计。

本文件是唯一实施入口。A线GPU会话与B线本地CPU实现可并行；A只决定反馈分支是否值得扩跑，不卡B。

## 1 研究问题与继承结论

| 已核实事实（017 v1） | 本轮含义 |
| --- | --- |
| P0两题均被12288截断，无完整P0，门控关闭+4 | 017原问题[P0,F] vs [P0,R]未被回答；不得把后续结果记成原P0流程成功 |
| F-db程序4demo全过＋query exact（通用写法，非硬编码） | 程序路线走得通；但db在016直出已命中，不算新题 |
| 291三份程序输出全部形状错失（F 26×6、R 28×8 vs 答案18×8），失败定位为多色物体被拆成同色小块＋重叠外接框重复取内容 | 291是A线唯一标的；B线原语针对此失败设计 |
| 6次调用3次撞12288上限；长程序生成agg约22tok/s | 12288对从零程序不够；但用户要求避免围绕同两题反复调token上限——A线**不调上限**（沿用12288），B线用结构约束治本 |
| 请求层无structured_outputs/guided参数（已grep确认）；CPU只有Python执行器，无对象原语 | B1/B2确为缺失机制，需新建 |

A线只回答：**已有完整程序时，一次真实demo反馈修复是否值得（相对同预算独立重启）**。部署成本必须加回生成起始程序的费用（沉没：P0-291 12288 ＋ F-291 9842 ＝ 22130 completion token）。

## 2 A线：F-291冻结程序的一次反馈对照（GPU，2次调用）

### 冻结物（`experiments/018/frozen/`，SHA记录）

- `F-291dc1e1_0.final.txt`（5345字节，v1原文，含散文＋代码块）与`.code.py`（5330字节，156行，本地`ast.parse`通过、不执行）。
- 不得改动一字节；反馈里不加入任何人工补丁。

### 会话流程（vdriver018a，会话2700秒）

1. 沿用017安装/服务8/C8链路；执行器预检（恒等函数）不过则基础设施停止。
2. 在GPU上**重新执行冻结F-291程序**于291全部4个demo（`execute_grid` 2秒/512MB、程序累计20秒），得真实`demo_results`；用`demo_feedback(demos, results, 3000)`构造反馈（先断言4条对齐）。这一步零模型调用，反馈必为真实执行结果。
3. F-fix调用：base prompt（017同builder、291干净题面）＋冻结final原文＋反馈；seed **5103**，12288上限、thinking≤8192、xhigh。
4. R2重启：base prompt only，同预算（同上限/thinking/seed 5103中新抽样——与已有的R-291/seed5102不同，避免复刻）。
5. 两调用并发（C8），执行各自输出（demo＋query），落盘复用017格式；不做[F,R]混合挑选。
6. 上限：新增模型调用恰2次，`completion_tokens`预算上限24576（超了记超预算事件但不截断会话证据）。

### 门控与记账

- 本地评分后：若F-fix query exact（旧组合没有的正确query）→ 开+4（起始程序＋反馈vs重启，协议另定，预算另批）；若仍无新增解 → **暂停这条线扩展**。
- 投入标尺：每GPU小时新增正确题数。A会话成本（配额小时，约0.7–1h）＋沉没22130 token一并记录。

## 3 B1：final改为受限操作列表（解码阶段约束）

保留思考阶段；只约束最终输出。请求经`/v1/chat/completions`的`extra_body`传约束（现有`chat_once`无此参数，需在018 vendored副本上加**唯一小补丁**`extra_body: dict | None`，只读透传；SHA与diff记录）。

约束要点（必须做实，逐条可验）：

- `enable_guided`只作用于final，`enable_in_reasoning=False`（推理过程不受约束）。
- Schema：顶层对象只含`steps`数组（`additionalProperties: false`）；每步`{"op": 枚举, "params": 对象}`；`max_steps=12`；数值有界（如平移−29..29、旋转0..3、每步参数白名单）。
- **禁止**`analysis`/`explanation`/`code`及一切自由字符串字段；所有对象层`additionalProperties: false`。
- 空白说明：JSON空白耗费仍受`max_tokens`硬上限约束（与现状同），schema禁的是额外字段泄漏，不是空白；如实记录，不夸大。
- CPU收到后先做独立结构校验（自写递归校验器、stdlib only），非法记`invalid_plan`不执行。
- 若固定vLLM版本拒绝约束字段（unknown field/400），记`structured_unsupported`并停止——不回退成无约束重跑（回退会污染对照）。

第一轮保留思考8192预算：只判断约束机制是否让"交出可执行结构"的比例上升，不指望缩短思考或保证规则正确。

## 4 B2：可靠对象原语＋多解释参数搜索（CPU）

### 原语（`experiments/018/congen/opspec.py`，stdlib only，小子集移植）

| 操作 | 行为约定 |
| --- | --- |
| `extract_objects` | `same_color: bool` × `connectivity: 4/8`；背景取众数色（第一版假设，如实记录局限）；返回对象列表，每对象保留**真实格点**`cells:[r,c,v]`＋派生bbox，不把整框当对象 |
| `transform_object` | 旋转（quarter_turns 0..3，逆时针为正）/镜像（axis）/平移（dr,dc），坐标约定统一文档化 |
| `filter_sort` | 按面积/行列位置筛选排序，`order`枚举 |
| `place` | 拼回网格：`background`显式、重叠→error、越界→error |
| `output` | 选择输出值（grid或某对象 raster） |

语义参考michaelhodel/arc-dsl的对象表示与几何操作（查阅后记来源与日期），只移植上表子集。

### 模板＋搜索

- 模型输出短模板：固定op序列，部分参数可为`{"choices": [...]}`候选。
- CPU枚举笛卡尔积，**上限32个完整候选、每题CPU总计≤10秒**（建议上限，实测记录实际耗时）。
- 同一候选配置必须**一次通过全部demo**，禁止每demo各选最优；边框按位置识别，禁止按颜色整删（物体内部可能同色）。
- 第一版覆盖对象提取/排列/变换/组合类；不宣称覆盖ARC2全域。

### 本轮交付

- `opspec.py`：schema常量、结构校验器、执行器、搜索器。
- 本地干跑：手造微型网格（非隐藏答案、非P6答案），断言提取/变换/搜索/越界拒绝行为；CPU耗时实测记录。
- B-verify GPU会话（另推小会话，schema＋约束＋模板提示，1–2题验证"可执行结构率"）：待本地干跑通过后再批预算，不与A会话合并。

## 5 第3项（有界程序编辑）：本轮只留设计，不实现

结构化程序就绪后，反馈输出改为`{"edits": [{step, op/params}]}`（最多改2–3步，参数仍受枚举/类型约束）；主控应用编辑得完整程序、重跑全demo，失败编辑不覆盖原候选。适合局部规则错；思路全错仍需新候选。排在B1/B2之后。

## 6 会话与配额门

- A会话2700秒：门＝2.7ks×2.3≈1.73h＋6h reserve≈7.73h；执行前重查ccjjllt余额。
- B-verify会话另批（小会话，门同公式）；本地CPU工作不耗配额。
- 不训练、不发布Notebook以外产物、不提交比赛；C6/981不读。

## 7 拟改动文件

- 新建：`docs/plan/018-*.md`（本文件）、`docs/progress/018-*.md`；`experiments/018/frozen/`（F-291两文件＋SHA）；`experiments/018/aq/vdriver018a.py`＋notebook（vendored probe＋`extra_body`零改动——A线不用约束，保持与017一致）；`experiments/018/congen/opspec.py`＋干跑报告。
- 018 vendored `run_reasoning_probe.py`：B-verify用副本才加`extra_body`补丁；A线副本与017一致（SHA`8970e7a7…`可直接复用，但仍独立存放以便核对）。

## 8 评分与报告规则

- 答案比对只在本地；P6为反复用开发集，不报公榜/泛化。
- A线报告：两调用finish/parse/demo/query四元组、EM（F-fix vs R2）、沉没＋新增token、配额小时、新题/小时；明确写"原P0流程未成功"限定语。
- B线报告：可执行结构率、非法计划率、CPU搜索耗时、demo全过率；失败案例定位到op/参数，不笼统归因。
