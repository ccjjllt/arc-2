# 019 — 全量可提交候选打通、首稿形成率与+4完整流程验证

日期：2026-10-20，北京时间。对应[019计划](../plan/019-submission-and-first-draft-pipeline.md)。复核起点为`efca1ba`（2026-10-20 10:07提交）。

**当前状态：已完成018实验证据与代码缺陷复核，019计划文档已建档；阶段0代码修复、P1全量提交演练、P2其余四题（+4）全流程验证与P3受约束生成修复均尚未运行。** 本轮尚未进行训练、未发布新Notebook、未做正式比赛提交，C6与981评估集未动。

## 1 对018（`efca1ba`）的独立复核结论

### 1.1 `291dc1e1`新增正确解成立，成本分三层记账

- 独立比对[`experiments/018/kaggle-v6/vprobe/run.json`](../../experiments/018/kaggle-v6/vprobe/run.json)中的`F-fix-291dc1e1_0`预测网格与官方`291dc1e1`答案，确认逐格一致（**exact**），且程序在全部4个`train`示例上4/4通过。
- 冻结输入程序[`F-291dc1e1_0.code.py`](../../experiments/018/frozen/F-291dc1e1_0.code.py)原本通过3/4个demo，送入F-fix的反馈（[`frozen-feedback.json`](../../experiments/018/kaggle-v6/vprobe/frozen-feedback.json)）为GPU端真实执行差异（459字符），无人工补丁。
- 同预算独立重启`R2-291dc1e1_0`在`thinking_token_budget=8192`下耗尽`12,288`总token（`finish_reason="length"`），其[`R2-291dc1e1_0.final.txt`](../../experiments/018/kaggle-v6/vprobe/A291/R2-291dc1e1_0.final.txt)包含12,260字符分析文字而无`def transform`或代码围栏，无有效候选。
- 三层token账本明确区分：
  1. 本次F-fix修复调用本身：**9,596 token**；
  2. 加回历史起始程序生成的完整修复路线成本（`9,596 + 22,130`）：**31,726 token**；
  3. 含实验对照R2的整次研究账本（`31,726 + 12,288`）：**44,014 token**。

### 1.2 已定位的四项确定性代码与协议缺陷

1. **`sha256_file`死循环与62位wheel哈希常量**：[`vdriver018a.py`](../../experiments/018/aq/vdriver018a.py)与[`vdriver018b.py`](../../experiments/018/bv/vdriver018b.py)中的`for ch in iter(lambda: h.read(1 << 20), 1):`因`read()`在EOF返回`b""`（永不等于整数`1`）而陷入无限空读。018 v6通过后台线程等待300秒超时跳过才继续执行，[`driver-watch.log`](../../experiments/018/kaggle-v6/vprobe/driver-watch.log)证实该后台线程在后续模型推理期间始终停留在该循环内。此外，两份驱动的`VLLM_WHEEL_SHA256`仅有62位，修完死循环后会立即因哈希不匹配退出（正确64位值见[`experiments/017/kaggle-v1/vprobe/run.json`](../../experiments/017/kaggle-v1/vprobe/run.json)）。已在018进展文档中更正原“存储卷stall”归因。
2. **限制thinking预算不等于交付代码**：R2证明模型在`<think>`受限后仍可在`final`通道继续长篇推理直至耗尽总预算。下一阶段首要指标定为**固定总预算内的首稿完整可执行率**。
3. **B线四重阻断**：
   - 请求字段使用嵌套`extra_body.guided_json`与请求级`enable_in_reasoning`，固定版本vLLM实际协议字段为顶层`structured_outputs: {"json": schema}`，未知字段会被静默忽略而非必返400；
   - `vdriver018b.py`设置`GEN_MAX=4096`小于`THINK_BUDGET=8192`，长思考会先撞总上限；
   - `opspec.py`的`place_objects`只能创建输入尺寸`H×W`画布，无法表达`291dc1e1`和`db118e2a`的变尺寸输出；
   - `guided_schema()`仅允许标量参数，未放行CPU搜索器支持的`{"choices": [...]}`结构。
4. **提交入口双槽覆盖陷阱**：[`submit_lib.py`](../../experiments/017/submit_entry/submit_lib.py)的`update_slots`在收到单个候选时会将`attempt_1`和`attempt_2`同时覆写为该候选，若逐候选直接调用会清除此前已有的不同候选；且需将格式校验通过率与非占位真实预测覆盖率分开报告。

## 2 下一步执行顺序（详见019计划）

1. **阶段0（CPU必改）**：修复`sha256_file`哨兵为`b""`、恢复64位`VLLM_WHEEL_SHA256`、在`submit_lib.py`中实现每题候选池去重选槽与真实覆盖率统计，同步到所有Notebook内嵌代码。
2. **第一优先级（P1）**：打通27B全量提交候选（真实challenge输入 → 保留全部demos与全部query → C8滚动求解 → 候选池持续写盘`submission.json` → 全局截止收尾），用公开输入完成端到端演练。
3. **第二优先级（P2）**：在单次模型加载会话中一次性批量跑完P6其余四题（`4c416de3_0`、`58c02a16_0`、`aa4ec2a5_0`、`fc4aaf52_0`）的“短首稿P0 + 一次真实反馈F vs 同预算重启R”，四题固定在分母中，完整报告`首稿完整率 → 示例全过率 → query exact`及全流程成本。
4. **第三优先级（P3）**：完成B线CPU最小修复（顶层`structured_outputs`、12288/8192预算、变尺寸/拼接原语、`choices` schema），先在本地跑通291示例干跑，再在已加载模型的GPU会话中用极小预算探针验真约束生效后执行任务。
