# 019 — 全量可提交候选打通、首稿形成率与+4完整流程验证

日期：2026-10-20，北京时间。对应[019计划](../plan/019-submission-and-first-draft-pipeline.md)。复核起点为`efca1ba`（2026-10-20 10:07提交）。

**最新状态（020复核更新）**：阶段0及部分代码修复已完成，P2（+4）与P1公开演练已运行，实际结果见§5–6；未新增正式比赛提交。P1真实覆盖160/172、双槽不同0；用户附件复核固定双槽0/158，非本轮独立重评分。旧主C6部分题已进入P1推理，不能继续称未动。后续以[020计划](../plan/020-score-validation-and-real-submission.md)为准，不再原样扩跑低预算直出。

以下§1–3保留当时研究及实施记录；其中“未运行”“确认集未动”只描述相应阶段，不能覆盖后续实际运行与本次纠正。

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

## 3 019实现subagent工作记录（2026-10-20，CPU本地，未运行GPU）

本节记录019首轮实现subagent的改动与人工核对结果。**GPU实验未运行**（未推送Notebook、未训练、无正式提交）；C6与981评估集未动。

### 3.1 阶段0确定性修复（018旧文件，仅改两缺陷点）

- `experiments/018/aq/vdriver018a.py`：`sha256_file`哨兵`1`→`b""`；`VLLM_WHEEL_SHA256`恢复64位`5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a`。后台300秒超时跳过逻辑按任务要求在018旧文件中保留未动（只改两缺陷点）。
- `experiments/018/bv/vdriver018b.py`：同上两处修改。
- grep确认：019新建文件+018两旧文件均无`iter(lambda: h.read(1 << 20), 1)`残留、无62位哈希残留；64位常量在`vdriver019_submit.py`、`vdriver019_p4.py`及两旧驱动中均存在。

### 3.2 P1新建文件（`experiments/019/submit/`）

- `submit_lib.py`（SHA`f2b4d4db4a30…`）：AST核对8个017共享函数逐字一致，仅新增`merge_candidate_pool`（all_demos_pass优先→demo_matches降序→来源program_fix>program_init>direct_grid→order升序，去重后≥2取前2、=1重复填、=0留`[[0]]`）、`coverage_report`（format_ok/单槽真实/双槽相异/占位分开报）、`_is_placeholder`与`SOURCE_PRIORITY`/`PLACEHOLDER`常量。绝不用query真值。
- `vdriver019_submit.py`（SHA`c8e24fa5…`）：全局单时钟`session_deadline`，T-600停新请求、T-300硬收尾原子写盘；阶段1直出（C8/4096/think2048/low/seed7101）+阶段2选择性程序首稿与一次反馈（12288/8192/xhigh/P0 seed5101/F seed5103）；`on_slot_done`回调持锁更新候选池并`atomic_json`持续写盘`/kaggle/working/submission.json`；同步`sha256_file`校验、无后台跳过线程。
- `run_reasoning_probe.py`（SHA`7aac22b6…`，与018/aq一致）、`solve_program.py`（SHA`03aaa381…`，与017一致）：vendored副本供Notebook内嵌。
- `notebook/arc2-019-submit-full.ipynb`（179,517字符）+`kernel-metadata.json`（id`ccjjllt/arc2-019-submit-full`）：内嵌4文件base64，经核对解码后与源文件逐字节一致；运行时从`/kaggle/input`自动解析`train/test`结构challenge JSON；会话39600秒。
- 人工核对：`python3 -c`验证有界读取、候选池排序（含来源优先级平局、去重保高优先级、单候选重复填、空池占位）、coverage三分（format3/single2/double1/placeholder1）、`update_slots`单候选双填兼容；`py_compile`通过。

### 3.3 P2新建文件（`experiments/019/p4/`）

- `vdriver019_p4.py`（SHA`3afe652b…`）：固定分母4题（`4c416de3_0`、`58c02a16_0`、`aa4ec2a5_0`、`fc4aaf52_0`，题面`p6-challenges.json`，SHA`03c5c50b…`与017冻结值一致；实测train为3/3/3/2个demo，驱动按实际长度执行不断言4）；P0用max_tokens12288/thinking8192/seed5101+final禁散文强契约（首行` ```python `）；Step1批量P0→真实demo_feedback（≤3000字符）；Step2对完整题F+R（seed5103）并行、不完整题仅R；双槽A_F=[P0,F]/A_R=[P0,R]，F改坏时P0保底；本地漏斗`funnel_local`记录gen_ok/finish/demo_matches/all_demos_pass，query exact留待取回后对照010 scoring.json。
- `run_reasoning_probe.py`、`solve_program.py`：同P1 vendored副本；`p6-challenges.json`：自017复制。
- `notebook/arc2-019-p4-plus4.ipynb`（200,995字符）+`kernel-metadata.json`（id`ccjjllt/arc2-019-p4-plus4`）：内嵌4文件，解码一致；会话5400秒。

### 3.4 P3新建文件（`experiments/019/congen/`）

- `opspec.py`：继承018全部6操作与搜索器；`guided_schema()`全部26个参数包装为anyOf[标量,{"choices"}]；新增`crop_region`（框内子区域归零提取）、`normalize_objects`（高>宽逆时针旋转90°并归零）、`pack_objects`（stack_v/stack_h/canvas+align/background/spacing，显式height/width，0=auto）、`place_objects`扩展height/width（blank画布显式尺寸）；输出门控放宽为place或pack其一+output。
- `run_reasoning_probe.py`：`chat_once`/`chat_wave`改用顶层`structured_outputs={"json": schema}`，删除`extra_body.guided_json`与请求级`enable_in_reasoning`（残留字样仅见于注释说明删除原因）；`--max-tokens`默认改为12288（微探针另传128且关闭thinking）。
- 人工核对：schema全参数anyOf包装；choices往返`validate_template`；normalize（4×1竖条→1×4横条）+canvas铺放；stack_v两对象拼接得变尺寸4×3画布；legacy模板`search_template`仍all_pass。**291真实题面4/4干跑门控未做**（新原语仅合成网格验证），进入GPU前必须先补。

### 3.5 未运行项与下一步交接（首轮实现时）

- 未运行：P1端到端演练、P2四题全流程、P3约束微探针与291干跑；Notebook均未推送；C6未动。
- GPU手动执行顺序（用户执行）：先推`arc2-019-p4-plus4`（约90分钟会话，摊薄加载成本，首跑附带1条128-token无thinking `structured_outputs`微探针验真约束）；通过后再推`arc2-019-submit-full`（约11小时会话，开始前核对活跃账号剩余额度满足会话上限×2.3+6h余量）。
- 已知局限：P1提交Notebook的challenge文件名解析为启发式（列出候选并断言），若官方文件名不同需按实际日志调整；P2四题train demo数为3/3/3/2（非4），评分分母仍为4题。

## 4 本人第三遍复核后亲修（2026-10-20，CPU本地，GPU未动）

针对独立review的三项阻断，两处小修已落实，一处P3门控经 bounded 实证判定为当前语言不可闭合，不再卡P1/P2。

### 4.1 P2两处小修+Notebook重打包

- `experiments/019/p4/vdriver019_p4.py`：F提示由嵌`p0["res"]content`全文改为嵌`p0["parsed"]code`（`p0code`变量），符合§5.3“携带P0完整代码”契约，避免散文污染反馈。
- `"d differ"`同步改名为`RENDER_DIFFER`：`p4/vdriver019_p4.py:398`、`congen/run_reasoning_probe.py:456`、`p4/run_reasoning_probe.py:441`、`submit/run_reasoning_probe.py:441`四处一致。
- 重打包两份Notebook内嵌blobs并更新`expected` SHA：p4 driver `4603a2d7…`、两份probe `6c76df00…`；解码回检与源逐字节一致、无哨兵残留、`py_compile`全过；`grep p0final|"d differ"`在`019/`零命中。

### 4.2 P3-291真实4/4干跑：bounded实证失败（负结果，证据保留）

- 方法：`same_color=False/conn=4`抽取（bg=8众数）+`filter area_lt 17/18`去帧 + 单一`sort` +`normalize rotate_tall_ccw`+`pack stack_v center/8`+`output`，`search_template`上限32候选/10秒，同一配置须全过4 demo。
- 结果：`sort=row/asc`过demo1+demo2（2/4），`sort=col/asc`仅过demo0（1/4），其余排序0/4；无单一静态排序4/4。
- 根因：F真解排序是条件的——demo0内区19×7（h>w）走列优先，demo1/2内区7×10/7×27（h<w）走行优先，demo3内区7×22但帧在右侧（col_dir=-1）需行内逆序；且demo0输出含1/2帧碎片，多数派bg抽取的合并/分裂与F的0角洪水+包围盒分组不一致。当前9原语无“按长边/帧侧切换排序”与“动态内区裁剪”，`choices`只能展开候选不能给各demo不同排序，故同一模板不可能4/4。
- 结论：B线现语言对291不闭合。P3-291任务验证不得进GPU；128-token无thinking `structured_outputs`微探针可随P2会话附带先行（与任务解耦）。B线后续需新增条件排序/动态裁剪原语或转向db/其它题，A线程序路线不受影响。

### 4.3 当前状态（2026-10-20 12:3x，P2已跑完，GPU COMPLETE）

- 配额快照（`kaggle quota`实测，刷新2026-10-24T00:00Z）：推前ccjjllt剩10.15h；P2跑后剩9.17h（消耗约0.98h，1742秒墙钟≈0.48h，账单倍率≈2.0×，非2.3）。teammatea剩2.34h；teammateb(=qingyu1232)剩15.19h（未动）。P1的39600秒（11h）按2.3+6需31.3h，当前三号均不够，先记结论等刷新/拆分。
- P2推送：`ccjjllt/arc2-019-p2-plus4` v1 COMPLETE，elapsed 1742.1秒（29分钟），其中安装加载700.3秒（占40%，较018v6的1193.8秒省约8分钟，同步哈希去跳过线程生效）；P0波435秒/35783 token，F/R波601秒/69464 token；模板`RENDER_DIFFER`通过；产物归档`experiments/019/p4/kaggle-v1/vprobe/`。

## 5 P2（+4）完整流程验证结果：反馈0/4，独立重启1/4（aa exact）

- 首稿完整率：P0 `gen_ok` 3/4（4c/aa/fc正常`stop`；58c `stop`但`program_rejected: forbidden AST node Pow`，模型用`**`写`10**9`被执行器拒绝，未进反馈，仅发R）。
- 示例全过率（/4）：P0 0/4（有程序但各0中）；F 0/4；R 1/4（仅`R-aa4ec2a5_0` 3/3全过）。
- Query exact（本地对照`010/scoring.json`，/4）：P0 0/4；F 0/4；R 1/4（`aa4ec2a5_0`与官方答案逐格一致）；双槽`A_F=[P0,F]` 0/4，`A_R=[P0,R]` 1/4。合并P6全集（含017/018的291 exact与db程序exact）：反馈臂新增仍仅291一题，重启臂新增aa一题。
- 全流程成本：11次调用（P04次+F3次+R4次），completion共105247 token；58c-P0因Pow被拒不计入反馈；fc-F `program_ready`但query无合法网格（执行失败）。
- 研究结论：**“短首稿+一次真实反馈”在本批+4上未重复291的新增解（F 0/4）；净新增解来自同预算独立重启（R 1/4 aa）。** 最终禁散文契约把首稿完整率抬到3/4（优于017/018截断），但0示例全过说明完整≠正确。Pow禁令误伤一个完整首稿，提示词/执行器需协同（禁`**`或放行纯常量幂）。
- C6门控（§5.4.5）：触发C6要求“短首稿+一次反馈”相对直出/重启有净新增且成本可进12h。本轮反馈臂净新增为0，不满足，**C6不跑**；aa的R解法可作为高置信程序槽位并入提交策略，但不宣称泛化提分。R-aa程序归档`experiments/019/p4/frozen/R-aa4ec2a5_0.code.py`（89行，1连通域+孔洞填充写法；本地沙盒`executor_unavailable`无法复验，GPU侧3/3+exact为准）。

## 6 P1试跑（90分钟盒，teammateb身份，GPU运行中）

- P1全量11h按2.3+6需31.3h，三号当前均不够；先推时间盒试跑验管线，不宣称覆盖率。
- CPU改动：`submit/vdriver019_submit.py`程序提示追加首行围栏契约+`**/pow()`禁用（58c-Pow教训），重打包full notebook（driver `b5282e9e…`）；试跑为同blobs、boot仅改`session_seconds 39600→5400`+`wait 41400→5580`，id `qingyu1232/arc2-019-p1-pilot`（与teammateb=qingyu1232身份匹配，剩15.19h，门限9.45h通过）。
- 已推：`qingyu1232/arc2-019-p1-pilot` v1（无标题派生警告），后台`logs --follow`跟进` /tmp/arc2-019-pilot-logs.txt`中。
- 试跑COMPLETE（~05:53 UTC，elapsed 5105.7秒/85分钟）：**安装加载合计662.5秒，已含安装262.5秒，不能重复相加**；阶段1直出172调用/3756秒/541310 token（144 tok/s）；阶段2P0排入172调用仅681秒后被全局截止中断（`usage_missing` 164/172），`stage2_fix`跳过；`submission.json` 120 task/172 query全部格式合法；单槽真实覆盖160/172（93%），占位12/172，双槽相异0/172。产物`experiments/019/submit-pilot/kaggle-v1/`。
- **并发纠正**：按请求发出/完成时间重算，最大在途8、平均7.8768、单请求平均172.01秒；3756/172≈21.8秒是流水线完成间隔。此前“平均并发1、服务端串行”的判断错误，不再为恢复C8开实验；程序长请求须单独实测，不能据直出吞吐保证11小时容纳两个全量程序波。见[020复算](../../experiments/020/evidence/019-manual-review.json)。
- 12个占位含`aa4ec2a5_0`。R-aa是已知开发题的正确程序，但不能把其硬编码注入称为泛化互补；新的候选生成和双槽选择须在不同题上验证。用户附件评分0/158意味着93%覆盖不足以支持当前直出配置继续扩跑。
- **后续决策已由020替换**：用户报告zhuating有完整GPU配额，先修输入选择、S2逐结果执行、完整反馈、task复用、候选排序与截止，再做D24/C24和正式提交。取消“等10月24日原样跑11小时”的安排。019 pilot已推理主C6公共evaluation的3task/5query及备用3task/4query；旧未运行确认身份退役。
