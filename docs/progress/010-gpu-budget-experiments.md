# 010 — 配额刷新后的研究决策与GPU交接

日期：2026-10-03。对应[010计划](../plan/010-gpu-budget-experiments.md)。本轮完成实时配额/规则核实、只读代码与权重键审计、效率资料更新、CPU物化与固定实验清单；没有启动GPU、Kaggle会话、模型训练/推理或比赛提交。

## 1. 当前判断

先修会导致实验失真的继承/格式接口，然后用共享底座的顺序实验比较原底座、G1、F0/F1小步SFT和有限候选策略。以每个配额小时得到多少可信的完整比较为目标。主模型仍为用户选定的Qwen3.8-27B。

不把009的token减少当成已实现GPU加速，也不把400task覆盖当成已提高泛化。不会为了完成实验表将两种格式都盲训到底。大量实现已写入010计划§8，由用户接手。

## 2. 30小时的真实含义

Kaggle MCP查询时刻2026-10-03 03:53:17 UTC（北京时间11:53:17）：used=0、reserved=0、total=108000秒，next refresh=2026-10-10T00:00:00Z。官方L4×4页面仍写两倍扣额。因此按约15小时机器墙钟分配，而非30小时L4。

证据：`experiments/010/quota-and-rules.json`，含本次官方Code Requirements、Data、Timeline、Upgraded Accelerators内容。评分仍为无网、整Notebook最多12小时、240隐藏task、每query两个候选。[官方算力说明](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators) · [代码要求](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/code-requirements)。

历史时长重新核算：G1纯训练200更新4567.7752秒（均值22.839秒/update），Notebook日志约5045秒；账户差22198.43秒=6.166配额小时。差额未解释，不当稳定扣额倍率。三次模型加载为291.333/535.183/611.137秒，即约4.86–10.19分钟；减少重复加载是直接可识别的固定成本节省。

旧2048评价20query：greedy均值68.736秒、p90约121.18秒；第二sample花1374.05秒无新增命中。旧题多数来自ARC1，这些数只能作历史参照，不能预测长ARC2题。

## 3. CPU审计发现四项实际风险

源码HEAD `1d0b8ab728084335477370616afa5809dcd6a16c`，开始时工作区干净。本轮未修改三个求解/训练脚本。

### 3.1 `--init-adapter`会静默漏掉G1权重

`train_grid_sft.py:836`起按保存权重键与模型state_dict逐字相交。实际G1 safetensors键例如`...lora_A.weight`，训练状态记录为`...lora_A.default.weight`。只读safetensors JSON header与pickle opcode，未反序列化模型：保存512键、训练512键，字面匹配0，正确加adapter名后匹配512。

当前函数打印复制数但不拒绝0，因而有“从fresh LoRA训练却记录为继承G1”的风险。证据独立复核保存在 `experiments/010/adapter-key-audit.json`。这不是已观测到某次G2训练失败：G2尚未跑。已有E0使用`PeftModel.from_pretrained`，不受此缺口影响。

[PEFT官方checkpoint说明](https://huggingface.co/docs/peft/developer_guides/checkpoint)解释保存时移除adapter名；正确加载应使用PEFT映射接口或已验证的from_pretrained路径。修正时保留新阶段独立optimizer/scheduler的设计。

### 3.2 F1的真实缺口在贯通，而非所有训练文本被转回空格

`encode_episode`直接消费已物化messages和target；F1文本确实能进入loss。问题在于：solve第1279/1338行构造原/转置prompt时仍用默认F0，只有解析器改为F1；训练zero-step和FORMATTER_NAME仍写F0；auto仅搜adapter目录，训练run_config却在父目录，旧`space_digits_v1`也不在别名表。

所以直接`--formatter f1_compact`会造成F0提示与F1解析失配，不能作为F1能力实验。需把prompt、目标、诊断、sidecar和解析统一。本轮细化了followup中“train默认仍空格”的概括，没有声称F1全链路已可用。

### 3.3 fair只做了任务公平，独立base身份被丢失

数据保留base_instance_id，但训练Episode没有该字段；picker第403/416/431行用family_id当base，而family_id=task_id。CPU两个独立base同task的例子被统计成1base。本轮400例在数据侧先固定，绕开此问题；不能把009的601task/725base模拟数字当作当前trainer已经实现的行为。

### 3.4 修复pair去重冲突

原/修复的task与instance均相同；episode_id不包含repair variant。实际CPU repair-only物化得到2条original、2条duplicate skip、0条repaired。新主pool使用`--skip-repair`，不把修复对照混入本周格式比较。以后质量A/B应替换匹配原例，不能让两臂同时含原例与修复例。

另外，008/010的自定义清单schema不能直接给旧`--split-file`；solve按task展开全部query、soft time guard、整题后落盘。train没有墙钟上限，多adapter／两格式训练臂共享会话未实现；现成F0 base/G1评价已经共用一次加载。这些实施要求已落实到计划，而不是仅列一个“可以开始训练”的结论。

## 4. 已完成同400例F0/F1物化

用现有CPU物化入口处理D2旧索引＋新2240base两种增强，不纳入repair：

| 来源 | 真实物化行数 |
| --- | ---: |
| 官方索引 | 1147 |
| 旧生成索引 | 1800 |
| 新2240base的两增强 | 4480 |
| 合计 | 7427 |

53条旧LOO不满足最新规则被排除计数，102条同输出suspect按规则保留。临时F0全池SHA `72a4ff920fe22289db2fc31e87cf2abd00628e8a4f9cc525b04c6cb7ed6ce037`；原007/008池未修改。

随后CPU执行 `experiments/010/prepare_paired_pilot.py`，约26.59秒，Python3.9.6/arm64/tokenizer-only；没有模型权重。排除009已记录17个新base的34个增强行，以及两格式共同4K域之外1946行；在其余候选中固定400例并双向还原网格核对。

| 项目 | 已准备值 |
| --- | --- |
| 任务/独立base/episode | 400 / 400 / 400，两种格式相同 |
| 来源 | 官方160、生成240 |
| 历史来源 | 旧ARC1 train69、旧ARC1 eval201、非ARC1 130 |
| 无V2生成器的非ARC1任务 | 32/55入选，不声称全覆盖 |
| F0参考长度桶 | 80/160/160，均完整≤4096 |
| F0 token | prompt669710＋答案/EOS90562＝760272 |
| F1 token | prompt383265＋答案/EOS49209＝432474 |

配对F1比F0少约43.1% full token，但本实验按相同原始例比较，不是相同token/墙钟。25/50/100更新的同seed消费前缀、逐行两格式长度和输出SHA保存到`paired-pilot-manifest.json`。训练器legacy路径的共同shuffle已计算；任何后续改动都应核对实际消费ID。

已落地 `paired-f0-400.jsonl`、`paired-f1-400.jsonl`。这是数据准备完成，不是GPU训练完成；继承、formatter、deadline等修正仍须先做。

随后用当前真实`encode_training_rows → legacy → Random(20261003).shuffle`与本地tokenizer独立复核：两文件均400/400保留，编码错误/诊断排除/超长均为0，F0最大4060、F1最大2211；全400消费顺序和25/50/100前缀统计均与manifest一致。此检查没有加载模型，不证明4K GPU显存可行。

## 5. 已冻结评价与本周预算

`evaluation-manifest.json`只按输入元数据冻结P12（12task/14query）、独立C12（12task/16query）、TTT P6（6task/6query）；H与E分开。P12包含已见的342dd610并有标记。C12在冻结方案前不读成绩，不能边调参边叫确认集。原008清单没变。

另备 `e0-p12-challenges.json` 和 `e0-p12-source-map.json`：将P12两来源题面合并，按来源和S/M/L交替排队，仍12task/14query、原query索引不变、无query答案。这样现成F0 E0可一次加载覆盖P12两个来源；运行时按source map生成独立评分标签文件，逐组结果仍分开报告。

`budget-analysis.json`存原始统计与各阶段上限。名义15小时分为：S0 2.5、两格式25/50更新4.5、单支续100 1、候选/可选TTT 2、冻结确认1.5、余量3.5。余量不预花；实扣异常时缩后续阶段，不并行开作业使账目难以归因。

F0/F1总LR horizon都固定100；先到25/50取读数，避免为两个无收益分支盲跑。对照同初始化/原始样本/顺序，不给F1单独换更容易或更长的池。只在两臂共同checkpoint比较。P12反馈反复使用是开发选择，不作最终泛化证据。

## 6. 效率研究的新判断

### 不裁词表，减少无监督prompt位置的输出头计算

TF5.8的Qwen3.5系实现有`logits_to_keep`，当前训练器未使用。microbatch1且答案为连续后缀时，可以保持完整主干输入，使用T+1个尾部logits和相应labels切片，让HF原有causal shift监督同一批答案与EOS。

G1真实token算术显示，head位置可减少约87.8%；新400例F0则从760272变90962。**这不是整模训练倍数，也不是已经获得显存收益。** 首会话最多15–20分钟复用同一模型，对短/2K/4K样本做原路径与新路径的loss/全部LoRA梯度/实际时间对照；有系统差异即不采用。具体调用与边界见计划§6。

来源：[TF5.8模型实现](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)、[loss实现](https://github.com/huggingface/transformers/blob/v5.8.0/src/transformers/loss/loss_utils.py)。[Cut Cross-Entropy研究](https://arxiv.org/abs/2411.09009)及[作者实现](https://github.com/apple-aiml-research/ml-cross-entropy)说明大词表logits有优化空间，但本周先用原生更小改动，未采用其patch或梯度filtering。

### GDN依赖有更明确的候选，但仍需实测

官方causal-conv1d1.7.0资产存在匹配torch2.10/cu12/cp312/Linux x86_64、cxx11abiTRUE的wheel；资产SHA `8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7`。FLA0.5.2的依赖与API有静态兼容证据。没有下载/安装或GPU验证。本周只在离线wheelhouse已备好、ABI核实后限时probe，不在GPU会话现场编译。

来源：[官方卷积资产](https://github.com/Dao-AILab/causal-conv1d/releases/expanded_assets/v1.7.0)、[FLA0.5.2依赖](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/pyproject.toml)、[chunk API](https://github.com/fla-org/flash-linear-attention/blob/v0.5.2/fla/ops/gated_delta_rule/chunk.py)。不能从包安装成功就认定所有GDN算子加速，更不能从推理通过推断后向可用。

### 更新两项历史信息

- [Unsloth当前依赖](https://github.com/unslothai/unsloth/blob/main/pyproject.toml)已允许Transformers至5.17，TF5.8不再被007记录的上限排除。[balanced多卡说明](https://unsloth.ai/docs/basics/multi-gpu-training-with-unsloth)也存在。暂不主迁移的原因是本项目环境和性能尚未验证，而非继续声称旧依赖硬冲突。
- [FlashQLA](https://github.com/QwenLM/FlashQLA)列出的SM90/100/103/120/121不含L4的SM89。[NVIDIA算力表](https://developer.nvidia.com/cuda/gpus)。不把其Hopper/Blackwell提速声明用于L4预算。

所有外部资料本次查阅日期为2026-10-03。没有找到可直接替代当前主线、已在本项目四L4预算下证明ARC2收益的新方法；不继续罗列高分但不可迁移的模型。

## 7. 交付与未运行边界

本轮新增010计划/进展、预算与评价manifest、配额规则证据、adapter键核对、同400例两格式数据及可重建CPU脚本；README更新当前指向。没有更改现有训练/推理脚本，没有消耗本轮GPU配额。

研究计划在核对范围内可用于实施交接；训练实现仍需修A/B与预算入口，GPU性能和解题效果均未验证。F0现成E0可单独运行；共享训练/多adapter会话、后缀logits、GDN内核、TTT等仍是建议实现/未运行，不能把计划接口写成已经可执行的命令。

## 8. 2026-10-03 CPU衔接补记（§2表A/B/D实现，不含GPU）

- A（`train_grid_sft.py`）：同targets优先`PeftModel.from_pretrained(...,is_trainable=True)`+逐张量验值；异targets用PEFT公开映射（`...lora_A.weight`→`...lora_A.default.weight`）规范化拷贝，字面相交仅诊断永不计成功；`verified==0`或任何缺失/shape/value不一致直接`RuntimeError`终止；optimizer/scheduler/cursor仍新阶段重建。纯CPU标准库复核真实G1文件：`literal=0、mapped=512/512`，与`adapter-key-audit.json`一致。
- B（solve+train）：双视角`build_messages`一律显式formatter；训练数据/CLI二选一统格式并拒混用；zero-step/解析/system日志/sidecar一致；auto兼容`space_digits_v1`，adapter无字段查父目录run_config；`adapter_step_N`各存`formatter.json`，不碰PEFT配置。`f1-train-consume`条改为已实现（CPU，待GPU验证）。
- D（solve+train）：`--query-file`显式query清单（008/010 schema不动，help注jq转换）；逐candidate/query增量JSONL落盘；`dedup_key`判重跳过completed，`budget_interrupted`重跑且不记模型错误；`--wall-clock-limit`仅update边界保存退出。S0–S4编排仍归用户§8 runner，help已写明。
- C按计划绕开：本周格式对照用已预选400行，`--skip-repair`；fair的base身份/repair碰撞记followup未动。`experiments/010`数据本轮只读核验（400配对token/顺序/ID、manifest前缀、P12/C12无交集、e0-p12无答案字段、预算合计15h）全PASS。
- 合规：只改2脚本+followup；007池/e0与010数据未动；无测试文件/CI；无GPU/推理/提交。GPU真机继承值与F1提示确认仍待首次会话按§9先跑F0 E0时核对。

## 9. E0 F0配对结果与扣额实测（`ccjjllt/e0-f0-pair-010` v1 COMPLETE）

- 会话：P12一次加载，`--compare-mode base_vs_adapter --formatter f0_space --max-new-tokens 2048 --n-sample 0 --time-budget 5400`，产物`experiments/010/kaggle-e0-v1/010/e0_f0_pair.json`（+queries JSONL+scoring+log）。
- 记分：运行时 scoring 含全部12 task（Kaggle bundle确有`arc-agi_evaluation_solutions.json`，E24按§9只用于评分、未进prompt）；test项全为input-only prompt包。
- 结果：14 query中13对完整配对，base 0/13、adapter 0/13，无新增无丢失（P12为代表性片：6非ARC1 H+6 E24，此0/0不证明底座/G1无能，只说明本片无信号）；`9aaea919 q0`的adapter臂被guard掐掉待补。parse基本全OK（仅`291dc1e1` adapter与`aa4ec2a5` base各1 FAIL），0截断。
- 用时：solve elapsed 5684s/27候选≈210s均值；短512tok约115s、1800tok约400s。P12完整28候选约需6000–6500s，5400预算偏小是缺臂原因。
- 扣额：会话前后账户差30364s=8.43配额小时，约1.6h墙钟对应**约5倍墙钟**，不是2倍（另有TPU +1004s归因不明，金额小暂不追）。剩余77636s=21.57h。按此倍率S1原文（4.5h墙钟）不可行，需缩范围后重排。
- 初始化决策（探索性记录）：P12无G1正向证据亦无回退；F0/F1对照取双臂fresh同初始化（干净比较，且避开待真机验证的init-adapter路径）。

## 10. 补缺会话结果与扣额修正（`ccjjllt/e0-fill-f1at0-010` v1 COMPLETE）

- F0补缺：`9aaea919 q0` adapter臂完成（parse OK、em False、450s/1800tok）。E0现14/14完整配对：base 0/14、adapter 0/14。产物`experiments/010/kaggle-e0fill-v1/`。
- F1@0零数据：14 query零候选。日志显示run1在t≈51s后静默至t≈6664s（零stderr、零逐query输出），循环启动时预算已爆，14行全记`budget_interrupted`。与E0会话同一签名（首载静默6600s、同会话二次加载约400s正常），判定为首载stall（冷存储读54B权重），非F1格式问题——F1能力与B链GPU验证仍未完成。
- 扣额修正：本轮账差15925s=4.42h，墙钟约7300s≈2.03h，即**约2.2倍墙钟**；E0轮“5倍”是6600s stall也被扣额所致。剩余61710s=17.1h≈7.8h墙钟。S1原文4.5h墙钟仍可执行，但50/100续跑与S2–S4需按剩余重排，不再假设2倍整。
- 教训：solve/train加载前后加心跳打印（stall可见）；会话首载预留波动；F1@0并入下次会话首载后顺手重测，不单独开会。

## 11. S1-25首跑：F1@0全绿、B链通，F0四步OOM转§6.1（`ccjjllt/s1-25-010` ERROR）

- F1@0（底座，`f1_compact`，960上限）：14/14候选全合法、em全0、1684s、无guard。B链GPU验证通过：紧凑提示→紧凑合法网格。
- F0训练：`FAILED_OOM`，8次OOM后停机，仅完成4更新（52–82s/更新，loss 0.066–0.098）；OOM集中于gen|104等3–4K长episode，短/中正常。`adapter_step_4`已保存可续，但4更新无比较价值，F0-25判失败。F1臂与双臂评价未执行。
- 结论：现HF栈+BF16在4×L4上装不下4K完整episode（与§5预判分支一致），不回退到只挑短题。下一步：CPU实现`--logits-mode suffix`+`--probe-suffix-logits`（只改train，默认full不变，非后缀契约拒收，探针输出loss/全LoRA梯度/时间/显存JSON且不自动切换）；GPU先15–20分钟探针（G1非零LoRA一例），通过后再定F0重试，F1-25（最长2211tok）先行。
- 本轮账差4602s=1.28h。累计used 50892s=14.14h，剩余57108s=15.86h。

## 12. 探针+F1-25会话（`ccjjllt/s1-probe-f1-010` COMPLETE）

- 探针FAILED（正确拒收）：调用漏传4096上限，池按默认1024/2048构建，`4k`桶无样→`refusing substitution`。实现按设计拒绝静默替代；suffix数值验证仍未完成。下次调用补`--max-len/--overflow-len 4096`。
- F1-25：`SHORTFALL`（stop-25命中）25/25更新、**零OOM**、loss 0.027–0.473；P12评价14 query、13 OK+1 FAIL、em 0/14。F1在现栈上可训。
- 对比现状：F0-25缺失（4更新OOM版不可比），F1-25完成但0/14。格式比较仍无结论，需F0侧有效checkpoint。
- 本轮账差6690s=1.86h。累计used 57582s=16h，剩余50418s=14h。
- 下会话分支（阈值现定）：探针max rel_err≤1e-3且min cosine≥0.999且loss差≤2倍噪声→F0-suffix-25+评价；否则F1-50 fresh重训+评价（F1可训，S2条件续跑）。判定阈值记录于此，不在会话内臨時改。

## 13. S2分支会话（`ccjjllt/s2-branch-010` COMPLETE）：走F1-50分支

- 探针（G1非零LoRA，short/2k/4k）：loss三例按位一致（diff 0.0）；梯度cross-path maxrel 0.044/0.197/0.103、mincos 0.99950/0.98725/0.99506，同路径噪声 0.0436/0.2199/0.1148、0.99915/0.98775/0.99398——全面落在噪声地板内，无系统差异；时间略快（约1–2%），显存此长度下未见分离（主干主导）。
- 阈值修正：预设1e-3绝对阈值被噪声地板证伪（微小norm张量同路径噪声本就~4e-2–2e-1）。按计划“无系统差异则采用”规则，suffix路径数值等价成立，允许用于F0重试；字面阈值未达标故会话内仍走F1分支，记录在此。
- F1-50：fresh重训50/50零OOM、loss最低0.0266（第38步）；P12评价13 OK+1 FAIL、em 0/14。F1训练稳定但25/50均无P12信号。
- 本轮账差29528s=8.2h（探针+F1-50+评价）。累计used 87110s=24.2h，剩余20890s=5.8h≈2.6h墙钟。本周期只够最后一个精简会话：F0-suffix-25+评价。

## 14. S3会话（`ccjjllt/s3-f0-suffix-010` ERROR，但训练本身成功）

- suffix路径17/17更新**零OOM**（full路径12次尝试8次OOM），终结了F0的显存阻塞。停机原因是我的`--wall-clock-limit 5400`后备先到（`SHORTFALL/wall_clock_limit`），`adapter_step_17`已保存；会话ERROR是我run_cell硬编码检查`adapter_step_25`所致，非训练失败。两个bug都记在此，下会话修正。
- 本轮账差11000s=3.06h。累计used 98109s=27.25h，剩余9891s=2.75h≈1.2h墙钟。
- 本周期最后一次会话：F0-suffix-25纯训练（不评价，快照检查按实际步数，wall-clock放宽到7200），约50分钟墙钟≈2h配额。评价与格式比较留待10-10刷新后。

## 15. S4会话与周期收官（`ccjjllt/s4-f0-train-010` COMPLETE）

- F0-suffix-25：25/25更新**零OOM**（stop_after_update命中，`SHORTFALL`正常退出），loss 0.012–0.146、末步0.020；`adapter_step_25`已保存。suffix路径彻底解决4K训练显存阻塞（full路径12试8OOM vs suffix 93次更新零OOM）。
- 账差11811s=3.28h。累计used 109921s，超出108000s总额约1921s（0.53h）——会话内超支，本周期无剩余。下次刷新北京时间10-10 08:00。
- 本周期交付：E0配对14/14（0/0）、F1@0链路、B链、探针数值等价、F1-25/50（0/14）、F0-suffix-25权重。**缺的唯一一块：F0-25的P12评价**——下周期首会话即评，双臂25步格式比较闭环。C12、50步、TTT/TTA、GDN均未启动。
- 烧额复盘（配额秒/墙钟）：E0 8.43h（含首载stall）、补缺4.42h、S1首跑1.28h、探针F1轮8.2h、S3训练1.28h（SHORTFALL早停省了）、S4 3.28h。首载stall约6600s出现两次（E0、补缺轮），S3/S4未出现；2.2倍是常态，5倍是stall假象。
