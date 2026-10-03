# 010 — 30小时配额内的GPU实验与实现交接

日期：2026-10-03。核实结果、已准备文件和来源见[同名progress](../progress/010-gpu-budget-experiments.md)。本计划用于用户接手实施与手动执行；表中GPU实验均未运行。固定Qwen3.8-27B、4×L4，不新增测试套件/CI。

## 1. 目标和预算单位

本周目标是用尽量少的加载和无效生成，回答：G1是否优于底座；紧凑F1能否在保持解题的同时节省成本；专项SFT是否改善新题；第二视角与小步TTT是否值得预算。不是尽可能多开Notebook、增加候选数或把每张卡利用率拉满。

实时配额：GPU used=0、reserved=0、total=108000秒，下次刷新UTC10月10日00:00，即北京时间08:00。按当前官方L4×4两倍扣额，**30配额小时≈15小时四卡机器墙钟**；不是30小时L4，也不是可分配120个独立GPU小时。12小时是单个比赛评分Notebook上限，另一个口径。

先承诺11.5小时墙钟、留3.5小时余量。007的6.17小时是账户配额差，纯训练1.269小时、日志约1.401小时，差额未解释；不能把6.17当训练墙钟，也不能按该账差推定新倍率。首次会话前后只保留一个活跃作业，核实扣额后再释放余量。

## 2. 先修会使实验失真的接口

以下是研究前提的具体修正，不是工程门禁；不需要建设测试框架。最少完成A/B/D即可开展继承式F0/F1对照；C可暂通过本轮预选配对数据绕开，修复数据支线本周不必启动。

| 优先级 | 现状与依据 | 最小修改及人工核对 |
| --- | --- | --- |
| A：G1权重继承 | `load_init_adapter_weights`直接按state_dict键相交；实际512个保存键与训练键字面交集0，补adapter名后512 | 同targets优先使用已有`PeftModel.from_pretrained(..., is_trainable=True)`；需拷贝时用PEFT公开state映射API。检查预期512个继承张量全部加载、值一致；复制0或非预期缺失立即终止。optimizer/scheduler/cursor仍按新阶段重建 |
| B：formatter贯通 | 物化F1文本能进入loss，但solve原/转置prompt仍两参调用F0；训练诊断和元数据仍写F0 | `build_messages`两个视角均显式formatter；训练从数据/CLI确定统一格式并拒混用；zero-step、解析、system日志、checkpoint sidecar一致。兼容旧`space_digits_v1`别名，每个`adapter_step_N`也保存格式；不要把自定义字段随意塞入PEFT配置 |
| C：task/base与repair身份 | Episode丢失base_instance_id；fair把family_id当base；repair原/修复episode_id碰撞，实测repaired写入0条 | 保留`family_id=task_id`，新增(source,task,base_instance_id)独立键；repair identity带variant但保留parent lineage。用实际物化→编码→选择路径人工核对。本周格式对照使用已预选的400行，不依赖此fair实现；`--skip-repair`避免混入错误质量对照 |
| D：预算、落盘、清单 | solve预算只在query前看，train无墙钟上限；solve只在整task完成后写结果；008清单schema不能直接作split-file | 增加显式query清单、按candidate/query落盘、跳过已完成项；在完整optimizer update边界提前保存退出。预算含setup/load/JIT/save，不能靠超时杀死optimizer.step。预算中断单独记录，不把未完成query当模型错误 |

最小GPU人工检查是加载一次真实G1，确认adapter继承值和实际F0/F1提示；本地compile/help或假tokenizer不能代替这些检查。A/B未修时，可先做已有F0 E0，但不要运行标成“继承G1”的训练或F1成绩比较。

## 3. 已冻结的评价清单

文件：`experiments/010/evaluation-manifest.json`，seed20261003，只从008已有输入侧分层元数据按SHA选择。没有查看query答案，原008清单不变。

| 清单 | task/query | 用途 |
| --- | --- | --- |
| P12 | H-new 6/7＋E24 6/7，共12/14 | 所有开发checkpoint、格式、视角的固定比较。`342dd610`明确是已见题 |
| C12 | H备用6/7＋E备用6/9，共12/16 | 初始化、格式、checkpoint与解码策略冻结后，独立确认一次 |
| TTT P6 | P12每来源每层1题，共6/6 | 条件性的0/8更新小实验 |

P12/C12无交集。所有query原始索引保留，多query不能只取test[0]。H与E分开计分；exact-match按query，统计不确定性按task聚类。P12反复用来挑方案后是开发数据，不称盲测；C12一旦反馈用于调参就失去确认身份。

首次GPU会话先得到P12的base/G1完整配对，再视时间补H-new/E24剩余题。H25历史桥接后置，不重复旧诊断8题及旧20个G1 greedy。未完成清单显式列出，不按时间截止前的容易题估算全体成绩。

评分时Kaggle官方solutions仅供独立评分路径按选中ID/query访问，绝不进入生成prompt、选样、训练或TTT；不要打印C12及其余备用答案。仅题面运行的EM应为null，不能伪装为已评分实验。

## 4. 已准备的F0/F1同样本对照

`experiments/010/paired-f0-400.jsonl`与`paired-f1-400.jsonl`已在CPU物化。它们包含**相同400个原始episode、400task、400base**，只是网格序列化不同；原始网格经双向转换核对。来源160官方/240生成，非ARC1任务130个，其中32个无V2生成器。

共同可行域为两种格式完整序列都≤4096，按F0长度选80/160/160行。F0共760272 token，F1共432474 token。这轮只回答相同训练例下的格式效果，**不同时把F1额外能装入的长题加入训练**。

来源是D2索引物化＋2240新base各两增强，总7427行；排除009已标记的17个新base对应34行、排除共同域外1946行，再在剩余候选按来源/task/base选择。本轮没有把修复pair混进来，也不声称剩余候选全部语义认证。

参数与顺序：

- 两臂同一原底座revision、同一验证后的初始LoRA张量、r8/alpha32、原targets、lr2e-5、warmup10、grad_accum4、microbatch1。初始化由E0决定，选择过程记作探索性决策。
- 两臂均采用scheduler总horizon100；在25/50暂停不把总horizon改成25/50。25/50/100对应同一消费顺序前100/200/400条，前缀token统计和期望ID顺序已存manifest。
- 可用现有legacy选择路径：`--sample-policy legacy`，**不传**`--length-mixture`，`--max-len 4096 --overflow-len 4096 --seed 20261003`。两文件行序相同、全都在上限内，现有同seed shuffle也相同。不得再分别按F1 token桶采样；核对日志实际ID前缀。
- 保存25/50/100的独立checkpoint；每次评价记录准确更新数和消费量。选最终checkpoint依靠开发解题与成本，不能只看loss。

现有CLI还没有“固定horizon100但停25/50并共享已加载模型”的会话入口，需按§8实现；不能直接把`--optimizer-updates`改25充当本对照。

## 5. 顺序实验与最高支出

所有时间包含各阶段实际加载、安装、保存；是分配上限，不是耗时承诺。能留在同一进程共享底座的阶段就连续运行；需要检查结果再决定的阶段保存后退出，不能让GPU空等人工研究。

| 阶段 | 上限：L4墙钟 / 配额 | 最小可回答的问题 |
| --- | --- | --- |
| S0：加载、必要数值探针、E0、选定初始化F1@0 | 2.5h / 5h | G1相对底座的新增/丢失；F1是否能按格式输出。先P12，再补固定E0余项 |
| S1：F0/F1分别25/50更新与P12评价 | 4.5h / 9h | 相同训练例、初始化和LR曲线下的格式/解题/时间差异 |
| S2：只给一支条件续到100 | 1h / 2h | 增加训练是否仍有收益；不足时保留共同50步结果 |
| S3：第二候选比较与可选TTT | 2h / 4h | sample还是transpose值得第二次机会；小步适配是否增加解题 |
| S4：冻结方案与原参照在C12确认 | 1.5h / 3h | 开发片信号能否在独立小片保留 |
| 保存/恢复、账差及未承诺余量 | 3.5h / 7h | 先保结果与补完整清单；核实实扣后才分配 |
| 合计 | 15h / 30h | 不保证每个可选分支全部完成 |

S0可能生成98次base/G1输出，加选定初始化的14次F1@0共112次；按旧题均值已经2.14小时，尚不含加载和数值探针。**不能承诺2.5小时把这些全部做完。** P12配对优先，F1@0和其余E0按明确顺序及成本准入执行。

S1在25更新主要检查格式或明显退化，50更新做初步比较。只有两臂最后共同完成的checkpoint能称配对比较。若出现OOM/跳样本，两臂实际消费不再相同，须按真实共同前缀报告，不能静默删长题继续宣称同数据。

S2不是自动续训：有新题净收益、互补性或明确性能价值才考虑。1道题的变化只算探索信号；两臂均无收益时优先把预算用于候选/适配或确认问题，而非为完成100步继续烧额度。

S3先复用现有greedy，只额外生成同视角sample和transpose greedy，形成两个各有2个候选的方案。不能把base、adapter、sample、transpose所有输出混成一个“@2”。若两候选覆盖没有增量，不做更多视角或PoE；若有覆盖但选2丢解，再研究排序。

TTT只在入口已实现且S3剩余预算足够时执行预定P6：0/8更新、accum4（最多192个microstep），每题重置同一离线checkpoint，仅demonstrations/LOO，query真值不参与；多query共享task适配。与不TTT但多视角的等墙钟方案分开比较。未实现则把预算返还确认或有效训练，不临时写半套TTT耗掉会话。

TTT的LOO不能机械继承离线D2“至少两个剩余demo”的筛选：本片存在只给两个示例的任务，留一后一个context仍可作已知示例适配，应显式记录而不是把整题跳过；任何时候均不加入待答query输出。

S4先做冻结checkpoint与选定原参照的greedy配对，共32次生成；有余量才补冻结第二候选策略。不能看C12分数后改参数再称盲确认。仍不足以推断私榜或金牌概率。

## 6. 先测一个不改变训练目标的性能优化

优先级：原生答案后缀logits → 已离线备好的GDN fastpath → 可选greedy batch2。性能探针合计最多约1小时墙钟，包含在表中阶段预算或经核实释放的余量，**不是另加一小时**；不要求三项全做。

### 6.1 答案后缀logits（首选，最多15–20分钟）

当前prompt标签虽为-100，仍为全部位置计算248320维输出头。对microbatch1且答案监督为连续后缀的情形，设prompt长P、答案含EOS长T，完整输入L=P+T；建议增加可选路径：

```python
# 建议接口；不是当前脚本已有行为。完整input_ids和主干梯度保留。
output = model(
    input_ids=input_ids,
    attention_mask=attention,
    labels=labels[:, P - 1:],
    logits_to_keep=T + 1,
    use_cache=False,
)
```

保留hidden位置P−1..L−1，HF自身shift负责第一个答案和EOS，最后一个logit仍ignore。不要再手动shift、不要只改logits而保留全长labels、不要detach prompt、不要裁词表。若不是当前连续后缀契约则明确拒绝此路径。

同一adapter与同一短/2K/4K训练episode，先不更新权重，比较完整路径与后缀路径的loss、全部LoRA梯度相对误差/余弦及重复同一路径的数值噪声；至少一例用G1非零LoRA避免零梯度掩盖问题。确认后再比较时间和各卡allocated/reserved；清理旧logits/grad，四卡同步，JIT和warmup分列。出现系统差异或无法解释的偏差，保留旧路径。

按G1真实训练量，头部位置由521364减少至63608，约减少87.8%；这**不是整模训练提速87.8%**。主干仍处理完整上下文。当前新400例F0的投影位置约可由760272降至90962；实际显存和速度均未测。HF生成通常已只算最后logit，不把此改进重复算作推理收益。

### 6.2 GDN fastpath（仅准备完成才测）

CPU预先准备独立、固定版本的离线wheelhouse，候选FLA0.5.2与causal-conv1d1.7.0；官方存在torch2.10/cu12/cp312/linux_x86_64/cxx11abiTRUE卷积wheel。GPU先核ABI再导入，不能现场联网、重装整套torch或编译数十分钟。

记录实际chunk/recurrent/conv/norm callable。单层前后向通过后再对真实episode看LoRA梯度、prefill、decode与显存；与同配置fallback配对才谈倍数。安装/ABI/数值失败或超过30–45分钟总增量就结束该分支，保留原栈继续主实验。新增GDN LoRA targets改变适配容量，是另一项研究，本周不混入F0/F1对照。

### 6.3 可选greedy batch=2

只在前两项未耗尽性能预算时，固定4条query按输入长度配对，fresh cache、left padding、正确attention mask与每行EOS，先短前缀再完整生成。比较完成query/s和输出变化，不把更短生成误算提速。当前训练microbatch保持1；直接改成batch CE会把按episode平均变成按答案token平均，也需要额外实现，本周不做。

## 7. 本周暂不铺开的方向

不同时比较rank、学习率、GDN targets、G3、量化、TP、Unsloth与新数据。它们会使配额耗尽而难以归因。

本次最新来源更新：Unsloth当前依赖已允许TF5.8，不能继续沿用007“<=5.5冲突”的理由；但本项目BF16四L4迁移收益仍未测。FlashQLA目前列出的GPU架构不含L4的SM89，不在本周试装。G3保留输出头/权重压缩价值，但先完成更轻的F1及后缀logits对照。

## 8. 共享模型会话：需要用户接手的实现

建议一个小的手动实验入口（如`scripts/run_budgeted_grid_experiment.py`），复用既有数据、生成、训练函数；不要新增训练平台或通用调度系统。以下是接口设计，不是已经可调用的CLI。

1. **加载与状态**：一个冻结BF16底座驻留四卡；只保存/恢复LoRA与必要optimizer状态。不要deepcopy27B。E0使用现有PEFT关闭/开启adapter路径。训练臂在同一初始LoRA快照上各自开始；选原底座时fresh LoRA保留正常A初始化、B为0，不能把A/B同时清零。
2. **阶段与调度**：scheduler horizon=100与`stop_after_update=25/50/100`分离。阶段停止保留optimizer/RNG/cursor；新臂重新建立这些状态。保存后可退出，跨会话恢复以实际消费ID为准，不重新采样。
3. **模式转换**：训练时use_cache=False、梯度checkpointing与LoRA梯度正确；评价时eval/no_grad和受支持cache。每次权重/adapter/formatter切换清空KV、GDN递归与卷积状态，不移植attention-only DFS回退。
4. **评价函数**：接收已加载model/tokenizer、arm、formatter、显式query清单、deadline、output；不再从函数内部重复from_pretrained。每query持久化原文、parsed grid、tokens、时长、状态；每arm独立分数，候选oracle与最终选2区分。
5. **预算控制**：按已测短/中/长样本成本作新update/query准入，预留至少一个保存窗口；多个arm配对优先完成同一query。运行中deadline触发不把未完成输出算模型错误。Notebook完整墙钟和账户配额差均记录，不能只计主cell。
6. **结果与恢复**：已完成结果按checkpoint hash/formatter/task/query/view/config判重；不同checkpoint结果不复用。默认不覆盖已有产物。调参所用P12与冻结C12输出目录分开。

实现顺序：修A/B → deadline/逐query保存 → 共享loaded-model评估 → 同初始化两训练臂 → 后缀logits探针 → 条件TTA/TTT。修C、repair质量分支与复杂batch随后再做。若共享训练重构成本过高，可先共享评估、两臂各加载一次；明确多花约5–10分钟/次，也优于仓促写错状态继承。

## 9. 当前即可运行的F0 E0与复现入口

现有`solve_grid.py`可运行F0 base/G1配对，但预算是soft guard、每task才落盘，因此要留最后一对生成的余量。不要将008/010的自定义清单直接传`--split-file`。本轮已准备`e0-p12-challenges.json`合并两来源题面、`e0-p12-source-map.json`提供顺序与评分文件路由：运行时从Kaggle的training/evaluation solutions各取选中task的原query标签写到一个独立评分JSON。配合map的`task_ids_cli`，现成入口可一次加载覆盖P12，不必为H/E分别重载；输出按map拆组报告。只有选中标签进入评分，不能用于prompt或TTT。

```sh
python scripts/solve_grid.py \
  --model-dir "$ARC_MODEL_DIR" --adapter "$ARC_G1_ADAPTER_DIR" \
  --challenges "$ARC_CHALLENGES" --solutions "$ARC_SCORING_SOLUTIONS" \
  --task-ids "$ARC_VERIFIED_TASK_IDS" \
  --compare-mode base_vs_adapter --formatter f0_space \
  --max-new-tokens 2048 --n-sample 0 --second-view none \
  --time-budget 3600 --out /kaggle/working/010/e0_f0_pair.json
```

3600仅示范已存在的参数，不是通用的一小时保证；paths、清单和预算需按阶段实际填写。比较模式的通用aggregate目前把base和adapter合成oracle，报告应读取逐臂记录与`pair_aggregates`，不要把它叫adapter部署@2。

本轮CPU配对文件的重建顺序：用现有materializer将D2 index＋newbase、`--skip-repair --formatter space`写到`/tmp/arc010-materialized/f0.jsonl`，再人工运行`python3 experiments/010/prepare_paired_pilot.py`。该脚本拒绝覆盖既有400行文件；完整输入SHA和原始消费顺序保存在manifest，不能在看到分数后重新挑训练seed。

每次GPU手动实验继承项目记录要求：数据/权重/代码revision、formatter、实际消费量、硬件/内核/精度、Notebook全流程墙钟、账户配额差、逐题新增/丢失及产物。不新增自动测试或CI。当前所有GPU效果与速度比较都必须写“未运行”，直到有实际输出。
