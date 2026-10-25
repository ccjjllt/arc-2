# 027 NVARC候选与推理研究核查

查阅日期：2026-10-25。仅研究记录；未启动GPU、未读取预留23题答案、未修改求解代码，也没有新的提分结果。已继承025 R2净0、026单续训权重最终净0的停止决定，不能用相同16题搜新权重参数／排序救该checkpoint。

## 三项原始来源

1. [NVARC作者2025方案说明](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc)，配合固定[原仓README](https://github.com/1ytic/NVARC/blob/846d0198efa752534594e321fc3289fc0a06c657/README.md)。作者明确：batch DFS带来数值非确定性；其batch-invariant实现本地更好，但Kaggle约慢17%，最终没有采用。原重评分已经给每个候选使用同一组8个增强，后续聚合候选出现次数与增强几何概率。故“再加入对称增强一致性”不能直接称作本方案缺失的验证器。这里不复述其2025榜分为本届结果。
2. [ARChitects 2025作者技术报告](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)，固定仓库commit `efcadc66a0fcc1ea7eca2c90eac066c54f8fc543`、[报告源码](https://github.com/LambdaLabsML/ARC2025_Solution_by_the_ARChitects/blob/efcadc66a0fcc1ea7eca2c90eac066c54f8fc543/page.md)。报告§Initial Autoregressive Improvements称，一次猜16—32 token的speculative DFS加速4.7倍，prefix-cache重评分加速5.8倍，继而将DFS路径概率门槛17%降至7%，重评分views从8增加到32。倍率属于其模型／实现，不能移作NVARC4B已验证收益。其最终masked-diffusion及shape模型不在当前范围。
3. [Land作者预印本v1](https://arxiv.org/html/2606.31543v1)，§3、§5、§7区分候选多样性与最终联合选择，讨论多数答案错误时保留少数正确假设。其holistic judge依赖完整推理轨迹、商业模型与长上下文，不能移植到只会16-token网格的4B上直接作为语言judge；只支持把候选来源互补与两槽选择分开核算的研究原则。

## 对当前精确V1的代码核对

已读`experiments/021/reference/failed-in-aimo-v1.source.py`，以及成功026外置运行源码：

- `turbo_dfs`每次调用model的`input_ids`形状为batch×1，逐token递归；没有上述16—32 token speculative块验证。
- `calc_scores`对每个新候选的8个增强重新前向完整query＋answer。`known_scores`只是同一query-grid的分数去重，没有缓存不同候选共用的prompt KV。
- `calc_scores`返回目标answer token的**NLL求和**；`kgmon`为出现次数减去增强NLL均值，并未按answer token数归一化。原概率阈值0.2也作用于路径累计NLL，不是平均token置信度。
- 原增强重评分是同一模型的多视角偏好，没有可执行程序或独立正确性证书。

## 可落地路线比较

| 路线 | 当前能证明的机制 | 成本及主要风险 | 优先级 |
| --- | --- | --- | --- |
| 双初始化、两槽交错去重 | 026说明两初始化候选存在互补；用来源身份保留多样性，避免直接比较未校准NLL | 两模型全量会约翻倍TTT／DFS；可能把原第2槽正确答案挤掉；局部确认正不等于12h全量可行 | 最快可确认，但必须同时解决预算 |
| answer长度归一化／不同NLL聚合 | 不需要新GPU生成，可从旧开发候选回放 | 短／长答案偏好会改变；没有正确性保证；025已否定只改第二槽最低现有NLL的R2 | 只作独立开发假设，不扫026面板 |
| 增加对称views | 平均更多同模型方向偏差 | 原版已有8view；增加到32会多前向，不能当独立验证 | 不作为最快下一步 |
| 只对known demos做LOO | 衡量该题模型是否能从其他示例泛化 | 同一task的所有query候选会拿到相同分数，无法区分候选；且TTT已看过被留出demo时存在记忆干扰 | 不宜作为直接候选排序 |
| candidate-conditioned demo LOO | 把query-input／候选-output当额外demo，再预测被留出的已知demo，可获得候选相关分数 | 属于本项目待验证推断；不是逻辑证书，错误候选也可能与已知demo兼容；每候选多次完整前向，且已见demo记忆削弱信号 | 研究储备，不阻塞当前交付 |
| prefix缓存＋speculative DFS | 原作者已有显著提速实证；精确V1仍缺两项，为更深搜索腾预算 | Qwen3／Unsloth KV缓存可变性、回溯裁剪、batch数值差异都需实测；不能只抄倍率直接降阈值 | 下一项生成改进最有实质依据 |

## 推荐冻结的最小候选规则

若本轮选择双初始化新方向，先冻结无调参的 **round-robin-two-slot**，不要为026的错误题设门槛：

1. 每个初始化仍独立用原kgmon得到去重候选顺序`b1,b2,...`、`c1,c2,...`。
2. 固定流`b1,c1,b2,c2,...`，按网格完全相同去重，前两个合法网格作为两槽。没有候选的一臂自然跳过；相同第一候选时回落到base第二个不同候选。
3. 不把两模型support／raw NLL直接加总，不按task ID、输出大小或已知答案白名单切换。规则本身不能保证保留base第二槽正确，必须记录新增和丢失。
4. 先在独立历史开发材料上完成规则身份与fallback核查；如果要使用预留23题，必须在其gold读取前冻结唯一策略和正式预算方案，只确认一次。026 16题不得拿来调排序权重、阈值或选择若干变体中的最好一个。

该规则属于基于互补动机的本项目推断，三项论文均未证明它对现有checkpoint一定提分；本报告未选择或评分预留答案。

## 正式提交预算边界

026两臂各2GPU的16题评价耗时约92分钟。若240隐藏任务耗时分布相似，双初始化全量粗略外推为23小时左右，而原版只有12小时。此为输入分布未知下的粗估，足以否定“确认23题跑全就自然适配正式全量”的推断。正式版本需保住原base覆盖，并有可部署信号约束补题数量／预算；或先实现实测提速。不建议直接把2＋2全量集成当正式提分版本上传。

不能直接把DFS门槛0.2改为0.07而不提速：会扩大搜索树，可能使540秒DFS和1200秒task限制截掉更多query。最有依据的后续代码方向是先做共享prefix缓存或speculative DFS中的一个，小型同权重配对核查实际速度和候选变化，然后才单独评估更深门槛。

## 027最终确认设计的补充审计

前文双初始化／2＋2为最初研究候选。027实际方案已收敛为**同一原sft139、顺序两个4GPU阶段，仅改变TTT数据增强seed1→2**；不使用026续训权重，也不把原base预算切半。Trainer／PEFT seed42、DFS增强seed2、原TTT／DFS／B4均保留。base结束后才用余时跑extra，最终固定base第一槽，extra第一合法真候选不同于base第一槽时替换第二槽，否则保持base第二槽；base没有真实候选时才使用extra两槽。此处只读审计，不产生确认分数。

确认包108007字节，SHA `bb1e8320a087a0abb57948077ff15f794c1206f53abdb2b235c2969b7cca87b0`，源码AST通过；两阶段共用PYTHONHASHSEED27027和同一global cutoff、分支目录分离，前一进程组完全退出后清理原导入marker。真实candidate count与合法网格分开检查，合法一格`[[0]]`不会被当作默认占位。无阻塞缺陷。该设计保住base计算覆盖，不等于保证历史公榜30.14；原batch数值非确定性和本轮新哈希种子仍会影响具体输出。

条件正式生成器`prepare_formal.py`在SHA `f4ad6d81409e0ee1573d06b713967ea311501727eb547066fce2ddf57a752904`版本接受只读静态审计：仅调用纯`build()`在内存构造，没有调用main、写正式包或启动GPU。formal全部源码AST通过，023原loader／decoder／solver／starter与研究包逐字相同，23个确认task ID均未出现在formal源码。输入在competition rerun时动态读取完整test challenges，否则只用023原四题，并核对其canonical SHA；总43200秒、global42600秒。base返回0后先写原B4备份及atomic顶层submission，随后extra；extra启动、运行或finalizer异常回落到该base备份并return0。extra预算截止只允许已完成atomic候选参与融合。正式上传仍以确认净收益和预定prefix条件通过为前提；本次没有生成正式Kaggle版本或submission。

## speculative DFS：已定位代码与公开范围

2026-10-25读取了ARChitects 2025报告仓库的完整固定树：`LambdaLabsML/ARC2025_Solution_by_the_ARChitects` commit `efcadc66a0fcc1ea7eca2c90eac066c54f8fc543`仅公开LLaDA预训练相关Python文件，**没有定位到报告所说16—32 token启发式speculative DFS实现**。不能写成已有可直接拷贝的2025函数。报告本身的页尾为CC BY-SA 4.0，仓库树未见通用LICENSE文件，不把报告许可擅自扩成所有代码许可。

可读的相近机制在作者2024公开实现，固定commit `d3ac3f6ebf6fb609bfdc782561ee99977ca35d95`：

| 来源 | 精确位置与作用 | 许可 |
| --- | --- | --- |
| [training_code/inference_tools.py](https://github.com/da-fr/arc-prize-2024/blob/d3ac3f6ebf6fb609bfdc782561ee99977ca35d95/training_code/inference_tools.py)；blob `1439b14dda230344243e8796c65744435b159c66` | `explore`第49行和`dfs`第85行：一次前向prompt＋已知guess，沿guess复用预计算logits，偏离guess时恢复单token前向；回溯显式裁剪KV。`infer_task`把当前best guess带给后续增强 | 文件头版权Daniel Franzen、Jan Disselhoff 2024及Apache-2.0；[LICENSE.txt](https://github.com/da-fr/arc-prize-2024/blob/d3ac3f6ebf6fb609bfdc782561ee99977ca35d95/LICENSE.txt)一致 |
| [kaggle_notebooks/arc-prize-2024_updated.ipynb](https://github.com/da-fr/arc-prize-2024/blob/d3ac3f6ebf6fb609bfdc782561ee99977ca35d95/kaggle_notebooks/arc-prize-2024_updated.ipynb)；blob `6a1d0161884d18d03bfcc09abeddd6d1d4d15008` | code cell index1导出`model_runner.py`；导出body中`turbo_dfs`472行、`inference_turbo_dfs`502行、`inference_step`522行、`calc_score`625行。仍是precomputed-path机制，明确断言不支持batch；不是报告中的完整16—32-token启发式batch实现 | Notebook内保留同一Apache-2.0文件头；采用时保留来源和修改说明 |

## 最小可实现提速备选，保持027本轮冻结

以下是后续实现步骤，未运行，不能改变本次027结果。

1. **优先尝试共享prefix重评分缓存，或单独做speculative DFS，二选一。** 前者不改搜索概率门槛／TTT；后者首先保持原累计NLL阈值`−log(0.2)`、ARC_TOKENS、EOS15、最大网格token和540／1200秒检查，避免把速度与更深搜索两个因素混在一起。
2. 若做prefix缓存，实际token序列作为cache key，每个task／query切换时清理，不能只用query ID假定prompt相同。8个D4 views通常只有两类prefix长度，但demo裁剪可能改变它，必须按**实际prefix token长度**分组、每组≤4；不要在右padding的KV后直接接answer而引入PAD孔位及RoPE错位。保存每组最后一个prefix logit与KV；首answer token用该logit，其余teacher forcing拼接，并在每个candidate之后把可变KV恢复到prefix。只缓存当前query，避免多query保留8view KV导致L4 OOM。输出NLL重新排回原8view顺序，仍求和、仍原kgmon。
3. 若做speculative DFS，可先参考2024的已知路径验证与KV回溯，但Qwen3／Unsloth当前缓存不能照搬旧tuple切片；先确认实际cache类型及in-place有效长度。草稿token只作提议，必须用target模型逐位置验证累计NLL、枚举所有仍允许的分支，不能改成只接受greedy路径；拒绝位置及回溯时丢弃未接受草稿KV。草稿长度先固定16、不扫描一组长度找最好。
4. batch适配保留当前4views布局、PAD失活beam与每行独立状态；对于draft早遇EOS／网格最大长度的行分别结束。第一个草稿token的概率来自prefix最后logit，随后token分别来自前一个草稿输入的logit，防一位偏移。不要复制2024额外`max_score_greedy`放宽，它与当前NVARC概率树定义不同。
5. 先用少数已开发任务手动比较新旧teacher-forced NLL、真实合法候选、实际速度和峰值显存，再做一个固定同预算配对；不新建测试体系。CUDA batch数值变化可能改变边界候选，不能无运行便宣称候选集合逐字不变。只有确认真实提速／候选收益后，才另行研究更低DFS门槛；报告倍率不能代作本项目实测。
