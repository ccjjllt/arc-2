# 011 训练数据小样本复核：新 base 与同 base 增强

日期：2026-10-11。仅 CPU 手工研究，未生成新数据、未训练、未修改数据入口；没有读取 C6/C12 或官方 evaluation query 答案。使用 Data 的数据质量工作流做这一项局部核查，不新增测试套件。

**结论：现有材料足够定位并修正具体质量问题，也有可用的小样本；尚不足以声称已经准备好一轮公平、足量的“独立新 base 对比多增强”GPU训练。** 最突出的问题不是所有生成器标签错误，而是旧 G2 的过滤政策会主动留下某些示例欠定的 episode。增加随机 base 和增加旋转分别解决不同问题，必须控制这一差异。

## 抽样与口径

先固定 009 已有可读规则重建的六族：`0b17323b、2072aba6、00dbd492、3b4c2228、8597cfd7、c8b7cc0f`。每族取 G2 当前保留的最小 instance 的 `ident|ident`、同 instance 的 `rot90|ident`、008 的 instance100，共18个呈现；加009修复试件的原始/修复两份，共20个。没有按本次核查结果替换差例。这是针对已知风险的目的抽样，不能据此估计448族的整体问题率。

20个呈现对应 **13个独立 query base、6个族**；修复版与原版共享同一query base，另借用了instance104的一条训练demo。新文件中的 `gen_instance:100` 必须连同 source、task_id 组成键，不能单独计独立实例。六族均不在 D2 的 holdout/diag 排除名单内。

G2文本按 `[demo N] in/out` 与 `[query] in/out` 边界反解；rot90按现有顺时针定义逆旋转回原向。只提取009分析脚本的 `components`、`oracle` 两个函数做复算，没有执行该脚本的落盘部分或调用生成器。逐族人工读规则、看下面的demo/query特征，再检查保存标签。

- 20×4＝**80/80个呈现pair**符合009记录的规则；这些并非80个独立pair，也不是重新证明生成器所有分支正确。
- 六组旋转的每组8张网格（3个demo各输入/输出＋query输入/目标）均精确等于原base旋转，**48/48**。它们增加方向经验，没有增加独立底层实例，也没有补出原episode缺少的映射证据。
- 12个G2呈现只有system/user消息，query输出区为空；新base与repair的目标独立存储。修复版query输入、target与原版完全相同；替换demo精确来自新base文件第1793行的demo0，且不等于query输入。没有发现本次样本的prompt构造把query目标直接写入query输出位置。
- “target等于某个demo输出”与“query输入直接重复demo”分别处理；前者在分类/计数任务很正常，不能统一当泄漏。

## 20个实际呈现与长度

G2行号指 `experiments/007/g2/episodes.jsonl`，new行号指 `experiments/008/data/gen_newbases.jsonl`。repair均来自 `experiments/009/episode-repair-pilot.jsonl:1` 的不同字段。长度为重新构造相应F0/F1提示、nonthinking模板、完整答案＋一个EOS的CPU tokenizer计数；不是训练用时或显存实测。tokenizer为本地Qwen缓存，Python3.9.6/Transformers4.57.6，模板SHA见末尾。

| # | family | 来源/instance/呈现 | 源行 | F0/F1 full token | 本次重点 |
|---:|---|---|---:|---:|---|
| 1 | 0b17323b | G2/2/ident | 12445 | 3708/2027 | 不同spacing输入，可读延伸规则 |
| 2 | 0b17323b | G2/2/rot90 | 13195 | 3708/2027 | 与#1同base |
| 3 | 0b17323b | new/100/ident | 8 | 3708/2027 | query输入直接等于demo0 |
| 4 | 2072aba6 | G2/1/ident | 12442 | 468/323 | 局部模板替换 |
| 5 | 2072aba6 | G2/1/rot90 | 13192 | 468/323 | 与#4同base |
| 6 | 2072aba6 | new/100/ident | 54 | 468/323 | 新占位组合，支持范围仍窄 |
| 7 | 00dbd492 | G2/0/ident | 12001 | 4612/2479 | query厚度3未在demo中展示 |
| 8 | 00dbd492 | G2/0/rot90 | 12751 | 4612/2479 | 保留同样缺失映射；F0超过4K |
| 9 | 00dbd492 | new/100/ident | 1 | 3520/1923 | query厚度1已展示 |
| 10 | 3b4c2228 | G2/6/ident | 12448 | 532/358 | demo计数1/3/1，query计数2 |
| 11 | 3b4c2228 | G2/6/rot90 | 13198 | 532/357 | 与#10同base |
| 12 | 3b4c2228 | new/100/ident | 109 | 498/339 | demo/query计数全1，常量捷径也成立 |
| 13 | 8597cfd7 | G2/1/ident | 12443 | 896/551 | demo胜者全4，query胜者2 |
| 14 | 8597cfd7 | G2/1/rot90 | 13193 | 896/545 | 旋转没有补出第二类别 |
| 15 | 8597cfd7 | new/100/ident | 232 | 860/531 | demo覆盖两种胜者 |
| 16 | c8b7cc0f | G2/1/ident | 12444 | 700/447 | demo框内计数3/3/4，query3 |
| 17 | c8b7cc0f | G2/1/rot90 | 13194 | 700/447 | 与#16同base |
| 18 | c8b7cc0f | new/100/ident | 349 | 764/481 | 框内计数全3，新query色可从输入复制 |
| 19 | 00dbd492 | repair/original/102 | 1.original | 1140/687 | 缺厚度2；有限映射类有6候选 |
| 20 | 00dbd492 | repair/repaired/102 | 1.repaired | 1200/719 | 增加厚度2 demo；同一假设类剩1候选 |

18/20同时处于F0/F1共同4K域；20/20 F1≤2479。不应通过只给F1加入#7/#8，混淆格式效果与新增长样本效果。若只研究数据而固定F1，仍需按真实长度记录，而不能把行数相同称为token/墙钟相同。

## 具体发现

**高置信度、影响大的旧过滤偏差：`8597cfd7`。** 规则比较红/黄在分隔线两侧的数量增量，输出仅有纯色2×2的2或4。旧G2保留实例的demo赢家为`[4,4,4]`，query为2；“永远输出4”完全拟合demo，却在query错。该缺口被rot90完整保留。new100的demo赢家`[2,4,4]`，query4，并没有缺少另一种类别。

这不仅是偶然选到难例。G2 manifest记载19个生成base因 `target_equals_demo_output` 被删除；`scripts/arc_episode_data.py:1050`附近的旧复现分支仍清楚实现这一规则。对该二类固定输出族，只要demo已经展示query所属类别，就必然被旧规则删除。因此旧池与新池的差异混合了 **过滤政策、示例覆盖和base数量**，不能直接归因于“新base更有价值”。下一轮应继续采用D2的flag-only政策；不能恢复全局同输出删除。

**任意映射欠定：`00dbd492`。** 规则按方框厚度1/2/3填8/4/3色。旧G2 instance0的demo只见厚度2、厚度1/2及空图，query含厚度3；3的映射并未展示。改变尚未展示的厚度3输出色仍能满足全部demo，因而不能称为干净的演示内推断例。new100 query只要求已展示的厚度1，风险不同。009原/修复对保持query及目标不变，用一条厚度2 demo替换空demo；本次再次确认所有网格及donor lineage，支持“补有区分力的示例”这一局部处理。6→1的候选数来自009预先定义的有限映射类，不是普遍唯一性证明或模型提分。

**新base也可能没有增加有效监督。** `0b17323b` new100的query直接等于demo0，应重配或在干净训练片中单列；`3b4c2228` new100的三个demo与query均输出计数1；`c8b7cc0f` new100也全部框内计数3。后两者源规则和标签正确，当前query也不能区分“真实计数”与常量计数捷径，不建议把它们当作增加计数覆盖的证据。

**可保留的小样本。** `2072aba6`新旧实例均支持逐格模板替换，新base换了占位组合；但随机生成只覆盖3×3前5位，不能声称已经覆盖整个九位输入域。`c8b7cc0f`旧base的框内计数3/4与框外干扰提供有效区分；query新颜色来自query输入本身，不能因为demo没出现该色就删。`8597cfd7` new100、`00dbd492` new100及修复版也有明确可保留依据。这里“可保留”仅表示本次未找到所检查的短规则反例，不是语义完备认证。

## 下一轮只做一个窄问题

先将问题限定为：**同一组干净family、同一固定formatter/初始化/更新数下，把第二次呈现用于“新独立base”还是“原base旋转”更有价值？**

1. 共享一个已审的anchor base；A追加其rot90，B追加同family不同base。两臂按family一一对应，禁止B顺便换修复质量、长度上限或筛选政策。旋转臂不增加base计数。
2. 本次最值得继续的小组是`2072aba6、00dbd492、8597cfd7、c8b7cc0f`。009既有记录可提供同族100/101候选，但本次只重审了100，不能把101写成本轮新认证。先逐对核demo关键类别和短规则反例，再决定是否形成约8–12个独立base的小消融。当前20呈现不能直接成为这一配对训练集。
3. 将同base增强与“修复示例欠定”分开。修复pair原始base键已经带variant，但当前 `_make_episode` 的episode_id仍由task/query_kind/instance/geometry/color构造（`scripts/arc_episode_data.py:538`）；若把原/修复同时物化，仍可能撞去重键。先明确variant和parent lineage，不能把raw修复完成等同于已能进入训练。
4. 不建议立即花GPU扩大SFT：目前只是4族有可复用审查起点，不足以对更广ARC2泛化作推断。先补齐上述配对、保持训练侧目标隔离和共同长度域；用现有P12评价后续模型，C6/C12保持未读。未经模型实验，不宣称独立base、修复或多增强哪一种提分。

## 来源与复核边界

- G2共30000行；本次只选择上述12个ARC-GEN呈现。`experiments/007/g2/manifest.json`固定ARC-GEN revision为`a15cbdb44c776610aeeb9f487a06af875d3d0878`，750个生成base各展开24种增强。
- 新base文件2240行、448族，每族instance100..104；本次取预定六族100。完整文件SHA与009记录一致。
- 规则复核使用已存的009人工重建函数；原`/tmp/arc007-arcgen`源码检出已不在本机，本次没有重新克隆、执行生成器或假称重新完成全部原源码审计。
- CPU统计本轮约数秒（不含人工阅读）；无PyTorch/TensorFlow/Flax模型加载。来源查阅日期为2026-10-11；没有新增模型指标。

| 来源 | 本次SHA256 |
|---|---|
| `experiments/007/g2/episodes.jsonl` | `d09dd0dec76adf9604394d893ef42ceb32bf8eead1b3b63b46175201bba3ca74` |
| `experiments/008/data/gen_newbases.jsonl` | `539fe79657a1cdb3f7363ed07507092c2adaf508774e2e27d7a31d6f597a0605` |
| `experiments/009/episode-repair-pilot.jsonl` | `f048dfb90f993c6be0b710a5d509c96757d1772aa86ba13c5bba7f70b479ba7f` |
| `experiments/009/analysis-source/arc009-semantics-audit.py` | `fea74c3243183e02e44b45a94828e2ab81cc589fd428d202db4967ee076da5e6` |
| `experiments/009/episode-semantics.json` | `9c1c4db28af92699f666ac9154d162f672ed3e07edd8b9dd999b41c7cfdee557` |
| `experiments/009/demo-replacement-analysis.json` | `37025b323aa6e41a58301712e1931f47ae9d6898ce2ddd98259b9f4637640615` |
| Qwen缓存`chat_template.jinja` | `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041` |

核查入口：上述固定行号、20项表、`scripts/arc_episode_data.py`的`build_messages/serialize_grid/apply_geometry`、009脚本中只读提取的`components/oracle`。这是一份研究备忘，不是新增自动测试流程。
