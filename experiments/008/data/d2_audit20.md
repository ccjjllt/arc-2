# D2 人工抽查20个base（CPU只读，未训练/未推理）

日期：2026-10-01。方法：只读 `experiments/007/raw` 官方题面与 `/tmp/arc007-arcgen` 生成器源码（commit a15cbdb）及 `experiments/007/g2/episodes.jsonl` 中对应 `ident|ident` 行的文本长度；未运行模型推理、未运行 `validate()`、未重跑 `generate()`。`suspect_same_as_demo` 仅标记保留，不删除。示例是否足以推断query规则为人工初判（足以/存疑/未确认），不是模型解题证明。

覆盖：长题（30x30）7、非ARC1 6、多色（>=7色）9、重复输出（suspect）5、LOO 3、生成侧 6。family_id=task_id，仅ID隔离，不称语义去重。

| # | task / base / query | origin / V2 | 长/色/重复 | 示例是否足以推断query规则（初判） | 共享变量/校验语义备注 |
|---|---|---|---|---|---|
| 1 | `e4888269` test:0（2 tests, 3 demos; qin 30x30 tgt 30x30） | non_arc1 / 有V2（151之一） | 长30x30 / 10色 / 非suspect | 未确认：3个demo对30x30大格是否覆盖query变化需试解，不能仅凭尺寸断定 | 官方端无共享变量；demos=3偏少，长题校验需完整episode训练，不删demo |
| 2 | `b74ca5d1` test:0（2 tests, 3 demos; 30x30->30x30） | non_arc1 / 无V2（55之一）靠官方 | 长 / 10色 / 非suspect | 未确认：同上；该任务靠官方覆盖，D2索引已保证1行 | 55任务代表；无生成器，丢失即永久丢失，已在D2官方turn0覆盖 |
| 3 | `b74ca5d1` test:1（同题第二个query） | non_arc1 / 无V2（55） | 长 / 10色 / 非suspect | 存疑：同题多query，G2只取test0，D2首轮仅索引test0；test1未覆盖，明确记录为未覆盖 | 多query任务仅首query进入首轮索引；test1需后续轮次补，不假装已覆盖 |
| 4 | `753ea09b` test:0（1 test, 3 demos; 30x30） | non_arc1 / 无V2（55） | 长 / 10色 / 非suspect | 未确认：3 demos对大格规则暴露程度待验证 | 同为55任务；官方turn0已覆盖；生成侧永远无法覆盖，配方变化需另记 |
| 5 | `17829a00` test:0（1 test, 2 demos; 16x16） | non_arc1 / 无V2（55） | 中16x16 / 7色 / 非suspect | 存疑：仅2个demo，示例稀疏，是否足以约束query规则需检查原题 | demo数最少一档；完整episode要求不删demo；2-demo任务的LOO仅1个候选 |
| 6 | `a2d730bd` test:0（1 test, 3 demos; 30x30, 5色） | non_arc1 / 有V2 | 长 / 5色 / 非suspect | 未确认 | 该V2生成器在G2曾报 `ValueError: Sample larger than population`（manifest throw示例），生成可用性存疑，D2复用G2已保留base不新造 |
| 7 | `fd096ab6` test:0（1 test, 2 demos; 30x30, 9色） | arc1_eval / 有V2 | 长 / 9色 / 非suspect | 存疑：仅2 demos，大格多色，规则欠定风险高 | 官方+生成双覆盖任务；生成侧是否有同族base需查索引，不把V2当新任务 |
| 8 | `f9d67f8b` test:0（1 test, 4 demos; 30x30, 9色） | arc1_eval / 有V2 | 长 / 9色 / 非suspect | 初步足以候选：4 demos为本批最多之一，但仍未试解，记为候选足以、未证明 | demo数相对充足；长题仍属真token桶未确认（tokenizer缺失），不宣称已训长桶 |
| 9 | `239be575` test:0（2 tests, 6 demos; qin 8x6 tgt 1x1） | arc1_train / 有V2 | 小目标1x1 / 3色 / suspect=True | 存疑：target等于某demo输出（base级suspect 25之一），保留不删；小目标易退化，需防模型抄demo | D2修正点：旧规则若删LOO会误删，本test本就保留；标记后抽样审查，不一律删除 |
| 10 | `642d658d` test:0（1 test, 3 demos; qin 27x22 tgt 1x1, 10色） | arc1_eval / 有V2 | 长输入/小输出 / 10色 / suspect=True | 存疑：大输入小输出+重复输出，示例是否足以区分抄答案与真规则需审查 | 同上标记保留；颜色多但目标1格，监督token极小，不能按行数加权 |
| 11 | `1a2e2828` loo:2（5 train, 4 demos; qin 11x11 tgt 1x1, 8色） | arc1_eval / 有V2 | 中 / 8色 / suspect=True（72之一） | 存疑：LOO suspect，旧G1/G2直接丢弃整base（24行）；D2保留并标记 | D2与G1差异点：该base在G1/G2被跳过，D2计入3853并给抽样机会，是否可学习待后续小消融 |
| 12 | `1fad071e` loo:0（3 train, 2 demos; 9x9->1x5, 3色） | arc1_train / 有V2 | 小 / 3色 / suspect=True | 存疑 | 同上；LOO demos=2，留一后示例更少，规则欠定风险高于test |
| 13 | `007bbfb7` loo:0（5 train, 4 demos; 3x3->9x9, 5色） | arc1_train / 有V2 | 小输入大输出 / 5色 / 非suspect（合法LOO） | 初步足以候选：4 demos且输入输出尺寸对比强，但未试解 | 合法LOO代表：G2官方无LOO，D2首轮turn1覆盖此类；答案mask监督目标明确 |
| 14 | `27a28665` test:0（3 tests, 7 demos; 3x3->1x1, 9色） | arc1_train / 有V2 | 小 / 9色 / suspect=True，多query | 存疑：7 demos充足但目标1x1且重复输出；同题3个test query，首轮仅test0 | 多query覆盖不足代表：test1/test2未进首轮索引；demos多但目标小，监督token权重需按token记不按行数 |
| 15 | `0b17323b` gen instance0（G2保留; demos 3, query 1） | arc1_eval / V2生成 | 生成 / 尺寸见G2行 / 非suspect（G2曾丢弃同族另instance的suspect 2例） | 存疑：`generate(spacing,start)` 每次调用独立 `common.randint`，4 slots独立种子，demos与query的spacing/start可能不一致，示例未必暴露query变化 | 共享变量：无跨slot共享；`validate()` 用固定 spacing/start（4,0/2,1/3,0）但D2/G2从未运行它；不能当规则oracle；需读源码+小批试看 |
| 16 | `2072aba6` gen instance0 | arc1_eval / V2生成 | 生成 / 3x3->6x6放大多色 | 存疑：`generate(colors)` 独立随机5位colors（和3-6约束），slots独立抽样，query的colors组合可能在demos未出现 | 同上无跨slot共享；`validate()` 固定3组colors，未运行；颜色语义需核对output 2x2块映射 |
| 17 | `00dbd492` gen instance0（G2 ident行 prompt约3551字符） | arc1_eval / V2生成 | 生成 / 大格（size 7-23随机） | 未确认：`generate(size,rows,cols,thicks)` 独立随机size/boxes/thicks+ `overlaps` 约束，query尺寸可能与demos完全不同 | 共享变量：同一次instance内4次调用参数各自独立，无传递；`validate()` 未运行；矩形合法≠可学习episode |
| 18 | `3b4c2228` gen instance0 | arc1_eval / V2生成 | 生成 / boxes+sizes+colors | 存疑：`generate(width,height,rows,cols,sizes,colors)` 内部 `draw()` 要求 sizes不全等且蓝色size2/color3计数1-3，否则返回None（G2重试逻辑），slots独立 | G2曾丢弃同族suspect（manifest示例）；D2复用保留base；`validate()` 未运行；合法矩形≠可学习，需查失败重试是否偏置分布 |
| 19 | `8597cfd7` gen instance0 | arc1_eval / V2生成 | 生成 / reds/yellows参数 | 存疑：`generate(height,reds,yellows)` 独立随机，query的reds/yellows分布可能偏移 | 官方同族 `8597cfd7` test:0本身也是suspect（test 25之一），生成与官方重复输出风险叠加，已分别标记 |
| 20 | `c8b7cc0f` gen instance0 | arc1_eval / V2生成 | 生成 / size/rows/cols/wide/tall/brow/bcol/color多参 | 未确认：多参独立随机，query所需变化在demos是否出现过未验证 | 参数最多一档；`validate()` 未运行；G2 manifest对该族无throw记录不等于语义已核对；需后续小批人工看格子 |

**共性结论（不做训练推断）：** 官方14个中4个suspect标记保留（9,10,11,12,14其一），G2旧规则下11,12两LOO会被删除，D2已改标记；3号、14号暴露多query仅首query覆盖；2,4,5号证实55任务只能靠官方。生成6个共同问题是4 slots独立种子、无跨slot参数共享，`validate()` 均未运行，不能当oracle；矩形/数值合法已由G2生成时检查，但示例是否足以推断query规则6个均为存疑/未确认，需后续小批看格子再定约束。长题7个真token桶均为未确认（本地无27B权重），不用 `overflow_len` 假装。

## 附录A（2026-10-01，CPU源码语义复核 + validate()抽样；原20条保留）

- ARC-GEN可用：`/tmp/arc007-arcgen` HEAD=`a15cbdb44c776610aeeb9f487a06af875d3d0878`，与`scripts/arc_episode_data.py:39`期望一致；pip无arc-gen包，调用为源码导入。G2池未动（30000行）。
- 新base：`experiments/008/data/gen_newbases.jsonl` 2240行（instance 100–104 × 448族），与G2的750合计2990≈3000；每base规划一致增强2个（ident|ident、rot90|ident，未展开为episode行）。base级失败0/2240；attempt级重试有（G2口径：generate_timeout / ValueError / IndexError / 非矩形含draw()->None），跨分片全量计数未保留、仅base级0失败为完整口径。
- validate()抽样：旧族30 + 新族30（含6存疑族），60/60返回矩形train/test（旧30/30、新30/30），失败0。结论：validate()传固定参数、绕过`if X is None`随机分支，不练跨slot独立性、不走draw()->None重试路径；2072aba6的validate用9元colors而随机分支只抽5元（见下），属分布外。因此validate()只能证“手选示例可构造”，不能当规则oracle。

| 族 | generate()语义（行号） | 示例是否足以暴露query变化 | 依据 |
|---|---|---|---|
| 0b17323b | L28随机`spacing∈2..4,start∈0..1`每slot独立；L29-36对角规则；validate L41-50固定(4,0),(2,1),(3,0) | 存疑 | 6种组合demos 3个覆盖不全；validate只盖3种且不管跨slot一致性 |
| 2072aba6 | L26-28随机只抽5元colors（`range(5)`）且和3..5，L30-39只动前5格/2x2块；validate L43-52用9元colors | 存疑（9格全规则接近不足；随机分布内5格仍存疑） | 后4格随机路径恒0，validate示例分布外；16种5元组合demos难盖全 |
| 00dbd492 | L32-40随机size 7..23/thicks/rows/cols+overlaps guard；L33 size<9时max_thick=2；validate L58-70固定size 15/9/13/7 | 存疑 | 小尺寸demos见不到thick=3（color 3）映射；query尺寸可与demos全不同 |
| 3b4c2228 | L32 sizes全等拒收，L40 blues计数1..3否则None（→非矩形重试）；L47-72随机宽高/红绿盒；validate L77-103全固定参数 | 存疑 | 输出对角1数=blues计数，3个demos未必盖1..3；validate不走随机搜索/拒收路径 |
| 8597cfd7 | L32-34要求reds/yellows递增且差值不等、bgcolor取大差方；L48-54随机height 9/11；validate L56-67固定5组 | 存疑 | query差值 ordering 可翻转bgcolor（2↔4），demos未必含两侧；官方同族test本身suspect叠加 |
| c8b7cc0f | L46内部count须3..6否则None（→重试）；L57-64随机size 7/9、color≠1、10%像素点、矩形框；validate L66-83固定点集 | 存疑（接近不足） | 输出=计数3..6上色，3个demos难盖query计数；validate绕过密度搜索 |

共性维持：6族仍无“足以”，矩形合法≠可学习；新2240 bases未改变该结论，只把待审查实例从750扩至2990。
