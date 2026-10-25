# 026 ARC2 额外 training 规则覆盖小挑战者：数据

日期：2026-10-25。只完成本地 CPU 数据准备；未启动 GPU、训练、评价或提交，不能由此声称提分。025 的诊断支持候选生成缺口，但没有可靠的“失败题 → 生成器 family”语义对应关系，因此这里不称为修复了特定失败规则。

## 实际交付

- **Notebook 实际输入**：[data-embedded-v1/train.jsonl](data-embedded-v1/train.jsonl)，1024 行：512 合成、512 官方 training 回放。压缩文件 [train.jsonl.gz](data-embedded-v1/train.jsonl.gz) 为 **430154 B**，base64 为 573540 字符；打包此文件，不打包全池。未压缩 SHA256：`33f3b4e6133be5b890b8f2efb76421390a463b04291f5669f6b5e44bcbd5f901`；gzip SHA256：`2556082aedb566816c6373aeda17afaeecc450964d289569fed35ae0f023d66a`。
- [冻结清单](data-embedded-v1/manifest.json) 保存全部 1024 个 episode ID、来源、祖先排除、采样方法及完整性哈希。其 gzip 约 8 KB。冻结时没有使用新模型输出。
- [全 CPU 候选池](data-v1/manifest.json)：1080 个合成训练 episode（90 family × 12），1051 个官方 training 原始 query 回放，2131 个唯一 query 输入；另有 16 个完整 family × 12＝192 个合成留出 episode。全池不嵌入 Notebook。
- [pilot](data-pilot-v1/manifest.json)：16 family × 4＝64 个合成 episode；其中 52 train、12 family_holdout；另有 52 个官方回放。另写的 [纯 stdlib 规则重建](data-pilot-v1/manual-rule-reconstruction.py) 未导入上游 DSL、generator 或 verifier，256/256 个合成 pair 与重建规则一致，见 [核查结果](data-pilot-v1/manual-rule-audit.json)。四张 contact sheet 人工查看了每族首个完整 episode，共 16 episode／64 pair；没有逐图人工查看全部 256 pair。

训练输入 schema 是 `episode_id, base_id, family_id, source, source_revision, ancestor_task_ids, split, train, query_input, query_output`，另有 lineage、质量标记和长度统计。模型文本由原始 NVARC `QwenFormatter` 生成，不读 metadata，不使用旧 27B chat 格式。只监督最终 query 答案及原 formatter 的 EOS。

## 来源与接触边界

合成源固定为 [water-vapor/synth-rearc d9ea8b2](https://github.com/water-vapor/synth-rearc/tree/d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0)，使用 [`arc2_training_minus_arc1_task_ids.txt`](https://github.com/water-vapor/synth-rearc/blob/d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0/data/task_lists/arc2_training_minus_arc1_task_ids.txt) 与 021 固定的 109 extra-training 差集；来源及选定 Python 文件 SHA 在全池 manifest 中。extra 是相对 ARC-GEN `a15cbdb44c776610aeeb9f487a06af875d3d0878` 的生成器覆盖差集，不代表底座未见规则。

固定版本源码落在 `/private/tmp/arc026-synth-rearc-src`，`.git/HEAD` 为上述 detached commit。只获取顶层 `dsl.py/utils.py`、包依赖和 109 个 task 子目录，没有下载其官方 evaluation 数据或合成 evaluation zip。生成 API 为 `synth_rearc.tasks.task_<id>.generate(diff_lb, diff_ub)`，返回独立 input/output pair；`verify(input)` 核对标签。

官方回放仅来自仓库已有的：

- `experiments/007/raw/arc-prize-2026/arc-agi_training_challenges.json`
- `experiments/007/raw/arc-prize-2026/arc-agi_training_solutions.json`

这两个文件的 SHA256 已记录。只取原 task 的 demonstrations 和原 query/target，不重组不同规则，也不通过 LOO 或重复增强凑回放行数。026 数据代码没有读取 evaluation solutions，也没有读取 025 query target 网格。`experiments/025/cohort-v1/challenges.json` 仅用于 task ID 排除检查。先前模型以及上游 NVARC 可能已训练于这些 training 规则；新 family 留出仅表示本轮续训未消费，不能称为模型未见验证。

## 预检查与隔离

109/109 个生成器各生成 1 个新 pair，均与同版本 verifier 一致。但 verifier 对本地官方 training demonstrations 为 335/338 一致；`17829a00`、`b74ca5d1`、`f18ec8cc` 各有 1 例不一致。这三族的合成材料及官方回放整族排除，不断言官方数据错误，不根据其合成自洽继续使用。

其余 106 族的官方 demonstration 全部匹配。全池 106 × 12 个 episode 均达到要求数量，无生成失败；每个生成 pair 重新由固定 verifier 核对。16 个 pilot 族另做源码规则解释和独立重建；其余族明确标为 `official_demo_and_pair_verifier_only`。**这些检查没有证明所有三示例 episode 都唯一可推断。** 跨 task ID 的同语义关系也尚未建立。

16 个 pilot 规则解释是：

| family | 本轮源码解释 | 核查边界 |
| --- | --- | --- |
| 14754a24 | 在受约束的局部加号里，将缺失的灰格改红 | 包含源码对黄色斜邻干扰的限制，未泛化到任意噪声 |
| 18419cfa | 在蓝框内部补齐红图案的双轴对称 | 只检查生成分布及原示例 |
| 2037f2c7 | 两个对象归一对齐后的轮廓异或，裁剪成蓝色输出 | 对象须满足生成器的分离条件 |
| 2de01db2 | 每行保留指定前景色的补集，删除其余噪声色 | 固定前景角色集合来自源码 |
| 3194b014 | 最大单色四连通块的颜色输出成 3×3 | 生成器确保唯一最大块 |
| 396d80d7 | 小颜色组决定填色，填大颜色组的纯斜邻背景 | 排除直邻及已有前景 |
| 5587a8d0 | 按颜色总量排序成同心方框 | 生成器使颜色计数互不相同 |
| 6df30ad6 | 最近点色决定灰对象重着色，并清掉点 | 源码平距时再按颜色数量排序 |
| 7bb29440 | 裁出红/品红标记总数最少的矩形 | 固定两种标记角色；生成器确保目标 |
| aa62e3f4 | 用对象最稀少色画其四邻外边界 | 对象和背景满足生成器约束 |
| aaecdb9a | 按五种固定颜色的八连通块计数画底齐柱图 | 固定颜色列序来自源码，不能认为任意三示例均已教全列序 |
| b15fca0b | 填两个红点间所有最短路径的并集 | 独立双向 BFS 核对，蓝格不可通行 |
| cd3c21df | 裁出归一后唯一不重复的多色对象 | 生成器为 1+2+2 对象频率 |
| d56f2372 | 裁出左右对称的颜色组 | 生成器确保唯一对称颜色组 |
| e729b7be | 把 180° 旋转后的图像只补在背景格 | 保留原前景，背景角色由最常见色决定 |
| f5c89df1 | 蓝图案相对绿锚的偏移复制到每个红锚 | 输出只保留复制后的蓝图案 |

以上是生成族规则解释，未据此给 025 的失败 query 贴语义标签。

## 独立性、示例覆盖与采样

每个合成 episode 为 **3 个新 demo＋1 个新 query**，使用独立 draw seed；全部固定 difficulty 范围 `[0,1]`。不保存 24 次旋转／换色扩增。全池训练侧 4320 个 pair 输入全部不同；1080 个 query 中，按同 family 的旋转／镜像／颜色重命名规范化后有 15 个结构重复，未将这些结构相似样本称为新规则。

拒绝同 episode 输入重复、训练合成池跨 episode 输入重复、全 demo 输出为常量、整 episode 恒等。相同 query 输出等于某个 demo 输出只标记，不删除；旧 007—011 已表明全局删除这个情况会造成类别示例缺失。单个恒等 pair 可保留并标记。`009d5c81`、`9bebae7a` 的固定符号形状，以及 `a6953f00` 的宽度奇偶分支，额外要求 query 的 cue 在至少一个 demo 中出现；其他条件映射未全量认证。`categorical_query_cue_covered=false` 表示未启用这三族的专项政策，不等于已确认覆盖失败。

留出由 seed `20261025`、106 合格祖先 ID 的固定 SHA 排序选最低 16，全部从合成 train 和官方 replay 排除：

`184a9768, 5adee1b2, 6df30ad6, 902510d5, 9f669b64, aa62e3f4, aaecdb9a, bc93ec48, bf32578f, c3fa4749, cc9053aa, d6542281, da6e95e5, db615bd4, e4888269, f5aa3634`。

实际 1024 行中：90 个合成 train family 每族 5 或 6 个 episode（28 族×5，62 族×6），全族轮转并在族内按长度四分位取样；四个桶实际为 128/129/128/127。官方回放每个祖先取 1 个原 query，共 512 family，按全长四分位各取 128。不按题内容、模型成绩或“容易程度”选择。两来源在消费顺序逐行交替。

原 `QwenFormatter` 和固定 16-token WordLevel tokenizer 的解析合约为 `full_length = grid_token_proxy + 6×(demo数+1)`。实际 tokenizer 尚由运行脚本复核；不把这个解析计数写成已运行 tokenizer。冻结片预计 min 90、median 1581.5、p90 4204、max 7464；只因完整序列超过 8192 显式排除全池中的官方 `f9d67f8b q0`，没有为凑长度删除 demo 或裁切 target。两来源约各半的行数并非各半 token：合成约 1224183 token、官方约 771176 token，实际监督 token 由训练入口记录。

## 手动运行

入口 [prepare_data.py](prepare_data.py) 只接受固定 training 来源，不接受任意 solution 路径；拒绝覆盖已有非空输出目录。生成运行使用 macOS arm64、Python 3.12.14、本地 CPU，无 GPU；全池约 39.252 秒，不含源码和依赖下载。冻结子集使用系统 Python 3.9.6。每 pair 上限 4 秒、每 family 上限 30 秒，失败写入 manifest，不默默凑数量。源码生成只依赖 stdlib 与 `utils.py` 的 matplotlib 导入；本轮 matplotlib 3.11.2 安装在临时目录，不修改项目环境。

```bash
PYTHONHASHSEED=0 PYTHONPATH=/private/tmp/arc026-data-deps \
  /Users/zhaohuaxiang/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/bin/python3 \
  experiments/026/prepare_data.py \
  --source-root /private/tmp/arc026-synth-rearc-src \
  --output-dir experiments/026/data-new-run --profile expanded

python3 experiments/026/prepare_data.py \
  --freeze-from experiments/026/data-v1 \
  --output-dir experiments/026/data-new-embedded --embedded-rows 1024
```

后续训练只使用冻结 1024 行和原 NVARC formatter；先做 tokenizer/CPU 检查、再由用户运行 GPU。训练初始化以外的 TTT、增强、DFS 和 B4 保持原配置，实际提分须由冻结真实题面板配对评价确认。
