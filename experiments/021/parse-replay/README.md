# 021 Direct 输出逐 query 解析回放

日期：2026-10-21，北京时间。本地CPU手动运行，未调用模型、未使用GPU、未改020冻结输出或评分产物。

**现有C24和Deep6响应中，逐query保留没有恢复任何合法预测，也没有新增正确解。** 本轮不支持为这项解析修改单独开GPU验证或宣称它能挽回12.50提交的损失。它仍可作为后续多query返回的局部健壮性改动，与模型能力提升分开记录。

| 集合 | task / query | 可匹配官方答案query | 原合法预测 | 新合法预测 | 原exact / 新exact | 新增合法 / 新增exact |
| --- | --- | --- | --- | --- | --- | --- |
| C24 | 24 / 26 | 26 | 21 | 21 | 17 / 17 | 0 / 0 |
| Deep6 | 6 / 9 | 8 | 8 | 8 | 2 / 2 | 0 / 0 |

合法预测按真实模型返回计数，不把占位`[[0]]`算作模型完成。C24为official training来源项目留出，17/26不能外推竞赛成绩。Deep6有1个输入未匹配query，仍保留在原9个query账本中，但不赋予exact；可匹配分母为8。两集合不能合并为隐藏分数估计。

## 解析政策

旧政策直接从当前020源文件AST中提取`parse_direct`函数执行，并调用其原`valid_grid`。每个task的旧状态和query结果均与冻结run记录逐项一致，每个已匹配query的exact也与既有冻结评分逐项一致。

新政策仍要求一次完整标准JSON对象，严格拒绝重复key、非标准常量、未知query key、含额外文本的返回和不完整代码围栏。对象解析成功后，独立验证每个预期query的完整网格：缺失或非法成员失败，其他完整合法成员保留。不截取分析文字中的数组、不猜截断内容、不补网格单元格。

## 为什么没有恢复

- C24的`984d8a3e`只有一个query，网格行宽不一致；没有可保留的合法兄弟query。
- C24的`aa62e3f4`、`b7fb29bc`及`c3fa4749`耗满长度后，final内容仍是分析文字，不能解析为完整JSON。包括多query的`c3fa4749`，也没有可独立接纳的完整对象。
- Deep6的`800d221b`同样是长度终止后的分析文字，没有完整JSON对象。
- 其余多query返回原本已完整合法；独立验证与旧政策结果相同。

## 来源与复现

人工命令：

```bash
python3 experiments/021/parse-replay/replay_direct.py \
  --output-dir experiments/021/parse-replay/run-v1
```

输出目录必须尚不存在；重复观察时指定新目录。该脚本是对现有响应的手动分析入口，不是自动测试套件。

原模型为`Qwen/Qwen3.8-27B`的Kaggle FP8 repack v1；原运行TP4/C8、xhigh、generation最大49152、thinking最大40960、MTP2。C24响应来自`c24-direct-ccjjllt-v3`及`c24-direct-qingyu1232-v1`，Deep6来自`deep-direct-qingyu1232-v1`。

评分使用已有020冻结账本中、已与完整demonstrations及query输入匹配的官方真值SHA256。先校验原challenge/manifest的冻结SHA和每query输入SHA，再比较规范化网格SHA；没有新读取或下载答案网格。官方revision为`f3283f727488ad98fe575ea6a5ac981e4a188e49`，C24对应training，Deep6对应evaluation。

实际回放约0.038秒，本地arm64 CPU、Python3.9.6；完整平台信息、全部响应/输入/源码哈希见[report.json](run-v1/report.json)，逐query结果见[query-ledger.jsonl](run-v1/query-ledger.jsonl)。旧代码执行未修改；新解析只存在于本分析脚本，尚未并入021 GPU求解器。
