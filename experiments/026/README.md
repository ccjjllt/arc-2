# 026 NVARC 4B额外规则覆盖续训配对

[研究计划](../../docs/plan/026-nvarc-extra-rule-training.md) · [实际进展](../../docs/progress/026-nvarc-extra-rule-training.md)

研究V1／362051855已COMPLETE。训练256步、1024例完整消费，合并成功；两臂各16题／23query全部完成原TTT与4批DFS。总墙钟7211.426秒，约120.19分，无报错／OOM。

| 指标 | 原sft139＋B4 | 续训＋B4 |
| --- | --- | --- |
| 最终两槽正确query | 4/23 | 4/23 |
| 完整task | 2/16 | 2/16 |
| raw／scored oracle | 6/23 | 9/23 |

新增2、丢失2、净增0。**结束此续训配方，保留原权重，不进入预留23题、不提交此challenger。** 这是公开开发面板，不是Kaggle成绩；正式最好仍30.14。

[完整汇总](paired-v1-summary.json) · [评分](scoring-v1-f328/score-report.json) · [逐query账本](scoring-v1-f328/query-ledger.jsonl) · [下载身份](execution/paired-v1-outcome.json) · [归档SHA核查](execution/artifact-verification-v1.json) · [冻结预测](frozen-v1/prediction-freeze.json)

## 冻结产物

- `data-pilot-v1/`：16个可读生成器族，CPU独立规则重建核查。
- `data-v1/`：106合格族全池，90训练／16族隔离；3个与官方demo不兼容的族隔离。全量verifier一致只属弱语义核查。
- `data-embedded-v1/`：实际输入1024 episode，512合成＋512官方training replay，保留family／长度覆盖；gzip430154字节。
- `panel-v1/`：训练前冻结16题／23query；分层使用025历史正确性，属于开发集。另有23题确认预留，不称盲测。
- `execution/`：配额、tokenizer来源、上传与启动的真实凭据；未产生的记录不可推断已经运行。

## 手动运行入口

`prepare_data.py`和`prepare_panel.py`仅本地处理；`train_specialist.py`含真实16-token的query-only标签检查与单GPU受限续训；`prepare_paired.py`从成功025包生成断网研究Notebook。生成第三方源码只落`/private/tmp`，不进入仓库。离线训练合并为普通BF16 checkpoint后，独立进程重新加载并创建原rank256 TTT adapter。

总会话上限10800秒，训练阶段含加载／保存／合并最多3600秒、评价至多7200秒（收尾计入总预算）。实际步骤与耗时以运行日志为准；训练loss不当作ARC分数。正常启动后停止查询，等待用户通知完成。

本轮已完成以下人工下载／冻结／评分步骤；复核入口为`/private/tmp/arc2-026-download-completed-v1.py`，仅下载元数据／日志／候选，跳过所有权重，保留远端checkpoint。然后先核查日志、预测和候选元数据；大权重留在Kaggle输出，第三方`.py`落私有tmp。核对来源和归档身份，再冻结：

```bash
python3 experiments/026/freeze_paired.py --run-root DOWNLOADED/026 \
  --manifest experiments/026/panel-v1/manifest.json --output experiments/026/frozen-v1
python3 experiments/020/score_frozen_outputs.py \
  --manifest experiments/026/panel-v1/manifest.json \
  --challenges experiments/026/panel-v1/challenges.json \
  --prediction base_B4=experiments/026/frozen-v1/submission-base_B4.json \
  --prediction challenger_B4=experiments/026/frozen-v1/submission-challenger_B4.json \
  --candidate-pool experiments/026/frozen-v1/candidate-pools.json \
  --official-revision f3283f727488ad98fe575ea6a5ac981e4a188e49 \
  --official-split evaluation --output-dir experiments/026/scoring-v1-f328 \
  --allow-gold-read
```

逐query核对已完成：新增7ed72f31_1、eee78d87_0，丢失7b0280bc_0、e8686506_0；最终净0。按预定条件停止，无后续GPU运行或正式比赛submission。
