# 026 NVARC 4B额外规则覆盖续训配对

[研究计划](../../docs/plan/026-nvarc-extra-rule-training.md) · [实际进展](../../docs/progress/026-nvarc-extra-rule-training.md)

研究V1已于2026-10-25 15:28北京时间接受启动，15:34确认前2步训练正常后停止查询，精确凭据见`execution/paired-v1-launch.json`。本轮只检验一个小规模续训权重。固定公开33.89精确V1，保留原TTT／DFS／B4；16题／23query上新base与challenger同次运行配对。尚无新评分，不宣称提分。

## 冻结产物

- `data-pilot-v1/`：16个可读生成器族，CPU独立规则重建核查。
- `data-v1/`：106合格族全池，90训练／16族隔离；3个与官方demo不兼容的族隔离。全量verifier一致只属弱语义核查。
- `data-embedded-v1/`：实际输入1024 episode，512合成＋512官方training replay，保留family／长度覆盖；gzip430154字节。
- `panel-v1/`：训练前冻结16题／23query；分层使用025历史正确性，属于开发集。另有23题确认预留，不称盲测。
- `execution/`：配额、tokenizer来源、上传与启动的真实凭据；未产生的记录不可推断已经运行。

## 手动运行入口

`prepare_data.py`和`prepare_panel.py`仅本地处理；`train_specialist.py`含真实16-token的query-only标签检查与单GPU受限续训；`prepare_paired.py`从成功025包生成断网研究Notebook。生成第三方源码只落`/private/tmp`，不进入仓库。离线训练合并为普通BF16 checkpoint后，独立进程重新加载并创建原rank256 TTT adapter。

总会话上限10800秒，训练阶段含加载／保存／合并最多3600秒、评价至多7200秒（收尾计入总预算）。实际步骤与耗时以运行日志为准；训练loss不当作ARC分数。正常启动后停止查询，等待用户通知完成。

用户通知完成后，可人工执行`/private/tmp/arc2-026-download-completed-v1.py`，仅下载元数据／日志／候选，跳过所有权重，保留远端checkpoint。然后先核查日志、预测和候选元数据；大权重留在Kaggle输出，第三方`.py`落私有tmp。核对来源和归档身份，再冻结：

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

之后逐query核对两槽新增／丢失、oracle与阶段覆盖。只有两槽净增严格为正才进入预留23题的一次确认；零或负结束此配方，不在16题上救分。此处不创建正式比赛submission。
