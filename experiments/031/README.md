# 031 NVARC TTT预算

当前[031 V1](https://www.kaggle.com/code/zhuating/arc2-031-ttt-budget64)普通配对运行中，实际scriptVersionId待完成输出确认；尚未创建正式submission，正式最好仍02330.14。研究设计和停止／晋级条件：[计划](../../docs/plan/031-nvarc-ttt-budget.md)；实际状态：[进展](../../docs/progress/031-nvarc-ttt-budget.md)。

面板8task／12query全部是已曝光公共开发数据。保留原128步学习率调度和完整n16增强，仅callback64停止；原DFS0.2／B4。ordinary配对与hidden完整单遍候选共用同V1，结果前冻结，不据普通结果修改hidden源码。

## 用户通知完成后的人工接续

1. 用用户授权的CLI读终态、固定V1源码及工件。CLI2.2.4的owner/slug/version后缀不能单独证明输出版本；使用此处只读response-hook helper，在官方CLI内部记录current_version_number及全部输出URL路径中的 `/kf/<scriptVersionId>/` 数字，绝不保存签名URL／token。

```bash
python3 experiments/030/kaggle_zhuating.py kernels status zhuating/arc2-031-ttt-budget64
/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python experiments/031/capture_cli_provenance.py experiments/031/execution/v1-completed-source.json kernels pull zhuating/arc2-031-ttt-budget64 -p /private/tmp/arc2-031-completed-source -m
/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python experiments/031/capture_cli_provenance.py experiments/031/execution/v1-completed-output-provenance.json kernels output zhuating/arc2-031-ttt-budget64 -p /private/tmp/arc2-031-completed-output --file-pattern '^(031/(?!.*inference_outputs/).*|submission[.]json)$' --page-size 200 --quiet
```

2. COMPLETE后确认current_version_number仍为1，所有cell源与冻结Notebook相同；输出实际scriptVersionId必须唯一，不能用kernelId141118487代替。对 `031/artifact-identities.json` 所列下载文件逐项核SHA／size，省略的项只能是冗余候选pickle，逐query有序raw／scored JSONL须完整。实际runtime源码只允许已观察过的IPython末尾补一个LF差异，其他差异阻断提交。顶层submission必须与challenger原B4字节相同，模型copy10文件及4×L4实际证据完整。TTT observer实际步数／LR及预算账由后续评分记录核实。

3. 先冻结原B4与候选池，再读本轮gold；复用030的通用无gold freezer，不复用030阈值晋级条件：

```bash
python3 experiments/030/freeze_pair.py --run-root /private/tmp/arc2-031-completed-output/031 --manifest experiments/031/panel-v1/manifest.json --output experiments/031/frozen-v1
python3 experiments/031/score_pair.py --frozen experiments/031/frozen-v1 --snapshot-manifest experiments/030/execution/official-kaggle-scoring-snapshot-v1.json --output experiments/031/scoring-v1 --allow-gold-read
```

固定官方快照若丢失，应通过同一授权Kaggle CLI重新下载challenge／solution并核对原SHA，另记本次来源；不要默默替换未确认数据源。评分必须完整12query，并由新独立代理用直接网格比较复核。

4. 若冻结条件全部通过，完成精确版本、当日名额及pending人工核查后，在既有授权下直接提交同一V1：

```bash
python3 experiments/030/kaggle_zhuating.py competitions submit arc-prize-2026-arc-agi-2 -k zhuating/arc2-031-ttt-budget64 -v 1 -f submission.json -m '031 original NVARC TTT64 original128-step schedule DFS0.2 B4'
```

记录真实submission ID、实际scriptVersion绑定及实际公榜结果，超过30.14才称正式提分。若条件失败，结束64步配方、不扫描步数／LR／seed／排序、不提交失败版本。不要创建自动监控或重复提交。
