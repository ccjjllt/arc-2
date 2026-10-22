# 023 当前交付入口

截至 **2026-10-22 15:40北京时间**的本地记录：忠实V1普通GPU运行已COMPLETE，原4task／5query评分为 **3/5 query、2/4完整task**。这是公开演示集合结果，不是比赛成绩。15:34正式额度为0，尚未创建本轮比赛submission；本次目录整理未重新查询远端。

| 要找什么 | 文件 |
| --- | --- |
| 下一次正式提交使用的冻结参数 | [formal-v1-ready.json](formal-v1-ready.json) |
| COMPLETE状态、版本与输出SHA | [execution/faithful-v1-outcome.json](execution/faithful-v1-outcome.json) |
| 独立运行核验 | [execution/ordinary-v1-runtime-verification.json](execution/ordinary-v1-runtime-verification.json) |
| 评分前预测冻结 | [execution/ordinary-v1-prediction-freeze.json](execution/ordinary-v1-prediction-freeze.json) |
| 四题输入身份与划分 | [ordinary-v1-input/manifest.json](ordinary-v1-input/manifest.json) |
| 原始输出与完整日志 | [ordinary-v1-output/submission.json](ordinary-v1-output/submission.json)、[full.log](ordinary-v1-output/full.log) |
| 输入匹配评分与逐query结果 | [score-report.json](scoring-ordinary-v1-f328/score-report.json)、[query-ledger.jsonl](scoring-ordinary-v1-f328/query-ledger.jsonl) |
| 打包入口与当时的准备清单 | [prepare_faithful_v1.py](prepare_faithful_v1.py)、[faithful-v1-package-manifest.json](faithful-v1-package-manifest.json) |
| 后置的R2事后回放 | [selection-replay/README.md](selection-replay/README.md) |

正式目标为`zhuating/arc2-023-failed-in-aimo-v1-faithful`，**kernel_version=1，scriptVersionId=361045383**。额度恢复后先重查同队额度、pending与该版本是否已经提交，再按就绪参数提交一次；无需重跑普通保存或更新版本。本页不构成定时监控。

`ordinary-v1-output/submission.json`保留120task的完整公开结构，只有原四题实际求解，不能将整个文件称为全量模型评测。完整候选池未取得，oracle仍未确认。

`faithful-v1-package-manifest.json`的`prepared_not_launched`是**打包当时的历史状态**；当前状态读上表的就绪参数和outcome，保留原清单字节。冻结包位于清单记录的私有临时目录；打包脚本依赖[021原V1](../021/reference/failed-in-aimo-v1.ipynb)与外部copy helper，不要批量清除`/private/tmp/arc2-*`。

后续研究顺序及完整经过见[023计划](../../docs/plan/023-score-recovery-and-selection.md)与[023进展](../../docs/progress/023-score-recovery-and-selection.md)。
