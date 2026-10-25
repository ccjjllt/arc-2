# 023 当前交付入口

截至 **2026-10-25 10:05北京时间**实查：正式submission **57582282已COMPLETE／公榜30.14**，绑定精确V1／361045383，与历史最好持平，未实现提分，私榜未返回。见[实际结果](execution/submission-57582282-result-20261025.json)。普通原4task／5query的3/5仍仅为公开演示集合结果。

| 要找什么 | 文件 |
| --- | --- |
| 已完成正式提交的冻结参数与状态 | [formal-v1-ready.json](formal-v1-ready.json) |
| 本次正式30.14结果 | [submission-57582282-result-20261025.json](execution/submission-57582282-result-20261025.json) |
| COMPLETE状态、版本与输出SHA | [execution/faithful-v1-outcome.json](execution/faithful-v1-outcome.json) |
| 独立运行核验 | [execution/ordinary-v1-runtime-verification.json](execution/ordinary-v1-runtime-verification.json) |
| 评分前预测冻结 | [execution/ordinary-v1-prediction-freeze.json](execution/ordinary-v1-prediction-freeze.json) |
| 四题输入身份与划分 | [ordinary-v1-input/manifest.json](ordinary-v1-input/manifest.json) |
| 原始输出与完整日志 | [ordinary-v1-output/submission.json](ordinary-v1-output/submission.json)、[full.log](ordinary-v1-output/full.log) |
| 输入匹配评分与逐query结果 | [score-report.json](scoring-ordinary-v1-f328/score-report.json)、[query-ledger.jsonl](scoring-ordinary-v1-f328/query-ledger.jsonl) |
| 打包入口与当时的准备清单 | [prepare_faithful_v1.py](prepare_faithful_v1.py)、[faithful-v1-package-manifest.json](faithful-v1-package-manifest.json) |
| 后置的R2事后回放 | [selection-replay/README.md](selection-replay/README.md) |

正式版本为`zhuating/arc2-023-failed-in-aimo-v1-faithful`，**kernel_version=1，scriptVersionId=361045383**；57582282结果已确认30.14。保留为下一轮对照，不原样重复提交或重跑普通保存。下一步一次B4／R2扩展配对尚未运行，不创建定时监控。

`ordinary-v1-output/submission.json`保留120task的完整公开结构，只有原四题实际求解，不能将整个文件称为全量模型评测。完整候选池未取得，oracle仍未确认。

`faithful-v1-package-manifest.json`的`prepared_not_launched`是**打包当时的历史状态**；当前状态读上表的就绪参数和outcome，保留原清单字节。冻结包位于清单记录的私有临时目录；打包脚本依赖[021原V1](../021/reference/failed-in-aimo-v1.ipynb)与外部copy helper，不要批量清除`/private/tmp/arc2-*`。

后续研究顺序及完整经过见[023计划](../../docs/plan/023-score-recovery-and-selection.md)与[023进展](../../docs/progress/023-score-recovery-and-selection.md)。
