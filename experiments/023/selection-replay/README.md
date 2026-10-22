# 已有E20候选的事后CPU回放

日期：2026-10-22，北京时间。只读取022冻结候选、原B4输出与输入对齐评分账本，未新增模型调用。

| 政策 | 正确query | 完整task | 相对B4新增／丢失 |
| --- | --- | --- | --- |
| B4原kgmon前二 | 3/27 | 2/20 | — |
| R1：最低mean_aug_nll前二 | 4/27 | 3/20 | 1／0 |
| R2：原首槽＋其余最低mean_aug_nll | 4/27 | 3/20 | 1／0 |

唯一新增`3a25b0d8_0`。R1/R2无序网格集合在全部query上相同，不是独立重复验证。21个query有候选、6占位，oracle仍5/27。政策没有使用task ID分支或gold选槽，只检查了这两条政策；但E20及其错误已经看过，所以这是事后开发诊断，不能称预注册或未见验证。

手动回放方法：

1. 读取`experiments/022/e20-output-v2/022-run/candidate-pools.json`和`experiments/022/paired-e20-v2/submission-B4.json`，候选池已按原kgmon排序和网格去重。
2. 对每个query按`mean_aug_nll`升序稳定排序；相同值保持原kgmon顺序。
3. R1取前两个不同网格；R2取原pool第一格，再取NLL最低且不同的一格。不足两个不同候选时保留原空槽占位（022为`[[0]]`）。保留全部task/query原顺序。
4. 写出两份submission，然后沿用已有`query-ledger.jsonl`中的输入对齐gold SHA评分。网格SHA算法为UTF-8编码的`json.dumps(grid, ensure_ascii=False, sort_keys=True, separators=(',', ':'))`再做SHA256。
5. 与B4逐query计算new/lost/net，并以所有query全对统计完整task。没有读取新答案网格或重新推理。

`mean_aug_nll`是答案token NLL之和在增强视角及记录间的均值，不是每token归一化loss或概率。原`kgmon=support-mean_aug_nll`，当前系统已有增强重评分；本次未新增评分模块。

产物：[完整报告](report.json)、[R1输出](submission-R1.json)、[R2输出](submission-R2.json)、[逐query回放账本](query-ledger.jsonl)。报告含输入与输出SHA、CPU平台、Python版本、耗时和上游推理版本。新GPU调用为0。后续只将R2作为待确认挑战者，不据本结果修改正式提交或宣称超过30.14。
