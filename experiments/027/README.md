# 027 原权重TTT视图集成确认与提交

[计划](../../docs/plan/027-nvarc-ttt-view-ensemble.md) · [进展](../../docs/progress/027-nvarc-ttt-view-ensemble.md) · [原始研究核查](research-notes.md)

研究V1／362101452 COMPLETE，精确Kaggle快照34/34输入匹配。**原B4为5/34，固定融合6/34，新增1／丢失0；前缀净增0／0／1，晋级条件通过。** 正式公榜尚未更新，最好仍30.14。

[确认完整汇总](confirmation-v1-summary.json) · [精确评分](scoring-v1-kaggle/score-report.json) · [晋级决定](promotion-v1.json) · [数据版本身份](execution/official-kaggle-scoring-snapshot.json) · [归档核查](execution/artifact-verification-v1.json)

**正式submission57634436已被接受**，绑定[精确V1／362150112](https://www.kaggle.com/code/zhuating/arc2-027-ttt-view-ensemble-formal?scriptVersionId=362150112)。普通4题／5query验证已COMPLETE，base／extra完整跑完，冻结融合4/5且无错误回退。初始提交绑定核对通过，状态字段及公／私榜分数尚未返回；当前正式最好仍30.14。

[提交回执](execution/competition-submission-v1.json) · [初始精确绑定](execution/submission-57634436-initial.json) · [普通完整汇总](formal-v1-summary.json) · [普通评分](formal-scoring-v1-kaggle/score-report.json)

## 唯一方案

固定原sft139，不用026权重。先4GPU以原TTT增强seed1完整跑base，再以seed2用余量跑extra；Trainer seed42、eval增强seed2、rank256／128步／DFS／B4不变。保留base第一槽，extra有真实候选且第一槽不同才补第二槽；相同／缺extra则保持base第二槽，base没真实候选时才fallback extra两槽。没有跨分支分数混合、ID白名单或种子扫描。

23task／34query确认输入及固定策略见`confirmation-v1/manifest.json`，query答案未读；它排除025四十、026十六与继承开发ID。旧公开推理／上游接触仍存在，不称盲测。分阶段预算让base保有原计算覆盖，不保证其得分一定与历史30.14相同。

## 产物与执行入口

- `prepare_confirmation.py`生成研究包，第三方Notebook只放`/private/tmp/arc2-027-v1-confirmation`。
- `freeze_ensemble.py`为固定纯stdlib selector及候选／预测无gold冻结CLI。
- `freeze_prefixes.py`在不读gold时按extra完成任务时间冻结前6／12／18任务可用时的fallback预测，检查部分补题的回归风险。
- `development-replay.json`为旧开发集少数预定义聚合回放，不含026面板或新确认评分。
- `execution/`为真实配额、提交名额、上传身份及运行状态；旧快照不当作实时状态。
- `confirmation-v1-package-manifest.json`与`confirmation-v1-manual-package-review.json`记录已上传源和手工审查。
- `prepare_formal.py`已生成正式包并上传普通V1验证；必须用隐藏test动态输入、原42600秒全局预算、base先冻结／extra出错回退。确认通过前不创建正式版本。

## 完成后的固定流程

正常启动后停止查询，用户通知完成才通过`/private/tmp/arc2-027-download-completed-v1.py`读取终态／下载小产物。先核对全部归档、执行源与原B4，再执行下列冻结，之后才能读gold：

```bash
python3 experiments/027/freeze_ensemble.py --run-root DOWNLOADED/027 \
  --manifest experiments/027/confirmation-v1/manifest.json --output experiments/027/frozen-v1
python3 experiments/027/freeze_prefixes.py --run-root DOWNLOADED/027 \
  --manifest experiments/027/confirmation-v1/manifest.json \
  --frozen-dir experiments/027/frozen-v1 --output experiments/027/prefixes-v1
python3 experiments/020/score_frozen_outputs.py \
  --manifest experiments/027/confirmation-v1/manifest.json \
  --challenges experiments/027/confirmation-v1/challenges.json \
  --prediction base_B4=experiments/027/frozen-v1/submission-base_B4.json \
  --prediction extra_B4=experiments/027/frozen-v1/submission-extra_B4.json \
  --prediction ensemble=experiments/027/frozen-v1/submission-ensemble.json \
  --prediction prefix25=experiments/027/prefixes-v1/submission-prefix25.json \
  --prediction prefix50=experiments/027/prefixes-v1/submission-prefix50.json \
  --prediction prefix75=experiments/027/prefixes-v1/submission-prefix75.json \
  --candidate-pool experiments/027/frozen-v1/candidate-pools-scored.json \
  --official-revision f3283f727488ad98fe575ea6a5ac981e4a188e49 \
  --official-split evaluation --output-dir experiments/027/scoring-v1-f328 --allow-gold-read
```

最终两槽query净增必须>0，可用前缀净增不得为负，base覆盖完整、输入全部匹配且无实现故障，才进入已授权正式提交流程。否则保留023并停止此固定方案，不用确认集挑种子、改规则救结果。oracle只是候选诊断，不能当公榜收益；以真实比赛结果判断是否提分。


## 本轮数据版本修正与正式接续

固定GitHub f328参考有4query真正输入不匹配（4a21e3da_0、f560132c_0、faa9f03d_0/1），初始30/34报告仅保留审计。之后官方Kaggle挑战文件SHA与实际推理源完全一致，配套solutions SHA也冻结；原30个gold哈希一致，完整34个均匹配。`score_kaggle_snapshot.py`复用020计分逻辑，未改变任何预测；`scoring-v1-kaggle`为正式采用的本地诊断。

普通验证完成的下载入口是`/private/tmp/arc2-027-download-formal-completed-v1.py`，输出将落`formal-v1-output/027`，来源代码落私有tmp。必须核对实际终态、base/extra/fusion、全部源和输出SHA、真实scriptVersion以及顶层submission.json，再按固定kernel V1直接提交；不能从研究scriptVersion362101452创建正式提交。提交名额与pending必须临近提交刷新。原四题为历史公开诊断，不将其当独立确认。


正式普通验证结束后，还可人工运行`/private/tmp/arc2-027-formal-completion-preflight.py`取得源cell一致性、实际输出scriptVersion集合、顶层submission.json绑定、提交名额和pending状态；此脚本只读不提交。严格绑定普通V1的实际输出，不能把kernelId140787141或研究版本362101452当作正式scriptVersion。

[正式接续参数](formal-v1-ready.json)已保存，scriptVersion／submission ref须在普通完成后以真实结果填写。


## 当前下一步

已提交且完成一次回执／绑定读取，停止远端查询，等待用户通知结果。人工入口`/private/tmp/arc2-027-read-submission-57634436.py`只读取57634436实际状态、分数和精确版本绑定；不要重提V1，也不要把普通Notebook的COMPLETE当作隐藏评测已经完成。正式回执和ready是当前事实，前文准备／下载命令保留作历史复核入口。
