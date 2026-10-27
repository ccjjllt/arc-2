# 030 原NVARC DFS搜索覆盖配对

2026-10-27 **pilot20 V1／362694905已COMPLETE并评分：两臂均6/28，新增2丢失2、净0，按预定规则结束0.1。** raw／scored oracle7→8，覆盖25→22query、四批22→13；15个共同TTT task的adapter SHA均不同，保留因果限制。[结果汇总](pilot20-v1-summary.json)、[评分](pilot20-scoring-v1/score-report.json)、[停止决定](pilot20-v1-decision.json)。不启动原40题扩展、不提交0.1、不扫阈值／seed或改排序救结果，保留原0.2／B4及正式30.14。以下命令为已执行流程与复现记录，不表示继续启动GPU。

研究决策及停止条件见[030计划](../../docs/plan/030-final-week-search-coverage.md)，实际交接见[030进展](../../docs/progress/030-final-week-search-coverage.md)。只比较原sft139／TTT／B4／runtime下的累计路径概率门槛0.2与0.1。

## 冻结材料

| 产物 | 状态／作用 |
| --- | --- |
| [panels-v1/manifest.json](panels-v1/manifest.json) | 初轮40task／58query，扩展40task／57query，task交集0；均已暴露回归材料 |
| [source-audit-v1.json](source-audit-v1.json) | 原V1、023与历史权重／源码SHA审计；原作者镜像等未知保持未知 |
| [pair-static-review-v1.json](pair-static-review-v1.json) | 本轮新代理对实际初轮／扩展包的静态审查，无阻塞；不是GPU成功证明 |
| [execution/manual-cpu-reader-check-v1.json](execution/manual-cpu-reader-check-v1.json) | 使用025已有候选复核原B4重放、完整58query与阶段事件汇总；未新评分 |
| [mechanism-v1-package-manifest.json](mechanism-v1-package-manifest.json) | 3task／5query、同adapter原DFS A1/A2=0.2、B=0.1，共15调用；不算准确率 |
| [initial-v1-package-manifest.json](initial-v1-package-manifest.json) | 首轮完整原B4配对，顺序base→challenger，4×L4 |
| [extension-v1-package-manifest.json](extension-v1-package-manifest.json) | 扩展配对，顺序challenger→base，预算与初轮相同；只在首轮通过后运行 |
| [ready-v1.json](ready-v1.json) | 当前有效临时包位置、源码及脚本SHA、就绪状态 |
| [panels-v1/pilot20/manifest.json](panels-v1/pilot20/manifest.json) | 当前额度受限先行20task／28query，读取030结果前input-only分层冻结；不是新盲测 |
| [pilot20-v1-package-manifest.json](pilot20-v1-package-manifest.json) | ≤7200秒总会话、每臂3300秒，机制通过后才可运行；[静态审查](pilot20-static-review-v1.json)通过 |
| [mechanism-v1-runtime-identity.json](mechanism-v1-runtime-identity.json) | 已验证实际运行源码身份，保留原包manifest；仅loader／decoder／starter多末尾1个IPython换行，solver逐字一致 |
| [execution/pilot20-v1-launch.json](execution/pilot20-v1-launch.json) | 历史启动记录；完成已另绑定362694905，见[工件身份](execution/pilot20-v1-artifact-verification.json) |
| [pilot20-frozen-v1/prediction-freeze.json](pilot20-frozen-v1/prediction-freeze.json) | 读取gold前精确重放B4并冻结两臂预测、候选、28query完整分母及事件 |
| [pilot20-scoring-v1/score-report.json](pilot20-scoring-v1/score-report.json) | 官方快照28/28匹配，6/28→6/28、净0；不是公榜分数 |

两臂均为6600秒部署槽；公共准备时间向每臂全额计费。global截止为本臂启动＋6600－公共准备－120，hard截止再晚60秒；模型导入、加载、TTT、DFS、重评分都在其中。最终B4导出及共同final import也计费，总账超过6600则比较无效。公共准备上限900秒。两个完整pair会话均最多14400秒，扩展没有额外加长每题计算。

同adapter小检查最多3600秒，首轮4小时、扩展4小时、普通验证预留1小时，共最多10墙钟小时，按L4系数2计划20配额小时，预留10配额小时。实际资源与扣费临运行重读；未核验正式隐藏rerun的额度扣费方式。

**当前余额执行修订：** zhuating实际启动前仅6.62配额小时，当前先做机制≤1小时及pilot20≤2小时，按系数2最多6配额小时，不直接启动需要约8配额小时上限的initial40。pilot20保持原阈值、hash、B4及净增>0且至少两task贡献新增的条件，不据其结果挑其他子集或阈值。该子集没有`2ba387bc`，不能宣称覆盖了已知第二槽独有正确风险；正式晋级前仍须原initial40的固定压力／覆盖复核及扩展结果。当前没有为后续40题会话假定额外余额。

用户已明确指定Kaggle CLI及token文件。所有后续账户操作使用以下子进程凭证选择入口，不更改全局默认认证；token不会写入日志／仓库：

```bash
python3 experiments/030/kaggle_zhuating.py config view
python3 experiments/030/kaggle_zhuating.py quota --format json
python3 experiments/030/kaggle_zhuating.py kernels status zhuating/arc2-030-threshold-mechanism
```

机制与pilot V1均已完成，两者都不要重复push；固定0.1配方已停止。普通`kernels logs`在运行中可能为空；历史启动检查采用有界`--follow`，关闭本地日志流不会停止远端会话。

## 手动准备与核对

当前已生成的Notebook都在`/private/tmp`，第三方生成源码不写入仓库。临时目录可能被清理；如缺失，按以下顺序重建，核对manifest SHA。不要自行升级原Unsloth／Transformers依赖。

```bash
kaggle kernels output zhuating/arc2-023-failed-in-aimo-v1-faithful -p /private/tmp/arc2-030-recovered-023 --file-pattern '^copy_model[.]py$' --quiet
```

复制脚本应为3253字节、SHA `5d51c4e6698ce5d574dd9997b140e0e370ea611fbee05d1de197b665a8458010`；下载命令自身不宣称版本绑定，以历史内容SHA为准。随后：

```bash
python3 experiments/023/prepare_faithful_v1.py --owner zhuating --slug arc2-023-failed-in-aimo-v1-faithful --out /private/tmp/arc2-030-023-base --copy-helper /private/tmp/arc2-030-recovered-023/copy_model.py
python3 experiments/030/prepare_mechanism.py --out /private/tmp/arc2-030-mechanism-v1 --base-package /private/tmp/arc2-030-023-base
python3 experiments/030/prepare_pair.py --owner zhuating --slug arc2-030-dfs-threshold-pair-initial --out /private/tmp/arc2-030-pair-initial-v3 --base-package /private/tmp/arc2-030-023-base --panel initial40
python3 experiments/030/prepare_pair.py --owner zhuating --slug arc2-030-dfs-threshold-pair-extension --out /private/tmp/arc2-030-pair-extension-v3 --base-package /private/tmp/arc2-030-023-base --panel extension40
```

生成器拒绝覆盖目录；已有有效包不重复生成。`v3`是本地准备修订，不是Kaggle远程版本；当前未创建任何030远程版本。重建023的Notebook应精确匹配 `0bdde6503f6ec616be467321649e9abb997288f07aa54e8b4302f76f32f3d119`。旧pair-v1／v2是本地CPU草稿，不能选其扩展包启动。

运行前用显式zhuating凭证的CLI只读刷新额度（保存时使用新文件名，不覆盖历史记录）：

```bash
python3 experiments/030/kaggle_zhuating.py quota --format json
```

当前pilot已完成且净0，勿重复上传或启动扩展。历史操作沿用[已有Kaggle工作流](../../docs/kaggle-workflow.md)，上传／运行是普通研究会话，**不能调用competition_submit_code提交这些带研究输入的包**；它们包含rerun拒绝逻辑。默认`read_preflight.py`仍读取全局认证，其结果不能代替显式zhuating CLI身份／额度。

先运行机制小检查；若A/A噪声失败或不完整，保留结果并人工分析，不自动重试。通过后运行初轮配对。初轮未通过停止0.1配方，不启动扩展。按用户既有习惯，确认启动健康后停止远端轮询，等待用户通知结束；没有预约或自动任务。

## 完成后先冻结，再评分

从实际完成的精确远程版本下载整个`030/`（机制为`030-mechanism/`），核对源码、输入、checkpoint和工件SHA。原B4及所有query分母保留，包括预算截止造成的缺失。然后：

```bash
python3 experiments/030/freeze_pair.py --run-root DOWNLOADED_OUTPUT/030 --output NEW_FROZEN_DIRECTORY
```

冻结器逐query重放原B4，保存两臂raw/scored池、第二槽和晚批覆盖诊断需要的事件、adapter指纹。它不生成任何新排序。

评分快照来源为027已核验的官方比赛JSON对，历史manifest在`experiments/027/execution/official-kaggle-scoring-snapshot.json`；其中旧`/private/tmp`文件已不存在。本轮没有为准备而读取／下载solutions。预测冻结后，才手动恢复官方challenge／solutions并写本轮snapshot manifest，核对以下SHA和完整输入一致：

- challenges：`e7c62a4bd211867c6b538f66b8013b81f299663c82ca062f49a52bf439d6e4e8`
- solutions：`84be4f4f39b79e82c36d565fc878830988b094917f052ee7069aef30b33ca8f1`

如果官方文件变化，明确记新快照并先核对全部输入，不能静默剔除query。评分命令示意：

```bash
python3 experiments/030/score_pair.py --frozen NEW_FROZEN_DIRECTORY --snapshot-manifest OFFICIAL_SNAPSHOT_MANIFEST.json --stage initial --mechanism-report experiments/030/mechanism-v1-output/mechanism-report.json --mechanism-manifest experiments/030/mechanism-v1-runtime-identity.json --output NEW_SCORING_DIRECTORY --allow-gold-read
```

`score_pair.py`先核对冻结工件，再读gold；机制报告必须与冻结机制包的源码、输入和种子一致。缺机制证据或A/A未确认，正净增也保持hold。初轮要求净增>0、至少两个不同task贡献新增；通过人工预算／晚批覆盖审查后，才运行扩展并以`--stage extension`评分。扩展仍须净增>0，才准备另外的正式挑战包。当前没有正式挑战包，也没有新正式成绩。

实际机制源码的IPython末尾换行差异已逐字验证，原包manifest没有改动。评分显式选择已绑定362686825的runtime identity，避免把正常序列化差异当成未知身份；它没有放宽A/A控制或修改实验结果。23项原工件在临时完整下载中已核对，第三方`sources/`保留在临时目录，仓库归档保留其SHA与身份记录。
