# 025 公开最高分V1候选选择确认

2026-10-25用户通知结束后实查，[研究Notebook](https://www.kaggle.com/code/zhuating/arc2-025-v1-selection-confirmation) **GPU V2／361992679为COMPLETE**，kernelId140698356，运行6684.238秒、未超时。40task均完成128步TTT，58query均获得搜索。GPU V1／361989657早期ERROR保留为失败记录。

**本轮确认未提分：B4=R2=15/58，新增0／丢失0，完整task均9/40；raw／scored oracle均17/58。** 58query全部与固定官方版本输入匹配。按预定条件保留原B4、结束R2扩展；下一方向是NVARC 4B候选生成能力，针对性数据／续训小对照尚未设计执行。此处为公开诊断成绩，正式最好仍为30.14。

| 结果 | 文件 |
| --- | --- |
| 完整汇总、覆盖归因与成本 | [confirmation-v2-summary.json](confirmation-v2-summary.json) |
| B4／R2输入匹配评分 | [score-report.json](scoring-v1-f328/score-report.json)、[逐query账本](scoring-v1-f328/query-ledger.jsonl) |
| 两份策略输出的事先冻结 | [prediction-freeze.json](replay-v1/prediction-freeze.json) |
| 实际运行终态与精确版本 | [confirmation-v2-outcome.json](execution/confirmation-v2-outcome.json) |
| 709项产物SHA核对 | [artifact-verification-v2.json](execution/artifact-verification-v2.json) |

起点是[Failed in AIMO精确V1／312063593](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593)，公开33.89。研究包经023已跑通的最小运行修复派生，保持原TTT、DFS和重评分；本项目正式30.14保留为对照。

| 产物 | 用途 |
| --- | --- |
| [cohort-v1/manifest.json](cohort-v1/manifest.json) | 固定40task／58query、分层seed、排除理由与数据接触限制 |
| [cohort-v1/challenges.json](cohort-v1/challenges.json) | 无query答案的冻结输入 |
| [prepare_cohort.py](prepare_cohort.py) | 重新生成输入划分的方法；现有v1不覆盖 |
| [prepare_confirmation.py](prepare_confirmation.py) | 从冻结最高分V1包装研究Notebook，不自行发起远程运行 |
| [replay_selection.py](replay_selection.py) | 无gold的同池B4／R2回放；核对原B4、保存完整分母与候选 |
| [confirmation-v2-package-manifest.json](confirmation-v2-package-manifest.json) | 实际运行包、原V1与输入的SHA、允许差异和时间预算 |
| [execution/confirmation-v1-launch.json](execution/confirmation-v1-launch.json) | 平台实际接受的启动回执 |
| [execution/confirmation-v2-launch.json](execution/confirmation-v2-launch.json) | 挂载核对后实际恢复的GPU V2回执 |
| [execution/remote-source-verification-v2.json](execution/remote-source-verification-v2.json) | 远端V2逐cell源一致性 |
| [execution/remote-source-verification-v1.json](execution/remote-source-verification-v1.json) | 远端V1逐cell源一致性 |
| [execution/public-notebook-reference.json](execution/public-notebook-reference.json) | 本次公开Code列表最高分参照核验 |
| [execution/confirmation-v2-startup-health.json](execution/confirmation-v2-startup-health.json) | 当时四个worker加载完成的启动证据；当前读终态outcome |

早期V1包在`/private/tmp/arc2-025-v1-selection-confirmation-compact`，Notebook SHA为`9a83821330e510d267ca7bdeed12a2f5aa34161ea0b0aeca22bc8e6dedd5dd08`，仅供失败追溯。输入采用gzip＋base64嵌入，解压后精确核对原字节SHA；修复了平台小于1MB源码的限制。早期未压缩包的400拒绝保存在execution，不代表已经运行过一次模型。

实际V2包位于`/private/tmp/arc2-025-v2-mount-ready`，SHA为`ca8c03dde65b9eca40ed5ea8e48938ef58737994b2a2e122b76edc5c671cf4ef`，清单为[confirmation-v2-package-manifest.json](confirmation-v2-package-manifest.json)。它增加最多60秒的原工具／固定模型就绪等待与同名目录识别，校验023已成功使用的模型配置指纹；求解及输出cell与V1相同，总预算6900秒。GPU实际utility为`/kaggle/usr/lib/pip-install-unsloth-flash-patch`，运行已完成，不能重复执行启动脚本。

实际V2平台上限6900秒，求解软截止6780秒、进程组硬截止6840秒，包含加载与收尾；未触及这些截止。原1200秒逐task软检查保持。全部58query保留分母，实际都已搜索。日志与raw候选导出有少量CPU／I/O开销。

## 本次人工取回与评分

本轮没有设置定时任务。当前V2人工Kaggle操作脚本为`/private/tmp/arc2-025-kaggle-v2.py`，使用当前研究账号；`inspect`只读远端状态／源码，`download`仅在终态下载，不能用`launch`重复发起已有Notebook。真实ref固定为`zhuating/arc2-025-v1-selection-confirmation`。V2结果目录为`confirmation-v2-output/`；下方示例使用该目录，旧V1失败产物保持原样。

实际产物位于`confirmation-v2-output/025/`，原第三方源单独留在私有临时目录。下方命令已执行完成，先冻结两份策略输出，再调用共用评分器：

```bash
python3 experiments/025/replay_selection.py \
  --manifest experiments/025/cohort-v1/manifest.json \
  --challenges experiments/025/cohort-v1/challenges.json \
  --baseline experiments/025/confirmation-v2-output/025/submission-B4.json \
  --ordered-candidates experiments/025/confirmation-v2-output/025/scored-candidates.jsonl \
  --raw-candidates experiments/025/confirmation-v2-output/025/raw-candidates-rank0.jsonl \
  --raw-candidates experiments/025/confirmation-v2-output/025/raw-candidates-rank1.jsonl \
  --raw-candidates experiments/025/confirmation-v2-output/025/raw-candidates-rank2.jsonl \
  --raw-candidates experiments/025/confirmation-v2-output/025/raw-candidates-rank3.jsonl \
  --output-dir experiments/025/replay-v1

python3 experiments/020/score_frozen_outputs.py \
  --manifest experiments/025/cohort-v1/manifest.json \
  --challenges experiments/025/cohort-v1/challenges.json \
  --prediction B4=experiments/025/replay-v1/submission-B4.json \
  --prediction R2=experiments/025/replay-v1/submission-R2.json \
  --candidate-pool experiments/025/replay-v1/candidate-pools-scored.json \
  --official-split evaluation \
  --official-revision f3283f727488ad98fe575ea6a5ac981e4a188e49 \
  --output-dir experiments/025/scoring-v1-f328 \
  --allow-gold-read
```

按输入匹配的逐query账本统计B4／R2新增、丢失、净增、scored oracle及完整task命中；raw池另与相同gold SHA比较。按rank事件核对搜索机会和TTT／搜索／评分耗时。raw只覆盖已返回的DFS批次，不能把硬中断前尚未返回的内部候选计为已导出。公开结果用于选择研究方向，正式评分才确认比赛增量。

详见[025计划](../../docs/plan/025-research-gap-and-next-step.md)与[025实际进展](../../docs/progress/025-research-gap-and-next-step.md)。
