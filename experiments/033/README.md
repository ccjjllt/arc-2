# 033 NVARC GPU NLL实现

**57737364已COMPLETE，2026-10-29实查公榜30.56，较023提高0.42，保留为当前正式最好。** 精确公开[Minimal Perfpatch V2／335137615](https://www.kaggle.com/code/mikelou1/arc-agi2-lb33-89-minimal-perfpatch?scriptVersionId=335137615)、zhuating／ccjjllt绑定通过，私榜未返回。[实际结果](execution/submission-57737364-result-20261029.json)。这是整体公开实现的正式效果，不是单热点因果证据；不把标题33.89当作本项目成绩。

源／普通产物及差异在 `reference/`，[身份清单](public-v2-manifest.json)、[研究计划](../../docs/plan/033-nvarc-gpu-nll-submission.md)、[实际进展](../../docs/progress/033-nvarc-gpu-nll-submission.md)。模型实际16词表，不能夸大传输优化收益；GPU归约和use_cache=False可能影响数值与候选，正式结果才决定是否超过02330.14。

用户通知出分后，人工只读CLI列表并核ref57737364／scriptVersion335137615：

```bash
/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python experiments/032/capture_submission_cli.py experiments/033/execution/submission-57737364-result-read.json competitions submissions arc-prize-2026-arc-agi-2 --page-size 20 --format json
```

不重复提交，不从公开Notebook ordinary状态推断本队hidden评测状态；新分数严格超过30.14才更新正式最好。接续见[ready-v2.json](ready-v2.json)。
