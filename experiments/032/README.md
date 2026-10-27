# 032 最高分公开Notebook直接提交

**已正式提交57694986，待评测。** 用户2026-10-27明确要求直接用最高分Notebook提交；CLI scoreDescending第一仍为Failed in AIMO，直接引用[原作者精确V1／312063593](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593)，历史外部33.89。本次没有复制／修改代码、没有使用023副本或031研究改动、没有新ordinary GPU运行。

实际提交人为zhuating、队伍ccjjllt、文件submission.json；服务端列表精确URL绑定通过，当前PENDING，公／私榜分数均未返回。受理与后续读取证据在 `execution/`，[当前接续记录](ready-v1.json)、[进展](../../docs/progress/032-highest-notebook-submission.md)。不要重复提交或把外部分数当成本队新分数。

用户通知评测完成后，只执行一次只读官方CLI列表并核对ref57694986与scriptVersion312063593，再记录真实结果：

```bash
/Users/zhaohuaxiang/.local/share/uv/tools/kaggle/bin/python experiments/032/capture_submission_cli.py experiments/032/execution/submission-57694986-result-read.json competitions submissions arc-prize-2026-arc-agi-2 --page-size 20 --format json
```

`capture_submission_cli.py`是手动CLI响应记录入口，读取用户指定zhuating token但不写出凭证；命令是只读submissions时不会创建submission。不要使用普通公开Notebook的完成状态替代本队hidden评测状态。不创建自动监控。
