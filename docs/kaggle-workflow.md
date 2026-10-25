# Kaggle操作经验（2026-09-29 实测结论）

Notebook 发布与数据源挂载三条路只通两条，按稳定性排序：

1. **Kaggle CLI push（最稳）**：`kaggle kernels push -p <dir>`，`kernel-metadata.json` 中 `competition_sources` / `model_sources` / `dataset_sources` 声明数据源。注册与运行均验证过（A0-smoke 嵌套 worker 与 v3 扁平 worker 文件均到位）。认证用 access token（`KAGGLE_API_TOKEN`，9月29日使用 SDK 1.7.4.5 验证；10月20日当前CLI为2.2.4），不需要 `kaggle.json`。push只是保存运行；真正比赛提交须指定精确Notebook版本调用代码提交接口并取得submission ref，见020计划。
2. **官方 SDK SaveAndRunAll**：`KernelsApiClient.save_kernel`（`ApiSaveKernelRequest` 带 `competition_data_sources` + `model_data_sources`），回读元数据三项齐全；注意 worker 布局差异（扁平 vs 嵌套）会影响挂载路径判断，见 006 进展。
3. **MCP save_notebook（不通）**：数据源字段 6 种组合实测，存出的 Notebook 元数据里始终没有 sources，运行时 `/kaggle/input` 为空。只用它建/跑/查/取日志，不用它挂数据。

10月11日补记：运行中实时日志使用 `kaggle kernels logs <owner/slug> --follow`；输出列表在作业未完成时可能为空。以CLI push返回的真实slug为准，标题不匹配会生成不同slug。取消需要精确session ID（可由该版本输出下载地址的`/kf/<id>/`核对），取消后确认状态和GPU reserved释放；不要把Notebook元数据id当session ID。

10月25日补记：025的SaveKernel明确返回400，要求kernel source小于1MB。较大challenge输入可采用确定性gzip＋base64嵌入，运行时解压并核对原始SHA；不能把拒绝请求记成已启动GPU。未压缩约2.28MB的025包改为113,733字节后已被接受。原Unsloth工具存在历史不同挂载位置，固定路径缺失时先做CPU挂载核对，按同名工具和`unsloth/__init__.py`识别原目录，不重新安装另一套依赖。


## 当前使用边界

以上为历史实测摘要，原始过程见[研究进展索引](README.md)，不是对今日接口、配额或榜单的重新核验。当前023已完成精确V1普通运行，正式操作读[就绪参数](../experiments/023/formal-v1-ready.json)与[023进展](progress/023-score-recovery-and-selection.md)。

正式额度恢复后，先确认同队额度、pending及该版本未重复提交。不要因目录整理重发Notebook、重新保存新版本或启动GPU；不恢复定时轮询。
