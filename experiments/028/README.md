# 028 原NVARC DFS路径复用机制实验

[计划](../../docs/plan/028-nvarc-dfs-throughput.md) · [进展](../../docs/progress/028-nvarc-dfs-throughput.md)

**当前结果：V2／362511155已COMPLETE，3题与15次调用全部完成。稳定候选题DFS调用耗时减1.57%；d35表面减6.63%但稳定缺1／增1候选网格，结束固定16-token配方，不晋级同预算或正式提交。** [完整汇总](pilot-v2-summary.json) · [工件核查](execution/artifact-verification-v2.json)

2026-10-26 21:43北京时间平台接受`zhuating/arc2-028-dfs-path-pilot`研究**V2**，最长5400秒。当前包SHA `0c9d58242c9dae02afb8afb411deda8a5e76ce5af14596b37b5a382fe40f5153`，来源[manifest](pilot-v2-package-manifest.json)及[回执](execution/pilot-v2-launch.json)。V1在TTT前因封装漏定义日志变量ERROR，已归档；V2只补此变量。启动健康另见execution，不把接受回执写成实验完成或提分。

固定原sft139、TTT128／seed1、eval2、DFS0.2；3task输入含5query，实际每task首query前两批，共3query。原版和路径proxy共用同一次TTT adapter，后批按原／改／改／原执行；草稿来自同次前批首个合法候选。最多预计算16token，原DFS callback保留。

这是raw候选／NLL／吞吐机制核查，**不执行B4、不评分完整task、不提交比赛**。完整机制结果已归档；B4、准确率与后续同预算收益未测量。正式最好仍02330.14，02729.72结束晋级。

产物入口：

- `panel-v1/`与`panel-selection.json`：无gold输入、选择规则及历史暴露。
- `dfs_path_reuse.py`：自有proxy／snapshot恢复，不复制第三方原DFS。
- `pilot_runtime.py`：同adapter、固定ABBA与GPU计量。
- `prepare_pilot.py`：生成第三方Notebook到`/private/tmp`，不自行启动。
- `execution/`：实际预检、上传回执、启动源与日志证据。

用户通知完成后人工执行`/private/tmp/arc2-028-download-completed-v2.py`读取精确V2输出，先核对身份与报告完整性，再比较三题实际复用、候选变化、NLL漂移和速度。旧V1为封装失败，不读取它来判断机制效果。未知runtime受控失败时如实结束／定位，不自动放宽cache断言或扫描参数。

21:50启动流已观察到rank0／1模型两片加载完成、无Traceback；rank2、TTT完整执行与path_A尚未观察。精确V2全部cell源一致、原模型指纹通过，详见[启动健康](execution/startup-health-v2.json)和[接续参数](pilot-v2-ready.json)。已停止远端查询，等待用户完成通知，不把加载正常当作机制验证通过。

用户通知后已核对真实V2终态与输出版本362511155、22项工件SHA及15次调用。原A／B完全一致，pathA／B各自一致；d35网格集合交集2/3。当前固定方案停止，不把表面变快当提分；先保留原023求解方案。
