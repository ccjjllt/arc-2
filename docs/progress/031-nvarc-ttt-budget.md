# 031 原NVARC TTT预算进展

## 当前状态

**031 V1／362740379已COMPLETE，完整12query评分2/12→2/12，增0／丢0／净0。** 第二槽压力与运行核查通过，但预定净增条件失败；结束固定64步配方，不正式提交、不扫描步数／LR／seed／排序救结果。保留原128步TTT／DFS0.2／B4和023正式30.14。[结果汇总](../../experiments/031/v1-summary.json)、[停止决定](../../experiments/031/v1-decision.json)。

已重新派出全新方向审计、CLI资源审计及Notebook实现子代理，没有沿用此前已完成的030代理。研究设计、预定停止／晋级条件见同编号计划。

## 输入冻结

人工执行 `python3 experiments/031/prepare_panel.py`，仅读取输入与成员元数据，生成8task／12query：221dfab4、2ba387bc、35ab12c3、67e490f4、88e364bc、c7f57c3e、d35bdbdc、dbff022c。

- challenge SHA：`2ffb3661a4b8588eb7dcfeb232d9ddec5de0122f343e6fc62697ed8198d8f581`。
- manifest SHA：`fd1382585393f7953f00cf1df555b1ff9cf39976f0f93dc411187aea7bbb029d`。
- 2ba是有意包含的历史第二槽压力例，其余7题按输入分层与固定哈希选取；全部是已曝光开发材料。

CLI再次显示zhuating GPU used27.92／30h、remaining2.08h，刷新10月31日08:00北京时间。此前完成的精确响应记录为7504.286配额秒、reserved0；本次展示值为四舍五入，未冒称重新取得精确值。未切换默认账户或打印凭证。

## 实现／效果分开记录

初始准备阶段，Notebook生成器实现ordinary8题等预算128／64配对，hidden完整test直接单遍64；原128步scheduler／n16数据保持，callback64停止，DFS0.2与B4不变。该阶段未上传或启动，随后上传事实见下节；不能把计划中的分数或时长填作结果。

## V1上传与启动

生成器已完成；CPU手工AST／JSON核查通过。原loader／decoder与023逐字相同，`train_args`、`peft_params`、`max_score`、`max_seq_length`与023的AST赋值相同。新独立代理复核原DFS两函数、评分及collator／Trainer定义无变，手工执行callback确认base64不停止、challenger64停止并拒绝64步scheduler；这些检查不是GPU效果证据。初审包ordinary输入被重排序列化的问题已修为保留冻结输入字节，未改变求解算法。

实际包 `/private/tmp/arc2-031-candidate-v1`，Notebook **93641字节**、SHA **10d2019a844bbd299dc6d89067f68581dce0b9f133ee4ab65358ed1068ae63fa**。预定条件在读本轮gold前已写入 `ready-v1.json`，引用计划SHA `30d3a8e66d06c2dfbaecf4c21bfdce597cdf6404ceaec07fdad9b6825b04be1b`。

通过用户授权的Kaggle CLI成功上传 [zhuating/arc2-031-ttt-budget64](https://www.kaggle.com/code/zhuating/arc2-031-ttt-budget64) **V1**，4×L4、断网、session最多3600秒；随后实际status为RUNNING。回读current_version_number=1、kernelId141118487，全部上传cell源与冻结包一致；kernelId不是scriptVersionId，后者尚待完成输出确认。没有创建competition submission。

首段实际日志记录Notebook **14:55:49北京时间**开始，输入为8task／12query且官方源SHA／面板SHA一致，4×NVIDIA L4各23034MiB。原utility已挂载，Python3.11.13；固定config与第一权重分片身份通过，第二片复制中。尚未核实完整模型加载、128步scheduler实际日志或64步停止，不能把boot成功称为完整验证。启动前精确quota：used100495.714／108000秒、reserved0，余额7504.286秒；本次会话预算7200配额秒为按系数2的上限规划，实际扣费仍待终态读取。

随后一次启动健康日志已确认：固定checkpoint10文件字节／SHA全部匹配023，复制7267271302字节、96.166秒；base rank0—3均实际启动，并观察到各轮两片加载完成消息。实际Unsloth2025.9.7、Transformers4.55.4、Torch2.8.0+cu128；未见Traceback或CUDA OOM。仍未见完整TTT stats，尤其challenger64的真实停止尚未验证，留到终态工件核查。证据见 `execution/v1-startup-health.json` 与原始日志。

启动健康确认后停止全部远端查询／日志连接，等待用户完成通知，不创建自动监控。人工接续命令见031 README：完成后先精确版本／来源／工件核验、无gold冻结，再运行031独立评分脚本；不依赖030已失败配方的晋级条件。031评分脚本仅手工AST通过，尚未执行评分或读本轮gold；当前没有新分数或正式submission。

## 用户通知完成后的实际结果

用户通知完成后，通过同一zhuating授权CLI实查COMPLETE。current_version_number仍为1，全部cell源与预冻结SHA `10d2019a...`一致；39项输出路径中的实际scriptVersionId唯一为 **362740379**。不是用kernelId或CLI后缀推定版本。

356项runtime工件manifest中，下载的35项全部SHA／size匹配；省略的321项仅为冗余候选pickle，有序raw／scored JSONL均完整。另有run-state、manifest本身、顶层submission与Notebook日志，共39下载文件。四个实际runtime源均仅有已知IPython末尾多1 LF，copy helper及固定checkpoint10文件身份一致。顶层 `submission.json`字节精确等于challenger原B4。非第三方源码的33项结果归档在 `experiments/031/completed-v1-output/`。

先运行通用无gold freezer，原B4两臂各12/12精确重放，10项预测／候选池／事件工件在本轮读gold前冻结；再以固定Kaggle官方challenge／solution快照评分，12/12输入及答案匹配。全新独立评分代理另以直接网格比较、oracle集合与完整task重算，结论一致。

| 指标 | base128 | challenger64 |
| --- | ---: | ---: |
| 最终两槽正确query | 2/12 | 2/12 |
| raw／scored oracle | 3/12 | 3/12 |
| 第一槽正确 | 1 | 1 |
| 仅第二槽正确 | 1 | 1 |
| 全query均正确task | 1/8 | 1/8 |
| 得到搜索的query | 12/12 | 12/12 |
| 完成搜索batch | 46/48 | 48/48 |
| 完成四批的query | 10/12 | 12/12 |
| 不同query-grid候选 | 114 | 118 |
| 完成TTT的task | 8/8，128步 | 8/8，64步 |
| 含公共准备／B4导出的部署账 | 1531.644秒 | 1409.424秒 |

增0／丢0／净0。`2ba387bc_0`两臂均第一槽错误、第二槽正确，风险检查确实覆盖并保住；`d35bdbdc_2`两臂第一槽正确。`d35bdbdc_1`两臂候选均含正确解但最终未选，这不是在本面板调排序救结果的许可。

实际4×L4，**14:55:49—15:42:30北京时间**，总墙钟 **2801.274秒（46.69分）**；公共setup118.160439秒，两臂各全额计入，solver global均1381.839561秒。公共收尾21.726636秒及逐臂B4导出计入部署账，均低于1620秒；两臂return0，无hard kill、未闭合阶段、Traceback或CUDA OOM。原soft/global结束提示属于原时限语义，没有从分母剔题。

真实TTT记录各8题均max_steps128、epoch1配置、warmup_ratio0.1、LR5e-5／cosine。两臂第64步LR均 `2.9415915362214692e-5`；base第128步归零，B第64步stop_requested=true且实际TrainOutput.global_step64。不是max_steps64或n8。base少的两批分别为 `88e364bc_1`、`dbff022c_0`各少1，B补齐却未新增正确query。

已闭合host阶段工作和TTT1471.233→760.671秒、DFS2698.113→3006.467秒、重评分483.350→505.464秒；部署账减少7.98%。阶段和不是CUDA profile或墙钟关键路径，独立TTT也不保证相同数值轨迹，不能把节省训练步骤当作同adapter纯因果收益或公榜提分。

## 停止与资源

预定判据中仅“最终两槽净增严格为正”失败，其余输入、第二槽、覆盖、训练路径和预算通过。按原计划 **结束固定64步配方，031不创建正式submission**，不扫描32／96步、LR、seed或排序。没有把本轮8题平局解释为隐藏全量覆盖必涨；本轮无新的公榜成绩，正式最好仍02330.14。

运行后精确quota used106153.032／108000秒、reserved0，余额1846.968配额秒（CLI显示0.51h，按当前4×L4系数约15.39分钟墙钟），本轮实耗5657.318配额秒；10月31日08:00北京时间刷新。没有启动新GPU会话、自动监控或额外submission。
