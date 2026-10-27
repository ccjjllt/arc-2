# 031 原NVARC TTT预算进展

## 当前状态

2026-10-27开始031，目标为在原NVARC上取得有实际配对依据的下一次正式尝试。0300.1净0已停止；正式最好仍为02330.14。本轮V1已启动普通配对，结果尚未返回，没有提分结论。

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
