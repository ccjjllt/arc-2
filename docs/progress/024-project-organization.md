# 024 项目清理与目录整理进展

日期：**2026-10-22，北京时间**。对应[024计划](../plan/024-project-organization.md)。Git基点`6ad6be6`，开始时工作树干净。本轮为用户授权的本地维护，研究主线仍为023。

## 1 当前项目情况

主线为从Failed in AIMO V1／312063593忠实复现可用正式起点。023普通GPU精确V1／361045383已COMPLETE，原4task／5query为3/5 query、2/4完整task；这是公开演示集合，不是比赛成绩。15:34已归档的同队正式额度为0，尚未创建023正式submission。最新正式成绩仍022／57504054的0.00，隐藏根因未确认。完整经过见[023进展](023-score-recovery-and-selection.md)。

本轮仅读取本地证据，没有重新查询比赛、额度或远端运行；没有GPU、模型加载、训练、推理、比赛提交或定时监控。R2、TRM和新训练继续后置。

## 2 实际清理

**释放冗余文件内容361,399,290字节，约344.66 MiB。**不含`.git`的项目文件占用约从1.31 GiB降至0.97 GiB；新导航文档仅增加少量文本。没有把Git内部对象计作垃圾或清理历史。

| 清理项 | 操作 | 释放内容字节 |
| --- | --- | ---: |
| Python／Unsloth缓存 | 删除7个缓存目录，合计56个缓存文件 | 4,170,430 |
| 012 K探针wheel | 删除3个本地下载，逐项SHA先匹配原manifest | 197,777,628 |
| 010重复adapter权重 | 保留step_4实体，另一读取路径改为相对软链接 | 159,451,232 |

删除的缓存目录：

- `scripts/__pycache__/`
- `experiments/011/kaggle-s1-pilots-v1/011/pilot-code/__pycache__/`
- `experiments/012/__pycache__/`
- `experiments/012/kaggle-cpu-preflight-v2/012/code/__pycache__/`
- `experiments/013/kaggle-baseline-output-v1/013-code/__pycache__/`
- `experiments/013/kaggle-baseline-output-v1/013-code/unsloth_compiled_cache/`
- `experiments/016/kaggle-v1/016-crb-code/__pycache__/`（原本为空）

012删除的文件为`causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl`、`fla_core-0.5.2-py3-none-any.whl`和`flash_linear_attention-0.5.2-py3-none-any.whl`。来源、大小和完整SHA保留在[原清单](../../experiments/012/wheel-manifest.json)，精确下载命令写入[恢复说明](../../experiments/012/wheels/README.md)。K路线已经停用，当前023不依赖它们；旧012打包器所有模式仍需先恢复wheel，本轮没有改旧代码规避该要求。未实际下载验证恢复命令，也未重新联网核查历史Dataset。

010保留实体为`experiments/010/kaggle-s1-v1/010/s1-f0/adapter_step_4/adapter_model.safetensors`。`adapter/adapter_model.safetensors`改为`../adapter_step_4/adapter_model.safetensors`相对链接；两个路径读取的SHA256均为`4187df14d4e8f70a4ffc3c54936a6c9329b96b8d2d828cf706ad226074ecfa51`，内容不变。该权重及链接沿用原ignore规则，仅为本地资产，不在Git差异中。

所有唯一adapter、007 optimizer state、episodes、原始数据、冻结源码／Notebook、日志、候选池、评分账本与正式就绪参数均保留。未按空文件统一删除，因为空`final.txt`和零字节训练状态可能是失败／截断证据。仓库外的`/private/tmp/arc2-*`冻结包未清理。

## 3 目录与阅读入口

- [根README](../../README.md)精简为当前主线、实际成绩和目录图，删除重复、过时的状态摘要；其依据仍在原progress。
- [docs/README.md](../README.md)收完整轮次索引，[Kaggle操作经验](../kaggle-workflow.md)收原首页历史操作记录。
- [experiments/README.md](../../experiments/README.md)区分023当前交付、020—022必要依赖及历史研究。
- [023/README.md](../../experiments/023/README.md)直接链接正式参数、运行凭据、输入输出与评分，明确历史准备清单不代表当前状态。
- [scripts/README.md](../../scripts/README.md)按用途说明11个共享脚本。
- `.gitignore`补充macOS元数据和Notebook检查点，合并重复的Python字节码规则。

保留所有编号目录和共享脚本的原路径。50个文件引用`scripts/`，脚本互相按同目录导入；023打包直接依赖021原V1，022配对依赖020评分器和E20输入，021打包也引用020源码。此次整理通过清理冗余和分层导航改善可读性，不扩成求解代码迁移。

## 4 手动核验与交接

- 本地Python3标准库逐文件核对：**2740项既有研究文件／唯一大资产的SHA均未变化**，包含全部原跟踪的实验文件、共享源码、计划和进展，以及未跟踪的权重、训练状态、episodes。
- 权重链接解析到保留的step_4实体，两个读取路径哈希相同；012三个下载删除前逐项通过清单SHA核验。
- 清理后缓存目录为空；新增／改写导航的本地链接及`git diff --check`经手动核对。
- 两个独立子代理分别审查清理边界和路径依赖，目录导航另作只读复核。未新增测试文件、目录或框架。

本轮没有新增模型／比赛研究结果。后续直接按[023当前产物入口](../../experiments/023/README.md)承接待正式额度步骤；无需因整理重跑普通保存或更新冻结Notebook版本。
