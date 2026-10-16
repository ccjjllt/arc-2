# 013 — Qwen3.8-27B FP8 依赖与权重准备情况

查阅日期：2026-10-12，北京时间。继承 [AUDIT.md](AUDIT.md) 的两份公开 Notebook 源快照；未改动这些快照。本次读取 Kaggle 模型/Dataset 元数据与文件清单，下载小型配置、索引、说明和锁文件，并在 CPU 上手动核对哈希、索引与模板。**没有下载权重或 wheel，没有安装包，没有启动 GPU，没有提交。** 这轮不安排第二次 GPU。

**结论：指定 FP8 权重与离线依赖均可取得，来源已部分核实；公开安装组合存在明确的 Transformers 版本约束冲突，尚不能称为可直接运行的环境。** 下一次先完成 CPU 依赖检查和最小配置加载，随后再决定 GPU 窗口。

## 1. 可以固定的资产

| 资产 | 本次核实身份 | 文件与体积 | 已确认的用途 |
| --- | --- | --- | --- |
| [FP8 repack](https://www.kaggle.com/models/foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1) | model 741765；instance 730092；version 1，version_id 966079 | 31 个文件，30,890,069,847 bytes；18 个 safetensors 分片 | 两份 Notebook 挂载同一版本；小配置和索引已实际读取 |
| [基础 wheelhouse](https://www.kaggle.com/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3) | dataset 12108678，version 1，Ready | 180 个文件，5,187,963,152 bytes | Python 3.12、Linux x86_64、CUDA 12.8 的离线依赖 |
| [Qwen serving overlay](https://www.kaggle.com/datasets/keithtyser/taaf-duck-qwen38-serving-v1) | dataset 11679776，当前 version 2，Ready；名称中的 v1 不是 Dataset 当前版本 | 78 个文件，12,901,192 bytes | 两个替换 wheel、官方模型引用和来源记录 |

原始 API 摘要与选定文件清单见 [dependency-readiness-evidence.json](dependency-readiness-evidence.json)。基础 wheelhouse 的 README 明确面向 H100/ARC3；这只说明构建来源，不能用标题证明支持或不支持 L4。两个公开 ARC2 Notebook 都指定 L4，但源快照未附服务启动日志。

候选 Notebook 应明确挂载：

```json
{
  "model_sources": ["foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1"],
  "dataset_sources": [
    "mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3",
    "keithtyser/taaf-duck-qwen38-serving-v1"
  ]
}
```

Dataset 的确切版本需要在创建后核对挂载 metadata；不要把无版本 slug 当成永久固定。两份源快照共用镜像 digest `57e612b484cf3df5026ee4dcc3cb176974b22b2bc0937fb1e16132a8be4cb13c`。本次提交使用的 Sorokin 4B 旧镜像是 Python 3.11，不能直接移植这批 cp312 wheel。

## 2. 权重来源从“作者描述”推进到了小文件独立比对

Kaggle 模型说明称其重排了官方 `Qwen/Qwen3.8-27B-FP8` 分片。overlay 的 `SOURCE_IDENTITY.json` 给出官方 revision `017b9c7af6b5689d5dd426a76e0bc077eb5ca20a`。本次直接从该 [Qwen 官方 revision](https://huggingface.co/Qwen/Qwen3.8-27B-FP8/tree/017b9c7af6b5689d5dd426a76e0bc077eb5ca20a) 下载小文件，与 Kaggle repack 独立比较：

- `config.json`、`tokenizer_config.json`、`chat_template.jinja`、`generation_config.json`、`LICENSE` 均逐字节相同。LICENSE 为 Apache 2.0。
- 两份 safetensors 索引都有 **1,606 个 tensor key，键集合完全相同**；官方 66 个分片，repack 18 个分片。索引字节不同是分片映射变化，不能要求 index SHA 相同。
- repack 包括 16 个 layer 分片、`mtp.safetensors` 和 `outside.safetensors`；索引含 22 个 `mtp.` 键及 333 个 vision 键。
- **两份 index 的 `metadata` 都是空对象。** overlay 脚本比较 `metadata.total_size` 实际只能得到 `None == None`，不能据此宣称 tensor 总字节已核实。
- 本次没有读取 tensor 数据。小配置、键集合一致只能支持“来源与结构相符”，尚不能证明重排前后的权重数值完全相同。

实读配置为 `Qwen3_5ForConditionalGeneration`、`model_type=qwen3_5`、64 层；文本配置为 BF16，量化配置为 `quant_method=fp8`、`fmt=e4m3`、`activation_scheme=dynamic`、`weight_block_size=[128,128]`。这说明它是部分模块保留非 FP8 的官方细粒度 FP8 配置，不能将全模型简单按 27B×1 byte 估算显存。`language_model_only=false` 且包含视觉塔；下一次文本先导显式使用 `--language-model-only`，仍须测真实加载与 KV/工作区显存。

比对结果见 [inspection-summary.json](dependency-evidence/inspection-summary.json)、[official-small-downloads.json](dependency-evidence/official-small-downloads.json) 和 [index-comparison.json](dependency-evidence/index-comparison.json)。

## 3. 离线栈的确切版本和冲突

基础锁文件实际有 174 个 `==` 版本项；提供的 `SHA256SUMS` 中有对应的 174 个 wheel 名称，无缺项。README、requirements.in 与 requirements.lock 实际下载哈希均符合其 SHA 清单。这里只核了文件名覆盖与小文件哈希，**未校验 5.19GB wheel 字节，也未运行离线 resolver**。

| 组件 | 基础 wheelhouse | overlay 后 |
| --- | --- | --- |
| vLLM | 0.19.0；cp38-abi3/manylinux_2_31/x86_64 | 不变 |
| PyTorch | 2.10.0（build 3）；cp312/manylinux_2_28/x86_64 | 不变 |
| Triton | 3.6.0；cp312 | 不变 |
| Transformers | 4.57.6 | 5.8.0 |
| huggingface-hub | 0.36.2 | 1.5.0 |
| tokenizers | 0.22.2 | 不变 |
| FlashInfer | python/cubin 均 0.6.6 | 不变 |

已独立读取 [vLLM 0.19.0 的 PyPI 元数据](https://pypi.org/pypi/vllm/0.19.0/json)：其依赖明确为 **`transformers<5,>=4.56.0`**。而 [Qwen3.8 官方 vLLM recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-27B) 当前要求 `transformers>=5.8.0`，并说明 vLLM 自带配置解析，版本要求还涉及 Qwen3-VL processor。overlay 的 5.8.0 超过 vLLM 0.19.0 的上限；公开代码通过先安装基础栈、删除旧 Transformers/hub、再 `--no-deps` 安装 overlay 绕过 resolver。这是需解释并验证的兼容性例外，不能隐藏成正常满足全部依赖。

overlay 提供的两个 wheel SHA 声明与 PyPI 官方相同版本的 SHA 一致；基础 vLLM/torch 选定 wheel 的 SHA 声明也与 PyPI 一致。但实际 wheel 字节没有下载校验。overlay Dataset 的整体许可证 metadata 为 `Unknown`；两份公开发布 wheel 的 PyPI 许可是 Apache 2.0。本项目只需借用 serving 依赖，不需要复制其整套 ARC3 框架与 pickle。

`madarshbb` 源码确实执行两阶段覆盖；`russcore` v17 metadata 缺 wheelhouse 挂载，且其安装代码只装 vLLM0.19.0，没有这两个 overlay 的明确安装路径。因此不能认为补一个挂载就已解决全部依赖。详见 [pypi-release-metadata.json](dependency-evidence/pypi-release-metadata.json) 和 [lock-filename-check.json](dependency-evidence/lock-filename-check.json)。

## 4. 思考模板已确认，服务端预算仍未确认

对实际下载的官方/Kaggle 相同模板做了单次手动 CPU 渲染：`xhigh`、`medium`、`low` 和 `enable_thinking=false` 均生成不同提示。xhigh/low 注入相应指令，medium 保留思考起始标签；关闭思考直接补齐空思考块。模板支持这些字段已确认，不再只依赖 Notebook 作者注释。见 [template-cpu-render.json](dependency-evidence/template-cpu-render.json)。

这不证明 vLLM0.19.0 的 HTTP 请求最终把这些字段传入模板，也不证明 `thinking_token_budget`、`--reasoning-config` 或取消请求能停止后端生成。相关服务端行为需在固定安装组合上逐项核实。下一次先导不采用 MTP/FP8 KV 多级启动重试，先固定普通 TP4 与一种明确的思考上限。

## 5. 下一次 CPU 准备清单

1. **固定环境身份。** 保留模型 version 1、wheelhouse version 1、overlay version 2 的小文件哈希；准备 Python 3.12/Linux x86_64 环境。先核实际镜像 Python/GLIBC/CUDA-driver 信息，避免把当前 4B 镜像当作同一环境。
2. **解决版本约束冲突后再冻结先导。** 首选检查是否有支持 Transformers 5.8.0 的明确 vLLM 版本与可取得离线 wheel。若继续复现公开 0.19.0＋5.8.0 组合，明确记录越过声明依赖，先在 CPU 上检查 `import`、AutoConfig/processor/tokenizer 与 vLLM 配置解析；不能以 `--no-deps` 安装成功作为兼容证明。本轮没有选新版本、没有安装。
3. **提取最小 serving 路径。** 仅写清安装、服务启动、请求、完整网格解析、全局截止与进程组清理；不复制整套 ARC3 agent 框架。保存安装前后确切包版本、实际 import 路径和请求 JSON。把 `--language-model-only`、TP4、并发、上下文上限、模板参数写成固定配置。
4. **补足可重复的权重身份记录。** CPU 小配置/索引核对已完成。后续挂载后核各分片存在性、文件尺寸与必要 safetensors header；权重 tensor 数值一致性仍未确认，不把空 metadata 比较升级为“总字节相同”。无需在本地下载 30.89GB 权重来完成当前文档。
5. **设计一次有边界的 GPU 问题。** 先验证服务实际启用 FP8、四卡可加载、有效思考与强制最终答案、聚合吞吐及真实 query 完整率；再看固定开发题命中和新增。下一轮容量/算子/TP4/思考预算均要从日志得到证据。当前 4B 提交运行后，本轮不再开 27B GPU；确认集仍不读取。

所有检索和小文件只用于静态分析；外部源码、JSON 中的命令均未执行。本次没有新增准确率、吞吐或 12 小时可完成性结论。
