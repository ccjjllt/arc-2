# 011：GDN 加速路径后续证据与最小探针

查阅时间：2026-10-11，最终核对 05:18 UTC（北京时间 13:18）。状态：只读源码、发行资产及本地产物核验；没有安装包、下载权重、启动 GPU 或修改 core。下述探针**未实现、未运行**。

同日S1 pilot结束后补记：已有研究作业的真实日志记录`torch_cuda="12.8"`、`cxx11_abi=true`、`triton="3.6.0"`，见[环境及运行身份摘录](execution/s1-pilots-publication.json)和`kaggle-s1-pilots-v1/arc2-011-s1-pilots.log`中的`KERNEL_ENV`。因此下文“实际ABI/Triton未确认”是05:18的历史状态，现已补齐这两项；einops、二进制导入、L4 launch/JIT、数值及训练兼容性仍未确认。这个研究作业没有安装候选内核，不能视为加速探针已经完成。

结论：存在与当前版本标签相符的离线候选包，值得先完成 CPU 准备，再用一次有硬时限的小探针确认。尚无证据保证整套在 L4 上可用或显著更快；本周不应直接占用 GPU 尝试 NVCC 源码编译、反复换依赖版本。固定 Qwen3.8-27B。

## 1. 真机事实和口径

- `execution/s0-v2-load-evidence.json` 的 `event.elapsed_s=310.335`；Torch `2.10.0+cu128`、Transformers `5.8.0`、Accelerate `1.13.0`、BF16。
- `event.runtime_identity.gdn` 共 48 层：全部 `causal_conv1d_fn=null`，其余三个算子均为 Transformers 的 `torch_*`，norm 全为 `Qwen3_5RMSNormGated`。这证明当前走回退路径，尚不能证明其占总耗时的比例。
- 本地 `execution/s0-v2-live.log` 快照去重后前六条 `candidate_done`：480/113.141、336/79.692、512/116.458、836/202.503、512/119.372、1432/340.859（token/秒），即 4.128–4.396 token/s。这是调用总时间口径，含 prefill；第六条为 `invalid_output`，不把速度当解题能力。
- Python 3.12、PEFT 0.19.1 是当前会话环境约束；上述 load JSON 自身没有记录它们，也没有记录 Triton、einops、CXX11 ABI、driver、compute capability，探针须补齐。不能由 torch 版本号替代实际 ABI 读取。

证据 SHA256：load JSON `350b58f090a622e913d47931ae24723a68a4ded19853860f010fb14c1a8a5f9c`；查阅时 live log `0c8b7098f62e4323f674f7af2ea56ddebfa959dfddd5998ac0e686ef399621d0`。live log 可继续更新，SHA 只对应该快照。

## 2. 可精确准备的候选资产

| 资产 | 官方来源与完整性 | 静态匹配及未确认项 |
| --- | --- | --- |
| `causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl` | [官方 v1.7.0 资产表](https://github.com/Dao-AILab/causal-conv1d/releases/expanded_assets/v1.7.0)，页面显示 187 MB；SHA256 `8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7` | Linux x86_64、Python 3.12、Torch 2.10、CUDA 12 标签吻合；**必须实读 `torch._C._GLIBCXX_USE_CXX11_ABI is True`**。二进制导入及 L4 launch 未确认。 |
| `fla_core-0.5.2-py3-none-any.whl` | [PyPI 0.5.2 元数据](https://pypi.org/pypi/fla-core/0.5.2/json)，819225 B；SHA256 `5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761` | Python ≥3.10，基础依赖 einops，CUDA extra 声明 Torch ≥2.7、Triton ≥3.3。当前 Torch 满足；实际 Triton/einops 未确认。 |
| `flash_linear_attention-0.5.2-py3-none-any.whl` | [PyPI 0.5.2 元数据](https://pypi.org/pypi/flash-linear-attention/0.5.2/json)，399590 B；SHA256 `dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400` | Python ≥3.10，依赖 `fla-core==0.5.2`、Transformers ≥4.45.0；TF 5.8 满足元数据约束，不等于该组合已经 GPU 验证。 |

[causal-conv1d 精确 wheel](https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl)、[fla-core 精确 wheel](https://files.pythonhosted.org/packages/2d/ed/dfe19c4da779957eb6a42a26812f9b4e2280bf757a17a71933ff59ffcb98/fla_core-0.5.2-py3-none-any.whl)、[flash-linear-attention 精确 wheel](https://files.pythonhosted.org/packages/90/d2/2070e3cf2148c5cce99ca4876633c4b7b89ec085323600bd3d47aeacd306/flash_linear_attention-0.5.2-py3-none-any.whl)。这些是已核实可定位的发布资产，本次没有下载后复核其内容。

FLA 已拆分 core 与外层模型包；本次建议同时准备两个版本固定的小 wheel（合计 1218815 B），避免靠旧版单包布局推断导入。[官方安装说明](https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.5.2/INSTALL.md)。纯 Python wheel 仍有运行时 Triton JIT；不能写成“无需编译”。

causal-conv1d [v1.7.0 构建源码](https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.7.0/setup.py)按 CUDA 主版本、Torch 主次版本、ABI 和 Python tag 匹配 wheel；源码安装找不到 wheel 会转本地编译。其编译目标含 sm80；NVIDIA 说明 Ampere 二进制前向兼容 Ada，因此未单列 sm89 不是充分拒绝理由，但实际 wheel 内二进制与 driver 仍须验证。[NVIDIA Ada compatibility guide](https://docs.nvidia.com/cuda/ada-compatibility-guide/index.html)。算子声明支持 BF16、卷积宽度 2/3/4、SiLU。[官方 README](https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/v1.7.0/README.md)。

不安装 FlashQLA/TileLang 或额外模型包；这次候选是标准 Triton GDN 路径。PEFT 0.19.1 保持不变：没有查到“上述 FLA+PEFT+L4 整套已验证”的上游证据，不能仅凭依赖范围给它认证。

## 3. TF 接入与训练边界

[TF 5.8 Qwen3.5 实现](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)分别导入 causal-conv1d 和 FLA；多 token 路径调用 chunk，单 token 缓存解码调用 fused recurrent。构造时把函数存入每个 GDN 实例，norm 类型也在构造时选定。**模型载入后安装包，不会自动替换现有 48 层。** 需新进程或完整、明确的探针内切换；不能只看安装成功。

[TF 可用性检测](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/utils/import_utils.py)要求可导入 fla、CUDA 可用且版本 ≥0.2.2，并有缓存。CPU-only 环境显示该能力不可用，并不能单独否定 GPU 路径。

[FLA chunk 源码](https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.5.2/fla/ops/gated_delta_rule/chunk.py)的 q/k/v、g、beta、initial_state、output_final_state、use_qk_l2norm_in_kernel 参数与 TF 调用静态对得上，且有 backward 实现。[fused recurrent 源码](https://raw.githubusercontent.com/fla-org/flash-linear-attention/v0.5.2/fla/ops/gated_delta_rule/fused_recurrent.py)的 backward 明确 `NotImplementedError`。首轮只验证 `eval/no_grad` 推理；decode 成功不能证明 TTT 训练成功，后者需另做 `use_cache=False` chunk 路径的非零 LoRA 梯度检查。

## 4. 最小同题性能探针（方案，未运行）

1. **先在 CPU 准备**：保留现有 Torch/TF/PEFT，记录平台、Python、实际 ABI、Triton/einops、CUDA runtime/driver；固定以上三个 wheel 并核 SHA。独立 Linux 进程从本地明确文件 `--no-index --no-deps` 安装，检查依赖、导入错误；不跑 `pip -U`，不上传源码 sdist，不在 GPU 会话解析新依赖。缺依赖先解决 CPU 准备，不升级底座栈。工作量粗估 30–60 分钟人工准备，尚未测量。
2. **单次 GPU 探针上限 30 分钟**，并计入主实验预算；不是额度承诺。JIT/小算子验证最多 10 分钟，超时或 ABI/launch/NaN 失败即结束，不现场 NVCC 编译、不穷举版本。load 使用已观测约 310 秒作预算参考，但保留 10 分钟硬上限。
3. **只加载一次相同 BF16 27B**，固定原底座、禁用 adapter、同 device map、同 tokenizer/template。使用已有 P12 的 `db118e2a` query 0 及它原有完整 prompt；不读 query 答案，不新增题族或改 prompt。先小规模 BF16 chunk/recurrent/conv 前向及 cache 数值检查，再加载模型。
4. **最小 A/B 可先固定 torch norm**：在独立探针中让构造使用原 `Qwen3_5RMSNormGated`，A 使用原四个回退函数，B 仅替换 48 层的 conv/chunk/recurrent 实例函数为已保存的官方导入。两臂不换权重/norm/LoRA，保留并恢复原函数引用；记录每层实际 provenance。这样能避免第二次载入或复制完整模型，但这是“GDN+conv 算子探针”，不声称已经验证 stock FusedRMSNormGated 全路径。全栈采用前仍须单独核验 fused norm。此切换尚未实现，不能当成已有命令可直接运行。
5. 同一 prompt、同一固定 64-token continuation（从 A 的 greedy 输出取得，不是标签），两臂 teacher-force 完全相同 token，均从空 cache 开始；分开记录 prefill/首 token 和后续 64 步。首次 JIT、一次 warmup 与随后两次交替测量分开，测时前后同步四卡。记录每卡 peak allocated/reserved、有限值、相对 RMS/logit 误差和每步 top1 一致性；误差若明显大于重复基线或出现首个 top1 分歧，先保留证据调查，不直接替换比赛后端。
6. 两次暖态配对都节省至少 15% 的完整固定工作量墙钟时间，才值得继续一题完整 greedy 输出比较；15% 是本轮成本决策门槛，不是上游保证。完整输出单列变化/解析状态，不推断比赛提分。若收益小或首编译很贵，保留当前可运行路径；不要为此重跑训练或全 P12。

## 5. 四卡利用率与带宽的解释边界

[NVIDIA L4 官方规格](https://www.nvidia.com/en-us/data-center/l4/)为 24 GB 显存、300 GB/s 显存带宽，PCIe Gen4 x16 标称 64 GB/s。当前证据的 device map 把连续层分到 4 卡；batch1 单序列的后层依赖前层输出，各卡不能各自推进同一 token。因此卡间低平均忙碌比例不等于还能直接提速 4 倍。

18 个 shard 的文件大小之和为 **55,563,006,776 B（55.563 GB）**。若理想化地把它当作每 token 都需流读一次的有效权重字节，并假设层串行且各卡均达标称带宽，则：

`sum_i(W_i / 300e9) = 0.185210 s/token`，倒数约 **5.399 token/s**；不能改算 `55.563 GB / (4 × 300 GB/s)`。这是该简化模型的带宽时间下界，**不是当前模型实测吞吐上限**：文件含 header、输入 embedding 等不必每 token 全读的部分，缓存复用、实际活跃张量流量、投影/norm/状态读取、PCIe、kernel launch 和 prefill 都没精确建模。更严谨下界须从实际活跃参数和 profiler 流量求值。

故 5.399 与观测约 4.3 的比值约 1.256，只能说明“串行流读已可能占重要比例”；不能承诺 kernel 优化快 25.6%，也不能断言最大仅快 25.6%。优化 GDN/conv 主要减少小 kernel 调度、状态读写及 prefill 的中间张量；不会消除 MLP/投影权重带宽成本。以后 batch2 可以在同一权重读取上服务两个序列，方向是提高总 token 吞吐，不保证单请求延迟减半，还需承担不同长度 padding、cache/显存和候选独立性的复杂度。本轮不重构 batch2，先取得上述有界配对证据。
