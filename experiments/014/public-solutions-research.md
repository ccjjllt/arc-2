# 014 公开方案研究：先恢复有效预算，再验证 27B 的新增解

查阅日期：2026-10-17（Asia/Shanghai）。本文件只记录外部资料与源码审读；未运行 GPU、未安装依赖、未提交比赛。冻结源码、元数据和小型 manifest 位于 `public-sources/`，证据索引见 `public-solutions-sources.json`。

## 决策摘要

1. **没有找到可以凭可读源码直接复现的公开金牌/奖金级 27B 版本。** 新公开的 Qwen3.8-27B Notebook 有作者明确指向版本的约 21.5 分报告，同时作者报告小改后跌至 13。它是可借用的部署样例，不能当作提升到 40+ 的证据。
2. **0.19.0 + Transformers 5.8 的依赖冲突有替代资产，尚未有本项目实跑。** 找到官方 commit 固定的 vLLM 0.27.2 开发版完整离线 wheelhouse，CPU 已读其 manifest/requirements；它省掉本地重新打包运行时的工作。是否能在实际 L4 镜像启动、吞吐是否足够，必须通过一次短先导验证。
3. **高吞吐新 fork 值得保留线索，但不宜直接押注。** 最新 27B Merlin 优化只在 sm86 RTX3090 + P2P 验证，27B 补丁的激活条件也限制 sm86；Kaggle 新 sm89 wheelhouse manifest 明确 `gpu_tested=false`。不能将作者 3090 的速度转写为 L4 速度。
4. **短期优先恢复已有解题路径的有效推理预算，停止广泛加数据训练。** 最新一手合成训练报告用了 A100 数十小时却没有明确 ARC2 增益；同题、同时间、两次实际提交答案的新增解更重要。
5. **所有候选并集均不等于可提交成绩。** 保留一条旧 top-1 只能保护该 top-1，不能保证不丢原 attempt-2 命中的题；新 27B 结果也必须经过实际二选验证。

## 1. 最有用的新 27B 公开版本：部署素材，不是高分保证

- [新 Notebook](https://www.kaggle.com/code/madarshbb/arc-agent-sub-q38-direct-b98k-c16-792ef9?scriptVersionId=358248812)：API 可读当前 v1，源码无输出 cell；权重为 `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`，仍是 013 已核的小文件身份。
- [作者 10 月 14 日的版本链接与分数报告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748082#3538430)：明确贴上述 `scriptVersionId`，称这一份约 21.5；同时称微小更改后 13。**证据是作者绑定版本的报告，未读账户提交明细、未复现分数。**
- 对比 013 源码，实际新增是 CUDA 动态链接修补和预算配置：context 81920→131072，client concurrency 10→16，thinking cap 40960→98304，max tokens 49152→106496。有效路径仍 `mode=direct`，不是标题写的已验证 Python 程序搜索。
- 新 `cuda_link_dir()` 寻找实际挂载的 `libcuda.so.1`，创建 `libcuda.so` 并加入 `LIBRARY_PATH`；作者注释解释 FlashInfer JIT 曾因 `-lcuda` 失败。这是具体可复用的启动修补。
- 离线安装仍为 vLLM0.19.0 基础 wheelhouse 后 `--no-deps` 覆盖 TF5.8/hub；没有解决声明依赖冲突。MTP/FP8-KV 多层 fallback 也没有附成功启动日志，不能确定有效层。
- 全题第一样本优先、渐进写答案、思考尾部强制最终网格可提取；98K token 与 131K context 不应照抄为本项目先导默认值，容易先消耗完预算而丢覆盖。

同一讨论中 mikelou1 10/16 报告约 40，称不含 NVARC 集成；但未提供源码/准确模型身份/版本，且前面明确不准备开源。这只能表明有值得继续问证的线索，不能把他的旧公开 `Minimal Perfpatch` 赋予 40 分，也不能把预计 60/85 当作已实现分数。另有 27B 公评 80%、公榜 26.81 的作者报告，未解释成本和评测口径；不作为我们选预算的依据。

## 2. 可更快进入 27B 实验的现成离线栈

[Qwen38 vLLM0272 wheelhouse v1](https://www.kaggle.com/datasets/nick2187/qwen38-vllm0272-cu130-wheelhouse-v1) 的 manifest 已实际下载：

- Dataset id 11654954，version 1；文件列表含 193 个 wheel，总 wheel 字节 4,900,988,522。
- vLLM 源 commit `1f7427bc0ad31fdad1b0933c77ea5acae15da2e0`，版本 `0.27.2rc1.dev91+g1f7427bc0`，来源 URL 指向 `wheels.vllm.ai`，wheel SHA256 在 manifest。
- Python3.12 / Linux x86_64 / CUDA13，Torch2.13，Transformers5.15.0，FlashInfer0.6.16.post3。requirements 允许 TF>=5.8；不是旧0.19的<5冲突。
- manifest 中“Qwen3.5 runtime/parsers/MTP fix存在”为发布者检查，**不是本项目 GPU 验证**；未下载 wheel 字节，也未完成独立完整依赖解析。
- Dataset 整体许可元数据为 Unknown；各运行时组件需依据各自许可证和来源记录使用。该资产来自 ARC3 serving，上层 agent 与本任务无关，不复制其业务框架。

[PyPI 原始依赖](https://pypi.org/pypi/vllm/0.27.0/json) 已独立读取：0.20–0.23 开始允许 TF5.8，0.24–0.28 为 TF>=5.5.3；0.29/0.30 已要求 TF>=5.10.4。因此“换最新版 vLLM，继续旧5.8 overlay”仍不成立。版本证据不是能力进步，只消除一个可定位的启动障碍。

建议下一个部署实验只固定一套：此 wheelhouse + 已核官方FP8 repack + 实际支持CUDA13的L4镜像；先测真实生成、完整结果、显存和聚合吞吐。若CUDA13/镜像不匹配，明确回退到已公开0.19+5.8的兼容性例外，并在短窗口内以真实请求验证，不能在一个窗口同时比较三套运行时。首轮先不启用 MTP/FP8-KV 叠加功能。

## 3. 新吞吐开源实现的价值与边界

[halt95/qwen38-27b-3090s v1.3.1](https://github.com/halt95/qwen38-27b-3090s/tree/v1.3.1)（10/13）提供 Qwen3.8-27B W4A16、FP8 KV、MTP K4、前缀缓存等具体实现；同一底座的量化部署方向，不代表更换基础模型。作者 quad3090 在32K的单流短生成约249 token/s，测量为 low effort、256输出token，并非ARC2工作量。它需要额外量化权重和补丁，目前仅sm86+P2P验证。

本次实读 `upstream/PIN`、27B patch README、serve脚本和依赖文件：核心27B优化经 `series_active` 限制为 dense Qwen3.5 架构且 sm86。Kaggle 10/14新 [sm89 wheelhouse](https://www.kaggle.com/datasets/nisshokuitsuki/qwen-merlin-h95-wheels) 虽含Torch2.13/TF5.18和sm89预编译FlashInfer模块，其描述仅为Flash-Next Merlin基础及sm89 QSA reader变更；manifest明确 `gpu_tested=false`，不能认定27B那11项补丁已移植、已启用或数值正确。

因此保留作后备加速研究，不把整个比赛剩余时间投入改fork。优先普通vLLM批处理路径，只有测出主要瓶颈且普通路径不足，才挑单项修补。

## 4. NVARC 公开改进应怎样借用

[ARC Baseline Rebuild v101](https://www.kaggle.com/code/yusuketogashi/arc-baseline-rebuild) 实读源码给出具体思路：训练后独立计算推理截止，依剩余题数分预算；评分增强种子由随机进程hash换为crc32；对零候选题降低DFS门槛重试。源码的实际公式是 `clip(1.5*fair_share - train_elapsed, 300, 1200)`，不是无限延长。若借用必须用本项目覆盖/时长记录检验，不能与多项变化捆绑后归因。

`Minimal Perfpatch` 的GPU端NLL计算只能当候选小优化；NVARC专用模型已裁词表的历史资料意味着不应假定在搬运15万维logits，未在本轮读取该权重config。无实际吞吐结果，不承诺显著加速。

另读 `luxluxshan/arc2-nvarc-v1` v3：两次64训练序列/8视角比单次128/16更分散，pool候选后保护pass-A top1。其“同一代码多次26.9–32.2”是作者报告，不能推断本项目25.69一定只是随机波动。保护top1不能保护原top2所有正确题；需按最终二答案实测。

## 5. 最新合成数据研究提示暂停无对照扩训练

[10/13 原作者实验报告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/748895) 描述新规则家族数据与NVARC追加训练：A10080GB，LoRA约68.5小时，新增块约57.2小时。合成留出改善，但真实ARC2公开eval基线17/172，完整342K LoRA18/172、新增块17/172；作者报告完整比赛pipeline也没有明确收益。这是外部作者报告而非本项目重跑，尚不能证明所有合成训练都无效；足以降低“再多训一些就会涨”的优先级。

其1369规则家族可作为未来数据覆盖与失败类别分析素材，但现在不下载大数据、不启动342K级训练。继续27B专项训练应先找到基座确实缺失、可用少量数据验证的能力，不能重复011/012无命中配置扩算。

## 下一步可执行顺序

1. 继承主审计对25.69的机制分析；在相同开发题、相同时间预算上与013冻结配置比较。历史30.14精确源码仍未取得，不声称已与它完成配对；源码可见的预算/候选风险不等于隐藏发生率已确认。
2. 并行准备最小27B vLLM先导，固定模型、运行时、输出协议；记录time-to-first-token、总decode、完整网格、截断、OOM/抢占、每题耗时及最终两个答案。
3. 先比较一条中等预算direct与有预算的最终答案恢复，获得真实覆盖后再增加程序执行反馈。未形成完整可执行程序前不要声称反馈方法失败。
4. 只有27B在旧方案错题上有真实新增，才设计时间切分和最终二答案选择。确认集保持未见，冻结配置后一次使用。
5. 不等待公开别人金牌版本，也不把未经运行的框架升级包装成研究成果。GPU先导完成与能力提升分别记录。

所有分数的证据级别、不能继承的口径及精确来源见JSON索引。

