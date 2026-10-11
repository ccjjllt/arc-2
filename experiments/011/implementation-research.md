# 011 实现与吞吐研究交接

查阅日期：2026-10-10（北京时间）。本文件是只读源码/已归档日志审计与一手资料调研；没有修改 `scripts/`，没有下载或加载模型权重，没有启动 GPU、训练、推理或竞赛提交。以下新增接口、性能改进和验收实验均**未实现/未运行**，除非单独注明。

固定条件：`Qwen/Qwen3.8-27B`、BF16、4×L4、单进程 `device_map="balanced"`，最终完整 Notebook 12 小时。本轮只优先两件事：**一次加载底座后顺序完成多个评价工作；在当前 HF 栈核实并补齐 GDN 快路径**。不同时迁移 TP、量化或推理服务框架。microbatch decode 暂留后项。

## 1. 先继承实际代码能力，而不是文件名

| 入口 | 已有能力 | 实际边界 |
| --- | --- | --- |
| `scripts/solve_grid.py:1419–1441`、`:1557`、`:1811` | 单次 CLI 调用加载一次底座/adapter，依次完成多个 query；base/G1 配对通过禁用 LoRA 共用底座 | 单次调用不是每 query 重载；重复运行 CLI、训练后另跑评价仍重新加载。`main()` 混合了参数、加载、循环与输出，没有现成的多 adapter 已加载模型入口 |
| `scripts/solve_grid.py:1090–1200` | `_run_one_candidate` 接受已加载 model/tokenizer，真实调用 generate，保留完整候选与解析结果 | 一次仅单样本；`generate_s` 包含模型 prefill+decode，`prepare_s` 只是 tokenizer/搬运；不是纯 decode tok/s |
| `scripts/train_grid_sft.py:3280`、`:3448`、`:2986` | `execute` 加载、建 LoRA 与 optimizer；`train_loop` 做真实训练；suffix logits 已有实现 | 未抽出共享会话训练入口。先完成推理共享；不要为省一次加载同时重写训练调度 |
| `scripts/run_budgeted_grid_experiment.py:214–262`、`:289–300` | CPU 阶段规划，若干模式切换辅助函数 | `evaluate_loaded_model` 只写 `not_run_no_gpu`，不生成；非 dry-run 退出。不是可运行共享会话编排器 |
| `scripts/ttt_per_task.py:298–309`、`:385–394` | CPU TTT 计划、LOO/mask 辅助逻辑 | GPU adapt/generate 主循环明确抛错，不能把已有文件视为可运行 TTT |

最新训练栈来自 `experiments/010/kaggle-s4-v1/010/s4-f0/run_config.json`：Python 3.12.13、torch 2.10.0+cu128、transformers 5.8.0、peft 0.19.1、accelerate 1.13.0。输入 cuda:0、lm_head cuda:3。后续记录实际 `hf_device_map`，不要只写 `balanced` 就假定完全没有 CPU/disk offload。

## 2. 归档日志更正与瓶颈实测缺项

010 进展后段对慢加载的归因不能原样继承。本次人工发起标准库 JSON 读取，直接核对原日志：

| 来源 | 原始记录 | 可支持的结论 |
| --- | --- | --- |
| `experiments/010/kaggle-e0-v1/e0-f0-pair-010.log:383`、`:477` | `model loaded in 312.7s`；`SOLVE_EXIT` 日志时刻约 5722s | 这份 E0 日志没有约 6600s 首载。不能用不存在于此日志的 stall 解释其账户配额差 |
| `experiments/010/kaggle-e0fill-v1/e0-fill-f1at0-010.log:443` | 首次 `model loaded in 6584.0s` | 确有一次很慢的加载，并吃尽该臂预算 |
| 同上 `:339–342` | 加载中持续输出 `Loading weights`，例如 t≈4711–4810s 仍有 73%→75% 进度 | 不是“零 stderr 静默 6600s”。冷存储读取可以作为假设，尚不能排除初始化、映射/搬运等因素 |
| 同一补缺会话后半段 | 第二次 `model loaded in 26.2s` | 同会话暖加载明显较快；不是“约 400s”。可减少重复底座加载，但不能承诺每次节省 6600s |
| S4 原运行日志，本轮根审阅另行核到 | `LOAD_DONE=3308.9s`；25 更新纯训练约37.6分钟 | 慢加载不止补缺会话一次；不能继承010末尾“S3/S4未出现”的笼统判断。S4该项由根研究记录归档，本子审计未另下载日志 |

两份日志均重复出现 GDN fallback 警告（E0 `:51`；补缺 `:47`）。**只有“至少一个快路径依赖未启用”的证据，没有逐算子占时或确认四个算子全在 fallback 的记录。**

目前缺的诊断很小但关键：

- 加载时：基座/adapter 各自时长、完整 `hf_device_map`、每卡 allocated/reserved、主机内存、日志心跳与阶段；I/O 与 GPU 搬运未拆，所以不写确定病因。
- 推理时：同步后的 prefill、首 token、固定长度续写与完整 query 的独立时长，实际输入/输出 token。现 `generate_s` 不能回答 GDN 主要拖慢 prefill 还是 recurrent decode。
- 只在固定短/中/长各一例采集很短的 GPU 利用率/算子概况，用来区分串行跨卡、GDN 与输出头占比；不建长期监控平台。
- 所有预算从 Notebook 初始化起算，分别报告安装/JIT、加载、计算、保存、账户配额差。账户差不当作单 Notebook 稳定扣额倍率。

## 3. 优先改进 A：共享底座，顺序执行多个冻结评价任务

这是减少重复初始化和加载的改进，**不会使四张卡变成四个并行 worker**。Accelerate 官方对这种朴素模型切分的说明是设备依次工作，不能把 4 卡容量与 4 倍吞吐混为一谈。[官方 big model inference](https://huggingface.co/docs/accelerate/main/en/concept_guides/big_model_inference#limits-and-further-development)，查阅 2026-10-10。

先以 F0-25 P12 待完成评价为任务 1；同一会话的后续诊断只在剩余时间足够时执行。已有 F1 结果不为“凑多任务”全部重跑。多个待评 adapter 可用同一 PEFT 包装注册唯一名称、逐个激活；PEFT 0.19.1 已有 `load_adapter`/`set_adapter`/`disable_adapter`，加载不自动激活，切换会影响 trainable 状态，应显式保持评价冻结。[PEFT 固定版本源码](https://raw.githubusercontent.com/huggingface/peft/v0.19.1/src/peft/peft_model.py)，查阅 2026-10-10。

### 拟实现接口（尚不存在）

```text
load_session(model_dir, fixed_runtime) -> {model, tokenizer, devices, identity}
activate_frozen_adapter(session, adapter_path_or_none, adapter_name) -> verified_identity
evaluate_loaded_queries(session, query_jobs, generation_config, deadline, journal) -> records
```

- 从 `solve_grid.main:1419–1444` 抽取唯一加载点；从 `:1583–1929` 抽取真实 query 流程，复用 `_run_one_candidate`。不能调用现在 runner 的假评价包装后声称已贯通。
- 一次只保留一个全量基座；不 `deepcopy`、不 merge/unmerge 不同 adapter。冻结 adapter 可保留少量具名副本，但要实测占用，缺显存时卸载不用的 adapter，不能靠盲目累积。
- 切换权重后丢弃所有旧 query 的 KV/GDN/conv 状态；保留磁盘上的任务记录不是跨权重复用 cache。现 `_drop_cached_decode_state:993` 可复用，但还要确保调用方不持有前次 `past_key_values`。
- 从训练切到评价时须 `disable_checkpointing`（train `:2090`）、`use_cache=True`、`eval()` 与 `inference_mode`。现在 runner `enter_eval_mode:171` 没调用 `disable_checkpointing`，不能直接当完整模式切换。
- 共享训练/TTT 留到第二步。多 adapter 时现 `freeze_non_lora:2106` 会把所有 `lora_` 都设为可训练，不适合直接用于“只训练当前臂”；需要显式选当前 adapter 的 optimizer 参数。TTT 还必须按 task 恢复相同初始 LoRA、重建 optimizer，不能把上一题权重延续到下一题。

### 手动验收与回退

在同一已加载会话取固定 2–3 个完整 query：顺序 A→B→A，核对 A 两次完整 token/grid、参数指纹、formatter、cache 隔离；base 禁用 adapter 与旧入口结果一致。记录加载次数必须为 1，分开报告加载节省和纯推理速度；后者可能没有改变。任何状态串扰都回退到原单 adapter CLI，保留已完成候选。只做小手动检查，不新增测试文件或 CI。

## 4. 优先改进 B：在固定 TF5.8 栈启用 GDN 快路径

官方固定权重 config 的 `model_type` 是 `qwen3_5`、text `qwen3_5_text`；64 层采用三层 linear/一层 full attention 的重复，conv kernel width=4。研究应核对 `qwen3_5` 实现，不因模型商品名搜索不存在的 `qwen3_8` 模块。[Qwen 固定 config](https://huggingface.co/Qwen/Qwen3.8-27B/blob/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0/config.json)，查阅 2026-10-10。

TF5.8 实际按模块成员分别选择 conv、chunk 和 recurrent 实现；聚合 warning 只表示四个函数并非全部可用。FLA 同时影响 gated RMSNorm 类，安装前后要记录它。新 forward/prefill 走 chunk，有旧 state 且 seq_len=1 走 recurrent；所以只跑一次短 forward 不够。[TF5.8 Qwen3.5 官方源码](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/models/qwen3_5/modeling_qwen3_5.py)，查阅 2026-10-10。

TF5.8 导入探测要求 CUDA、可见 `fla` 且版本至少 0.2.2；causal-conv1d 独立探测。[TF5.8 import_utils](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/utils/import_utils.py)，查阅 2026-10-10。当前 FLA 安装文档已采用 `fla-core` 与 `flash-linear-attention` 两包和 backend extras；不能照抄旧笔记无版本 `pip install -U`，也不能认为 FLA 自带 Triton conv 就自动满足 TF5.8 的 `causal_conv1d` 导入。[FLA 官方安装说明](https://raw.githubusercontent.com/fla-org/flash-linear-attention/main/INSTALL.md)，查阅 2026-10-10。causal-conv1d 官方支持 BF16、width 2/3/4，匹配本模型的 dtype/width，但该信息不保证本机 wheel ABI 和完整图兼容。[causal-conv1d 官方说明](https://raw.githubusercontent.com/Dao-AILab/causal-conv1d/main/README.md)，查阅 2026-10-10。

### 先离线准备（CPU 可做；本轮未执行准备产物）

1. 固定原 torch/TF/PEFT/accelerate 和模型/adapter，新增依赖只选一个可复现 FLA commit/release 与 causal-conv1d wheel 候选，保存完整版本、文件 SHA、许可与所需依赖。当前版本组合兼容性**未确认**；网页新版本不是项目已装版本。
2. 在与 Kaggle Linux x86_64、Python3.12、torch2.10.0/cu128 及 C++ ABI 对应的环境准备 wheel；本地 M5 只能下载、检查 metadata/源码，不能证明 CUDA 二进制兼容。没有匹配 wheel 时先写构建交接，不在正式 GPU 预算里无界源码编译。
3. 安装候选使用离线 wheelhouse 和约束文件，不升级已验证栈；依赖须在首次 import transformers/fla 前安装。完整保存原 wheelhouse/镜像回退入口。
4. 预选短/约2K/约4K 三条完整输入及固定后缀 token，不删 demo，不用 query 金标准作为候选筛选；冻结 tokenizer/template/formatter 和随机种子。F0/F1 不与本次核优化同时改变。

### 拟实现的短探针（尚不存在）

```text
inspect_gdn_runtime(model) -> per_layer_callable_names, norm_classes, versions
probe_gdn_backend(session, fixed_cases, backend, warmups, repeats) -> numeric_and_timing_report
```

每层记录 `causal_conv1d_fn`、`causal_conv1d_update`、`chunk_gated_delta_rule`、`recurrent_gated_delta_rule` 的 module/name 与 norm class，并记录 `hf_device_map`。不用“warning 消失”替代调用路径证据。必要的 baseline/fast 切换只在探针中保留原成员引用，恢复时逐层复原；不能仅把包级 flag 改掉就假定已构造的模型会重绑函数。

手动真机验收按下列顺序，预算事前封顶，失败即停：

1. **数值**：同短/2K/4K 输入比较 prefill 最后若干 logits、固定 token 后缀的 cached recurrent 连续多步输出、有限值与 cache shape。先用同路径重复测噪声，再判断 cross-path 差异；若梯度路径以后用于训练/TTT，再补相同答案 mask 下 loss 与非零 LoRA 梯度。只通过推理不能声称训练也可用。
2. **实际输出**：固定少量完整 greedy query 与原路径配对，看解析、完整网格、首次分歧位置。数值差异可能放大成生成分歧；有分歧时保留两份输出并检查任务 EM，不用“kernel 名正确”盖过行为变化。
3. **速度**：JIT/warmup 单列；warm 后每长度重复 3 次，分别计 prefill、固定128 token续写、完整 query、每卡峰值。固定 continuation 用同一预存 token，避免输出变短制造吞吐假象；完整 greedy 的真实输出长度另报。同步在计时边界覆盖所有相关 GPU。
4. **采纳**：先要求没有系统数值异常、没有 OOM/非法状态、完整候选无明确退步；再要求端到端收益覆盖冷启动/JIT成本。可预定“暖态典型 query 中位时间至少降10%”作为继续研究门槛，但不是保证，也不是金牌或比赛准确率证据。

若依赖导入/ABI/数值失败，用原已验证环境继续 F0-25 评价，不能让安装失败吞掉主要研究；若只 prefill 改善，按真实输入/输出分布算收益，不能拿长 prefill 倍数套在长 decode 上。内核改进不增加 GDN LoRA targets，两项需分开研究。

## 5. 为什么 microbatch decode 此轮不排前二

batch=2 可能提高每次模型权重扫描的有效工作量，但并不能改变 balanced 的串行设备布局。当前 solver 的 decode 只取 `sequences[0]`（`:1055`），一个候选只接一份 prompt；采样还临时替换进程级 `torch.multinomial`（`:1019`），不能拿多线程同时调用 generate 充当批处理。变长输入需左 padding、正确 attention/cache、逐样本 EOS/输出长度、独立 seed、失败单条回退和逐候选落盘，均未实现。待共享加载与 GDN 核归因完成后，再对相近长度的两个 greedy query做 batch1/2 小对照；本轮不展开该分支。

## 6. 在共享执行之前修正持久化语义

这些是省 GPU 预算所必需的局部改动，不是建设测试体系。

| 当前源码问题 | 影响 | 最小修正要求（均未实现） |
| --- | --- | --- |
| runner `should_skip_completed:208` 给 `query_dedup_key` 传 6 个位置参数；solver `:473` 只接 5 个 | 调用即 TypeError，现共享评价壳甚至不能可靠判重 | 统一一个候选/工作项身份 schema，调用签名一致，不靠吞异常继续 |
| runner `:210–211` 将 `load_completed_queries` 结果当 dict；solver `:478–506` 返回 `(done, interrupted)` | 即使前一个参数错修掉，membership 仍不会按 key 命中 | 显式解包；一次读取建立索引，增量更新，别每 query重读全文件 |
| solver `checkpoint_fingerprint:432–453` 实为路径+adapter配置；没有 adapter权重内容，base仅路径 | 同路径覆写权重后可能跳过旧结果，路径改变却可能重复做同权重 | adapter文件 SHA256+base不可变revision/数据源版本/已核对分片manifest；运行前算一次，别每query读54GB |
| `run_config_hash:456` 没含题面/模板/代码内容或 kernel backend身份 | 修改输入或执行实现仍可误复用旧行，性能对照被“跳过” | 加题面/模板/formatter版本、代码版本、kernel运行指纹；性能实验 run_id 单列，明确何时允许复用输出 |
| solver `:1814–1825` 完成候选后仅放内存，直到 `:1927/1929` 才写整 query | 第二臂执行时被外部杀掉会丢第一臂；有 partial budget 行时下一次会重跑整query | 每候选结束立即journal；key含checkpoint/task/query/view/candidate种类/sample_index/seed/config，query行由候选聚合 |
| 现 time guard 只在开始 query/candidate 前检查 `:1801` | 一个长候选可越过 deadline；不能称硬12h保护 | 预留保存窗口，以最近长度实测进行 admission；必要时用受支持 stopping criterion 中止当前生成并记 incomplete，不当模型错误 |

`append_query_jsonl:509` 每次 open/append/flush 是已有基础；文件尾损坏跳过逻辑也已有。后续只需确保已完成候选和错误/预算中断的状态可区分，保留旧记录、产生新身份；不要重写原007/010证据文件。普通格式错误是已完成模型结果，资源/中断错误可重试，但不能在汇总里默默丢失。

CPU 手动预演可用几个内存字典/临时 JSONL 核对“同身份跳过、换权重重算、第一候选已落盘只补第二、尾行中断不丢前行”的四个场景；不写新测试文件/目录/框架。真机先跑一个双臂 query，人工停止于臂间，回读确认续跑只补缺臂，再开展更贵评价。

## 7. 最小交接顺序

1. 先核对并取得待评价 F0-suffix-25 的真实权重与身份，完成已承诺的 P12 比较；不要让优化分支替代这个研究问题。
2. CPU 准备输入、tokenization、权重/代码manifest、候选journal语义；把真实 solver 抽成共享加载接口并做少量手动复核。
3. 一次4L4会话先完成主要评价；若预算可容纳，再测 A→B→A状态隔离和已离线准备的短 kernel探针。依赖失败立即回退，不现场升级一整套栈。
4. 实际报告拆开：加载次数/时长、计算速度、数值与完整答案、GPU预算、未完成项。任何单项通过都不替代 ARC2 解题收益与12小时完整提交验证。

本文件提出的共享调用接口、GDN探针、候选journal修正尚未实施；当前没有新增性能数字。

## 8. thinking/preserve_thinking 的本地 CPU 实证

2026-10-10 手动执行一次 tokenizer-only `AutoTokenizer.from_pretrained(local_path, local_files_only=True, trust_remote_code=False)` 和 `apply_chat_template`；本地 Python3.9.6、transformers4.57.6、jinja2 3.1.6、tokenizers0.22.2、`Qwen2TokenizerFast`，环境没有 PyTorch/TensorFlow/Flax，无模型可加载。它验证模板字符串行为，不是Kaggle TF5.8模型运行。缓存目录为 `/Users/zhaohuaxiang/.cache/qwen38-tokenizer`（项目原固定 revision `1d4bf0f2` 的缓存）；模板 SHA256 为 `c3cf9e34abf4f9e36c2d72165aa9c132d3e2a725b6c2586aaa3a8af9d7a81041`。未保存测试脚本，结果记录于本节。

模板真实分支见缓存 `chat_template.jinja:46–55`、`:110–119`、`:163–169`：

| 传给 apply_chat_template 的设置 | 本次实际渲染结果 |
| --- | --- |
| `enable_thinking=False` | 不插入effort指令，generation prompt含空的闭合think块 |
| 不传thinking/effort | 默认xhigh指令，generation prompt以开放think块结束 |
| `enable_thinking=True, reasoning_effort="low"` | 插入简短思考指令，开放think块 |
| `enable_thinking=True, reasoning_effort="medium"` | 开放think块，没有额外effort文案 |
| `enable_thinking=True, reasoning_effort="high"` | 明确报错；支持的只有xhigh/medium/low |
| `enable_thinking=False, reasoning_effort="high"` | effort分支不执行，仍是关闭think；不应因没报错而称high受支持 |
| `preserve_thinking`不传或True | 保留历史assistant的`reasoning_content` |
| `preserve_thinking=False` | 去掉最后真实user问题之前的assistant思考；该问题之后工具链里的assistant思考仍保留 |

最后两行用两轮user、两条带不同marker的assistant `reasoning_content`、一条tool反馈实际核对：False移除旧marker但保留当前marker；tool被渲染为tool_response。模板不自动替调用方从一次原始生成中提取/保存思考，调用方必须正确维护 assistant 的 `reasoning_content`、最终 `content` 和工具响应。

**以上参数属于模板渲染，不是可以原样塞进 HF `model.generate()` 的 API 参数。** `max_new_tokens` 控制生成总上限，low只是模板文案，不提供硬思考token配额。当前 `solve_grid.py:1614` 固定关闭think，现解析器要求整段输出是网格；打开think后必须另建明确的think/final分离逻辑，未闭合、没有最终答案或超预算要记incomplete，不能从思考里找“最后一个矩形”冒充答案。

有界程序支线可保持很小：固定同一底座、同一low模板、6个预冻结开发tasks，两臂共用首次程序提案；A第二次独立提案，B第二次读首提案、已知demo的CPU执行反馈并只修一次。两臂同为最多2次调用、每次总输出上限（例如768，含思考与程序）、相同墙钟截止；必须完整输出代码才执行，无query答案反馈；统一按demo一致性选最多两个最终query网格。B保留显式任务历史/思考，A从同原题重新开始，此实验衡量“历史＋执行反馈”整体价值，不单独归因于preserve开关。先设支线总预算，硬上限覆盖加载、两臂、执行和保存；不为它先训练或扩通用agent框架。入口、解析、限时执行与该A/B均**未实现/未运行**，先完成主线F0-25评价。

## 9. 只保留一个可验证加载策略候选

固定TF5.8已经默认异步物化权重：`modeling_utils.py`调用 `core_model_loading.convert_and_load_state_dict_in_model`；后者 `GLOBAL_WORKERS=min(4, os.cpu_count() or 4)`，环境变量 `HF_DEACTIVATE_ASYNC_LOAD` 为真，或disk offload/即时量化时才走同步。两个文件未找到旧 `HF_ENABLE_PARALLEL_LOADING`，不把旧教程开关抄进本项目。[TF5.8 modeling_utils](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/modeling_utils.py)、[TF5.8 core_model_loading](https://raw.githubusercontent.com/huggingface/transformers/v5.8.0/src/transformers/core_model_loading.py)，查阅2026-10-10。

**唯一候选是受预算保护的一次同步加载**：在下一次本来就需要的会话、首次from_pretrained之前设置 `HF_DEACTIVATE_ASYNC_LOAD=1`，其他版本/权重/映射不改。假说是避免某些挂载存储或物化/搬运的并发争用，不是已证明会更快。正式采用前先记录运行时两个源文件位置/SHA、版本、该环境变量和实际device_map；仅版本字符串一致不保证所运行源码与网页完全相同。

不专开多个27B加载作A/B，不为预热再完整读一遍权重，不预复制54GB到容量未核实的工作目录。事前从总预算分出加载上限，例如20分钟；由父进程监看单个完整工作进程的加载完成标记与心跳，加载成功后该进程继续所有任务并共享模型。截止前未完成则保留日志、终止该工作进程并结束本次尝试，不能在同一计费会话反复from_pretrained重试。该父进程截止机制**未实现**，当前solve/train的软time guard不能中断from_pretrained。

一次成功或失败只能更新可行性与预算假设；与旧日志的冷/暖环境并不配对，不能把时间差写成加载开关的严格因果收益。失败回退是在下一次经预算安排的会话取消该变量、恢复原默认异步加载，同时继续坚持一次基座加载，不把现有可用路径全部替换。该候选实验**未运行**。
