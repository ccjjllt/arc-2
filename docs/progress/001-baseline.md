# 001 — 比赛、算力与已有基线：实际进展

日期：2026-09-29。对应计划：`docs/plan/001-baseline.md`。

状态：完成官方信息与算力调研、研究目录和协作约定；已直连 Kaggle MCP，但实际数据读取未通过认证，基线代码与成绩尚未独立核实。

## 实际完成的修改与产物

开始时当前项目目录为空。本轮新增：

- `AGENTS.md`：保存研究优先、不写测试文件、所有测试手动进行的协作约定。
- `README.md`：比赛、基线和研究文档入口。
- `docs/plan/001-baseline.md`：本轮调研及基线交接计划。
- `docs/progress/001-baseline.md`：本文件，记录已确认事实与未完成项。
- `docs/plan/002-strategy-improvement.md`：后续推理预算、候选选择、定向微调与互补解题器实验设计。

未新增训练、推理或测试代码，未安装训练依赖，未启动 GPU 作业，未上传、修改或提交 Kaggle Notebook。

## 基线核实状态

| 项目 | 已知情况与证据性质 |
| --- | --- |
| 用户目标 | 争取比赛金牌 |
| 起始分数 | 用户自述约 30 分；精确值、名次、对应 submission 尚未读取 |
| Notebook | 用户提供 `ccjjllt/reproduce-nvarc-2025-results` |
| 指定版本 | URL 中的 `scriptVersionId=314617319` |
| 模型与组件 | 尚未读取指定版本代码，不能确认权重、TTT、TRM、PoE 是否实际启用 |
| 日志与资源 | 未取得该版本的运行日志、实际硬件和阶段耗时 |

[用户指定版本](https://www.kaggle.com/code/ccjjllt/reproduce-nvarc-2025-results?scriptVersionId=314617319)。不要用上游实现或 latest 替代该版本。

## Kaggle MCP 的实际检查

- 本机 Codex 与 OpenCode 配置均指向 `https://www.kaggle.com/mcp`，认证引用 `KAGGLE_MCP_TOKEN`。
- 当前聊天原生工具列表没有暴露 Kaggle 工具；随后已直接通过 MCP 协议连接现有服务，`initialize` 和 `tools/list` 成功。
- 实际调用 `get_notebook_info` 返回 `Permission 'kernels.get' was denied`。
- 实际调用 `get_competition_leaderboard` 和 `search_competition_submissions` 返回 `Unauthenticated`。
- 当前执行进程未取得配置引用的认证变量。因此只能确认端点与工具发现可用，不能确认已获得用户数据访问权限。
- MCP 查询参数 `versionLabelNullable` 不应直接当作网页 URL 的 `scriptVersionId`；成功取得元数据后，需核对实际版本 ID。
- 没有显示或保存凭据，没有修改任何认证或权限配置。后续应让会话取得现有有效认证，再重新执行只读工具调用；无需把密钥写入项目文档。

这个问题不影响公开规则调研，但使本轮无法完成具体 Notebook 审阅及账号配额核实。

## 比赛概况：已查证事实

ARC-AGI-2 是静态网格推理：根据少量输入/输出示例归纳规则，预测新的输入对应的完整输出网格。每个 test input 提交两个候选；任一个与真值完全一致才得分，最终对 test outputs 的命中情况汇总。局部像素正确不等于解对。[官方赛道说明](https://arcprize.org/competitions/2026/arc-agi-2)

| 项目 | 2026 页面信息 |
| --- | --- |
| 开赛 | 2026-03-25 |
| 报名、合队截止 | 2026-10-26 |
| 最终提交截止 | 2026-11-02 23:59 UTC，即北京时间 11-03 07:59 |
| 获奖者公布 | 2026-12-04 |
| 本赛道总奖池 | 700,000 美元 |
| 代码提交 | Kaggle Notebook，CPU/GPU 运行上限 12 小时，关闭联网，生成 `submission.json` |
| 外部资源 | 允许符合规则的公开外部数据及预训练模型；离线环境需事先准备模型和依赖 |
| 提交数量 | 每日最多 1 次，最终最多选择 2 次 submission |
| 队伍人数 | 最多 5 人 |

来源：[Kaggle Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/description)、[2026 Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)。赛程若更新，以官方页面为准。

评测时不能调用联网大模型 API。赛前可以在其他算力平台进行允许的训练，再把可复现、可离线加载的产物带回 Kaggle；不能把 Colab TPU 当作提交 Notebook 的在线外部算力。[比赛要求](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/description)

## 金牌目标与榜单口径

金牌依最终排名分配，没有固定分数门槛。按 Kaggle 竞赛奖牌规则，参赛队伍至少 250 时，金牌名额为 `10 + floor(队伍数 × 0.002)`；例如 2,000 队对应前 14 名。85% 是特殊奖金相关条件，不是金牌条件。[Kaggle 奖牌规则](https://www.kaggle.com/progression/competitions)

本轮能读取的官方榜单搜索缓存标注为约两周前：1,942 队，第 13 名 33.47 显示金牌，第 14 名同分但显示银牌。它只能说明历史分界，不能代表 9 月 29 日实时金牌线。同分也不保证同名次。最终根据私榜决定，不应按“30 再加 3.47 就能金牌”安排研究。[榜单来源](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)

2025 NVARC 的 24.03% 是当届 private eval 结果，不能直接与用户 2026 公榜约 30 分比较。[2025 官方结果](https://arcprize.org/blog/arc-prize-2025-results-analysis)

## L4×4 能否微调

可以，建议作为当前主力。2026 官方升级说明允许关联本赛的开发 Notebook 使用 L4×4，并要求所有此类会话关闭联网。公布的配额消耗是普通 T4×2/P100 的两倍，且保留调整倍率的权利。具体账户每周额度尚未读取，不能假定“无限免费训练”或固定可用小时数。[升级说明](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview/upgraded-accelerators)、[Kaggle Staff 公告](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/discussion/689054)

每张 L4 为 24GB，四张卡物理显存合计 96GB，但不是一张 96GB 卡。普通数据并行复制模型；分片训练需要额外支持和通信。模型可单卡容纳时，ARC 逐题 TTT 可考虑四个独立 worker。[NVIDIA 规格](https://www.nvidia.com/en-us/data-center/l4/)、[PyTorch FSDP 说明](https://docs.pytorch.org/tutorials/intermediate/FSDP1_tutorial.html)

建议先复用 ARC 专用权重，手动研究 LoRA/TTT、候选生成与排序。离线继续 SFT 也可研究，但模型规模、长度、batch、优化器和额度决定可行范围；不能仅凭权重大小判断训练能否放入显存。

参考成本很重要：NVARC 作者报告 Qwen3-4B 完整 SFT 使用 32 张 H100 训练 27 小时；ARChitects 报告其离线 LoRA 用 8 张 H100 训练 39 小时，同时提交阶段在 L4 上做每题 128 步 rank-32 LoRA。这支持“L4 可用于适配”，不支持“L4 能轻易重做冠军全套离线训练”。[NVARC 作者方案](https://www.kaggle.com/competitions/arc-prize-2025/writeups/nvarc)、[ARChitects 技术报告](https://lambdalabsml.github.io/ARC2025_Solution_by_the_ARChitects/)

## Colab v5e-1 与本地电脑的定位

| 资源 | 本轮判断 | 建议用途 |
| --- | --- | --- |
| M5 / 32GB | 可以开发和做轻量实验；未进行速度或显存实测 | 数据准备、分析、可视化、CPU 搜索、小样本手动运行 |
| Kaggle L4×4 | CUDA 生态和最终提交环境一致，优先 | 现有基线、逐题适配、可控规模的继续训练、完整手动运行 |
| Colab v5e-1 | 单芯片档，v5e 每芯片 16GB HBM，并非八芯片 128GB | 有现成 TPU 实现时的小模型/LoRA 备选研究 |

来源：[Google v5e 规格](https://docs.cloud.google.com/tpu/docs/v5e)、[Google Colab 设备文档](https://github.com/googlecolab/google-colab-cli/blob/main/docs/01_session_management.md)。Cloud TPU 文档中的整台主机资源不能直接套用到 Colab 单设备会话。

技术判断：基于 CUDA、Unsloth、FlashAttention 或特定量化内核的代码不能原样迁移到 TPU，需要适配 JAX/PyTorch XLA 及对应算子。当前优先研究已有方案，所以暂不需要为了这条路线配置 Colab MCP。若后续选定明确的 TPU 实验，再处理接入。

## 研究效果：本轮没有新增结果

| 项目 | 状态 |
| --- | --- |
| 指定版本复现 | 未运行 |
| 新训练/微调 | 未运行 |
| 新提交 | 无 |
| 提分结果 | 无 |
| 本地 vs L4 vs TPU 性能对比 | 未实测；上述是规格与实现适配分析 |

已手动阅读官方材料和 MCP 响应，并检查文档内容与配对关系。没有运行自动化测试，也没有创建测试文件。

## 下一轮需要继承

1. 首先恢复现有 MCP 认证的会话可见性，读取精确版本；继续沿用可用基线。
2. 先查正确答案是“生成不了”“选不出来”还是“没时间算”，再决定是否继续微调。
3. 上游 NVARC 的训练配方含 `arc2_evaluation6`。未经训练覆盖核对，不能把 public evaluation 当作未见验证集。[上游配方](https://github.com/1ytic/NVARC/tree/main/ARChitects)
4. 开源与可复现是获奖相关要求。Kaggle Rules 与 ARC Prize 总站的许可措辞、85% bonus 分配说明存在差异，临近成果发布时对照最新官方规则或主办方说明处理，不在本轮自行作统一的许可结论。[Kaggle Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)、[ARC Prize 总则](https://arcprize.org/competitions/2026)
5. 按 `002-strategy-improvement.md` 手动开展实验，在同名 progress 文件中报告真正完成的实现与结果。

所有网络资料查阅日期为 2026-09-29。部分 Kaggle 页面直接抓取为空，本轮使用其官方页面的搜索索引正文；动态排名和账号数据不作实时已核实处理。
