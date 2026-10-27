# 030 赛前一周研究审计与搜索覆盖：进展

日期：2026-10-27，北京时间。计划见[030计划](../plan/030-final-week-search-coverage.md)。

**最新状态：用户指定zhuating token及Kaggle CLI后，机制V1已启动／RUNNING，实际4×L4、固定权重复制和rank0加载正常；尚无终态或新分数。** 此前0余额实属默认CLI账号qingyu1232，zhuating启动前可用6.62小时。执行入口见[030 README](../../experiments/030/README.md)，精确产物见[ready-v1.json](../../experiments/030/ready-v1.json)。

## 已完成的工作

- 三个新的只读子代理分别核验历史原始工件、实时比赛／Notebook信息、作者开源与论文；主代理复核025—029材料、原V1的剪枝／batch／重评分实现和论文Table 3。
- 直接访问Kaggle官方页面，取消公开代码列表不可访问数据源排除；目前可见最高仍是Failed in AIMO33.89，分数绑定精确V1／312063593。没有证据证明存在可复现的更高公开Notebook。
- 今日公榜第16名69.44带金牌标记，第8名75.14；最终私榜未知。不能把33.89当作临近金牌的分数。
- 只读CLI／SDK确认当前配置身份普通GPU余额0，下次刷新10月31日08:00北京时间；无新GPU、训练、正式提交、自动监控或云资源支出。
- 独立复核已存query ledger及events：025首槽14、仅第二槽1；17个oracle正确中14个首见第1批、3个首见第3批。此为既有运行工件的CPU审计，不是新模型结果。
- 建立030研究设计：原sft139／TTT／B4／runtime，只比较DFS累计路径门槛0.2与0.1；预算计入后续视角、query和重评分。首轮与扩展均需最终两槽净增，失败不扫阈值。

## 来源与实查记录

### 比赛与公开Notebook

2026-10-27约10:00—10:06北京时间的浏览器页面读取：

- [Code列表](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?competitionId=133469&sortBy=scoreDescending&excludeNonAccessedDatasources=false)：第一Failed in AIMO33.89；Learned from AIMO、Minimal Perfpatch、twin-lb33-verbatim、Reproduce NVARC、NVARC+等32.22。
- [精确V1](https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593)：Version1 of2，33.89与V1绑定、sft139 BF16 V1、Apache-2.0、普通26m11s。
- [Leaderboard](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/leaderboard)：Tufa Labs88.06；第8 Dawid Kopiczko75.14；第16 Lashen Learning69.44／gold；第17 ảc69.17／silver。只记今日快照，不预测最终名次。
- [Overview](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview)、[Rules](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/rules)、[Data](https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data)：11月3日07:59北京时间最终截止；每日1份、最终最多2份；隐藏240task；每个test output取两槽最佳exact后微平均。网页说明不是服务端代码审计。

### 额度

现成认证的Kaggle CLI／SDK只读结果摘录，**本节是工具结果转录，不是新保存的原始响应文件**：

```json
{
  "checked_at_utc": "2026-10-27T02:05:35.559478+00:00",
  "quota_refresh_time": "2026-10-31T00:00:00",
  "gpu_used_seconds": 120810.666,
  "gpu_reserved_seconds": 0.0,
  "gpu_total_seconds": 108000.0,
  "is_pay_to_scale_enabled": false
}
```

剩余按`max(0,total-used)`为0。只代表当前配置账号；未查询其他账号。TPU仍有额度，但没有为此移植当前NVARC运行时。普通会话额度与正式隐藏rerun的扣减关系本轮未核验。今日提交列表最新仍为027／57634436 COMPLETE29.72、023／57582282 COMPLETE30.14；未见pending。正式可提交次数需临操作重新读取。

### 历史结果与新审计结论

| 轮次 | 已运行结果 | 对本轮的约束 |
| --- | --- | --- |
| 023 | 正式30.14 | 仍保留；没有复现作者33.89 |
| 025 | B4=R2=15/58，raw=scored oracle17/58 | 排序最多补当前池2个；41个失败池外，但其原因不能直接归为缺训练 |
| 026 | base=challenger=4/23、增2丢2，oracle6→9 | 结束配方；oracle增加不是晋级依据 |
| 027 | C23固定融合5→6/34，正式29.72 | 结束方案；C23缺base第二槽独有正确，且隐藏extra覆盖未知 |
| 028／029 | 候选变化与数值定位；未测B4／准确率 | runtime支线结束 |

压力样本`2ba387bc_0`及晚批首次正确`d35bdbdc_1`、`7ed72f31_1`、`7b0280bc_0`来自025已存gold指纹与候选／事件匹配，不是对新方案的评分。025的40题＋027的23题＋历史排除57题覆盖当前120题；只说明没有当前账本可认证的新保留池，不能声称120题全部读过答案。

### 研究来源

本轮直接阅读[Product of Experts v2 Table3](https://arxiv.org/html/2505.07859v2#S5.T3)、[MARC原论文](https://openreview.net/pdf?id=asgBo3FNdg)及作者实现、[2026-10-09预算研究](https://arxiv.org/abs/2610.13660)、[2026-10-23候选接口研究](https://arxiv.org/abs/2610.32235)。详细适用边界见030计划§4。外部效果没有转写为本项目效果。

## 当前状态与接续

**完成：研究审计、CPU实现、面板与预算冻结、实际包生成及独立审查。未完成：GPU机制／配对、扩展实际评价、正式挑战包及新分数。** 没有新增测试文件、框架或CI，也没有创建后续自动任务。

接续先临GPU执行读取实际额度；现有条件下把10月31日刷新后的窗口用于单因素配对。每个实际实验仍须记录数据版本／划分、权重／代码SHA、配置、硬件、耗时、两槽与oracle指标及产物。若无法在预算内给出正向证据，保留023，不为赶时间盲投。

## 用户授权后的CPU执行

用户要求开始并重新派代理。本轮新建`r030_source_audit_new`、`r030_panels_new`、`r030_budget_review_new`，分别承担来源审计、面板／冻结计分脚本、预算／机制包；没有沿用上一轮代理。主代理实现完整配对包装并整合审查修复。

旧`/private/tmp/arc2-*`全部缺失。只读下载023已完成输出中的3253字节`copy_model.py`，内容SHA与历史`5d51c4e...`一致，再用仓库生成器重建023；Notebook SHA精确复现`0bdde6503f6ec616be467321649e9abb997288f07aa54e8b4302f76f32f3d119`。没有为恢复而启动Notebook。

输入已冻结：初轮40task／58query，challenge SHA`38df4c7126f160f9dd045b4b0c9fb37ccb4f24dfbb51c8d32aa7b50f132ebeb9`；扩展40task／57query，SHA`6c76e81dfcbf00a0efb1f16d0da67a0240bc91f961d3ba8c540f3ead74e6db66`。扩展从剩余80题按input-only长度三档×单多query比例确定性选择，两面板task交集0、包含11个已消费C23 task；没有读取solutions或预测挑选扩展。

| 研究包 | 本地有效包／Notebook SHA | 已冻结范围 |
| --- | --- | --- |
| 同adapter机制 | `/private/tmp/arc2-030-mechanism-v1`；`2093702f934709070efe7bfcaa9a21350d4d84abf6d640d93b645e14c99ea14e` | 3task／5query，第1批原DFS A1/A2=0.2、B=0.1，共15调用；≤3600秒，不算B4／准确率 |
| 初轮完整配对 | `/private/tmp/arc2-030-pair-initial-v3`；`8a6361e5ff76011ab1af63c736ecd0c014336a46c296c9794b9e142997d8966c` | 40／58，base→challenger，每臂6600秒部署槽、总会话≤14400秒 |
| 扩展完整配对 | `/private/tmp/arc2-030-pair-extension-v3`；`c26c3d36e21860c262e18f075d635adbb06fd7c968df68e235530af69b8d43c2` | 40／57，反转challenger→base，预算与初轮相同；只在前置通过后运行 |

`v3`仅为本地CPU准备修订，尚无任何030远程版本。旧扩展草稿较长的计算预算已废弃，随后补齐最终B4导出成本。当前以[ready-v1.json](../../experiments/030/ready-v1.json)列出的SHA为准。

完整pair顺序独占4×L4、独立TTT，固定hash27030，唯一求解参数差异为累计概率门槛。每臂加载、TTT、DFS、重评分、B4导出均计入6600秒；公共准备／final imports向每臂全额计费，总账超时明确失败。原540秒DFS、1200秒task软检查及四batch布局保持。研究包有rerun拒绝逻辑，不能比赛提交。

新增六个手动脚本：准备面板、机制包、完整配对、只读preflight、无gold冻结、显式官方快照评分。冻结脚本重放原B4并保留完整分母、raw/scored候选与worker事件；评分先验证工件SHA，噪声证据缺失或A/A失败保持hold。没有测试套件、CI、自动重试或预约。

手动核查：新代理逐字确认原DFS／推理／重评分函数与loader／decoder保持，实际包AST、输入无答案、SHA、预算和分母通过[独立静态审查](../../experiments/030/pair-static-review-v1.json)。主代理用025已有真实候选复核新reader：58query B4完全复现，无未开始query、55query完成四批，阶段区间闭合；见[CPU核查](../../experiments/030/execution/manual-cpu-reader-check-v1.json)。这不是030模型运行或新评分。

11:02:21北京时间保存[实际SDK preflight](../../experiments/030/execution/preflight-20261027.json)：GPU used120810.666／total108000秒、reserved0、remaining0；10月31日08:00刷新。同队num_today0、num_allowed_now1。普通GPU与正式隐藏rerun扣费关系未核验。本机CPU完成打包与历史工件审阅；GPU实际硬件／耗时、adapter指纹执行、A/A结果、EM@2和oracle均为未运行／未确认。

CPU准备时的预算为机制1＋初轮4＋扩展4＋普通验证预留1＝最多10墙钟小时，按系数2计划20配额小时、预留10。该阶段尚未启动GPU。随后已按下节实际账号额度修订执行；没有正式提交或自动监控。

## 指定zhuating凭证后的实际启动

用户补充可以使用zhuating配额、必须用Kaggle CLI，token在`.kaggle/`。先对默认认证打印用户名，确认之前余额0属于**qingyu1232**；随后仅读取用户指定目录中的`access_token_zhuating`，在子进程覆盖`KAGGLE_API_TOKEN`，CLI `config view`明确返回**zhuating**。没有输出／写入token或修改全局默认认证。

本次另新派`r030_zhuating_launch_schema`及`r030_mechanism_prelaunch_new`审查CLI操作与实际机制／pilot包，未沿用上一批代理。

zhuating的CLI quota显示GPU used23.38h、remaining6.62h、total30h，刷新仍10月31日08:00；这是CLI四舍五入读数，没有reserved字段。[实查记录](../../experiments/030/execution/zhuating-cli-quota-20261027.json)。不得把默认账号qingyu1232的0余额写成zhuating额度，也不把两个账号余额相加。

通过CLI `kernels push -p /private/tmp/arc2-030-mechanism-v1 -t 3600 --accelerator NvidiaL4`成功创建[机制Notebook](https://www.kaggle.com/code/zhuating/arc2-030-threshold-mechanism) **V1**，随后CLI status为RUNNING。回读全部cell源码与冻结SHA`2093702f934709070efe7bfcaa9a21350d4d84abf6d640d93b645e14c99ea14e`对应内容逐字一致；断网及三类数据源正确。CLI未返回实际scriptVersionId，仍为未确认，不以kernel版本号代替它。

一次最长40秒的CLI `logs --follow`取得真实启动证据：**11:19:55北京时间**开始执行，4张NVIDIA L4各23034MiB，Python3.11.13；原Unsloth2025.9.7／Transformers4.55.4／Torch2.8.0+cu128。完整7267271302字节权重复制耗时67.925秒，config及两片SHA与023相同；rank0已完成两片加载。该日志范围未观察Traceback。其他rank与TTT／15调用完整性尚未确认，不能把加载正常写成机制通过。[启动记录](../../experiments/030/execution/mechanism-v1-launch.json)、[实际日志](../../experiments/030/execution/mechanism-v1-startup.log)。普通`kernels logs`此前两次为空，不据此判错；后续不继续空轮询。

为在当前额度内取得先行结果，在读取任何030预测／新gold前冻结pilot20：从初轮40题按原长度tier×单多query比例、固定seed `030:pilot20:quota20261027`确定性选20题／28query，保留全部query，未强制纳入压力题。challenge SHA`94d947c6d3da5f1080700de205bb2a0791764a65b79cb2e2b181f5f91bf58f12`，包SHA`60a5454bbe50beafcf49a959d2840df6d0c321f0e8d766810c8aea0bce38cc63`；总上限7200秒、每臂3300秒，core源码与initial-v3一致。[新代理静态审查](../../experiments/030/pilot20-static-review-v1.json)无阻塞。机制终态通过、退出及重读实际余额后才能启动，当前未上传该pilot。

当前先行预算机制≤1h＋pilot≤2h，按系数2最多6配额小时。pilot仍要求净增>0且至少两task贡献新增，不选其他子集或阈值救结果；但它未含2ba，不能声称覆盖已知第二槽独有正确风险，正式晋级仍需原40题压力／覆盖复核及扩展。原initial40与extension40包保留、没有重复启动或假定当前余额可支持它们。没有正式提交或自动任务；按用户既有偏好，确认健康后停止远端查询，等待用户通知结束。
