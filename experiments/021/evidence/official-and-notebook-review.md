# 021 官方比赛与公开 Notebook 核对

查阅日期：2026-10-21（Asia/Shanghai）。本轮通过网页搜索、Kaggle 实时浏览器页面和本地源审读核对；未运行 GPU、未提交、未修改项目工程。

## 官方条件（本轮已实时读取 Kaggle 页面正文）

- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/overview ：最终提交截止 2026-11-02 23:59 UTC，即北京时间 2026-11-03 07:59；入场与合队截止 2026-10-26 23:59 UTC；GPU/CPU 最多12小时；断网；允许公开可获取的外部数据与预训练模型；产物 submission.json。L4×4 总96GB；配额消耗为旧 T4×2/P100 的2倍，官方保留调整权。
- 评分按每个 task test output 的两槽 exact-match 中任一命中记1，再按全部 test outputs 平均；应以 query 微平均作主要指标。不能直接继承旧 NVARC 本地 task 加权评分。
- https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/data ：当前正文明确 rerun 使用240个 unseen tasks，部分含两个 query。隐藏 query 总数未给出，必须动态读取。公开 test 文件只是占位，不能据其内容断定真实隐藏分布。
- https://arcprize.org/competitions/2026 ：同样确认11月2日截止、断网与开源要求。

## 可绑定精确版本的4B起点

- https://www.kaggle.com/code/koushikrudra/failed-in-aimo?scriptVersionId=312063593
- 实时页面明确：33.89 / V1；Version1 of2；L4×4；qwen3_4b_grids15_sft139 BF16 V1；Apache2.0；显示普通保存运行26m11s。
- 该普通运行只执行4个公开题，源码末段白名单0934a4d8、36a08778、981571dc、aa4ec2a5，不能将26m11s外推为隐藏240题总耗时。
- 从明确V1页点击下载所得：/Users/zhaohuaxiang/Downloads/failed-in-aimo.ipynb，63719 bytes，9cells，SHA256 71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d。
- 代码提取 experiments/021/reference/failed-in-aimo-v1.source.py；只供源码审读，未经运行。
- 原V1含4B逐题重置LoRA、训练128增强序列、推理16视角、batched DFS、逆变换去重、频次减增强平均NLL排序；rank256 alpha32、lr5e-5、1epoch、8192上下文、4独立GPU worker、全局12h-600s、每题1200s。
- 完整4B流程指该4B分支；不能等同2025获胜方案完整ensemble，官方2025分析将获胜方案描述为TTT模型与TRM组件组合。
- 重要陷阱：本地 experiments/014/public-sources/koushikrudra-failed-in-aimo.metadata.json 是V2，source.py含额外Qwen2.5-Coder7B LegC。不能用这份V2源证明V1/33.89实现。
- V1 validate_submission 使用 score += 1 / len(v)，返回task加权计数，与本届官网query平均不同；交接评分应独立按官方定义计算。
- V1评分增强种子仍为hash(bk)，复现实验固定PYTHONHASHSEED可减少进程随机性；若改为稳定hash须记为单项改动，不伪称字节级原版。

## 高分公开Notebook检索的实际边界

本轮实时进入官方比赛Code页，选择Public Score排序，页面URL：
https://www.kaggle.com/competitions/arc-prize-2026-arc-agi-2/code?competitionId=133469&sortBy=scoreDescending&excludeNonAccessedDatasources=true

此当前可见列表第一条仍为Failed in AIMO33.89；随后Learned from AIMO32.22、mikelou1 Minimal Perfpatch32.22、Luxshan NVARC+32.22、Dale Zhong NVARC Plus Ckpt31.94、Yusuke Baseline Rebuild31.81。仅核验列表分数，未对这些fork逐一绑定最佳源码版本，不能称它们具有独立复现增分。默认带excludeNonAccessedDatasources筛选，故不声称全网最高。

Hotness页题为Triple-Pass NVARC+ SOTA(35.9% LB)的mayankraaz条目实际显示31.39，另同题名sushanthtiruvaipati为26.53；标题分数不是该源码实测成绩。27B公开Notebook讨论仍缺完整12小时高分可复现源码，已归档的35→45是作者自报，不能按已验证10分增益写方案。

2025官方回顾 https://arcprize.org/blog/arc-prize-2025-results-analysis 可支撑：NVARC私榜24.03，ARChitects16.53，MindsAI12.64。这些是2025私榜，不能与2026公榜33.89或本项目12.50直接求增益。

## 对交接的建议

1. 三臂比较完整预算4B、让出预算4B、同总预算4B+27B，以隔离压缩4B损失与27B互补收益。
2. 先冻结V1源码作为控制，正式按query评分并保存真实两槽；oracle并集仅诊断选择损失。
3. 总隐藏运行240任务预算需要包含安装、4B加载/TTT、27B加载切换、推理、落盘，不能把普通4题保存运行时当整轮推理预算。
4. 更高分团队方法未公开绑定，不能从榜分推断其模型或训练方法；不要以金牌图标（Notebook投票奖章）证明比赛金牌。

## 证据类型与参数补充

官方/data、/overview、Failed in AIMO V1分数版本与Code排序列表均来自本轮实时浏览器AX正文，不是搜索缓存。最初web检索的比赛/leaderboard内容有一个月缓存，只用于定位，不用作当前排名；本地013快照是2026-10-12历史官方API归档。ARC Prize 2026与2025回顾来自本轮web open的一手页面。

V1提取代码SHA256：b129bd6c316ee611722a1c75b4e757b277bd4332efb348bf7561de22462505cd。源码的1200秒是外层解码循环检查，DFS内部另有540秒/全局截止判断，不等于可抢占的严格逐题1200秒硬上限。正式预算分析以实际wall time为准。
