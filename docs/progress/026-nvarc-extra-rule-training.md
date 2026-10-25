# 026 NVARC额外训练规则覆盖挑战者进展

日期：2026-10-25。对应[026计划](../plan/026-nvarc-extra-rule-training.md)。研究V1已COMPLETE并评分：base与challenger均4/23，新增2／丢失2、净增0；候选oracle6→9。按预定停止条件结束本续训配方，保留原权重，未实现最终两槽提分。

## 已完成

用户授权继续在NVARC上提分。重新派遣数据、训练格式、独立研究审计及配对包装代理；未复用历史代理。独立审计发现原重评分使用`hash(bk)`而025未设置`PYTHONHASHSEED`，因此本轮改用同一次运行的新base／challenger配对，不能直接和旧15/58做训练因果比较。

[冻结开发面板](../../experiments/026/panel-v1/manifest.json)包含16task／23query，输入SHA `bd3b26cf78c4986d2cce6d72b457ce763f845e0e2f66d91233fc00a2a9f059f0`。固定seed与预定边际约束选择，历史025为4全对／1部分对／11全错；没有读取query答案网格、生成器规则或新模型预测选题。另保留23题的一次确认身份。

07:11:48 UTC账号`zhuating`配额读取可用64286.086秒，约17.86配额小时，reserved0；这是账号快照，不是本次实验成本。见[account-start](../../experiments/026/execution/account-start.json)。拟定4×L4合并实验最多约3小时，按2倍额度约6配额小时；实际硬件、运行时间与消耗仍未确认。

数据生成和训练脚本的手动核查结果、最终上传身份和启动事实将在实际完成后追加。未创建新的正式比赛submission；当前最好仍为023正式公榜30.14。

## 数据冻结与CPU核查

固定生成器commit `d9ea8b2fc21ac4189f1a895aaeef84cf74df16b0`，109族均可生成并通过自身verifier，但与官方training demo核对335/338一致；`17829a00`、`b74ca5d1`、`f18ec8cc`各有不一致，整族隔离，不据此宣称官方错误。合格106族分90训练／16族隔离；隔离族也从官方training回放排除。16可读pilot族另用独立stdlib规则重建核对256/256 pair一致；扩展106族并未全做独立语义证明，保留弱核查标签和任意cue限制。

[实际嵌入数据](../../experiments/026/data-embedded-v1/manifest.json)为1024 episode：90合成族共512条（每族5或6条，族内长度四分位均衡），官方training 512个祖先族各1条，按全长四分位各128条。训练JSONL SHA `33f3b4e6133be5b890b8f2efb76421390a463b04291f5669f6b5e44bcbd5f901`，4870801字节、gzip430154字节。唯一全长超8192的官方池行在选择前显式排除，不删demo或截答案。training／holdout祖先、训练族／评价面板交集均为空。模型上游历史接触仍未排除，不能称未见规则。

小型真实tokenizer文件已从固定Kaggle模型V1逐文件下载，均与025模型复制记录SHA一致，见[tokenizer来源](../../experiments/026/execution/tokenizer-source.json)。此前27B模板未复用。训练脚本单独核对query-only监督、header、EOS、prompt前缀和全序列。离线SFT对最长7464 token样例启用Unsloth梯度检查点，减少L4激活显存压力；原逐题TTT仍保持原配置。

调度最外层严格交替两种来源，来源内轮转family×长度桶。因此任意完整accum4步骤均为2合成＋2官方回放，避免训练时间提前到达后来源失衡。每个episode最多走一遍，不重复小pilot凑满步数。运行时仍需确认真实trainer保留标签与顺序、有限loss与显存。

## 最终手动核查与实际启动

[真实1024条CPU核查](../../experiments/026/execution/cpu-manual-audit.json)通过：512＋512、每连续4个microbatch为2＋2、无重复、无超长跳过或删demo。token桶1024/2048/4096/8192分别346/301/269/108；总输入1,998,306 token，末query＋EOS监督228,017 token，最大7464。训练文件SHA `03cde88fd31b45b191313bd72f08002f1e130e87abbe367f09805b83f10a92bf`。CPU运行不等于GPU训练已通过，GPU阶段额外检查trainer保留原标签与episode序号。

最终包装`/private/tmp/arc2-026-v1-paired-training`为697150字节，Notebook SHA `73d31197073408e6507ddfb3ec8adcfdc11b2bf3f9e5b0188b1e0154c934c914`。loader／decoder与成功025包逐字一致，5项嵌入物解压SHA核查通过；两个独立队列各16个相同sorted ID＋2 sentinel。见[包清单](../../experiments/026/paired-v1-package-manifest.json)与[手动包装审查](../../experiments/026/paired-v1-manual-package-review.json)。这些记录只代表准备时事实。

**15:28:04北京时间，平台接受研究Notebook `zhuating/arc2-026-v1-paired-training` V1**，kernelId140735362，平台上限10800秒，请求L4×4、断网、沿用成功025镜像／utility和固定sft139 BF16 V1。启动前可用64286.086配额秒、reserved0。真实[启动回执](../../experiments/026/execution/paired-v1-launch.json)已保存。本次为研究训练与公开开发面板评价，没有创建正式比赛submission；真实硬件／训练步骤／终态以随后日志为准。


远端15:28:47读取为RUNNING、V1、NvidiaL4、断网，模型／utility／镜像数据源正确，全部cell源与本地冻结包一致。见[远端源码核对](../../experiments/026/execution/remote-source-verification-v1.json)。平台metadata的lastRunTime日期仍异常，本轮实际时间按请求／启动日志记录。

实时日志确认4×NVIDIA L4、每卡23034MiB、Python3.11.13；固定权重复制7,267,271,302字节、78.479秒，两个大权重片SHA与025一致。实际utility本轮为嵌套路径`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`，resolver正常；不能将025 GPU的平铺目录当作所有后续硬件统一路径。训练子进程已打印Unsloth初始化信息，首批训练更新仍需核查。无定时监控。


## 启动正常，停止查询等待用户

**15:34:22北京时间**，实时SSE已确认前2个optimizer step完成；前8个episode累计输入18,543 token、监督2,138 token，原query-only有限loss检查通过。第1步日志loss0.0065、grad_norm1.8125，第2步观测峰值显存8,576,153,600字节（约7.99GiB）。SFT实际为1卡、batch1／accum4、1024例／最多256步，Unsloth2025.9.7／Transformers4.55.4／Torch2.8.0+cu128，36层完成patch。未观察到Traceback或CUDA OOM；不以低loss宣称规则已学会或比赛提分。

[启动健康记录](../../experiments/026/execution/paired-v1-startup-health.json)与实时原日志已保存。至此停止所有远端状态查询／日志流，不设自动化，等待用户通知结束。训练合并、两臂TTT／DFS、搜索覆盖、checkpoint身份和配对评分尚未完成核验；后续以真实终态为准。用户通知后先人工执行已准备的`/private/tmp/arc2-026-download-completed-v1.py`读取终态／下载小产物（跳过大权重），核对归档与实际scriptVersionId，然后按026 README冻结预测并评分。若失败，先查具体错误，不重跑多个配方或直接正式提交。

## 完整运行、评分与停止决定

用户通知“跑完了你看看”后读取终态并下载。研究Notebook **V1／scriptVersionId362051855为COMPLETE**，训练与评价子进程均return0、无硬超时。原公开高分参考仍是V1／312063593，两个版本身份不可混用。本轮没有启动新GPU任务或创建正式比赛submission。

### 实际训练和成本

- 实际4×NVIDIA L4、每卡23034MiB；离线SFT单卡、rank32／lr1e-5／accum4、query-only监督、Unsloth梯度检查点。真实**256步、1024唯一episode全部消费**，512合成＋512官方training replay，每accum4均2＋2，与冻结顺序完全相同；输入1,998,306 token、监督228,017 token，最长7464，无超长跳过／删demo或重复补步。
- 固定数据仍为training JSONL SHA `33f3b4e6133be5b890b8f2efb76421390a463b04291f5669f6b5e44bcbd5f901`，固定原模型config SHA `932e9430395ba3eea4087cba1769241bc7d0dc8d574846862d6983659456a2d0`；训练脚本仍为`03cde88fd31b45b191313bd72f08002f1e130e87abbe367f09805b83f10a92bf`。隔离族、quarantine族和评价panel的祖先交集为空；上游公开接触限制仍然存在。
- 单卡训练进程1549.380秒（约25.82分），其中optimizer阶段1354.353秒；mean loss0.0198572，峰值8,768,711,168字节。32步仅存pilot，同一优化器继续至256，没有按验证挑checkpoint。
- 合并成功，72/72探针出现非零BF16变化；完整普通BF16权重7,267,233,496字节，16vocab、EOS及结构保持，tokenizer字节与base一致，随后训练进程退出，再由新进程加载并创建原rank256逐题adapter。合并model config SHA `63573aa9bd7ea58f45a4d96b6cd36fce0b079932205e42f36a5f2bc6981dd20b`；配置差别为pad_token_id None→13及版本元数据，未改模型结构。完整权重留在该精确Kaggle V1输出，未下载到仓库。
- Notebook实际**15:28:30—17:28:42北京时间**，墙钟**7211.426秒（120.19分）**，评价5501.794秒（91.70分）、末尾冻结35.940秒。没有用平台异常lastRunTime推算时间。
- 18:23:08北京时间账号剩余49,783.118配额秒（约13.83小时），reserved0；used相对启动前增加14,502.968配额秒。这是整个账号快照差，非单作业计费明细，见[完成后配额](../../experiments/026/execution/account-after-v1.json)。

### 求解和归档核查

两臂均**16/16 task完成128步TTT，23/23 query全部完成4批DFS，92/92批正常返回并导出**；阶段起止无缺失。无Traceback、OOM或非有限loss。base有2题超过原1200秒软检查，但没有任何搜索覆盖缺口。challenger首次任务比base晚约109秒，两臂最后任务仍分别早于共同截止约1943／1701秒，因此没有因加载差异减少搜索。

base raw／scored均315条、193个query-grid，22/23 query有合法候选；challenger均325条、213个query-grid，23/23有候选。各自raw／scored网格集合完全一致，没有正确raw候选未被重评分的丢失。554项小文件／源码身份核对通过，包括506个candidate pickle；4个大权重文件跳过本地下载，保留远端身份。执行的loader／decoder／solver源码仅有IPython为无末尾换行的writefile body补上一个换行，starter和训练脚本与冻结源字节一致，无算法改动。原B4用候选顺序重放，两臂全部23query均复现一致；没有运行R2或新选择策略救本轮结果。

### 配对评分

先冻结两份预测与完整候选，再读取官方ARC-AGI-2 commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`。**23/23 query均以完整demo和query输入匹配，0版本不匹配**，官方gold哈希与025同题一致；评分使用的预测SHA与冻结完全相同。

| 指标 | 原sft139＋原B4 | 续训权重＋原B4 |
| --- | --- | --- |
| 最终两槽正确query | **4/23（17.39%）** | **4/23（17.39%）** |
| 完整task全query正确 | 2/16 | 2/16 |
| 相对新基线新增／丢失 | — | **2／2，净增0** |
| raw／scored候选oracle | 6/23 | 9/23 |
| 有正确候选但未进入两槽 | 2个query | 5个query |

新增`7ed72f31_1`、`eee78d87_0`，丢失`7b0280bc_0`、`e8686506_0`。前者一项来自候选生成新增、一项来自原来已有正确候选的选择变化；丢失一项在新候选池中没有正确答案，另一项仍有正确候选却未进入两槽。候选oracle新得5、丢2、净3，说明有候选互补现象，但不能计作最终提交收益，也不能据此扩大本配方训练。

本16题在025旧运行B4为6/23；本次新base为4/23。025的Python哈希未固定，运行／数据子集也不同，不能把旧6→新4解释为训练退分。本轮归因只用新base4与challenger4的同轮比较。

**按预定条件结束本续训配方，保留原sft139／B4，不进入预留23题扩验，不用这16题调参数或重排序救checkpoint。** 本轮结果属于公开开发面板，不是Kaggle公榜17.39，正式最好仍为023的30.14。下一轮若研究候选验证／选择，必须另行预先定义方案和开发／确认边界；本次没有启动下一轮。

### 结果入口

- [完整汇总](../../experiments/026/paired-v1-summary.json)、[真实终态与下载](../../experiments/026/execution/paired-v1-outcome.json)、[小产物SHA核查](../../experiments/026/execution/artifact-verification-v1.json)。
- [预测冻结](../../experiments/026/frozen-v1/prediction-freeze.json)、[评分报告](../../experiments/026/scoring-v1-f328/score-report.json)、[逐query计分](../../experiments/026/scoring-v1-f328/query-ledger.jsonl)、[原B4重放核对](../../experiments/026/execution/manual-B4-reproduction-v1.json)。
- 原始运行日志、训练消费与候选在`experiments/026/paired-v1-output/026/`；第三方源码单独落`/private/tmp/arc2-026-output-v1/`，未加入仓库。
