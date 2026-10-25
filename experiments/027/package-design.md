# 027 运行包装与预算只读设计

查阅日期：2026-10-25。此文件仅记录落地研究，未生成上传Notebook、未启动GPU、未读取预留23题gold、未调用远端API。来源为当前本地Kaggle SDK 2.2.4、011挂载实测、023正式提交凭据与026真实阶段事件。

**后续实际决定已更新：** 主任务冻结的027执行方法是原sft139顺序4GPU两遍：base只用TTT增强seed1完整扫原sorted队列，再由同4GPU用剩余时间跑extra增强seed2；Trainer／PEFT seed42及DFS增强seed2保持原样。共同研究上限10800秒，不给base切半预算。下文2+2／TTT优化seed42/43仅保留为已审计而未采用的备选，不能当作当前运行配置。当前可执行包装以`prepare_confirmation.py`和`/private/tmp/arc2-027-v1-confirmation/package-manifest.json`为准。新假设不加载026权重，不使用026十六panel；C23由主任务另行冻结为新策略一次确认，非复活026训练配方。

## 当前边界

最新替换的AGENTS仍明确：026净增0后结束该配方，保留原sft139／B4，不进入预留23题，不在026的16题上调参或重排序救checkpoint。因此不能把复用026权重的新编号直接当作已经获准送23题的方案。本设计优先提出原权重双TTT种子，避免依赖被停止的离线checkpoint；任何后续集合和选槽规则须在执行前另行冻结。

## 首选落地：同一原权重、两个TTT优化种子

两臂均读取同一份已校验的`sft139 BF16 V1`本地拷贝，GPU0/1为seed42，GPU2/3为seed43，两个独立但相同sorted task队列。每臂本地rank0/1；保留026验证过的物理rank串行import，记录每臂模型ready和阶段时间。

仅将每臂的PEFT `random_state`和Trainer `seed`设为预定的42／43，用统一“TTT优化种子”控制初始化和shuffle。原TTT数据增强seed1、n16、rank256、128step、lr5e-5、DFS增强n2／seed2、重评分、B4与1200秒task软检查保持不变。两臂共享固定PYTHONHASHSEED，不能让Python进程哈希为选槽增加另一随机因素。不搜索更多seed，也不根据26题目结果换seed。

无需挂载026训练输出、无新增SFT、无需保存或合并新权重。可直接以026成功包装删除训练cell和训练子进程，沿用loader／decoder与原solver，从一个已经拷贝的模型路径启动两个arm。最终保存每臂原B4、raw／scored候选、原读取顺序和阶段事件，再额外保存**事先固定**的组合预测；组合规则应由独立开发证据决定，不在本文件发明参数。

研究包装继续禁止competition rerun。正式包装必须独立启用隐藏`arc-agi_test_challenges.json`，保留原全局42600秒和43200秒平台上限，保存`/kaggle/working/submission.json`；不能直接提交带研究输入覆盖／rerun拒绝的026派生包。

## 实测预算与2+2的覆盖风险

026的16task／23query在两臂各2L4上完整完成4批DFS／query，实际评价5501.794秒。逐task时间合计：

| 指标 | base | 026 challenger（成本参考） |
| --- | ---: | ---: |
| 总task rank-seconds | 10123.302 | 10217.373 |
| 平均rank-seconds／task | 632.706 | 638.586 |
| 最大单task秒 | 1248.813 | 1112.787 |
| 2worker任务时间分别 | 5007.319／5115.983 | 4965.009／5252.363 |
| 模型ready距评价开始秒 | 84.417／141.077 | 193.673／246.247 |

以上是已分层且按旧正确性挑出的开发面板，不是全量时间分布。原权重双seed成本预计相近，但尚未运行，不能保证。

- 23／24task各双臂的均值粗估约2.1小时，另留加载、队列尾部与保存缓冲，**10800秒研究上限**较合理；建议global end=session−120、进程硬截止=session−60。全部query始终保留分母，覆盖不完整时不得宣称完整双臂能力对照。
- 对120task，单臂2卡均值粗估`120×632.706/2≈37962秒`（10.55小时），加上加载和队列尾部可能接近12小时；对240task则约21.1小时，明显不够。按均值原4worker42600秒可分摊约269task，而2+2每臂约134task。**不能假设隐藏测试只有120题。** 正式运行前应从真实输入取得task／query数和输入长度分布，判定两臂是否能覆盖；该判断不读取答案。
- 若2+2使两臂只覆盖同一批排序靠前任务，新增多样性会被晚题占位抵消。为了保持原4worker覆盖而新增分阶段／任务抢占／模型切换，会成为额外调度因素，当前没有验证，不建议为赶提交同时改造。
- 因此若主要改进为**同一候选池上的CPU选槽**，应优先保持原4worker原权重；它几乎不增加TTT／搜索成本，比双seed更易快速成为可提交版本。2+2是生成多样性实验，不是默认可直接上线的提升。

完成026后账号快照可用49783.118配额秒、约13.83配额小时、reserved0。3小时L4×4研究按2倍约21600配额秒，余约28183秒／7.83配额小时。普通12小时L4×4提交准备运行需约24配额小时，现有额度不够；比赛隐藏重跑的扣额方式仍未确认，不能假定免费。每次动作前都需由主任务读当前快照，不能把此历史读数当现值。

## 确认数据建议（未决定、未运行）

不使用026十六题调种子或选槽，不进入被禁用的预留23题。若主任务选择无训练、固定42／43的独立生成多样性假设，可考虑一次预先冻结025四十题扣除026十六题后的24题；它有历史B4／R2及公开模型接触，不能称盲测。该备选也不能用来继续搜索旧R2或微调B4参数，是否进入应由主任务根据最新研究设计明确。

## 026 checkpoint挂载事实，作为历史复用能力备查

以下仅说明技术路径，不代表推荐重推该权重。

- 精确输出为`zhuating/arc2-026-v1-paired-training` V1，scriptVersionId **362051855**；普通checkpoint相对路径为`026/training/merged_model`，marker为`026/training/merge_ready.json`，manifest SHA `29b97043d9caebb153d780492ee3e2665398e97f23ff3d4dffffcee4315651e9`。
- 模型config SHA `63573aa9bd7ea58f45a4d96b6cd36fce0b079932205e42f36a5f2bc6981dd20b`；两个safetensors大小3987521952与3279711544字节，总7267233496字节；index SHA `8d28e18b4e624395ca74b61294a802de7db16678de20e1cff4491cfbc3c73d4d`。小tokenizer SHA可从026 `merge_ready.json`逐项核对。没有完整大权重SHA，不能宣称重新下载已校验了权重字节。
- Kaggle SDK的`kernels_push`逐项调用`validate_kernel_string`，接受`owner/slug/version`并原样传到`kernel_data_sources`。故metadata应使用`zhuating/arc2-026-v1-paired-training/1`，不能把scriptVersionId362051855作为版本后缀。011已实际使用`ccjjllt/s4-f0-train-010/1`挂载成功，路径见`experiments/011/kaggle-cpu-preflight-v1/011/code/run.json`。这证明历史机制可用，尚未证明026输出在新会话的实际路径或服务端绑定。
- 预计checkpoint路径：`/kaggle/input/notebooks/zhuating/arc2-026-v1-paired-training/026/training/merged_model`。仅为待核实路径，不能写死后直接耗GPU。允许的resolver应只扫描预定父层：`/kaggle/input/notebooks/*/*`、`/kaggle/input/*`、同名sys.path目录，归一化slug后查`026/training/merge_ready.json`；不递归扫描所有packages。唯一根目录才接受，marker／manifest／config／tokenizer SHA和权重size均须匹配，否则失败而不落到latest或重新训练。
- 原utility仍独立保留`sorokin/pip-install-unsloth-flash-patch`。026实际utility为`/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch`，025 GPU曾为平铺`/kaggle/usr/lib/pip-install-unsloth-flash-patch`；保留成功resolver，不把其中一条当统一路径。原模型固定`model_sources=[sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1]`。
- 若将来合法独立假设确实需要此输出，先做≤180秒CPU source-mount核对（仅小文件，禁递归／禁导入模型），再在GPU会话copy原模型和merged模型各一次并核对小文件。CPU探针只准备，没有在本代理运行。若版本固定不被平台接受或marker不一致，停止复用；不要悄悄删掉`/1`读取latest。

## 正式提交入口与执行顺序

023使用的官方调用是`api.competition_submit_code(file_name='submission.json', competition=..., kernel='owner/slug', kernel_version=<Notebook版本号>, message=...)`，普通Notebook push不是正式提交。脚本参考`/private/tmp/arc2-023-submit-formal-v1-20261024.py`，仅作为接口与流程样本，不能原样执行或重复提交361045383。

1. 主任务先固定新方法、开发／确认边界和预算，完成一次有意义的配对评价；未运行／未评分不承诺公榜提升。
2. 如晋级，单独生成正式包，核对hidden input分支、模型source、断网、L4×4、原12小时预算及submission.json；先普通运行必要的小验证，而非把研究guard删除后未经核对直交。
3. 保存真实Notebook版本与scriptVersionId、执行源码SHA、普通输出绑定，再实时读取同队每日额度／pending，确保该精确版本未重复提交。023旧`num_allowed_now=0`是历史读数，不代表今日额度。
4. 正式调用必须指定新Notebook确切版本号，拿到submission ref后检查绑定；不要因传输未知结果重复请求，也不要把普通COMPLETE当正式成绩。
5. 按用户习惯确认正常启动后停止查询，等待用户通知完成；不创建定时监控。


## 正式接入的必要行为（研究C23尚未实现为正式包）

当前C23包装明确保留研究异常：base错误不运行extra；extra非截止异常使研究ERROR但最终导出可诊断base／extra候选；共享硬截止下extra部分输出单独标记。它有研究rerun guard及固定C23输入，不能正式提交。

如C23晋级，正式包继续原V1的42600秒全局求解、43200秒平台总时长，先完整4worker base，再用剩余时长跑extra；没有2+2并行截掉base覆盖。**base清洁完成后必须立即冻结原B4的`/kaggle/working/submission.json`及候选身份快照，然后才启动extra。** Extra异常不能让已完成base整体失败：正式state必须明确`base_fallback_after_extra_error`，使用原base两槽并保留错误证据，不能保留部分extra却谎称正常ensemble。共享截止造成的extra部分覆盖可按事前固定规则，仅使用已原子写完的pickle；base第一槽仍按既定helper保留。

普通4题验证分支也应真实运行base四worker后extra四worker，以核对两阶段进程退出、marker清理、重新加载与最终fallback；不能仅在单臂普通分支跑通而让隐藏分支首次运行顺序两阶段。正式fallback行为尚未执行或验证，需在晋级后单独小范围接入；本代理没有启动研究或正式提交。
