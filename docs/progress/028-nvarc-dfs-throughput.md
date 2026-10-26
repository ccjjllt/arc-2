# 028 NVARC DFS路径复用小实验进展

日期：2026-10-26。对应[028计划](../plan/028-nvarc-dfs-throughput.md)。用户授权继续优化后，本轮已实现并启动一次有上限的机制实验；**完整结果、真实提速、同预算最终得分和比赛提分均未取得**。

## 面板与固定对照

重新派出三个代理，分别负责实际cache源码研究／proxy实现、无gold面板冻结、实际上传包只读审查，没有沿用旧代理。取题只使用027既有24题开发集合，排除026十六题与C23；按每query已知prompt网格cell均值分为8／8／8的短中长桶，每桶按旧025 DFS批次耗时中位数取第4题。不按正确率、oracle、候选数或新输出取题。

| 层次 | task | 保留query数 | 平均已知prompt cells | 旧DFS批次中位秒 |
| --- | --- | ---: | ---: | ---: |
| S | d35bdbdc | 3 | 700 | 23.75 |
| M | 271d71e2 | 1 | 1670 | 50.45 |
| L | 7b80bb43 | 1 | 2797 | 81.46 |

源输入保留3task／5query，query答案字段为0；实际机制测量**每task首query的前两批原增强**，共3query，不是全量5query计分。长度为网格cell代理，旧耗时为host事件区间，均不冒称真实token／同步GPU测量。输入11690字节，SHA **341e7bac741286b6cde44c29279f2105b9bae6cbe9df028c6c98e3bd2c6826cb**。见[面板manifest](../../experiments/028/panel-v1/manifest.json)与[全部24题选择记录](../../experiments/028/panel-selection.json)。该面板是历史开发材料，不称未见确认集。

每题只执行一次原sft139的TTT rank256／128步、增强seed1、Trainer／PEFT seed42；eval增强seed2、hash27027、DFS概率阈值0.2。旧版和proxy改版共用**同一内存adapter**。先用首批原DFS生成草稿；取按原返回顺序第一个合法candidate，转换到第二批各view。没有草稿时明确记录fallback。第二批固定顺序`original_A → path_A → path_B → original_B`，不按结果挑顺序或草稿。

## 运行时约束与实际实现

本机原utility源码与官方[Unsloth2025.9.7发行包](https://pypi.org/project/unsloth/2025.9.7/)核对，2026-10-26由新代理内存读取，未安装或落盘：wheel314642字节，SHA **a15c55a064c0b86c5b462662b21a0f2d55f9004394db01f72c522aa669fec744**。`qwen3.py` SHA `74e159d0835a3238f0be669fa49c52cae0f2588c3eb758a050ea8f0e459a3ece`，`llama.py` SHA `0ec7ec3200e48f7be97a9893d3f5380c3dae63f9a975ecb935607cf74fba03f4`。

重要限制是`llama.py`非空past走fastdecode，并硬性要求`q_len==1`；不能直接把16-token带cache调用塞入原DFS。Qwen3 cache为每层(K,V)，fastdecode第一次复制到内部paged缓冲，之后主要使用传入cache长度、内部paged内容；单纯切snapshot长度却保留旧paged会有回溯错误。

因此自有[dfs_path_reuse.py](../../experiments/028/dfs_path_reuse.py)只实现proxy，**原`turbo_dfs`和`inference_turbo_dfs`函数体保持不变**。首次无past一次前向prompt＋最多16个既有草稿token，clone不可变KV和相应logits；全部batch行（含PAD行）匹配时才跳过一次forward。首次偏离即停用本次复用；任何snapshot祖先重新进入真实decode前清理paged及其临时buffer，使正确前缀重新复制进去。snapshot按storage／offset／stride／长度识别，不把可变paged view当不可变备份。未知cache类型直接受控失败并记录诊断。

这是保守的前16token机制验证，不是2025报告所述完整speculative DFS，也不承诺大倍率；克隆与重建开销可能抵消收益。概率树、候选顺序、EOS15、最大输出长度和单次540秒限制仍由原callback执行。多次ABBA测量受本次5400秒会话上限约束，重复测量可超过原单task1200秒，**不能把本轮总耗时当生产task预算或全量覆盖证据**。未来端到端评价仍须保留原1200秒task／42600秒全局约束。

[pilot_runtime.py](../../experiments/028/pilot_runtime.py)记录每call起止、deadline可能截断标记、实际prompt tokens SHA、草稿来源、模型forward次数／token数、GPU显存、NLL差异、完整返回序列集合及顺序；计时包含prefill、clone和恢复，并在边界同步CUDA。原版重复漂移由A／B两次提供参照。仅比较raw候选、NLL和吞吐，**本机制pilot不执行B4，不计最终准确率**，不能据此宣称排序保真或比赛提分。

## 实际包装与启动

[prepare_pilot.py](../../experiments/028/prepare_pilot.py)读取冻结025／027来源，第三方求解源码只生成于`/private/tmp/arc2-028-dfs-pilot-v1-ready`。实际Notebook为**69027字节**，SHA **2d1d554f24cc20e566772cfdb00bd69a7f530e7c197f6ac8c84f7e87edde31ba**；helper SHA **b658d01533d599cc4074caf3f60bceb0b98cd61b5d87de3cc0e6ea8b9dce2e34**，runner SHA **0551b066357f15c82715b14bf536b072de8b51f7a45826e74724fb144807862a**。记录见[包manifest](../../experiments/028/pilot-v1-package-manifest.json)。

初次内部审阅包SHA056548…，随后仅为runner补call起止／截断与返回顺序诊断，实际上传包为上面的2d1d…；原求解函数及helper未因此改变。手工AST解析、四个原函数AST身份对照和无query答案检查通过。发现的`precompute_tokens`接口不一致已在上传前修复，没有以该错误包启动GPU。

**21:24:21北京时间**只读预检可用GPU26932.892秒（7.48配额小时），reserved0、无pending submission；**21:39:08—21:39:13北京时间**临近启动再核对相同可用量，平台接受[arc2-028-dfs-path-pilot](https://www.kaggle.com/code/zhuating/arc2-028-dfs-path-pilot) **V1**，kernelId140991448（不是scriptVersion）。模型／utility／镜像均沿用成功原sft139环境，申请L4×4，实际仅3个worker，硬件库存与实际版本另以boot核对。

会话API上限5400秒、solver全局5200秒、进程组硬截止5320秒，包含安装、加载与收尾；不做离线续训、不创建competition submission。回执见[launch](../../experiments/028/execution/pilot-v1-launch.json)，此时只证明平台接受，完整运行与提速仍未确认。

## V1启动错误与V2修复

**21:41:25北京时间**实查V1已ERROR，全部cell源与上传包一致。完整日志显示第三个cell的`boot.update`引用`raw_rerun_flag`，新封装setup漏定义该日志变量，触NameError。已定位到实际堆栈，**模型挂载与config校验已通过，但尚未进行权重本地复制、模型加载、TTT或DFS**，不把这次失败解释为新算法无效。

V1实际scriptVersion **362510062**，日志及5项产物已下载至`experiments/028/pilot-v1-output/`，第三方Notebook／HTML仅在私有tmp，见[终态凭据](../../experiments/028/execution/pilot-v1-outcome.json)。首次启动包的审阅未捕捉跨cell变量遗漏，这是本轮封装错误，已修正。

V2唯一Notebook变化是在setup补`raw_rerun_flag=os.getenv('KAGGLE_IS_COMPETITION_RERUN')`。原求解函数、helper、runner、面板及配置均未变化，逐cell对照只变化cell2且只增加这一行；没有重新挑题、换seed或放宽cache检查。

**21:43:04北京时间**平台接受同Notebook **V2**，SHA **0c9d58242c9dae02afb8afb411deda8a5e76ce5af14596b37b5a382fe40f5153**、69092字节。见[V2包manifest](../../experiments/028/pilot-v2-package-manifest.json)与[V2回执](../../experiments/028/execution/pilot-v2-launch.json)。临近启动GPU可用26841.172秒（约7.46配额小时）、reserved0；相比V1前减少91.72配额秒，以账户快照差值记录，不能将其冒称算法耗时。V2实际scriptVersion及完整结果仍待取得，kernelId仍不能当scriptVersion。

完成接续入口已准备`/private/tmp/arc2-028-download-completed-v2.py`，仅在终态后读取、下载和归档，不创建新运行或提交；用户通知后使用这个V2入口，避免取旧V1失败输出。

## 完成后读取与判断

正常启动后停止远端查询，按用户既有方式等待通知，不建定时监控。通知完成后先核对精确V1终态、实际scriptVersion、全部报告和产物SHA；第三方runtime源码下载到私有tmp，仓库只留数据与诊断。

先看3份task报告、TTT128步、原／改两次配对均闭合，有无cache不支持／OOM／异常或deadline。然后看draft可得性、实际跳过forward数、完整批次候选与NLL差异相对原版重复漂移、同顺序比较、真实耗时与峰值显存。没有实际复用或没有稳定收益，就结束这个固定16-token配方，不用此面板扫描草稿长度救结果。截断的批次不能作为完整树一致证据。

机制通过后，再做独立冻结的同预算搜索评价；只有生成覆盖、候选oracle与最终两槽的实测证据支持，才走正式提交。027融合保持结束，02330.14仍为正式最好。

## V2启动加载已观察，停止查询

**21:44:05北京时间**只读确认V2为RUNNING、所有cell源与SHA0c9d…实际包一致，输入3task／5query、实际测量3query。boot显示4×NVIDIA L4各23034MiB、Python3.11.13、原模型config指纹及utility挂载正确；实际模型复制7267271302字节，88.249秒。

截至**21:50:02北京时间**的有界启动流，rank0／1均出现启动及两片checkpoint100%加载，未见Traceback或CUDA OOM。rank2、TTT完整统计、首个path_A调用尚未观察，不能写成三题已运行完或缓存算法已通过。SSE存在重复回放，不将消息次数当真实步骤或推算各rank完成时刻。证据见[启动健康](../../experiments/028/execution/startup-health-v2.json)与[源绑定](../../experiments/028/execution/source-verification-v2.json)。

按用户既有等待方式，现在停止所有远端状态／日志读取，无后台轮询或定时任务。接续状态存[ready](../../experiments/028/pilot-v2-ready.json)，实际V2 scriptVersion仍未知。用户通知完成后执行`/private/tmp/arc2-028-download-completed-v2.py`；该入口先严格核对当前V2与全部cell源，再检查终态、读取输出URL真实scriptVersion、归档诊断及SHA。实际吞吐、候选变化、NLL漂移及提分仍未确认。
