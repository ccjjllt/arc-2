# 015 — Qwen3.8-27B 完整推理与程序反馈

日期：2026-10-18，北京时间。对应[计划](../plan/015-qwen38-reasoning-research.md)。

version 7 跑完了冻结 P2 和 P6。两臂 query EM@2 都是 **1/6**，命中的不是同一题。14/18 次调用在输出上限处截断，最终通道为空。这不是比赛分。程序反馈和 P12 未运行。

## 1 比赛与配额

查询时刻 2026-10-18 15:04 北京，CLI 身份 jiyujia。原始记录在[competition-snapshot.json](../../experiments/015/competition-snapshot.json)。

| 项 | 值 |
| --- | --- |
| 公榜 | 3015 队。本队 jiyujia（ccjjllt, jiyujia）第 1013 名，公榜分 **30.14** |
| 最新提交 | 57103498 COMPLETE，该次 publicScore **25.69**，私榜空。公榜名次用的是队伍最佳分 30.14 |
| 榜首 / 第 8 / 第 16 | 88.06 / 68.19 / 58.33 |
| 金牌线参照 | 超过 1000 队时 `10+floor(0.002×队数)`，3015 队约为第 16 名。最终以私榜、有效队数和资格为准 |
| 截止 | UTC 2026-11-02 23:59，北京时间 2026-11-03 07:59 |
| 今日提交 | 已用 0，剩余 1 |
| GPU 配额 | 15:04 北京 used 42945.436 秒，剩 18.070712 小时。推送前 15:08 北京 used 43375.494 秒，剩 **17.951252 小时**，reserved 仍是 0，刷新 UTC 2026-10-24 00:00 |

时长用 `timedelta.total_seconds()`。07:04:19Z 到 07:08:30Z，used 增加 430 秒，墙钟约 251 秒，折合 RTX 会话大约 1.71 倍配额。018 自 14:57 北京起，硬限 5400 秒，推送时大约还要 2.24 配额小时。L4×4 按 2 倍、再留计划里的 2.3 系数和 6 小时，本次会话上限定为 **15200 秒**。

## 2 已做的修正

延续 014 已定位的问题，只改推理和计量。

- 两个 pip 分支都显式安装 `wheels/vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl`，安装前核对 SHA256 `5cbeedeb…304a`。
- 流式请求带 `stream_options.include_usage`，先读 usage 再读 choices。
- 调用墙钟和安装+加载的 20 分钟共用绝对截止；静默超过 180 秒才断开，避免把长 prefill 当成失败。
- 完整 reasoning / final 单独落盘。只有 `finish_reason=stop`、颜色 0–9、边长 1–30 的闭合网格算合法答案。
- 协议之后有一次 4 路、每路最多 256 token 的并发波，用同一波次墙钟算聚合 tok/s。
- `scripts/run_reasoning_probe.py` 与 `experiments/014/v-probe/run_reasoning_probe.py` 字节相同。

本地手动检查（没有新建测试文件）：安装命令两条分支都带上该 wheel；`10 11` 与未闭合 fence、`length` 都不算答案；标准 SSE 的 usage-only 尾块得到 `completion_tokens=11`；并发波把 11+7 加总。这些是假响应，不是 GPU 生成。

## 3 两次推送

同一 slug [jiyujia/arc2-015-v-t-27b-reasoning](https://www.kaggle.com/code/jiyujia/arc2-015-v-t-27b-reasoning)，id_no 139255458。挂载是 FP8 权重、vLLM wheelhouse、比赛数据、断网、`NvidiaL4`。

**version 5** 于 07:09:25Z 启动，约 309 秒后 `EVIDENCE-EXIT status=install_failed`，返回码 0。4×L4、Python 3.12.3、driver 580.178.04，wheel SHA 与冻结值一致。3.12 没有 ensurepip，走了 3.13 pip 的 `--python-version 3.12 --target`。把 vLLM wheel 和 requirements 放进同一次解析后，pip 报 `No matching distribution found for torchaudio==2.11.0`。这份 wheelhouse 的 requirements 含 torch==2.13.0 和 torchvision，不含 torchaudio。原始文件在 [kaggle-v5](../../experiments/015/kaggle-v5/run.json)。没有生成，T 未开始。

**version 6** 安装改为先装 requirements，再 `--no-deps` 装同一个 vLLM wheel。07:31Z 左右结束，`elapsed_s` 668.8，状态 `recheck_not_admitted`。vLLM `0.27.2rc1.dev91+g1f7427bc0` 从 `/kaggle/working/vprobe/site/vllm/__init__.py` 导入，torch `2.13.0+cu130`。安装 258.2 秒，安装加加载 658.5 秒。xhigh/low 模板渲染不同。两次 hello 都是 `finish_reason=stop` 且带 usage：xhigh 37 completion token、1.8 秒、20.23 tok/s；low 32 token、1.4 秒、23.34 tok/s。4 路并发波墙钟 1.7 秒、139 completion token、聚合 81.66 tok/s，usage 没有缺失。最终文本都是换行后的 `[[1]]`，没有闭合围栏，所以 `valid_answer` 为 false，T 没有启动。这是探针口令和围栏解析不一致，不是 CUDA 失败。原文在 [kaggle-v6](../../experiments/015/kaggle-v6/run.json)。短请求速度不能当成后面 32K ARC 请求的速度。

**version 7** 于 07:34Z 推送，状态 RUNNING。探针里正常结束且正文正好是 `[[1]]` 的回复算完整探针答案；ARC 题仍只接受闭合的 0–9 围栏。会话上限 14626 秒。vdriver SHA256 `fad17d46657f966ca692bb2a73c88624b2a9a215e2db5b16c8ca6f6c85499984`。

## 4 这次会话里要跑的实验

嵌入字节与磁盘源码一致：

| 文件 | SHA256 |
| --- | --- |
| vdriver.py | version 7：`fad17d46657f966ca692bb2a73c88624b2a9a215e2db5b16c8ca6f6c85499984`。version 6 是 `08e0d169…0fe3f2` |
| run_reasoning_probe.py | `af29f0419d90dda6cfa1df0753884cc7ea609ebea25d9527989f4c5246d64be6` |
| p2-prompts.json | `9f0c23dc53994fc21a937ebfc1ea51d11af4377281ed07fa44bdcb723854dd83` |
| p6-extra-prompts.json | `76f3b6324e4ee28bebc12b63f2e0d6834e6c65883beb0090892c1f7940eeb572` |

- 复验：两档模板、hello 各最多 4096 token、然后 4 并发短请求。两档都产生 token 且 usage 齐全、至少一份合法答案，才进入 T。
- T-high：xhigh 一次，32768 token，seed 1001，两题并发 2。只有一个答案时第二槽重复该答案。
- T-low：low 两次，各 16384 token，seed 2001 与 2002，填满 4 并发。
- 先跑冻结 P2（291dc1e1_0、db118e2a_0）。P2 完整且剩余时间盖过按 P2 实测成本放大的 4 道额外题时，才跑 P6 里未跑过的 4 题。不按对错换题。
- 评分在取回结果后本地做，query EM@2。不把 981571dc 读进来。

version 7 的逐题结果见下一节。程序反馈、P12 和新的比赛提交未运行。训练没有启动。

## 5 version 7 结果

Notebook [jiyujia/arc2-015-v-t-27b-reasoning](https://www.kaggle.com/code/jiyujia/arc2-015-v-t-27b-reasoning) version 7，`elapsed_s` 7009.1，状态 `complete_with_evidence`。4×L4、driver 580.178.04、Python 3.12.3。vLLM `0.27.2rc1.dev91+g1f7427bc0`，torch `2.13.0+cu130`。安装加加载 658.3 秒。会话上限 14626 秒，实际没用满。原始产物在 [kaggle-v7](../../experiments/015/kaggle-v7/scores.json)。结束后 jiyujia 剩 12.94 配额小时，reserved 0；`arc3-018` 已结束。

主指标是 query EM@2：两个槽里有一个完整网格与标准答案逐格相同即命中。没有最终网格的 query 留在分母里，算错。题目来自已经用过的开发集，不是未见确认，也不是公榜分。

| 范围 | T-high | T-low | 实际墙钟 | completion token |
| --- | --- | --- | --- | --- |
| P2，2 query | 0/2 | 1/2 | 2046.1 秒 | 96463 |
| P6，6 query | 1/6 | 1/6 | 高臂约 4106 秒，低臂约 2234 秒 | 高臂 155733，低臂 186047 |

P6 的两道 P2 题复用了同配置结果，没有重跑。高臂墙钟是 P2 的一次 2 并发加上额外 4 题的两次 2 并发。低臂是 P2 的一次 4 并发加上额外 8 次调用的两次 4 并发。

| query | T-high | T-low |
| --- | --- | --- |
| 291dc1e1_0 | length，32768 token，最终为空 | 两次都 length，16384，最终为空 |
| 4c416de3_0 | length，32768，最终为空 | 两次都 length，16384，最终为空 |
| 58c02a16_0 | stop，15941 token，16×16 与答案一致；第二槽是这一份的重复 | 两次都 length，16384，最终为空 |
| aa4ec2a5_0 | length，32768，最终为空 | 两次都 length，16384，最终为空 |
| db118e2a_0 | stop，8720 token，最终通道为空 | 两次都 stop，8480 与 13727 token，两份 15×15 相同且都与答案一致 |
| fc4aaf52_0 | length，32768，最终为空 | 两次都 length，16384，最终为空 |

18 次任务调用里，14 次 `finish_reason=length` 且最终通道为空，1 次 stop 但最终通道为空，3 次 stop 并给出合法网格。思考通道里出现过的围栏没有一份等于标准答案，所以没有把“思考里的矩形”算进分数。两臂各对一题，并集 2/6 只是诊断上界，不是一种已经提交的选 2 规则。

短探针 4 并发聚合约 78 tok/s。长请求上，高臂波次聚合约 31–46 tok/s，低臂约 75–87 tok/s。低臂墙钟更短，token 更多，因为两次采样经常都顶满 16384，而高臂有两次提前 stop。低臂的唯一命中比高臂便宜。58c02a16 的高臂实际只用 15941 token 即命中，低于低臂的 16384 上限；低臂两次在上限处截断。因此这里只能说明不同策略命中不同，不能把高臂命中归因于更大的上限，effort、seed、并发也没有独立控制。该解释由016复核纠正，原始结果不变。

同一套 32K/16K 上限再扩到 P12，主要会增加空的最终通道。这次没有启动 P12，也没有做程序反馈。下一步若还用这台 27B，应先让输出在上限内离开思考并写出最终网格。
