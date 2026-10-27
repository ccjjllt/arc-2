# 030 源码差异审计 V1

日期：2026-10-27，北京时间。新子代理 `r030_source_audit_new` 执行。仅CPU源码／历史指纹核对，未读新query答案、未运行模型／GPU、未提交。机器可读明细见 [source-audit-v1.json](source-audit-v1.json)。

## 结论

- 原V1／312063593的Notebook SHA为 `71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d`，与仓库冻结身份一致。
- 原V1四个`%%writefile`源码均可从仓库恢复；loader、decoder、starter直接匹配023指纹，solver仅替换模型存储路径后匹配023。继承025与027纯源码包装变换也逐字复现各自成功包的solver指纹。
- 历史默认临时包 `/private/tmp/arc2-023-faithful-v1-frozen` 等已不存在，不能继续把这些默认路径视作可用依赖。主代理已恢复copy helper到 `/private/tmp/arc2-030-recovered-023/copy_model.py`；本审计实读3253字节，SHA `5d51c4e6698ce5d574dd9997b140e0e370ea611fbee05d1de197b665a8458010`，与023 manifest一致。下载本身不被冒称精确V1源绑定；内容SHA绑定通过。
- 023、025、027研究、027普通正式包的四份历史model-copy记录，全部10个文件的bytes/SHA完全一致。模型config为 `932e9430395ba3eea4087cba1769241bc7d0dc8d574846862d6983659456a2d0`；两片权重分别 `379de219f74a1402a67d26fe7dcacdacadbf686947e08e41117dd9c884ee4529`、`3b8470b83355ccbc527afcb7a77664bd679e63899a80e79169c016658e6b92e0`，共7,267,271,302字节。这是历史成功拷贝指纹，未在本地重新读取权重。
- 原作者镜像digest未知；023采用已成功的镜像digest `320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868`。原工具仅有挂载来源／版本日志，没有完整目录固定SHA，不得宣称整个历史环境已逐字复制。

## 核心源码指纹

| 组件 | 原V1 SHA | 023复建SHA | 与023记录 |
| --- | --- | --- | --- |
| arc_loader.py | `d01cd56167e534ae156706ef62ab11662fc7b403964614a7a55131940bde970c` | `d01cd56167e534ae156706ef62ab11662fc7b403964614a7a55131940bde970c` | 一致 |
| arc_decoder.py | `965cfd910777d9bbec9681c6c15c5e2ea3924569734248046f5e58f16cb28222` | `965cfd910777d9bbec9681c6c15c5e2ea3924569734248046f5e58f16cb28222` | 一致 |
| arc_solver.py | `a7faf4d4eb9e6ba257653ad7beae697805157c02c16dd0be5acd1c166ae67a7a` | `bb55ff101a01fc9d1b583dca67a59219c0dfa3eed3b6a387d61a4f62e668161d` | 一致 |
| starter.py | `c3f0b49bd7fed7679eb8d128abc07517f4ec423613a8c20f17eb8b8b982ab3d4` | `c3f0b49bd7fed7679eb8d128abc07517f4ec423613a8c20f17eb8b8b982ab3d4` | 一致 |

025 solver复建SHA `bc462db7d58b962000ab34686f4d7bd5802cc90ae8296facf676ba6f56af629b`，starter `0b0fa56b7485a7b5498c86d0638c5a93ec2f0bd5bf5ddaaf32d339e869f3aab3`；均匹配成功V2。027 solver复建SHA `6b9ffc044d1f81e97c989d3b50597bc681f9b1f75928c0cf1315c8edec3be4db`，匹配成功V1。以上为纯函数变换与SHA核对，没有执行导出的第三方模型源码。

## 030需要保留与披露

原sft139 BF16 V1、rank256 rsLoRA、alpha32、PEFT／Trainer seed42、TTT增强seed1、1epoch（既有运行128步）、8192长度、原16搜索视角／4批与8重评分视角、540秒DFS／1200秒task软检查、原kgmon B4均保留。原NLL是answer token求和，没有长度归一化。正式全局42600秒从第一原cell开始，原sorted四worker队列无model-ready barrier。

唯一求解假设为累计路径概率0.2对0.1。新Python hash seed必须在父环境里冻结并对两臂一致，因原重评分确实用`hash(bk)`选择增强；这是相对023未显式固定hash的已知差异。短研究截止、观测导出和输入子集另行披露，不当作忠实12小时复现。030不得带入027的TTT seed2额外分支／第二槽融合，也不得带入028／029的草稿prefill或cache修改。

本次尚未审计新030实际生成包SHA；应在生成后做单独审查。外部33.89与本项目30.14差距原因仍未确认，源码可重建不构成效果或环境等价证明。
