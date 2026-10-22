# 023 恢复有效提交与候选选择研究进展

日期：**2026-10-22，北京时间**。对应[023交接计划](../plan/023-score-recovery-and-selection.md)。审阅Git基点`f2cefe876ac339ccf327171fd3cd7c81b8cf3f1b`；本轮开始工作树干净。用户请求了解当前现状、审视其TRM构想并写执行交接，本轮未把研究建议扩成GPU执行授权。

## 1 已完成的研究与事实修订

- 阅读最新README、020—022计划／进展、原始4B候选与配对记录、正式零分凭据和用户粘贴构想。确认57504054为实际COMPLETE／0.00，隐藏执行证据仍缺，不能断言是模型能力、加载或分支路由造成。
- 联网核对2026官方Overview、Data、Rules与ARC Prize主站。计分是query两槽exact微平均；截止为北京时间11月3日07:59。Kaggle Rules与主站对Bonus分配及许可有文字差别，计划明确以正式规则及主办方澄清办理，未承诺85%必得15万美元。
- 通过GitHub工具读取NVARC冻结commit `846d0198efa752534594e321fc3289fc0a06c657`的论文、TRM说明及实际适配代码。论文§4.4明确最终比赛提交使用Qwen3-4B单独系统；2B集成有退分、4B集成未增，纠正“补TRM就是补全冠军必需流程”的推断。
- 核实TRM有作者提供的权重入口，但新任务需要重建embedding并适配。未下载权重或运行TRM。NVARC约2小时TRM阶段与另一篇TRM适配论文的长预算不是同一实现，未混用。
- 核实用户引用的68.17／46.99来自作者自建177题而非官方ARC2评估；原文表6还写45.99，不能拿它作确定竞赛分数。核对Johan Land的半私有72.9%及API成本限定、2026递归研究的ARC1／ARC2区别。
- 公开Code排序和榜单本轮访问遇到超时，未取得实时最高公开Notebook或金牌线。精确Failed in AIMO V1／33.89沿用10月21日已冻结的原页面与源码证据，不写成10月22日实时排名。未依据第三方榜单推断高分队伍算法。

来源链接与查阅日期、原论文页码、版本证据集中列于023计划。NVARC PDF原件通过GitHub插件临时保存为`/private/tmp/arc2-nvarc-2025-research.pdf`；Git blob SHA `9a3771265bce947f62df809bcef2e55dfbe319dd`是Git对象标识，**不是SHA256**。

## 2 已有候选的手动CPU实验

问题：原候选oracle5/27、最终3/27，能否仅调整已有评分的选槽获得低成本信号？只比较R1最低NLL前二与R2保留原首槽＋剩余最低NLL，未进行阈值搜索。

| 记录项 | 实际值 |
| --- | --- |
| 数据 | 022固定公开E20，20task／27query；25项f328官方revision、2项旧revision完整输入匹配；既有开发集合 |
| 原模型与推理 | `sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1`；E20 V2／360845520；原4×L4推理3749.775秒 |
| 本轮执行 | 本地Darwin arm64、Python3；标准库CPU回放，无GPU、无新模型调用 |
| 候选与配置 | 原122个去重评分候选；原`kgmon=support−mean_aug_nll`；R1/R2稳定排序、保持原占位与全部分母 |
| 运行时间 | 主线程落盘回放约0.0264秒；子代理另做一次只读独立计数，输入相同，不是独立研究样本 |
| 结果 | B4 3/27、2/20完整task；R1/R2均4/27、3/20完整task，new1/lost0/net+1 |
| 覆盖 | 21query有真实候选、6占位；oracle仍5/27 |
| 产物 | `experiments/023/selection-replay/`，含两政策输出、逐query账本、输入输出SHA、完整平台和耗时 |

唯一新增`3a25b0d8_0`，原排名第3；`142ca369_0`原第5仍错。R1/R2所有query的无序候选集合相同，仅一题槽位次序不同，不能当两次独立正证据。此次没有改变生成、TTT或重新评分，也没有恢复6个占位。

结论限定：这是事后公开开发集回放。R2有进入一次新集合确认的理由，但未证明泛化、比赛提分或可达5/27；仍可能丢失其他题原第二槽的正确答案。下一轮只冻结R2，不继续在E20调参。

## 3 形成的交接与当前状态

形成顺序：限时回查旧零分／取得新的共用全量路径证据 → 可用4B正式起点 → R2一次扩展同候选确认 → 按证据决定是否需要覆盖调度或TRM挑战。TRM必须比较相同总预算下被挤占的4B收益，不因参数少或冠军仓库包含它而直接上主线。

- **已完成：**调研、事实纠正、手动CPU回放、023计划／进展及README入口。
- **未修改：**求解核心、模型权重、现有评分脚本、历史冻结输出和正式Notebook。
- **未运行：**新GPU、40题确认、覆盖调度对照、TRM适配／推理、续训、全量演练及新比赛提交。
- **未操作：**账户／团队、最终提交选择、定时监控；没有恢复自动轮询。
- **比赛效果：**没有新公榜；最新仍57504054的0.00，历史最好30.14。零分根因未确认。

手动核对新增JSON计数与SHA、计划链接、阶段状态和Git变更范围。独立复核发现初次落盘对单候选query重复了首槽，已恢复022原第二槽`[[0]]`并同步SHA、账本及政策说明；结果仍为4/27，未新增／删除正确解，修正耗时另记report。未新增测试文件、框架或CI。后续实际执行需继续追加本进展，不能把计划预算、外部论文分数或CPU回放写成GPU／比赛结果。

## 4 用户后续收敛：先忠实复现高分Notebook，再优化

2026-10-22用户明确提出先参考`koushikrudra/failed-in-aimo`取得高分起点，再在其基础上持续优化。已据此修改023计划、README和AGENTS中的优先级；当前只推进原V1／312063593的忠实复现，R2、TRM、新训练和40题确认后置，不作为首份复现提交的前置。

本轮重新访问用户URL及指定版本未取得网页正文，沿用10月21日已归档的V1／33.89证据，没有声称核实最新分数、当前排名或金牌线。33.89不能称金牌保证。

只读对比原V1源码和022配置确认：原全局求解42600秒，022正式求解截止41100秒；原1200秒是解码外层软检查，022fair首轮约660秒且训练可中止；022另加逐题60秒评分预留及评分截止。这些会改变实际训练和搜索机会，不能统称加载修复，也不能把022的0.00说成原V1能力结果。尚无证据认定这些差异中哪一项导致零分。

首份复现改从冻结原Notebook派生，保留原方法与时间规则，只在挂载、依赖、加载发生直接阻塞时做最小补丁，记录差异和实际成本。此前计划中的统一完整入口重构与完整普通全量演练不再是默认前置；原普通四题成功仍不等于隐藏成功，最终以精确版本实际提交结果报告。未修改求解代码，未启动GPU、提交或监控。

## 5 用户授权推进后：独立复现包已准备，等待共享GPU释放

2026-10-22用户要求开始推进研究，并澄清后续可以使用新子代理，但不要调用此前三个旧子代理。旧代理均已完成，当前接口没有删除历史记录能力；本轮另建`faithful_v1_audit_023`完成只读独立审计，没有再让旧代理承担工作。

已新增[手动打包入口](../../experiments/023/prepare_faithful_v1.py)，从冻结原V1直接生成独立包`/private/tmp/arc2-023-faithful-v1-frozen/`，目标私有Notebook为`zhuating/arc2-023-failed-in-aimo-v1-faithful`。Notebook SHA256为`0bdde6503f6ec616be467321649e9abb997288f07aa54e8b4302f76f32f3d119`；[版本清单](../../experiments/023/faithful-v1-package-manifest.json)和[审计记录](../../experiments/023/execution/prelaunch-source-audit.json)已保存。

相对原V1的运行适配只有：显式GPU／L4配置；当前固定模型挂载路径；将同一权重复制至本地盘；原Unsloth utility进入子进程PYTHONPATH；同等求解命令增加非零退出传播。复制与加载时间计入原42600秒。`starter.py`保持原字节，`arc_solver.py`只改变权重存储路径；原四题、worker导入／marker顺序、完整单epoch、1200秒软限制、DFS、增强评分和kgmon最终两槽均保留。没有移植022的fair调度、TTT截止回调、评分预留或四模型ready屏障。

独立静态审计通过。依赖utility仍由无版本ref挂载，历史V1镜像digest也未暴露；当前包使用022已成功运行的镜像，实际依赖版本需运行时记录，不能称依赖字节级复现。这里只是运行包和静态核对完成，不是GPU复现成功。

资源的实际变化：

- **12:31北京时间**只读核对zhuating：可用32300.548秒（8.9724配额小时）、reserved0，见[初始快照](../../experiments/023/execution/account-start.json)。
- 同队当天`numToday=1`、累计10次；最新提交仍57504054／0.00，没有pending。按每日最多1次规则，正式提交需等额度恢复并在提交前重查，见[只读就绪记录](../../experiments/023/execution/readiness.json)。
- 启动前再次核对发现新GPU预留；**12:37北京时间**可用0、reserved86207.944秒。实际占用来自**12:36启动的另一项目`zhuating/p015-fsdp-capacity`**，状态RUNNING；同账号另有ARC3运行，未修改任何其他项目。见[再次核对](../../experiments/023/execution/account-before-launch-recheck.json)与[占用记录](../../experiments/023/execution/quota-occupancy.json)。
- 本地启动前检查在调用SaveKernel前停止，**本轮没有创建远端Notebook、启动GPU或创建比赛submission**；不是平台已经接受后失败。

用户选择等待现有作业结束，随后明确“我等结束了告诉你”。据此停止主动轮询、保持无定时任务；等待用户通知后再查当前配额、同名作业和正式额度，然后启动上述冻结包。不得根据旧快照自动认为资源已释放，也不得取消其他项目作业。

## 6 用户通知配额释放后：忠实V1普通GPU运行已启动

2026-10-22用户通知“你开始吧现在有配额了”。**14:55北京时间**复查zhuating：GPU可用18293.165秒（5.0814配额小时）、reserved0，见[恢复快照](../../experiments/023/execution/account-resume.json)。冻结包仍逐SHA匹配，不复用其他项目作业。

首次SaveKernel请求在TLS代理握手时超时；已先只读确认远端未创建同名Notebook、同队无pending后才重试，没有重复启动。原请求与错误保留在[第一次尝试](../../experiments/023/execution/faithful-v1-launch-attempt1.json)，重新查询单独保存在`readiness-after-timeout.json`，未覆盖历史就绪快照。

**14:59:19北京时间**平台接受`zhuating/arc2-023-failed-in-aimo-v1-faithful` **V1**，kernelId **140168317**，Notebook SHA仍`0bdde6503f6ec616be467321649e9abb997288f07aa54e8b4302f76f32f3d119`。请求显式NvidiaL4、断网、平台43200秒；原42600秒全局时钟、原四题及原求解语义保持。见[真实启动记录](../../experiments/023/execution/faithful-v1-launch.json)。这是普通GPU运行，**不是正式比赛提交**。

15:01只读回查状态RUNNING，远端metadata确认GPU开启、NvidiaL4、断网、权重BF16 V1与原Unsloth utility挂载、currentVersionNumber1。源下载至`/private/tmp/arc2-023-remote-v1/`供新子代理`runtime_verification_023`独立核对。metadata的lastRunTime出现不一致日期，实际启动时刻使用本次请求／响应时间及运行日志，不据该字段倒推。

实时日志已确认同一checkpoint **7,267,271,302 bytes**在**71.180秒**完成本地复制，随后写出原四份源码。复制仍计入原全局预算；此成本不当作后续加载／求解时间。当前尚未报告训练完成或正确数。SSE重连会重放部分日志，禁止按重复块累计多次阶段耗时。

原四题共**4task／5query**，普通最终文件仍保留整个公开evaluation结构，不能将未选中占位当模型全量成绩。原候选目录`/kaggle/inference_outputs`没有新增归档改动，若平台取不到候选则本轮oracle记未确认。最后ordinary benchmark不是query微平均，结果需冻结后独立按输入版本匹配评分。

当前同队仍已用当天1次正式额度，最新正式成绩仍57504054／0.00。普通运行完成后冻结输出与版本、记录实际结果，再在正式额度恢复后按同一精确版本提交；不提前宣称已复现33.89。没有创建定时任务。

## 7 忠实V1普通运行COMPLETE：原四题实际3/5，等待正式提交额度

**2026-10-22 15:31北京时间**下载冻结时，官方状态为COMPLETE。完整一页47项输出清单绑定精确 **scriptVersionId361045383**、V1／kernelId140168317；未将kernelId代替版本ID。见[输出冻结记录](../../experiments/023/execution/faithful-v1-outcome.json)。新的普通运行已完成，不是仍在排队或只完成源码准备。

远端10/10 cell与冻结包一致，四份实际写出的源码均只是`%%writefile`正文末尾额外一个LF，代码无漂移；复制helper SHA一致。boot确认普通分支、固定sft139 BF16 V1、原42600秒时钟；四worker日志均为NVIDIA L4，Unsloth2025.9.7／Transformers4.55.4／Torch2.8.0+cu128／Triton3.4.0与原版吻合。新子代理`runtime_verification_023`只读独立验收通过，见[运行验收](../../experiments/023/execution/ordinary-v1-runtime-verification.json)。

| task | 实际TTT步数／epoch | TTT秒 | 整题秒 | 去重后的DFS批次日志 |
| --- | --- | --- | --- | --- |
| 0934a4d8 | 128／1.0 | 399.542 | 563.7 | 4 |
| 36a08778 | 128／1.0 | 192.063 | 1103.3 | 8 |
| 981571dc | 128／1.0 | 810.543 | 1211.2 | 4 |
| aa4ec2a5 | 128／1.0 | 301.356 | 647.5 | 4 |

四题均有实际DFS、候选重评分、finished和worker done记录，无Traceback。981的整题超过1200秒，符合保留的原版外层软检查语义，没有引入022式TTT硬截断。日志最后一个持久事件的会话相对时钟为**1637.606秒，约27.3分钟**；它不是平台单独暴露的精确总耗时。四题并行耗时不能求和当墙钟。

权重复制71.180秒／7,267,271,302字节；此成本已在原全局时钟内。账户前后used差为3278.802配额秒，按2倍折算约1639.401秒，与日志量级相符，但共享账户快照不是独立计费证明。

### 7.1 输出冻结和独立实际评分

原完整文件保留120task／172query结构；实际求解只有原四题的**5query**，其余167query两槽均为占位。所选5query共9个非占位网格槽，981的第二槽仍为`[[0]]`。不把完整JSON结构当公开120题模型全覆盖。

最终预测在读取新评分gold前冻结：[冻结凭据](../../experiments/023/execution/ordinary-v1-prediction-freeze.json)。当前官方公开evaluation输入SHA与008快照一致（`e7c62a4b…e4e8`），保留原4题5query输入清单，未根据结果选题。实际挂载输入没有单独导出哈希，这一限制保留在清单中。

随后人工调用既有`experiments/020/score_frozen_outputs.py`，以官方固定commit `f3283f727488ad98fe575ea6a5ac981e4a188e49`的evaluation数据匹配全部ordered demos及query输入：**5/5匹配，0未确认／版本不匹配**。结果见[评分报告](../../experiments/023/scoring-ordinary-v1-f328/score-report.json)和[逐query账本](../../experiments/023/scoring-ordinary-v1-f328/query-ledger.jsonl)：

| query | 原最终两槽exact |
| --- | --- |
| 0934a4d8_0 | 否 |
| 36a08778_0 | 是，首槽 |
| 36a08778_1 | 否 |
| 981571dc_0 | 是，首槽 |
| aa4ec2a5_0 | 是，首槽 |

**实际为3/5 query（60%），2/4完整task。**日志`Reload score: 2.5`是原task加权累计值，不是query正确数、公开120题准确率或比赛分数。旧013的2/4只覆盖3题4query，不能直接与本次5query分母比较。本轮仍为已经开发／可能被上游训练接触的公开原演示集合，不是未见验证，也未证明复现33.89或实现比赛提分。

原候选位于`/kaggle/inference_outputs`，未在这次忠实普通包中新增归档；平台输出不能取得完整池，**raw/scored oracle未确认**。不为补oracle修改已完成V1或额外重跑GPU。

### 7.2 正式版本已就绪，当前额度为0

**15:34北京时间**再次实查zhuating：reserved0、可用15014.363配额秒（约4.1707小时）；同队`num_today=1`、`num_total=10`、**num_allowed_now=0**，最新正式提交仍57504054／COMPLETE／0.00。见[运行后实查](../../experiments/023/execution/postrun-account-and-submission-state.json)。因此本轮没有创建新正式submission，不能声称可靠全量正式基线已经建立。

[正式就绪参数](../../experiments/023/formal-v1-ready.json)已冻结：Notebook `zhuating/arc2-023-failed-in-aimo-v1-faithful`，**kernel_version=1**，精确scriptVersionId361045383，文件`submission.json`。额度恢复后先重查同队额度、pending和是否已提交该精确版本，再创建一次code submission；无需为了提交重跑普通保存或重发新版本。

本轮完成了忠实原四题GPU流程、实际评分、版本与产物冻结。当前只剩正式额度恢复后的全量计分步骤；R2、TRM和新训练仍后置，未创建定时监控或操作其他项目。
