# 030 实际配对包静态审查 V1

2026-10-27，新子代理 `r030_source_audit_new`。只读实际active v3包及生成器，未运行GPU、未读gold、未提交。明细见 [pair-static-review-v1.json](pair-static-review-v1.json)。

**静态审查未发现启动阻塞；这不等于实际GPU验证或提分通过。**

| active包 | Notebook SHA256 | 字节 | task/query | session/每臂slot |
| --- | --- | --- | --- | --- |
| initial-v3 | `8a6361e5ff76011ab1af63c736ecd0c014336a46c296c9794b9e142997d8966c` | 123865 | 40/58 | 14400/6600秒 |
| extension-v3 | `c26c3d36e21860c262e18f075d635adbb06fd7c968df68e235530af69b8d43c2` | 132541 | 40/57 | 14400/6600秒 |

旧 `extension-v1` 使用更长预算，已废弃，不得用于执行或结果晋级。旧pair v2已由v3替代；v3相对已审v2仅最终化cell8变化。当前两包均固定4小时总cap、每臂6600秒，初轮A→B、扩展B→A。

核对通过：

- `turbo_dfs`、`inference_turbo_dfs`、`calc_scores`逐字与原V1一致；loader与decoder逐字一致。PEFT、train_args和max_seq_length的AST一致；原16views／4批布局、1200秒task软检查和540秒DFS保持。
- 唯一求解参数差异为`-log(0.2)`对`-log(0.1)`；共同hash27030在fresh子进程启动前设定。原B4及遍历顺序冻结保持，未混入027额外TTT或融合。
- 解压实际输入后，初轮40/58、扩展40/57，test项均仅含input。研究rerun拒绝，最终化不读query gold；完整分母经原get_submission保留，包括未搜索占位。
- 两包都远小于1MB，所有导出Python和其余普通代码cell通过AST解析，实际源码SHA与manifest匹配。
- 共同setup用时对两臂各扣一次；两臂各自从启动时计算相同global/hard duration，前一臂工作不会侵占后一臂预算。每臂全4卡、独立process group。timeout杀整个group并等待；下一臂前清原4个import markers。
- solver只导出完整atomic pickle，incomplete目录不会被读入；原重评分允许在soft global截止后至hard kill之间收尾，规则两臂一致。

需在运行／晋级前保留的限制：

1. 本机Python无torch，adapter指纹的BF16→CPU uint8字节view仅做静态审查；需实际机制运行确认。代码调用现有get_peft_model_state_dict(default)，不改变RNG或参数；指纹开销计入task/global墙钟。
2. v2的最终化计时缺口已修复：v3把共同final imports/input读取对每臂各计一次，再计各臂B4／有序scored导出时间；accounted deployment=shared setup＋child elapsed＋final common＋branch final。缺失child计时或超过6600秒会设置deployment_budget_accounting_failure／return1，不能进入效果比较。实际计时仍须运行取得。
3. A/A噪声控制、实际TTT adapter指纹一致性和截止覆盖均未取得。不同面板反转arm顺序只能减少系统性先后偏差，不能完全排除JIT／文件缓存影响。
4. 使用全新干净Kaggle会话；交互式重复执行旧working目录可能留旧pickle，不能将其混入当前运行。
