# 021 Qwen3.8 27B E20 单轮候选包

日期：2026-10-21，北京时间。**已准备，未push，未运行。** 由主执行者在4B对照实际运行后决定启动。

研究目的为公开E20的候选覆盖诊断。它允许27B观察全部20task／27query，不代表最终隐藏240task中只给部分任务分配27B的B2部署预算；不能把这批全覆盖候选直接当同预算融合成绩。

## 固定配置

| 项目 | 配置 |
| --- | --- |
| 账号／slug | `zhuating/arc2-021-qwen-e20-direct32k` |
| 输入 | `experiments/020/evaluation-e20/challenges.json`，20task／27query，SHA256 `ad12e367296c958b1f84a1c2bf7c2a8858cfb09db4dbc8e152d5735890b67dda` |
| 权重 | `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1` |
| 依赖 | 已验证的`zhuating/arc2-020-runtime-recovery` v1，沿用原193个wheel与vLLM SHA |
| 服务 | TP4、C8、MTP2、mamba cache align |
| 生成 | xhigh，thinking24576，total32768，名义final空间8192；原sampling与P0 seed5101 |
| 调用 | 每task一次Direct1，覆盖其全部query，最多20次模型调用；不附加CPU规则、repair或强制收尾 |
| 时钟 | 总7200秒，包含安装加载；T−600秒停新请求，T−300秒停生成，Notebook parent guard另留45秒 |
| 解析 | 原020整task严格JSON解析；本轮CPU逐query回放没有恢复，故未加入解析改动 |

使用`--p0-only`排除跟随调用。继承driver的`M0/M1`产物名称代表同一份共享P0的重复视图，并非两个独立研究臂，不产生额外模型调用。主输出仍为`/kaggle/working/submission.json`，原始回复和候选池均逐次保存。

7200秒不保证全20task完成。按完整27query分母报告未完成、超时和占位；公开E20早在019暴露过输入，不能称模型未见集。评分前冻结新输出SHA，再按官方版本的完整demo和query输入匹配。

## 相对020的修改

- `submit/vdriver020_submit.py`：`finish`默认退出码由0改1，记录`exit_code`；末尾显式成功退出仍为0。推理、prompt、sampling、候选选择和parser均未改。
- `prepare_notebook.py`：读取021目录副本，适配嵌套路径的repo root，运行时解包到独立的`/kaggle/working/021-qwen-code`，更新Notebook用途说明。
- 其他driver依赖文件与020逐字相同。差异见[code-diff.patch](code-diff.patch)，完整源SHA与原版比较见[research-manifest.json](research-manifest.json)。

Notebook是研究包，保留`assert not rerun`，不供比赛提交。

## 包装和产物

手动包装命令，**不会push**：

```bash
python3 experiments/021/qwen-e20/prepare_notebook.py \
  --owner zhuating --slug arc2-021-qwen-e20-direct32k \
  --cohort experiments/020/evaluation-e20/challenges.json \
  --session-seconds 7200 \
  --runtime-kernel zhuating/arc2-020-runtime-recovery \
  --policy M1 --gen-max 32768 --thinking-max 24576 \
  --mtp-tokens 2 --generation-mode direct --direct-attempts 1 --p0-only \
  --output experiments/021/qwen-e20/notebook
```

准备好的[Notebook](notebook/arc2-021-qwen-e20-direct32k.ipynb)约116KB，SHA256 `c69c98b6caf4960b866464e6a1944f0b0ee88a706185c3e2622194fbd78d6cdb`。挂载元数据见[kernel-metadata.json](notebook/kernel-metadata.json)。

已手动解包核对源码SHA、编译Python源码/Notebook cells、校验E20固定SHA及20/27数量、确认query无output、CPU补槽关闭且P0-only开启。没有远端操作或模型调用；这只说明包内容符合配置，不代表GPU可运行或解题有效。
