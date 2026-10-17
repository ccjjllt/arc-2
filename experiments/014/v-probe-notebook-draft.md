# V Notebook打包入口草案(014-5,未运行GPU)

## 挂载(创建Notebook时固定,开机首cell核对version_number)
- model: `foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1`
- dataset: `nick2187/qwen38-vllm0272-cu130-wheelhouse-v1` version **1**
  (commit 1f7427bc0ad31fdad1b0933c77ea5acae15da2e0, py3.12/Torch2.13/TF5.15/CUDA13)
- machine: 4×L4, internet off
- **不用**013旧Python3.11镜像(57e612b4…);新wheel为cp312,旧镜像不兼容。
  b98k镜像 `e5452ce6…` 的Python版本开机实测`python --version`为准,不预设。

## cell顺序(总上限1h会话;GPU计时含下载/安装/JIT)
1. 环境证据:`python --version; nvidia-smi; pip --version`;记录driver/CUDA。
   若python≠3.12或driver不支持CUDA13 → 按证据退出,不升级包、不混装旧torch。
2. 离线安装(≤20min,计时):
   `pip install --no-index --find-links <wheelhouse>/wheels -r <wheelhouse>/requirements-qwen38-runtime.txt`
   (借madarshbb-b98k-v1 L1934-1937;新wheelhouse自带requirements,禁用`--no-deps`覆盖TF5.8旧做法)
   超时/失败 → 落盘`run.json`退出。
3. 模板核对:`python scripts/run_reasoning_probe.py --skip-serve`
   渲染`rendered-prompt-xhigh/low.txt`,两档sha不同才继续(已CPU验证:308 vs 237字符)。
4. 短协议(≤5min):`python scripts/run_reasoning_probe.py --prompt-file <P2一题prompt>`
   成功标准=真实生成+reasoning/final区分+finish_reason/usage/TTFT落盘;
   “端口已打开”不算成功。失败按证据退出。
5. 余量进P2(291dc1e1,db118e2a各1 query):同脚本逐query调用,并发≤4;
   是否扩P6/P12由实测聚合R按014-6准入公式决定,不短测外推。

## 不照抄项
131K context / C16并发 / 98K思考 / max_tokens106496;MTP(`--speculative-config`)/
`--kv-cache-dtype fp8`/多级server_ladder;旧0.19+TF5.8两阶段`--no-deps`覆盖;
顶层`reasoning_effort`字段(对Qwen无效,见russcore L1659-1660)。
