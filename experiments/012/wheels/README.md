# 012 旧K探针wheel下载位置

2026-10-22目录清理已删除三个本地wheel，共 **197,777,628字节**。删除前逐项SHA256匹配[原始清单](../wheel-manifest.json)；清单记录它们已发布为`ccjjllt/arc2-012-k-wheels`，本轮未重新联网核验。保留`dataset-metadata.json`。

K快速后端已停止采用，当前023不使用这些文件。**旧`experiments/012/prepare_notebooks.py`所有模式（包括`--check-only`）仍会读取wheel；重新运行该旧打包器前需先恢复。**本轮不为清理修改旧研究代码。

如需恢复，在项目根目录手动执行以下下载命令。这些固定版本直链来自[011来源记录](../../011/gdn-followup-evidence.md)，本轮未下载或安装：

```sh
curl -fL 'https://github.com/Dao-AILab/causal-conv1d/releases/download/v1.7.0/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' -o 'experiments/012/wheels/causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
curl -fL 'https://files.pythonhosted.org/packages/2d/ed/dfe19c4da779957eb6a42a26812f9b4e2280bf757a17a71933ff59ffcb98/fla_core-0.5.2-py3-none-any.whl' -o 'experiments/012/wheels/fla_core-0.5.2-py3-none-any.whl'
curl -fL 'https://files.pythonhosted.org/packages/90/d2/2070e3cf2148c5cce99ca4876633c4b7b89ec085323600bd3d47aeacd306/flash_linear_attention-0.5.2-py3-none-any.whl' -o 'experiments/012/wheels/flash_linear_attention-0.5.2-py3-none-any.whl'
shasum -a 256 experiments/012/wheels/*.whl
```

将输出与`wheel-manifest.json`三项SHA逐一比较，匹配后再运行旧打包器；下载和安装不会自动触发。此次清理不代表重新采用K路线。
