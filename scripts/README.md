# 共享研究脚本

这里保留已有27B／网格研究的共享实现，**当前023忠实V1打包入口在[experiments/023](../experiments/023/README.md)**。脚本互相按同目录导入，历史Notebook也记录了原路径，因此不按用途搬动源码。

| 用途 | 脚本 | 说明 |
| --- | --- | --- |
| 数据 | [arc_episode_data.py](arc_episode_data.py) | episode数据池、表示与采样 |
| 词表 | [convert_grid_vocab.py](convert_grid_vocab.py) | 网格词表转换研究 |
| 训练 | [train_grid_sft.py](train_grid_sft.py) | 网格SFT、LoRA、权重初始化和续训 |
| 逐题适配 | [ttt_per_task.py](ttt_per_task.py) | 既有TTT研究入口 |
| 网格推理 | [solve_grid.py](solve_grid.py) | 网格生成、候选与推理 |
| 运行支持 | [grid_runtime.py](grid_runtime.py) | 网格运行的共享工具 |
| 程序推理 | [solve_program.py](solve_program.py) | 程序生成与受约束执行 |
| 思考探针 | [run_reasoning_probe.py](run_reasoning_probe.py) | 27B思考／程序研究 |
| 预算运行 | [run_budgeted_grid_experiment.py](run_budgeted_grid_experiment.py) | 常驻底座与分阶段预算 |
| 依赖探针 | [probe_gdn_inference.py](probe_gdn_inference.py) | GDN性能与数值研究；采用与否读012进展 |
| 初始可行性 | [r0_qwen38_lora_feasibility.py](r0_qwen38_lora_feasibility.py) | 006起点的27B LoRA诊断 |

实际命令、数据划分和配置以[对应计划／进展](../docs/README.md)为准；冻结实验目录里的源码副本代表当时版本。这里只做人工发起的研究，不新增测试套件或CI。
