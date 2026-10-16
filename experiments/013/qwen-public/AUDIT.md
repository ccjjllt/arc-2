# Qwen3.8 public notebooks — source-only audit
Retrieved 2026-10-12 (Asia/Shanghai) with Kaggle get_notebook_info. No GPU run, no dependency installation, no submission. No output cells were provided for either readable notebook. Source comments, forum scores, and planned token rates are not reproduced measurements.

## madarshbb/arc-qwen-3-8-27b-agent
URL: https://www.kaggle.com/code/madarshbb/arc-qwen-3-8-27b-agent
Metadata: current_version_number=2; last_run_time=2026-10-09T09:05:30.430489100Z; NvidiaL4; internet off.
Despite the agent title and large included REPL implementation, active config sets mode=direct. It returns grid code blocks, not complete Python programs. No program-synthesis evaluation in the active path.
Model mount: foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1. The actual weight/config contents were not inspected.
Dataset mounts: mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3 and keithtyser/taaf-duck-qwen38-serving-v1. The latter overlays serving wheels including transformer/hub replacements.
Serving: vLLM OpenAI server, tensor parallel 4, prefix caching, gpu_memory_utilization=0.90, context 81920, max_num_seqs=16, client concurrency=10.
Options ladder tries FP8 KV + 2-token MTP speculation, then drops features, ultimately plain serving. This is fallback code, not proof the fastest option successfully ran.
Budget: max_tokens=49152, thinking_token_budget=40960, xhigh template, temperature=1.0. If reasoning-budget unsupported, code drops that parameter for later requests; this can leave long thinking uncontrolled.
One no-thinking rescue with 8192 tokens from final 16000 characters of reasoning. Active direct mode works around incomplete reasoning with a grid, not with completed executable code.
Scheduling: first samples for every task ahead of second samples; smaller prompts first; stop new tasks with <600s; incremental submission writes, input-copy floor; deadline 12h minus 35min. tok_per_s=140 is a config estimate, not observed data.
Reusable: offline serving bootstrap, TP4 batching, direct baseline, bounded answer rescue, incremental full-coverage scheduling.
Risks: 81920 context/FP8 KV/MTP/support stack need real validation; upstream source availability/licenses and weight provenance need checking. Large included inactive code should not be copied wholesale.

## russcore/arcagi2-qwen38-freeform-direct
URL: https://www.kaggle.com/code/russcore/arcagi2-qwen38-freeform-direct
Metadata: current_version_number=17; last_run_time=2026-10-09T09:05:33.690659300Z; NvidiaL4; internet off.
Model mount is the same FP8 repack v1. Source pins vLLM=0.19.0 and searches a wheelhouse whose hint is arc3-vllm-h100-wheelhouse-v3.
WARNING: metadata dataset_data_sources is [""], while runtime explicitly needs the wheelhouse. Do not assume copying v17 provides every dependency.
Serving: vLLM TP4, 65536 context, gpu_memory_utilization=.90, prefix caching, max_num_seqs/client concurrency=24; --dtype bfloat16 is set, while FP8 quantization is expected via model config, not an explicit --quantization flag. Actual weight quantization config not read.
Agent: 10 turns, run_python tool with at most 4 simultaneous tool executions. It finally emits a grid; complete standalone transform(input) source is not required.
xhigh pass: tool-turn cap 40960, direct cap 57344, episode 3600s before adaptive full-coverage reduction. medium spare pass: 20480 tokens and 1800s. Global hidden-run wall=43020s, reserve=180s.
Forced final when last turn or (after first turn) <8 minutes remain: no tools, enable_thinking=False, low effort, cap4096. A separate no-thinking 4096-token recovery uses reasoning tail.
Scheduler: all query attempt_1 xhigh+Python before spare-time risk-ordered attempt_2 medium+Python. Concurrency-aware episode wall adapts to remaining wall and query count.
Default saved version is floor-only and does not run model. Explicit ARC_FREEFORM_DIAGNOSTIC=1 starts diagnostic, KAGGLE_IS_COMPETITION_RERUN=true selects hidden rerun. A saved notebook success is not evidence inference ran.
Source comment claims v7 ~255 aggregate tokens/s at C24 vs ~158 at C8, zero preemption, 97.2% KV. No corresponding raw logs/outputs supplied here: treat as author report. It cannot be directly divided by project's 4.54 single-sequence tokens/s to assert acceleration.
Reusable: short extraction of serving configuration, forced-final protocol and coverage scheduling; full ~188KB source contains substantial general runtime/gate code not aligned with this research project's no-framework preference.

## Third notebook
https://www.kaggle.com/code/khoinguyennguyen/discovery-by-acc-of-observations-arc-agi2-tir
get_notebook_info returned Permission 'kernels.get' was denied. No source inspected; no implementation claims made; no bypass attempted.

## Proposed experimental priority
Preserve selected Qwen3.8-27B. First compare practical inference serving, not another training round: use known FP8 repack or a provenance-checked export, vLLM TP4 with concurrency 4->8->16 as memory permits, measure aggregate throughput, per-query wall, parseable grids, ARC2 exact match and projected full-run wall on a fixed small development cohort. Start without speculative decoding/FP8 KV until plain TP4 works, then evaluate optional speed features independently. Reuse direct grid and forced-final recovery as baseline, then tool-augmented program attempts. Keep held-out confirmation cohort untouched until selecting a configuration. This is a proposed manual experiment; none was run here.

Files .source.py preserve notebook Markdown and code/magics as text for inspection, not directly executable Python.

