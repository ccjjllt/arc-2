import os, shutil

# Fresh outputs avoid mixing old local/debug files with this submission run.
shutil.rmtree("/kaggle/inference_outputs_main", ignore_errors=True)
os.makedirs("/kaggle/inference_outputs_main", exist_ok=True)

# Stage-1: original-strength baseline. This is the score anchor.
# Keep train_aug=16, eval_aug=2, min_prob=0.20, dfs=540, no beam truncation.
!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 ARC_OUTPUT_DIR=/kaggle/inference_outputs_main ARC_RUN_TAG=base ARC_SKIP_DONE=0 ARC_TASK_ORDER=lexicographic ARC_BUDGET_MODE=fixed ARC_MAX_PUZZLE_SECONDS=1200 ARC_MIN_PUZZLE_SECONDS=420 ARC_NPROCS=4 ARC_TRAIN_AUG_N=16 ARC_TRAIN_DATA_SEED=1 ARC_EVAL_SEEDS=2 ARC_EVAL_AUG_NS=2 ARC_MIN_TOKEN_PROBS=0.20 ARC_DFS_TIME_LIMITS=540 ARC_MAX_SCORED_BEAMS_ROUND0=0 ARC_MAX_SCORED_BEAMS_ROUNDN=0 ARC_SCORE_SEED_OFFSET=13 ARC_WORKER_MARKER_TAG=base python starter.py --end-time {global_end_time}

# Stage-2: coverage rescue only. Unique=1 is accepted because a stable single grid can be the correct ARC answer.
# This fixes v8's wasteful repeated reruns on high-confidence single-answer tasks.
!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 ARC_OUTPUT_DIR=/kaggle/inference_outputs_main ARC_RUN_TAG=rescue_a ARC_SKIP_DONE=1 ARC_SKIP_DONE_MODE=quality ARC_SKIP_DONE_MIN_FILES_PER_TEST=10 ARC_SKIP_DONE_MIN_UNIQUE_PER_TEST=1 ARC_SKIP_DONE_MIN_CANDIDATES_PER_TEST=6 ARC_FOCUS_MODE=quality ARC_FOCUS_HARD_N=16 ARC_TASK_ORDER=hard_first ARC_BUDGET_MODE=adaptive_parallel ARC_MAX_PUZZLE_SECONDS=840 ARC_MIN_PUZZLE_SECONDS=220 ARC_NPROCS=4 ARC_AUGMENT_MODE=extended ARC_TRAIN_AUG_N=8 ARC_TRAIN_DATA_SEED=7 ARC_EVAL_SEEDS=23 ARC_EVAL_AUG_NS=1 ARC_MIN_TOKEN_PROBS=0.16 ARC_DFS_TIME_LIMITS=190 ARC_ROUND1_REQUIRE_INCOMPLETE=1 ARC_MAX_SCORED_BEAMS_ROUND0=10 ARC_MAX_SCORED_BEAMS_ROUNDN=8 ARC_SCORE_SEED_OFFSET=17 ARC_WORKER_MARKER_TAG=rescue_a python starter.py --end-time {global_end_time}

# Stage-3: narrow last pass for genuinely low-coverage outputs, not low-unique outputs.
!UNSLOTH_DISABLE_STATISTICS=1 TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas OMP_NUM_THREADS=12 ARC_OUTPUT_DIR=/kaggle/inference_outputs_main ARC_RUN_TAG=rescue_b ARC_SKIP_DONE=1 ARC_SKIP_DONE_MODE=quality ARC_SKIP_DONE_MIN_FILES_PER_TEST=14 ARC_SKIP_DONE_MIN_UNIQUE_PER_TEST=1 ARC_SKIP_DONE_MIN_CANDIDATES_PER_TEST=8 ARC_FOCUS_MODE=quality ARC_FOCUS_HARD_N=8 ARC_TASK_ORDER=hard_first ARC_BUDGET_MODE=adaptive_parallel ARC_MAX_PUZZLE_SECONDS=700 ARC_MIN_PUZZLE_SECONDS=200 ARC_NPROCS=4 ARC_AUGMENT_MODE=extended ARC_RESCUE_AGGRESSIVE=1 ARC_TRAIN_AUG_N=8 ARC_TRAIN_DATA_SEED=11 ARC_EVAL_SEEDS=37 ARC_EVAL_AUG_NS=1 ARC_MIN_TOKEN_PROBS=0.13 ARC_DFS_TIME_LIMITS=130 ARC_MAX_SCORED_BEAMS_ROUND0=8 ARC_SCORE_SEED_OFFSET=23 ARC_WORKER_MARKER_TAG=rescue_b python starter.py --end-time {global_end_time}

# Guarded final selection:
# attempt_1 = kgmon anchored to baseline candidates when available;
# attempt_2 = best distinct full-probability candidate from all candidates.
!ARC_INFERENCE_DIRS=/kaggle/inference_outputs_main ARC_SELECTION_MODE=diverse_dual ARC_SUBMISSION_PATH=/kaggle/working/submission.json python build_submission.py
