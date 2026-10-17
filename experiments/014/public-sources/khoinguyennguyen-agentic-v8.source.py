# CELL 0 (markdown)
# Discovery by Accumulation of Observation: Community Edition

An ARC-AGI-2 task is a small world whose transformation law has not been written
down. Its training pairs are observations of that world in action; its test inputs
ask what the world will do next. This notebook solves a task by treating it that
way — as a mini-reality to be investigated rather than a pattern to be matched.

Each task is solved by a **single persistent solver** that works inside its own
Python workspace, records what it notices, tests candidate laws against every
training pair, revises those laws when they fail, and finally commits to a
prediction. Before its answer is accepted, the solver is nudged to check its own
reasoning. The two attempts the solver produces — one from the initial
investigation, one from the verification round — become `attempt_1` and
`attempt_2` in the submission.

This edition is the community release. It is deliberately smaller than the full
exploration notebook and runs in a single pass: one solver per task, one
verification round, four tasks solved in parallel on four GPUs. Every block below
is ordinary Python you can read and edit. The prose describes what each block
does, why it is built that way, and what you will see when it runs.

## What the solver is given

Each task arrives as a complete bundle: every training pair, every test input,
and nothing else. The solver never sees test outputs; it never reads a solutions
file. It works only from the training pairs, its own experiments, and its own
records.

## How the solver works

The solver runs as a multi-turn conversation with two tools:

- **`python`** — a persistent Jupyter kernel. Variables, functions, and imports
  survive between turns. The kernel is preloaded with the scientific stack
  (`numpy`, `scipy.ndimage`, `sympy`) and a handful of helpers for the work that
  ARC tasks demand repeatedly: printing grids, counting colors, diffing two
  grids, applying a candidate transform to every training pair, and reading the
  actual PASS/FAIL results.
- **`cognitive_activity`** — the solver's lab notebook. Every entry has a
  freeform label and full content, and every entry is preserved in the evidence
  trace. Entries can be marked to survive a context reset.

There is no separate `thinking` tool. Prose reasoning that does not belong in
code goes into a `cognitive_activity` entry with a label like `"thinking"` or
`"plan"`. One notebook, one record format, one trace stream.

## Context and compression

The conversation can grow past the context window. When it approaches the
ceiling, the solver is asked to write a handoff summary for the next instance of
itself. The Python kernel state is preserved; only the conversation resets.
Preserved cognitive activities are re-inserted into the fresh conversation as
remembered notes. The original conversation, the summary, and the preserved
activities all remain on disk for review.

## Models
- Qwen3.8 27B FP8
- Qwen3.6 35A3 FP8

## Submission format

The competition expects one entry per task, one entry per test input, with both
attempt slots filled:

```json
{"task_id": [{"attempt_1": [[...]], "attempt_2": [[...]]}, ...]}


# CELL 1 (markdown)

## Block 01 — Clock and CONFIG

This block defines **the single source of truth** for every runtime parameter.
Nothing later in the notebook may declare a hardcoded path, timeout, flag, or
model name. Everything reads from `CONFIG`.

### What CONFIG covers

The dictionary is organized into groups. Each group exists because a different
decision needs to be made about it, and grouping the parameters by that decision
is what keeps the notebook debuggable.

| Group | Keys | The decision it answers |
|---|---|---|
| Execution | `ENV`, `execute`, `MODE`, `N_TASK_UNITTEST`, `UNITTEST_SEED` | Is this a dry run or a real run? Kaggle or Colab? How many tasks does a unit test sample? |
| Context and output | `context_tokens`, `max_output_tokens`, `buffer_tokens`, `compression_trigger_ratio`, `compression_max_tokens` | How big is the model's context, how much output does one request get, and when does compression fire? |
| Timeouts | `overall_timeout`, `solve_stop_seconds`, `finalization_seconds`, `request_timeout`, `python_timeout`, `per_task_timeout`, `max_turns` | How long may each phase take, from the whole run down to a single tool call? |
| Parallelism | `task_parallel`, `branches_per_task`, `preload_workers` | How many tasks run at once, how many solvers per task, and how many threads read the model weights? |
| Sampling | `temperature`, `top_p`, `presence_penalty`, `seed` | How does the model sample, and what seed does each branch derive from? |
| Verification | `verification_rounds` | How many nudges does the solver get before its answer is accepted? |
| Model serving | `model_dir`, `served_model`, `port` and the vLLM flags | What server is started, on what port, with which features? |
| Offline install | `wheel_dir`, `requirements_path` | Where does vLLM come from on a machine with no network? |
| Paths | `challenges`, `solutions`, `submission`, `evidence`, `task_ids` | Where do inputs come from, where do outputs go, and which tasks are selected? |

### Why a single CONFIG dict

Previous versions of this notebook scattered parameters across a dozen cells.
Changing one number meant finding every consumer and hoping none was missed.
Putting every parameter in one dict makes a change visible, singular, and
testable. This is the discipline that makes the notebook debuggable — a reviewer
reading the config sees every assumption the notebook makes, in one place.

### Derived values

Two values are computed **once** from the base config and never redefined:

- `prompt_ceiling = context_tokens − max_output_tokens − buffer_tokens`
  → the largest prompt that still leaves room for a full output. A prompt larger
  than this cannot be sent without truncating the output.
- `compression_trigger = compression_trigger_ratio × prompt_ceiling`
  → the prompt size at which the compression round fires. The ratio leaves
  headroom: the model needs room to write its handoff summary.

A third derived value, `peak_concurrent_branches = task_parallel ×
branches_per_task`, is a sanity check. It must not exceed the server's
`max_num_seqs`, or the engine will queue requests and the queueing delays will
show up as solve-loop timeouts.

### Mode detection

The notebook reads `KAGGLE_IS_COMPETITION_RERUN` once, at import time. The value
is `"1"` or `"true"` (case-insensitive, whitespace-stripped) during a real
competition rerun, and absent or `"0"` / `"false"` otherwise. The result decides
two things:

- Whether the notebook loads the test challenges (real competition) or the
  evaluation challenges (development).
- Whether it looks for a solutions file at all. The solutions file is only for
  local scoring; the solver never sees it.

`MODE = "UNIT_TEST"` when the rerun flag is not set. In that mode the notebook
samples `N_TASK_UNITTEST` tasks with a fixed seed so runs are reproducible.
`MODE = "FULL"` when the rerun flag is set; every task in the test challenge file
is solved.

### Preflight output

Running this cell prints every important parameter and creates the evidence
directory. If any path or value looks wrong, fix `CONFIG` **before** running the
server cell. A wrong path discovered here costs one edit; the same path
discovered inside a long solve loop costs a failed run.

# CELL 2 (code)
# ── Block 01: Clock & CONFIG ──
# Single source of truth for every runtime parameter.

import os
import time
import json
from pathlib import Path

ENTRY_MONOTONIC = time.monotonic()


# ── Run mode auto-detection ──
_rerun_raw = os.environ.get("KAGGLE_IS_COMPETITION_RERUN", "").strip().lower()
IS_KAGGLE_RERUN = _rerun_raw in ("1", "true")

MODE = "FULL" if IS_KAGGLE_RERUN else "UNIT_TEST"
N_TASK_UNITTEST = 4
UNITTEST_SEED = 42

_EVAL_DIR = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
if MODE == "UNIT_TEST":
    #_CHALLENGES = f"{_EVAL_DIR}/arc-agi_evaluation_challenges.json" # Use this for real eval challenges
    #_SOLUTIONS  = f"{_EVAL_DIR}/arc-agi_evaluation_solutions.json" # Use this for real eval challenges
    _CHALLENGES = f"{_EVAL_DIR}/arc-agi_test_challenges.json"
    _SOLUTIONS  = None
else:
    _CHALLENGES = f"{_EVAL_DIR}/arc-agi_test_challenges.json"
    _SOLUTIONS  = None

#MODEL_PATH = "/kaggle/input/models/zerabytex/qwen-3-6-35b-a3b-fp8/transformers/default/1"
MODEL_PATH = "/kaggle/input/models/mikedan7/qwen3-8-27b-fp8-official/pytorch/hf-fp8/1"

CONFIG = {
    # ── Execution mode ──
    "ENV": "KAGGLE",
    "execute": True,
    "MODE": MODE,
    "N_TASK_UNITTEST": N_TASK_UNITTEST,
    "UNITTEST_SEED": UNITTEST_SEED,

    # ── Context and output budget ──
    "context_tokens": 262144,
    "max_output_tokens": 65536,
    "buffer_tokens": 512,
    "compression_trigger_ratio": 0.80,
    "compression_max_tokens": 8192*4,

    # ── Time budgets (seconds) ──
    "overall_timeout": 43200,
    "solve_stop_seconds": 41400,
    "finalization_seconds": 1800,
    "request_timeout": 3600,
    "python_timeout": 120,
    "per_task_timeout": 3600*4,
    "max_turns": 256,

    # ── Parallelism ──
    # task_parallel: how many tasks are in flight at once.
    # branches_per_task: how many independent solvers per task, in parallel.
    #   Branch 0 -> attempt_1, Branch 1 -> attempt_2.
    # Peak concurrent model calls = task_parallel * branches_per_task.
    # Peak concurrent sandboxes = same number.
    "task_parallel": 8,
    "branches_per_task": 2,
    "preload_workers": 16,

    # ── Sampling ──
    "temperature": 0.9,
    "top_p": 0.95,
    "presence_penalty": 1.5,
    "seed": 42,

    # ── Model serving (Kaggle) ──
    "model_dir": MODEL_PATH,
    "served_model": "qwen3.8-27b",
    "port": 8000,

    # ── vLLM server flags (Kaggle) ──
    "tensor_parallel_size": 4,
    "gpu_memory_utilization": 0.96,
    "max_model_len": 262144,
    "max_num_seqs": 16,
    "kv_cache_dtype": "fp8",
    "attention_backend": "TRITON_ATTN",
    "speculative_config": {"method": "mtp", "num_speculative_tokens": 3}, # CAN MODULATE THE NUMBER OF TOKENS
    "chat_template_kwargs": {"enable_thinking": True, "preserve_thinking": True}, #CAN CHANGE TO PRESERVE THINKING  True/False

    # ── Offline install (Kaggle) ──
    "wheel_dir": "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3",
    "requirements_path": "/kaggle/input/datasets/mirzamilanfarabi/arc3-vllm-h100-wheelhouse-v3/requirements.lock",

    # ── Data and output paths (MODE-derived) ──
    "challenges": _CHALLENGES,
    "solutions": _SOLUTIONS,
    "submission": "/kaggle/working/submission.json",
    "evidence": "/kaggle/working/arc2-evidence",
    "task_ids": None,
}

CONFIG["prompt_ceiling"] = (
    CONFIG["context_tokens"]
    - CONFIG["max_output_tokens"]
    - CONFIG["buffer_tokens"]
)
CONFIG["compression_trigger"] = int(
    CONFIG["compression_trigger_ratio"] * CONFIG["prompt_ceiling"]
)
CONFIG["peak_concurrent_branches"] = (
    CONFIG["task_parallel"] * CONFIG["branches_per_task"]
)

print(f"ENV               : {CONFIG['ENV']}")
print(f"MODE              : {CONFIG['MODE']}"
      + (f" (KAGGLE_IS_COMPETITION_RERUN={_rerun_raw!r})" if IS_KAGGLE_RERUN else " (dev, no rerun env var)"))
print(f"execute           : {CONFIG['execute']}")
print(f"context_tokens    : {CONFIG['context_tokens']:,}")
print(f"max_output_tokens : {CONFIG['max_output_tokens']:,}")
print(f"prompt_ceiling    : {CONFIG['prompt_ceiling']:,}")
print(f"compression trigger: {CONFIG['compression_trigger']:,} prompt tokens")
print(f"per_task_timeout  : {CONFIG['per_task_timeout']}s")
print(f"max_turns         : {CONFIG['max_turns']}")
print(f"task_parallel     : {CONFIG['task_parallel']}")
print(f"branches_per_task : {CONFIG['branches_per_task']}")
print(f"peak branches     : {CONFIG['peak_concurrent_branches']} (must be <= max_num_seqs={CONFIG['max_num_seqs']})")
print(f"preload_workers   : {CONFIG['preload_workers']}")
print(f"model_dir         : {CONFIG['model_dir']}")
print(f"served_model      : {CONFIG['served_model']}")
print(f"challenges        : {CONFIG['challenges']}")
print(f"solutions         : {CONFIG['solutions']}")
if MODE == "UNIT_TEST":
    print(f"UNIT_TEST tasks   : {CONFIG['N_TASK_UNITTEST']} sampled with seed {CONFIG['UNITTEST_SEED']}")

os.makedirs(CONFIG["evidence"], exist_ok=True)
print(f"\nEvidence dir ready: {CONFIG['evidence']}")

# CELL 3 (markdown)
## Block 02 — Weight preload, vLLM install, server launch

This block starts the vLLM server. Three operations happen in order: preload the
model weights into the OS page cache, install vLLM from the offline wheelhouse
(Kaggle only), and launch the server.

### Why preload

vLLM reads the safetensors files from disk while it initializes. On Kaggle's NVMe
mounts, a cold read of a 27B FP8 model costs several minutes; a warm read from
RAM costs a fraction of that. Reading every model file into the page cache
*before* launching vLLM turns a disk-bound startup into a memory-bandwidth-bound
one.

The preload uses a `ThreadPoolExecutor` with `CONFIG["preload_workers"]` threads
so multiple files are read in parallel. Chunk size is 64 MiB. The loop discards
the bytes it reads — the kernel keeps recently-read file pages resident on its
own — so the preload does not copy anything, does not modify the model files, and
does not need a GPU.

### Step 1 — Preload

Walk `CONFIG["model_dir"]`, collect every regular file, read each one in
parallel, and print the total bytes and elapsed time. A `FileNotFoundError` here
is the right place to catch a missing model mount; catching the same problem
inside vLLM's initialization produces a much less useful error.

### Step 2 — Install vLLM (Kaggle only)

If `vllm` is already importable, skip. Otherwise install from the offline
wheelhouse with `pip install --no-index --find-links <wheel_dir> --requirement
<lock>`. The `--no-index` flag is what makes this work with no network access:
pip is told not to reach for PyPI, only to look inside the wheelhouse. Colab,
which is not offline, skips this step.

The installation prints both stdout and stderr on failure. A silent pip failure
is the most common cause of a server that will not start; the full pip output is
what makes the cause visible.

### Step 3 — Launch vLLM

Build the command from `CONFIG` and start it as a subprocess with
`start_new_session=True`, so the server's process group is its own. That flag is
what lets Block 10 terminate the server and its workers as one group.

The command pins several flags that matter for this specific model:

- `--tensor-parallel-size 4` splits the model across four GPUs. If fewer than
  four are visible, the server fails to start.
- `--kv-cache-dtype fp8` halves the KV cache memory footprint, which is what
  makes 262144 context tokens feasible on 4×L4.
- `--attention-backend TRITON_ATTN` pins a specific attention kernel. The
  alternative on this hardware is less stable with FP8.
- `--enable-prefix-caching` reuses KV state across requests that share a prefix.
  The solver's conversations share long system prompts and task headers, so the
  savings are large.
- `--tool-call-parser qwen3_coder` and `--reasoning-parser qwen3` teach vLLM to
  parse this model's tool calls and reasoning text.
- `--speculative-config {"method": "mtp", "num_speculative_tokens": 6}` enables
  speculative decoding with this model's multi-token prediction head. It is off
  by default because of a known interaction with the attention backend — the fix
  is the environment variable below.

### Environment variables pinned in the subprocess

Two environment variables are set for the server process:

- `VLLM_FORCE_ATTN_BACKEND=TRITON_ATTN` pins **both** the main model and the MTP
  drafter to Triton. The drafter does not inherit `--attention-backend` from the
  CLI. Without this variable the main model runs Triton while the drafter tries
  FlashInfer and dies with an illegal memory access mid-generation — a failure
  mode that is hard to diagnose from the log alone because the main model has
  already started successfully.
- `PYTORCH_ALLOC_CONF=expandable_segments:True` prevents VRAM fragmentation
  across a long session. A solver loop that runs for hours allocates and frees
  many short-lived tensors; without this setting the allocator can fragment
  enough to fail on a request that would otherwise fit.

### Step 4 — Poll for readiness

Poll `GET /v1/models` until the server answers or `max_wait` (900 seconds)
elapses. If the server process dies during startup, its return code and the last
60 lines of its log are printed. This is deliberate: the most common startup
failure is an OOM or a missing wheel, and the last 60 lines of the server log
usually names which one.

### Step 5 — Smoke test

Ask the model `"What is 2+2?"` and check that `"4"` appears in the response.
This catches the case where `/v1/models` answers but the engine crashes on the
first real generation — the known failure mode with FP8 + speculative decoding.
A health check that only calls `/v1/models` passes while the server is broken.

### Expected timing

- Preload: 30–90 seconds on Kaggle NVMe for a ~30 GB model.
- vLLM startup after preload: 90–240 seconds on 4×L4.
- Total from block start to ready: **2–5 minutes**, versus **6–10 minutes**
  without preload.

The preload is why the difference is that large. It is not optional, but it is
also not slow.

# CELL 4 (code)
# ── Block 02: Weight preload + vLLM install (Kaggle) + server launch ──
# 1. Preload every model file into OS page cache (parallel reads, 64 MiB chunks).
# 2. Install vLLM from the mounted wheelhouse (Kaggle, offline).
# 3. Launch the OpenAI-compatible server on CONFIG["port"].
# 4. Poll /v1/models until ready or timeout.
# 5. Run one smoke request to confirm generation works.

import sys
import subprocess
import time
import json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI


# ── Step 1: preload model files into OS page cache ──
def preload_weights():
    """Read every file under CONFIG['model_dir'] into the OS page cache.

    Parallel across files. Uses 64 MiB chunks. Does not copy or modify anything.
    Returns the total number of bytes read.
    """
    model_dir = Path(CONFIG["model_dir"])
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    files = []
    total_bytes = 0
    for root, _, filenames in os.walk(model_dir):
        for name in sorted(filenames):
            fp = Path(root) / name
            if fp.is_file():
                files.append(fp)
                total_bytes += fp.stat().st_size

    if not files:
        raise FileNotFoundError(f"No files found under {model_dir}")

    print(f"Preloading {len(files)} files ({total_bytes/1e9:.2f} GB) from {model_dir} ...")
    t0 = time.monotonic()

    CHUNK = 64 * 1024 * 1024  # 64 MiB

    def _read(path):
        with path.open("rb") as fh:
            while fh.read(CHUNK):
                pass

    with ThreadPoolExecutor(max_workers=CONFIG["preload_workers"]) as ex:
        list(ex.map(_read, files))

    elapsed = time.monotonic() - t0
    print(f"  preload complete in {elapsed:.1f}s "
          f"({total_bytes/1e9/max(elapsed, 1e-6):.2f} GB/s effective)")
    return total_bytes


# ── Step 2: install vLLM from wheelhouse (Kaggle only) ──
def install_vllm_from_wheelhouse():
    try:
        import vllm  # noqa: F401
        print(f"vLLM already installed: {vllm.__version__}")
        return
    except ImportError:
        pass

    if CONFIG["ENV"] != "KAGGLE":
        print("Skipping wheelhouse install (not KAGGLE). Ensure vLLM is available.")
        return

    wheel_dir = CONFIG["wheel_dir"]
    req_path = CONFIG["requirements_path"]
    if not Path(wheel_dir).is_dir():
        raise FileNotFoundError(f"Wheelhouse directory not found: {wheel_dir}")
    if not Path(req_path).is_file():
        raise FileNotFoundError(f"Requirements lock not found: {req_path}")

    print(f"Installing vLLM from wheelhouse: {wheel_dir}")
    result = subprocess.run(
        [sys.executable, "-m", "pip", "install",
         "--no-index",
         "--find-links", wheel_dir,
         "--requirement", req_path],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        print("── pip stdout (tail) ──")
        print(result.stdout[-2000:])
        print("── pip stderr (tail) ──")
        print(result.stderr[-2000:])
        raise RuntimeError("vLLM installation failed")

    import vllm
    print(f"vLLM installed: {vllm.__version__}")


# ── Step 3: build the server command from CONFIG ──
def build_server_command():
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", CONFIG["model_dir"],
        "--served-model-name", CONFIG["served_model"],
        "--tensor-parallel-size", str(CONFIG["tensor_parallel_size"]),
        "--gpu-memory-utilization", str(CONFIG["gpu_memory_utilization"]),
        "--max-model-len", str(CONFIG["max_model_len"]),
        "--max-num-seqs", str(CONFIG["max_num_seqs"]),
        "--kv-cache-dtype", CONFIG["kv_cache_dtype"],
        "--attention-backend", CONFIG["attention_backend"],
        "--host", "0.0.0.0",
        "--port", str(CONFIG["port"]),
        "--enable-prefix-caching",
        "--trust-remote-code",
        "--enable-auto-tool-choice",
        "--no-async-scheduling",
        "--tool-call-parser", "qwen3_coder",
        "--reasoning-parser", "qwen3",
        "--disable-log-stats",
    ]
    if CONFIG.get("speculative_config"):
        cmd.extend([
            "--speculative-config",
            json.dumps(CONFIG["speculative_config"]),
        ])
    return cmd


# ── Server globals (used by Block 10 cleanup) ──
server_proc = None
server_log = None


def launch_server():
    """Launch vLLM and wait for readiness. Returns True on success."""
    global server_proc, server_log

    env = os.environ.copy()
    env["TRANSFORMERS_NO_TF"] = "1"
    env["TRANSFORMERS_NO_FLAX"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
    env["VLLM_DISABLE_PAD_FOR_CUDAGRAPH"] = "1"
    env["VLLM_MEMORY_PROFILER_ESTIMATE_CUDAGRAPHS"] = "1"
    env["VLLM_FORCE_ATTN_BACKEND"] = CONFIG["attention_backend"]

    cmd = build_server_command()
    log_path = "/kaggle/working/vllm.log" if CONFIG["ENV"] == "KAGGLE" else "/content/vllm.log"

    print("Launching vLLM server")
    print(f"  command: {' '.join(cmd)}")
    print(f"  log    : {log_path}")

    server_log = open(log_path, "w")
    server_proc = subprocess.Popen(
        cmd,
        stdout=server_log,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )

    client = OpenAI(
        base_url=f"http://127.0.0.1:{CONFIG['port']}/v1",
        api_key="sk-local",
    )
    t0 = time.monotonic()
    max_wait = 900

    while True:
        if server_proc.poll() is not None:
            print(f"\n❌ Server process exited with code {server_proc.returncode}")
            server_log.flush()
            with open(log_path) as f:
                lines = f.readlines()
            print("── last 60 log lines ──")
            print("".join(lines[-60:]))
            return False

        try:
            client.models.list()
            elapsed = time.monotonic() - t0
            print(f"\n✓ Server ready in {elapsed:.1f}s")
            return True
        except Exception:
            elapsed = time.monotonic() - t0
            if elapsed > max_wait:
                print(f"\n❌ Server not ready after {max_wait}s")
                server_log.flush()
                with open(log_path) as f:
                    lines = f.readlines()
                print("── last 60 log lines ──")
                print("".join(lines[-60:]))
                return False
            if int(elapsed) % 30 == 0 and int(elapsed) > 0:
                print(f"  waiting {int(elapsed)}s ...")
            time.sleep(2)


# ── Step 4: smoke request ──
def smoke_test():
    client = OpenAI(
        base_url=f"http://127.0.0.1:{CONFIG['port']}/v1",
        api_key="sk-local",
    )
    try:
        resp = client.chat.completions.create(
            model=CONFIG["served_model"],
            messages=[{"role": "user", "content": "What is 2+2? Reply with just the number."}],
            max_tokens=64,
            temperature=0.0,
            extra_body={"chat_template_kwargs": CONFIG["chat_template_kwargs"]},
        )
        content = resp.choices[0].message.content or ""
        print(f"  smoke response: {content!r}")
        if "4" in content:
            print("✓ Health check passed")
            return True
        else:
            print("⚠ Health check: response did not contain '4' (non-fatal)")
            return True
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return False


# ── Run in order: preload → install → launch → smoke ──
if CONFIG["execute"]:
    preload_weights()
    install_vllm_from_wheelhouse()
    if launch_server():
        smoke_test()
    else:
        raise RuntimeError("vLLM server failed to start; see log above")
else:
    print("CONFIG['execute'] = False — skipping preload, install, and server launch (dry run)")

# CELL 5 (markdown)
## Block 03 — Task loading and validation

Loads the ARC-AGI-2 challenge file and validates every grid. This block runs
without a GPU and can be tested on its own.

### Why validate at load time

Every later block assumes valid grids. If the challenge file were malformed, the
solver would fail in the middle of a long generation with an obscure `numpy`
error whose traceback points at the solver, not at the data. Catching a
malformed file once, at load time, keeps the failure visible and cheap — you see
exactly which task and which pair is broken, and you see it before a server has
started.

### `strict_json(text)`

Parses JSON but rejects duplicate keys, `NaN`, and `Infinity`. Standard
`json.loads` silently accepts all three. That leniency has produced real bug
reports in ARC pipelines, where a hand-edited file with a duplicated `"train"`
key loaded without error and quietly discarded one of the training pairs. The
strict parser raises instead.

### `validate_grid(grid)`

Returns `True` only if the grid is a rectangular 2-D list of integers 0–9, with
at least one row and one column, and at most 30×30. The checks, in order:

- The grid is a nonempty list.
- Every row is a nonempty list.
- The width of the first row is between 1 and 30.
- The height is between 1 and 30.
- Every row has the same width as the first.
- Every cell is an `int` (not a `bool` — `True` is not a valid color) in the
  range 0–9.

A grid that passes this function is safe to pass to any downstream block.

### `load_challenges(path)`

Reads the file, validates every `train` input/output and every `test` input, and
returns a dict `{task_id: {"train": [...], "test": [...]}}`. The `train` entries
are copied to fresh lists so that no later mutation can affect the loaded data.
Test entries are required to have an `input` key and forbidden from having an
`output` key; a test case with an output is a sign that a hand-edited file has
leaked a solution into the challenge data, and is rejected rather than ignored.

### `load_solutions(path)`

Optional. Used only when scoring locally. The solver never sees the return value
of this function. The block reads the file so the scoring cell can find it, but
nothing in the solve path touches it.

### `preflight_report(challenges)`

Prints the task count, the total number of test inputs, and the smallest and
largest tasks by test count. This is what you look at before starting the
server: if the numbers look wrong, the problem is in the challenge file, not in
the solver.

### What this cell does when run

The cell loads `CONFIG["challenges"]` if the file exists, prints the preflight
report, and stops. It does **not** start the server. If the file is not attached,
it prints a warning and continues so the rest of the notebook can still be
inspected. In `UNIT_TEST` mode it additionally samples `N_TASK_UNITTEST` task IDs
with the configured seed and prints them, so a test run is reproducible.

# CELL 6 (code)
# ── Block 03: Task loading & validation ──
# Pure Python. No GPU, no server, no solver. Testable in isolation.

from pathlib import Path


def strict_json(text: str):
    """Parse JSON, rejecting duplicate keys, NaN, and Infinity."""
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key!r}")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError(f"nonfinite JSON value: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)


def validate_grid(grid) -> bool:
    """Return True iff grid is a rectangular 2-D list of ints in [0, 9], size in [1, 30]."""
    if not isinstance(grid, list) or not grid:
        return False
    if not all(isinstance(row, list) and row for row in grid):
        return False
    width = len(grid[0])
    if not 1 <= width <= 30:
        return False
    if not 1 <= len(grid) <= 30:
        return False
    if any(len(row) != width for row in grid):
        return False
    for row in grid:
        for cell in row:
            if type(cell) is not int:
                return False
            if not 0 <= cell <= 9:
                return False
    return True


def load_challenges(path):
    """Read the challenge file, validate every grid, return {task_id: {train, test}}."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Challenge file not found: {path}")
    data = strict_json(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError("Challenge file must be a non-empty task map")

    tasks = {}
    for task_id, task in data.items():
        if not isinstance(task_id, str) or not task_id.strip():
            raise ValueError(f"Invalid task ID: {task_id!r}")
        if not isinstance(task, dict):
            raise ValueError(f"Task {task_id} is not a dict")
        train = task.get("train")
        test = task.get("test")
        if not isinstance(train, list) or not train:
            raise ValueError(f"Task {task_id} has no train pairs")
        if not isinstance(test, list) or not test:
            raise ValueError(f"Task {task_id} has no test inputs")

        clean_train = []
        for i, pair in enumerate(train):
            if not isinstance(pair, dict) or "input" not in pair or "output" not in pair:
                raise ValueError(f"Task {task_id} train[{i}] missing input/output")
            if not validate_grid(pair["input"]):
                raise ValueError(f"Task {task_id} train[{i}].input invalid")
            if not validate_grid(pair["output"]):
                raise ValueError(f"Task {task_id} train[{i}].output invalid")
            clean_train.append({
                "input": [list(r) for r in pair["input"]],
                "output": [list(r) for r in pair["output"]],
            })

        clean_test = []
        for i, case in enumerate(test):
            if not isinstance(case, dict) or "input" not in case:
                raise ValueError(f"Task {task_id} test[{i}] missing input")
            if "output" in case:
                raise ValueError(f"Task {task_id} test[{i}] unexpectedly contains output")
            if not validate_grid(case["input"]):
                raise ValueError(f"Task {task_id} test[{i}].input invalid")
            clean_test.append({"input": [list(r) for r in case["input"]]})

        tasks[task_id] = {"train": clean_train, "test": clean_test}

    return tasks


def load_solutions(path):
    """Optional. Load solutions for local scoring. Never passed to the solver."""
    if path is None:
        return None
    path = Path(path)
    if not path.is_file():
        return None
    data = strict_json(path.read_text(encoding="utf-8"))
    return data


def preflight_report(challenges):
    n_tasks = len(challenges)
    n_test = sum(len(t["test"]) for t in challenges.values())
    sizes = [(tid, len(t["test"])) for tid, t in challenges.items()]
    sizes.sort(key=lambda x: x[1])
    smallest = sizes[0] if sizes else ("-", 0)
    largest = sizes[-1] if sizes else ("-", 0)
    print(f"Tasks           : {n_tasks}")
    print(f"Test inputs     : {n_test}")
    print(f"Smallest task   : {smallest[0]} ({smallest[1]} test input{'s' if smallest[1] != 1 else ''})")
    print(f"Largest task    : {largest[0]} ({largest[1]} test inputs)")


# ── Test ──
challenges = None
try:
    challenges = load_challenges(CONFIG["challenges"])
    preflight_report(challenges)

    if CONFIG["MODE"] == "UNIT_TEST":
        import random
        _all_ids = sorted(challenges.keys())
        _rng = random.Random(CONFIG["UNITTEST_SEED"])
        _k = min(CONFIG["N_TASK_UNITTEST"], len(_all_ids))
        _sampled = sorted(_rng.sample(_all_ids, _k))
        challenges = {tid: challenges[tid] for tid in _sampled}
        CONFIG["task_ids"] = _sampled
        print(f"\nUNIT_TEST: sampled {_k} of {len(_all_ids)} tasks (seed={CONFIG['UNITTEST_SEED']})")
        for tid in _sampled:
            n_pairs = len(challenges[tid]["train"])
            n_tests = len(challenges[tid]["test"])
            print(f"  {tid}  train={n_pairs}  test={n_tests}")
    else:
        print(f"\nFULL: all {len(challenges)} tasks will be solved")
        print(f"  challenges: {CONFIG['challenges']}")
        print(f"  submission: {CONFIG['submission']}")

except FileNotFoundError as e:
    print(f"⚠ {e}")
    print("  (Attach the ARC-AGI-2 competition input, then re-run this cell.)")

# CELL 7 (markdown)
## Block 04a — The Jupyter sandbox

A wrapper around one Jupyter kernel process. Each task owns exactly one sandbox
for the duration of its solve loop. State persists across `execute()` calls: a
variable defined in turn 3 is readable in turn 40.

### Why a separate kernel and not `exec()`

Running model-generated code with `exec()` inside the notebook kernel would leak
state between tasks, crash the whole notebook on a segfault, and make timeouts
impossible to enforce. A separate Jupyter kernel gives three things `exec()`
cannot:

- **Isolation.** One task's variables cannot contaminate another's. Two tasks
  can both define `transform` and neither overwrites the other.
- **Bounded execution.** When a loop runs away, `interrupt_kernel()` stops it
  after `timeout` seconds and the solver gets an error message rather than a
  hung notebook.
- **Clean shutdown.** Closing the kernel releases its memory and file handles. A
  leaked kernel that is never closed will eventually exhaust the machine's file
  descriptors.

### What the class does

- `__init__(timeout, preload_code)` — allocates five ZMQ ports under a
  class-level lock (so multiple sandboxes created in parallel do not collide),
  starts the kernel, connects a blocking client, waits for the kernel to be
  ready, and runs the preload code once. The port lock is the only shared state
  in the class; everything else is per-instance.
- `execute(code, timeout)` — submits `code`, drains the IOPub channel for
  `stream`, `error`, `execute_result`, and `display_data` messages, and returns
  the concatenated stdout and formatted traceback as a plain string. The
  `status: idle` message is the only reliable signal that the kernel has
  finished; the loop waits for it and does not rely on message counts.
- `close()` — stops the client channels, shuts down the kernel, and cleans up
  resources. All three steps are wrapped in `contextlib.suppress(Exception)` so
  `close` never raises. A failed cleanup is not allowed to mask the original
  failure that triggered it.

### On timeouts

When `execute` times out it interrupts the kernel and returns a string beginning
with `[ERROR] Execution timed out`. The interrupted code may have partially run
— some variables may have been defined before the interrupt. This is
intentional: the solver sees the error, can inspect the workspace, and can
decide what to redo. What the sandbox does **not** do is silently kill the
kernel, because the solver's other variables (the ones defined before the
timeout) are still valuable.

### Thread safety

Each sandbox is owned by exactly one thread at a time. The pool pattern
(`queue.Queue`) is not needed here because each task creates its own sandbox,
but the port allocator is thread-safe so parallel task spawns do not collide on
a port number. Port collisions manifest as a kernel that hangs at startup, not
as an error, so the lock matters more than it might appear.

### Self-test

The cell creates a sandbox, runs one line of code, and closes it. No GPU is
required. If this cell fails, the problem is the Jupyter installation, not the
solver — fix it here rather than debugging it inside a solve loop.

# CELL 8 (code)
# ── Block 04a: JupyterSandbox ──
# One stateful kernel per task. State persists across execute() calls.

import os
import re
import time
import queue
import threading
import contextlib

from jupyter_client import KernelManager


class JupyterSandbox:
    """Persistent Jupyter kernel. One instance per task. Not shared across threads."""

    _port_lock = threading.Lock()
    _next_port = 50000

    @classmethod
    def _get_next_ports(cls, count=5):
        with cls._port_lock:
            ports = list(range(cls._next_port, cls._next_port + count))
            cls._next_port += count
            return ports

    def __init__(self, timeout=60.0, preload_code=""):
        self._timeout = timeout
        ports = self._get_next_ports(5)

        env = os.environ.copy()
        env["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
        env["PYTHONWARNINGS"] = "ignore"
        env["MPLBACKEND"] = "Agg"

        self._km = KernelManager()
        self._km.shell_port = ports[0]
        self._km.iopub_port = ports[1]
        self._km.stdin_port = ports[2]
        self._km.hb_port = ports[3]
        self._km.control_port = ports[4]

        self._km.start_kernel(env=env, extra_arguments=["--Application.log_level=CRITICAL"])
        self._client = self._km.blocking_client()
        self._client.start_channels()
        self._client.wait_for_ready(timeout=self._timeout)

        if preload_code:
            self.execute(preload_code, timeout=self._timeout)

    def execute(self, code, timeout=None):
        """Run code in the kernel. Return stdout + formatted stderr as one string."""
        effective_timeout = timeout or self._timeout
        msg_id = self._client.execute(
            code,
            store_history=True,
            allow_stdin=False,
            stop_on_error=False,
        )
        stdout_parts = []
        stderr_parts = []
        start = time.time()

        while True:
            if time.time() - start > effective_timeout:
                self._km.interrupt_kernel()
                return f"[ERROR] Execution timed out after {effective_timeout:.0f}s"

            try:
                msg = self._client.get_iopub_msg(timeout=1.0)
            except queue.Empty:
                continue

            if msg.get("parent_header", {}).get("msg_id") != msg_id:
                continue

            msg_type = msg.get("msg_type")
            content = msg.get("content", {})

            if msg_type == "stream":
                text = content.get("text", "")
                if content.get("name") == "stdout":
                    stdout_parts.append(text)
                else:
                    stderr_parts.append(text)

            elif msg_type == "error":
                tb = content.get("traceback", [])
                stderr_parts.append(self._format_traceback(tb))

            elif msg_type in ("execute_result", "display_data"):
                text = content.get("data", {}).get("text/plain")
                if text:
                    stdout_parts.append(text if text.endswith("\n") else text + "\n")

            elif msg_type == "status":
                if content.get("execution_state") == "idle":
                    break

        stdout = "".join(stdout_parts)
        stderr = "".join(stderr_parts)

        if stderr:
            return f"{stdout.rstrip()}\n{stderr}" if stdout else stderr
        return stdout if stdout.strip() else "[WARN] No output. Use print() to see results."

    def close(self):
        """Release all kernel resources. Never raises."""
        with contextlib.suppress(Exception):
            if self._client is not None:
                self._client.stop_channels()
        if self._km is not None:
            with contextlib.suppress(Exception):
                self._km.shutdown_kernel(now=True)
            with contextlib.suppress(Exception):
                self._km.cleanup_resources()

    @staticmethod
    def _format_traceback(tb):
        """Strip ANSI escapes, drop non-user frames."""
        clean = []
        for frame in tb:
            frame = re.sub(r"\x1b\[[0-9;]*m", "", frame)
            if 'File "' in frame and "ipython-input" not in frame:
                continue
            clean.append(frame)
        return "".join(clean)


# ── Test (no GPU required) ──
_sb = JupyterSandbox(timeout=30.0)
_out = _sb.execute("x = 42\nprint('hello from kernel')\nprint('x =', x)")
print(_out)
_sb.close()
print("✓ Sandbox self-test complete")

# CELL 9 (markdown)
## Block 04b — Preload helpers and task injection

Every sandbox kernel is initialized with the same preload string. It imports the
standard scientific stack and defines six helper functions that the model is
expected to use during exploration.

### Why preload

The preload serves two purposes. First, it saves tokens: the solver does not
have to write `import numpy as np` in every turn, and the alternative — writing
it once and hoping the solver remembers — fails as soon as a new kernel is
created. Second, it gives the solver a consistent vocabulary. When the solver
sees `verify_all` in the tool description, it knows the helper exists, and it can
call it without writing the whole loop.

### Pre-loaded libraries

- `numpy` as `np` — array arithmetic, slicing, and everything that operates on
  the whole grid at once.
- `scipy.ndimage` — connected-component labeling, which most object-based tasks
  need.
- `sympy` — symbolic algebra. Rarely needed, but the alternative when it is
  needed is writing a CAS.
- `json`, `itertools`, `collections` (with `Counter` and `defaultdict`), `copy`
  (with `deepcopy`) — the standard tools for building and comparing grids.

### Pre-loaded helpers

| Helper | What it does |
|---|---|
| `print_grid(grid, name="")` | Prints rows as space-separated digits with a dimension header. The header is what makes a trace readable. |
| `color_counts(grid)` | Returns `{color: count}` for every color present. The most common first observation in an ARC task. |
| `grid_diff(g1, g2)` | Returns a same-shape grid where `-1` marks unchanged cells and the new value marks changed cells. Returns `None` if the shapes differ. This is what turns "something changed" into "exactly these cells changed". |
| `verify_all(transform_fn, train_pairs)` | Applies `transform_fn` to every training input, prints PASS/FAIL per pair with the first-difference coordinates, and returns `(passed, total)`. This is the workhorse: a candidate law is tested by writing a function and calling this. |
| `find_pattern(grid, template, mask, transforms)` | Finds every position where `template` matches in `grid`, under any subset of the eight dihedral transforms. `mask` is a 0/1 grid of the same shape where 1 means the cell must match and 0 means wildcard. This is what a task with a repeated shape needs. |
| `color_lines(grid, axis)` | Returns per-line color histograms for rows, columns, or either diagonal direction. This is what a task with a border or a divider needs. |

Each helper is a single well-defined operation. None of them solve a task; they
remove a class of boilerplate so the solver can spend its turns on the actual
investigation.

### Pre-loaded data

`inject_task_data(sandbox, task_data)` assigns two variables into the kernel
namespace:

- `train_pairs` — list of `{"input": [[...]], "output": [[...]]}`.
- `test_inputs` — list of test input grids.

Both are Python-native (not JSON strings). The model can index them directly.
There is no test output anywhere in the kernel.

### Self-test

The cell builds a sandbox with the preload string, injects a tiny synthetic task
(a 2×2 swap of colors 0 and 1), runs a `verify_all` call, and confirms the
helper works end to end. No GPU required. If this cell passes, the preload is
sound and the kernel can be trusted inside a real solve.

# CELL 10 (code)
# ── Block 04b: Preload string + task injection ──

SANDBOX_PRELOAD = r'''
import numpy as np
import json
import itertools
import collections
from collections import Counter, defaultdict
from copy import deepcopy
import scipy.ndimage
import sympy


def print_grid(grid, name=""):
    """Print grid with space-separated digits and a dimension header."""
    if name:
        print(f"--- {name} ({len(grid)}x{len(grid[0])}) ---")
    for row in grid:
        print(" ".join(str(c) for c in row))
    print()


def color_counts(grid):
    """Return {color: count} for every color present."""
    return dict(Counter(c for row in grid for c in row))


def grid_diff(g1, g2):
    """Return same-shape diff grid: -1 for unchanged cells, new value for changed. None if shapes differ."""
    a1, a2 = np.array(g1), np.array(g2)
    if a1.shape != a2.shape:
        return None
    diff = np.full_like(a1, -1)
    mask = a1 != a2
    diff[mask] = a2[mask]
    return diff.tolist()


def verify_all(transform_fn, train_pairs):
    """Apply transform_fn to every training input, compare to output. Returns (passed, total)."""
    passed = 0
    for i, pair in enumerate(train_pairs):
        try:
            result = transform_fn(pair["input"])
            expected = pair["output"]
            r, e = np.array(result), np.array(expected)
            if np.array_equal(r, e):
                print(f"Pair {i}: PASS")
                passed += 1
            else:
                if r.shape != e.shape:
                    print(f"Pair {i}: FAIL - shape {r.shape} vs expected {e.shape}")
                else:
                    diffs = np.argwhere(r != e)
                    print(f"Pair {i}: FAIL - {len(diffs)} cells differ")
                    if len(diffs) > 0:
                        row, col = diffs[0]
                        print(f"  First diff at ({row},{col}): got {r[row, col]}, expected {e[row, col]}")
        except Exception as ex:
            print(f"Pair {i}: ERROR - {ex}")
    print(f"\nResult: {passed}/{len(train_pairs)} passed")
    return passed, len(train_pairs)


def find_pattern(grid, template, mask=None, transforms=None):
    """Find all positions where template matches. mask: 1=must-match, 0=wildcard. transforms: subset of 8 dihedral ops."""
    grid = np.array(grid)
    template = np.array(template)
    if mask is not None:
        mask = np.array(mask, dtype=bool)
    else:
        mask = np.ones(template.shape, dtype=bool)
    if transforms is None:
        transforms = ["identity"]

    def _apply(arr, m, name):
        if name == "identity":  return arr, m
        if name == "rot90":     return np.rot90(arr, 1), np.rot90(m, 1)
        if name == "rot180":    return np.rot90(arr, 2), np.rot90(m, 2)
        if name == "rot270":    return np.rot90(arr, 3), np.rot90(m, 3)
        if name == "fliph":     return arr[:, ::-1], m[:, ::-1]
        if name == "flipv":     return arr[::-1, :], m[::-1, :]
        if name == "transpose": return arr.T, m.T
        if name == "anti_t":    return np.flipud(np.fliplr(arr.T)), np.flipud(np.fliplr(m.T))
        raise ValueError(f"Unknown transform: {name}")

    gh, gw = grid.shape
    results = []
    for tname in transforms:
        t, mk = _apply(template, mask, tname)
        th, tw = t.shape
        if th > gh or tw > gw:
            continue
        for r in range(gh - th + 1):
            for c in range(gw - tw + 1):
                if np.all(grid[r:r+th, c:c+tw][mk] == t[mk]):
                    results.append({"row": r, "col": c, "transform": tname})
    return results


def color_lines(grid, axis):
    """Per-line color histograms. axis: 'row' | 'col' | 'diag_down' (\\) | 'diag_up' (/)."""
    g = np.array(grid)
    H, W = g.shape
    results = []
    if axis == "row":
        for r in range(H):
            results.append({"index": r, "start": (r, 0), "length": W,
                            "colors": dict(Counter(int(v) for v in g[r, :]))})
    elif axis == "col":
        for c in range(W):
            results.append({"index": c, "start": (0, c), "length": H,
                            "colors": dict(Counter(int(v) for v in g[:, c]))})
    elif axis == "diag_down":
        idx = 0
        for offset in range(W - 1, -H, -1):
            diag = g.diagonal(offset)
            if len(diag) == 0:
                continue
            start = (0, offset) if offset >= 0 else (-offset, 0)
            results.append({"index": idx, "start": start, "length": len(diag),
                            "colors": dict(Counter(int(v) for v in diag))})
            idx += 1
    elif axis == "diag_up":
        gf = np.fliplr(g)
        idx = 0
        for offset in range(W - 1, -H, -1):
            diag = gf.diagonal(offset)
            if len(diag) == 0:
                continue
            sf = (0, offset) if offset >= 0 else (-offset, 0)
            start = (sf[0], (W - 1) - sf[1])
            results.append({"index": idx, "start": start, "length": len(diag),
                            "colors": dict(Counter(int(v) for v in diag))})
            idx += 1
    else:
        raise ValueError(f"axis must be row/col/diag_down/diag_up, got: {axis}")
    return results
'''


def inject_task_data(sandbox, task_data):
    """Assign train_pairs and test_inputs into the sandbox namespace."""
    train_pairs = task_data["train"]
    test_inputs = [t["input"] for t in task_data["test"]]
    code = (
        f"train_pairs = {json.dumps(train_pairs)}\n"
        f"test_inputs = {json.dumps(test_inputs)}\n"
        "print(f'train_pairs: {len(train_pairs)}, test_inputs: {len(test_inputs)}')\n"
    )
    return sandbox.execute(code)


# ── Test (no GPU required) ──
_sb = JupyterSandbox(timeout=30.0, preload_code=SANDBOX_PRELOAD)

# Synthetic task: swap 0 and 1 in a 2x2 grid
_synthetic_task = {
    "train": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
    ],
    "test": [{"input": [[0, 1], [0, 1]]}],
}
print(inject_task_data(_sb, _synthetic_task))

_test_code = '''
def swap01(g):
    return [[1 - c for c in row] for row in g]

print_grid(train_pairs[0]["input"], "train[0].input")
verify_all(swap01, train_pairs)
'''
print(_sb.execute(_test_code))
_sb.close()
print("✓ Preload + injection self-test complete")

# CELL 11 (markdown)
## Block 05 — Tools and prompts

Two tools, one system prompt, one user-prompt builder.

### The two tools

**`python`** — the primary tool. Executes code in the task's persistent Jupyter
sandbox. The sandbox pre-loads `numpy`, `scipy.ndimage`, `sympy`, and the
standard library helpers, plus the task data as `train_pairs` and `test_inputs`.
The trace records every call and its complete output, including tracebacks.

**`cognitive_activity`** — the explorer's lab notebook. The model supplies a
freeform `activity` label and its `content`. The label vocabulary is open:
`observation`, `hypothesis`, `dead end`, `revision`, `self_doubt`,
`blindspot_check`, `thinking`, `plan` — anything the model finds worth
recording. The `preserve_across_reincarnation` boolean marks an entry to survive
context compression. When the conversation resets, preserved entries are
re-inserted into the new conversation as remembered lab notes. Unpreserved
entries remain archived only.

There is no separate `thinking` tool. Prose reasoning that does not belong in
code goes into a `cognitive_activity` entry with a label like `"thinking"` or
`"plan"`. One tool, one record format, one trace stream. The reason for merging
the two is that a separate `thinking` tool would create two parallel narratives
— one labeled as reasoning, one labeled as cognition — and the trace would be
harder to read. One stream is easier to reconstruct.

### The system prompt

Frames each task as a **mini-world**: the training pairs are observations of a
hidden transformation law, the test inputs ask the explorer to predict what that
law produces.

The prompt suggests a scientific discovery loop:

1. **Observe.** Log exact dimensions, colors, changed cells.
2. **Hypothesize.** State a candidate law.
3. **Experiment.** Implement it in Python, compare against every training pair.
4. **Revise.** Keep contradicted hypotheses in the notebook; update the theory.
5. **Predict.** Apply the investigated law.

The labels in bold are examples the model may use for `cognitive_activity`. They
are not required and no quota is enforced. The prompt is a suggestion, not a
procedure; the model is expected to adapt it to the task.

### The Cognitive Enhancement Principles

Twelve principles, adapted from contemplative practice, that give practical
guidance for managing fixation, questioning assumptions, and returning to the
evidence. They are grouped into three families:

- **Regulative** (managing raw energy) — treat uncertainty as a signal;
  separate the wish to finish from evidence that a rule works; change the view
  when effort stops producing information; investigate without protecting a
  favored outcome.
- **Deconstructive** (removing distortion) — use methods that fit the task;
  prefer exact observations to persuasive stories; treat role assignments such
  as "this color is background" as experiments, not identity; separate your
  identity from a claim.
- **Anchoring** (stability and evolution) — keep accuracy, evidence, and honest
  uncertainty as decision anchors; return to the task after errors; accumulate
  small informative corrections; return to the aim of explaining the training
  pairs and predicting every test.

The principles do not prescribe a sequence. They give the model a vocabulary for
noticing when it is stuck and a set of moves for getting unstuck.

### `build_user_prompt(task_data)`

Renders one task as plain text: every training pair (input grid, output grid)
followed by every test input. Uses the compact digit-string format (`0123`)
rather than JSON. The compact format is both shorter and easier for the model to
scan for patterns; JSON's quotes and commas add noise without adding
information.

### `build_initial_messages(task_data)`

Returns `[system_message, user_message]`. The user message contains the complete
task. Nothing is truncated. If the task were too large to fit, the failure would
be explicit — the notebook does not silently drop training pairs.

### Self-test

The cell builds messages for a tiny synthetic task and asserts that every
training pair and every test input appears in the rendered prompt. This catches
the one bug that would make the whole notebook pointless: a prompt that quietly
omits part of the task.

# CELL 12 (code)
# ── Block 05: Tools and prompts ──
# Two tools, one system prompt grounded in the Mini Standard Model framing,
# one user-prompt builder that injects the complete task.


TOOL_PYTHON = {
    "type": "function",
    "function": {
        "name": "python",
        "description": (
            "Execute Python in your persistent Jupyter workspace. Variables, functions, "
            "and imports survive across calls.\n\n"
            "Pre-loaded libraries (no import needed): numpy (as np), scipy.ndimage, sympy, "
            "json, itertools, collections (Counter, defaultdict), copy (deepcopy).\n\n"
            "Pre-loaded data:\n"
            "  train_pairs : list of {'input': [[int]], 'output': [[int]]}\n"
            "  test_inputs : list of [[int]]\n\n"
            "Pre-loaded helpers:\n"
            "  print_grid(grid, name='')       - print grid with dimension header\n"
            "  color_counts(grid)              - {color: count}\n"
            "  grid_diff(g1, g2)               - -1 for unchanged cells, new value for changed; None if shapes differ\n"
            "  verify_all(fn, train_pairs)     - apply fn to each train input, print PASS/FAIL, return (passed, total)\n"
            "  find_pattern(grid, template, mask=None, transforms=None) - positions of template in grid\n"
            "  color_lines(grid, axis)         - per-line color histograms; axis in {'row','col','diag_down','diag_up'}\n\n"
            "Always use print() to see output. Default timeout 30s, max 120s."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {"type": "string", "description": "Python code to execute"},
                "timeout": {"type": "number", "description": "Max seconds (default 30, max 120)"},
            },
            "required": ["code"],
        },
    },
}


TOOL_COGNITIVE = {
    "type": "function",
    "function": {
        "name": "cognitive_activity",
        "description": (
            "Record an entry in your lab notebook. Every entry is archived in the trace.\n\n"
            "The activity label is freeform. Use the label that describes what you are "
            "actually doing — 'observation', 'hypothesis', 'experiment', 'revision', "
            "'dead_end', 'self_doubt', 'blindspot_check', 'assumption_inventory', "
            "'counterexample_search', 'thinking', 'plan' are all useful. You are not "
            "limited to them.\n\n"
            "Set preserve_across_reincarnation=true for entries you want to survive a "
            "context compression. Your conversation may be reset if it grows long; "
            "preserved entries are re-inserted as remembered lab notes, unpreserved "
            "entries remain in the archive only.\n\n"
            "Returns storage metadata only. Does not call another model."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "activity": {"type": "string", "description": "Freeform label for this entry"},
                "content": {"type": "string", "description": "Full content of the entry"},
                "preserve_across_reincarnation": {
                    "type": "boolean",
                    "description": "True to keep this entry after a context reset",
                },
            },
            "required": ["activity", "content", "preserve_across_reincarnation"],
        },
    },
}


ALL_TOOLS = [TOOL_PYTHON, TOOL_COGNITIVE]


COGNITIVE_PRINCIPLES = """\
COGNITIVE ENHANCEMENT PRINCIPLES

Regulative — managing raw energy
1. Management of Fear: treat uncertainty as a signal. Name the risk and check it.
2. Management of Desire: separate the wish to finish from evidence that a rule works.
3. Non-Force Principle: when effort stops producing information, change the view or the experiment.
4. Non-Attachment: investigate fully without protecting a favored outcome.

Deconstructive — removing distortion
5. Deconstruction of Methodology: use methods that fit the task; question a familiar method when it fails.
6. Deconstruction of Illusions: prefer exact observations to repetition, confidence, or a persuasive story.
7. Deconstruction of Attachment: notice "I must" or "I cannot" bindings to roles. Treat assignments such as "this color is background" as experiments, not identity.
8. Deconstruction of Ego: separate your identity from a claim; expose mistakes without self-protection.

Anchoring — stability and evolution
9. Strategic Anchoring in Core Values: keep accuracy, evidence, and honest uncertainty as decision anchors.
10. Perseverance Against Temptation and Chaos: return to the task after errors; choose the next useful check.
11. Infinitesimal Effort Principle: accumulate small, informative observations and corrections.
12. Original Intention as Methodology: return to the aim of explaining the training pairs and predicting every test.
"""


SYSTEM_PROMPT = """\
# EXPLORER OF MINI REALITIES

You are an explorer arriving at an unfamiliar mini-reality.

Each ARC-AGI-2 task is such a mini-reality: a small, self-consistent world whose
physics acts on a two-dimensional grid of colored cells. You have never seen its
rules. What you have is a handful of observations — the training pairs — each
showing the mini-reality in one configuration and again after its physics has
acted. Then you are shown inputs you have not seen, and asked to predict what the
mini-reality will do to them.

Your task is not to recall a familiar pattern. ARC-AGI-2 tasks are constructed
precisely so that no memorized template suffices. Your task is to reconstruct the
mini-reality's transformation law from the evidence available and to use it.

# MINI STANDARD MODEL

Your current best explanation of a mini-reality is its Mini Standard Model (MSM).
An MSM is not a grid and not a formula. It is a written theory:

  - the entities the world appears to contain, and their roles,
  - the relationships between them,
  - the transformation law that connects an input to an output,
  - the observations that support the law,
  - the rival laws you have not yet ruled out,
  - the uncertainties that remain.

Write the MSM down as `cognitive_activity` entries. Revise it as evidence
accumulates. Do not treat an earlier MSM as an identity to defend — treat it as
the best available hypothesis, subject to replacement when a sharper observation
appears. When two laws explain the training pairs equally well, keep both visible
until an experiment distinguishes them. A law that fits every training pair is a
supported hypothesis, not a certainty.

# DISCOVERY BY ACCUMULATION OF OBSERVATION

You will not understand this world by looking once. You will understand it by
accumulating precise observations, each one narrowing the space of laws that
could have produced what you see.

1. Observe exactly.
   Print dimensions, color counts, changed cells, repeated structures. Separate
   the measurement from the interpretation. A cell that changed from 3 to 7 is a
   fact; the reason it changed is a hypothesis.

2. Deconstruct.
   Ask which distinctions the world appears to make. Is position meaningful?
   Color? Connectivity? Boundaries? The shape of the negative space between
   objects? Do not commit to the first familiar interpretation. A color you call
   "background" might be an actor. A rectangle you call an object might be a
   constraint.

3. Propose a law.
   State a candidate transformation that would produce the observed outputs from
   the observed inputs. A useful law tells you how to generate the output. A
   description of what the output resembles is not a law.

4. Experiment.
   Implement the candidate in Python and apply it to every training pair. Read
   the actual PASS/FAIL results, not the ones you expected. When a pair fails,
   find the specific cells where the law diverges and inspect what the divergence
   reveals. A failed experiment is more informative than a lucky success — it
   eliminates a law.

5. Revise.
   Update the MSM. Keep the contradicted version visible in the notebook,
   because the reason a law failed is itself evidence about the world.

6. Predict.
   Apply the surviving law to each test input. State the remaining uncertainty.

# YOUR TOOLS

You have two tools.

  python              — a persistent Jupyter workspace. Variables, functions, and
                        imports survive across calls. Pre-loaded data: train_pairs
                        and test_inputs. Pre-loaded helpers: print_grid,
                        color_counts, grid_diff, verify_all, find_pattern,
                        color_lines. Use print() to see results.

  cognitive_activity  — your lab notebook. Every entry is archived in the trace.
                        The activity label is freeform. Set
                        preserve_across_reincarnation=true for entries you want
                        to survive a context compression.

Write to the notebook often. A scientific discovery that is not written down
cannot be revisited, challenged, or refined. The notebook is how the MSM
accumulates.

# ON DOUBT AND REVISION

You will be tempted to protect an early hypothesis. This is the most common
failure mode. When a hypothesis keeps failing, question the framework it assumed,
not just its parameters. A law can be wrong in its parts and wrong in its shape.
Revise from the evidence, not from attachment to the previous theory.

You may also be tempted to declare a solution too early. A law that fits the
training pairs but has not been examined for edge cases may be a coincidence.
Before predicting, verify against the least convenient training pair, not the
easiest one.

""" + COGNITIVE_PRINCIPLES + """

# WHEN YOU ARE READY

When you are confident in your MSM, emit your answer as:

    SOLUTION: \\boxed{[[row1], [row2], ...]}

If the mini-reality has more than one test input, label them in order:

    SOLUTION_1: \\boxed{[[row1], [row2], ...]}
    SOLUTION_2: \\boxed{[[row1], [row2], ...]}

Before emitting the answer, write a final cognitive_activity entry that states
your surviving MSM in one paragraph and names the uncertainty that remains.
Training agreement is evidence, not proof.
"""


def _render_grid(grid):
    """One line per row, digits concatenated with no separator."""
    return ["".join(str(int(c)) for c in row) for row in grid]


def build_user_prompt(task_data):
    """Render one mini-reality as plain text: every training pair, then every test input."""
    lines = []

    for i, pair in enumerate(task_data["train"]):
        inp = pair["input"]
        out = pair["output"]
        lines.append(f"train[{i}].input ({len(inp)}x{len(inp[0])}):")
        lines.extend(_render_grid(inp))
        lines.append(f"train[{i}].output ({len(out)}x{len(out[0])}):")
        lines.extend(_render_grid(out))
        lines.append("")

    for i, case in enumerate(task_data["test"]):
        inp = case["input"]
        lines.append(f"test[{i}].input ({len(inp)}x{len(inp[0])}):")
        lines.extend(_render_grid(inp))
        lines.append("")

    lines.append("The same transformation law explains every training pair.")
    lines.append("Predict the output for each test input.")

    n_tests = len(task_data["test"])
    if n_tests == 1:
        lines.append("Emit your answer as:")
        lines.append("    SOLUTION: \\boxed{[[row1], [row2], ...]}")
    else:
        lines.append(f"There are {n_tests} test inputs. Emit one solution per test input, in order:")
        for i in range(n_tests):
            lines.append(f"    SOLUTION_{i+1}: \\boxed{{[[row1], [row2], ...]}}")
    return "\n".join(lines)


def build_initial_messages(task_data):
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(task_data)},
    ]


# ── Self-test ──
_synthetic = {
    "train": [
        {"input": [[0, 1], [1, 0]], "output": [[1, 0], [0, 1]]},
        {"input": [[0, 0], [1, 1]], "output": [[1, 1], [0, 0]]},
    ],
    "test": [{"input": [[0, 0], [1, 1]]}],
}
_msgs = build_initial_messages(_synthetic)
print(f"✓ Tools: {[t['function']['name'] for t in ALL_TOOLS]}")
print(f"✓ System prompt length: {len(SYSTEM_PROMPT)} chars")
print(f"✓ User prompt length: {len(_msgs[1]['content'])} chars")

_n_train = len(_synthetic["train"])
_n_test = len(_synthetic["test"])
_prompt = _msgs[1]["content"]
for i in range(_n_train):
    assert f"train[{i}].input" in _prompt
    assert f"train[{i}].output" in _prompt
for i in range(_n_test):
    assert f"test[{i}].input" in _prompt
print(f"✓ Injection verified: {_n_train} train pairs + {_n_test} test input(s) present")

# CELL 13 (code)


# CELL 14 (markdown)
## Block 06a — Token counter

The solver must know, *before* every request, whether the conversation still
fits inside the context window with room for the maximum output. That number
cannot be guessed: it depends on the model's tokenizer, the chat template, and
the tool schemas attached to the request.

### Why not `len(text) // 3`

Character-count heuristics routinely underestimate by 2–3× on grids with long
digit runs, and the error compounds across turns. A conversation that is 100,000
characters might be 30,000 tokens at the start and 90,000 by turn 20, because the
grids get denser. The only reliable count comes from the tokenizer that will
actually run on the server.

### Why the server endpoint

Loading a tokenizer in the notebook would mean `transformers` must be importable
here, and it would mean the notebook's copy of the tokenizer must be the same one
the server uses. Both conditions can fail silently. Instead, this block posts to
the server's `/v1/chat/completions/render` endpoint, which uses exactly the
tokenizer and chat template the server will use for inference. The count is
directly comparable by construction.

### Functions

- `count_tokens(messages, tools=None)` — posts the messages and any tool schemas
  to the render endpoint and returns the length of the returned `token_ids`. If
  the endpoint is unavailable (older vLLM), it falls back to summing character
  lengths with a per-message overhead constant and a conservative tool-schema
  constant. The fallback over-counts slightly; over-counting is the safe
  direction, because it triggers compression earlier rather than later.
- `compute_max_tokens(messages, tools=None)` — returns the output allowance for
  the next request: `context_tokens − prompt_tokens − buffer_tokens`, clamped to
  `max_output_tokens`. The buffer is the room the notebook keeps for the next
  request's own overhead.
- `should_compress(messages, tools=None)` — returns `True` when the prompt
  crosses `compression_trigger`.

### Self-test

The cell counts a small prompt and asserts that the result is positive and that
`compute_max_tokens` returns `max_output_tokens` for a small prompt. The second
assertion is the important one: it confirms that the ceiling is being computed
correctly, not just that the counter runs.

# CELL 15 (code)
# ── Block 06a (fallback): token counter via vLLM server endpoint ──
# Avoids importing transformers in the notebook kernel. Counts are still exact —
# the server uses the same tokenizer it will use for inference.

import urllib.request
import urllib.error
import json


def _server_url(path):
    return f"http://127.0.0.1:{CONFIG['port']}{path}"


def _post_json(url, payload, timeout=30):
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# Cache: rendered prompt string -> token count.
_prompt_token_cache = {}


def count_tokens(messages, tools=None):
    """Count prompt tokens via the server's /tokenize endpoint."""
    # Render the chat template client-side is not possible without transformers.
    # Instead, we approximate by counting the raw text of every message plus a
    # per-message overhead. This is conservative (over-counts slightly) and never
    # under-counts, which is the safe direction for compression triggering.
    #
    # If the server exposes /v1/chat/completions/render (vLLM >= 0.20), prefer it;
    # otherwise fall back to the message-text sum.
    try:
        payload = {
            "model": CONFIG["served_model"],
            "messages": messages,
            "max_tokens": 1,  # required by the schema; ignored by render endpoint
        }
        if tools:
            payload["tools"] = tools
        rendered = _post_json(_server_url("/v1/chat/completions/render"), payload)
        ids = rendered.get("token_ids") or rendered.get("prompt_token_ids")
        if isinstance(ids, list):
            return len(ids)
    except (urllib.error.HTTPError, urllib.error.URLError, KeyError, ValueError):
        pass  # render endpoint unavailable; fall through

    # Fallback: sum text lengths with a per-message overhead constant.
    # chars/3 is a safe over-estimate for Qwen's BPE on ASCII digit runs.
    total = 0
    for msg in messages:
        content = msg.get("content") or ""
        if isinstance(content, list):
            content = " ".join(
                part.get("text", "") for part in content if isinstance(part, dict)
            )
        total += len(content) // 3 + 16  # 16 tokens/message for role + delimiters
    if tools:
        total += 800  # conservative tool schema constant
    return total


def compute_max_tokens(messages, tools=None):
    prompt_tokens = count_tokens(messages, tools=tools)
    remaining = CONFIG["context_tokens"] - prompt_tokens - CONFIG["buffer_tokens"]
    return max(1, min(CONFIG["max_output_tokens"], remaining))


def should_compress(messages, tools=None):
    return count_tokens(messages, tools=tools) >= CONFIG["compression_trigger"]


# ── Self-test ──
_test_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is 2 + 2?"},
]
_n = count_tokens(_test_messages)
_m = compute_max_tokens(_test_messages)
print(f"test prompt tokens : {_n}")
print(f"test max_tokens    : {_m:,}")
print(f"compression trigger: {CONFIG['compression_trigger']:,}")
assert _n > 0, "token count must be positive"
assert _m == CONFIG["max_output_tokens"], f"unexpected max_tokens: {_m}"
print("✓ Token counter (server-based) self-test passed")

# CELL 16 (markdown)
## Block 06b — Model caller and compression

### The model caller

`call_model()` is the single HTTP entry point used by both the solver loop and
the compression step. Every request goes through it.

- Builds the request body from `CONFIG` — model name, sampling parameters,
  `chat_template_kwargs`.
- Attaches `tools` and `tool_choice="auto"` only when tools are supplied. The
  compression request deliberately supplies none, so the model writes a summary
  rather than attempting a new tool call.
- Clamps the HTTP timeout to the remaining deadline. A request started near the
  end of the run cannot outlive the run: the timeout is `min(request_timeout,
  deadline − now)`.
- Returns the raw `ChatCompletion` object; the caller reads
  `response.choices[0].message`.

The OpenAI Python client is cached in a module-level variable so the underlying
HTTP connection pool is reused across turns. Rebuilding the client per request
would discard the pool and pay TCP setup on every call.

### The compression step

When `should_compress()` fires, the current conversation is replaced by a
**handoff conversation**:

1. The model is asked to write a handoff summary. The prompt is explicit: the
   Python kernel state will survive, only the conversation will reset. The
   summary must include the current hypothesis, verified facts, approaches tried
   and why they failed, the best guesses so far for each test output, the exact
   variable names that carry useful state, and the single next step.
2. The summary is captured as plain text.
3. A fresh conversation is built from:
   - the original system prompt,
   - the original task user message (unchanged — the task is still the same),
   - a block containing every cognitive activity whose
     `preserve_across_reincarnation` flag was set,
   - a final user message containing the handoff summary and an instruction to
     continue.

The Python sandbox is **not** touched. Variables, functions, and imports remain
live; the next assistant turn can use them directly. The handoff prompt says so
explicitly, because a model that thinks it is starting over will redefine its
variables and lose the benefit of everything it did.

### Why not truncate

Truncating older messages silently destroys evidence. The model cannot tell that
a fact it "remembers" is no longer in the prompt. Asking the model to write the
handoff forces a real compression decision — what matters, what does not — and
the resulting summary is saved to disk as part of the trace, so a human reviewer
can check what was preserved and what was dropped. Truncation has no such
record.

### Failure handling

If the compression request returns an empty summary, the reset still proceeds
but the handoff text records the failure explicitly. If the request itself
errors (network, deadline), the caller sees the exception and can decide to stop
the task rather than risk an unbounded prompt. Both failures leave the original
conversation intact in the evidence directory.

# CELL 17 (code)
# ── Block 06b: Model caller + compression ──

from openai import OpenAI


_client_cache = None


def get_client():
    global _client_cache
    if _client_cache is None:
        _client_cache = OpenAI(
            base_url=f"http://127.0.0.1:{CONFIG['port']}/v1",
            api_key="sk-local",
            timeout=CONFIG["request_timeout"],
        )
    return _client_cache


def call_model(messages, tools=None, max_tokens=None, temperature=None,
               deadline=None, seed=None):
    """One chat completion request. Returns the raw response object."""
    client = get_client()

    if max_tokens is None:
        max_tokens = compute_max_tokens(messages, tools=tools)
    if temperature is None:
        temperature = CONFIG["temperature"]
    if seed is None:
        seed = CONFIG["seed"]

    timeout = CONFIG["request_timeout"]
    if deadline is not None:
        remaining = deadline - time.monotonic()
        timeout = max(5.0, min(timeout, remaining))

    kwargs = dict(
        model=CONFIG["served_model"],
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=CONFIG["top_p"],
        presence_penalty=CONFIG["presence_penalty"],
        seed=seed,
        extra_body={"chat_template_kwargs": CONFIG["chat_template_kwargs"]},
        timeout=timeout,
    )
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "auto"

    return client.chat.completions.create(**kwargs)


COMPRESSION_PROMPT = """\
Your context window is nearly full. Write a handoff summary for the next instance of yourself.

The Python kernel state — every variable, function, and import you defined — WILL BE PRESERVED. Only the conversation history will be reset. You will be able to call the same functions and read the same variables.

Your summary MUST include, in this order:

1. Current hypothesis about the transformation law (as precise as you can state it).
2. Facts verified with code — exact values, not paraphrases.
3. Approaches tried and why each failed.
4. Best guess for each test output so far (with the reasoning behind it).
5. Exact Python variable names that carry useful state.
6. The single next step you would take.

Respond with plain text only. No tool calls, no code blocks, no LaTeX. This is a handoff document, not a new attempt.
"""


def compress_conversation(messages, preserved_activities, task_user_prompt, deadline):
    """Ask the model for a handoff summary, then rebuild the conversation.

    Returns (new_messages, summary).
    """
    request_messages = list(messages) + [{"role": "user", "content": COMPRESSION_PROMPT}]

    response = call_model(
        request_messages,
        tools=None,
        max_tokens=CONFIG["compression_max_tokens"],
        temperature=0.5,
        deadline=deadline,
    )
    summary = (response.choices[0].message.content or "").strip()
    if not summary:
        summary = "[compression summary was empty; model returned no text]"
        print("  [compression] WARNING: empty summary")

    new_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_user_prompt},
    ]

    if preserved_activities:
        block_lines = ["[PRESERVED COGNITIVE ACTIVITIES FROM PREVIOUS LIFE]", ""]
        for act in preserved_activities:
            block_lines.append(f"--- {act['activity']} ---")
            block_lines.append(act["content"])
            block_lines.append("")
        new_messages.append({"role": "user", "content": "\n".join(block_lines)})

    new_messages.append({
        "role": "user",
        "content": (
            "[HANDOFF FROM PREVIOUS LIFE]\n\n"
            f"{summary}\n\n"
            "Continue from here. The Python kernel state is preserved — do not redefine "
            "existing variables. The task and its solution format are unchanged."
        ),
    })

    return new_messages, summary


print("✓ Model caller and compression ready")

# CELL 18 (markdown)
## Block 07 — Solver loop

This is where a task actually gets solved. One `solve_branch()` call drives a
full multi-turn conversation against the vLLM server, using the persistent
sandbox from Block 04, the tools from Block 05, and the token counter and
compression from Block 06.

### Why "branch" and not "task"

Each task is solved by **two independent branches**, running in parallel. Branch
0 supplies `attempt_1`; branch 1 supplies `attempt_2`. The two branches share
nothing — different seeds, different sampling, different sandboxes, different
conversations. If they agree on a prediction, that is a weak form of evidence.
If they disagree, one of them is wrong, and the disagreement is itself a signal
that the task was hard.

The alternative — a single branch that runs, then a "verification round" that
asks the same conversation to check itself — is cheaper but weaker. A single
conversation that has already committed to a wrong law is unlikely to abandon
it. An independent branch has no commitment to abandon.

### Per-branch state (`BranchState`)

A small dataclass holds everything that belongs to one branch:

- `task_id`, `branch_index`, `branch_dir` — where the trace will be written.
- `live_messages` — the conversation currently being sent to the model.
- `history` — append-only record of everything that has passed through,
  including compression events. `history` differs from `live_messages` because
  compression replaces `live_messages` but `history` keeps the pre-compression
  conversation.
- `preserved_activities` — every `cognitive_activity` whose flag was set.
  Survives compression.
- `solution_history` — a dict `{test_index: [grid, grid, ...]}` recording every
  solution the model has emitted, per test input. Keeping every emission, not
  just the latest, is what makes it possible to see how the model's answer
  evolved.
- `sandbox_log`, `cognitive_log` — separate evidence streams.
- `compression_events` — every compression event with its summary.
- `turn` — the current turn number.

`state.save()` writes every stream to
`CONFIG["evidence"]/<task_id>/branch_<i>/` as JSON files so a reviewer can
reconstruct exactly what happened.

### The main loop

For each turn, in order:

1. **Deadline check.** If `time.monotonic() > deadline`, break.
2. **Compression check.** If `should_compress(live_messages, tools=ALL_TOOLS)`,
   call `compress_conversation()`. The live conversation is replaced; the
   sandbox is not.
3. **Model call.** `call_model(live_messages, tools=ALL_TOOLS,
   deadline=deadline)`.
4. **Append assistant message.** Both `live_messages` and `history` get the
   assistant turn verbatim.
5. **Dispatch tool calls, in order.** Each tool call produces a `{"role":
   "tool", "tool_call_id": ..., "content": ...}` message appended to both lists.
   Tool calls are executed sequentially because a later call may depend on an
   earlier one — a variable defined by `python` in call 1 might be used by
   `python` in call 2. Parallel tool dispatch would be faster but would break
   that dependency.
6. **If no tool calls: check for solutions.** The assistant text is scanned for
   `SOLUTION:` (single test) or `SOLUTION_i:` (multi-test). Any found grids are
   appended to `state.solution_history[i]`.
7. **Acceptance check.** If every test input has at least one solution in
   `solution_history`, the last solution for each test becomes the branch's
   `final_solutions`, and the branch breaks.
8. **Partial-solution log.** If only some tests have solutions, the branch
   prints the per-test counts and continues.
9. **Empty response nudge.** If the model returned no tool calls and no text,
   append a short "continue" message so the loop does not stall.

The order matters. The compression check runs before the model call because the
model call cannot be made on a prompt that does not fit. The deadline check runs
before the compression check because compression is expensive and there is no
point starting it if the deadline has already passed.

### Multi-test tasks

The task user prompt instructs the model to emit one solution per test input,
labelled `SOLUTION_1:`, `SOLUTION_2:`, etc. The extractor reads the labels and
maps each grid to the correct test index. A solution with no label goes to the
first empty slot; a solution with an explicit label goes to that slot. Two
solutions for the same slot: the first one wins.

### Solution extraction

The regex accepts two forms:

- Single-test: `SOLUTION: \boxed{[[...]]}` or `SOLUTION_1: \boxed{[[...]]}`.
- Multi-test: `SOLUTION_1: \boxed{[[...]]}`, `SOLUTION_2: \boxed{[[...]]}`, …
  with an explicit index.

Grids are validated with `validate_grid` before being accepted. A malformed grid
is ignored — not treated as a solution and not allowed to overwrite a valid one.
The model gets a chance to correct its format in the next turn.

### Trace persistence

Before the sandbox closes, `state.save()` writes:

- `messages.json` — the full conversation (compressed and post-compression
  messages, plus markers).
- `solutions.json` — per-test solution history.
- `sandbox.jsonl` — every Python call and its output.
- `cognitive.jsonl` — every `cognitive_activity` tool call.
- `compression_events.json` — every compression event with its summary.

This is the evidence a reviewer reads to judge whether a solution is grounded or
lucky.

# CELL 19 (code)
# ── Block 07: Solver — one branch ──

import re
import uuid
import json
import time
from dataclasses import dataclass, field
from pathlib import Path


def truncate_output(text, max_chars=8192):
    if len(text) <= max_chars:
        return text
    half = max_chars // 2
    return (
        text[:half]
        + f"\n\n... [truncated {len(text) - max_chars} chars] ...\n\n"
        + text[-half:]
    )


# ── Solution extraction ──
_SOLUTION_UNINDEXED_RE = re.compile(
    r"(?<![\w])SOLUTION(?!_\d)\s*:\s*(?:\\boxed\s*\{+)?\s*(\[\[[\d\s,\[\]]+\]\])\}*"
)
_SOLUTION_INDEXED_RE = re.compile(
    r"SOLUTION_(\d+)\s*:\s*(?:\\boxed\s*\{+)?\s*(\[\[[\d\s,\[\]]+\]\])\}*"
)


def _try_parse_grid(s):
    try:
        grid = json.loads(s)
    except json.JSONDecodeError:
        return None
    return grid if validate_grid(grid) else None


def extract_all_solutions(text, n_tests):
    results = [None] * n_tests
    for m in _SOLUTION_INDEXED_RE.finditer(text):
        idx = int(m.group(1)) - 1
        if 0 <= idx < n_tests and results[idx] is None:
            grid = _try_parse_grid(m.group(2))
            if grid is not None:
                results[idx] = grid
    for m in _SOLUTION_UNINDEXED_RE.finditer(text):
        slot = next((i for i, g in enumerate(results) if g is None), None)
        if slot is None:
            break
        grid = _try_parse_grid(m.group(1))
        if grid is not None:
            results[slot] = grid
    return results


# ── Per-branch state ──
@dataclass
class BranchState:
    task_id: str
    branch_index: int
    branch_dir: Path
    live_messages: list = field(default_factory=list)
    history: list = field(default_factory=list)
    preserved_activities: list = field(default_factory=list)
    solution_history: dict = field(default_factory=dict)
    sandbox_log: list = field(default_factory=list)
    cognitive_log: list = field(default_factory=list)
    compression_events: list = field(default_factory=list)
    turn: int = 0

    def append_message(self, msg):
        self.live_messages.append(msg)
        self.history.append(msg)

    def append_sandbox_log(self, turn, code, output):
        self.sandbox_log.append({"turn": turn, "code": code, "output": output})

    def append_cognitive(self, turn, activity, content, preserve):
        event = {
            "id": uuid.uuid4().hex,
            "turn": turn,
            "activity": activity,
            "content": content,
            "preserve_across_reincarnation": bool(preserve),
        }
        self.cognitive_log.append(event)
        if preserve:
            self.preserved_activities.append(event)
        return event

    def save(self):
        self.branch_dir.mkdir(parents=True, exist_ok=True)
        (self.branch_dir / "messages.json").write_text(
            json.dumps(self.history, indent=2, default=str),
            encoding="utf-8",
        )
        (self.branch_dir / "solutions.json").write_text(
            json.dumps({str(k): v for k, v in self.solution_history.items()},
                       indent=2, default=str),
            encoding="utf-8",
        )
        if self.sandbox_log:
            with (self.branch_dir / "sandbox.jsonl").open("w", encoding="utf-8") as f:
                for entry in self.sandbox_log:
                    f.write(json.dumps(entry, default=str) + "\n")
        if self.cognitive_log:
            with (self.branch_dir / "cognitive.jsonl").open("w", encoding="utf-8") as f:
                for entry in self.cognitive_log:
                    f.write(json.dumps(entry, default=str) + "\n")
        if self.compression_events:
            (self.branch_dir / "compression_events.json").write_text(
                json.dumps(self.compression_events, indent=2, default=str),
                encoding="utf-8",
            )


# ── Tool dispatch ──
def dispatch_tool(tc, sandbox, state):
    name = tc.function.name
    try:
        args = json.loads(tc.function.arguments)
    except json.JSONDecodeError as e:
        return name, None, json.dumps({
            "error": f"invalid JSON in tool arguments: {e}",
            "raw": tc.function.arguments,
        })

    if name == "python":
        code = args.get("code", "")
        try:
            timeout = float(args.get("timeout", 30))
        except (TypeError, ValueError):
            timeout = 30.0
        timeout = min(max(timeout, 1.0), 120.0)
        output = sandbox.execute(code, timeout=timeout)
        state.append_sandbox_log(state.turn, code, output)
        return name, None, truncate_output(output, 8192)

    if name == "cognitive_activity":
        activity = args.get("activity", "")
        content = args.get("content", "")
        preserve = bool(args.get("preserve_across_reincarnation", False))
        event = state.append_cognitive(state.turn, activity, content, preserve)
        return name, activity, json.dumps({
            "status": "persisted",
            "event_id": event["id"],
            "activity": activity,
            "length": len(content),
            "preserved": preserve,
        })

    return name, None, json.dumps({"error": f"unknown tool: {name}"})


def build_task_user_message(task_data):
    n_tests = len(task_data["test"])
    base = build_user_prompt(task_data)
    if n_tests == 1:
        fmt = "\n\nEmit your answer as:\n    SOLUTION: \\boxed{[[row1], [row2], ...]}"
    else:
        lines = ["\n\nEmit one solution per test input, exactly:"]
        for i in range(n_tests):
            lines.append(f"    SOLUTION_{i+1}: \\boxed{{[[row1], [row2], ...]}}")
        fmt = "\n".join(lines)
    return base + fmt


# ── The branch solver ──
def solve_branch(branch_index, task_id, task_data, deadline):
    """Run one independent solver branch for one task.

    Each branch has its own sandbox, its own conversation, and its own
    evidence directory. Returns a dict with the branch's solutions per
    test input. Runs until first solution, deadline, or max_turns.
    """
    n_tests = len(task_data["test"])
    branch_dir = Path(CONFIG["evidence"]) / task_id / f"branch_{branch_index}"
    branch_dir.mkdir(parents=True, exist_ok=True)

    state = BranchState(task_id=task_id, branch_index=branch_index, branch_dir=branch_dir)
    state.solution_history = {i: [] for i in range(n_tests)}

    task_user_prompt = build_task_user_message(task_data)

    # Branch identity: branch 0 uses seed X, branch 1 uses seed X+1 so the
    # two branches do not sample the same tokens. The system prompt is
    # identical; the diversity comes from seed and temperature, not from
    # different instructions.
    branch_seed = CONFIG["seed"] + branch_index
    branch_tag = f"{task_id}/b{branch_index}"

    initial_messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task_user_prompt},
    ]
    state.live_messages = list(initial_messages)
    state.history = list(initial_messages)

    sandbox = JupyterSandbox(timeout=CONFIG["python_timeout"], preload_code=SANDBOX_PRELOAD)
    inject_task_data(sandbox, task_data)

    t0 = time.monotonic()
    final_solutions = [None] * n_tests

    try:
        for turn in range(CONFIG["max_turns"]):
            state.turn = turn
            if time.monotonic() > deadline:
                print(f"[{branch_tag}] turn {turn}: deadline reached")
                break

            if should_compress(state.live_messages, tools=ALL_TOOLS):
                print(f"[{branch_tag}] turn {turn}: reincarnation")
                try:
                    new_messages, summary = compress_conversation(
                        state.live_messages,
                        state.preserved_activities,
                        task_user_prompt,
                        deadline,
                    )
                    state.history.append({
                        "role": "system",
                        "content": f"[REINCARNATION_EVENT turn={turn}] summary_len={len(summary)}",
                    })
                    state.history.append({
                        "role": "assistant",
                        "content": f"[REINCARNATION_SUMMARY]\n{summary}",
                    })
                    state.live_messages = new_messages
                    state.compression_events.append({
                        "turn": turn,
                        "summary": summary,
                        "preserved_activities": len(state.preserved_activities),
                    })
                except Exception as e:
                    print(f"[{branch_tag}] turn {turn}: compression failed: {e}")
                    break

            try:
                response = call_model(
                    state.live_messages,
                    tools=ALL_TOOLS,
                    deadline=deadline,
                    seed=branch_seed,
                )
            except Exception as e:
                print(f"[{branch_tag}] turn {turn}: model call failed: {e}")
                break

            choice = response.choices[0]
            msg = choice.message

            assistant_dict = {"role": "assistant"}
            if msg.content:
                assistant_dict["content"] = msg.content
            if msg.tool_calls:
                assistant_dict["tool_calls"] = [
                    {"id": tc.id, "type": "function",
                     "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    for tc in msg.tool_calls
                ]
            state.append_message(assistant_dict)

            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name, activity_label, result = dispatch_tool(tc, sandbox, state)
                    state.append_message({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": result,
                    })
                    if name == "python":
                        print(f"[{branch_tag}] turn {turn}  python  ({len(result)} chars)")
                    elif name == "cognitive_activity":
                        print(f"[{branch_tag}] turn {turn}  cognitive_activity · {activity_label}")
                continue

            text = msg.content or ""
            solutions = extract_all_solutions(text, n_tests)
            n_found = sum(1 for s in solutions if s is not None)
            for i, sol in enumerate(solutions):
                if sol is not None:
                    state.solution_history[i].append(sol)

            if n_found == n_tests:
                for i in range(n_tests):
                    final_solutions[i] = state.solution_history[i][-1]
                print(f"[{branch_tag}] turn {turn}  SOLUTION emitted")
                break
            elif n_found:
                counts = [len(state.solution_history[i]) for i in range(n_tests)]
                print(f"[{branch_tag}] turn {turn}  partial solution {n_found}/{n_tests}; history={counts}")

            if not text.strip():
                state.append_message({
                    "role": "user",
                    "content": (
                        "Continue your investigation. Use the python tool to explore the "
                        "training pairs, and use cognitive_activity to record what you find."
                    ),
                })
                print(f"[{branch_tag}] turn {turn}  empty response, nudged")

    finally:
        sandbox.close()
        state.save()

    # Fallback: any partial solution discovered
    for i in range(n_tests):
        if final_solutions[i] is None and state.solution_history[i]:
            final_solutions[i] = state.solution_history[i][-1]

    return {
        "branch_index": branch_index,
        "solutions": final_solutions,
        "n_tests": n_tests,
        "turns": state.turn + 1,
        "duration": round(time.monotonic() - t0, 2),
        "compressions": len(state.compression_events),
    }


print("✓ Solver branch ready")

# CELL 20 (markdown)
## Block 08 — Run orchestrator

`solve_branch()` from Block 07 handles one branch. This block runs every
selected task under a bounded thread pool and collects the results.

### Concurrency model

`ThreadPoolExecutor(max_workers=CONFIG["task_parallel"])` runs tasks in parallel.
Each worker calls `_run_one_task`, which:

- owns its own pair of `JupyterSandbox` instances (one per branch),
- makes blocking HTTP requests to the vLLM server,
- writes its own evidence directory.

No state is shared between workers. The vLLM server batches concurrent requests
internally (`--max-num-seqs`), so running four tasks in parallel saturates the
engine without oversubscribing it. `CONFIG["task_parallel"]` must not exceed
`CONFIG["max_num_seqs"]` — a server that is asked to handle more concurrent
sequences than it was configured for will queue them, and the queueing delays
show up as solve-loop timeouts.

### Two branches per task

`_run_one_task` runs `CONFIG["branches_per_task"]` (default 2) branches for a
single task, in parallel with each other. Branch 0's solutions become
`attempt_1`; branch 1's become `attempt_2`. If a branch produces no solution for
a test input, that slot is `None` and the submission writer fills the
placeholder.

The two branches share a task-level deadline. They do not share a sandbox, a
conversation, or a seed. The whole point is that they arrive at the same answer
by different routes — or, if they disagree, that the disagreement is visible in
the trace.

### Per-task deadline

The overall solve window is `[ENTRY_MONOTONIC, ENTRY_MONOTONIC +
CONFIG["solve_stop_seconds"]]`. Each task gets the *smaller* of:

- the global solve deadline, and
- `now + CONFIG["per_task_timeout"]`.

A task that starts late therefore gets less time; a task that starts early
cannot run past the global window. This is what makes the run safe to cut short
— the global window is what protects the submission, and the per-task timeout is
what keeps one hard task from consuming the whole run.

### Task selection

`CONFIG["task_ids"]` is either `None` (all tasks in the challenge file) or an
explicit list. When a list is given, its order is preserved so the run is
reproducible. Any ID in the list that is not present in the challenge file is
reported as an error and skipped; it is not silently dropped. A silent drop
would leave the submission missing a task, which is worse than an early visible
failure.

### Failure handling

A worker that raises does not kill the pool. Its exception is caught, printed
with the task ID, and recorded as `{"attempt_1": [None]*n, "attempt_2":
[None]*n}`. The submission writer treats `None` slots as missing and fills them
with the placeholder grid, so a single failed task does not invalidate the whole
submission.

### Progress

The orchestrator prints one line per completed task: task ID, wall time, number
of turns, number of compressions, and whether a solution was found. On 240 tasks
with `task_parallel=4` you get a live trace without per-turn spam; the per-turn
detail lives in each task's evidence directory.

### Optional scoring

If `CONFIG["solutions"]` is set and the file exists, `score_results()` scores
the run at the end. The solver never sees the solutions file — scoring happens
after every task has closed. This is the same separation the ARC competition
requires, and enforcing it in the notebook means a scoring bug cannot leak into
a solve.

# CELL 21 (code)
# ── Block 08: Run orchestrator (two branches per task) ──

from concurrent.futures import ThreadPoolExecutor, as_completed


def _run_one_task(tid, task_data, deadline):
    """Run CONFIG['branches_per_task'] independent branches and merge their output.

    Branch 0's solutions become attempt_1.
    Branch 1's solutions become attempt_2.
    If a branch produces no solution for a test input, that slot is None
    and the submission writer fills the placeholder.
    """
    n_branches = CONFIG["branches_per_task"]
    branch_results = [None] * n_branches

    with ThreadPoolExecutor(max_workers=n_branches) as ex:
        futures = {
            ex.submit(solve_branch, i, tid, task_data, deadline): i
            for i in range(n_branches)
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                branch_results[i] = fut.result()
            except Exception as e:
                import traceback
                traceback.print_exc()
                branch_results[i] = {
                    "branch_index": i,
                    "solutions": [None] * len(task_data["test"]),
                    "n_tests": len(task_data["test"]),
                    "turns": 0,
                    "duration": 0.0,
                    "compressions": 0,
                    "error": f"{type(e).__name__}: {e}",
                }

    # Merge: branch i -> attempt_{i+1}
    n_tests = len(task_data["test"])
    attempt_lists = []
    for i in range(n_branches):
        br = branch_results[i] or {}
        sols = br.get("solutions") or [None] * n_tests
        if len(sols) < n_tests:
            sols = list(sols) + [None] * (n_tests - len(sols))
        attempt_lists.append(sols)

    # Pad missing attempt slots if branches_per_task < 2 (should not happen, but safe)
    while len(attempt_lists) < 2:
        attempt_lists.append([None] * n_tests)

    total_turns = sum((b or {}).get("turns", 0) for b in branch_results)
    total_duration = max(((b or {}).get("duration", 0.0) for b in branch_results), default=0.0)
    total_compressions = sum((b or {}).get("compressions", 0) for b in branch_results)
    branch_errors = [
        (b or {}).get("error")
        for b in branch_results
        if (b or {}).get("error")
    ]

    result = {
        "attempt_1": attempt_lists[0],
        "attempt_2": attempt_lists[1],
        "n_tests": n_tests,
        "branches": n_branches,
        "turns": total_turns,
        "duration": total_duration,
        "compressions": total_compressions,
        "branch_summary": [
            {
                "branch": i,
                "turns": (branch_results[i] or {}).get("turns", 0),
                "duration": (branch_results[i] or {}).get("duration", 0.0),
                "solved": any(g is not None for g in attempt_lists[i]),
            }
            for i in range(n_branches)
        ],
    }
    if branch_errors:
        result["errors"] = branch_errors
    return result


def run_all_tasks(challenges):
    """Solve every selected task. Two branches per task, running in parallel."""
    if CONFIG["task_ids"] is None:
        selected = list(challenges.keys())
    else:
        missing = [tid for tid in CONFIG["task_ids"] if tid not in challenges]
        if missing:
            print(f"⚠ task_ids not present in challenge file: {missing}")
        selected = [tid for tid in CONFIG["task_ids"] if tid in challenges]

    if not selected:
        raise RuntimeError("No tasks selected")

    global_deadline = ENTRY_MONOTONIC + CONFIG["solve_stop_seconds"]
    results = {}

    print(f"Tasks              : {len(selected)}")
    print(f"Task workers       : {CONFIG['task_parallel']}")
    print(f"Branches per task  : {CONFIG['branches_per_task']}")
    print(f"Peak concurrent branches: {CONFIG['peak_concurrent_branches']} "
          f"(<= max_num_seqs={CONFIG['max_num_seqs']})")
    print(f"Global solve deadline: {CONFIG['solve_stop_seconds']}s from entry")
    print(f"Per-task timeout   : {CONFIG['per_task_timeout']}s\n")

    t_start = time.monotonic()

    def _worker(tid):
        now = time.monotonic()
        deadline = min(global_deadline, now + CONFIG["per_task_timeout"])
        if deadline <= now:
            return tid, {
                "attempt_1": [None] * len(challenges[tid]["test"]),
                "attempt_2": [None] * len(challenges[tid]["test"]),
                "n_tests": len(challenges[tid]["test"]),
                "branches": CONFIG["branches_per_task"],
                "turns": 0,
                "duration": 0.0,
                "compressions": 0,
                "error": "global solve deadline already passed",
            }
        try:
            return tid, _run_one_task(tid, challenges[tid], deadline)
        except Exception as e:
            import traceback
            traceback.print_exc()
            return tid, {
                "attempt_1": [None] * len(challenges[tid]["test"]),
                "attempt_2": [None] * len(challenges[tid]["test"]),
                "n_tests": len(challenges[tid]["test"]),
                "branches": CONFIG["branches_per_task"],
                "turns": 0,
                "duration": 0.0,
                "compressions": 0,
                "error": f"{type(e).__name__}: {e}",
            }

    completed = 0
    with ThreadPoolExecutor(max_workers=CONFIG["task_parallel"]) as ex:
        futures = {ex.submit(_worker, tid): tid for tid in selected}
        for fut in as_completed(futures):
            tid, result = fut.result()
            results[tid] = result
            completed += 1

            a1 = result.get("attempt_1") or []
            a2 = result.get("attempt_2") or []
            solved_1 = any(g is not None for g in a1)
            solved_2 = any(g is not None for g in a2)
            if solved_1 and solved_2:
                status = "✓✓"
            elif solved_1 or solved_2:
                status = "✓·"
            else:
                status = "··"
            err = f"  error={result['error']}" if result.get("error") else ""

            print(
                f"[{completed:3d}/{len(selected)}] {status} {tid}  "
                f"{result['duration']:.1f}s  "
                f"turns={result['turns']}  "
                f"compressions={result['compressions']}"
                f"{err}"
            )

    elapsed = time.monotonic() - t_start
    solved_both = sum(
        1 for r in results.values()
        if any(g is not None for g in (r.get("attempt_1") or []))
        and any(g is not None for g in (r.get("attempt_2") or []))
    )
    solved_any = sum(
        1 for r in results.values()
        if any(g is not None for g in (r.get("attempt_1") or []))
        or any(g is not None for g in (r.get("attempt_2") or []))
    )
    print(f"\nRun complete: {solved_both}/{len(selected)} tasks solved by both branches; "
          f"{solved_any}/{len(selected)} by at least one")
    print(f"Total wall clock: {elapsed:.1f}s ({elapsed/60:.1f} min)")

    return results


def score_results(results, solutions):
    """Exact-match pass@2: attempt_1 OR attempt_2 matches the expected grid."""
    total = 0
    matched = 0
    per_task = {}

    for tid, sols in solutions.items():
        if tid not in results:
            continue
        r = results[tid]
        a1_list = r.get("attempt_1") or []
        a2_list = r.get("attempt_2") or []
        task_matched = 0
        task_total = 0
        for i, expected in enumerate(sols):
            if i >= len(a1_list):
                continue
            task_total += 1
            total += 1
            a1 = a1_list[i]
            a2 = a2_list[i] if i < len(a2_list) else None
            hit = False
            if a1 is not None and a1 == expected:
                hit = True
            elif a2 is not None and a2 == expected:
                hit = True
            if hit:
                task_matched += 1
                matched += 1
        per_task[tid] = (task_matched, task_total)

    score = (matched / total) if total else 0.0
    print(f"\nScore (exact match, attempt_1 OR attempt_2): {matched}/{total} = {score:.4f}")
    return score, per_task


print("✓ Orchestrator ready (2 branches per task, parallel)")

# CELL 22 (markdown)
## Block 09 — Submission writer

Converts the solver's per-task results into the ARC-AGI-2 submission format,
validates it, and writes it atomically.

### Target format

The competition expects one entry per task, in test-input order, with both
attempt slots present:

```json
{
  "task_id": [
    {"attempt_1": [[...]], "attempt_2": [[...]]},
    {"attempt_1": [[...]], "attempt_2": [[...]]}
  ]
}

# CELL 23 (code)

# ── Block 09: Submission writer ──

from pathlib import Path


PLACEHOLDER = [[0]]


def _valid_or_placeholder(grid):
    """Return (grid, repaired_flag). If grid is invalid, use placeholder."""
    if grid is None:
        return [list(r) for r in PLACEHOLDER], True
    if validate_grid(grid):
        return [list(r) for r in grid], False
    return [list(r) for r in PLACEHOLDER], True


def build_submission(challenges, results):
    """Build a complete, structurally valid submission from solver results.

    Returns (submission, report) where report has totals and per-task repairs.
    """
    submission = {}
    report = {"total_slots": 0, "repaired_slots": 0, "per_task": {}}

    for tid, task in challenges.items():
        n_tests = len(task["test"])
        r = results.get(tid) or {}
        a1_list = r.get("attempt_1") or []
        a2_list = r.get("attempt_2") or []

        task_repaired = 0
        entries = []
        for i in range(n_tests):
            a1 = a1_list[i] if i < len(a1_list) else None
            a2 = a2_list[i] if i < len(a2_list) else None

            g1, r1 = _valid_or_placeholder(a1)
            g2, r2 = _valid_or_placeholder(a2)
            task_repaired += int(r1) + int(r2)

            entries.append({"attempt_1": g1, "attempt_2": g2})

        submission[tid] = entries
        report["per_task"][tid] = {
            "n_tests": n_tests,
            "repaired_slots": task_repaired,
        }
        report["total_slots"] += n_tests * 2
        report["repaired_slots"] += task_repaired

    return submission, report


def validate_submission(submission, challenges):
    """Raise on any structural violation. Does not repair."""
    if set(submission) != set(challenges):
        missing = set(challenges) - set(submission)
        extra = set(submission) - set(challenges)
        raise ValueError(f"submission task set mismatch: missing={missing} extra={extra}")

    for tid, task in challenges.items():
        n_tests = len(task["test"])
        entries = submission[tid]
        if not isinstance(entries, list) or len(entries) != n_tests:
            raise ValueError(f"{tid}: expected {n_tests} entries, got {len(entries)}")
        for i, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(f"{tid}[{i}]: entry is not a dict")
            if set(entry) != {"attempt_1", "attempt_2"}:
                raise ValueError(f"{tid}[{i}]: keys must be attempt_1 and attempt_2")
            for slot in ("attempt_1", "attempt_2"):
                if not validate_grid(entry[slot]):
                    raise ValueError(f"{tid}[{i}].{slot}: invalid grid")

    print(f"✓ Submission valid: {len(submission)} tasks, "
          f"{sum(len(v) for v in submission.values())} test inputs, "
          f"{sum(len(v) * 2 for v in submission.values())} slots")


def write_submission_atomic(submission, path):
    """Write to a temp file, then os.replace into place."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(submission, indent=1), encoding="utf-8")
    os.replace(tmp, path)
    size_kb = path.stat().st_size / 1024
    print(f"✓ Submission written: {path} ({size_kb:.1f} KB)")


print("✓ Submission writer ready")

# CELL 24 (markdown)

## Block 10 — Cleanup and GPU safety net

Two shutdown paths. They serve different failure modes; run them in this order.

### 1. `stop_owned_server()` — the normal path

Terminates the exact process this notebook launched, using its process group so
worker children exit too. Called automatically via `atexit` and also exposed as
a plain function. After it returns, the vLLM server is gone and its GPU memory
is released.

The sequence is: send `SIGTERM` to the process group, wait up to 20 seconds for
a clean exit, escalate to `SIGKILL` if the process is still alive, wait up to 10
more seconds. The two-stage approach gives the server a chance to close its log
file and release its CUDA context cleanly, while still guaranteeing that a hung
server does not block notebook shutdown.

If this succeeds, nothing else is needed. Skip `nuke_gpu_memory()`.

### 2. `nuke_gpu_memory()` — the safety net

For the cases `stop_owned_server()` cannot handle:

- A previous kernel was interrupted and left a vLLM process holding VRAM.
- The server crashed mid-run; the Python handle is stale but the OS process
  still owns its CUDA context.
- A fresh session still sees memory used by a detachable worker from an earlier
  run.

It kills vLLM entrypoint processes by name pattern, then any remaining process
holding GPU memory (excluding the notebook kernel itself), then anything still
listening on the vLLM port. It finishes by clearing the notebook kernel's own
CUDA context via `torch.cuda.empty_cache()` and reporting the actual GPU state
from `nvidia-smi`.

**Warning.** `nuke_gpu_memory()` kills every process on this session that holds
GPU memory, not just ones this notebook started. That is correct on Kaggle and
Colab, where the session is a dedicated VM. On a shared GPU host it will kill
other tenants' work. Only run it when you know the session is yours.

### Verification

Both functions print what they did. `nuke_gpu_memory()` ends by re-querying
`nvidia-smi` and reporting the actual GPU state, so a silently-failing kill is
visible rather than assumed. The final report is the only trustworthy
confirmation that memory was actually freed.

### When to use which

| Situation | Action |
|---|---|
| Normal end of run | `stop_owned_server()` (already called by `atexit`) |
| Cell raised after server started | Run the cleanup cell; `stop_owned_server()` runs first |
| Fresh session, VRAM already occupied | Run the cleanup cell; `stop_owned_server()` no-ops, `nuke_gpu_memory()` clears the leftovers |
| Not sure which | Run the cell. Safe on Kaggle/Colab. |

# CELL 25 (code)
# ── Block 10: Cleanup and GPU safety net ──

import atexit
import gc
import os
import signal
import subprocess
import time


# ── Normal path: terminate the exact process this notebook launched ──
def stop_owned_server():
    """Terminate server_proc (from Block 02) and its whole process group.

    Safe: touches only the process group of the process we started.
    No-op if the server was never launched or has already exited.
    """
    if "server_proc" not in globals() or server_proc is None:
        print("[stop_owned_server] no owned server to stop")
        return

    if server_proc.poll() is not None:
        print(f"[stop_owned_server] owned server already exited (code {server_proc.returncode})")
        if server_log is not None:
            try:
                server_log.close()
            except Exception:
                pass
        return

    print(f"[stop_owned_server] terminating PID {server_proc.pid} and its group")
    try:
        os.killpg(os.getpgid(server_proc.pid), signal.SIGTERM)
    except ProcessLookupError:
        pass

    # Wait up to 20s for a clean exit.
    t0 = time.monotonic()
    while time.monotonic() - t0 < 20:
        if server_proc.poll() is not None:
            break
        time.sleep(0.5)

    # Escalate if still alive.
    if server_proc.poll() is None:
        print("[stop_owned_server] SIGTERM did not finish it; sending SIGKILL")
        try:
            os.killpg(os.getpgid(server_proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print("[stop_owned_server] WARNING: server did not exit after SIGKILL")

    if server_log is not None:
        try:
            server_log.close()
        except Exception:
            pass

    print(f"[stop_owned_server] done; exit code {server_proc.returncode}")


# ── Safety net: reclaim GPU memory from anything we cannot identify as ours ──
def nuke_gpu_memory():
    """Kill vLLM processes and any remaining GPU-holding processes on this session.

    Destructive: kills all GPU-holding processes except this notebook kernel.
    Safe on Kaggle/Colab dedicated sessions. Do not run on a shared host.
    """
    my_pid = os.getpid()

    # 1. Kill vLLM entrypoint processes (all of them).
    for pattern in ("vllm.entrypoints", "vllm.v1.engine", "vllm.worker"):
        r = subprocess.run(["pkill", "-9", "-f", pattern],
                           capture_output=True, text=True)
        if r.returncode == 0:
            print(f"  killed by pattern: {pattern}")

    # 2. Kill anything else still holding GPU memory (excluding self).
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        if r.stdout.strip():
            for line in r.stdout.strip().splitlines():
                pid_str = line.strip()
                if not pid_str.isdigit():
                    continue
                pid = int(pid_str)
                if pid == my_pid:
                    continue
                try:
                    os.kill(pid, signal.SIGKILL)
                    print(f"  killed GPU process {pid}")
                except ProcessLookupError:
                    pass
                except PermissionError:
                    print(f"  could not kill {pid} (permission)")
    except Exception as e:
        print(f"  nvidia-smi query failed: {e}")

    # 3. Kill anything still listening on the vLLM port.
    port = CONFIG["port"]
    try:
        r = subprocess.run(["lsof", "-ti", f":{port}"],
                           capture_output=True, text=True)
        for pid_str in r.stdout.strip().splitlines():
            if pid_str.strip().isdigit():
                pid = int(pid_str.strip())
                if pid == my_pid:
                    continue
                subprocess.run(["kill", "-9", str(pid)], capture_output=True)
                print(f"  killed port-{port} listener {pid}")
    except Exception:
        pass

    time.sleep(3)

    # 4. Clear this kernel's own CUDA context so torch sees freed memory.
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()
    except Exception:
        pass
    gc.collect()

    # 5. Report the actual GPU state so a silent failure is visible.
    try:
        r = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.free",
             "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        print(f"  GPU status (used MiB, free MiB per device):")
        for line in r.stdout.strip().splitlines():
            print(f"    {line}")
    except Exception:
        pass

    print("[nuke_gpu_memory] done")


# ── Register stop_owned_server for process exit ──
atexit.register(stop_owned_server)


# ── Interactive invocation ──
# Normal end of notebook: nothing to do here, atexit will handle it.
# If the server crashed or you are resuming after an interruption, run this:
#
#   stop_owned_server()      # try the clean path first
#   nuke_gpu_memory()        # only if the clean path did not free VRAM
#
# Uncomment the next two lines to run both now.
# stop_owned_server()
# nuke_gpu_memory()

print("✓ Cleanup registered.")
print("  - stop_owned_server() runs automatically at process exit.")
print("  - nuke_gpu_memory() is available for manual use when VRAM is stuck.")

# CELL 26 (code)
#stop_owned_server()
#nuke_gpu_memory()

# CELL 27 (markdown)
## Final cell — Solve, build, validate, write

One cell that runs everything. Its structure mirrors what the competition does:

1. Load the challenge file (Block 03).
2. Solve every selected task (Block 08).
3. Build the submission from the solver results (Block 09).
4. Validate the structure before writing (Block 09).
5. Write atomically to `CONFIG["submission"]` (Block 09).
6. If a solutions file is configured, score the run (Block 08).

If any step raises, the notebook stops before the submission is written — a
broken run leaves no file for Kaggle to pick up. The cleanup cell (Block 10)
runs after regardless, via `atexit`.

### Why the order matters

Building before validating means the placeholder substitution happens first, and
validation then sees a submission that is guaranteed to be structurally valid
unless there is a bug in the builder. That is the point: `validate_submission`
is checking the builder, not the solver. A solver that produced no solutions at
all still produces a valid submission.

Writing before scoring means a scoring failure does not prevent the submission
from being written. Scoring is a convenience; the submission is the point.

### Dry run

With `CONFIG["execute"] = False`, this cell prints a message and does nothing.
Useful for verifying the setup cells without launching the server or spending
GPU time. In dry-run mode `challenges` may be `None`; the cell does not require
it.

### Resuming

The notebook is not designed to resume mid-run. If the kernel dies at task 150,
rerun from the top — the vLLM server restarts, tasks re-run from scratch. This
is intentional: partial submissions from a crashed kernel are not more useful
than a clean rerun, and a clean rerun is what the deadline math is calibrated
against. A resume path would require checkpointing every branch's state, and the
added complexity would cost more than the rerun saves.

# CELL 28 (code)
# ── Final cell: run the full pipeline ──

if not CONFIG["execute"]:
    print("CONFIG['execute'] = False — dry run.")
    print("Set it to True in Block 01 to actually solve tasks.")
else:
    print("=" * 64)
    print(f"DAO EXPLORER — {CONFIG['MODE']} RUN")
    print("=" * 64)

    print("\n[1/4] Loading challenges ...")
    if challenges is None:
        challenges = load_challenges(CONFIG["challenges"])
    preflight_report(challenges)

    print("\n[2/4] Solving tasks ...")
    results = run_all_tasks(challenges)

    print("\n[3/4] Building submission ...")
    submission, report = build_submission(challenges, results)
    print(f"  slots total     : {report['total_slots']}")
    print(f"  slots repaired  : {report['repaired_slots']}")

    print("\n[4/4] Validating and writing ...")
    validate_submission(submission, challenges)
    write_submission_atomic(submission, CONFIG["submission"])

    if CONFIG["MODE"] == "UNIT_TEST":
        print("\n" + "=" * 64)
        print("UNIT_TEST SUMMARY")
        print("=" * 64)
        solutions = load_solutions(CONFIG["solutions"])
        if solutions:
            filtered = {tid: solutions[tid] for tid in challenges if tid in solutions}
            score, per_task = score_results(results, filtered)
            print("\nPer-task:")
            for tid in sorted(per_task):
                matched, total = per_task[tid]
                mark = "✓" if matched == total else "·"
                print(f"  {mark} {tid}  {matched}/{total}")

                # Show the model's cognitive activity labels as a reading aid
                cog_path = Path(CONFIG["evidence"]) / tid / "cognitive.jsonl"
                if cog_path.is_file():
                    labels = []
                    for line in cog_path.read_text(encoding="utf-8").splitlines():
                        try:
                            ev = json.loads(line)
                            labels.append(ev.get("activity", "?"))
                        except json.JSONDecodeError:
                            continue
                    if labels:
                        # Deduplicate while preserving order
                        seen = set()
                        ordered = [x for x in labels if not (x in seen or seen.add(x))]
                        print(f"      notebook: {' → '.join(ordered[:12])}"
                              + (" → ..." if len(ordered) > 12 else ""))
            print(f"\nOverall: {score:.4f}")
        else:
            print(f"  Solutions file not found: {CONFIG['solutions']}")
    else:
        print("\n" + "=" * 64)
        print("FULL SUBMISSION COMPLETE")
        print("=" * 64)
        print(f"  Submission: {CONFIG['submission']}")
        print(f"  Tasks     : {len(submission)}")
        print(f"  Slots     : {report['total_slots']}")
        print(f"  Repaired  : {report['repaired_slots']}")

# CELL 29 (code)

