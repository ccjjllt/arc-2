import glob
import json
import os
import subprocess
import sys

CODE_DIR = None
for cand in glob.glob('/kaggle/input/**/paired-f0-400.jsonl', recursive=True):
    CODE_DIR = os.path.dirname(cand)
    break
print('CODE_DIR:', CODE_DIR, flush=True)
if not CODE_DIR:
    raise SystemExit('CODE_NOT_FOUND')

WORK_DIR = '/kaggle/working/010'
os.makedirs(WORK_DIR, exist_ok=True)
ch_path = os.path.join(CODE_DIR, 'e0-p12-challenges.json')
map_path = os.path.join(CODE_DIR, 'e0-p12-source-map.json')
TASK_IDS_CLI = 'db118e2a,291dc1e1,58c02a16,4c416de3,fc4aaf52,aa4ec2a5,e3f79277,1818057f,342dd610,21897d95,52df9849,9aaea919'

train_sol_hits = glob.glob('/kaggle/input/competitions/**/arc-agi_training_solutions.json', recursive=True)
eval_sol_hits = glob.glob('/kaggle/input/competitions/**/arc-agi_evaluation_solutions.json', recursive=True)
route = json.load(open(map_path, encoding='utf-8'))['route']
train_sol = json.load(open(train_sol_hits[0], encoding='utf-8')) if train_sol_hits else {}
eval_sol = json.load(open(eval_sol_hits[0], encoding='utf-8')) if eval_sol_hits else {}
scoring = {}
for entry in route:
    tid = entry['task_id']
    fn = entry['scoring_solutions_filename']
    if fn == 'arc-agi_training_solutions.json' and tid in train_sol:
        scoring[tid] = train_sol[tid]
    elif fn == 'arc-agi_evaluation_solutions.json' and tid in eval_sol:
        scoring[tid] = eval_sol[tid]
scoring_path = os.path.join(WORK_DIR, 'scoring.json')
json.dump(scoring, open(scoring_path, 'w', encoding='utf-8'))
print('SCORED_TASKS:', sorted(scoring), flush=True)

model_dir = None
for path in glob.glob('/kaggle/input/**/config.json', recursive=True):
    try:
        cfg = json.load(open(path, encoding='utf-8'))
    except Exception:
        continue
    if cfg.get('model_type') in ('qwen3_5', 'qwen3'):
        model_dir = os.path.dirname(path)
        break
print('MODEL_DIR:', model_dir, flush=True)
if not model_dir:
    raise SystemExit('MODEL_DIR_NOT_FOUND')

adapter_dir = None
for path in glob.glob('/kaggle/input/**/adapter_config.json', recursive=True):
    cand = os.path.dirname(path)
    if os.path.isfile(os.path.join(cand, 'adapter_model.safetensors')) and 'arc2-007-g1-adapter' in cand:
        adapter_dir = cand
        break
print('G1_ADAPTER_DIR:', adapter_dir, flush=True)
if not adapter_dir:
    raise SystemExit('G1_ADAPTER_DIR_NOT_FOUND')


def run_cmd(cmd, tag, ok_codes=(0,)):
    print('CMD:', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd)
    print('EXIT', proc.returncode, tag, flush=True)
    if proc.returncode not in ok_codes:
        raise SystemExit('%s_FAILED %d' % (tag, proc.returncode))
    return proc.returncode


# Stage 1: suffix-logits probe on G1 non-zero LoRA (short/2k/4k, no training).
# Non-zero exit tolerated: F1 train below always uses default full path.
probe_out = os.path.join(WORK_DIR, 'probe')
code = run_cmd([
    sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
    '--model-dir', model_dir,
    '--episodes', os.path.join(CODE_DIR, 'paired-f0-400.jsonl'),
    '--out-dir', probe_out,
    '--init-adapter', adapter_dir,
    '--probe-suffix-logits', 'short,2k,4k',
    '--seed', '20261003',
], 'PROBE', ok_codes=(0, 1, 2))
probe_json = os.path.join(probe_out, 'suffix_probe.json')
print('PROBE_JSON_EXISTS', os.path.isfile(probe_json), 'exit', code, flush=True)
if os.path.isfile(probe_json):
    print('PROBE_JSON', open(probe_json, encoding='utf-8').read()[:3000], flush=True)

# Stage 2: F1-25 fresh training (default full path regardless of probe).
f1_out = os.path.join(WORK_DIR, 's1-f1')
run_cmd([
    sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
    '--model-dir', model_dir,
    '--episodes', os.path.join(CODE_DIR, 'paired-f1-400.jsonl'),
    '--out-dir', f1_out,
    '--optimizer-updates', '100',
    '--scheduler-horizon', '100',
    '--stop-after-update', '25',
    '--grad-accum', '4',
    '--microbatch', '1',
    '--max-len', '4096',
    '--overflow-len', '4096',
    '--save-every', '25',
    '--lr', '2e-5',
    '--weight-decay', '0.01',
    '--warmup-updates', '10',
    '--clip-norm', '1.0',
    '--seed', '20261003',
    '--formatter', 'f1_compact',
    '--wall-clock-limit', '7200',
], 'TRAIN_S1_F1')
snap_f1 = os.path.join(f1_out, 'adapter_step_25')
print('SNAPSHOT_FILES:', sorted(os.listdir(snap_f1)) if os.path.isdir(snap_f1) else None, flush=True)
if not os.path.isfile(os.path.join(snap_f1, 'adapter_config.json')):
    raise SystemExit('SNAPSHOT_ADAPTER_MISSING s1-f1')

# Stage 3: P12 greedy eval of F1-25.
eval_path = os.path.join(WORK_DIR, 's1_f1_25_eval.json')
run_cmd([
    sys.executable, os.path.join(CODE_DIR, 'solve_grid.py'),
    '--model-dir', model_dir,
    '--adapter', snap_f1,
    '--challenges', ch_path,
    '--solutions', scoring_path,
    '--task-ids', TASK_IDS_CLI,
    '--formatter', 'f1_compact',
    '--max-new-tokens', '960',
    '--n-sample', '0',
    '--second-view', 'none',
    '--seed', '20261003',
    '--time-budget', '4200',
    '--out', eval_path,
], 'EVAL_S1_F1_25', ok_codes=(0, 2))
payload = json.load(open(eval_path, encoding='utf-8'))
print('N_QUERIES', len(payload.get('queries') or []), 'EVAL_S1_F1_25', flush=True)
for key in ('pair_aggregates', 'aggregates'):
    if payload.get(key) is not None:
        print(key.upper(), json.dumps(payload[key], ensure_ascii=False)[:2000], flush=True)

print('PROBE_F1_25_DONE', flush=True)
