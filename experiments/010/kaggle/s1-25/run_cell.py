import glob
import json
import os
import subprocess
import sys

CODE_DIR = None
for cand in glob.glob('/kaggle/input/**/solve_grid.py', recursive=True):
    cand_dir = os.path.dirname(cand)
    if os.path.isfile(os.path.join(cand_dir, 'paired-f0-400.jsonl')):
        CODE_DIR = cand_dir
        break
print('CODE_DIR:', CODE_DIR, flush=True)
if not CODE_DIR:
    raise SystemExit('CODE_NOT_FOUND')

for name in ('paired-f0-400.jsonl', 'paired-f1-400.jsonl'):
    path = os.path.join(CODE_DIR, name)
    n = sum(1 for _ in open(path, encoding='utf-8'))
    print('EPISODES_FILE:', name, n, flush=True)
    if n != 400:
        raise SystemExit('PAIRED_400_COUNT_MISMATCH %s' % name)

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


def run_cmd(cmd, tag, ok_codes=(0,)):
    print('CMD:', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd)
    print('EXIT', proc.returncode, tag, flush=True)
    if proc.returncode not in ok_codes:
        raise SystemExit('%s_FAILED %d' % (tag, proc.returncode))


def summarize_solve(out_path, tag):
    payload = json.load(open(out_path, encoding='utf-8'))
    print('N_QUERIES', len(payload.get('queries') or []), tag, flush=True)
    for key in ('pair_aggregates', 'aggregates'):
        if payload.get(key) is not None:
            print(key.upper(), json.dumps(payload[key], ensure_ascii=False)[:2000], flush=True)


# Stage 1: F1 zero-step probe on base (B-chain GPU check + cold-load warmup).
run_cmd([
    sys.executable, os.path.join(CODE_DIR, 'solve_grid.py'),
    '--model-dir', model_dir,
    '--challenges', ch_path,
    '--solutions', scoring_path,
    '--task-ids', TASK_IDS_CLI,
    '--formatter', 'f1_compact',
    '--max-new-tokens', '960',
    '--n-sample', '0',
    '--second-view', 'none',
    '--seed', '20261003',
    '--time-budget', '3600',
    '--out', os.path.join(WORK_DIR, 's1_f1_at0.json'),
], 'F1AT0', ok_codes=(0, 2))
summarize_solve(os.path.join(WORK_DIR, 's1_f1_at0.json'), 'F1AT0')


def train_arm(episodes, out_name, formatter):
    out_dir = os.path.join(WORK_DIR, out_name)
    run_cmd([
        sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
        '--model-dir', model_dir,
        '--episodes', os.path.join(CODE_DIR, episodes),
        '--out-dir', out_dir,
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
        '--formatter', formatter,
        '--wall-clock-limit', '7200',
    ], 'TRAIN_' + out_name.upper())
    snap = os.path.join(out_dir, 'adapter_step_25')
    print('SNAPSHOT:', snap, os.path.isdir(snap), flush=True)
    print('SNAPSHOT_FILES:', sorted(os.listdir(snap)) if os.path.isdir(snap) else None, flush=True)
    if not os.path.isfile(os.path.join(snap, 'adapter_config.json')):
        raise SystemExit('SNAPSHOT_ADAPTER_MISSING ' + out_name)
    return snap


# Stage 2/3: fresh-init F0/F1 arms, same seed/order (B=0 functional equivalence).
snap_f0 = train_arm('paired-f0-400.jsonl', 's1-f0', 'f0_space')
snap_f1 = train_arm('paired-f1-400.jsonl', 's1-f1', 'f1_compact')


def eval_arm(adapter_dir, formatter, out_name):
    out_path = os.path.join(WORK_DIR, out_name)
    run_cmd([
        sys.executable, os.path.join(CODE_DIR, 'solve_grid.py'),
        '--model-dir', model_dir,
        '--adapter', adapter_dir,
        '--challenges', ch_path,
        '--solutions', scoring_path,
        '--task-ids', TASK_IDS_CLI,
        '--formatter', formatter,
        '--max-new-tokens', '2048' if formatter == 'f0_space' else '960',
        '--n-sample', '0',
        '--second-view', 'none',
        '--seed', '20261003',
        '--time-budget', '4200',
        '--out', out_path,
    ], 'EVAL_' + out_name.upper(), ok_codes=(0, 2))
    summarize_solve(out_path, 'EVAL_' + out_name.upper())


# Stage 4: P12 greedy per arm.
eval_arm(snap_f0, 'f0_space', 's1_f0_25_eval.json')
eval_arm(snap_f1, 'f1_compact', 's1_f1_25_eval.json')

print('S1_25_DONE', flush=True)
