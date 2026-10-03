import glob
import json
import os
import subprocess
import sys

CODE_DIR = '/kaggle/working/010/code'
WORK_DIR = '/kaggle/working/010'
os.makedirs(WORK_DIR, exist_ok=True)

TASK_IDS_CLI = 'db118e2a,291dc1e1,58c02a16,4c416de3,fc4aaf52,aa4ec2a5,e3f79277,1818057f,342dd610,21897d95,52df9849,9aaea919'

ch_path = os.path.join(CODE_DIR, 'e0-p12-challenges.json')
map_path = os.path.join(CODE_DIR, 'e0-p12-source-map.json')

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
    if os.path.isfile(os.path.join(cand, 'adapter_model.safetensors')):
        adapter_dir = cand
        break
print('ADAPTER_DIR:', adapter_dir, flush=True)
if not adapter_dir:
    raise SystemExit('ADAPTER_DIR_NOT_FOUND')


def run_solve(out_name, extra, budget):
    out_path = os.path.join(WORK_DIR, out_name)
    cmd = [
        sys.executable,
        os.path.join(CODE_DIR, 'solve_grid.py'),
        '--model-dir', model_dir,
        '--challenges', ch_path,
        '--solutions', scoring_path,
        '--seed', '20261003',
        '--time-budget', str(budget),
        '--out', out_path,
    ] + extra
    print('CMD:', ' '.join(cmd), flush=True)
    proc = subprocess.run(cmd)
    print('SOLVE_EXIT', proc.returncode, out_name, flush=True)
    if os.path.isfile(out_path):
        payload = json.load(open(out_path, encoding='utf-8'))
        print('N_QUERIES', len(payload.get('queries') or []), out_name, flush=True)
        for key in ('pair_aggregates', 'aggregates'):
            if payload.get(key) is not None:
                print(key.upper(), json.dumps(payload[key], ensure_ascii=False)[:2000], flush=True)
    if proc.returncode not in (0, 2):
        raise SystemExit(proc.returncode)


# Run 1: F1 zero-step probe on base (B-chain GPU check: compact prompt + parse).
run_solve('e0_f1_at0.json', [
    '--task-ids', TASK_IDS_CLI,
    '--formatter', 'f1_compact',
    '--max-new-tokens', '960',
    '--n-sample', '0',
    '--second-view', 'none',
], 3600)

# Run 2: missing F0 adapter arm for 9aaea919 q0 (E0 pair completion).
run_solve('e0_f0_fill.json', [
    '--adapter', adapter_dir,
    '--task-ids', '9aaea919',
    '--formatter', 'f0_space',
    '--max-new-tokens', '2048',
    '--n-sample', '0',
    '--second-view', 'none',
], 1500)

print('E0_FILL_F1AT0_DONE', flush=True)
