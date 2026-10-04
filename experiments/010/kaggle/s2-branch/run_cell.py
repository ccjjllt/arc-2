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


# Stage 1: suffix probe, fixed pool bounds (short/2k/4k need 4096).
probe_out = os.path.join(WORK_DIR, 'probe')
run_cmd([
    sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
    '--model-dir', model_dir,
    '--episodes', os.path.join(CODE_DIR, 'paired-f0-400.jsonl'),
    '--out-dir', probe_out,
    '--init-adapter', adapter_dir,
    '--max-len', '4096',
    '--overflow-len', '4096',
    '--probe-suffix-logits', 'short,2k,4k',
    '--seed', '20261003',
], 'PROBE')
probe_json = os.path.join(probe_out, 'suffix_probe.json')
print('PROBE_JSON_EXISTS', os.path.isfile(probe_json), flush=True)

# Pre-declared adoption rule (progress/010 §12): max rel_err<=1e-3,
# min cosine>=0.999, loss diff<=2x repeat noise. Missing fields -> fail.
adopt = False
if os.path.isfile(probe_json):
    pj = json.load(open(probe_json, encoding='utf-8'))
    print('PROBE_JSON', json.dumps(pj, ensure_ascii=False)[:3000], flush=True)
    try:
        grad = pj.get('lora_grad', {})
        loss = pj.get('loss', {})
        noise = max(loss.get('repeat_noise_full_abs', 0) or 0,
                    loss.get('repeat_noise_suffix_abs', 0) or 0, 1e-12)
        diff = abs(loss.get('full_1', 0) - loss.get('suffix_1', 0))
        adopt = (grad.get('max_rel_error', 1) <= 1e-3
                 and grad.get('min_cosine', 0) >= 0.999
                 and diff <= 2 * noise)
    except Exception as exc:
        print('PROBE_RULE_EVAL_ERROR', str(exc)[:200], flush=True)
        adopt = False
print('ADOPT_SUFFIX', adopt, flush=True)


def train_arm(episodes, out_name, formatter, stop, extra=()):
    out_dir = os.path.join(WORK_DIR, out_name)
    run_cmd([
        sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
        '--model-dir', model_dir,
        '--episodes', os.path.join(CODE_DIR, episodes),
        '--out-dir', out_dir,
        '--optimizer-updates', '100',
        '--scheduler-horizon', '100',
        '--stop-after-update', str(stop),
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
    ] + list(extra), 'TRAIN_' + out_name.upper())
    snap = os.path.join(out_dir, 'adapter_step_%d' % stop)
    if not os.path.isfile(os.path.join(snap, 'adapter_config.json')):
        raise SystemExit('SNAPSHOT_ADAPTER_MISSING ' + out_name)
    return snap


def eval_arm(adapter_dir, formatter, out_name, max_new):
    out_path = os.path.join(WORK_DIR, out_name)
    run_cmd([
        sys.executable, os.path.join(CODE_DIR, 'solve_grid.py'),
        '--model-dir', model_dir,
        '--adapter', adapter_dir,
        '--challenges', ch_path,
        '--solutions', scoring_path,
        '--task-ids', TASK_IDS_CLI,
        '--formatter', formatter,
        '--max-new-tokens', str(max_new),
        '--n-sample', '0',
        '--second-view', 'none',
        '--seed', '20261003',
        '--time-budget', '4200',
        '--out', out_path,
    ], 'EVAL_' + out_name.upper(), ok_codes=(0, 2))
    payload = json.load(open(out_path, encoding='utf-8'))
    print('N_QUERIES', len(payload.get('queries') or []), 'EVAL_' + out_name.upper(), flush=True)
    for key in ('pair_aggregates', 'aggregates'):
        if payload.get(key) is not None:
            print(key.upper(), json.dumps(payload[key], ensure_ascii=False)[:2000], flush=True)


if adopt:
    # F0 with memory-saving suffix path.
    snap = train_arm('paired-f0-400.jsonl', 's2-f0', 'f0_space', 25,
                     extra=('--logits-mode', 'suffix'))
    eval_arm(snap, 'f0_space', 's2_f0_25_eval.json', 2048)
else:
    # F1 fresh to 50 (trains cleanly; conditional S2 continuation).
    snap = train_arm('paired-f1-400.jsonl', 's2-f1-50', 'f1_compact', 50)
    eval_arm(snap, 'f1_compact', 's2_f1_50_eval.json', 960)

print('S2_DONE', flush=True)
