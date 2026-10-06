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

# F0-25 with validated suffix path. No eval this session (quota).
f0_out = os.path.join(WORK_DIR, 's4-f0')
cmd = [
    sys.executable, os.path.join(CODE_DIR, 'train_grid_sft.py'),
    '--model-dir', model_dir,
    '--episodes', os.path.join(CODE_DIR, 'paired-f0-400.jsonl'),
    '--out-dir', f0_out,
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
    '--formatter', 'f0_space',
    '--logits-mode', 'suffix',
    '--wall-clock-limit', '7200',
]
print('CMD:', ' '.join(cmd), flush=True)
proc = subprocess.run(cmd)
print('EXIT', proc.returncode, 'TRAIN_S4_F0', flush=True)
if proc.returncode != 0:
    raise SystemExit('TRAIN_S4_F0_FAILED %d' % proc.returncode)

summary_path = os.path.join(f0_out, 'SUMMARY.json')
summary = json.load(open(summary_path, encoding='utf-8'))
print('SUMMARY', json.dumps(summary, ensure_ascii=False)[:2000], flush=True)
completed = int(summary.get('completed_updates', 0))
snap = os.path.join(f0_out, 'adapter_step_%d' % completed)
print('SNAPSHOT:', snap, os.path.isdir(snap), flush=True)
if completed < 25:
    print('NOTE_UNDER_25_NO_EVAL_THIS_CYCLE', flush=True)
if not os.path.isfile(os.path.join(snap, 'adapter_config.json')):
    raise SystemExit('SNAPSHOT_ADAPTER_MISSING s4-f0')

print('S4_DONE', flush=True)
