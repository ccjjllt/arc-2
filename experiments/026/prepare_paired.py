#!/usr/bin/env python3
"""Prepare a bounded SFT + paired NVARC V1 research notebook, without launch.

Third-party source stays in the temporary generated package. Evaluation query
answers are prohibited; the only embedded targets belong to the training pool.
"""
from __future__ import annotations

import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/private/tmp/arc2-025-v2-mount-ready')
REFERENCE_SHA = '71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d'
RUN_ROOT = '/kaggle/working/026'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one source anchor: ' + repr(before[:120]))
    return source.replace(before, after, 1)


def code_cell(source):
    return dict(cell_type='code', metadata={}, execution_count=None,
                outputs=[], source=source.splitlines(keepends=True))


def embed(destination, raw):
    payload = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    return ("payload = gzip.decompress(base64.b64decode(" + repr(payload) + "))\n"
            "if hashlib.sha256(payload).hexdigest() != " + repr(sha(raw)) + ":\n"
            "    raise RuntimeError('Embedded artifact identity mismatch')\n"
            "Path(" + repr(destination) + ").write_bytes(payload)\n")


STARTER = '''%%writefile starter_paired.py
import argparse, json, os, time
from pathlib import Path
import torch
import torch.multiprocessing as mp

def local_worker(physical_rank, queues, end_time, paths):
    arm = 'base' if physical_rank < 2 else 'challenger'
    local_rank = physical_rank % 2
    os.environ['CUDA_VISIBLE_DEVICES'] = str(physical_rank)
    os.environ['ARC026_ARM'] = arm
    os.environ['ARC026_MODEL_PATH'] = paths[arm]
    os.environ['ARC026_PHYSICAL_RANK'] = str(physical_rank)
    torch.set_default_device('cpu')
    # Preserve the original process-ordered import to protect shared Unsloth
    # patch/compile caches. Worker rank is local to its arm after importing.
    if physical_rank > 0:
        while not Path('/kaggle/arc026-worker%d' % (physical_rank - 1)).exists():
            if time.time() >= end_time:
                raise TimeoutError('Paired import ordering exhausted evaluation budget')
            time.sleep(2)
    from arc_solver import worker
    Path('/kaggle/arc026-worker%d' % physical_rank).write_text('Ok')
    print(json.dumps(dict(event='paired_worker_imported', arm=arm,
                          rank=local_rank, physical_rank=physical_rank,
                          model_path=paths[arm], time=time.time())), flush=True)
    worker(local_rank, queues[0 if arm == 'base' else 1], end_time)
    print(json.dumps(dict(event='paired_worker_finished', arm=arm,
                          rank=local_rank, physical_rank=physical_rank,
                          time=time.time())), flush=True)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--end-time', type=float, required=True)
    parser.add_argument('--base-model', required=True)
    parser.add_argument('--challenger-model', required=True)
    args = parser.parse_args()
    if os.environ.get('PYTHONHASHSEED') != '26026':
        raise RuntimeError('Both arms require the frozen Python hash seed')
    data = json.loads(Path('/kaggle/working/026/input/challenges.json').read_text())
    manager = mp.Manager()
    queues = [manager.Queue(), manager.Queue()]
    for queue in queues:
        for key in sorted(data):
            queue.put(key)
        for _ in range(2):
            queue.put(None)
    paths = dict(base=args.base_model, challenger=args.challenger_model)
    mp.spawn(local_worker, args=(queues, args.end_time, paths), nprocs=4)
'''


LAUNCH = '''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path

run_root = Path('/kaggle/working/026')
state = dict(notebook_started=notebook_started, session_deadline=session_deadline,
             training_stage_deadline=training_stage_deadline,
             input_identity=input_identity, training_identity=training_identity,
             source='Frozen V1 312063593 / successful 025 wrapper',
             python_hash_seed=26026, research_only=True, gold_read=False)
def save_state():
    (run_root / 'run-state.json').write_text(json.dumps(state, indent=2))
def run_child(command, environment, log_path, deadline, stage):
    record = dict(started_at=time.time(), hard_deadline=deadline, timed_out=False,
                  command=command)
    remaining = deadline - time.time()
    if remaining <= 0:
        return dict(record, return_code=-1, status='stage_budget_exhausted')
    with log_path.open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True,
                                   errors='replace', bufsize=1, start_new_session=True)
        record['process_group_id'] = process.pid
        def tee():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print('[' + stage + '] ' + line, end='', flush=True)
        thread = threading.Thread(target=tee, daemon=True)
        thread.start()
        try:
            record['return_code'] = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            record.update(timed_out=True, status='hard_deadline')
            os.killpg(process.pid, signal.SIGKILL)
            record['return_code'] = process.wait(timeout=10)
        finally:
            thread.join(timeout=2)
    record.setdefault('status', 'completed' if record['return_code'] == 0 else 'child_error')
    record.update(finished_at=time.time(), elapsed_s=time.time()-record['started_at'])
    return record

child_env = dict(os.environ)
child_env.update(UNSLOTH_DISABLE_STATISTICS='1', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                 OMP_NUM_THREADS='12', PYTHONUNBUFFERED='1', PYTHONHASHSEED='26026')
training_env = dict(child_env, CUDA_VISIBLE_DEVICES='0', WORLD_SIZE='1',
                    ARC026_ARM='training', ARC026_PHYSICAL_RANK='0',
                    ARC026_SOLVER_DEADLINE=str(training_stage_deadline))
for name in ['LOCAL_RANK', 'RANK', 'MASTER_ADDR', 'MASTER_PORT']:
    training_env.pop(name, None)
command = [sys.executable, '-u', 'train_specialist.py', '--model-path', '/kaggle/temp/arc026-base',
           '--train-jsonl', str(run_root / 'input/train.jsonl'),
           '--data-manifest', str(run_root / 'input/data-manifest.json'),
           '--expected-data-sha256', training_identity['train_sha256'],
           '--source-dir', str(Path.cwd()), '--output-dir', str(run_root / 'training'),
           '--max-steps', '256', '--pilot-steps', '32', '--train-seconds', '2700',
           '--max-seconds', '3600', '--stage-deadline-unix', str(training_stage_deadline)]
state['training'] = run_child(command, training_env, run_root / 'training.log',
                              training_stage_deadline, 'training')
save_state()
marker = run_root / 'training/merge_ready.json'
merged = run_root / 'training/merged_model'
if state['training']['return_code'] != 0 or not marker.is_file():
    state.update(status='training_failed_no_paired_evaluation', return_code=1)
else:
    ready = json.loads(marker.read_text())
    state['merge_ready'] = ready
    if not (merged / 'config.json').is_file() or (merged / 'adapter_config.json').exists():
        raise RuntimeError('Challenger must be a merged ordinary model with fresh TTT adapter')
    # The training process has fully exited; no trained model object or CUDA
    # allocation is reused by any rank-256 per-task TTT process.
    evaluation_started = time.time()
    global_end_time = min(evaluation_started + 7200, session_deadline - 120)
    hard_deadline = min(global_end_time + 60, session_deadline - 60)
    state.update(evaluation_started_at=evaluation_started,
                 global_end_time=global_end_time, solver_hard_deadline=hard_deadline)
    child_env['ARC026_SOLVER_DEADLINE'] = str(global_end_time)
    for rank in range(4):
        Path('/kaggle/arc026-worker%d' % rank).unlink(missing_ok=True)
    command = [sys.executable, '-u', 'starter_paired.py', '--end-time', str(global_end_time),
               '--base-model', '/kaggle/temp/arc026-base', '--challenger-model', str(merged)]
    state['evaluation'] = run_child(command, child_env, run_root / 'worker.log',
                                    hard_deadline, 'evaluation')
    state.update(status='completed' if state['evaluation']['return_code'] == 0 else 'evaluation_error',
                 return_code=state['evaluation']['return_code'])
save_state()
print(json.dumps(dict(event='paired_stages_finished', **state)), flush=True)
'''


FINAL = '''import hashlib, json, os, platform, shutil, time
from pathlib import Path
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

run_root = Path('/kaggle/working/026')
data = ArcDataset.from_file(str(run_root / 'input/challenges.json'))
state = json.loads((run_root / 'run-state.json').read_text())
state['arms'] = {}
for arm in ['base', 'challenger']:
    arm_root = run_root / arm
    store = arm_root / 'inference_outputs'
    file_order = os.listdir(store)
    (arm_root / 'candidate-file-order.json').write_text(json.dumps(file_order, indent=2))
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    original_listdir = os.listdir
    try:
        os.listdir = lambda path: list(file_order) if str(path) == str(store) else original_listdir(path)
        decoder.load_decoded_results(str(store))
    finally:
        os.listdir = original_listdir
    submission = data.get_submission(decoder.run_selection_algo())
    raw = (json.dumps(submission, ensure_ascii=False, sort_keys=True) + '\\n').encode()
    (arm_root / 'submission-B4.json').write_bytes(raw)
    count = 0
    with (arm_root / 'scored-candidates.jsonl').open('w') as output:
        for query_key, records in decoder.decoded_results.items():
            for record_id, record in records.items():
                source_file, index = record_id.rsplit('.out', 1)
                row = dict(query_key=query_key, record_id=record_id, source_file=source_file,
                           record_index=int(index), solution=record['solution'].tolist(),
                           score_aug=[float(value) for value in record['score_aug']],
                           beam_score=float(record['beam_score']))
                output.write(json.dumps(row, allow_nan=False) + '\\n')
                count += 1
    state['arms'][arm] = dict(scored_records=count, candidate_file_count=len(file_order),
                               prediction_sha256=hashlib.sha256(raw).hexdigest())
sources = run_root / 'sources'
for filename in ['arc_loader.py', 'arc_decoder.py', 'arc_solver.py', 'starter_paired.py',
                 'copy_model.py', 'train_specialist.py']:
    shutil.copy2(filename, sources / filename)
identities = {}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in {'artifact-identities.json', 'run-state.json'}:
        relative = path.relative_to(run_root)
        # Weight identities were recorded by training at save time. Avoid a
        # second full 7-GB checkpoint pass within the final output time reserve.
        if relative.parts[0] == 'training' and path.suffix in {'.safetensors', '.bin'}:
            identities[str(relative)] = dict(bytes=path.stat().st_size, sha256=None,
                                             note='Weight not rehashed; see training save/merge manifest')
        else:
            raw = path.read_bytes()
            identities[str(relative)] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
(run_root / 'artifact-identities.json').write_text(json.dumps(identities, indent=2))
state.update(finished_at=time.time(), elapsed_s=time.time()-notebook_started,
             planned_tasks=len(data.keys),
             planned_queries=sum(len(data.queries[key]['test']) for key in data.keys),
             python=platform.python_version(), gold_read=False)
(run_root / 'run-state.json').write_text(json.dumps(state, indent=2))
print(json.dumps(dict(event='paired_outputs_frozen', **state)), flush=True)
if state['return_code'] != 0:
    raise RuntimeError('Bounded paired experiment incomplete; inspect 026/run-state.json')
'''


def paired_solver(source):
    source = source.replace('/kaggle/working/025', RUN_ROOT)
    source = source.replace('ARC025_SOLVER_DEADLINE', 'ARC026_SOLVER_DEADLINE')
    source = replace_once(source, "import json as _observation_json\n",
                          "import json as _observation_json\n"
                          "_arm = os.environ['ARC026_ARM']\n"
                          "_arm_root = '/kaggle/working/026/' + _arm\n")
    source = replace_once(source,
                          "record = dict(timestamp=time.time(), rank=rank, event=event, task_id=task_id,",
                          "record = dict(timestamp=time.time(), arm=_arm, rank=rank,\n"
                          "                  physical_rank=int(os.environ['ARC026_PHYSICAL_RANK']),\n"
                          "                  event=event, task_id=task_id,")
    source = replace_once(source, "with open('/kaggle/working/026/events-rank%d.jsonl' % rank, 'a') as output:",
                          "with open(_arm_root + '/events-rank%d.jsonl' % rank, 'a') as output:")
    source = replace_once(source, "with open('/kaggle/working/026/raw-candidates-rank%d.jsonl' % rank, 'a') as output:",
                          "with open(_arm_root + '/raw-candidates-rank%d.jsonl' % rank, 'a') as output:")
    source = replace_once(source, 'model_name="/kaggle/temp/arc025-model",',
                          'model_name=os.environ["ARC026_MODEL_PATH"],')
    source = replace_once(source, '    dir_outputs = "/kaggle/working/026/inference_outputs"',
                          '    dir_outputs = _arm_root + "/inference_outputs"')
    source = replace_once(source, 'partial = os.path.join("/kaggle/working/026/incomplete", subkey)',
                          'partial = os.path.join(_arm_root + "/incomplete", subkey)')
    source = replace_once(source,
                          '    collator = QwenDataCollatorForCompletionOnlyLM(',
                          '    _event(rank, "model_and_fresh_ttt_adapter_ready", model_path=os.environ["ARC026_MODEL_PATH"])\n'
                          '    collator = QwenDataCollatorForCompletionOnlyLM(')
    return source


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ['owner', 'slug']:
        parser.add_argument('--' + name, required=True)
    for name in ['out', 'manifest', 'challenges', 'data-manifest']:
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--train-jsonl', '--train-data', dest='train_jsonl', required=True, type=Path)
    parser.add_argument('--trainer', type=Path, default=Path(__file__).with_name('train_specialist.py'))
    parser.add_argument('--base-package', type=Path, default=BASE)
    parser.add_argument('--session-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.session_seconds != 10800:
        parser.error('This frozen first paired design uses a 10800-second session')
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated third-party sources must stay outside the repository')
    if out.exists():
        raise FileExistsError(out)
    metadata = json.loads((args.base_package / 'kernel-metadata.json').read_text())
    prior = json.loads((args.base_package / 'package-manifest.json').read_text())
    raw = (args.base_package / metadata['code_file']).read_bytes()
    if sha(raw) != prior['notebook_sha256'] or prior['reference_sha256'] != REFERENCE_SHA:
        raise ValueError('Successful source package identity mismatch')
    notebook = json.loads(raw)
    challenge_raw, manifest_raw = args.challenges.read_bytes(), args.manifest.read_bytes()
    challenges, manifest = json.loads(challenge_raw), json.loads(manifest_raw)
    if any('output' in q for task in challenges.values() for q in task['test']):
        raise ValueError('Evaluation query outputs are prohibited')
    task_ids = sorted(challenges)
    n_queries = sum(len(task['test']) for task in challenges.values())
    if len(task_ids) != manifest['n_tasks'] or n_queries != manifest['n_queries']:
        raise ValueError('Panel manifest denominator mismatch')
    if task_ids != sorted(entry['task_id'] for entry in manifest['entries']):
        raise ValueError('Panel manifest task mismatch')
    train_raw, data_manifest_raw = args.train_jsonl.read_bytes(), args.data_manifest.read_bytes()
    data_manifest = json.loads(data_manifest_raw)
    if data_manifest.get('train_jsonl_sha256', sha(train_raw)) != sha(train_raw):
        raise ValueError('Training data differs from its frozen manifest')
    train_rows = [json.loads(line) for line in train_raw.decode().splitlines() if line.strip()]
    if not train_rows or any(row.get('split') != 'train' for row in train_rows):
        raise ValueError('Embed only the frozen training episode pool')
    if any(str(row['family_id']) in task_ids for row in train_rows):
        raise ValueError('Training family overlaps evaluation panel')
    input_identity = dict(challenges_sha256=sha(challenge_raw), manifest_sha256=sha(manifest_raw),
                          n_tasks=len(task_ids), n_queries=n_queries, task_ids=task_ids)
    training_identity = dict(train_sha256=sha(train_raw), manifest_sha256=sha(data_manifest_raw),
                             pool_episodes=len(train_rows), source_path=str(args.train_jsonl))
    notebook['cells'][0] = code_cell('import time\nnotebook_started = time.time()\n'
                                    'session_deadline = notebook_started + 10800\n'
                                    'training_stage_deadline = notebook_started + 3600\n'
                                    'global_end_time = training_stage_deadline\n')
    setup = '''from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess
run_root = Path('/kaggle/working/026')
for directory in ['input', 'sources', 'training']:
    (run_root / directory).mkdir(parents=True, exist_ok=True)
for arm in ['base', 'challenger']:
    for directory in ['inference_outputs', 'incomplete']:
        (run_root / arm / directory).mkdir(parents=True, exist_ok=True)
    for rank in range(2):
        (run_root / arm / ('events-rank%d.jsonl' % rank)).write_text('')
        (run_root / arm / ('raw-candidates-rank%d.jsonl' % rank)).write_text('')
raw_rerun_flag = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '')
if raw_rerun_flag.strip().lower() not in {'', '0', 'false', 'no'}:
    raise RuntimeError('Research paired package cannot be competition submitted')
os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
'''
    for name, content in [('manifest.json', manifest_raw), ('challenges.json', challenge_raw),
                          ('train.jsonl', train_raw), ('data-manifest.json', data_manifest_raw)]:
        setup += embed(RUN_ROOT + '/input/' + name, content)
    trainer_raw = args.trainer.read_bytes()
    setup += embed('train_specialist.py', trainer_raw)
    setup += 'input_identity = ' + repr(input_identity) + '\n'
    setup += 'training_identity = ' + repr(training_identity) + '\n'
    runtime = ''.join(notebook['cells'][2]['source'])
    runtime = runtime[runtime.index('import base64, hashlib, json, os, subprocess, sys\n'):]
    runtime = runtime.replace('/kaggle/working/025', RUN_ROOT).replace('/kaggle/temp/arc025-model', '/kaggle/temp/arc026-base')
    before = runtime[runtime.index('boot.update(session_seconds='):runtime.index("(run_root / 'boot.json')")]
    runtime = replace_once(runtime, before,
                           "boot.update(session_seconds=10800, training_stage_deadline=training_stage_deadline,\n"
                           "            evaluation_max_seconds=7200, session_deadline=session_deadline,\n"
                           "            input_identity=input_identity, training_identity=training_identity,\n"
                           "            raw_rerun_flag=raw_rerun_flag, gpu_inventory=gpu_read.stdout,\n"
                           "            gpu_inventory_return_code=gpu_read.returncode, research_only=True, gold_read=False)\n")
    notebook['cells'][2] = code_cell(setup + runtime)
    notebook['cells'][5] = code_cell(paired_solver(''.join(notebook['cells'][5]['source'])))
    notebook['cells'][6] = code_cell(STARTER)
    notebook['cells'][7] = code_cell(LAUNCH)
    notebook['cells'][8] = code_cell(FINAL)
    source_hashes = {}
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['execution_count'], cell['outputs'] = None, []
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            name = header.split()[1]
            ast.parse(body, filename=name)
            source_hashes[name] = sha(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)
    ast.parse(trainer_raw.decode())
    name = args.slug + '.ipynb'
    notebook_raw = (json.dumps(notebook, ensure_ascii=False, indent=2) + '\n').encode()
    if len(notebook_raw) >= 1000000:
        raise ValueError('Notebook exceeds Kaggle 1 MB source cap: %d bytes; freeze a smaller pool before packaging' % len(notebook_raw))
    metadata.update(id=args.owner + '/' + args.slug, title=args.slug.replace('-', ' '), code_file=name, is_private=True)
    result = dict(status='prepared_not_launched', ref=metadata['id'], notebook_sha256=sha(notebook_raw),
                  notebook_bytes=len(notebook_raw), reference_script_version_id=312063593,
                  reference_sha256=REFERENCE_SHA, base_package_notebook_sha256=sha(raw),
                  trainer_sha256=sha(trainer_raw), runtime_source_sha256=source_hashes,
                  input_identity=input_identity, training_identity=training_identity,
                  session_seconds=10800, training_stage_seconds=3600, evaluation_max_seconds=7200,
                  training_optimizer_step_cap=256, training_seconds=2700,
                  python_hash_seed=26026, arms=dict(base_physical_gpus=[0, 1], challenger_physical_gpus=[2, 3]),
                  preserved=['Original V1 loader and decoder B4', 'Original rank256 one-epoch per-task TTT',
                             'Original DFS, augmentation seeds1/2, rescoring, 1200s task soft check',
                             'Separate identical sorted queues; all query denominators retained'],
                  research_changes=['Bounded one-GPU offline continuation merged into ordinary BF16 checkpoint',
                                    'Fully exit training before fresh per-task TTT workers',
                                    'Paired two-worker arms with common global deadline and Python hash seed',
                                    'Original import serialization uses physical rank; worker rank local to each arm',
                                    'Deterministic compressed training/input embedding with SHA checks'],
                  warning='Development paired public panel; upstream model exposure exists; this is not a blind or formal competition score.')
    out.mkdir(parents=True)
    (out / name).write_bytes(notebook_raw)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'package-manifest.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
