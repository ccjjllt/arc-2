#!/usr/bin/env python3
"""Prepare one original-weight sequential TTT-view ensemble confirmation.

Reads frozen sources and label-free inputs; never launches compute or submits.
Generated third-party notebook sources must remain outside the repository.
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
RUN_ROOT = '/kaggle/working/027'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected one source anchor: ' + repr(before[:120]))
    return source.replace(before, after, 1)


def code_cell(source):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=source.splitlines(keepends=True))


def embed(destination, raw):
    packed = base64.b64encode(gzip.compress(raw, mtime=0)).decode()
    return ('payload = gzip.decompress(base64.b64decode(' + repr(packed) + '))\n'
            'if hashlib.sha256(payload).hexdigest() != ' + repr(sha(raw)) + ':\n'
            '    raise RuntimeError("Embedded artifact identity mismatch")\n'
            'Path(' + repr(destination) + ').write_bytes(payload)\n')


def solver_source(source):
    source = source.replace('/kaggle/working/025', RUN_ROOT)
    source = source.replace('/kaggle/temp/arc025-model', '/kaggle/temp/arc027-model')
    source = source.replace('ARC025_SOLVER_DEADLINE', 'ARC027_SOLVER_DEADLINE')
    source = replace_once(source, 'import json as _observation_json\n',
                          'import json as _observation_json\n'
                          '_branch = os.environ["ARC027_BRANCH"]\n'
                          '_ttt_aug_seed = int(os.environ["ARC027_TTT_AUG_SEED"])\n'
                          'if (_branch, _ttt_aug_seed) not in [("base", 1), ("extra", 2)]:\n'
                          '    raise RuntimeError("Unexpected frozen TTT-view branch")\n'
                          '_branch_root = "/kaggle/working/027/" + _branch\n')
    source = replace_once(source,
                          'record = dict(timestamp=time.time(), rank=rank, event=event, task_id=task_id,',
                          'record = dict(timestamp=time.time(), branch=_branch, ttt_augmentation_seed=_ttt_aug_seed,\n'
                          '                  rank=rank, event=event, task_id=task_id,')
    source = replace_once(source,
                          "with open('/kaggle/working/027/events-rank%d.jsonl' % rank, 'a') as output:",
                          "with open(_branch_root + '/events-rank%d.jsonl' % rank, 'a') as output:")
    source = replace_once(source,
                          "with open('/kaggle/working/027/raw-candidates-rank%d.jsonl' % rank, 'a') as output:",
                          "with open(_branch_root + '/raw-candidates-rank%d.jsonl' % rank, 'a') as output:")
    source = replace_once(source,
                          '    dir_outputs = "/kaggle/working/027/inference_outputs"',
                          '    dir_outputs = _branch_root + "/inference_outputs"')
    source = replace_once(source,
                          'partial = os.path.join("/kaggle/working/027/incomplete", subkey)',
                          'partial = os.path.join(_branch_root + "/incomplete", subkey)')
    source = replace_once(source,
                          '        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=1)',
                          '        train_ds = puzzle_ds.augment(n=16, shfl_keys=True, seed=_ttt_aug_seed)')
    return source


LAUNCH = '''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path

run_root = Path('/kaggle/working/027')
state = dict(notebook_started=notebook_started, session_deadline=session_deadline,
             global_end_time=global_end_time, solver_hard_deadline=solver_hard_deadline,
             input_identity=input_identity, python_hash_seed=27027,
             source='Original V1 312063593 / successful 025 wrapper',
             policy='Sequential original-weight TTT-view ensemble; base seed1 then extra seed2',
             research_only=True, evaluation_gold_read=False)

def save_state():
    (run_root / 'run-state.json').write_text(json.dumps(state, indent=2))

def run_branch(branch, augmentation_seed):
    record = dict(branch=branch, ttt_augmentation_seed=augmentation_seed,
                  started_at=time.time(), global_end_time=global_end_time,
                  hard_deadline=solver_hard_deadline, timed_out=False)
    if time.time() >= global_end_time:
        return dict(record, status='skipped_no_solver_budget', return_code=None)
    # Original import markers belong to a process group, not a whole notebook.
    # The prior branch has fully exited before clearing them and loading afresh.
    for rank in range(4):
        Path('/kaggle/worker%d' % rank).unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.update(UNSLOTH_DISABLE_STATISTICS='1',
                       TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                       OMP_NUM_THREADS='12', PYTHONUNBUFFERED='1',
                       PYTHONHASHSEED='27027', ARC027_BRANCH=branch,
                       ARC027_TTT_AUG_SEED=str(augmentation_seed),
                       ARC027_SOLVER_DEADLINE=str(global_end_time))
    command = [sys.executable, '-u', 'starter.py', '--end-time', str(global_end_time)]
    record['command'] = command
    with (run_root / branch / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, errors='replace',
                                   bufsize=1, start_new_session=True)
        record['process_group_id'] = process.pid
        def tee():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print('[' + branch + '] ' + line, end='', flush=True)
        thread = threading.Thread(target=tee, daemon=True)
        thread.start()
        try:
            record['return_code'] = process.wait(timeout=max(0.1, solver_hard_deadline-time.time()))
        except subprocess.TimeoutExpired:
            record.update(timed_out=True, status='shared_global_hard_deadline')
            os.killpg(process.pid, signal.SIGKILL)
            record['return_code'] = process.wait(timeout=10)
        finally:
            thread.join(timeout=2)
    record.setdefault('status', 'completed' if record['return_code'] == 0 else 'solver_error')
    record.update(finished_at=time.time(), elapsed_s=time.time()-record['started_at'])
    return record

# No half-budget cut is imposed: base receives the original entire remaining
# session budget. A second pass only runs after all base workers have exited.
state['base'] = run_branch('base', 1)
save_state()
if state['base']['return_code'] == 0:
    state['extra'] = run_branch('extra', 2)
else:
    state['extra'] = dict(branch='extra', ttt_augmentation_seed=2,
                          status='skipped_base_failed', return_code=None, timed_out=False)
if state['base']['return_code'] != 0:
    state.update(status='base_error_partial_outputs', return_code=1)
elif state['extra'].get('timed_out'):
    state.update(status='completed_extra_deadline_partial', return_code=0)
elif state['extra']['return_code'] is None:
    state.update(status='completed_extra_skipped', return_code=0)
elif state['extra']['return_code'] != 0:
    state.update(status='extra_error_partial_outputs', return_code=1)
else:
    state.update(status='completed', return_code=0)
state['solver_finished_at'] = time.time()
save_state()
print(json.dumps(dict(event='sequential_view_stages_finished', **state)), flush=True)
'''


FINAL = '''import hashlib, json, os, platform, shutil, time
from pathlib import Path
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
from freeze_ensemble import select_ensemble_query

run_root = Path('/kaggle/working/027')
data = ArcDataset.from_file(str(run_root / 'input/challenges.json'))
state = json.loads((run_root / 'run-state.json').read_text())
state['arms'] = {}
submissions, candidate_counts = {}, {}
for branch in ['base', 'extra']:
    branch_root = run_root / branch
    store = branch_root / 'inference_outputs'
    file_order = os.listdir(store)
    (branch_root / 'candidate-file-order.json').write_text(json.dumps(file_order, indent=2))
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    original_listdir = os.listdir
    try:
        os.listdir = lambda path: list(file_order) if str(path) == str(store) else original_listdir(path)
        decoder.load_decoded_results(str(store))
    finally:
        os.listdir = original_listdir
    submissions[branch] = data.get_submission(decoder.run_selection_algo())
    raw = (json.dumps(submissions[branch], ensure_ascii=False, sort_keys=True) + '\\n').encode()
    (branch_root / 'submission-B4.json').write_bytes(raw)
    candidate_counts[branch] = {key: len(records) for key, records in decoder.decoded_results.items()}
    count = 0
    with (branch_root / 'scored-candidates.jsonl').open('w') as output:
        for query_key, records in decoder.decoded_results.items():
            for record_id, record in records.items():
                source_file, index = record_id.rsplit('.out', 1)
                row = dict(query_key=query_key, record_id=record_id, source_file=source_file,
                           record_index=int(index), solution=record['solution'].tolist(),
                           score_aug=[float(value) for value in record['score_aug']],
                           beam_score=float(record['beam_score']))
                output.write(json.dumps(row, allow_nan=False) + '\\n')
                count += 1
    state['arms'][branch] = dict(scored_records=count, candidate_file_count=len(file_order),
                                 prediction_sha256=hashlib.sha256(raw).hexdigest(),
                                 queries_with_real_candidates=sum(value > 0 for value in candidate_counts[branch].values()))
ensemble, decisions = {}, []
for task_id in sorted(submissions['base']):
    ensemble[task_id] = []
    for index, base_slots in enumerate(submissions['base'][task_id]):
        key = task_id + '_' + str(index)
        slots, audit = select_ensemble_query(base_slots, submissions['extra'][task_id][index],
                    candidate_counts['base'].get(key, 0), candidate_counts['extra'].get(key, 0))
        ensemble[task_id].append(slots)
        decisions.append(dict(task_id=task_id, query_index=index, query_key=key, **audit))
raw = (json.dumps(ensemble, ensure_ascii=False, sort_keys=True) + '\\n').encode()
(run_root / 'submission-ensemble.json').write_bytes(raw)
Path('/kaggle/working/submission.json').write_bytes(raw)
with (run_root / 'ensemble-decisions.jsonl').open('w') as output:
    for decision in decisions:
        output.write(json.dumps(decision, allow_nan=False) + '\\n')
sources = run_root / 'sources'
for name in ['arc_loader.py', 'arc_decoder.py', 'arc_solver.py', 'starter.py',
             'copy_model.py', 'freeze_ensemble.py']:
    shutil.copy2(name, sources / name)
identities = {}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in {'artifact-identities.json', 'run-state.json'}:
        content = path.read_bytes()
        identities[str(path.relative_to(run_root))] = dict(bytes=len(content),
                                         sha256=hashlib.sha256(content).hexdigest())
(run_root / 'artifact-identities.json').write_text(json.dumps(identities, indent=2))
state.update(finished_at=time.time(), elapsed_s=time.time()-notebook_started,
             planned_tasks=len(data.keys),
             planned_queries=sum(len(data.queries[key]['test']) for key in data.keys),
             ensemble_prediction_sha256=hashlib.sha256(raw).hexdigest(),
             ensemble_decisions=len(decisions), python=platform.python_version(), evaluation_gold_read=False)
(run_root / 'run-state.json').write_text(json.dumps(state, indent=2))
print(json.dumps(dict(event='sequential_view_outputs_frozen', **state)), flush=True)
if state['return_code'] != 0:
    raise RuntimeError('Sequential-view experiment has solver error; partial outputs frozen in 027')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('owner', 'slug'):
        parser.add_argument('--'+name, required=True)
    for name in ('out', 'manifest', 'challenges'):
        parser.add_argument('--'+name, required=True, type=Path)
    parser.add_argument('--base-package', type=Path, default=BASE)
    parser.add_argument('--selection-helper', type=Path, default=Path(__file__).with_name('freeze_ensemble.py'))
    parser.add_argument('--session-seconds', type=int, default=10800)
    args = parser.parse_args()
    if args.session_seconds != 10800:
        parser.error('This frozen sequential-view confirmation uses a 10800-second session')
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated third-party notebook sources must remain outside the repository')
    if out.exists():
        raise FileExistsError(out)
    prior = json.loads((args.base_package/'package-manifest.json').read_text())
    metadata = json.loads((args.base_package/'kernel-metadata.json').read_text())
    prior_raw = (args.base_package/metadata['code_file']).read_bytes()
    if sha(prior_raw) != prior['notebook_sha256'] or prior['reference_sha256'] != REFERENCE_SHA:
        raise ValueError('Successful 025 package identity mismatch')
    notebook = json.loads(prior_raw)
    manifest_raw, challenge_raw = args.manifest.read_bytes(), args.challenges.read_bytes()
    manifest, challenges = json.loads(manifest_raw), json.loads(challenge_raw)
    if not isinstance(challenges, dict) or not challenges:
        raise ValueError('Expected nonempty challenge map')
    if any(set(query) != {'input'} for task in challenges.values() for query in task['test']):
        raise ValueError('Evaluation query answer or unexpected fields prohibited')
    task_ids = sorted(challenges)
    n_queries = sum(len(task['test']) for task in challenges.values())
    if sorted(entry['task_id'] for entry in manifest['entries']) != task_ids:
        raise ValueError('Manifest task identities differ')
    if manifest['n_tasks'] != len(task_ids) or manifest['n_queries'] != n_queries:
        raise ValueError('Manifest query/task denominators differ')
    if manifest.get('input_artifact', {}).get('sha256', sha(challenge_raw)) != sha(challenge_raw):
        raise ValueError('Manifest input SHA differs')
    forbidden = json.loads((ROOT/'experiments/026/panel-v1/manifest.json').read_text())
    if set(task_ids)&{entry['task_id'] for entry in forbidden['entries']}:
        raise ValueError('026 sixteen-task panel cannot enter this new confirmation')
    helper_raw = args.selection_helper.read_bytes()
    helper_tree = ast.parse(helper_raw.decode())
    if len([n for n in helper_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'select_ensemble_query']) != 1:
        raise ValueError('Expected frozen standalone selection helper function')
    input_identity = dict(challenges_sha256=sha(challenge_raw), manifest_sha256=sha(manifest_raw),
                          n_tasks=len(task_ids), n_queries=n_queries, task_ids=task_ids)
    notebook['cells'][0] = code_cell('import time\nnotebook_started = time.time()\n'
                                    'session_deadline = notebook_started + 10800\n'
                                    'global_end_time = session_deadline - 120\n'
                                    'solver_hard_deadline = session_deadline - 60\n')
    setup = '''from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess
run_root = Path('/kaggle/working/027')
for directory in ['input', 'sources']:
    (run_root / directory).mkdir(parents=True, exist_ok=True)
for branch in ['base', 'extra']:
    for directory in ['inference_outputs', 'incomplete']:
        (run_root / branch / directory).mkdir(parents=True, exist_ok=True)
    for rank in range(4):
        (run_root / branch / ('events-rank%d.jsonl' % rank)).write_text('')
        (run_root / branch / ('raw-candidates-rank%d.jsonl' % rank)).write_text('')
raw_rerun_flag = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '')
if raw_rerun_flag.strip().lower() not in {'', '0', 'false', 'no'}:
    raise RuntimeError('Research sequential-view confirmation cannot be competition submitted')
os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
'''
    for name, content in [('manifest.json', manifest_raw), ('challenges.json', challenge_raw)]:
        setup += embed(RUN_ROOT+'/input/'+name, content)
    setup += embed('freeze_ensemble.py', helper_raw)
    setup += 'input_identity = '+repr(input_identity)+'\n'
    runtime = ''.join(notebook['cells'][2]['source'])
    runtime = runtime[runtime.index('import base64, hashlib, json, os, subprocess, sys\n'):]
    runtime = runtime.replace('/kaggle/working/025', RUN_ROOT).replace('/kaggle/temp/arc025-model', '/kaggle/temp/arc027-model')
    before = runtime[runtime.index('boot.update(session_seconds='):runtime.index("(run_root / 'boot.json')")]
    runtime = replace_once(runtime, before,
                           'boot.update(session_seconds=10800, solver_global_seconds=10680, solver_hard_seconds=10740,\n'
                           '            session_deadline=session_deadline, input_identity=input_identity, raw_rerun_flag=raw_rerun_flag,\n'
                           '            gpu_inventory=gpu_read.stdout, gpu_inventory_return_code=gpu_read.returncode,\n'
                           '            research_only=True, evaluation_gold_read=False, python_hash_seed=27027,\n'
                           '            ttt_augmentation_seeds=dict(base=1, extra=2), sequential_four_gpu_passes=True,\n'
                           '            offline_continuation_training=False)\n')
    runtime = runtime.replace("event='faithful_v1_boot'", "event='sequential_view_boot'")
    notebook['cells'][2] = code_cell(setup+runtime)
    notebook['cells'][5] = code_cell(solver_source(''.join(notebook['cells'][5]['source'])))
    notebook['cells'][6] = code_cell(''.join(notebook['cells'][6]['source']).replace('/kaggle/working/025', RUN_ROOT))
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
    name = args.slug+'.ipynb'
    notebook_raw = (json.dumps(notebook, ensure_ascii=False, indent=2)+'\n').encode()
    if len(notebook_raw) >= 1000000:
        raise ValueError('Notebook exceeds Kaggle 1MB source cap')
    metadata.update(id=args.owner+'/'+args.slug, title=args.slug.replace('-', ' '), code_file=name, is_private=True)
    if metadata['kernel_sources'] != ['sorokin/pip-install-unsloth-flash-patch']:
        raise ValueError('This method must not attach 026 checkpoint outputs')
    result = dict(status='prepared_not_launched', ref=metadata['id'], notebook_sha256=sha(notebook_raw),
                  notebook_bytes=len(notebook_raw), reference_script_version_id=312063593,
                  reference_sha256=REFERENCE_SHA, base_package_notebook_sha256=sha(prior_raw),
                  input_identity=input_identity, selection_helper_sha256=sha(helper_raw),
                  runtime_source_sha256=source_hashes, session_seconds=10800,
                  solver_global_seconds=10680, solver_hard_seconds=10740, python_hash_seed=27027,
                  branches=dict(base=dict(gpus=[0,1,2,3],ttt_augmentation_seed=1),
                                extra=dict(gpus=[0,1,2,3],ttt_augmentation_seed=2)),
                  scheduling='Base four-worker original sorted queue first; extra same queue only after base process group exits, using remaining shared session budget; no half-budget base cutoff.',
                  preserved=['Original sft139 BF16 V1 only; no offline-trained weight',
                             'Original rank256 128-step TTT, random_state42 and Trainer seed42',
                             'Original DFS n2 seed2, rescoring, 1200-second task soft check',
                             'Original loader/decoder B4 for each branch, original four-worker starter'],
                  changes=['TTT data augmentation seed1 versus seed2 across sequential passes',
                           'Fixed common Python hash seed; per-branch atomic candidates and event logs',
                           'Frozen no-parameter base-first/extra-first-slot ensemble helper'],
                  warning='One-shot public strategy confirmation; all planned queries retain denominator even if extra deadline truncates. Not a blind or formal competition score.')
    out.mkdir(parents=True)
    (out/name).write_bytes(notebook_raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
