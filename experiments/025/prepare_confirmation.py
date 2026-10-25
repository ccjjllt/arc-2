"""Package one manual NVARC V1 candidate-selection confirmation run.

The third-party solver is read from the frozen 023 package and is written only
to the requested temporary package directory. This command never launches GPU
work or submits to Kaggle. No gold answers are embedded or read by the notebook.
"""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
BASE = Path('/private/tmp/arc2-023-faithful-v1-frozen')
REFERENCE_SHA = '71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d'
RUN_ROOT = '/kaggle/working/025'
CHALLENGES = RUN_ROOT + '/input/challenges.json'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def replace_once(source, before, after):
    if source.count(before) != 1:
        raise ValueError('Expected exactly one source anchor: ' + repr(before[:100]))
    return source.replace(before, after, 1)


def code_cell(source):
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
            'outputs': [], 'source': source.splitlines(keepends=True)}


OBSERVERS = '''
import json as _observation_json

def _event(rank, event, task_id=None, **fields):
    record = dict(timestamp=time.time(), rank=rank, event=event, task_id=task_id,
                  global_end_time=float(os.environ['ARC025_SOLVER_DEADLINE']), **fields)
    with open('/kaggle/working/025/events-rank%d.jsonl' % rank, 'a') as output:
        output.write(_observation_json.dumps(record, allow_nan=False) + '\\n')

def _raw_candidate(rank, query_key, subkey, beam_score, solution):
    record = dict(timestamp=time.time(), query_key=query_key, source_subkey=subkey,
                  beam_score=float(beam_score), solution=solution.tolist())
    with open('/kaggle/working/025/raw-candidates-rank%d.jsonl' % rank, 'a') as output:
        output.write(_observation_json.dumps(record, allow_nan=False) + '\\n')

'''


def instrument_solver(source):
    source = replace_once(source, '/kaggle/temp/arc023-model', '/kaggle/temp/arc025-model')
    source = replace_once(source, 'def worker(rank, queue, end_time):',
                          OBSERVERS + 'def worker(rank, queue, end_time):')
    source = replace_once(source, '    arc_test_set = ArcDataset.from_file(test_path)',
                          '    test_path = ' + repr(CHALLENGES) + '\n'
                          '    arc_test_set = ArcDataset.from_file(test_path)')
    source = replace_once(source, '    dir_outputs = "/kaggle/inference_outputs"',
                          '    dir_outputs = "/kaggle/working/025/inference_outputs"')
    source = replace_once(source, '        start_time = time.time()',
                          '        start_time = time.time()\n'
                          '        _event(rank, "task_start", key)')
    source = replace_once(source, '        model = FastLanguageModel.for_training(model)',
                          '        _event(rank, "ttt_start", key)\n'
                          '        model = FastLanguageModel.for_training(model)')
    source = replace_once(source, '            stats = trainer.train()',
                          '            stats = trainer.train()\n'
                          '            _event(rank, "ttt_end", key, global_step=int(stats.global_step),\n'
                          '                   training_loss=float(stats.training_loss))')
    source = replace_once(source,
                          '                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)',
                          '                _event(rank, "search_start", key, subkeys=subkeys,\n'
                          '                       query_keys=list(dict.fromkeys(s.split(".")[0] for s in subkeys)))\n'
                          '                dfs_result = inference_turbo_dfs(model, tokens, max_new_tokens, max_score, end_time)\n'
                          '                _event(rank, "search_end", key, subkeys=subkeys,\n'
                          '                       returned_beams=sum(len(beams) for _, beams in dfs_result))\n'
                          '                raw_valid = 0\n'
                          '                for raw_subkey_id, raw_beams in dfs_result:\n'
                          '                    raw_subkey = subkeys[raw_subkey_id]\n'
                          '                    for raw_beam_score, raw_tokens in raw_beams:\n'
                          '                        raw_array = formatter.convert_tokens_to_array(raw_tokens)\n'
                          '                        if raw_array is not None:\n'
                          '                            raw_solution = puzzle_ds_multi.invert_mod(raw_array, raw_subkey, inv_perm=True)\n'
                          '                            _raw_candidate(rank, raw_subkey.split(".")[0], raw_subkey, raw_beam_score, raw_solution)\n'
                          '                            raw_valid += 1\n'
                          '                _event(rank, "raw_batch_saved", key, subkeys=subkeys, legal_raw_records=raw_valid)')
    source = replace_once(source,
                          '                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")',
                          '                            _event(rank, "score_start", key, query_key=bk, subkey=subkey)\n'
                          '                            print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")')
    source = replace_once(source,
                          '                            known_scores[grid_id] = augmented_scores',
                          '                            known_scores[grid_id] = augmented_scores\n'
                          '                            _event(rank, "score_end", key, query_key=bk, subkey=subkey)')
    source = replace_once(source,
                          '                        with bz2.BZ2File(os.path.join(dir_outputs, subkey), "w") as f:\n'
                          '                            pickle.dump(decoded_result, f)',
                          '                        partial = os.path.join("/kaggle/working/025/incomplete", subkey)\n'
                          '                        with bz2.BZ2File(partial, "w") as f:\n'
                          '                            pickle.dump(decoded_result, f)\n'
                          '                        os.replace(partial, os.path.join(dir_outputs, subkey))\n'
                          '                        _event(rank, "scored_file_saved", key, query_key=bk,\n'
                          '                               subkey=subkey, records=len(decoded_result))')
    source = replace_once(source,
                          '        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")',
                          '        print(f"[Rank {rank}] finished {key} in {spend_time:.1f}s")\n'
                          '        _event(rank, "task_end", key, elapsed_s=spend_time)')
    return source


FINAL = '''import bz2, hashlib, json, os, pickle, platform, shutil, time
from pathlib import Path
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

run_root = Path('/kaggle/working/025')
data = ArcDataset.from_file(str(run_root / 'input/challenges.json'))
store = run_root / 'inference_outputs'
file_order = os.listdir(store)
(run_root / 'candidate-file-order.json').write_text(json.dumps(file_order, indent=2))
decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
# Freeze the exact file traversal used by the original decoder on this filesystem.
original_listdir = os.listdir
try:
    os.listdir = lambda path: list(file_order) if str(path) == str(store) else original_listdir(path)
    decoder.load_decoded_results(str(store))
finally:
    os.listdir = original_listdir
submission = data.get_submission(decoder.run_selection_algo())
prediction_bytes = (json.dumps(submission, ensure_ascii=False, sort_keys=True) + '\\n').encode()
(run_root / 'submission-B4.json').write_bytes(prediction_bytes)
Path('/kaggle/working/submission.json').write_bytes(prediction_bytes)
record_count = 0
with (run_root / 'scored-candidates.jsonl').open('w') as output:
    for query_key, records in decoder.decoded_results.items():
        for record_id, record in records.items():
            source_file, record_index = record_id.rsplit('.out', 1)
            row = dict(query_key=query_key, record_id=record_id, source_file=source_file,
                       record_index=int(record_index), solution=record['solution'].tolist(),
                       score_aug=[float(value) for value in record['score_aug']],
                       beam_score=float(record['beam_score']))
            output.write(json.dumps(row, allow_nan=False) + '\\n')
            record_count += 1
sources = run_root / 'sources'
sources.mkdir(exist_ok=True)
for filename in ['arc_loader.py', 'arc_decoder.py', 'arc_solver.py', 'starter.py', 'copy_model.py']:
    shutil.copy2(filename, sources / filename)
state = json.loads((run_root / 'run-state.json').read_text())
state.update(finished_at=time.time(), elapsed_s=time.time()-notebook_started,
             scored_records=record_count, prediction_sha256=hashlib.sha256(prediction_bytes).hexdigest(),
             candidate_file_count=len(file_order), planned_tasks=len(data.keys),
             planned_queries=sum(len(data.queries[key]['test']) for key in data.keys),
             python=platform.python_version(), gold_read=False)
identities = {}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in {'artifact-identities.json', 'run-state.json'}:
        raw = path.read_bytes()
        identities[str(path.relative_to(run_root))] = dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw))
(run_root / 'artifact-identities.json').write_text(json.dumps(identities, indent=2))
(run_root / 'run-state.json').write_text(json.dumps(state, indent=2))
print(json.dumps(dict(event='confirmation_outputs_frozen', **state)), flush=True)
if state['return_code'] != 0:
    raise RuntimeError('Solver did not complete: partial outputs were frozen; inspect 025/run-state.json')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--challenges', required=True, type=Path)
    parser.add_argument('--base-package', type=Path, default=BASE)
    parser.add_argument('--session-seconds', type=int, default=7200)
    args = parser.parse_args()
    if not 600 <= args.session_seconds <= 7200:
        parser.error('--session-seconds must be between 600 and 7200')
    session_seconds = args.session_seconds
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated third-party sources must stay outside the repository')
    if out.exists():
        raise FileExistsError(out)
    base_manifest = json.loads((args.base_package / 'package-manifest.json').read_text())
    base_metadata = json.loads((args.base_package / 'kernel-metadata.json').read_text())
    base_bytes = (args.base_package / base_metadata['code_file']).read_bytes()
    if base_manifest['reference_sha256'] != REFERENCE_SHA or sha(base_bytes) != base_manifest['notebook_sha256']:
        raise ValueError('Frozen 023 package identity mismatch')
    notebook = json.loads(base_bytes)
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    challenge_bytes = args.challenges.read_bytes()
    challenges = json.loads(challenge_bytes)
    if not challenges or not isinstance(challenges, dict):
        raise ValueError('Expected nonempty task dictionary')
    task_ids = sorted(challenges)
    if any('output' in query for task in challenges.values() for query in task['test']):
        raise ValueError('Research test queries must not contain output labels')
    if manifest.get('n_tasks') != len(challenges):
        raise ValueError('Manifest task count mismatch')
    if sorted(entry['task_id'] for entry in manifest['entries']) != task_ids:
        raise ValueError('Manifest task identities mismatch')
    query_count = sum(len(task['test']) for task in challenges.values())
    if manifest.get('n_queries') != query_count:
        raise ValueError('Manifest query count mismatch')
    input_identity = dict(challenges_sha256=sha(challenge_bytes), manifest_sha256=sha(manifest_bytes),
                          n_tasks=len(task_ids), n_queries=query_count, task_ids=task_ids)
    notebook['cells'][0] = code_cell('import time\nnotebook_started = time.time()\n'
                                    f'session_deadline = notebook_started + {session_seconds}\n'
                                    'global_end_time = session_deadline - 120\n'
                                    'solver_hard_deadline = session_deadline - 60\n')
    runtime = ''.join(notebook['cells'][2]['source'])
    input_setup = '''from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess
run_root = Path('/kaggle/working/025')
for directory in ['input', 'inference_outputs', 'incomplete', 'sources']:
    (run_root / directory).mkdir(parents=True, exist_ok=True)
for rank in range(4):
    (run_root / ('events-rank%d.jsonl' % rank)).write_text('')
    (run_root / ('raw-candidates-rank%d.jsonl' % rank)).write_text('')
raw_rerun_flag = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '')
if raw_rerun_flag.strip().lower() not in {'', '0', 'false', 'no'}:
    raise RuntimeError('Research confirmation package cannot be competition submitted')
# The original solver treats any nonempty rerun string as true.
os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
'''
    for filename, raw in [('manifest.json', manifest_bytes), ('challenges.json', challenge_bytes)]:
        compressed = gzip.compress(raw, mtime=0)
        input_setup += 'payload = gzip.decompress(base64.b64decode(' + repr(base64.b64encode(compressed).decode()) + '))\n'
        input_setup += 'if hashlib.sha256(payload).hexdigest() != ' + repr(sha(raw)) + ':\n    raise RuntimeError("Input identity mismatch")\n'
        input_setup += '(run_root / "input" / ' + repr(filename) + ').write_bytes(payload)\n'
    input_setup += 'input_identity = ' + repr(input_identity) + '\n'
    runtime = runtime.replace('/kaggle/temp/arc023-model', '/kaggle/temp/arc025-model')
    runtime = replace_once(runtime,
                           "utility = Path('/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch')\n"
                           "if not (utility / 'unsloth/__init__.py').is_file():\n"
                           "    raise RuntimeError('Original mounted Unsloth utility is unavailable')\n",
                           """import time
mount_lookup_started = time.monotonic()
mount_lookup_deadline = mount_lookup_started + 60
def same_utility_name(path):
    return 'pip_install_unsloth_flash_patch' in str(path).replace('-', '_')
known_utility = Path('/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch')
while True:
    utility_candidates = set()
    if (known_utility / 'unsloth/__init__.py').is_file():
        utility_candidates.add(str(known_utility.resolve()))
    else:
        possible = [Path(entry or '.') for entry in sys.path]
        for parent, pattern in [('/kaggle/usr/lib/notebooks', '*/*'),
                                ('/kaggle/input/notebooks', '*/*'),
                                ('/kaggle/input', '*')]:
            if Path(parent).is_dir():
                possible.extend(Path(parent).glob(pattern))
        for path in possible:
            if same_utility_name(path) and (path / 'unsloth/__init__.py').is_file():
                utility_candidates.add(str(path.resolve()))
    lookup_record = dict(event='original_utility_mount_lookup', roots=sorted(utility_candidates),
                         elapsed_s=time.monotonic()-mount_lookup_started, sys_path=sys.path)
    (run_root / 'mount-diagnostics.json').write_text(json.dumps(lookup_record, indent=2))
    print(json.dumps(lookup_record), flush=True)
    if utility_candidates or time.monotonic() >= mount_lookup_deadline:
        break
    time.sleep(min(2, max(0, mount_lookup_deadline-time.monotonic())))
if len(utility_candidates) != 1:
    raise RuntimeError('Expected exactly one mounted original Unsloth utility; roots=' + repr(sorted(utility_candidates)))
utility = Path(next(iter(utility_candidates)))
""")
    runtime = replace_once(runtime,
                           "model_source = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')\n"
                           "if not (model_source / 'config.json').is_file():\n"
                           "    raise RuntimeError('Expected fixed sft139 BF16 V1 checkpoint mount is unavailable')\n",
                           """model_source = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')
while True:
    if (model_source / 'config.json').is_file():
        candidates = {str(model_source.resolve())}
    else:
        candidates = {str(path.parent.resolve()) for path in Path('/kaggle/input/models').glob(
            '*/qwen3_4b_grids15_sft139/transformers/bfloat16/1/config.json')}
    if candidates or time.monotonic() >= mount_lookup_deadline:
        break
    time.sleep(min(2, max(0, mount_lookup_deadline-time.monotonic())))
if len(candidates) != 1:
    raise RuntimeError('Expected exactly one fixed sft139 BF16 V1 checkpoint; roots=' + repr(sorted(candidates)))
model_source = Path(next(iter(candidates)))
lookup_record.update(model_roots=sorted(candidates), elapsed_s=time.monotonic()-mount_lookup_started)
(run_root / 'mount-diagnostics.json').write_text(json.dumps(lookup_record, indent=2))
if hashlib.sha256((model_source / 'config.json').read_bytes()).hexdigest() != '932e9430395ba3eea4087cba1769241bc7d0dc8d574846862d6983659456a2d0':
    raise RuntimeError('Mounted fixed model config differs from successful 023 V1')
""")
    runtime = replace_once(runtime, 'notebook_started=global_end_time-42600', 'notebook_started=notebook_started')
    runtime = replace_once(runtime, "Path('023-boot.json').write_text(json.dumps(boot, indent=2))",
                           "gpu_read = subprocess.run(['nvidia-smi', '--query-gpu=index,name,memory.total', '--format=csv,noheader'], capture_output=True, text=True)\n"
                           f"boot.update(session_seconds={session_seconds}, solver_global_seconds={session_seconds-120}, solver_hard_seconds={session_seconds-60},\n"
                           "            session_deadline=session_deadline, input_identity=input_identity, raw_rerun_flag=raw_rerun_flag,\n"
                           "            gpu_inventory=gpu_read.stdout, gpu_inventory_return_code=gpu_read.returncode,\n"
                           "            research_only=True, gold_read=False)\n"
                           "(run_root / 'boot.json').write_text(json.dumps(boot, indent=2))")
    runtime = runtime.replace('/kaggle/working/023-model-copy.json', RUN_ROOT + '/model-copy.json')
    notebook['cells'][2] = code_cell(input_setup + runtime)
    solver_source = ''.join(notebook['cells'][5]['source'])
    notebook['cells'][5] = code_cell(instrument_solver(solver_source))
    starter_source = ''.join(notebook['cells'][6]['source'])
    starter_source = replace_once(starter_source,
                                  '    with open(test_path, "r") as f:',
                                  '    test_path = ' + repr(CHALLENGES) + '\n'
                                  '    with open(test_path, "r") as f:')
    starter_source = replace_once(starter_source,
                                  '        if not rerun_mode:\n'
                                  '            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:\n'
                                  '                continue\n', '')
    notebook['cells'][6] = code_cell(starter_source)
    notebook['cells'][7] = code_cell('''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
solver_env = dict(os.environ)
solver_env.update(UNSLOTH_DISABLE_STATISTICS='1', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
                  OMP_NUM_THREADS='12', PYTHONUNBUFFERED='1', ARC025_SOLVER_DEADLINE=str(global_end_time))
state = dict(solver_started_at=time.time(), notebook_started=notebook_started,
             global_end_time=global_end_time, hard_deadline=solver_hard_deadline,
             session_deadline=session_deadline, input_identity=input_identity,
             source='Frozen faithful 023 / original V1 312063593', timed_out=False)
log_path = Path('/kaggle/working/025/worker.log')
remaining = solver_hard_deadline - time.time()
if remaining <= 0:
    state.update(return_code=-1, status='setup_exhausted_budget')
else:
    with log_path.open('w') as log:
        process = subprocess.Popen([sys.executable, '-u', 'starter.py', '--end-time', str(global_end_time)],
                                   env=solver_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', bufsize=1, start_new_session=True)
        state['worker_pgid'] = process.pid
        def tee_worker_output():
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', flush=True)
        tee_thread = threading.Thread(target=tee_worker_output, daemon=True)
        tee_thread.start()
        try:
            state['return_code'] = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            state.update(timed_out=True, status='solver_hard_deadline')
            os.killpg(process.pid, signal.SIGKILL)
            state['return_code'] = process.wait(timeout=10)
        finally:
            tee_thread.join(timeout=2)
    state.setdefault('status', 'completed' if state['return_code'] == 0 else 'solver_error')
state['solver_finished_at'] = time.time()
Path('/kaggle/working/025/run-state.json').write_text(json.dumps(state, indent=2))
print(json.dumps(dict(event='solver_finished', **state)), flush=True)
''')
    notebook['cells'][8] = code_cell(FINAL)
    source_hashes = {}
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            filename = header.split()[1]
            ast.parse(body, filename=filename)
            source_hashes[filename] = sha(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)
    metadata = dict(base_metadata)
    filename = args.slug + '.ipynb'
    metadata.update(id=args.owner + '/' + args.slug, title=args.slug.replace('-', ' '),
                    code_file=filename, is_private=True)
    notebook_bytes = (json.dumps(notebook, ensure_ascii=False, indent=2) + '\n').encode()
    package_manifest = dict(reference_script_version_id=312063593, reference_sha256=REFERENCE_SHA,
                            base_package_notebook_sha256=sha(base_bytes), notebook_sha256=sha(notebook_bytes),
                            ref=metadata['id'], input_identity=input_identity, runtime_source_sha256=source_hashes,
                            session_seconds=session_seconds, solver_global_seconds=session_seconds-120, solver_hard_seconds=session_seconds-60,
                            status='prepared_not_launched',
                            input_embedding='Deterministic gzip plus base64; decompressed input bytes verified by SHA256',
                            changes=['Fixed subset challenges and sorted ordinary queue; research rerun guard',
                                     'Resolve same named original utility and fixed BF16 V1 under current mount owner aliases',
                                     'Wait at most 60 seconds total for original utility/model markers; count wait in session budget',
                                     f'Research global solver cutoff {session_seconds-120}s, hard process-group cutoff {session_seconds-60}s, total cap {session_seconds}s',
                                     'Working-directory candidate persistence, atomic scored-file writes',
                                     'Per-rank observation events, initialized empty files and live stdout tee',
                                     'Legal raw candidate export before rescoring uses deterministic decode/inversion without RNG calls',
                                     'Exact original decoder B4 and ordered scored records frozen before any gold read'],
                            preserved=['Original loader and decoder', 'Original one-epoch rank256 TTT',
                                       'Original 1200s task soft check, DFS, augmentations and rescoring',
                                       'Original sorted task order and worker marker/import behavior',
                                       'No PYTHONHASHSEED override or fair scheduling'],
                            warning='Short research global time is not a 12-hour replication; incomplete queries remain in the denominator.',
                            observation_overhead='Extra deterministic raw decoding/inversion and JSONL I/O add small CPU/wall overhead, which can affect deadline coverage.')
    out.mkdir(parents=True)
    (out / filename).write_bytes(notebook_bytes)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'package-manifest.json').write_text(json.dumps(package_manifest, indent=2) + '\n')
    print(json.dumps(package_manifest, indent=2))


if __name__ == '__main__':
    main()
