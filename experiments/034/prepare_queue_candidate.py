#!/usr/bin/env python3
"""Build one manually launched033-derived queue-order candidate without GPU/gold."""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REFERENCE = ROOT/'experiments/033/reference/arc-agi2-lb33-89-minimal-perfpatch.ipynb'
REFERENCE_SHA = 'ee49384a10a4296dc0bf0670ae67325521f798d5337ed57bee50c8a443a378f4'
SMOKE_IDS = ['2ba387bc', 'd35bdbdc']


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code(source):
    return dict(cell_type='code', metadata={}, execution_count=None,
                outputs=[], source=source.splitlines(keepends=True))


def replace_once(source, old, new):
    if source.count(old) != 1:
        raise ValueError('Source anchor count changed: '+old[:70])
    return source.replace(old, new)


CLOCK = '''import os, time
notebook_started = time.time()
raw_rerun_flag = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '')
is_competition_rerun = raw_rerun_flag.strip().lower() not in {'', '0', 'false', 'no'}
if is_competition_rerun:
    os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
else:
    os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
global_end_time = notebook_started + (42600 if is_competition_rerun else 720)
'''


QUEUE = '''    # Visible inputs/demos only. Never read a test-query output.
    def grid_cells(grid):
        return sum(len(row) for row in grid)

    priority_rows = []
    for key in data:
        if not rerun_mode and key not in ['2ba387bc', 'd35bdbdc']:
            continue
        task = data[key]
        n_queries = len(task['test'])
        if n_queries < 1:
            raise ValueError('Task has no test query: '+key)
        visible_cells = sum(grid_cells(pair['input']) + grid_cells(pair['output'])
                            for pair in task['train'])
        visible_cells += sum(grid_cells(query['input']) for query in task['test'])
        priority = Fraction(visible_cells, n_queries)
        priority_rows.append(dict(task_id=key, visible_cell_cost=visible_cells,
            n_test_queries=n_queries, priority_numerator=priority.numerator,
            priority_denominator=priority.denominator))
    priority_rows.sort(key=lambda row: (Fraction(row['priority_numerator'],
        row['priority_denominator']), row['task_id']))
    actual_order = [row['task_id'] for row in priority_rows]
    expected_ids = set(data) if rerun_mode else {'2ba387bc', 'd35bdbdc'}
    if len(actual_order) != len(set(actual_order)) or set(actual_order) != expected_ids:
        raise RuntimeError('Queue membership must preserve every selected task exactly once')
    queue_manifest = dict(mode='hidden_test' if rerun_mode else 'ordinary_order_smoke',
        source_path=test_path, source_sha256=hashlib.sha256(source_raw).hexdigest(),
        priority_rule='visible_cells(train inputs+outputs and test inputs) / n_test_queries; exact Fraction, task_id ties',
        original_reference='033 V2/335137615', actual_order=actual_order,
        n_tasks=len(actual_order), n_queries=sum(row['n_test_queries'] for row in priority_rows),
        entries=priority_rows, query_outputs_read=False)
    with open('queue-manifest.json', 'w') as f:
        json.dump(queue_manifest, f, indent=2)
    print(json.dumps(dict(event='queue_priority_manifest', **queue_manifest)), flush=True)

    queue = mp.Manager().Queue()
    for key in actual_order:
        queue.put(key)
'''


LAUNCH = '''import ast, hashlib, json, os, re, subprocess, sys, time
from pathlib import Path
command = [sys.executable, '-u', 'starter.py', '--end-time', str(global_end_time)]
environment = dict(os.environ)
environment.update(UNSLOTH_DISABLE_STATISTICS='1',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12')
solver_started = time.time()
with Path('worker.log').open('w') as log:
    process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1)
    for line in process.stdout:
        log.write(line); log.flush()
        print(line, end='', flush=True)
    return_code = process.wait()
solver_finished = time.time()
queue_manifest = json.loads(Path('queue-manifest.json').read_text())
raw_log = Path('worker.log').read_text()
finished_tasks = re.findall(r'\\[Rank \\d+\\] finished ([0-9a-f]+) in ', raw_log)
training_steps = {key:int(step) for key,step in re.findall(
    r'training stats for puzzle ([0-9a-f]+): TrainOutput\\(global_step=(\\d+)', raw_log)}
started_batches = {}
for raw_subkeys in re.findall(r'\\[Rank \\d+\\] decoding (\\[[^\\]\\n]*\\])', raw_log):
    for key in set(subkey.split('.')[0] for subkey in ast.literal_eval(raw_subkeys)):
        started_batches[key] = started_batches.get(key, 0) + 1
expected_query_keys = {row['task_id']+'_'+str(i) for row in queue_manifest['entries']
    for i in range(row['n_test_queries'])}
ordinary_smoke_complete = (return_code == 0
    and solver_finished <= global_end_time
    and set(finished_tasks) == {'2ba387bc','d35bdbdc'} and len(finished_tasks) == 2
    and training_steps == {'2ba387bc':128,'d35bdbdc':128}
    and set(started_batches) == expected_query_keys
    and all(value == 4 for value in started_batches.values())
    and 'timeout after' not in raw_log
    and 'Traceback (most recent call last)' not in raw_log)
launch_record = dict(command=command, started_at=solver_started,
    finished_at=solver_finished, elapsed_s=solver_finished-solver_started,
    return_code=return_code, is_competition_rerun=is_competition_rerun,
    ordinary_smoke_complete=ordinary_smoke_complete if not is_competition_rerun else None,
    finished_tasks=finished_tasks, training_steps=training_steps,
    started_batches_by_query=started_batches, no_new_hash_seed=True)
Path('launcher-record.json').write_text(json.dumps(launch_record,indent=2))
if return_code != 0:
    raise RuntimeError('034 solver returned nonzero; no submission permitted: '+str(return_code))
'''


IDENTITY = '''
from pathlib import Path
import hashlib, time
queue_manifest = json.loads(Path('queue-manifest.json').read_text())
prediction_raw = Path('submission.json').read_bytes()
run_identity = dict(original_reference='033 V2/335137615',
    mode='hidden_test' if is_competition_rerun else 'ordinary_order_smoke',
    source_input_sha256=queue_manifest['source_sha256'],
    queue_manifest_sha256=hashlib.sha256(Path('queue-manifest.json').read_bytes()).hexdigest(),
    actual_order=queue_manifest['actual_order'],
    actual_task_count=queue_manifest['n_tasks'], actual_query_count=queue_manifest['n_queries'],
    smoke_task_count=None if is_competition_rerun else 2,
    smoke_query_count=None if is_competition_rerun else 4,
    submission_schema_tasks=len(submission),
    submission_schema_queries=sum(len(rows) for rows in submission.values()),
    prediction_sha256=hashlib.sha256(prediction_raw).hexdigest(),
    global_end_time=global_end_time, notebook_started=notebook_started,
    finished_at=time.time(), elapsed_s=time.time()-notebook_started,
    ordinary_smoke_complete=launch_record['ordinary_smoke_complete'],
    query_gold_read=False, solver_loader_decoder_unchanged=True,
    quality_or_score_gain='not_measured', no_new_hash_seed=True)
Path('run-identity.json').write_text(json.dumps(run_identity,indent=2))
print(json.dumps(dict(event='034_run_identity',**run_identity)),flush=True)
if not is_competition_rerun and not launch_record['ordinary_smoke_complete']:
    raise RuntimeError('034 ordinary order smoke incomplete; partial outputs preserved, no submission permitted')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or out == ROOT or ROOT in out.parents:
        raise ValueError('Choose a new temporary directory outside the repository')
    raw_reference = REFERENCE.read_bytes()
    if sha(raw_reference) != REFERENCE_SHA:
        raise ValueError('033 reference changed')
    notebook = json.loads(raw_reference)
    original_cells = {index: ''.join(notebook['cells'][index]['source']) for index in [3,4,5]}
    notebook['cells'][0]['source'] = [
        '# 034 Visible-cell cost per query queue priority\n',
        '\nOnly the starter task order differs from033 in hidden reruns. '
        'The two-task ordinary smoke is not an accuracy or speed comparison.\n']
    notebook['cells'][1] = code(CLOCK)
    starter = ''.join(notebook['cells'][6]['source'])
    starter = replace_once(starter, 'import json\n',
        'import json\nimport hashlib\nfrom fractions import Fraction\n')
    starter = replace_once(starter,
        '    with open(test_path, "r") as f:\n        data = json.load(f)\n',
        '    with open(test_path, "rb") as f:\n        source_raw = f.read()\n'
        '    data = json.loads(source_raw)\n')
    old_queue = '''    queue = mp.Manager().Queue()

    for key in sorted(data.keys()):
        if not rerun_mode:
            if key not in ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]:
                continue
        queue.put(key)
'''
    notebook['cells'][6] = code(replace_once(starter, old_queue, QUEUE))
    notebook['cells'][7] = code(LAUNCH)
    final = ''.join(notebook['cells'][8]['source'])
    final = replace_once(final,
        '    data = data.load_replies("/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json")\n', '')
    final = replace_once(final, '''if not rerun_mode:
    decoder.benchmark_selection_algos()
    with open("submission.json", "r") as f:
        reload_submission = json.load(f)
    print("*** Reload score:", data.validate_submission(reload_submission))''', '')
    notebook['cells'][8] = code(final+IDENTITY)
    for index, expected in original_cells.items():
        if ''.join(notebook['cells'][index]['source']) != expected:
            raise ValueError('033 solver/loader/decoder must remain byte identical')
    runtime_sources = {}
    for current in notebook['cells']:
        if current['cell_type'] != 'code':
            continue
        current['execution_count'] = None; current['outputs'] = []
        source = ''.join(current['source'])
        if source.startswith('%%writefile '):
            head, body = source.split('\n',1); name = head.split()[1]
            ast.parse(body, filename=name); runtime_sources[name] = sha(body.encode())
        elif source and '!pip ' not in source:
            ast.parse(source)
    metadata = json.loads((REFERENCE.parent/'kernel-metadata.json').read_bytes())
    metadata.pop('id_no', None)
    metadata.update(id='zhuating/arc2-034-queue-query-cost',
        title='ARC2 034 visible cell cost per query queue',
        code_file='arc2-034-queue-query-cost.ipynb',is_private=True)
    raw = (json.dumps(notebook,ensure_ascii=False,indent=2)+'\n').encode()
    manifest = dict(prepared_at_utc=datetime.now(timezone.utc).isoformat(),
        status='prepared_not_launched',ref=metadata['id'],
        notebook_sha256=sha(raw),notebook_bytes=len(raw),
        reference033=dict(ref='mikelou1/arc-agi2-lb33-89-minimal-perfpatch',
            kernel_version=2,script_version_id=335137615,notebook_sha256=REFERENCE_SHA),
        runtime_source_sha256=runtime_sources,
        solver_loader_decoder_byte_unchanged=True,
        factor='Global task queue order by exact visible_cell_cost/n_test_queries, ties task_id',
        preserved=['original033 GPU-NLL','TTT128, augmentation and schedule','DFS0.2',
            'B4','4workers','hidden42600 global and1200soft task budget','original hash behavior'],
        ordinary_smoke=dict(task_ids=SMOKE_IDS,n_tasks=2,n_queries=4,
            global_seconds=720,planned_CLI_session_seconds=840,gold_read=False),
        hidden=dict(all_actual_test_tasks=True,global_seconds=42600),
        efficacy='not_run_not_proven',GPU_started=False,submission_created=False)
    out.mkdir(parents=True)
    (out/metadata['code_file']).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,ensure_ascii=False,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
