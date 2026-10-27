"""Prepare a bounded same-adapter original-DFS threshold diagnostic.

CPU packaging only. No model, query answers, GPU session or API is used here.
The generated notebook uses original DFS at 0.2 twice and 0.1 once on each
query's first original batch. It neither rescores candidates nor selects B4.
"""
from __future__ import annotations

import argparse
import ast
import contextlib
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
RUN_ROOT = '/kaggle/working/030-mechanism'
TASKS = ['2ba387bc', '7b0280bc', 'd35bdbdc']
ORDER = {
    '2ba387bc': ['A1', 'A2', 'B'],
    '7b0280bc': ['B', 'A1', 'A2'],
    'd35bdbdc': ['A2', 'B', 'A1'],
}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def code(source):
    return dict(cell_type='code', metadata={}, execution_count=None, outputs=[],
                source=source.splitlines(keepends=True))


MECHANISM = r'''
def run_task_mechanism(model, tokenizer, formatter, puzzle_ds_multi, eval_ds,
                       batches, max_new_tokens, end_time, rank, task_id):
    import hashlib, inspect, json, platform
    from pathlib import Path

    root = Path('/kaggle/working/030-mechanism/mechanism')
    path = root / (task_id + '.json')
    order_by_task = {'2ba387bc': ['A1', 'A2', 'B'],
                     '7b0280bc': ['B', 'A1', 'A2'],
                     'd35bdbdc': ['A2', 'B', 'A1']}
    first_batches = {}
    for batch in batches:
        query_keys = {subkey.split('.')[0] for subkey in batch}
        if len(query_keys) != 1:
            raise ValueError('Expected one query per original batch')
        query_key = next(iter(query_keys))
        first_batches.setdefault(query_key, batch)
    if len(first_batches) != len(puzzle_ds_multi.keys):
        raise ValueError('First-batch selection lost a task query')

    def value_sha(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True,
            separators=(',', ':'), allow_nan=False).encode()).hexdigest()

    def adapter_sha():
        digest = hashlib.sha256()
        state = get_peft_model_state_dict(model, adapter_name='default')
        for name, tensor in sorted(state.items()):
            digest.update(name.encode())
            digest.update(str(tuple(tensor.shape)).encode())
            digest.update(str(tensor.dtype).encode())
            raw = tensor.detach().contiguous().view(torch.uint8).cpu().numpy()
            digest.update(raw.tobytes())
            del raw
        return digest.hexdigest()

    report = dict(task_id=task_id, rank=rank, status='running',
        scope='Same-adapter first original batch for every query; no gold, rescoring, B4 or accuracy.',
        ttt_adapter_reused=True, adapter_object_id=id(model),
        hash_seed=os.environ.get('PYTHONHASHSEED'), order=order_by_task[task_id],
        query_keys=list(first_batches), selected_batch_index_per_query=0,
        extra_third_batch=False, records=[], python=platform.python_version(),
        torch_version=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), gold_read=False,
        original_dfs_sha256=hashlib.sha256(inspect.getsource(turbo_dfs).encode()).hexdigest(),
        original_inference_sha256=hashlib.sha256(inspect.getsource(inference_turbo_dfs).encode()).hexdigest())

    def save():
        partial = path.with_suffix('.json.partial')
        partial.write_text(json.dumps(report, indent=2, allow_nan=False) + '\n')
        os.replace(partial, path)

    report['adapter_sha256_before'] = adapter_sha()
    save()
    for query_key, subkeys in first_batches.items():
        prompts = [tokenizer.encode(eval_ds.get(s, formatter)['input']) for s in subkeys]
        prompt_sha = value_sha(prompts)
        for label in order_by_task[task_id]:
            if time.time() + 5 >= end_time:
                report['status'] = 'global_budget_partial'
                save()
                return
            probability = 0.1 if label == 'B' else 0.2
            torch.cuda.synchronize()
            started = time.time()
            record = dict(label=label, query_key=query_key, subkeys=subkeys,
                probability_cutoff=probability, max_score=float(-np.log(probability)),
                prompt_tokens_sha256=prompt_sha, started_at=started,
                global_end_time=end_time, status='running')
            report['records'].append(record)
            save()
            result = inference_turbo_dfs(model, prompts, max_new_tokens,
                                        -np.log(probability), end_time)
            torch.cuda.synchronize()
            dfs_finished = time.time()
            rows = []
            for row, beams in result:
                for nll, output_tokens in beams:
                    tokens = [int(token) for token in output_tokens]
                    grid = formatter.convert_tokens_to_array(tokens)
                    solution = None if grid is None else puzzle_ds_multi.invert_mod(
                        grid, subkeys[row], inv_perm=True).tolist()
                    rows.append(dict(row=int(row), subkey=subkeys[row],
                                     nll=float(nll), tokens=tokens, solution=solution))
            # Keep original traversal order and separate token/grid signatures;
            # do not reinterpret these rows as correctness or a selector score.
            token_rows = [dict(row=r['row'], tokens=r['tokens']) for r in rows]
            grid_rows = [dict(row=r['row'], solution=r['solution'])
                         for r in rows if r['solution'] is not None]
            record.update(status='complete', finished_at=dfs_finished,
                dfs_wall_seconds=dfs_finished-started,
                budget_boundary_reached=dfs_finished >= end_time,
                call540_boundary_reached=dfs_finished-started >= 540,
                returned_beams=len(rows), legal_grid_records=len(grid_rows),
                token_sha256=value_sha(token_rows), grid_sha256=value_sha(grid_rows),
                full_result_sha256=value_sha(rows), rows=rows)
            save()
            _event(rank, 'mechanism_call_end', task_id, label=label,
                   query_key=query_key, probability_cutoff=probability,
                   dfs_wall_seconds=dfs_finished-started,
                   token_sha256=record['token_sha256'],
                   grid_sha256=record['grid_sha256'])
    report['adapter_sha256_after'] = adapter_sha()
    report['adapter_unchanged'] = report['adapter_sha256_before'] == report['adapter_sha256_after']
    report['repeat_controls'] = []
    for query_key in first_batches:
        pair = {record['label']: record for record in report['records']
                if record['query_key'] == query_key and record['status'] == 'complete'}
        report['repeat_controls'].append(dict(query_key=query_key,
            A1_A2_tokens_equal=pair['A1']['token_sha256'] == pair['A2']['token_sha256'],
            A1_A2_grids_equal=pair['A1']['grid_sha256'] == pair['A2']['grid_sha256'],
            A1_A2_full_result_equal=pair['A1']['full_result_sha256'] == pair['A2']['full_result_sha256']))
    report['status'] = 'complete' if report['adapter_unchanged'] else 'adapter_changed'
    save()
'''


FINAL = r'''import hashlib, json, shutil, time
from pathlib import Path
run_root = Path('/kaggle/working/030-mechanism')
state = json.loads((run_root / 'run-state.json').read_text())
for name in ['arc_solver.py', 'arc_loader.py', 'arc_decoder.py', 'starter.py', 'copy_model.py']:
    shutil.copy2(name, run_root / 'sources' / name)
reports = [json.loads(path.read_text()) for path in sorted((run_root/'mechanism').glob('*.json'))]
state.update(finished_at=time.time(), elapsed_s=time.time()-notebook_started,
    completed_task_reports=sum(r['status']=='complete' for r in reports),
    planned_mechanism_calls=15,
    completed_mechanism_calls=sum(sum(c['status']=='complete' for c in r['records']) for r in reports),
    planned_tasks=3, planned_queries=5, research_only=True, gold_read=False,
    accuracy_computed=False, B4_computed=False, competition_submission_created=False)
(run_root/'run-state.json').write_text(json.dumps(state,indent=2)+'\n')
controls=[]
for report in reports:
    for query_key in report['query_keys']:
        pair={record['label']:record for record in report['records']
              if record['query_key']==query_key and record['status']=='complete'}
        row=dict(task_id=report['task_id'],query_key=query_key,
                 adapter_unchanged=report.get('adapter_unchanged'),
                 status='unconfirmed',reason='missing_control_or_incomplete_task')
        if 'A1' in pair and 'A2' in pair and report['status']=='complete':
            a1,a2=pair['A1'],pair['A2']
            bounded=any(record['budget_boundary_reached'] or record['call540_boundary_reached']
                        for record in [a1,a2])
            row.update(A1_tokens_sha256=a1['token_sha256'],A2_tokens_sha256=a2['token_sha256'],
                tokens_equal=a1['token_sha256']==a2['token_sha256'],
                grids_equal=a1['grid_sha256']==a2['grid_sha256'],
                tokens_grids_nll_equal=a1['full_result_sha256']==a2['full_result_sha256'],
                time_boundary_reached=bounded)
            if bounded:
                row['reason']='time_truncated_control'
            elif not report.get('adapter_unchanged'):
                row.update(status='failed',reason='adapter_changed')
            elif row['tokens_equal'] and row['grids_equal'] and row['tokens_grids_nll_equal']:
                row.update(status='passed',reason='exact_repeated_original_result')
            else:
                row.update(status='failed',reason='repeated_original_result_differs')
        controls.append(row)
complete=(state['return_code']==0 and state['completed_task_reports']==3 and
          state['completed_mechanism_calls']==15 and len(controls)==5)
aa_status=('failed' if any(row['status']=='failed' for row in controls) else
           'passed' if complete and all(row['status']=='passed' for row in controls) else 'unconfirmed')
mechanism_report=dict(schema_version=1,status='completed' if complete else 'incomplete',
    aa_control_status=aa_status,planned_tasks=3,planned_queries=5,planned_calls=15,
    completed_tasks=state['completed_task_reports'],completed_calls=state['completed_mechanism_calls'],
    source='Original DFS twice at0.2 plus once at0.1; same per-task TTT adapter',
    python_hash_seed=27030,probability_cutoffs=dict(A1=0.2,A2=0.2,B=0.1),
    input_identity=state['input_identity'],query_controls=controls,
    runtime_source_sha256={path.name:hashlib.sha256(path.read_bytes()).hexdigest()
                           for path in sorted((run_root/'sources').glob('*.py'))},
    original_dfs_function_sha256={report['task_id']:dict(
        turbo_dfs=report['original_dfs_sha256'],
        inference_turbo_dfs=report['original_inference_sha256']) for report in reports},
    checkpoint_files={row['path']:dict(bytes=row['bytes'],sha256=row['sha256'])
                      for row in json.loads((run_root/'model-copy.json').read_text())['files']},
    accuracy_computed=False,B4_computed=False,gold_read=False,
    limits='First original batch only; this is a repeat-control report, not a score or efficacy result')
(run_root/'mechanism-report.json').write_text(json.dumps(mechanism_report,indent=2)+'\n')
identities={}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in {'artifact-identities.json','run-state.json'}:
        raw=path.read_bytes()
        identities[str(path.relative_to(run_root))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(run_root/'artifact-identities.json').write_text(json.dumps(identities,indent=2)+'\n')
print(json.dumps(dict(event='mechanism_outputs_frozen',**state)),flush=True)
if state['return_code'] != 0 or state['completed_task_reports'] != 3:
    raise RuntimeError('030 threshold mechanism incomplete; partial evidence retained, no automatic retry')
'''


def build(args):
    source_path = ROOT / 'experiments/025/prepare_confirmation.py'
    spec = importlib.util.spec_from_file_location('prepare025_for030_mechanism', source_path)
    common = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(common)
    initial_dir = HERE / 'panels-v1/initial40'
    initial_manifest = json.loads((initial_dir/'manifest.json').read_text())
    initial_inputs = json.loads((initial_dir/'challenges.json').read_text())
    selected = {task_id: initial_inputs[task_id] for task_id in TASKS}
    if sum(len(task['test']) for task in selected.values()) != 5:
        raise ValueError('Unexpected mechanism query count')
    if any(set(query) != {'input'} for task in selected.values() for query in task['test']):
        raise ValueError('Mechanism query labels are forbidden')
    challenge_raw = (json.dumps(selected, ensure_ascii=False, indent=2)+'\n').encode()
    mechanism_manifest = dict(cohort='030_known_pressure_first_batch_mechanism',
        n_tasks=3, n_queries=5, task_order=TASKS, gold_read=False,
        source_panel_manifest_sha256=sha((initial_dir/'manifest.json').read_bytes()),
        source_panel_challenges_sha256=sha((initial_dir/'challenges.json').read_bytes()),
        entries=[entry for entry in initial_manifest['entries'] if entry['task_id'] in TASKS],
        exposure='Previously scored025 tasks selected by known behavior; mechanism diagnosis only, no independent confirmation',
        selected_original_batch_index_per_query=0, extra_third_batch=False)

    with tempfile.TemporaryDirectory(prefix='arc2-030-mechanism-build-', dir='/private/tmp') as temporary:
        temp = Path(temporary)
        write_json(temp/'manifest.json', mechanism_manifest)
        (temp/'challenges.json').write_bytes(challenge_raw)
        previous_argv = sys.argv
        try:
            sys.argv = [str(source_path), '--owner', args.owner, '--slug', args.slug,
                '--out', str(temp/'base025'), '--manifest', str(temp/'manifest.json'),
                '--challenges', str(temp/'challenges.json'), '--base-package', str(args.base_package),
                '--session-seconds', '3600']
            with contextlib.redirect_stdout(io.StringIO()):
                common.main()
        finally:
            sys.argv = previous_argv
        metadata = json.loads((temp/'base025/kernel-metadata.json').read_text())
        notebook = json.loads((temp/'base025'/metadata['code_file']).read_bytes())

    # Keep exactly the original numerical DFS/prefill definitions. Replace only
    # the post-TTT production search/rescoring block with diagnostic calls.
    source = ''.join(notebook['cells'][5]['source']).split('\n', 1)[1]
    marker = '        with torch.inference_mode():'
    if source.count(marker) != 1:
        raise ValueError('Unexpected original inference-block boundary')
    original_functions = {}
    parsed = ast.parse(source)
    for function in parsed.body:
        if isinstance(function, ast.FunctionDef) and function.name in {'turbo_dfs','inference_turbo_dfs'}:
            original_functions[function.name] = ast.get_source_segment(source, function)
    if len(original_functions) != 2:
        raise ValueError('Original DFS definitions not found')
    solver = source[:source.index(marker)]
    solver += '''        with torch.inference_mode():
            run_task_mechanism(model, tokenizer, formatter, puzzle_ds_multi, eval_ds,
                batches, max_new_tokens, end_time, rank, key)
        _event(rank, 'task_end', key, elapsed_s=time.time()-start_time)
'''
    solver = common.replace_once(solver, 'def worker(rank, queue, end_time):',
                                 MECHANISM+'\ndef worker(rank, queue, end_time):')
    solver = solver.replace('/kaggle/working/025', RUN_ROOT).replace(
        '/kaggle/temp/arc025-model', '/kaggle/temp/arc030-mechanism-model').replace(
        'ARC025_SOLVER_DEADLINE', 'ARC030MECH_SOLVER_DEADLINE')
    for function in ast.parse(solver).body:
        if isinstance(function, ast.FunctionDef) and function.name in original_functions:
            if ast.get_source_segment(solver, function) != original_functions[function.name]:
                raise ValueError('Original DFS numerical source changed')
    notebook['cells'][5] = code('%%writefile arc_solver.py\n'+solver)
    notebook['cells'][8] = code(FINAL)
    for index in [2, 6, 7]:
        source = ''.join(notebook['cells'][index]['source'])
        source = source.replace('/kaggle/working/025', RUN_ROOT).replace(
            '/kaggle/temp/arc025-model', '/kaggle/temp/arc030-mechanism-model').replace(
            'ARC025_SOLVER_DEADLINE', 'ARC030MECH_SOLVER_DEADLINE')
        if index == 2:
            source = common.replace_once(source, "for directory in ['input', 'inference_outputs', 'incomplete', 'sources']:",
                "for directory in ['input', 'inference_outputs', 'incomplete', 'sources', 'mechanism']:")
            checkpoint = json.loads((ROOT/'experiments/023/ordinary-v1-output/023-model-copy.json').read_bytes())
            expected_checkpoint = {row['path']:dict(bytes=row['bytes'],sha256=row['sha256'])
                                   for row in checkpoint['files']}
            source += '\nexpected_checkpoint_files='+repr(expected_checkpoint)+'\n'
            source += "copy_record=json.loads((run_root/'model-copy.json').read_text())\n"
            source += "observed_checkpoint_files={row['path']:dict(bytes=row['bytes'],sha256=row['sha256']) for row in copy_record['files']}\n"
            source += "if observed_checkpoint_files!=expected_checkpoint_files:\n    raise RuntimeError('Mechanism checkpoint bytes differ from successful023')\n"
        if index == 7:
            source = common.replace_once(source, "OMP_NUM_THREADS='12', PYTHONUNBUFFERED='1',",
                "OMP_NUM_THREADS='12', PYTHONUNBUFFERED='1', PYTHONHASHSEED='27030',")
            source = common.replace_once(source, "log_path = Path(" + repr(RUN_ROOT+'/worker.log') + ")",
                "for rank in range(4):\n    Path('/kaggle/worker%d' % rank).unlink(missing_ok=True)\n"
                "log_path = Path(" + repr(RUN_ROOT+'/worker.log') + ")")
        notebook['cells'][index] = code(source)

    source_hashes = {}
    for cell in notebook['cells']:
        source = ''.join(cell['source'])
        if source.startswith('%%writefile '):
            header, body = source.split('\n', 1)
            ast.parse(body)
            source_hashes[header.split()[1]] = sha(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['execution_count'] = None
            cell['outputs'] = []
    raw = (json.dumps(notebook, ensure_ascii=False, indent=2)+'\n').encode()
    if len(raw) >= 1_000_000:
        raise ValueError('Kaggle source exceeds oneMB limit')
    manifest = dict(ref=metadata['id'], status='prepared_not_launched',
        scope='Same-adapter original-DFS threshold mechanism only; no B4, accuracy, gold or submission',
        notebook_sha256=sha(raw), notebook_bytes=len(raw),
        package_generator_sha256=sha(Path(__file__).read_bytes()),
        runtime_source_sha256=source_hashes,
        original_dfs_function_sha256={name:sha(value.encode()) for name,value in original_functions.items()},
        original_dfs_functions_source_equal=True,
        session_seconds=3600, global_seconds=3480, hard_seconds=3540,
        hardware_requested='4xL4; original four-worker import schedule; only three tasks',
        task_ids=TASKS, planned_tasks=3, planned_queries=5, planned_calls=15,
        probability_cutoffs=dict(A1=0.2,A2=0.2,B=0.1), order=ORDER,
        python_hash_seed=27030, ttt_augmentation_seed=1, eval_augmentation_seed=2,
        machine_report_relative_path='030-mechanism/mechanism-report.json',
        machine_report_status_values=['completed','incomplete'],
        aa_control_status_values=['passed','failed','unconfirmed'],
        expected_checkpoint_files=expected_checkpoint,
        source_panel_identity=mechanism_manifest,
        source_challenges_sha256=sha(challenge_raw),
        limitations=['First original batch only, not fulltask coverage or same-budget efficacy',
            '540second DFS and shared3600second session may leave calls incomplete',
            'Existing024/025/026/027 answers are not loaded; known task behavior only motivates pressure selection',
            'A/A changes near deadlines must not be called numerical drift without checking time truncation'])
    return raw, metadata, manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--base-package', type=Path, default=Path('/private/tmp/arc2-030-023-base'))
    parser.add_argument('--owner', default='zhuating')
    parser.add_argument('--slug', default='arc2-030-threshold-mechanism')
    args = parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    if args.out.resolve() == ROOT or ROOT in args.out.resolve().parents:
        raise ValueError('Generated third-party code must remain outside repository')
    raw, metadata, manifest = build(args)
    args.out.mkdir(parents=True)
    (args.out/metadata['code_file']).write_bytes(raw)
    write_json(args.out/'kernel-metadata.json', metadata)
    write_json(args.out/'package-manifest.json', manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
