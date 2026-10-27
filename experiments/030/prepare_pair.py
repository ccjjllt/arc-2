#!/usr/bin/env python3
"""Prepare one manual, equal-budget original NVARC DFS threshold pair.

This only builds a label-free research notebook in a temporary directory.
It does not publish a notebook, launch a GPU, read solutions, or submit.
"""
from __future__ import annotations

import argparse
import ast
from contextlib import redirect_stdout
from datetime import datetime, timezone
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ROOT = '/kaggle/working/030'
HASH_SEED = 27030


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module('manual030_025', HERE.parent/'025/prepare_confirmation.py')


def code_cell(source):
    return dict(cell_type='code', metadata={}, outputs=[], execution_count=None,
                source=source.splitlines(keepends=True))


def solver_source(source):
    source = source.replace('/kaggle/working/025', RUN_ROOT)
    source = source.replace('/kaggle/temp/arc025-model', '/kaggle/temp/arc030-model')
    source = source.replace('ARC025_SOLVER_DEADLINE', 'ARC030_SOLVER_DEADLINE')
    source = legacy.replace_once(source, 'import json as _observation_json\n',
        '''import json as _observation_json
import hashlib as _adapter_hashlib
_branch = os.environ['ARC030_BRANCH']
_dfs_threshold = float(os.environ['ARC030_DFS_THRESHOLD'])
if (_branch, _dfs_threshold) not in [('base', 0.2), ('challenger', 0.1)]:
    raise RuntimeError('Unexpected frozen030 arm/threshold')
_branch_root = '/kaggle/working/030/' + _branch

def _adapter_fingerprint(model):
    digest = _adapter_hashlib.sha256()
    state = get_peft_model_state_dict(model, adapter_name='default')
    for name in sorted(state):
        tensor = state[name].detach().contiguous().cpu()
        digest.update((name + ':' + str(tensor.dtype) + ':' + str(tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
''')
    source = legacy.replace_once(source,
        'record = dict(timestamp=time.time(), rank=rank, event=event, task_id=task_id,',
        'record = dict(timestamp=time.time(), branch=_branch, dfs_threshold=_dfs_threshold,\n'
        '                  rank=rank, event=event, task_id=task_id,')
    source = legacy.replace_once(source,
        "with open('/kaggle/working/030/events-rank%d.jsonl' % rank, 'a') as output:",
        "with open(_branch_root + '/events-rank%d.jsonl' % rank, 'a') as output:")
    source = legacy.replace_once(source,
        "with open('/kaggle/working/030/raw-candidates-rank%d.jsonl' % rank, 'a') as output:",
        "with open(_branch_root + '/raw-candidates-rank%d.jsonl' % rank, 'a') as output:")
    source = legacy.replace_once(source,
        '    dir_outputs = "/kaggle/working/030/inference_outputs"',
        '    dir_outputs = _branch_root + "/inference_outputs"')
    source = legacy.replace_once(source,
        'partial = os.path.join("/kaggle/working/030/incomplete", subkey)',
        'partial = os.path.join(_branch_root + "/incomplete", subkey)')
    source = legacy.replace_once(source, '    max_score = -np.log(0.2)',
                                 '    max_score = -np.log(_dfs_threshold)')
    source = legacy.replace_once(source,
        '            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)',
        '            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)\n'
        '            _event(rank, "adapter_fingerprint", key, sha256=_adapter_fingerprint(model))')
    return source


LAUNCH = '''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path

run_root = Path('/kaggle/working/030')
config = __CONFIG__
pair_ready_at = time.time()
common_setup_s = pair_ready_at - notebook_started
state = dict(notebook_started=notebook_started, pair_ready_at=pair_ready_at,
             session_deadline=session_deadline, input_identity=input_identity,
             config=config, common_setup_seconds=common_setup_s,
             setup_charged_to_each_arm=True, python_hash_seed=27030,
             status='running', return_code=None, research_only=True,
             evaluation_gold_read=False, noise_control_status='unknown',
             source='Frozen V1 312063593 / restored exact023 / 025 observations')

def save_state():
    temporary = run_root / 'run-state.json.partial'
    temporary.write_text(json.dumps(state, indent=2))
    os.replace(temporary, run_root / 'run-state.json')

def run_arm(branch):
    started = time.time()
    # Shared setup is fully charged to EACH hypothetical deployment.
    # Other-arm work never consumes this arm's task or global clock.
    global_duration = config['arm_slot_seconds'] - common_setup_s - 120
    hard_duration = config['arm_slot_seconds'] - common_setup_s - 60
    deadline = started + global_duration
    hard_deadline = started + hard_duration
    record = dict(branch=branch, dfs_threshold=config['thresholds'][branch],
                  started_at=started, shared_setup_seconds_charged=common_setup_s,
                  global_end_time=deadline, hard_deadline=hard_deadline,
                  effective_solver_global_seconds=global_duration,
                  timed_out=False, status='running')
    if global_duration <= 0 or hard_deadline > session_deadline - 120:
        return dict(record, status='equal_budget_unavailable', return_code=None)
    for rank in range(4):
        Path('/kaggle/worker%d' % rank).unlink(missing_ok=True)
    environment = dict(os.environ)
    environment.update(UNSLOTH_DISABLE_STATISTICS='1',
        TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12',
        PYTHONUNBUFFERED='1', PYTHONHASHSEED='27030', ARC030_BRANCH=branch,
        ARC030_DFS_THRESHOLD=str(config['thresholds'][branch]),
        ARC030_SOLVER_DEADLINE=str(deadline))
    command = [sys.executable, '-u', 'starter.py', '--end-time', str(deadline)]
    record['command'] = command
    with (run_root / branch / 'worker.log').open('w') as log:
        process = subprocess.Popen(command, env=environment, stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, text=True, errors='replace', bufsize=1,
            start_new_session=True)
        record['process_group_id'] = process.pid
        def tee():
            for line in process.stdout:
                log.write(line); log.flush()
                print('['+branch+'] '+line, end='', flush=True)
        thread = threading.Thread(target=tee, daemon=True)
        thread.start()
        try:
            record['return_code'] = process.wait(timeout=max(0.1, hard_deadline-time.time()))
            record['status'] = 'completed' if record['return_code'] == 0 else 'solver_error'
        except subprocess.TimeoutExpired:
            record.update(timed_out=True, status='budget_cutoff')
            os.killpg(process.pid, signal.SIGKILL)
            record['return_code'] = process.wait(timeout=10)
        finally:
            thread.join(timeout=2)
        # Also stop any surviving descendant before another import-marker cycle.
        try:
            os.killpg(process.pid, signal.SIGKILL)
            record['descendant_group_kill_requested'] = True
        except ProcessLookupError:
            record['descendant_group_kill_requested'] = False
    record.update(finished_at=time.time(), elapsed_s=time.time()-started)
    record['deployment_elapsed_seconds'] = common_setup_s + record['elapsed_s']
    return record

save_state()
if common_setup_s > config['setup_allowance_seconds']:
    state.update(status='setup_exceeded_frozen_allowance', return_code=1)
else:
    for branch in config['arm_order']:
        state[branch] = run_arm(branch)
        save_state()
        if state[branch]['return_code'] != 0 and not state[branch]['timed_out']:
            break
    acceptable = all(branch in state and
        (state[branch]['return_code'] == 0 or state[branch]['timed_out'])
        for branch in ['base','challenger'])
    if acceptable:
        state.update(status=('completed_budget_cutoffs' if any(state[b]['timed_out']
            for b in ['base','challenger']) else 'completed'), return_code=0)
    else:
        state.update(status='pair_error_partial_outputs', return_code=1)
state['solver_finished_at'] = time.time()
save_state()
print(json.dumps(dict(event='threshold_pair_finished', **state)), flush=True)
'''


FINAL = '''import time
final_common_started = time.time()
import hashlib, json, os, platform, shutil
from pathlib import Path
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

run_root = Path('/kaggle/working/030')
data = ArcDataset.from_file(str(run_root/'input/challenges.json'))
state = json.loads((run_root/'run-state.json').read_text())
state['arms'] = {}
final_common_setup_s = time.time() - final_common_started
state['final_common_setup_seconds'] = final_common_setup_s
for branch in ['base','challenger']:
    final_arm_started = time.time()
    root = run_root/branch
    store = root/'inference_outputs'
    order = os.listdir(store)
    (root/'candidate-file-order.json').write_text(json.dumps(order,indent=2))
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    original = os.listdir
    try:
        os.listdir = lambda path: list(order) if str(path)==str(store) else original(path)
        decoder.load_decoded_results(str(store))
    finally:
        os.listdir = original
    prediction = data.get_submission(decoder.run_selection_algo())
    raw = (json.dumps(prediction,ensure_ascii=False,sort_keys=True)+'\\n').encode()
    (root/'submission-B4.json').write_bytes(raw)
    count=0
    with (root/'scored-candidates.jsonl').open('w') as output:
        for key,records in decoder.decoded_results.items():
            for record_id,row in records.items():
                filename,index=record_id.rsplit('.out',1)
                value=dict(query_key=key,record_id=record_id,source_file=filename,
                    record_index=int(index),solution=row['solution'].tolist(),
                    score_aug=[float(v) for v in row['score_aug']],
                    beam_score=float(row['beam_score']))
                output.write(json.dumps(value,allow_nan=False)+'\\n');count+=1
    state['arms'][branch]=dict(scored_records=count,candidate_file_count=len(order),
        prediction_sha256=hashlib.sha256(raw).hexdigest(),
        queries_with_real_candidates=len(decoder.decoded_results))
    final_arm_s = time.time()-final_arm_started
    terminal = state.get(branch,{})
    charged = terminal.get('deployment_elapsed_seconds')
    accounted = None if charged is None else charged+final_common_setup_s+final_arm_s
    exceeded = accounted is None or accounted>state['config']['arm_slot_seconds']
    state['arms'][branch].update(finalization_elapsed_seconds=final_arm_s,
        accounted_deployment_elapsed_seconds=accounted,deployment_budget_exceeded=exceeded)
    if exceeded:
        state.update(status='deployment_budget_accounting_failure',return_code=1)
# A research-only placeholder file always represents the original arm.
# No fusion, no ground-truth access, and no automatic challenge selection.
shutil.copyfile(run_root/'base/submission-B4.json','/kaggle/working/submission.json')
for name in ['arc_loader.py','arc_decoder.py','arc_solver.py','starter.py','copy_model.py']:
    shutil.copy2(name,run_root/'sources'/name)
identities={}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in {'artifact-identities.json','run-state.json'}:
        raw=path.read_bytes()
        identities[str(path.relative_to(run_root))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(run_root/'artifact-identities.json').write_text(json.dumps(identities,indent=2))
state.update(finished_at=time.time(),elapsed_s=time.time()-notebook_started,
    planned_tasks=len(data.keys),planned_queries=sum(len(data.queries[k]['test']) for k in data.keys),
    python=platform.python_version(),evaluation_gold_read=False)
(run_root/'run-state.json').write_text(json.dumps(state,indent=2))
print(json.dumps(dict(event='threshold_pair_outputs_frozen',**state)),flush=True)
if state['return_code'] != 0:
    raise RuntimeError('Pair failed; partial outputs saved, no efficacy comparison permitted')
'''


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--base-package', required=True, type=Path)
    parser.add_argument('--panel', choices=['pilot20','initial40','extension40'], default='initial40')
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or out == ROOT or ROOT in out.parents:
        raise ValueError('Use a new temporary package directory outside the repository')
    panel = HERE/'panels-v1'/args.panel
    config = dict(panel=args.panel, session_seconds=7200 if args.panel=='pilot20' else 14400,
        arm_slot_seconds=3300 if args.panel=='pilot20' else 6600,
        setup_allowance_seconds=900, final_reserve_seconds=120,
        arm_order=['challenger','base'] if args.panel=='extension40' else ['base','challenger'],
        thresholds=dict(base=0.2,challenger=0.1), python_hash_seed=HASH_SEED,
        gpu_ids=[0,1,2,3], ttt_augmentation_seed=1, search_augmentation_seed=2,
        task_soft_limit_seconds=1200, dfs_limit_seconds=540,
        hypothesis='Wider original DFS under equal total deployment budget improves final B4 EM@2',
        decision='Initial net>0 and gained>=2 tasks; extension net>0; noise and deadline audit required; no threshold scan')
    temporary_base = out.with_name(out.name+'-025-preparation')
    previous = list(sys.argv)
    try:
        sys.argv = ['prepare_confirmation.py','--owner',args.owner,'--slug',args.slug,
            '--out',str(temporary_base),'--manifest',str(panel/'manifest.json'),
            '--challenges',str(panel/'challenges.json'),'--base-package',str(args.base_package),
            '--session-seconds','7200']
        with redirect_stdout(io.StringIO()):
            legacy.main()
    finally:
        sys.argv = previous
    metadata = json.loads((temporary_base/'kernel-metadata.json').read_bytes())
    notebook = json.loads((temporary_base/metadata['code_file']).read_bytes())
    notebook['cells'][0] = code_cell('import time\nnotebook_started = time.time()\n'
        f"session_deadline = notebook_started + {config['session_seconds']}\n"
        f"setup_deadline = notebook_started + {config['setup_allowance_seconds']}\n"
        'global_end_time = session_deadline - 120\nsolver_hard_deadline = session_deadline - 60\n')
    setup = ''.join(notebook['cells'][2]['source'])
    setup = setup.replace('/kaggle/working/025',RUN_ROOT).replace('/kaggle/temp/arc025-model','/kaggle/temp/arc030-model')
    setup = legacy.replace_once(setup,
        "for directory in ['input', 'inference_outputs', 'incomplete', 'sources']:",
        "for directory in ['input','sources','base/inference_outputs','base/incomplete','challenger/inference_outputs','challenger/incomplete']:")
    setup = legacy.replace_once(setup,
        "for rank in range(4):\n    (run_root / ('events-rank%d.jsonl' % rank)).write_text('')\n    (run_root / ('raw-candidates-rank%d.jsonl' % rank)).write_text('')",
        "for branch in ['base','challenger']:\n    for rank in range(4):\n        (run_root/branch/('events-rank%d.jsonl'%rank)).write_text('')\n        (run_root/branch/('raw-candidates-rank%d.jsonl'%rank)).write_text('')")
    setup = legacy.replace_once(setup, "str(global_end_time)], check=True)","str(setup_deadline)], check=True)")
    setup = legacy.replace_once(setup,
        'boot.update(session_seconds=7200, solver_global_seconds=7080, solver_hard_seconds=7140,',
        "boot.update(session_seconds="+str(config['session_seconds'])+", solver_global_seconds=None, solver_hard_seconds=None,")
    setup = legacy.replace_once(setup, '            research_only=True, gold_read=False)',
        '            research_only=True, gold_read=False, python_hash_seed=27030,\n'
        '            threshold_pair_config='+repr(config)+')')
    checkpoint = json.loads((ROOT/'experiments/023/ordinary-v1-output/023-model-copy.json').read_bytes())
    expected = {r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in checkpoint['files']}
    setup += "\ncopy_record=json.loads((run_root/'model-copy.json').read_text())\n"
    setup += 'expected_checkpoint_files='+repr(expected)+'\n'
    setup += "observed_checkpoint_files={r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in copy_record['files']}\n"
    setup += "if observed_checkpoint_files!=expected_checkpoint_files:\n    raise RuntimeError('Checkpoint bytes differ from successful023')\n"
    notebook['cells'][2] = code_cell(setup)
    notebook['cells'][5] = code_cell(solver_source(''.join(notebook['cells'][5]['source'])))
    notebook['cells'][6] = code_cell(''.join(notebook['cells'][6]['source']).replace('/kaggle/working/025',RUN_ROOT))
    notebook['cells'][7] = code_cell(LAUNCH.replace('__CONFIG__',repr(config)))
    notebook['cells'][8] = code_cell(FINAL)
    hashes={}
    for cell in notebook['cells']:
        source=''.join(cell.get('source',[]))
        if source.startswith('%%writefile '):
            header,body=source.split('\n',1);name=header.split()[1]
            ast.parse(body,filename=name);hashes[name]=sha(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)
    raw=(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n').encode()
    if len(raw)>=1000000:
        raise ValueError('Kaggle source exceeds1MB')
    prior=json.loads((temporary_base/'package-manifest.json').read_bytes())
    manifest=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='prepared_not_launched',
        ref=metadata['id'],notebook_sha256=sha(raw),notebook_bytes=len(raw),
        reference_script_version_id=312063593,reference_sha256=legacy.REFERENCE_SHA,
        base_023_notebook_sha256=prior['base_package_notebook_sha256'],
        input_identity=prior['input_identity'],runtime_source_sha256=hashes,config=config,
        expected_checkpoint_files=expected,evaluation_gold_read=False,
        source_025_preparation_sha256=prior['notebook_sha256'],
        controlled_solver_difference='Only original cumulative path probability threshold0.2 versus0.1',
        common_changes=['Fixed common hash27030','Input-only subset, full denominator',
            'Independent process groups with original sorted4-worker queues',
            'Equal arm deployment slots charging all common setup to each arm',
            'Original candidate export plus TTT adapter fingerprints; observation overhead counted',
            'Research rerun rejected; no ensemble or automatic formal challenge selection'],
        limitations=['Public exposed regression, not a fresh blind confirmation',
            'Independent full TTT, adapter equality must be measured',
            'Subset/deployment-budget pressure is not hidden240-task replication',
            'Same-adapter A/A/B mechanism is a separate run, not performed by this package'])
    out.mkdir(parents=True)
    (out/metadata['code_file']).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(manifest,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
