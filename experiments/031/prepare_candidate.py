#!/usr/bin/env python3
"""Prepare the manually launched031 TTT-budget candidate, without gold or GPU.

Ordinary execution is a frozen128/64-step deployment-budget pair. Competition
reruns use only the64-step challenger on the complete actual test dictionary.
The64-step stop retains the original128-step cosine schedule and data order.
"""
from __future__ import annotations

import argparse
import ast
import base64
from contextlib import redirect_stdout
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
RUN_ROOT = '/kaggle/working/031'


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


legacy = load_module('manual031_025', HERE.parent / '025/prepare_confirmation.py')
sha = legacy.sha
cell = legacy.code_cell
replace_once = legacy.replace_once


SETUP = '''from pathlib import Path
import base64, gzip, hashlib, json, os, subprocess
run_root = Path('/kaggle/working/031')
for directory in ['input','sources']:
    (run_root/directory).mkdir(parents=True,exist_ok=True)
branches = ['challenger'] if is_competition_rerun else ['base','challenger']
for branch in branches:
    for directory in ['inference_outputs','incomplete']:
        (run_root/branch/directory).mkdir(parents=True,exist_ok=True)
    for rank in range(4):
        (run_root/branch/('events-rank%d.jsonl'%rank)).write_text('')
        (run_root/branch/('raw-candidates-rank%d.jsonl'%rank)).write_text('')
source_path = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/' +
    ('arc-agi_test_challenges.json' if is_competition_rerun else 'arc-agi_evaluation_challenges.json'))
source_raw = source_path.read_bytes()
full_challenges = json.loads(source_raw)
if not isinstance(full_challenges,dict) or not full_challenges:
    raise ValueError('Expected nonempty actual competition task dictionary')
if is_competition_rerun:
    challenges = full_challenges
else:
    frozen_raw = gzip.decompress(base64.b64decode(__PANEL_PAYLOAD__))
    if hashlib.sha256(frozen_raw).hexdigest() != __PANEL_SHA__:
        raise RuntimeError('Frozen ordinary input payload identity mismatch')
    challenges = json.loads(frozen_raw)
    actual_subset = {key:full_challenges[key] for key in sorted(challenges)}
    if actual_subset != challenges:
        raise RuntimeError('Ordinary panel differs from actual Kaggle evaluation inputs')
    manifest_raw = gzip.decompress(base64.b64decode(__MANIFEST_PAYLOAD__))
    if hashlib.sha256(manifest_raw).hexdigest() != __MANIFEST_SHA__:
        raise RuntimeError('Frozen ordinary manifest identity mismatch')
    (run_root/'input/manifest.json').write_bytes(manifest_raw)
if any(set(query)!={'input'} for task in challenges.values() for query in task['test']):
    raise ValueError('Query answers are forbidden in solver inputs')
challenge_raw = ((json.dumps(challenges,ensure_ascii=False,sort_keys=True,separators=(',',':'))+'\\n').encode()
                 if is_competition_rerun else frozen_raw)
(run_root/'input/challenges.json').write_bytes(challenge_raw)
input_identity = dict(mode='hidden_test' if is_competition_rerun else 'frozen_public_pair8',
    source_path=str(source_path),source_sha256=hashlib.sha256(source_raw).hexdigest(),
    challenges_sha256=hashlib.sha256(challenge_raw).hexdigest(),
    n_tasks=len(challenges),n_queries=sum(len(t['test']) for t in challenges.values()),
    task_ids=sorted(challenges),ordinary_frozen_challenges_sha256=__PANEL_SHA__,
    ordinary_frozen_manifest_sha256=__MANIFEST_SHA__)
(run_root/'input/runtime-input-manifest.json').write_text(json.dumps(input_identity,indent=2))
'''


OBSERVER = '''import json as _observation_json
import hashlib as _adapter_hashlib
from transformers import TrainerCallback
_branch = os.environ['ARC031_BRANCH']
_ttt_limit = int(os.environ['ARC031_TTT_LIMIT'])
if (_branch,_ttt_limit) not in [('base',128),('challenger',64)]:
    raise RuntimeError('Unexpected frozen031 branch or TTT limit')
_branch_root = '/kaggle/working/031/' + _branch

class _OriginalScheduleStepObserver(TrainerCallback):
    def __init__(self,rank,key):
        self.rank,self.key=rank,key
    def on_train_begin(self,args,state,control,**kwargs):
        if int(state.max_steps)!=128:
            raise RuntimeError('031 requires the original128-step training schedule')
        _event(self.rank,'ttt_schedule',self.key,max_steps=int(state.max_steps),
            stop_after_steps=_ttt_limit,num_train_epochs=float(args.num_train_epochs),
            warmup_ratio=float(args.warmup_ratio),learning_rate=float(args.learning_rate),
            lr_scheduler_type=str(args.lr_scheduler_type))
        return control
    def on_step_end(self,args,state,control,**kwargs):
        if int(state.global_step) in (64,128):
            optimizer=kwargs.get('optimizer')
            rates=[] if optimizer is None else [float(g['lr']) for g in optimizer.param_groups]
            _event(self.rank,'ttt_checkpoint',self.key,global_step=int(state.global_step),
                max_steps=int(state.max_steps),epoch=float(state.epoch),learning_rates=rates,
                stop_requested=(_ttt_limit==64))
        if _ttt_limit==64 and int(state.global_step)>=64:
            control.should_training_stop=True
        return control

def _adapter_fingerprint(model):
    digest=_adapter_hashlib.sha256()
    adapter=get_peft_model_state_dict(model,adapter_name='default')
    for name in sorted(adapter):
        tensor=adapter[name].detach().contiguous().cpu()
        digest.update((name+':'+str(tensor.dtype)+':'+str(tuple(tensor.shape))).encode())
        digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
'''


def solver_source(source):
    source = source.replace('/kaggle/working/025',RUN_ROOT)
    source = source.replace('/kaggle/temp/arc025-model','/kaggle/temp/arc031-model')
    source = source.replace('ARC025_SOLVER_DEADLINE','ARC031_SOLVER_DEADLINE')
    source = replace_once(source,'import json as _observation_json\n',OBSERVER)
    source = replace_once(source,
        'record = dict(timestamp=time.time(), rank=rank, event=event, task_id=task_id,',
        'record = dict(timestamp=time.time(),branch=_branch,ttt_limit=_ttt_limit,rank=rank,event=event,task_id=task_id,')
    source = replace_once(source,
        "with open('/kaggle/working/031/events-rank%d.jsonl' % rank, 'a') as output:",
        "with open(_branch_root+'/events-rank%d.jsonl'%rank,'a') as output:")
    source = replace_once(source,
        "with open('/kaggle/working/031/raw-candidates-rank%d.jsonl' % rank, 'a') as output:",
        "with open(_branch_root+'/raw-candidates-rank%d.jsonl'%rank,'a') as output:")
    source = replace_once(source,'    dir_outputs = "/kaggle/working/031/inference_outputs"',
        '    dir_outputs = _branch_root + "/inference_outputs"')
    source = replace_once(source,'partial = os.path.join("/kaggle/working/031/incomplete", subkey)',
        'partial = os.path.join(_branch_root + "/incomplete", subkey)')
    source = replace_once(source,'                args=UnslothTrainingArguments(**train_args),',
        '                args=UnslothTrainingArguments(**train_args),\n'
        '                callbacks=[_OriginalScheduleStepObserver(rank,key)],')
    source = replace_once(source,'            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)',
        '            model = trainer.accelerator.unwrap_model(model, keep_fp32_wrapper=False)\n'
        '            _event(rank,"adapter_fingerprint",key,sha256=_adapter_fingerprint(model))')
    return source


LAUNCH = '''import json, os, signal, subprocess, sys, threading, time
from pathlib import Path
config=__CONFIG__
pair_ready_at=time.time()
common_setup_s=pair_ready_at-notebook_started
state=dict(notebook_started=notebook_started,pair_ready_at=pair_ready_at,
    session_deadline=session_deadline,input_identity=input_identity,config=config,
    common_setup_seconds=common_setup_s,setup_charged_to_each_arm=True,
    python_hash_seed=27031,is_competition_rerun=is_competition_rerun,
    status='running',return_code=None,evaluation_gold_read=False,
    source='Frozen original V1 312063593 / exact023 / 025 observations')
def save_state():
    temporary=run_root/'run-state.json.partial'
    temporary.write_text(json.dumps(state,indent=2))
    os.replace(temporary,run_root/'run-state.json')
def run_arm(branch):
    started=time.time()
    if is_competition_rerun:
        deadline=notebook_started+42600
        hard_deadline=notebook_started+43140
    else:
        deadline=started+config['arm_slot_seconds']-common_setup_s-120
        hard_deadline=started+config['arm_slot_seconds']-common_setup_s-60
    record=dict(branch=branch,ttt_stop_steps=config['ttt_steps'][branch],started_at=started,
        shared_setup_seconds_charged=common_setup_s,global_end_time=deadline,
        hard_deadline=hard_deadline,effective_solver_global_seconds=deadline-started,
        timed_out=False,status='running')
    if deadline<=started or hard_deadline>session_deadline-60:
        return dict(record,status='budget_unavailable',return_code=None)
    for rank in range(4):
        Path('/kaggle/worker%d'%rank).unlink(missing_ok=True)
    environment=dict(os.environ)
    environment.update(UNSLOTH_DISABLE_STATISTICS='1',TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas',
        OMP_NUM_THREADS='12',PYTHONUNBUFFERED='1',PYTHONHASHSEED='27031',
        ARC031_BRANCH=branch,ARC031_TTT_LIMIT=str(config['ttt_steps'][branch]),
        ARC031_SOLVER_DEADLINE=str(deadline))
    command=[sys.executable,'-u','starter.py','--end-time',str(deadline)]
    record['command']=command
    with (run_root/branch/'worker.log').open('w') as log:
        process=subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,start_new_session=True)
        record['process_group_id']=process.pid
        def tee():
            for line in process.stdout:
                log.write(line);log.flush()
                print('['+branch+'] '+line,end='',flush=True)
        thread=threading.Thread(target=tee,daemon=True);thread.start()
        try:
            record['return_code']=process.wait(timeout=max(0.1,hard_deadline-time.time()))
            record['status']='completed' if record['return_code']==0 else 'solver_error'
        except subprocess.TimeoutExpired:
            record.update(timed_out=True,status='budget_cutoff')
            os.killpg(process.pid,signal.SIGKILL)
            record['return_code']=process.wait(timeout=10)
        finally:
            thread.join(timeout=2)
        try:
            os.killpg(process.pid,signal.SIGKILL)
            record['descendant_group_kill_requested']=True
        except ProcessLookupError:
            record['descendant_group_kill_requested']=False
    record.update(finished_at=time.time(),elapsed_s=time.time()-started)
    record['deployment_elapsed_seconds']=common_setup_s+record['elapsed_s']
    return record
save_state()
if not is_competition_rerun and common_setup_s>config['setup_allowance_seconds']:
    state.update(status='setup_exceeded_frozen_allowance',return_code=1)
else:
    for branch in branches:
        state[branch]=run_arm(branch);save_state()
        if state[branch]['return_code']!=0 and not state[branch]['timed_out']:
            break
    acceptable=all(b in state and (state[b]['return_code']==0 or state[b]['timed_out']) for b in branches)
    state.update(status=('completed_budget_cutoffs' if any(state[b]['timed_out'] for b in branches)
        else 'completed') if acceptable else 'solver_error_partial_outputs',return_code=0 if acceptable else 1)
state['solver_finished_at']=time.time();save_state()
print(json.dumps(dict(event='ttt_budget_stages_finished',**state)),flush=True)
'''


FINAL = '''import time
final_common_started=time.time()
import hashlib, json, os, platform, shutil
from pathlib import Path
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder
data=ArcDataset.from_file(str(run_root/'input/challenges.json'))
state=json.loads((run_root/'run-state.json').read_text())
state['arms']={}
final_common_setup_s=time.time()-final_common_started
state['final_common_setup_seconds']=final_common_setup_s
for branch in branches:
    final_arm_started=time.time()
    root=run_root/branch;store=root/'inference_outputs'
    order=os.listdir(store)
    (root/'candidate-file-order.json').write_text(json.dumps(order,indent=2))
    decoder=ArcDecoder(data.split_multi_replies(),n_guesses=2)
    original=os.listdir
    try:
        os.listdir=lambda path:list(order) if str(path)==str(store) else original(path)
        decoder.load_decoded_results(str(store))
    finally:
        os.listdir=original
    prediction=data.get_submission(decoder.run_selection_algo())
    raw=(json.dumps(prediction,ensure_ascii=False,sort_keys=True)+'\\n').encode()
    (root/'submission-B4.json').write_bytes(raw)
    count=0
    with (root/'scored-candidates.jsonl').open('w') as output:
        for key,records in decoder.decoded_results.items():
            for record_id,row in records.items():
                filename,index=record_id.rsplit('.out',1)
                value=dict(query_key=key,record_id=record_id,source_file=filename,
                    record_index=int(index),solution=row['solution'].tolist(),
                    score_aug=[float(v) for v in row['score_aug']],beam_score=float(row['beam_score']))
                output.write(json.dumps(value,allow_nan=False)+'\\n');count+=1
    final_arm_s=time.time()-final_arm_started
    charged=state.get(branch,{}).get('deployment_elapsed_seconds')
    accounted=None if charged is None else charged+final_common_setup_s+final_arm_s
    limit=43200 if is_competition_rerun else config['arm_slot_seconds']
    exceeded=accounted is None or accounted>limit
    state['arms'][branch]=dict(scored_records=count,candidate_file_count=len(order),
        prediction_sha256=hashlib.sha256(raw).hexdigest(),
        queries_with_real_candidates=len(decoder.decoded_results),
        finalization_elapsed_seconds=final_arm_s,accounted_deployment_elapsed_seconds=accounted,
        deployment_budget_exceeded=exceeded)
    if exceeded:
        state.update(status='deployment_budget_accounting_failure',return_code=1)
# One frozen challenger policy in ordinary and hidden runs; never fuse arms.
shutil.copyfile(run_root/'challenger/submission-B4.json','/kaggle/working/submission.json')
state['active_prediction_policy']='original_B4_after_TTT64_original128step_schedule'
state['prediction_sha256']=hashlib.sha256(Path('/kaggle/working/submission.json').read_bytes()).hexdigest()
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
print(json.dumps(dict(event='ttt_budget_outputs_frozen',**state)),flush=True)
if state['return_code']!=0:
    raise RuntimeError('031 execution failed; partial outputs preserved, no efficacy claim permitted')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ['owner','slug']:
        parser.add_argument('--'+name,required=True)
    for name in ['out','base-package','manifest','challenges']:
        parser.add_argument('--'+name,required=True,type=Path)
    args=parser.parse_args()
    out=args.out.resolve()
    if out.exists() or out==ROOT or ROOT in out.parents:
        raise ValueError('Use a new temporary package directory outside repository')
    temporary=out.with_name(out.name+'-025-preparation')
    previous=list(sys.argv)
    try:
        sys.argv=['prepare_confirmation.py','--owner',args.owner,'--slug',args.slug,
            '--out',str(temporary),'--manifest',str(args.manifest),'--challenges',str(args.challenges),
            '--base-package',str(args.base_package),'--session-seconds','3600']
        with redirect_stdout(io.StringIO()):
            legacy.main()
    finally:
        sys.argv=previous
    metadata=json.loads((temporary/'kernel-metadata.json').read_bytes())
    prior=json.loads((temporary/'package-manifest.json').read_bytes())
    notebook=json.loads((temporary/metadata['code_file']).read_bytes())
    panel_raw=args.challenges.read_bytes();manifest_raw=args.manifest.read_bytes()
    config=dict(ordinary_session_seconds=3600,hidden_session_seconds=43200,
        hidden_global_seconds=42600,arm_slot_seconds=1620,setup_allowance_seconds=600,
        arm_order=['base','challenger'],ttt_steps=dict(base=128,challenger=64),
        original_scheduler_steps=128,python_hash_seed=27031,dfs_threshold=0.2,
        task_soft_limit_seconds=1200,dfs_limit_seconds=540,gpu_ids=[0,1,2,3],
        ttt_augmentation_seed=1,search_augmentation_seed=2,selection='original_B4',
        interpretation='Exploratory exposed public equal-budget pair, not independent hidden-score evidence')
    notebook['cells'][0]=cell('''import os,time
notebook_started=time.time()
raw_rerun_flag=os.getenv('KAGGLE_IS_COMPETITION_RERUN','')
is_competition_rerun=raw_rerun_flag.strip().lower() not in {'','0','false','no'}
if is_competition_rerun:
    os.environ['KAGGLE_IS_COMPETITION_RERUN']='1'
else:
    os.environ.pop('KAGGLE_IS_COMPETITION_RERUN',None)
session_deadline=notebook_started+(43200 if is_competition_rerun else 3600)
setup_deadline=notebook_started+(42600 if is_competition_rerun else 600)
global_end_time=notebook_started+42600 if is_competition_rerun else session_deadline-120
solver_hard_deadline=session_deadline-60
''')
    setup=SETUP.replace('__PANEL_PAYLOAD__',repr(base64.b64encode(gzip.compress(panel_raw,mtime=0)).decode()))
    setup=setup.replace('__PANEL_SHA__',repr(sha(panel_raw)))
    setup=setup.replace('__MANIFEST_PAYLOAD__',repr(base64.b64encode(gzip.compress(manifest_raw,mtime=0)).decode()))
    setup=setup.replace('__MANIFEST_SHA__',repr(sha(manifest_raw)))
    runtime=''.join(notebook['cells'][2]['source'])
    marker='import base64, hashlib, json, os, subprocess, sys\n'
    if runtime.count(marker)!=1:
        raise ValueError('Original runtime start anchor changed')
    runtime=marker+runtime.split(marker,1)[1]
    runtime=runtime.replace('/kaggle/working/025',RUN_ROOT).replace('/kaggle/temp/arc025-model','/kaggle/temp/arc031-model')
    runtime=replace_once(runtime,'str(global_end_time)], check=True)','str(setup_deadline)], check=True)')
    runtime=replace_once(runtime,'boot.update(session_seconds=3600, solver_global_seconds=3480, solver_hard_seconds=3540,',
        "boot.update(session_seconds=43200 if is_competition_rerun else 3600,solver_global_seconds=42600 if is_competition_rerun else None,solver_hard_seconds=43140 if is_competition_rerun else None,")
    runtime=replace_once(runtime,'            research_only=True, gold_read=False)',
        '            research_only=False,gold_read=False,python_hash_seed=27031,ttt_budget_config='+repr(config)+')')
    copy=json.loads((ROOT/'experiments/023/ordinary-v1-output/023-model-copy.json').read_bytes())
    expected={r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in copy['files']}
    runtime+="\ncopy_record=json.loads((run_root/'model-copy.json').read_text())\n"
    runtime+='expected_checkpoint_files='+repr(expected)+'\n'
    runtime+="observed_checkpoint_files={r['path']:dict(bytes=r['bytes'],sha256=r['sha256']) for r in copy_record['files']}\n"
    runtime+="if observed_checkpoint_files!=expected_checkpoint_files:\n    raise RuntimeError('Fixed checkpoint bytes differ from successful023')\n"
    notebook['cells'][2]=cell(setup+runtime)
    notebook['cells'][5]=cell(solver_source(''.join(notebook['cells'][5]['source'])))
    notebook['cells'][6]=cell(''.join(notebook['cells'][6]['source']).replace('/kaggle/working/025',RUN_ROOT))
    notebook['cells'][7]=cell(LAUNCH.replace('__CONFIG__',repr(config)))
    notebook['cells'][8]=cell(FINAL)
    hashes={}
    for current in notebook['cells']:
        source=''.join(current.get('source',[]))
        if source.startswith('%%writefile '):
            header,body=source.split('\n',1);name=header.split()[1]
            ast.parse(body,filename=name);hashes[name]=sha(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)
    raw=(json.dumps(notebook,ensure_ascii=False,indent=2)+'\n').encode()
    if len(raw)>=1000000:
        raise ValueError('Notebook exceeds Kaggle source limit')
    result=dict(created_at_utc=datetime.now(timezone.utc).isoformat(),status='prepared_not_launched',
        ref=metadata['id'],notebook_sha256=sha(raw),notebook_bytes=len(raw),
        reference_script_version_id=312063593,reference_sha256=legacy.REFERENCE_SHA,
        base_023_notebook_sha256=prior['base_package_notebook_sha256'],
        ordinary_input_identity=prior['input_identity'],runtime_source_sha256=hashes,
        config=config,expected_checkpoint_files=expected,evaluation_gold_read=False,
        source_025_preparation_sha256=prior['notebook_sha256'],
        controlled_solver_difference='Stop original128-step TTT trajectory after step64 versus128; original full training data and128-step LR schedule retained',
        limitations=['Independent TTT may be numerically different; not a same-adapter causal comparison',
            'Ordinary short exposed panel is not a blind confirmation or full hidden-budget replication',
            'TTT64 may broaden costly DFS or reduce quality, so saved training steps do not guarantee total speed or score',
            'No GPU execution, efficacy measurement or competition submission is performed by this generator'])
    out.mkdir(parents=True)
    (out/metadata['code_file']).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
