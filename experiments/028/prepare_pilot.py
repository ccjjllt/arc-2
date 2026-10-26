"""Generate one bounded manual DFS pilot. Does not launch GPU or submit."""
import argparse
import ast
import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
BASE = Path('/private/tmp/arc2-027-formal-v1')
SOURCE = Path('/private/tmp/arc2-025-v2-mount-ready')


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def code(source):
    return dict(cell_type='code',metadata={},execution_count=None,outputs=[],
                source=source.splitlines(keepends=True))


def embed(name, raw):
    data = base64.b64encode(gzip.compress(raw,mtime=0)).decode()
    return (f'payload = gzip.decompress(base64.b64decode({data!r}))\n'
            f'assert hashlib.sha256(payload).hexdigest() == {sha(raw)!r}\n'
            f'Path({name!r}).write_bytes(payload)\n')


LAUNCH = '''import json, os, signal, subprocess, sys, threading, time
environment = dict(os.environ, UNSLOTH_DISABLE_STATISTICS='1',
    TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12',
    PYTHONUNBUFFERED='1', PYTHONHASHSEED='27027',
    ARC028_SOLVER_DEADLINE=str(global_end_time))
for rank in range(3):
    Path('/kaggle/worker%d' % rank).unlink(missing_ok=True)
state = dict(started_at=time.time(), notebook_started=notebook_started,
    session_deadline=session_deadline, global_end_time=global_end_time,
    hard_deadline=solver_hard_deadline, input_identity=input_identity,
    status='running', research_only=True, gold_read=False,
    policy='Same-adapter original DFS versus16-token known-path prefill proxy; fixed ABBA',
    model_source=boot['model_source'], model_config_sha256=boot['model_config_sha256'])
def save_state():
    (run_root/'run-state.json').write_text(json.dumps(state,indent=2)+'\\n')
save_state()
command = [sys.executable,'-u','starter.py','--end-time',str(global_end_time)]
with (run_root/'worker.log').open('w') as output:
    process = subprocess.Popen(command,env=environment,stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,text=True,errors='replace',bufsize=1,start_new_session=True)
    def tee():
        for line in process.stdout:
            output.write(line); output.flush(); print(line,end='',flush=True)
    thread=threading.Thread(target=tee,daemon=True);thread.start()
    try:
        state['return_code']=process.wait(timeout=max(0.1,solver_hard_deadline-time.time()))
        state['status']='complete' if state['return_code']==0 else 'solver_error'
    except subprocess.TimeoutExpired:
        os.killpg(process.pid,signal.SIGKILL)
        state.update(status='hard_deadline',return_code=process.wait(timeout=10))
    finally:
        thread.join(timeout=2)
state.update(finished_at=time.time(),elapsed_s=time.time()-notebook_started)
save_state()
'''

FINAL = '''import hashlib, json, shutil, time
sources=run_root/'sources';sources.mkdir(exist_ok=True)
for name in ['arc_solver.py','arc_loader.py','arc_decoder.py','starter.py',
             'pilot_runtime.py','dfs_path_reuse.py','copy_model.py']:
    shutil.copy2(name,sources/name)
reports=[]
for path in sorted((run_root/'pilot').glob('*.json')):
    reports.append(json.loads(path.read_text()))
state.update(finished_at=time.time(),elapsed_s=time.time()-notebook_started,
             completed_task_reports=sum(r['status']=='complete' for r in reports),
             planned_tasks=input_identity['n_tasks'],challenge_queries=input_identity['n_queries'],
             measured_queries_planned=3,formal_submission_created=False)
save_state()
identities={}
for path in sorted(run_root.rglob('*')):
    if path.is_file() and path.name not in ['artifact-identities.json','run-state.json']:
        raw=path.read_bytes()
        identities[str(path.relative_to(run_root))]=dict(bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
(run_root/'artifact-identities.json').write_text(json.dumps(identities,indent=2)+'\\n')
print(json.dumps(dict(event='dfs_pilot_finished',**state)),flush=True)
if state['return_code']!=0 or state['completed_task_reports']!=3:
    raise RuntimeError('DFS mechanism pilot incomplete; inspect recorded error, no automatic retry')
'''


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--owner',default='zhuating')
    parser.add_argument('--slug',default='arc2-028-dfs-path-pilot')
    parser.add_argument('--label',default='v1',help='Local artifact label; not a claimed scriptVersion')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():
        raise ValueError('Use a new output directory')
    nbpath=next(BASE.glob('*.ipynb'))
    original=nbpath.read_bytes()
    assert sha(original)=='3c8551fbf483bd1a4755722a0b7585408c2ed789f5d8ae5b3ed618f8a4200919'
    formal=json.loads(original)
    old_raw=next(SOURCE.glob('*.ipynb')).read_bytes()
    previous=json.loads((ROOT/'experiments/025/confirmation-v2-package-manifest.json').read_text())
    assert sha(old_raw)==previous['notebook_sha256']
    source_nb=json.loads(old_raw)
    panel=(HERE/'panel-v1/challenges.json').read_bytes()
    data=json.loads(panel)
    assert len(data)==3 and sum(len(t['test']) for t in data.values())==5
    assert all(set(q)=={'input'} for t in data.values() for q in t['test'])
    helper=(HERE/'dfs_path_reuse.py').read_bytes()
    runtime=(HERE/'pilot_runtime.py').read_bytes()
    solver=''.join(source_nb['cells'][5]['source']).split('\n',1)[1]
    assert solver.count('        with torch.inference_mode():')==1
    solver=solver[:solver.index('        with torch.inference_mode():')]
    solver=solver.replace('/kaggle/working/025','/kaggle/working/028').replace('/kaggle/temp/arc025-model','/kaggle/temp/arc028-model').replace('ARC025_SOLVER_DEADLINE','ARC028_SOLVER_DEADLINE')
    solver+='''        from pilot_runtime import run_task_pilot
        with torch.inference_mode():
            run_task_pilot(model, tokenizer, formatter, puzzle_ds_multi, eval_ds,
                batches, max_new_tokens, max_score, end_time, rank, key,
                original_inference=inference_turbo_dfs)
        _event(rank, "task_end", key, elapsed_s=time.time()-start_time)
'''
    starter=''.join(formal['cells'][6]['source']).split('\n',1)[1]
    starter=starter.replace('/kaggle/working/027','/kaggle/working/028')
    starter=starter.replace('range(4)','range(3)').replace('nprocs=4','nprocs=3')
    setup='''from pathlib import Path
import base64, gzip, hashlib, json, os
run_root=Path('/kaggle/working/028')
raw_rerun_flag=os.getenv('KAGGLE_IS_COMPETITION_RERUN')
for name in ['input','pilot','inference_outputs','incomplete','sources']:
    (run_root/name).mkdir(parents=True,exist_ok=True)
'''
    setup+=embed('/kaggle/working/028/input/challenges.json',panel)
    setup+=embed('dfs_path_reuse.py',helper)+embed('pilot_runtime.py',runtime)
    setup+=f'''input_identity=dict(n_tasks=3,n_queries=5,measured_queries=3,
    task_ids={sorted(data)!r},challenges_sha256={sha(panel)!r},
    source='Frozen old025 development panel excluding026 andC23; firstquery only for mechanism comparison')
(run_root/'input/runtime-manifest.json').write_text(json.dumps(input_identity,indent=2))
'''
    mounted=''.join(formal['cells'][2]['source'])
    mounted=mounted[mounted.index('import base64, hashlib, json, os, subprocess, sys\n'):]
    mounted=mounted.replace('/kaggle/working/027','/kaggle/working/028').replace('/kaggle/temp/arc027-model','/kaggle/temp/arc028-model')
    mounted=mounted.replace('session_seconds=43200, solver_global_seconds=42600, solver_hard_seconds=43140','session_seconds=5400, solver_global_seconds=5200, solver_hard_seconds=5320')
    mounted=mounted.replace('research_only=False','research_only=True')
    mounted=mounted.replace('ttt_augmentation_seeds=dict(base=1, extra=2), sequential_four_gpu_passes=True','ttt_augmentation_seed=1, paired_same_adapter=True, worker_count=3')
    mounted=mounted.replace("event='formal_sequential_view_boot'","event='dfs_path_pilot_boot'")
    nb=dict(formal)
    nb['cells']=[
        code('import time\nnotebook_started=time.time()\nsession_deadline=notebook_started+5400\nglobal_end_time=notebook_started+5200\nsolver_hard_deadline=notebook_started+5320\n'),
        formal['cells'][1],code(setup+mounted),formal['cells'][3],formal['cells'][4],
        code('%%writefile arc_solver.py\n'+solver),code('%%writefile starter.py\n'+starter),
        code(LAUNCH),code(FINAL)]
    for cell in nb['cells']:
        s=''.join(cell['source'])
        if s.startswith('%%writefile'):
            ast.parse(s.split('\n',1)[1])
        elif not s.startswith('!'):
            ast.parse(s)
        cell['execution_count']=None;cell['outputs']=[]
    ast.parse(helper);ast.parse(runtime)
    raw=(json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode()
    assert len(raw)<1_000_000
    metadata=json.loads((BASE/'kernel-metadata.json').read_text())
    metadata.update(id=args.owner+'/'+args.slug,title=args.slug.replace('-',' '),
                    code_file=args.slug+'.ipynb')
    out.mkdir(parents=True)
    (out/metadata['code_file']).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    manifest=dict(ref=metadata['id'],notebook_sha256=sha(raw),notebook_bytes=len(raw),
        source_formal_sha256=sha(original),source025_notebook_sha256=sha(old_raw),
        source_solver_sha256=sha(solver.encode()),helper_sha256=sha(helper),
        pilot_runtime_sha256=sha(runtime),input_sha256=sha(panel),
        session_seconds=5400,solver_global_seconds=5200,hard_seconds=5320,
        planned_tasks=3,challenge_queries=5,measured_queries=3,
        ttt_steps=128,ttt_seed=1,hash_seed=27027,precompute_tokens=16,
        dfs_probability=0.2,B4='unchanged; no full-task accuracy measured in this mechanism pilot',
        api_execution='Not launched',formal_submission=False)
    (out/'package-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    (HERE/('pilot-'+args.label+'-package-manifest.json')).write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    main()
