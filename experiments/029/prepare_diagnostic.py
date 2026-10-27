"""Prepare one manual, bounded NVARC numerical/cache diagnosis; no API calls."""
import argparse
import ast
import importlib.util
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
spec=importlib.util.spec_from_file_location('p028',ROOT/'experiments/028/prepare_pilot.py')
common=importlib.util.module_from_spec(spec)
spec.loader.exec_module(common)
sha,code,embed=common.sha,common.code,common.embed


def build():
    reference_raw=next(common.BASE.glob('*.ipynb')).read_bytes()
    assert sha(reference_raw)=='3c8551fbf483bd1a4755722a0b7585408c2ed789f5d8ae5b3ed618f8a4200919'
    formal=json.loads(reference_raw)
    old_raw=next(common.SOURCE.glob('*.ipynb')).read_bytes()
    assert sha(old_raw)=='ca8c03dde65b9eca40ed5ea8e48938ef58737994b2a2e122b76edc5c671cf4ef'
    old=json.loads(old_raw)
    solver=''.join(old['cells'][5]['source']).split('\n',1)[1]
    assert solver.count('        with torch.inference_mode():')==1
    solver=solver[:solver.index('        with torch.inference_mode():')]
    solver=solver.replace('/kaggle/working/025','/kaggle/working/029').replace('/kaggle/temp/arc025-model','/kaggle/temp/arc029-model').replace('ARC025_SOLVER_DEADLINE','ARC029_SOLVER_DEADLINE')
    solver+='''        from numerics_diagnostic import run_task_diagnostic
        with torch.inference_mode():
            run_task_diagnostic(model, tokenizer, formatter, puzzle_ds_multi, eval_ds,
                batches, max_new_tokens, max_score, end_time, rank, key,
                original_inference=inference_turbo_dfs)
        _event(rank, "task_end", key, elapsed_s=time.time()-start_time)
'''
    starter=''.join(formal['cells'][6]['source']).split('\n',1)[1]
    starter=starter.replace('/kaggle/working/027','/kaggle/working/029')
    starter=starter.replace('range(4)','range(1)').replace('nprocs=4','nprocs=1')
    panel=(HERE/'input-v1/challenges.json').read_bytes()
    inputs=json.loads(panel)
    assert list(inputs)==['d35bdbdc'] and len(inputs['d35bdbdc']['test'])==3
    assert all(set(q)=={'input'} for q in inputs['d35bdbdc']['test'])
    runtime=(HERE/'numerics_diagnostic.py').read_bytes()
    helper=(ROOT/'experiments/028/dfs_path_reuse.py').read_bytes()
    assert sha(helper)=='b658d01533d599cc4074caf3f60bceb0b98cd61b5d87de3cc0e6ea8b9dce2e34'
    setup='''from pathlib import Path
import base64, gzip, hashlib, json, os
run_root=Path('/kaggle/working/029')
raw_rerun_flag=os.getenv('KAGGLE_IS_COMPETITION_RERUN')
for name in ['input','diagnostic','inference_outputs','incomplete','sources']:
    (run_root/name).mkdir(parents=True,exist_ok=True)
'''
    setup+=embed('/kaggle/working/029/input/challenges.json',panel)
    setup+=embed('dfs_path_reuse.py',helper)+embed('numerics_diagnostic.py',runtime)
    setup+=f'''input_identity=dict(n_tasks=1,n_queries=3,measured_queries=1,
    task_ids=['d35bdbdc'],challenges_sha256={sha(panel)!r},
    source='Previously used028 affected development task selected for numerical diagnosis, not accuracy validation')
(run_root/'input/runtime-manifest.json').write_text(json.dumps(input_identity,indent=2))
'''
    mounted=''.join(formal['cells'][2]['source'])
    mounted=mounted[mounted.index('import base64, hashlib, json, os, subprocess, sys\n'):]
    mounted=mounted.replace('/kaggle/working/027','/kaggle/working/029').replace('/kaggle/temp/arc027-model','/kaggle/temp/arc029-model')
    mounted=mounted.replace('session_seconds=43200, solver_global_seconds=42600, solver_hard_seconds=43140','session_seconds=2400, solver_global_seconds=2220, solver_hard_seconds=2340')
    mounted=mounted.replace('research_only=False','research_only=True')
    mounted=mounted.replace('ttt_augmentation_seeds=dict(base=1, extra=2), sequential_four_gpu_passes=True','ttt_augmentation_seed=1, paired_same_adapter=True, worker_count=1')
    mounted=mounted.replace("event='formal_sequential_view_boot'","event='numerics_diagnostic_boot'")
    launch=common.LAUNCH.replace('ARC028','ARC029').replace('range(3)','range(1)')
    launch=launch.replace('Same-adapter original DFS versus16-token known-path prefill proxy; fixed ABBA',
                          'Same-adapter root, fixed-token and original-snapshot restore diagnosis; no strategy selection')
    final=common.FINAL.replace("'pilot_runtime.py'","'numerics_diagnostic.py'")
    final=final.replace("run_root/'pilot'","run_root/'diagnostic'")
    final=final.replace('completed_task_reports=sum','completed_task_reports=sum').replace('measured_queries_planned=3','measured_queries_planned=1')
    final=final.replace('dfs_pilot_finished','numerics_diagnostic_finished').replace("state['completed_task_reports']!=3","state['completed_task_reports']!=1")
    final=final.replace('DFS mechanism pilot incomplete','Numerical diagnosis incomplete')
    nb=dict(formal)
    nb['cells']=[code('import time\nnotebook_started=time.time()\nsession_deadline=notebook_started+2400\nglobal_end_time=notebook_started+2220\nsolver_hard_deadline=notebook_started+2340\n'),
        formal['cells'][1],code(setup+mounted),formal['cells'][3],formal['cells'][4],
        code('%%writefile arc_solver.py\n'+solver),code('%%writefile starter.py\n'+starter),code(launch),code(final)]
    for cell in nb['cells']:
        source=''.join(cell['source'])
        if source.startswith('%%writefile'): ast.parse(source.split('\n',1)[1])
        elif not source.startswith('!'): ast.parse(source)
        cell['execution_count']=None;cell['outputs']=[]
    ast.parse(runtime);ast.parse(helper)
    identity=dict(source_solver_sha256=sha(solver.encode()),diagnostic_sha256=sha(runtime),
                  path_helper_sha256=sha(helper),input_sha256=sha(panel),
                  source_formal_sha256=sha(reference_raw),source025_notebook_sha256=sha(old_raw),
                  session_seconds=2400,solver_global_seconds=2220,hard_seconds=2340,
                  planned_tasks=1,source_queries=3,measured_queries=1,
                  ttt_steps=128,ttt_augmentation_seed=1,trainer_seed=42,eval_seed=2,
                  hash_seed=27027,restore_depths=[0,1,13],fixed_token_count=16,
                  scope='Root/fixed-token/snapshot diagnosis only. No accuracy, B4 or competition submission.',
                  api_execution='Not launched')
    return nb,identity


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',required=True)
    parser.add_argument('--owner',default='zhuating')
    parser.add_argument('--slug',default='arc2-029-nvarc-cache-diagnostic')
    args=parser.parse_args()
    out=Path(args.out)
    if out.exists():raise ValueError('Output directory already exists')
    nb,identity=build()
    raw=(json.dumps(nb,ensure_ascii=False,indent=1)+'\n').encode()
    assert len(raw)<1_000_000
    metadata=json.loads((common.BASE/'kernel-metadata.json').read_text())
    metadata.update(id=args.owner+'/'+args.slug,title=args.slug.replace('-',' '),code_file=args.slug+'.ipynb')
    identity.update(ref=metadata['id'],notebook_sha256=sha(raw),notebook_bytes=len(raw))
    out.mkdir(parents=True)
    (out/metadata['code_file']).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    (HERE/'diagnostic-v1-package-manifest.json').write_text(json.dumps(identity,indent=2)+'\n')
    print(json.dumps(identity,indent=2))


if __name__=='__main__':main()
