#!/usr/bin/env python3
"""Prepare a conditional formal notebook only after 027 confirmation succeeds.

No API calls, GPU work or competition submission occur here. This generator
never reads C23 answers. A caller must verify confirmation and prefix results
before invoking it and before uploading its output.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location('confirmation_packager', HERE/'prepare_confirmation.py')
_common = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_common)
ROOT, sha, code_cell, embed, replace_once = (_common.ROOT, _common.sha, _common.code_cell,
                                           _common.embed, _common.replace_once)
RESEARCH = Path('/private/tmp/arc2-027-v1-confirmation')
FROZEN_RESEARCH_SHA = 'bb1e8320a087a0abb57948077ff15f794c1206f53abdb2b235c2969b7cca87b0'


SETUP = '''from pathlib import Path
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
is_competition_rerun = raw_rerun_flag.strip().lower() not in {'', '0', 'false', 'no'}
if is_competition_rerun:
    os.environ['KAGGLE_IS_COMPETITION_RERUN'] = '1'
else:
    os.environ.pop('KAGGLE_IS_COMPETITION_RERUN', None)
source_path = Path('/kaggle/input/competitions/arc-prize-2026-arc-agi-2/' +
                   ('arc-agi_test_challenges.json' if is_competition_rerun else
                    'arc-agi_evaluation_challenges.json'))
source_raw = source_path.read_bytes()
full_challenges = json.loads(source_raw)
if not isinstance(full_challenges, dict) or not full_challenges:
    raise ValueError('Expected actual competition challenge dictionary')
if is_competition_rerun:
    challenges = full_challenges
else:
    original_ordinary_ids = ['0934a4d8', '36a08778', '981571dc', 'aa4ec2a5']
    challenges = {key: full_challenges[key] for key in original_ordinary_ids}
if any(set(query) != {'input'} for task in challenges.values() for query in task['test']):
    raise ValueError('Solver input must not include query answers')
challenge_raw = (json.dumps(challenges, ensure_ascii=False, sort_keys=True,
                           separators=(',', ':')) + '\\n').encode()
if not is_competition_rerun and hashlib.sha256(challenge_raw).hexdigest() != EXPECTED_ORDINARY_CANONICAL_SHA:
    raise RuntimeError('Original four-task ordinary inputs differ from the frozen 023 reference')
(run_root / 'input/challenges.json').write_bytes(challenge_raw)
input_identity = dict(mode='hidden_test' if is_competition_rerun else 'original_ordinary_four',
                      source_path=str(source_path), source_sha256=hashlib.sha256(source_raw).hexdigest(),
                      challenges_sha256=hashlib.sha256(challenge_raw).hexdigest(),
                      n_tasks=len(challenges), n_queries=sum(len(task['test']) for task in challenges.values()),
                      task_ids=sorted(challenges), ordinary_reference_sha256=ORDINARY_REFERENCE_SHA)
(run_root / 'input/runtime-input-manifest.json').write_text(json.dumps(input_identity, indent=2))
'''


FREEZE_BRANCH = '''
from arc_loader import ArcDataset
from arc_decoder import ArcDecoder

data = ArcDataset.from_file(str(run_root / 'input/challenges.json'))
branch_submissions, branch_counts = {}, {}
def freeze_branch(branch):
    root = run_root / branch
    store = root / 'inference_outputs'
    order = os.listdir(store)
    (root / 'candidate-file-order.json').write_text(json.dumps(order, indent=2))
    decoder = ArcDecoder(data.split_multi_replies(), n_guesses=2)
    original_listdir = os.listdir
    try:
        os.listdir = lambda path: list(order) if str(path) == str(store) else original_listdir(path)
        decoder.load_decoded_results(str(store))
    finally:
        os.listdir = original_listdir
    prediction = data.get_submission(decoder.run_selection_algo())
    raw = (json.dumps(prediction, ensure_ascii=False, sort_keys=True) + '\\n').encode()
    (root / 'submission-B4.json').write_bytes(raw)
    counts = {key: len(records) for key, records in decoder.decoded_results.items()}
    total = 0
    with (root / 'scored-candidates.jsonl').open('w') as output:
        for key, records in decoder.decoded_results.items():
            for record_id, record in records.items():
                source_file, index = record_id.rsplit('.out', 1)
                row = dict(query_key=key, record_id=record_id, source_file=source_file,
                           record_index=int(index), solution=record['solution'].tolist(),
                           score_aug=[float(v) for v in record['score_aug']],
                           beam_score=float(record['beam_score']))
                output.write(json.dumps(row, allow_nan=False) + '\\n')
                total += 1
    branch_submissions[branch], branch_counts[branch] = prediction, counts
    state.setdefault('arms', {})[branch] = dict(scored_records=total,
        candidate_file_count=len(order), prediction_sha256=hashlib.sha256(raw).hexdigest(),
        queries_with_real_candidates=sum(v > 0 for v in counts.values()))
    return raw

def write_top_level(raw):
    target = Path('/kaggle/working/submission.json')
    temporary = target.with_suffix('.json.partial')
    temporary.write_bytes(raw)
    os.replace(temporary, target)
'''


STAGES = '''
# Persist base before starting extra. A later extra exception cannot erase it.
state['base'] = run_branch('base', 1)
save_state()
if state['base']['return_code'] == 0:
    base_raw = freeze_branch('base')
    (run_root / 'submission-base-backup.json').write_bytes(base_raw)
    write_top_level(base_raw)
    state.update(base_prediction_frozen_at=time.time(),
                 base_prediction_backup_sha256=hashlib.sha256(base_raw).hexdigest(),
                 active_prediction_policy='base_B4_backup')
    save_state()
    try:
        state['extra'] = run_branch('extra', 2)
    except Exception as error:
        state['extra'] = dict(branch='extra', ttt_augmentation_seed=2,
                              return_code=-1, timed_out=False, status='launcher_exception',
                              error=type(error).__name__ + ': ' + str(error)[:300])
    extra = state['extra']
    if extra.get('return_code') not in (0, None) and not extra.get('timed_out'):
        state.update(status='base_fallback_after_extra_error', return_code=0,
                     active_prediction_policy='base_B4_backup')
    elif extra.get('timed_out'):
        state.update(status='completed_extra_deadline_partial', return_code=0)
    elif extra.get('return_code') is None:
        state.update(status='completed_extra_skipped', return_code=0,
                     active_prediction_policy='base_B4_backup')
    else:
        state.update(status='completed', return_code=0)
else:
    state['extra'] = dict(branch='extra', ttt_augmentation_seed=2,
                          return_code=None, timed_out=False, status='skipped_base_failed')
    state.update(status='base_error_partial_outputs', return_code=1)
state['solver_finished_at'] = time.time()
save_state()
print(json.dumps(dict(event='formal_sequential_stages_finished', **state)), flush=True)
'''


FINAL = '''import hashlib, json, os, platform, shutil, time
from pathlib import Path
from freeze_ensemble import select_ensemble_query

if state['base']['return_code'] == 0:
    use_extra = state['status'] not in {'base_fallback_after_extra_error', 'completed_extra_skipped'}
    decisions = []
    if use_extra:
        try:
            freeze_branch('extra')
            ensemble = {}
            for task_id in sorted(branch_submissions['base']):
                ensemble[task_id] = []
                for index, base_slots in enumerate(branch_submissions['base'][task_id]):
                    key = task_id + '_' + str(index)
                    slots, audit = select_ensemble_query(base_slots,
                        branch_submissions['extra'][task_id][index],
                        branch_counts['base'].get(key, 0), branch_counts['extra'].get(key, 0))
                    ensemble[task_id].append(slots)
                    decisions.append(dict(task_id=task_id, query_index=index, query_key=key, **audit))
            raw = (json.dumps(ensemble, ensure_ascii=False, sort_keys=True) + '\\n').encode()
            (run_root / 'submission-ensemble.json').write_bytes(raw)
            state['active_prediction_policy'] = 'frozen_base_first_extra_first_ensemble'
        except Exception as error:
            state.update(status='base_fallback_after_extra_error',
                         extra_finalize_error=type(error).__name__ + ': ' + str(error)[:300],
                         active_prediction_policy='base_B4_backup')
            use_extra = False
            decisions = []
    if not use_extra:
        raw = (run_root / 'submission-base-backup.json').read_bytes()
    write_top_level(raw)
    state['prediction_sha256'] = hashlib.sha256(raw).hexdigest()
    with (run_root / 'ensemble-decisions.jsonl').open('w') as output:
        for row in decisions:
            output.write(json.dumps(row, allow_nan=False) + '\\n')
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
             planned_tasks=input_identity['n_tasks'], planned_queries=input_identity['n_queries'],
             python=platform.python_version(), evaluation_gold_read=False)
save_state()
print(json.dumps(dict(event='formal_sequential_outputs_frozen', **state)), flush=True)
if state['return_code'] != 0:
    raise RuntimeError('Base solver failed; diagnostic artifacts saved')
'''


def build(research_notebook, helper_raw, ordinary_raw, ordinary_ref_raw):
    """Return a new notebook in memory; does not execute or write it."""
    notebook = json.loads(json.dumps(research_notebook))
    ordinary = json.loads(ordinary_raw)
    ordinary_canonical = (json.dumps(ordinary, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':'))+'\n').encode()
    notebook['cells'][0] = code_cell('import time\nnotebook_started = time.time()\n'
                                    'session_deadline = notebook_started + 43200\n'
                                    'global_end_time = notebook_started + 42600\n'
                                    'solver_hard_deadline = session_deadline - 60\n')
    setup = ('EXPECTED_ORDINARY_CANONICAL_SHA = '+repr(sha(ordinary_canonical))+'\n'
             'ORDINARY_REFERENCE_SHA = '+repr(sha(ordinary_ref_raw))+'\n'+SETUP)
    setup += embed('freeze_ensemble.py', helper_raw)
    runtime = ''.join(notebook['cells'][2]['source'])
    runtime = runtime[runtime.index('import base64, hashlib, json, os, subprocess, sys\n'):]
    runtime = replace_once(runtime, 'boot.update(session_seconds=10800, solver_global_seconds=10680, solver_hard_seconds=10740,',
                           'boot.update(session_seconds=43200, solver_global_seconds=42600, solver_hard_seconds=43140,')
    runtime = replace_once(runtime, 'research_only=True, evaluation_gold_read=False, python_hash_seed=27027,',
                           'research_only=False, evaluation_gold_read=False, python_hash_seed=27027,')
    runtime = runtime.replace("event='sequential_view_boot'", "event='formal_sequential_view_boot'")
    notebook['cells'][2] = code_cell(setup+runtime)
    launch = ''.join(notebook['cells'][7]['source'])
    launch = launch[:launch.index('# No half-budget cut is imposed:')]
    launch = replace_once(launch, 'research_only=True, evaluation_gold_read=False)',
                           'research_only=False, evaluation_gold_read=False)')
    launch = replace_once(launch, 'import json, os, signal, subprocess, sys, threading, time',
                           'import hashlib, json, os, signal, subprocess, sys, threading, time')
    notebook['cells'][7] = code_cell(launch+FREEZE_BRANCH+STAGES)
    notebook['cells'][8] = code_cell(FINAL)
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('owner', 'slug'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--research-package', type=Path, default=RESEARCH)
    parser.add_argument('--helper', type=Path, default=HERE/'freeze_ensemble.py')
    args = parser.parse_args()
    out = args.out.resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated third-party sources must stay outside the repository')
    if out.exists():
        raise FileExistsError(out)
    prior = json.loads((args.research_package/'package-manifest.json').read_text())
    metadata = json.loads((args.research_package/'kernel-metadata.json').read_text())
    research_raw = (args.research_package/metadata['code_file']).read_bytes()
    if sha(research_raw) != prior['notebook_sha256'] or sha(research_raw) != FROZEN_RESEARCH_SHA:
        raise ValueError('Frozen research package identity changed')
    helper_raw = args.helper.read_bytes()
    if sha(helper_raw) != prior['selection_helper_sha256']:
        raise ValueError('Frozen no-parameter selector changed')
    ordinary_raw = (ROOT/'experiments/023/ordinary-v1-input/challenges.json').read_bytes()
    ordinary_ref_raw = (ROOT/'experiments/023/ordinary-v1-input/manifest.json').read_bytes()
    ordinary_ref = json.loads(ordinary_ref_raw)
    if sha(ordinary_raw) != ordinary_ref['input_artifact']['sha256']:
        raise ValueError('Original ordinary input identity changed')
    notebook = build(json.loads(research_raw), helper_raw, ordinary_raw, ordinary_ref_raw)
    source_hashes = {}
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []));cell['metadata'] = {}
        if cell['cell_type'] == 'code':cell['execution_count'],cell['outputs']=None,[]
        if source.startswith('%%writefile '):
            header,body=source.split('\n',1);name=header.split()[1]
            ast.parse(body,filename=name);source_hashes[name]=sha(body.encode())
        elif source and not source.startswith('!'):ast.parse(source)
    raw = (json.dumps(notebook,ensure_ascii=False,indent=2)+'\n').encode()
    if len(raw)>=1000000:raise ValueError('Formal notebook exceeds source cap')
    name=args.slug+'.ipynb'
    metadata.update(id=args.owner+'/'+args.slug,title=args.slug.replace('-',' '),code_file=name,is_private=True)
    result=dict(status='prepared_not_launched_conditional_on_confirmation',ref=metadata['id'],
        notebook_sha256=sha(raw),notebook_bytes=len(raw),research_notebook_sha256=sha(research_raw),
        reference_script_version_id=312063593,reference_sha256=prior['reference_sha256'],
        selection_helper_sha256=sha(helper_raw),runtime_source_sha256=source_hashes,
        session_seconds=43200,solver_global_seconds=42600,solver_hard_seconds=43140,
        python_hash_seed=27027,ordinary_tasks=4,ordinary_queries=5,
        input_policy='Dynamic competition test input on rerun; exact original four tasks otherwise. No C23 data embedded.',
        scheduling=prior['scheduling'],base_backup_before_extra=True,
        extra_error_policy='Explicit base_fallback_after_extra_error; preserve only base two slots.',
        extra_deadline_policy='Only complete atomic pickle records may contribute through frozen helper; all queries remain in denominator.',
        gates='Do not invoke/upload until C23 final net gain and predeclared partial-prefix criteria pass; this tool does not infer success or submit.',
        execution='Not run; no runtime, Kaggle version or submission ref exists for this generated artifact.')
    out.mkdir(parents=True)
    (out/name).write_bytes(raw)
    (out/'kernel-metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    (out/'package-manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    main()
