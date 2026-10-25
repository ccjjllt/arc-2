#!/usr/bin/env python3
"""Manually score frozen027 predictions against exact official Kaggle inputs.

Reuses020 scoring logic. Targets stay in the local official download snapshot;
no model call, new selection, or experiment launch occurs here.
"""
import argparse
import hashlib
import importlib.util
import json
from datetime import datetime,timezone
from pathlib import Path
from types import SimpleNamespace

HERE=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('frozen020',HERE.parent/'020/score_frozen_outputs.py')
score=importlib.util.module_from_spec(spec);spec.loader.exec_module(score)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--snapshot-manifest',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--allow-gold-read',action='store_true')
    args=p.parse_args()
    if not args.allow_gold_read:p.error('Use --allow-gold-read only after complete and prefix freeze')
    frozen=HERE/'frozen-v1';prefixes=HERE/'prefixes-v1'
    prediction=[name+'='+str(frozen/('submission-'+name+'.json')) for name in ['base_B4','extra_B4','ensemble']]+[name+'='+str(prefixes/('submission-'+name+'.json')) for name in ['prefix25','prefix50','prefix75']]
    config=SimpleNamespace(manifest=HERE/'confirmation-v1/manifest.json',challenges=HERE/'confirmation-v1/challenges.json',prediction=prediction,candidate_pool=[frozen/'candidate-pools-scored.json'])
    manifest,challenges,policies,candidates,identities=score.load_inputs(config)
    complete_freeze=json.loads((frozen/'prediction-freeze.json').read_text());prefix_freeze=json.loads((prefixes/'prediction-freeze.json').read_text())
    for item in identities['predictions']:
        want=(prefix_freeze if item['policy'].startswith('prefix') else complete_freeze)['outputs']['submission-'+item['policy']+'.json']['sha256']
        if item['sha256']!=want:raise ValueError('Prediction changed after original no-gold freeze')
    provenance=json.loads(args.snapshot_manifest.read_text())
    args.output.mkdir(parents=True,exist_ok=False)
    score.write_json(args.output/'prediction-freeze.json',dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),cohort=manifest['cohort'],planned_tasks=manifest['n_tasks'],planned_queries=manifest['n_queries'],artifacts=identities,original_complete_freeze=str((frozen/'prediction-freeze.json').resolve()),original_prefix_freeze=str((prefixes/'prediction-freeze.json').resolve()),original_predictions_unchanged=True,official_source_kind='Official Kaggle competition challenge/solution snapshot',snapshot_manifest=provenance,note='Original full and prefix predictions were frozen before all gold access. This followup resolves fixed-GitHub input-version mismatches using the actual competition snapshot; no prediction changed.'))
    data={}
    for name,record in provenance['files'].items():
        raw=Path(record['path']).read_bytes()
        if len(raw)!=record['bytes'] or hashlib.sha256(raw).hexdigest()!=record['sha256']:raise ValueError('Official snapshot identity changed')
        data[name]=json.loads(raw)
    official_ch=data['arc-agi_evaluation_challenges.json'];solutions=data['arc-agi_evaluation_solutions.json']
    rows=[];sources=[]
    for entry in manifest['entries']:
        key=entry['task_id'];source_task=official_ch[key]
        if source_task!=challenges[key] or len(solutions[key])!=len(source_task['test']):raise ValueError('Official inputs/query count differ '+key)
        official=dict(train=source_task['train'],test=[dict(input=query['input'],output=target) for query,target in zip(source_task['test'],solutions[key])])
        sources.append(dict(task_id=key,status='loaded',source_kind='Kaggle competition evaluation JSON pair',challenge_sha256=provenance['files']['arc-agi_evaluation_challenges.json']['sha256'],solutions_sha256=provenance['files']['arc-agi_evaluation_solutions.json']['sha256'],full_task_content_matches=True))
        for query in entry['queries']:
            idx,qkey=query['query_index'],query['query_key'];gold,status,indices=score.resolve_gold(challenges[key],official,idx)
            row=dict(task_id=key,query_index=idx,query_key=qkey,source_group=entry.get('source_group'),tier=entry.get('tier'),input_sha256=query['input_sha256'],gold_match_status=status,official_matching_indices=indices,gold_sha256=score.fingerprint(gold) if gold is not None else None,policies={})
            for name,pred in policies.items():
                result=score.score_prediction(pred,key,idx,gold)
                if gold is not None and name in candidates:
                    oracle=any(candidate.get('grid')==gold for candidate in candidates[name].get(qkey,[]))
                    result.update(candidate_oracle_exact=oracle,correct_candidate_not_selected=oracle and not result['exact'])
                row['policies'][name]=result
            rows.append(row)
    with (args.output/'query-ledger.jsonl').open('w') as stream:
        for row in rows:stream.write(json.dumps(row,allow_nan=False)+'\n')
    report=score.summarize(rows,policies);report.update(cohort=manifest['cohort'],planned_tasks=manifest['n_tasks'],official_sources=sources,snapshot_manifest=provenance,interpretation='Input-matched public Kaggle evaluation diagnostic only; not leaderboard score.')
    score.write_json(args.output/'score-report.json',report)
    print(json.dumps({key:value for key,value in report.items() if key not in ['official_sources','snapshot_manifest']},ensure_ascii=False))


if __name__=='__main__':main()
