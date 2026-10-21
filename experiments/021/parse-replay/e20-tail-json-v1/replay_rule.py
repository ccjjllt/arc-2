#!/usr/bin/env python3
"""Manual no-gold E20 structural audit and frozen tail-JSON diagnostic derivation.

Rule fixed before scoring: scan every opening brace of each saved final content;
accept only one complete standard JSON object ending at content end (whitespace
permitted), with exactly every expected query key and every grid strictly valid.
No character, grid cell, key, query, or answer is repaired. Duplicate keys and
nonstandard JSON constants are rejected. Apply uniformly to all 20 replies.
"""
from __future__ import annotations
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
RAW_BASE = ROOT/'experiments/021/qwen-e20-output-v1/vprobe'
FREEZE = ROOT/'experiments/021/execution/qwen-e20-v1-artifact-freeze.json'
CHALLENGE = ROOT/'experiments/020/evaluation-e20/challenges.json'
TOKENIZER = Path('/private/tmp/arc2-020-tokenizer-v1/tokenizer.json')
TOKENIZER_SHA = '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
AUDIT = ROOT/'experiments/021/execution/qwen-e20-parse-audit-v1.json'

def stamp(): return datetime.now(timezone.utc).isoformat()
def sha_bytes(value): return hashlib.sha256(value).hexdigest()
def ident(path):
    raw=path.read_bytes()
    try: name=str(path.relative_to(ROOT))
    except ValueError: name=str(path)
    return dict(path=name,sha256=sha_bytes(raw),bytes=len(raw))
def read(path): return json.loads(path.read_text())
def gridsha(grid):return sha_bytes(json.dumps(grid,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode())
def unique_object(pairs):
    obj={}
    for key,value in pairs:
        if key in obj: raise ValueError('duplicate JSON key')
        obj[key]=value
    return obj
def reject_constant(value):raise ValueError('nonstandard JSON constant')

def main():
    started=time.monotonic(); fixed_at=stamp()
    if AUDIT.exists() or (HERE/'submission.json').exists() or (HERE/'candidate-freeze.json').exists():
        raise FileExistsError('Do not overwrite a prior diagnostic freeze')
    frozen=read(FREEZE); frozen_map={r['path']:r for r in frozen['files']}
    for path in [RAW_BASE/'run.json',RAW_BASE/'submission-M1.json',*sorted((RAW_BASE/'programs').glob('*.reply.json'))]:
        identity=ident(path)
        if identity['sha256']!=frozen_map[identity['path']]['sha256']:
            raise ValueError('Frozen artifact changed: '+str(path))
    run=read(RAW_BASE/'run.json');challenges=read(CHALLENGE)
    if ident(CHALLENGE)['sha256']!=run['challenges_sha256'] or len(challenges)!=20:
        raise ValueError('E20 input changed')
    if sum(len(t['test']) for t in challenges.values())!=27:raise ValueError('E20 query count changed')
    spec=importlib.util.spec_from_file_location('arc021_parse_replay',ROOT/'experiments/021/parse-replay/replay_direct.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    old_parse,valid_grid,old_source=module.load_old_parser()
    decoder=json.JSONDecoder(object_pairs_hook=unique_object,parse_constant=reject_constant)
    import tokenizers
    if ident(TOKENIZER)['sha256']!=TOKENIZER_SHA:raise ValueError('Tokenizer identity changed')
    tok=tokenizers.Tokenizer.from_file(str(TOKENIZER))
    endthink_ids=tok.encode('</think>',add_special_tokens=False).ids
    derived=deepcopy(read(RAW_BASE/'submission-M1.json'))
    rows=[];changed=[]
    rule='For each of all 20 final strings, scan every opening brace. Accept exactly one complete standard JSON object whose end is followed only by whitespace, whose root keys equal all expected query keys, and whose every grid is valid; reject duplicates/nonstandard constants. No text/cell modification. If no unique valid tail object, retain original submission entry unchanged.'
    for tid,task in challenges.items():
        path=RAW_BASE/'programs'/('P0-'+tid+'.reply.json'); reply=read(path);rec,res=reply['rec'],reply['res']
        old=old_parse(rec,res,task); archived=run['tasks'][tid]['P0']
        if old['status']!=archived['status'] or old['query_results']!=archived['query_results']:
            raise ValueError('Old replay mismatch '+tid)
        strict_grids,strict_errors,strict_status=module.per_query_parse(rec,res,task,valid_grid)
        content=res.get('content') or '';reason=res.get('reasoning') or ''
        keys=['query%d'%i for i in range(len(task['test']))]
        matches=[]
        if res.get('finish') in ('stop','length') and not rec.get('client_error') and not rec.get('cancelled_by_client'):
            for start,char in enumerate(content):
                if char!='{':continue
                try:
                    value,end=decoder.raw_decode(content,start)
                    if content[end:].strip() or not isinstance(value,dict) or set(value)!=set(keys):continue
                    grids=[valid_grid(value[key]) for key in keys]
                    matches.append(dict(start_character=start,end_character=end,grids=grids,
                                        exact_extracted_text_sha256=sha_bytes(content[start:end].encode())))
                except (ValueError,TypeError):pass
        tail_accepted=len(matches)==1
        before=deepcopy(derived[tid])
        if tail_accepted:
            derived[tid]=[{'attempt_1':grid,'attempt_2':grid} for grid in matches[0]['grids']]
        if derived[tid]!=before:changed.append(tid)
        rcount=len(tok.encode(reason,add_special_tokens=False).ids)
        ccount=len(tok.encode(content,add_special_tokens=False).ids)
        row={'task_id':tid,'queries':len(keys),'reply':ident(path),'finish':res.get('finish'),
             'old_status':old['status'],'old_error':old.get('validation_error'),
             'old_legal_queries':len(old['query_results']),'strict_per_query_status':strict_status,
             'strict_per_query_legal_queries':sum(g is not None for g in strict_grids),
             'strict_per_query_errors':strict_errors,'content_chars':len(content),'reasoning_chars':len(reason),
             'reencoded_reasoning_tokens':rcount,'reencoded_content_tokens':ccount,
             'end_think_reencoded_tokens':len(endthink_ids),
             'completion_tokens':rec.get('usage',{}).get('completion_tokens'),
             'usage_minus_reencoded_text_and_endthink':rec.get('usage',{}).get('completion_tokens',0)-rcount-ccount-len(endthink_ids),
             'reasoning_reaches_requested_boundary':rcount+len(endthink_ids)==run['config']['thinking_max'],
             'unique_valid_tail_object_count':len(matches),'tail_policy_admitted':tail_accepted,
             'derived_entry_differs':derived[tid]!=before,
             'tail_candidates':[{k:v for k,v in m.items() if k!='grids'}|{'query_shapes':[[len(g),len(g[0])] for g in m['grids']],
                                 'grid_sha256':[gridsha(g) for g in m['grids']]} for m in matches]}
        if old['status']!='direct_ready':
            row.update(content_first_240=content[:240],content_last_240=content[-240:],reasoning_last_160=reason[-160:])
        rows.append(row)
    raw=json.dumps(derived,ensure_ascii=False,separators=(',',':'),allow_nan=False)+'\n'
    (HERE/'submission.json').write_text(raw)
    candidate_freeze={'frozen_at_utc':stamp(),'status':'derived_diagnostic_candidate_frozen_before_derived_gold_scoring',
                      'policy_fixed_at_utc':fixed_at,'policy':rule,'source_freeze':ident(FREEZE),
                      'source_submission':ident(RAW_BASE/'submission-M1.json'),'derived_submission':ident(HERE/'submission.json'),
                      'script':ident(Path(__file__)),'challenge':ident(CHALLENGE),'changed_task_ids':changed,
                      'scope':'All20 replies uniformly replayed; no gold, no model calls, original outputs unchanged.',
                      'accuracy_claim':'none; pending separately input-matched scoring by primary agent'}
    (HERE/'candidate-freeze.json').write_text(json.dumps(candidate_freeze,ensure_ascii=False,indent=2)+'\n')
    invalid=[r for r in rows if r['old_status']!='direct_ready'];length=[r for r in invalid if r['finish']=='length']
    total_tokens=sum(r['completion_tokens'] for r in rows)
    audit={'audited_at_utc':stamp(),'query_gold_read':False,'new_model_calls':0,'GPU_used':False,
           'raw_outputs_modified':False,'source_freeze':ident(FREEZE),'source_run':ident(RAW_BASE/'run.json'),
           'frozen_artifact_hashes_verified':True,'old_parse_function_sha256':sha_bytes(old_source.encode()),
           'tokenizer':ident(TOKENIZER)|{'library':'tokenizers','version':tokenizers.__version__,
                                      'encoding':'Saved reasoning/content independently, add_special_tokens=False; no raw generated token stream available'},
           'summary':{'tasks':20,'queries':27,'issued_calls':20,'generation_finished':20,'old_accepted_tasks':14,
                      'old_legal_queries':sum(r['old_legal_queries'] for r in rows),
                      'strict_per_query_legal_queries':sum(r['strict_per_query_legal_queries'] for r in rows),
                      'strict_per_query_recovered_queries':sum(r['strict_per_query_legal_queries']-r['old_legal_queries'] for r in rows),
                      'invalid_tasks':len(invalid),'invalid_queries':sum(r['queries'] for r in invalid),
                      'length_invalid_tasks':len(length),'length_invalid_queries':sum(r['queries'] for r in length),
                      'length_invalid_tokens':sum(r['completion_tokens'] for r in length),
                      'length_invalid_share_completion_tokens':sum(r['completion_tokens'] for r in length)/total_tokens,
                      'all_invalid_tokens':sum(r['completion_tokens'] for r in invalid),
                      'all_invalid_share_completion_tokens':sum(r['completion_tokens'] for r in invalid)/total_tokens,
                      'sum_completion_tokens':total_tokens,'tasks_reaching_reencoded_thinking_boundary':sum(r['reasoning_reaches_requested_boundary'] for r in rows),
                      'tail_rule_accepted_tasks':sum(r['tail_policy_admitted'] for r in rows),
                      'tail_rule_legal_queries':sum(r['queries'] for r in rows if r['tail_policy_admitted']),
                      'tail_rule_changed_task_ids':changed,'tail_rule_recovered_legal_queries':sum(r['queries'] for r in rows if r['derived_entry_differs']),
                      'tail_rule_accuracy':'not assessed'},
           'conclusions':['Strict per-query parsing cannot rescue any reply because every rejection occurs before a complete root JSON object is admitted.',
                          'Five length failures have final-channel analysis rather than a final JSON object; this is final delivery failure, not global timeout or sibling-query discard.',
                          '7491f3cf stopped normally and has explanatory prose before one complete valid tail JSON object; a uniform tail-object extraction diagnostic is frozen separately.',
                          'Saved-text retokenization supports budget-triggered reasoning-to-final switching; it cannot reconstruct the original raw token stream or prove longer thinking would improve answers.'],
           'derived_candidate_freeze':ident(HERE/'candidate-freeze.json'),'rows':rows,
           'local_cpu_wall_seconds':time.monotonic()-started,'platform':platform.platform(),'python':sys.version}
    AUDIT.write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(audit['summary'],ensure_ascii=False));print(json.dumps(candidate_freeze['derived_submission']))

if __name__=='__main__':main()
