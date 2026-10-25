#!/usr/bin/env python3
"""Manually freeze both completed arms and candidate pools before any gold read."""
import argparse
import hashlib
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, lines=False):
    raw=path.read_bytes()
    return ([json.loads(line) for line in raw.decode().splitlines() if line.strip()] if lines else json.loads(raw)),dict(path=str(path.resolve()),bytes=len(raw),sha256=sha(raw))


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def fingerprint(grid):
    return sha(json.dumps(grid,sort_keys=True,separators=(',',':')).encode())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest,mident=read(args.manifest)
    state,sident=read(args.run_root/'run-state.json')
    if state['status']!='completed' or state['return_code']!=0:
        raise ValueError('Incomplete experiment; inspect manually before freezing a comparison')
    if state['input_identity']['challenges_sha256']!=manifest['input_artifact']['sha256']:
        raise ValueError('Runtime inputs differ from frozen panel')
    keys={q['query_key'] for e in manifest['entries'] for q in e['queries']}
    if state['planned_queries']!=len(keys):
        raise ValueError('Query denominator differs')
    outputs,candidates,ledger,inputs={},{},[],[mident,sident]
    args.output.mkdir(parents=True,exist_ok=False)
    for arm in ('base','challenger'):
        policy=arm+'_B4'
        root=args.run_root/arm
        prediction,pident=read(root/'submission-B4.json')
        if pident['sha256']!=state['arms'][arm]['prediction_sha256']:
            raise ValueError('Runtime final prediction hash differs')
        scored,cident=read(root/'scored-candidates.jsonl',True)
        inputs.extend([pident,cident])
        pools=defaultdict(dict)
        for row in scored:
            if row['query_key'] not in keys: raise ValueError('Candidate outside cohort')
            pools[row['query_key']][fingerprint(row['solution'])]=row['solution']
        raw=defaultdict(set)
        for path in sorted(root.glob('raw-candidates-rank*.jsonl')):
            records,ident=read(path,True);inputs.append(ident)
            for row in records:
                if row['query_key'] not in keys: raise ValueError('Raw candidate outside cohort')
                raw[row['query_key']].add(fingerprint(row['solution']))
        candidates[policy]={key:[{'grid':grid} for _,grid in sorted(pools[key].items())] for key in sorted(keys)}
        for key in sorted(keys):
            ledger.append(dict(arm=arm,query_key=key,raw_candidate_sha256=sorted(raw[key]),scored_candidate_sha256=sorted(pools[key]),
                raw_unique_candidates=len(raw[key]),scored_unique_candidates=len(pools[key])))
        destination=args.output/('submission-'+policy+'.json')
        destination.write_bytes((root/'submission-B4.json').read_bytes())
        outputs[destination.name]=dict(sha256=pident['sha256'],bytes=pident['bytes'])
    write(args.output/'candidate-pools.json',candidates)
    with (args.output/'candidate-ledger.jsonl').open('w') as stream:
        for row in ledger: stream.write(json.dumps(row)+'\n')
    for name in ('candidate-pools.json','candidate-ledger.jsonl'):
        raw=(args.output/name).read_bytes();outputs[name]=dict(sha256=sha(raw),bytes=len(raw))
    freeze=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),cohort=manifest['cohort'],n_tasks=manifest['n_tasks'],n_queries=len(keys),
        gold_read=False,outputs=outputs,inputs=inputs,note='Both runtime B4 outputs copied byte-for-byte before gold. Candidate pools are diagnostics only.')
    write(args.output/'prediction-freeze.json',freeze)
    print(json.dumps(freeze,indent=2))


if __name__=='__main__':main()
