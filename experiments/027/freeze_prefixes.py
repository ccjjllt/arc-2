#!/usr/bin/env python3
"""Freeze predeclared partial-extra fallback predictions before reading gold."""
import argparse
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime,timezone
from pathlib import Path


def read(path,lines=False):
    raw=path.read_bytes()
    return ([json.loads(s) for s in raw.decode().splitlines() if s.strip()] if lines else json.loads(raw)),dict(path=str(path.resolve()),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root',required=True,type=Path)
    parser.add_argument('--manifest',required=True,type=Path)
    parser.add_argument('--frozen-dir',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest,ident=read(args.manifest);inputs=[ident]
    _,ident=read(args.frozen_dir/'prediction-freeze.json');inputs.append(ident)
    base,ident=read(args.frozen_dir/'submission-base_B4.json');inputs.append(ident)
    ensemble,ident=read(args.frozen_dir/'submission-ensemble.json');inputs.append(ident)
    completed={}
    for path in sorted((args.run_root/'extra').glob('events-rank*.jsonl')):
        events,ident=read(path,True);inputs.append(ident)
        for event in events:
            if event['event']=='task_end':
                key=event['task_id']
                if key in completed:raise ValueError('Duplicate extra task completion')
                completed[key]=event['timestamp']
    allowed=set(base)
    if set(completed)-allowed:raise ValueError('Extra event outside panel')
    ordered=sorted(completed,key=lambda key:(completed[key],key))
    args.output.mkdir(parents=True,exist_ok=False)
    outputs={}
    for percent in (25,50,75):
        requested=math.ceil(manifest['n_tasks']*percent/100)
        selected=set(ordered[:requested])
        prediction={key:ensemble[key] if key in selected else base[key] for key in base}
        filename='submission-prefix%d.json'%percent
        raw=(json.dumps(prediction,ensure_ascii=False,sort_keys=True)+'\n').encode()
        (args.output/filename).write_bytes(raw)
        outputs[filename]=dict(sha256=hashlib.sha256(raw).hexdigest(),bytes=len(raw),target_complete_tasks=requested,actual_complete_tasks=len(selected),target_reached=len(ordered)>=requested,task_ids=sorted(selected))
    record=dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),gold_read=False,cohort=manifest['cohort'],n_tasks=manifest['n_tasks'],n_queries=manifest['n_queries'],rule='Use already frozen final ensemble only for first ceil(N*25/50/75%) completed extra tasks, sorted by task_end time then task_id; otherwise retain full base slots. No new selection or gold read.',completed_extra_task_order=ordered,outputs=outputs,inputs=inputs)
    (args.output/'prediction-freeze.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(record,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
