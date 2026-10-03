"""Manual CPU preparation of one matched F0/F1 research pilot; never runs a model.

Input is the 7427-row materialization recorded in the 010 progress document.
Run at repository root after materialization. Existing output files are refused.
"""
import hashlib
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

from transformers import PreTrainedTokenizerFast

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
import arc_episode_data as data

SEED = 20261003
N = 400
LIMITS = [1024, 2048, 4096]
QUOTAS = [80, 160, 160]


def digest(x):
    return hashlib.sha256(str(x).encode()).hexdigest()


def main():
    start = time.perf_counter()
    src = Path('/tmp/arc010-materialized/f0.jsonl')
    dest = ROOT / 'experiments/010'
    outputs = [dest / 'paired-f0-400.jsonl', dest / 'paired-f1-400.jsonl', dest / 'paired-pilot-manifest.json']
    if any(p.exists() for p in outputs):
        raise RuntimeError('Refusing to overwrite an existing pilot')
    paths = [src, ROOT/'experiments/007/g1/split.json', ROOT/'experiments/008/local-audit.json',
             ROOT/'experiments/009/episode-semantics.json', ROOT/'experiments/008/data/gen_newbases.jsonl']
    rows = [json.loads(line) for line in src.read_text().splitlines()]
    split = json.loads(paths[1].read_text())
    train_ids = set(split['train'])
    arc1 = json.loads(paths[2].read_text())['sources']['arc1']
    old_train, old_eval = set(arc1['training']), set(arc1['evaluation'])
    newer = train_ids - old_train - old_eval
    gen_ids = {json.loads(line)['task_id'] for line in paths[4].read_text().splitlines()}
    protected = newer - gen_ids
    semantic = json.loads(paths[3].read_text())
    flagged = {(x['family'], int(x['instance'])) for key in ['query_input_duplicates', 'color_mapping_or_interaction_flags']
               for x in semantic[key]}
    cache = Path.home()/'.cache/qwen38-tokenizer'
    cfg = json.loads((cache/'tokenizer_config.json').read_text())
    tok = PreTrainedTokenizerFast(tokenizer_file=str(cache/'tokenizer.json'), eos_token=cfg['eos_token'],
        pad_token=cfg['pad_token'], chat_template=(cache/'chat_template.jinja').read_text())
    def lengths(ep):
        text = tok.apply_chat_template(ep['messages'], tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p = len(tok.encode(text, add_special_tokens=False))
        t = len(tok.encode(ep['target_grid_text'], add_special_tokens=False)) + 1
        return {'prompt': p, 'target_eos': t, 'full': p+t}
    eligible, excluded = [], Counter()
    for row in rows:
        tid = row['task_id']
        if tid not in train_ids:
            raise RuntimeError('Non-training task in materialized pool: '+tid)
        source = row['source']
        if source == 'arcgen_v2' and (tid, int(row.get('query_index', -1))) in flagged:
            excluded['known_009_flagged_newbase_rows'] += 1
            continue
        f0 = dict(row)
        f1 = data.convert_episode_formatter(f0, 'compact')
        back = data.convert_episode_formatter(f1, 'space') if f1 is not None else None
        if back is None or back['messages'] != f0['messages'] or back['target_grid_text'] != f0['target_grid_text']:
            raise RuntimeError('Roundtrip mismatch: '+row['episode_id'])
        a, b = lengths(f0), lengths(f1)
        if a['full'] > 4096 or b['full'] > 4096:
            excluded['outside_common_4k_domain'] += 1
            continue
        bucket = next(i for i,v in enumerate(LIMITS) if a['full'] <= v)
        base_key = '%s|%s|%s|%s' % (source, tid, row['query_kind'], row['query_index'])
        origin = 'arc1_train' if tid in old_train else 'arc1_eval' if tid in old_eval else 'non_arc1'
        for ep, lens in [(f0,a),(f1,b)]:
            ep['origin'] = origin
            ep['base_instance_id'] = base_key
            ep['prompt_tokens'] = lens['prompt']
            ep['target_tokens'] = lens['target_eos']
            ep.pop('d2_bucket', None)
        eligible.append({'id':row['episode_id'],'task':tid,'base':base_key,'source':source,'origin':origin,
                         'bucket':bucket,'f0':f0,'f1':f1,'lengths':{'f0':a,'f1':b}})
    task_n, base_n, source_n, bucket_n = Counter(), Counter(), Counter(), Counter()
    selected = []
    for i in range(N):
        b = max((j for j in range(3) if bucket_n[j]<QUOTAS[j]),
                key=lambda j:(QUOTAS[j]*(i+1)/N-bucket_n[j], -j))
        desired = 'official_train' if source_n['official_train'] < round((i+1)*0.4) else 'arcgen_v2'
        candidates = [r for r in eligible if r['bucket']==b and r['source']==desired]
        if not candidates:
            candidates = [r for r in eligible if r['bucket']==b]
        if not candidates:
            raise RuntimeError('Not enough candidates in bucket '+str(b))
        row = min(candidates, key=lambda r:(
            0 if r['source']=='official_train' and r['task'] in protected and task_n[r['task']]==0 else 1,
            task_n[r['task']], base_n[r['base']], digest('%s:%s'%(SEED,r['id']))))
        selected.append(row);eligible.remove(row)
        task_n[row['task']]+=1;base_n[row['base']]+=1;source_n[row['source']]+=1;bucket_n[b]+=1
    def stats(items):
        return {'rows':len(items),'tasks':len({r['task'] for r in items}),'bases':len({r['base'] for r in items}),
                'sources':dict(Counter(r['source'] for r in items)),
                'origins':dict(Counter(r['origin'] for r in items)),
                'protected_tasks':len({r['task'] for r in items}&protected),
                'f0_reference_buckets':dict(Counter(r['bucket'] for r in items)),
                'tokens':{f:{k:sum(r['lengths'][f][k] for r in items) for k in ['prompt','target_eos','full']}
                          for f in ['f0','f1']}}
    for f,p in zip(['f0','f1'],outputs[:2]):
        p.write_text(''.join(json.dumps(r[f],ensure_ascii=False,separators=(',',':'))+'\n' for r in selected))
    # Existing legacy path shuffles both identically when every row fits max_len.
    consumed = list(selected);random.Random(SEED).shuffle(consumed)
    manifest = {'date':'2026-10-03','status':'CPU_PREPARED_NOT_GPU_VALIDATED','seed':SEED,
        'selection':'Same raw episodes for F0/F1. F0 full-length buckets 80/160/160. Approximate40/60 source target, least-seen task/base and explicit55-task protection. Then existing legacy shuffle, same seed in both arms.',
        'paired_comparison':'same examples/targets/order, NOT equal tokens or equal wall-clock',
        'quality_scope':'Only17 known flagged new bases removed; remaining pool is not fully semantic-certified. Repair pilot deliberately excluded.',
        'input_rows':len(rows),'excluded':dict(excluded),'protected_task_count':len(protected),
        'protected_task_ids':sorted(protected),'stats':stats(selected),
        'prefix_by_optimizer_update':{str(u):stats(consumed[:u*4]) for u in [25,50,100]},
        'expected_legacy_consumption_order':[r['id'] for r in consumed],
        'rows':[{'episode_id':r['id'],'task_id':r['task'],'base_instance_id':r['base'],'lengths':r['lengths']} for r in selected],
        'required_training_flags':{'sample_policy':'legacy','length_mixture':None,'max_len':4096,'overflow_len':4096,'seed':SEED,'grad_accum':4,'lr_horizon_updates':100},
        'warning':'Do not independently resample by F1 token buckets. Fix init-adapter and formatter/metadata paths before GPU training. Max time and shared-model runner are not implemented by this preparation.',
        'input_sha256':{str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths},
        'output_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in outputs[:2]},
        'code_sha256':{str(ROOT/'scripts/arc_episode_data.py'):hashlib.sha256((ROOT/'scripts/arc_episode_data.py').read_bytes()).hexdigest()},
        'elapsed_seconds':time.perf_counter()-start}
    outputs[2].write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({'stats':manifest['stats'],'excluded':manifest['excluded'],'elapsed_seconds':manifest['elapsed_seconds']},ensure_ascii=False))


if __name__=='__main__':
    main()
