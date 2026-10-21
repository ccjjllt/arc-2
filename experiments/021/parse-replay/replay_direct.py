#!/usr/bin/env python3
"""Manual CPU replay of frozen 020 direct responses; no model calls or gold download.

Compares the exact archived parse_direct function with a strict per-query policy.
Only a complete standard JSON object is accepted. Duplicate/unknown keys reject
the object; valid expected members survive missing or invalid sibling members.
Existing official-input-matched gold hashes are used for exact comparison, after
checking the frozen challenge/manifest identities and per-query input hashes.
"""
from __future__ import annotations
import argparse
import ast
from collections import Counter
from datetime import datetime, timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import platform
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / 'experiments/020/submit/vdriver020_submit.py'
LIB = ROOT / 'experiments/020/submit/submit_lib.py'
COHORTS = [
    dict(name='C24', manifest='experiments/020/cohorts/confirmation-c24-manifest.json',
         challenges='experiments/020/cohorts/confirmation-c24-challenges.json',
         score_dir='experiments/020/scoring-c24-v1', policy='Direct1PlusLine',
         runs=['experiments/020/c24-direct-ccjjllt-v3/vprobe',
               'experiments/020/c24-direct-qingyu1232-v1/vprobe']),
    dict(name='Deep6', manifest='experiments/020/deep-direct/manifest.json',
         challenges='experiments/020/deep-p0/challenges.json',
         score_dir='experiments/020/scoring-deep-direct-v1', policy='DeepDirect',
         runs=['experiments/020/deep-direct-qingyu1232-v1/vprobe']),
]

def fingerprint(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                    separators=(',', ':')).encode()).hexdigest()

def identity(path):
    path = Path(path)
    raw = path.read_bytes()
    return {'path': str(path.relative_to(ROOT)), 'sha256': hashlib.sha256(raw).hexdigest(),
            'bytes': len(raw)}

def read(path):
    return json.loads(Path(path).read_text())

def load_old_parser():
    spec = importlib.util.spec_from_file_location('arc020_submit_lib', LIB)
    library = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(library)
    source = SOURCE.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)
                and n.name == 'parse_direct')
    isolated = ast.Module(body=[node], type_ignores=[])
    namespace = dict(json=json, hashlib=hashlib, SL=library)
    exec(compile(isolated, str(SOURCE), 'exec'), namespace)
    return namespace['parse_direct'], library.valid_grid, ast.get_source_segment(source, node)


def per_query_parse(rec, res, task, valid_grid):
    count = len(task['test'])
    grids, errors = [None] * count, [None] * count
    def fail(message):
        return grids, [message] * count, 'object_rejected'
    if (res.get('finish') not in ('stop', 'length') or rec.get('client_error')
            or rec.get('cancelled_by_client')):
        return fail('generation_incomplete')
    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value
    def reject_constant(value):
        raise ValueError('nonstandard JSON constant')
    try:
        text = (res.get('content') or '').strip()
        if text.startswith('```'):
            lines = text.splitlines()
            if len(lines) < 3 or lines[0].strip() not in ('```json', '```') or lines[-1].strip() != '```':
                raise ValueError('expected one complete JSON fence')
            text = '\n'.join(lines[1:-1])
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        keys = ['query%d' % i for i in range(count)]
        if not isinstance(value, dict):
            raise ValueError('expected object')
        if set(value) - set(keys):
            raise ValueError('unknown query key')
    except (ValueError, TypeError) as exc:
        return fail(str(exc))
    for index, key in enumerate(keys):
        if key not in value:
            errors[index] = 'missing query key'
        else:
            try:
                grids[index] = valid_grid(value[key])
            except (ValueError, TypeError) as exc:
                errors[index] = str(exc)
    return grids, errors, 'all_queries_valid' if all(g is not None for g in grids) else 'partial_or_invalid'


def run(output):
    started = time.monotonic()
    started_at = datetime.now(timezone.utc).isoformat()
    old_parse, valid_grid, old_function = load_old_parser()
    source_files = [identity(SOURCE), identity(LIB), identity(Path(__file__))]
    all_rows, cohort_reports, artifact_ids = [], [], []
    for cohort in COHORTS:
        challenges_path, manifest_path = ROOT / cohort['challenges'], ROOT / cohort['manifest']
        score_dir = ROOT / cohort['score_dir']
        challenges, manifest = read(challenges_path), read(manifest_path)
        freeze, prior_report = read(score_dir / 'prediction-freeze.json'), read(score_dir / 'score-report.json')
        if identity(challenges_path)['sha256'] != freeze['artifacts']['challenges']['sha256']:
            raise ValueError('frozen challenge identity mismatch')
        if identity(manifest_path)['sha256'] != freeze['artifacts']['manifest']['sha256']:
            raise ValueError('frozen manifest identity mismatch')
        entries = {r['task_id']: r for r in manifest['entries']}
        if set(entries) != set(challenges):
            raise ValueError('cohort task IDs mismatch')
        ledgers = [json.loads(line) for line in (score_dir / 'query-ledger.jsonl').read_text().splitlines() if line]
        ledger_by_key = {r['query_key']: r for r in ledgers}
        for filename in [challenges_path, manifest_path, score_dir/'prediction-freeze.json',
                         score_dir/'score-report.json', score_dir/'query-ledger.jsonl']:
            artifact_ids.append(identity(filename))
        raw_by_task, archived_runs, run_sources = {}, {}, {}
        for rel in cohort['runs']:
            base = ROOT / rel
            archived = read(base/'run.json')
            artifact_ids.append(identity(base/'run.json'))
            if set(raw_by_task) & set(archived['tasks']):
                raise ValueError('run shards overlap')
            for task_id, task_record in archived['tasks'].items():
                reply = base/'programs'/('P0-'+task_id+'.reply.json')
                raw_by_task[task_id] = read(reply)
                artifact_ids.append(identity(reply))
                archived_runs[task_id] = task_record['P0']
                run_sources[task_id] = {'run': rel, 'source_sha256': archived['source_sha256']}
        if set(raw_by_task) != set(challenges):
            raise ValueError('raw replies do not cover original cohort')
        summary = Counter(tasks=len(challenges), queries=0, matched_queries=0, old_legal=0, new_legal=0,
                          recovered_legal=0, lost_legal=0, old_exact=0, new_exact=0, recovered_exact=0, lost_exact=0)
        invalid_tasks = []
        for task_id, task in challenges.items():
            entry = entries[task_id]
            if fingerprint(task['train']) != entry['demo_pairs_sha256']:
                raise ValueError('demo input mismatch')
            raw = raw_by_task[task_id]
            old = old_parse(raw['rec'], raw['res'], task)
            archived = archived_runs[task_id]
            if old['status'] != archived['status'] or old['query_results'] != archived['query_results']:
                raise ValueError('old parser replay differs from archived parse: '+task_id)
            new_grids, errors, status = per_query_parse(raw['rec'], raw['res'], task, valid_grid)
            old_grids = [r['grid'] for r in old['query_results']] if old['gen_ok'] else [None] * len(task['test'])
            if not old['gen_ok']:
                invalid_tasks.append({'task_id': task_id, 'queries': len(task['test']),
                                      'old_error': old.get('validation_error'), 'new_status': status,
                                      'new_errors': errors, 'finish': raw['res'].get('finish'),
                                      'completion_tokens': raw['rec'].get('usage',{}).get('completion_tokens'),
                                      'content_chars': len(raw['res'].get('content',''))})
            for index, query in enumerate(task['test']):
                key = f'{task_id}_{index}'
                prior = ledger_by_key[key]
                query_entry = next(q for q in entry['queries'] if q['query_index'] == index)
                input_hash = fingerprint(query['input'])
                if input_hash != query_entry['input_sha256'] or input_hash != prior['input_sha256']:
                    raise ValueError('query input mismatch: '+key)
                matched = prior['gold_match_status'] == 'matched'
                gold_hash = prior.get('gold_sha256') if matched else None
                old_grid, new_grid = old_grids[index], new_grids[index]
                old_exact = old_grid is not None and matched and fingerprint(old_grid) == gold_hash
                new_exact = new_grid is not None and matched and fingerprint(new_grid) == gold_hash
                old_prior = prior['policies'][cohort['policy']]
                # C24 CPU rules admitted no query; DeepDirect policy is pure direct.
                if matched and bool(old_prior['exact']) != old_exact:
                    raise ValueError('old replay correctness differs from frozen prior ledger: '+key)
                row = dict(cohort=cohort['name'], task_id=task_id, query_index=index, query_key=key,
                           input_sha256=input_hash, gold_match_status=prior['gold_match_status'],
                           gold_sha256=gold_hash, old_legal=old_grid is not None, new_legal=new_grid is not None,
                           old_grid_sha256=fingerprint(old_grid) if old_grid is not None else None,
                           new_grid_sha256=fingerprint(new_grid) if new_grid is not None else None,
                           old_exact=old_exact if matched else None, new_exact=new_exact if matched else None,
                           old_status=old['status'], old_error=old.get('validation_error'), new_error=errors[index],
                           response_completion_sha256=old['completion_sha256'])
                all_rows.append(row)
                summary.update(queries=1, matched_queries=int(matched), old_legal=int(old_grid is not None),
                               new_legal=int(new_grid is not None),
                               recovered_legal=int(old_grid is None and new_grid is not None),
                               lost_legal=int(old_grid is not None and new_grid is None),
                               old_exact=int(old_exact), new_exact=int(new_exact),
                               recovered_exact=int(new_exact and not old_exact), lost_exact=int(old_exact and not new_exact))
        expected = prior_report['policies'][cohort['policy']]['exact_queries']
        if summary['old_exact'] != expected:
            raise ValueError('old exact aggregate differs')
        cohort_reports.append(dict(cohort=cohort['name'], **summary,
                                   official_revision=freeze['official_revision'], official_split=freeze['official_split'],
                                   unmatched_queries=summary['queries']-summary['matched_queries'],
                                   old_em2=summary['old_exact']/summary['matched_queries'],
                                   new_em2=summary['new_exact']/summary['matched_queries'],
                                   invalid_tasks=invalid_tasks, model='Qwen/Qwen3.8-27B FP8 repack v1',
                                   archived_model_root=read(ROOT/cohort['runs'][0]/'run.json')['model_root'],
                                   raw_run_source_identities=run_sources))
    output.mkdir(parents=True, exist_ok=False)
    report = dict(started_at_utc=started_at, finished_at_utc=datetime.now(timezone.utc).isoformat(),
                  local_cpu_wall_seconds=time.monotonic()-started,
                  hardware=dict(machine=platform.machine(), platform=platform.platform(), python=sys.version,
                                GPU_used=False), new_model_calls=0, source_files=source_files,
                  old_parse_function_sha256=hashlib.sha256(old_function.encode()).hexdigest(),
                  policy='Complete strict JSON object; duplicate/unknown keys reject whole object; independently retain complete valid expected members; no text/array recovery or inferred cells.',
                  scoring='Compare canonical grid SHA256 to previously input-matched official gold SHA256 from frozen 020 ledgers. No gold grids read or downloaded. Old replay verified against archived parse and frozen exact result for every matched query.',
                  cohorts=cohort_reports, input_artifacts=artifact_ids)
    (output/'report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    (output/'query-ledger.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in all_rows))
    print(json.dumps({k:v for k,v in report.items() if k in ['local_cpu_wall_seconds','new_model_calls']}))
    print(json.dumps([{k:c[k] for k in ['cohort','tasks','queries','matched_queries','old_legal','new_legal','recovered_legal','old_exact','new_exact','recovered_exact']} for c in cohort_reports],ensure_ascii=False))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    run(args.output_dir)
