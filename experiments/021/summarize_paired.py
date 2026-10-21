"""Manually summarize already-scored policies without reading any answer grids."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def identity(path):
    raw = path.read_bytes()
    return raw, {'path': str(path.resolve()), 'sha256': hashlib.sha256(raw).hexdigest()}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--ledger', type=Path, required=True)
    ap.add_argument('--baseline', required=True)
    ap.add_argument('--candidate-pool', action='append', default=[], metavar='POLICY=PATH',
                    help='Optional ranked candidate export; a policy map or a query-key map.')
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    report_raw, report_id = identity(args.report)
    ledger_raw, ledger_id = identity(args.ledger)
    report = json.loads(report_raw)
    rows = [json.loads(line) for line in ledger_raw.decode().splitlines() if line.strip()]
    policies = list(report['policies'])
    if args.baseline not in policies:
        raise ValueError('Baseline is absent from score report')
    if len(rows) != report['planned_queries']:
        raise ValueError('Ledger length differs from score report planned denominator')
    if len({r['query_key'] for r in rows}) != len(rows):
        raise ValueError('Ledger contains duplicate query keys')
    if any(set(r['policies']) != set(policies) for r in rows):
        raise ValueError('Ledger policies differ from score report')
    matched = [r for r in rows if r['gold_match_status'] == 'matched']
    if len(matched) != report['input_matched_queries']:
        raise ValueError('Matched ledger denominator differs from score report')
    by_task = {}
    for row in rows:
        by_task.setdefault(row['task_id'], []).append(row)
    fully_matched = {key: values for key, values in by_task.items()
                     if all(r['gold_match_status'] == 'matched' for r in values)}
    candidate_pools, candidate_ids = {}, []
    query_keys = {r['query_key'] for r in rows}
    for spec in args.candidate_pool:
        name, sep, filename = spec.partition('=')
        if not sep or name not in policies or name in candidate_pools:
            raise ValueError('Candidate pool must be a unique known POLICY=PATH')
        raw, ident = identity(Path(filename))
        data = json.loads(raw)
        data = data[name] if name in data and isinstance(data[name], dict) else data
        if not isinstance(data, dict) or set(data) - query_keys \
                or any(not isinstance(value, list) for value in data.values()):
            raise ValueError('Candidate pool must map cohort query keys to ranked rows')
        candidate_pools[name] = data
        candidate_ids.append(dict(ident, policy=name))
    output = {
        'cohort': report.get('cohort'), 'baseline': args.baseline,
        'planned_tasks': len(by_task), 'planned_queries': len(rows),
        'input_matched_queries': len(matched),
        'unresolved_queries': len(rows) - len(matched),
        'fully_input_matched_tasks': len(fully_matched),
        'input_match_status_counts': dict(Counter(r['gold_match_status'] for r in rows)),
        'sources': {'score_report': report_id, 'query_ledger': ledger_id,
                    'candidate_pools': candidate_ids},
        'policies': {}, 'comparisons': {},
        'interpretation': 'Public input-matched diagnostic results, not a Kaggle leaderboard score.',
        'limitations': [
            'Matched queries form the exact-match denominator; unresolved inputs remain separately counted.',
            'The [[0]] flag is a grid-value heuristic, not proof of placeholder provenance.',
            'A ranked source candidate pool is needed to distinguish generated slots from filler grids.',
            'This summary reads scored booleans and hashes; no official query answer grids are read.',
        ],
    }
    for name in policies:
        scored = [r['policies'][name] for r in matched]
        all_records = [r['policies'][name] for r in rows]
        exact = sum(bool(r['exact']) for r in scored)
        if exact != report['policies'][name]['exact_queries']:
            raise ValueError('Ledger exact count differs from score report: ' + name)
        task_exact = sum(all(r['policies'][name]['exact'] for r in values)
                         for values in fully_matched.values())
        result = {
            'exact_queries': exact, 'matched_query_denominator': len(matched),
            'query_micro_em2': exact / len(matched) if matched else None,
            'all_queries_exact_tasks': task_exact,
            'fully_matched_task_denominator': len(fully_matched),
            'full_task_exact_rate': task_exact / len(fully_matched) if fully_matched else None,
            'prediction_status_counts': dict(Counter(r['prediction_status'] for r in all_records)),
            'prediction_present_queries': sum(bool(r['prediction_present']) for r in all_records),
            'valid_queries': sum(r['prediction_status'] == 'valid' for r in all_records),
            'distinct_valid_two_slots_queries': sum(bool(r['distinct_valid_attempts']) for r in all_records),
            'both_slots_equal_zero_1x1_queries': sum(bool(r.get('both_placeholder')) for r in all_records),
            'exact_queries_with_both_slots_zero_1x1': sum(
                bool(r.get('both_placeholder')) and bool(r['exact']) for r in scored),
            'slot_1_exact_queries': sum(bool(r['attempt_1_exact']) for r in scored),
            'slot_2_exclusive_exact_queries': sum(
                bool(r['attempt_2_exact']) and not r['attempt_1_exact'] for r in scored),
        }
        if name in candidate_pools:
            pool = candidate_pools[name]
            result['source_candidate_coverage'] = {
                'queries_with_at_least_one_ranked_candidate': sum(bool(pool.get(key)) for key in query_keys),
                'queries_with_at_least_two_ranked_candidates': sum(len(pool.get(key, [])) >= 2 for key in query_keys),
                'queries_without_source_candidates': sum(not pool.get(key) for key in query_keys),
                'note': 'Assumes supplied rows are the final unique grid ranking used by the policy; no deduplication or rescoring is performed.',
            }
        output['policies'][name] = result
    for base in policies:
        for variant in policies:
            if base == variant:
                continue
            gained, lost = [], []
            for row in matched:
                a, b = row['policies'][base], row['policies'][variant]
                if b['exact'] and not a['exact']:
                    gained.append(row)
                elif a['exact'] and not b['exact']:
                    lost.append(row)
            output['comparisons'][variant + '_vs_' + base] = {
                'new_queries': len(gained), 'lost_queries': len(lost),
                'net_queries': len(gained) - len(lost),
                'micro_em2_delta': (len(gained) - len(lost)) / len(matched) if matched else None,
                'new_query_keys': [r['query_key'] for r in gained],
                'lost_query_keys': [r['query_key'] for r in lost],
                'new_task_ids': sorted({r['task_id'] for r in gained}),
                'lost_task_ids': sorted({r['task_id'] for r in lost}),
                'baseline_slot_2_exclusive_lost_query_keys': [
                    r['query_key'] for r in lost
                    if r['policies'][base]['attempt_2_exact']
                    and not r['policies'][base]['attempt_1_exact']],
            }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open('x') as stream:
        json.dump(output, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')
    print(json.dumps({'out': str(args.out), 'cohort': output['cohort'],
                      'matched_queries': len(matched),
                      'policies': {name: {'exact': v['exact_queries'], 'micro_em2': v['query_micro_em2']}
                                   for name, v in output['policies'].items()}}, ensure_ascii=False))


if __name__ == '__main__':
    main()
