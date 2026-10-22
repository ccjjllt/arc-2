"""Manually pair frozen E20 outputs; no inference, training or answer download.

Policies declared before the 4B result is available:
* FillMissing: add a different real 27B grid only where 4B has <2 candidates.
* ReplaceSecond: keep 4B slot1; use a different real 27B grid in slot2.
Both rules use prediction/source information only, never correctness or task IDs.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT = ROOT / 'experiments/021'
PRIMARY_REV = 'f3283f727488ad98fe575ea6a5ac981e4a188e49'
SUPPLEMENT_REV = 'bebec789c46f4be8f690a321ad852ecdb1f3f362'
POLICIES = {
    'B4': 'Unchanged frozen 4B final two slots.',
    'Q27': 'Unchanged frozen 27B Direct1 output; M0/M1 are duplicates.',
    'FillMissing': 'If 4B has fewer than two real unique candidates, preserve its existing first real candidate and append a different real 27B grid. With no 4B candidate, use the real 27B grid in both slots. Otherwise retain B4.',
    'ReplaceSecond': 'For every query, preserve B4 attempt_1 and put a real 27B grid in attempt_2 only if it differs from B4 attempt_1; otherwise retain B4.',
}


def read(path):
    raw = path.read_bytes()
    return json.loads(raw), {'path': str(path.resolve()),
                            'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}


def write(path, value):
    with path.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, allow_nan=False, indent=2)
        stream.write('\n')


def load_module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--b4-submission', type=Path, required=True)
    ap.add_argument('--b4-candidate-pool', type=Path, required=True)
    ap.add_argument('--b4-pool-policy', default='B0')
    ap.add_argument('--b4-raw-candidate-pool', type=Path,
                    help='Optional original legal inverse-transformed DFS grids; independent of scored ranking.')
    ap.add_argument('--b4-raw-pool-policy', default='B0')
    ap.add_argument('--b4-wall-seconds', type=float)
    ap.add_argument('--b4-cost-note', default='4B end-to-end measured cost not supplied.')
    ap.add_argument('--q27-submission', type=Path, default=DEFAULT / 'qwen-e20-output-v1/submission.json')
    ap.add_argument('--q27-candidate-pool', type=Path, default=DEFAULT / 'qwen-e20-output-v1/vprobe/candidate-pools.json')
    ap.add_argument('--q27-pool-policy', default='M1')
    ap.add_argument('--q27-run', type=Path, default=DEFAULT / 'qwen-e20-output-v1/vprobe/run.json')
    ap.add_argument('--manifest', type=Path, default=ROOT / 'experiments/020/evaluation-e20/manifest.json')
    ap.add_argument('--challenges', type=Path, default=ROOT / 'experiments/020/evaluation-e20/challenges.json')
    ap.add_argument('--primary-ledger', type=Path, default=DEFAULT / 'scoring-qwen-e20-v1/query-ledger.jsonl')
    ap.add_argument('--version-supplement', type=Path, default=DEFAULT / 'execution/qwen-e20-version-supplement-v1.json')
    ap.add_argument('--output-dir', type=Path, required=True)
    args = ap.parse_args()
    scorer_path = ROOT / 'experiments/020/score_frozen_outputs.py'
    scorer = load_module(scorer_path, 'arc020score')
    args.prediction = ['B4=' + str(args.b4_submission), 'Q27=' + str(args.q27_submission)]
    args.candidate_pool = []
    manifest, challenges, policies, _, inputs = scorer.load_inputs(args)
    entries = {q['query_key']: (entry, q) for entry in manifest['entries'] for q in entry['queries']}
    expected = set(entries)

    def pool(path, policy, allow_grid_rows=False):
        value, ident = read(path)
        if isinstance(value, dict) and policy in value and isinstance(value[policy], dict):
            value = value[policy]
        if not isinstance(value, dict) or set(value) - expected:
            raise ValueError('Candidate pool is not a direct cohort map or named policy map: ' + str(path))
        result = {}
        for key in expected:
            rows = value.get(key, [])
            if not isinstance(rows, list):
                raise ValueError('Candidate pool rows must be a list')
            seen, grids = set(), []
            for row in rows:
                grid = row.get('grid') if isinstance(row, dict) else row if allow_grid_rows else None
                if not scorer.grid_is_valid(grid):
                    raise ValueError('Invalid source candidate grid: ' + key)
                h = scorer.fingerprint(grid)
                if h not in seen:
                    seen.add(h)
                    grids.append(grid)
            result[key] = grids
        return result, ident

    b4pool, b4pool_id = pool(args.b4_candidate_pool, args.b4_pool_policy)
    rawpool, rawpool_id = (pool(args.b4_raw_candidate_pool, args.b4_raw_pool_policy, True)
                           if args.b4_raw_candidate_pool else (None, None))
    scored_absent_from_raw = ({key: len({scorer.fingerprint(g) for g in b4pool[key]}
                                       - {scorer.fingerprint(g) for g in rawpool[key]})
                              for key in expected
                              if {scorer.fingerprint(g) for g in b4pool[key]}
                              - {scorer.fingerprint(g) for g in rawpool[key]}}
                             if rawpool is not None else None)
    qpool, qpool_id = pool(args.q27_candidate_pool, args.q27_pool_policy)
    qrun, qrun_id = read(args.q27_run)
    if qrun['challenges_sha256'] != inputs['challenges']['sha256']:
        raise ValueError('Q27 runtime used different frozen inputs')
    policies['FillMissing'] = copy.deepcopy(policies['B4'])
    policies['ReplaceSecond'] = copy.deepcopy(policies['B4'])
    routed = {'FillMissing': set(), 'ReplaceSecond': set(challenges)}
    inserted = {'FillMissing': [], 'ReplaceSecond': []}
    for key, (entry, query) in entries.items():
        task_id, idx = entry['task_id'], query['query_index']
        b = policies['B4'].get(task_id, [])[idx]
        q = policies['Q27'].get(task_id, [])[idx]
        if not all(scorer.grid_is_valid(b.get(slot)) for slot in ('attempt_1', 'attempt_2')):
            raise ValueError('4B needs complete valid final slot objects, retaining filler for missing predictions')
        if set(b) != {'attempt_1', 'attempt_2'}:
            raise ValueError('Unexpected 4B final slot object')
        if not b4pool[key] and any(b[slot] != [[0]] for slot in ('attempt_1', 'attempt_2')):
            raise ValueError('4B source pool is missing provenance for non-filler final output: ' + key)
        if len(b4pool[key]) == 1 and b['attempt_2'] not in ([[0]], b['attempt_1']):
            raise ValueError('4B second slot is absent from its source pool: ' + key)
        for index, grid in enumerate(b4pool[key][:2]):
            if b['attempt_' + str(index + 1)] != grid:
                raise ValueError('4B pool must be its final unique ranked grid list: ' + key)
        if set(q) != {'attempt_1', 'attempt_2'} or not all(
                scorer.grid_is_valid(q.get(slot)) for slot in ('attempt_1', 'attempt_2')):
            raise ValueError('Unexpected Q27 final slot object')
        if not qpool[key] and any(q[slot] != [[0]] for slot in ('attempt_1', 'attempt_2')):
            raise ValueError('Q27 source pool is missing provenance for non-filler output: ' + key)
        if len(qpool[key]) > 1 or (qpool[key] and (
                q['attempt_1'] != qpool[key][0] or q['attempt_2'] != qpool[key][0])):
            raise ValueError('Expected one real Q27 Direct1 candidate per generated query')
        qgrid = qpool[key][0] if qpool[key] else None
        if len(b4pool[key]) < 2:
            routed['FillMissing'].add(task_id)
            if qgrid is not None:
                if not b4pool[key]:
                    policies['FillMissing'][task_id][idx] = {'attempt_1': qgrid, 'attempt_2': qgrid}
                    inserted['FillMissing'].append(key)
                elif qgrid != b['attempt_1']:
                    policies['FillMissing'][task_id][idx]['attempt_2'] = qgrid
                    inserted['FillMissing'].append(key)
        if qgrid is not None and qgrid != b['attempt_1']:
            policies['ReplaceSecond'][task_id][idx]['attempt_2'] = qgrid
            inserted['ReplaceSecond'].append(key)

    args.output_dir.mkdir(parents=True, exist_ok=False)
    for name, prediction in policies.items():
        write(args.output_dir / ('submission-' + name + '.json'), prediction)
    write(args.output_dir / 'prediction-freeze.json', {
        'frozen_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'policies': POLICIES, 'inputs': inputs,
        'candidate_pools': {'B4_scored': b4pool_id, 'B4_raw_optional': rawpool_id,
                            'Q27': qpool_id}, 'q27_run': qrun_id,
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'derived_submission_hashes': {name: read(args.output_dir / ('submission-' + name + '.json'))[1]
                                      for name in policies},
        'note': 'All predictions and label-free policy outputs frozen before reading scored reference hashes.',
    })

    ledger_raw = args.primary_ledger.read_bytes()
    prior = [json.loads(line) for line in ledger_raw.decode().splitlines() if line.strip()]
    prior_freeze, prior_freeze_id = read(args.primary_ledger.parent / 'prediction-freeze.json')
    if (prior_freeze['official_revision'] != PRIMARY_REV
            or prior_freeze['official_split'] != 'evaluation'
            or prior_freeze['artifacts']['challenges']['sha256'] != inputs['challenges']['sha256']):
        raise ValueError('Primary scored reference is bound to a different revision or challenge')
    for row in prior:
        if row['query_key'] not in entries or row['input_sha256'] != entries[row['query_key']][1]['input_sha256']:
            raise ValueError('Primary scored reference query input differs from frozen manifest')
    supplement, supplement_id = read(args.version_supplement)
    truth = {r['query_key']: (r['gold_sha256'], PRIMARY_REV)
             for r in prior if r['gold_match_status'] == 'matched'}
    if supplement['source']['revision'] != SUPPLEMENT_REV \
            or not all(supplement['matching_checks'][key] for key in
                       ('all_ordered_demonstrations_equal', 'all_query_inputs_equal')):
        raise ValueError('Supplement does not establish the expected input-matching official version')
    if supplement['frozen_inputs']['challenge_sha256'] != inputs['challenges']['sha256']:
        raise ValueError('Supplement was produced for different inputs')
    for row in supplement['queries']:
        if row['query_key'] in truth:
            raise ValueError('Supplement overlaps primary matched queries')
        truth[row['query_key']] = (row['official_gold_sha256'], SUPPLEMENT_REV)
    if set(truth) != expected or set(r['query_key'] for r in prior) != expected:
        raise ValueError('Reference ledger/supplement does not cover exactly the full frozen cohort')
    rows = []
    for key, (entry, query) in entries.items():
        h, revision = truth[key]
        if len(h) != 64:
            raise ValueError('Missing reference fingerprint')
        row = {'task_id': entry['task_id'], 'query_index': query['query_index'],
               'query_key': key, 'source_group': entry.get('source_group'), 'tier': entry.get('tier'),
               'gold_match_status': 'matched', 'gold_sha256': h, 'official_revision': revision,
               'input_sha256': query['input_sha256'], 'policies': {}}
        for name, prediction in policies.items():
            result = scorer.score_prediction(prediction, entry['task_id'], query['query_index'], None)
            accepted = result['prediction_status'] == 'valid'
            result['attempt_1_exact'] = accepted and result['attempt_sha256'][0] == h
            result['attempt_2_exact'] = accepted and result['attempt_sha256'][1] == h
            result['exact'] = result['attempt_1_exact'] or result['attempt_2_exact']
            row['policies'][name] = result
        b4_oracle = any(scorer.fingerprint(g) == h for g in b4pool[key])
        qreal = any(scorer.fingerprint(g) == h for g in qpool[key])
        raw_oracle = (any(scorer.fingerprint(g) == h for g in rawpool[key])
                      if rawpool is not None else None)
        row['diagnostics'] = {'B4_scored_candidate_oracle_exact': b4_oracle,
                              'Q27_real_candidate_exact': qreal,
                              'scored_candidate_union_oracle_exact': b4_oracle or qreal,
                              'selected_union_oracle_exact': row['policies']['B4']['exact'] or qreal,
                              'B4_scored_unique_candidates': len(b4pool[key]),
                              'B4_raw_candidate_oracle_exact': raw_oracle,
                              'B4_raw_unique_candidates': len(rawpool[key]) if rawpool is not None else None,
                              'raw_correct_absent_from_scored_export': raw_oracle and not b4_oracle if raw_oracle is not None else None,
                              'raw_correct_not_selected': raw_oracle and not row['policies']['B4']['exact'] if raw_oracle is not None else None,
                              'raw_candidate_union_oracle_exact': raw_oracle or qreal if raw_oracle is not None else None}
        rows.append(row)
    with (args.output_dir / 'query-ledger.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    report = scorer.summarize(rows, policies)
    report.update(cohort=manifest['cohort'], planned_tasks=manifest['n_tasks'],
                  official_version_counts={rev: sum(r['official_revision'] == rev for r in rows)
                                           for rev in (PRIMARY_REV, SUPPLEMENT_REV)},
                  interpretation='Input-version-aligned public E20 pairing; mixed official revisions explicitly recorded; no new inference or 12-hour ensemble run.')
    write(args.output_dir / 'score-report.json', report)
    subprocess.run([sys.executable, str(DEFAULT / 'summarize_paired.py'), '--report',
                    str(args.output_dir / 'score-report.json'), '--ledger',
                    str(args.output_dir / 'query-ledger.jsonl'), '--baseline', 'B4', '--out',
                    str(args.output_dir / 'paired-summary.json')], check=True)

    calls = {c['task_id']: c for c in qrun['calls'] if c['arm'] == 'P0'}
    if set(calls) != set(challenges) or len(qrun['calls']) != len(challenges):
        raise ValueError('Expected one existing Q27 P0 call per E20 task')
    cost = {'B4_measured_wall_seconds': args.b4_wall_seconds, 'B4_cost_note': args.b4_cost_note,
            'Q27_existing_full_E20_wall_seconds': qrun['elapsed_s'],
            'Q27_existing_setup_seconds_included': qrun['install_load_s'],
            'Q27_existing_completion_tokens': qrun['new_output_tokens_total'],
            'logical_calls': {name: {'tasks': len(ids), 'task_ids': sorted(ids),
                                     'observed_tokens_for_required_calls': sum(calls[k]['usage']['completion_tokens'] for k in ids),
                                     'sum_observed_request_seconds_not_stage_wall': sum(calls[k]['wall_s'] for k in ids),
                                     'inserted_query_keys': inserted[name]} for name, ids in routed.items()},
            'warning': 'Subset call costs use existing full-E20 observations. They are not a measured routed-stage wall time; batching and setup change costs. End-to-end 12h deployment is unverified.'}
    write(args.output_dir / 'cost-diagnosis.json', cost)
    exact = lambda field: sum(bool(r['diagnostics'][field]) for r in rows)
    unique = {'Q27_unique_correct_vs_B4_final2': [r['query_key'] for r in rows if r['diagnostics']['Q27_real_candidate_exact'] and not r['policies']['B4']['exact']],
              'Q27_unique_correct_vs_B4_scored_candidate_oracle': [r['query_key'] for r in rows if r['diagnostics']['Q27_real_candidate_exact'] and not r['diagnostics']['B4_scored_candidate_oracle_exact']],
              'B4_scored_candidate_oracle_exact': exact('B4_scored_candidate_oracle_exact'),
              'scored_candidate_union_oracle_exact': exact('scored_candidate_union_oracle_exact'),
              'selected_union_oracle_exact': exact('selected_union_oracle_exact'),
              'scored_correct_not_selected_query_keys': [r['query_key'] for r in rows if r['diagnostics']['B4_scored_candidate_oracle_exact'] and not r['policies']['B4']['exact']],
              'raw_pool_status': 'provided' if rawpool is not None else 'not_available_not_evaluated',
              'B4_raw_candidate_oracle_exact': exact('B4_raw_candidate_oracle_exact') if rawpool is not None else None,
              'raw_candidate_union_oracle_exact': exact('raw_candidate_union_oracle_exact') if rawpool is not None else None,
              'raw_correct_absent_from_scored_export_query_keys': [r['query_key'] for r in rows if r['diagnostics']['raw_correct_absent_from_scored_export']] if rawpool is not None else None,
              'raw_correct_not_selected_query_keys': [r['query_key'] for r in rows if r['diagnostics']['raw_correct_not_selected']] if rawpool is not None else None,
              'Q27_unique_correct_vs_B4_raw_candidate_oracle': [r['query_key'] for r in rows if r['diagnostics']['Q27_real_candidate_exact'] and not r['diagnostics']['B4_raw_candidate_oracle_exact']] if rawpool is not None else None,
              'scored_candidates_absent_from_raw_export_counts': scored_absent_from_raw,
              'reference_sources': {'primary_ledger': {'path': str(args.primary_ledger), 'sha256': hashlib.sha256(ledger_raw).hexdigest()}, 'primary_prediction_freeze': prior_freeze_id, 'supplement': supplement_id},
              'warning': 'ScoredCandidateOracle covers only the supplied rescored/ranked export, never all raw DFS generations. Optional RawCandidateOracle covers the supplied legal inverse-transformed raw export; its completeness must be established separately. Raw-correct absent from scored export does not by itself prove no scoring computation occurred. Oracle labels never choose either policy slots; unavailable raw evidence cannot justify a training conclusion.'}
    write(args.output_dir / 'oracle-and-unique.json', unique)
    paired, _ = read(args.output_dir / 'paired-summary.json')
    lines = ['| 政策或诊断 | 正确 query / 27 | 全部query正确 task / 20 | 相对4B 新增 / 丢失 / 净增 | 额外逻辑27B task调用 |',
             '| --- | --- | --- | --- | --- |']
    for name in ('B4', 'Q27', 'FillMissing', 'ReplaceSecond'):
        v = paired['policies'][name]
        delta = paired['comparisons'].get(name + '_vs_B4')
        change = '—' if delta is None else '%d / %d / %+d' % (delta['new_queries'], delta['lost_queries'], delta['net_queries'])
        count = 0 if name == 'B4' else len(challenges) if name == 'Q27' else len(routed[name])
        lines.append('| %s | %d / %d | %d / %d | %s | %d |' % (name, v['exact_queries'], len(rows), v['all_queries_exact_tasks'], manifest['n_tasks'], change, count))
    for label, value in (('4B ScoredCandidateOracle', unique['B4_scored_candidate_oracle_exact']),
                         ('4B已选两槽与27B oracle并集', unique['selected_union_oracle_exact']),
                         ('4B评分候选与27B oracle并集', unique['scored_candidate_union_oracle_exact'])):
        lines.append('| %s | %d / %d | — | 仅诊断上界 | 不代表可提交政策 |' % (label, value, len(rows)))
    if rawpool is not None:
        for label, value in (('4B RawCandidateOracle', unique['B4_raw_candidate_oracle_exact']),
                             ('4B原始导出候选与27B oracle并集', unique['raw_candidate_union_oracle_exact'])):
            lines.append('| %s | %d / %d | — | 仅所给raw导出范围 | 不代表可提交政策 |' % (label, value, len(rows)))
    lines += ['', '所有query保留。25项沿用主官方revision、2项沿用完整输入匹配的旧官方revision；这是公开E20配对诊断。',
              'Q27为同一Direct1结果的重复双槽。两种融合规则已预声明，不按答案选择题或阈值；未新增推理，也未验证12小时全量融合。',
              'ScoredCandidateOracle仅含已评分排名池；RawCandidateOracle只有提供原始合法逆变换grid池时才计算，缺失记未确认，不能据此归因训练不足。',
              '实际完整E20成本和所需逻辑调用见 cost-diagnosis.json；请求时长之和不能当阶段墙钟。', '']
    (args.output_dir / 'paired-table.md').write_text('\n'.join(lines))
    print(json.dumps({'output_dir': str(args.output_dir), 'policies': report['policies'],
                      'unique_vs_B4_final2': unique['Q27_unique_correct_vs_B4_final2']}, ensure_ascii=False))


if __name__ == '__main__':
    main()
