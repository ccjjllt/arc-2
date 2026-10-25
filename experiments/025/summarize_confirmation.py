#!/usr/bin/env python3
"""Summarize one frozen B4/R2 run using existing scoring hashes only.

No answers are fetched, predictions selected, models called, or jobs launched.
Example: python3 experiments/025/summarize_confirmation.py --replay-dir REPLAY \
  --score-dir SCORING --run-dir DOWNLOAD/025 --output NEW_REPORT.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from pathlib import Path


def read(path, identities, lines=False, optional=False):
    if optional and not path.is_file():
        return None
    raw = path.read_bytes()
    identities[str(path.resolve())] = dict(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    return [json.loads(s) for s in raw.decode().splitlines() if s.strip()] if lines else json.loads(raw)


def indexed(rows):
    result = {row['query_key']: row for row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate query keys in a ledger')
    return result


def event_queries(event):
    if event.get('query_key'):
        return [event['query_key']]
    return event.get('query_keys', list(dict.fromkeys(s.split('.')[0] for s in event.get('subkeys', []))))


def phase_costs(events, solver_end):
    phases = {}
    for phase in ('task', 'ttt', 'search', 'score'):
        opened, completed, unmatched_ends = defaultdict(deque), [], []
        for event in events:
            kind = event.get('event')
            if kind not in (phase + '_start', phase + '_end'):
                continue
            key = (event.get('rank'), event.get('task_id'))
            if phase == 'search':
                key += (tuple(event.get('subkeys', [])),)
            elif phase == 'score':
                key += (event.get('query_key'), event.get('subkey'))
            if kind.endswith('_start'):
                opened[key].append(event)
            elif opened[key]:
                start = opened[key].popleft()
                duration = event['timestamp'] - start['timestamp']
                if duration < 0:
                    raise ValueError('Negative observed phase duration')
                completed.append(dict(rank=key[0], task_id=key[1], seconds=duration))
            else:
                unmatched_ends.append(event)
        pending = []
        for queue in opened.values():
            for event in queue:
                item = dict(event)
                item['elapsed_to_solver_end_s'] = max(0, solver_end - event['timestamp']) if isinstance(solver_end, (float, int)) else None
                pending.append(item)
        totals = defaultdict(float)
        for item in completed:
            totals[str(item['rank'])] += item['seconds']
        phases[phase] = dict(completed_count=len(completed), completed_rank_seconds=sum(totals.values()),
                             completed_seconds_by_rank=dict(totals), unmatched_starts=pending,
                             unmatched_ends=unmatched_ends)
    return dict(phases=phases, note='Rank-seconds sum parallel workers, not wall time. Phases nest: do not add task, TTT, search and score totals. Unmatched intervals are censored observations, not measured compute durations.')


def coverage(row, candidate, events_available, query_events):
    starts = sum(e.get('event') == 'search_start' for e in query_events)
    returns = sum(e.get('event') == 'search_end' for e in query_events)
    saved = sum(e.get('event') == 'raw_batch_saved' for e in query_events)
    raw, scored = candidate.get('raw_unique_candidates'), candidate['scored_unique_candidates']
    matched = row['gold_match_status'] == 'matched'
    gold = row.get('gold_sha256') if matched else None
    raw_hashes, scored_hashes = candidate.get('raw_candidate_sha256'), candidate['scored_candidate_sha256']
    raw_correct = gold in raw_hashes if gold and raw_hashes is not None else None
    scored_correct = gold in scored_hashes if gold else None
    if scored:
        reason = ('gold_unresolved' if not matched else 'scored_candidates_without_correct' if not scored_correct
                  else 'correct_scored_candidate_not_selected' if not row['policies']['B4']['exact'] else 'B4_correct')
    elif raw:
        reason = 'raw_candidates_no_scored_export'
    elif not events_available or raw is None:
        reason = 'coverage_unconfirmed'
    elif not starts:
        reason = 'no_search_start_recorded'
    elif starts != returns or returns != saved:
        reason = 'search_or_raw_export_incomplete'
    else:
        reason = 'searched_no_legal_raw_in_returned_batches'
    return dict(reason=reason, search_starts=starts, search_returns=returns, raw_batch_exports=saved,
                raw_unique_candidates=raw, scored_unique_candidates=scored,
                raw_oracle_exact=raw_correct, scored_oracle_exact=scored_correct,
                raw_correct_absent_from_scored_export=raw_correct is True and scored_correct is False,
                correct_not_selected={name: scored_correct and not row['policies'][name]['exact'] if matched else None for name in ('B4', 'R2')})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('replay-dir', 'score-dir', 'run-dir', 'output'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--package-manifest', type=Path)
    parser.add_argument('--execution-outcome', type=Path)
    args = parser.parse_args()
    identities = {}
    freeze = read(args.replay_dir / 'prediction-freeze.json', identities)
    candidates = indexed(read(args.replay_dir / 'query-candidate-ledger.jsonl', identities, lines=True))
    score = read(args.score_dir / 'score-report.json', identities)
    score_freeze = read(args.score_dir / 'prediction-freeze.json', identities)
    rows = read(args.score_dir / 'query-ledger.jsonl', identities, lines=True)
    by_key = indexed(rows)
    if set(by_key) != set(candidates) or len(rows) != freeze['n_queries'] or len(rows) != score['planned_queries']:
        raise ValueError('Frozen and scored query denominators differ')
    if freeze['cohort'] != score['cohort'] or score_freeze['planned_queries'] != len(rows):
        raise ValueError('Frozen and scored cohort identities differ')
    for name in ('B4', 'R2'):
        expected = freeze['outputs']['submission-' + name + '.json']['sha256']
        used = [item['sha256'] for item in score_freeze['artifacts']['predictions'] if item['policy'] == name]
        if used != [expected]:
            raise ValueError('Scoring did not use the frozen ' + name + ' prediction file')
    for row in rows:
        if row['input_sha256'] != candidates[row['query_key']]['input_sha256']:
            raise ValueError('Candidate and scoring query input hashes differ')
    state = read(args.run_dir / 'run-state.json', identities, optional=True)
    boot = read(args.run_dir / 'boot.json', identities, optional=True)
    artifacts = read(args.run_dir / 'artifact-identities.json', identities, optional=True)
    package = read(args.package_manifest, identities) if args.package_manifest else None
    outcome = read(args.execution_outcome, identities) if args.execution_outcome else None
    event_files = sorted(args.run_dir.glob('events-rank*.jsonl'))
    events = []
    for path in event_files:
        events.extend(read(path, identities, lines=True))
    events.sort(key=lambda e: e['timestamp'])
    expected_event_files = {'events-rank%d.jsonl' % rank for rank in range(4)}
    complete_event_files = {p.name for p in event_files} == expected_event_files
    per_query_events = defaultdict(list)
    for event in events:
        for key in event_queries(event):
            if key not in by_key:
                raise ValueError('Event query is outside frozen cohort')
            per_query_events[key].append(event)
    details = []
    for row in rows:
        key = row['query_key']
        result = coverage(row, candidates[key], complete_event_files, per_query_events[key])
        details.append(dict(query_key=key, task_id=row['task_id'], gold_match_status=row['gold_match_status'],
                            B4_exact=row['policies']['B4'].get('exact'), R2_exact=row['policies']['R2'].get('exact'), **result))
    matched = [row for row in details if row['gold_match_status'] == 'matched']
    grouped = defaultdict(list)
    for row in details:
        grouped[row['task_id']].append(row)
    full_tasks = [items for items in grouped.values() if all(x['gold_match_status'] == 'matched' for x in items)]
    policies = {}
    for name in ('B4', 'R2'):
        exact = sum(x[name + '_exact'] is True for x in matched)
        if exact != score['policies'][name]['exact_queries']:
            raise ValueError('Scoring report and per-query exact totals differ')
        policies[name] = dict(exact_queries=exact, matched_denominator=len(matched),
                              exact_fraction=exact / len(matched) if matched else None,
                              complete_tasks=sum(all(x[name + '_exact'] is True for x in task) for task in full_tasks),
                              complete_task_denominator=len(full_tasks),
                              correct_scored_candidates_not_selected=[x['query_key'] for x in matched if x['correct_not_selected'][name]])
    gained = [x['query_key'] for x in matched if x['R2_exact'] and not x['B4_exact']]
    lost = [x['query_key'] for x in matched if x['B4_exact'] and not x['R2_exact']]
    oracles = {}
    for kind in ('raw', 'scored'):
        known = [x for x in matched if x[kind + '_oracle_exact'] is not None]
        oracles[kind] = dict(exact_queries=sum(x[kind + '_oracle_exact'] is True for x in known),
                             known_matched_denominator=len(known), unconfirmed_matched_queries=len(matched) - len(known))
    report = dict(created_at_utc=datetime.now(timezone.utc).isoformat(), cohort=freeze['cohort'],
                  planned_tasks=freeze['n_tasks'], planned_queries=len(rows), input_matched_queries=len(matched),
                  unresolved_queries=[x['query_key'] for x in details if x['gold_match_status'] != 'matched'],
                  policies=policies, R2_vs_B4=dict(gained_queries=gained, lost_queries=lost, net_queries=len(gained)-len(lost)),
                  candidate_oracles=oracles,
                  raw_correct_absent_from_scored_export=[x['query_key'] for x in matched if x['raw_correct_absent_from_scored_export']],
                  coverage_counts=dict(Counter(x['reason'] for x in details)), queries=details,
                  events=dict(files_complete=complete_event_files, count=len(events),
                              **phase_costs(events, state.get('solver_finished_at') if state else None)),
                  run_state=state, actual_hardware=boot.get('gpu_inventory') if boot else None,
                  runtime_boot=boot, package_manifest=package, execution_outcome=outcome,
                  source_artifact_identities=artifacts, scoring_input_identities=score_freeze['artifacts'],
                  analysis_input_identities=identities,
                  limitations=['Public input-matched diagnosis, not a Kaggle score or unseen-task validation.',
                               'Raw oracle covers supplied returned-batch exports; interrupted internal DFS candidates are unobserved.',
                               'Coverage categories describe persisted evidence, not a proven causal failure mechanism.',
                               'Missing run state, hardware, version or source identity remains null/unconfirmed.'])
    if state and 'elapsed_s' in state:
        report['observed_notebook_wall_seconds'] = state['elapsed_s']
    with args.output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write('\n')
    print(json.dumps({k: report[k] for k in ('planned_tasks', 'planned_queries', 'input_matched_queries', 'policies', 'R2_vs_B4', 'candidate_oracles', 'coverage_counts')}, ensure_ascii=False))


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError, TypeError) as error:
        raise SystemExit('Summary stopped: %s: %s' % (type(error).__name__, error))
