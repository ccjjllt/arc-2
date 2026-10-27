#!/usr/bin/env python3
"""Manually score the frozen031 pair against an explicit local Kaggle snapshot.

Nothing is downloaded, trained, ranked or submitted. Gold is opened only after
--allow-gold-read and verification of every artifact in the no-gold freeze.
The decision is evidence for manual review of the exact V1, never a score promise.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import importlib.util
import json
import math
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
SCORE_PATH = HERE.parent/'020/score_frozen_outputs.py'
spec = importlib.util.spec_from_file_location('score020_for031', SCORE_PATH)
score = importlib.util.module_from_spec(spec)
spec.loader.exec_module(score)
POLICIES = ('base_B4', 'challenger_B4')
PRESSURE_QUERY = '2ba387bc_0'


def read_lines(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def finite(value):
    return type(value) in (int, float) and math.isfinite(value)


def runtime_audit(state, runtime, events, task_ids, query_keys):
    """Inspect observed deployment clocks and TTT checkpoints without query gold."""
    cfg = state['config']
    configuration_ok = (state.get('is_competition_rerun') is False
        and state.get('python_hash_seed') == 27031
        and cfg.get('arm_slot_seconds') == 1620
        and cfg.get('ttt_steps') == dict(base=128, challenger=64)
        and cfg.get('original_scheduler_steps') == 128
        and cfg.get('dfs_threshold') == 0.2
        and cfg.get('selection') == 'original_B4'
        and cfg.get('task_soft_limit_seconds') == 1200
        and cfg.get('dfs_limit_seconds') == 540
        and cfg.get('ttt_augmentation_seed') == 1
        and cfg.get('search_augmentation_seed') == 2)
    evidence = {}
    for arm, stop_step in [('base', 128), ('challenger', 64)]:
        rows = [row for row in events if row['arm'] == arm]
        if any(row.get('branch') != arm or row.get('ttt_limit') != stop_step
               or row.get('task_id') not in task_ids for row in rows):
            raise ValueError('Frozen event branch/task/TTT limit differs: '+arm)
        by_task = defaultdict(list)
        starts, ends = {key: 0 for key in query_keys}, {key: 0 for key in query_keys}
        for row in rows:
            by_task[row['task_id']].append(row)
            if row.get('event') in {'search_start', 'search_end'}:
                keys = row.get('query_keys') or list(dict.fromkeys(
                    subkey.split('.')[0] for subkey in row.get('subkeys', [])))
                for key in keys:
                    if key not in query_keys:
                        raise ValueError('Search event outside fixed query denominator')
                    (starts if row['event'] == 'search_start' else ends)[key] += 1
        if (starts != runtime['arms'][arm]['runtime_events']['started_batches_by_query']
                or ends != runtime['arms'][arm]['runtime_events']['completed_batches_by_query']):
            raise ValueError('Frozen runtime coverage summary differs from raw events')
        task_evidence, bad = {}, []
        for task_id, task_rows in sorted(by_task.items()):
            schedules = [r for r in task_rows if r['event'] == 'ttt_schedule']
            completed = [r for r in task_rows if r['event'] == 'ttt_end']
            checkpoints = [r for r in task_rows if r['event'] == 'ttt_checkpoint']
            began = any(r['event'] == 'ttt_start' for r in task_rows)
            schedule_ok = len(schedules) == 1 and all(
                r.get('max_steps') == 128 and r.get('stop_after_steps') == stop_step
                and r.get('num_train_epochs') == 1.0
                and r.get('warmup_ratio') == 0.1
                and r.get('learning_rate') == 5e-5
                and 'cosine' in str(r.get('lr_scheduler_type')).lower() for r in schedules)
            checkpoints_ok = all(r.get('max_steps') == 128
                and r.get('global_step') in ({64, 128} if arm == 'base' else {64})
                and r.get('stop_requested') is (arm == 'challenger')
                and isinstance(r.get('learning_rates'), list) and bool(r['learning_rates'])
                and all(finite(v) and 0 <= v <= 5e-5 for v in r['learning_rates'])
                for r in checkpoints)
            if schedules and not schedule_ok or not checkpoints_ok:
                bad.append(task_id)
            completed_ok = None
            if completed:
                required = [64, 128] if arm == 'base' else [64]
                completed_ok = (len(completed) == 1 and completed[0].get('global_step') == stop_step
                    and finite(completed[0].get('training_loss')) and schedule_ok and checkpoints_ok
                    and [r['global_step'] for r in checkpoints] == required
                    and all(v > 0 for r in checkpoints if r['global_step'] == 64 for v in r['learning_rates'])
                    and all(abs(v) <= 1e-12 for r in checkpoints if r['global_step'] == 128 for v in r['learning_rates']))
                if not completed_ok:
                    bad.append(task_id)
            task_evidence[task_id] = dict(ttt_started=began, ttt_completed=bool(completed),
                schedule=schedules, checkpoints=checkpoints,
                final_global_steps=[r.get('global_step') for r in completed],
                completed_training_path_verified=completed_ok)
        completed_tasks = [key for key, row in task_evidence.items() if row['ttt_completed']]
        terminal, final = state[arm], state['arms'][arm]
        setup = state['common_setup_seconds']
        timing_ok = (finite(setup) and 0 <= setup <= 600
            and math.isclose(terminal['shared_setup_seconds_charged'], setup, abs_tol=1e-4)
            and math.isclose(terminal['global_end_time']-terminal['started_at'], 1620-setup-120, abs_tol=1e-4)
            and math.isclose(terminal['hard_deadline']-terminal['started_at'], 1620-setup-60, abs_tol=1e-4)
            and final.get('deployment_budget_exceeded') is False
            and finite(final.get('accounted_deployment_elapsed_seconds'))
            and final['accounted_deployment_elapsed_seconds'] <= 1620)
        evidence[arm] = dict(searched_queries=sum(v > 0 for v in starts.values()),
            completed_search_batches=sum(ends.values()), queries_with_four_completed_batches=sum(v == 4 for v in ends.values()),
            started_batches_by_query=starts, completed_batches_by_query=ends,
            ttt_completed_tasks=completed_tasks,
            ttt_unclosed_tasks=[key for key, row in task_evidence.items() if row['ttt_started'] and not row['ttt_completed']],
            tasks_not_started=sorted(set(task_ids)-set(by_task)),
            task_training_evidence=task_evidence, invalid_training_evidence_tasks=sorted(set(bad)),
            observed_training_paths_verified=bool(completed_tasks) and not bad,
            deployment_budget_verified=timing_ok,
            accounted_deployment_elapsed_seconds=final.get('accounted_deployment_elapsed_seconds'))
    return dict(configuration_matches_frozen031=configuration_ok, arms=evidence,
        training_and_budget_verified=configuration_ok and all(
            row['observed_training_paths_verified'] and row['deployment_budget_verified']
            for row in evidence.values()),
        interpretation='Unclosed TTT after a declared budget cutoff is reported, not assigned zero cost or removed from the query denominator. Independent TTT does not establish identical numerical trajectories.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--frozen', required=True, type=Path)
    parser.add_argument('--snapshot-manifest', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--allow-gold-read', action='store_true')
    args = parser.parse_args()
    if not args.allow_gold_read:
        parser.error('Use --allow-gold-read only after both predictions, candidate pools and runtime evidence are frozen')
    frozen, freeze_id = score.read_json(args.frozen/'prediction-freeze.json')
    if frozen.get('gold_read') is not False or frozen.get('B4_reproduced_for_both_arms') is not True:
        raise ValueError('Missing completed no-gold original B4 freeze')
    for name, expected in frozen['outputs'].items():
        raw = (args.frozen/name).read_bytes()
        if len(raw) != expected['bytes'] or score.sha(raw) != expected['sha256']:
            raise ValueError('Frozen artifact changed: '+name)
    cfg = SimpleNamespace(manifest=args.frozen/'input/manifest.json',
        challenges=args.frozen/'input/challenges.json',
        prediction=[name+'='+str(args.frozen/('submission-'+name+'.json')) for name in POLICIES],
        candidate_pool=[args.frozen/'candidate-pools-scored.json'])
    manifest, challenges, policies, _, identities = score.load_inputs(cfg)
    if (frozen['n_tasks'], frozen['n_queries'], manifest['n_tasks'], manifest['n_queries']) != (8, 12, 8, 12):
        raise ValueError('031 requires the frozen8-task/12-query denominator')
    ledger_rows = read_lines(args.frozen/'candidate-ledger.jsonl')
    ledger = {(r['arm']+'_B4', r['query_key']): r for r in ledger_rows}
    query_keys = {q['query_key'] for e in manifest['entries'] for q in e['queries']}
    if len(ledger) != len(ledger_rows) or set(ledger) != {(p, q) for p in POLICIES for q in query_keys}:
        raise ValueError('Candidate ledger does not cover each arm/query exactly once')
    runtime, runtime_id = score.read_json(args.frozen/'runtime-summary.json')
    state, state_id = score.read_json(args.frozen/'run-state.json')
    observed = runtime_audit(state, runtime, read_lines(args.frozen/'worker-events.jsonl'), set(challenges), query_keys)
    ready, ready_id = score.read_json(HERE/'ready-v1.json')
    provenance, provenance_id = score.read_json(args.snapshot_manifest)
    if provenance.get('competition') != 'arc-prize-2026-arc-agi-2':
        raise ValueError('Snapshot is not labelled as the target competition')
    names = ('arc-agi_evaluation_challenges.json', 'arc-agi_evaluation_solutions.json')
    for name in names:
        if not Path(provenance['files'][name]['path']).is_file():
            raise FileNotFoundError('Explicit local official snapshot unavailable: '+name)
    # No query gold is opened above this point.
    data = {}
    for name in names:
        record = provenance['files'][name]
        raw = Path(record['path']).read_bytes()
        if len(raw) != record['bytes'] or score.sha(raw) != record['sha256']:
            raise ValueError('Official snapshot identity differs: '+name)
        data[name] = json.loads(raw)
    official_inputs, solutions = (data[name] for name in names)
    rows, sources = [], []
    for entry in manifest['entries']:
        task_id = entry['task_id']
        task = official_inputs[task_id]
        if task != challenges[task_id] or len(solutions[task_id]) != len(task['test']):
            raise ValueError('Full official input or denominator differs: '+task_id)
        official = dict(train=task['train'], test=[dict(input=q['input'], output=g)
            for q, g in zip(task['test'], solutions[task_id])])
        sources.append(dict(task_id=task_id, source_kind='Exact Kaggle challenge/solution JSON pair', full_task_content_matches=True))
        for query in entry['queries']:
            idx, key = query['query_index'], query['query_key']
            gold, status, matched = score.resolve_gold(challenges[task_id], official, idx)
            if status != 'matched':
                raise ValueError('Official gold cannot be uniquely input-matched: '+key+' '+status)
            gold_sha = score.fingerprint(gold)
            row = dict(task_id=task_id, query_index=idx, query_key=key,
                source_group=entry.get('source_group'), tier=entry.get('tier'), gold_match_status=status,
                official_matching_indices=matched, input_sha256=query['input_sha256'], gold_sha256=gold_sha, policies={})
            for name in POLICIES:
                result = score.score_prediction(policies[name], task_id, idx, gold)
                candidate = ledger[(name, key)]
                oracle = gold_sha in candidate['scored_candidate_sha256']
                result.update(raw_candidate_oracle_exact=gold_sha in candidate['raw_candidate_sha256'],
                    candidate_oracle_exact=oracle, correct_candidate_not_selected=oracle and not result['exact'],
                    second_slot_only_exact=result['attempt_2_exact'] and not result['attempt_1_exact'])
                row['policies'][name] = result
            rows.append(row)
    if len(rows) != 12 or {r['query_key'] for r in rows} != query_keys:
        raise ValueError('Not every fixed query was scored')
    report = score.summarize(rows, policies)
    gains = [r for r in rows if r['policies']['challenger_B4']['exact'] and not r['policies']['base_B4']['exact']]
    losses = [r for r in rows if r['policies']['base_B4']['exact'] and not r['policies']['challenger_B4']['exact']]
    net = len(gains)-len(losses)
    for name in POLICIES:
        report['policies'][name].update(
            raw_oracle_queries=sum(r['policies'][name]['raw_candidate_oracle_exact'] for r in rows),
            scored_oracle_queries=sum(r['policies'][name]['candidate_oracle_exact'] for r in rows),
            first_slot_exact_queries=sum(r['policies'][name]['attempt_1_exact'] for r in rows),
            second_slot_only_exact_queries=sum(r['policies'][name]['second_slot_only_exact'] for r in rows))
    pressure = next(r for r in rows if r['query_key'] == PRESSURE_QUERY)
    a, b = observed['arms']['base'], observed['arms']['challenger']
    criteria = dict(all_frozen_queries_input_matched=True, net_queries_strictly_positive=net > 0,
        base_pressure_second_slot_only_reproduced=pressure['policies']['base_B4']['second_slot_only_exact'],
        challenger_pressure_still_correct=pressure['policies']['challenger_B4']['exact'],
        challenger_searched_queries_at_least_base=b['searched_queries'] >= a['searched_queries'],
        challenger_completed_batches_at_least_base=b['completed_search_batches'] >= a['completed_search_batches'],
        observed_training_and_budget_verified=observed['training_and_budget_verified'])
    passed = all(criteria.values())
    decision = 'eligible_for_exact_v1_formal_review' if passed else 'stop_fixed64_recipe'
    report.update(scored_at_utc=datetime.now(timezone.utc).isoformat(), cohort=manifest['cohort'],
        planned_tasks=8, fixed_query_denominator=12, frozen_input_identity=identities,
        original_prediction_freeze=freeze_id, runtime_summary_identity=runtime_id, run_state_identity=state_id,
        preregistered_ready_record=ready_id, preregistered_criteria=ready['criteria'],
        official_snapshot_manifest=provenance_id, official_sources=sources,
        paired=dict(gained_queries=[r['query_key'] for r in gains], lost_queries=[r['query_key'] for r in losses],
            net_queries=net, gain_task_ids=sorted({r['task_id'] for r in gains}),
            loss_task_ids=sorted({r['task_id'] for r in losses}),
            lost_base_second_slot_only_queries=[r['query_key'] for r in losses if r['policies']['base_B4']['second_slot_only_exact']]),
        pressure_query=pressure, runtime_evidence=runtime, observed_runtime_audit=observed,
        promotion_criteria=criteria, decision=decision,
        failed_criteria=[name for name, value in criteria.items() if not value],
        mandatory_manual_review=dict(exact_V1_source_and_actual_script_version_id='required_before_submission',
            checkpoint_bytes_and_actual_GPU='required_before_submission',
            ordinary_top_level_equals_challenger_B4='required_before_submission',
            no_runtime_error_and_declared_budget_cutoffs_only='required_before_submission',
            daily_submission_allowance_and_pending='required_before_submission'),
        scoring_helper_sha256=score.sha(SCORE_PATH.read_bytes()),
        interpretation=('Exposed public exploratory regression, not independent blind confirmation or hidden leaderboard score. '
            'Independent TTT numerical trajectories can differ. A positive result permits manual review of this exact V1 only; '
            'it cannot guarantee public-score improvement. Invalid runtime evidence is an execution issue to review, not proof of algorithm failure. '
            'No automatic submission or alternate step/LR/seed/ranking selection is performed.'))
    args.output.mkdir(parents=True, exist_ok=False)
    with (args.output/'query-ledger.jsonl').open('x') as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False)+'\n')
    score.write_json(args.output/'score-report.json', report)
    print(json.dumps(dict(cohort=manifest['cohort'], policies=report['policies'], paired=report['paired'],
        promotion_criteria=criteria, decision=decision), ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
