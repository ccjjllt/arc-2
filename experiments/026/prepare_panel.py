#!/usr/bin/env python3
"""Freeze one development panel before training, using archived 025 correctness flags.

No query solution grids are read. This is outcome-stratified development data,
not an unseen holdout. A separate 23-task reservation is never scored here.
"""
import argparse
import copy
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SEED = '026:paired-training:20261025'
COUNTS = {'L_multi': 3, 'L_single': 3, 'M_multi': 2, 'M_single': 3, 'S_multi': 2, 'S_single': 3}


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def read(path, lines=False):
    raw = path.read_bytes()
    value = [json.loads(line) for line in raw.decode().splitlines() if line.strip()] if lines else json.loads(raw)
    return value, dict(path=str(path.resolve()), sha256=sha(raw), bytes=len(raw))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    old, old_ident = read(ROOT / 'experiments/025/cohort-v1/manifest.json')
    source, source_ident = read(ROOT / 'experiments/025/cohort-v1/challenges.json')
    ledger, ledger_ident = read(ROOT / 'experiments/025/scoring-v1-f328/query-ledger.jsonl', True)
    grouped = defaultdict(list)
    for row in ledger:
        if row['gold_match_status'] != 'matched':
            raise ValueError('Cannot assign a historical outcome to unresolved gold')
        grouped[row['task_id']].append(row['policies']['B4']['exact'])
    category = {key: 'all_correct' if all(values) else 'partial' if any(values) else 'all_wrong' for key, values in grouped.items()}
    if Counter(category.values()) != Counter(all_correct=9, partial=3, all_wrong=28):
        raise ValueError('025 historical outcomes changed')
    strata = defaultdict(list)
    for entry in old['entries']:
        strata[entry['tier']+'_'+entry['query_group']].append(entry['task_id'])
    # Dynamic programming enforces both predeclared margins. A seeded hash cost
    # breaks ties without consulting any new model predictions or rule mapping.
    costs = {key: int(sha((SEED+':'+key).encode()), 16) for key in source}
    states = {(0, 0): (0, [])}
    for name, wanted in sorted(COUNTS.items()):
        following = {}
        for combo in itertools.combinations(sorted(strata[name]), wanted):
            full = sum(category[key] == 'all_correct' for key in combo)
            partial = sum(category[key] == 'partial' for key in combo)
            cost = sum(costs[key] for key in combo)
            for (a,b),(before,chosen) in states.items():
                state = (a+full,b+partial)
                if state[0]>4 or state[1]>1:
                    continue
                candidate = (before+cost,sorted(chosen+list(combo)))
                if state not in following or candidate < following[state]:
                    following[state] = candidate
        states = following
    if (4,1) not in states:
        raise ValueError('Predeclared margins are infeasible; do not silently relax')
    chosen = states[(4,1)][1]
    entries = []
    for entry in old['entries']:
        if entry['task_id'] in chosen:
            entry = copy.deepcopy(entry)
            entry['source_group'] = 'public_evaluation_NVARC026_development_panel'
            entry['historical_025_B4_task_outcome'] = category[entry['task_id']]
            entries.append(entry)
    challenges = {key: source[key] for key in chosen}
    if any('output' in query for task in challenges.values() for query in task['test']):
        raise ValueError('Query answer entered challenges')
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output/'challenges.json', challenges)
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort='026_NVARC4B_extra_training_paired16', status='frozen_before_training_not_run',
        selection_seed=SEED, n_tasks=len(entries), n_queries=sum(e['n_test_queries'] for e in entries),
        selection_rule='Select 16 from 025 forty: enforce 4 all-correct / 1 partial / 11 all-wrong tasks and L_multi3/L_single3/M_multi2/M_single3/S_multi2/S_single3; minimize sum of seeded task SHA256 integers across feasible subsets. Uses archived 025 B4 correctness flags only, no query grids, no generator mapping, no new predictions.',
        strata=COUNTS, historical_outcome_counts=dict(Counter(category[key] for key in chosen)),
        entries=entries, sources=[old_ident,source_ident,ledger_ident],
        input_artifact=read(args.output/'challenges.json')[1],
        pythonhashseed=26026, policies=['base_B4','challenger_B4'],
        limitations=['Outcome-stratified development panel; not a representative public-score estimate.',
            'Historical public inference/scoring and upstream exposure; not model-unseen or blind.',
            'New base and challenger are paired in the same run; old 025 score is diagnostic only.'],
        promotion_rule='Only strictly positive final two-slot query net gain advances this one fixed checkpoint to a single 23-task confirmation; zero/negative ends the recipe without panel-based retuning.')
    write(args.output/'manifest.json',manifest)
    full, full_ident = read(ROOT/'experiments/008/raw-eval/arc-agi_evaluation_challenges.json')
    excluded = {item['task_id'] for item in old['exclusions']}
    remaining = sorted(set(full)-excluded-set(source))
    if len(remaining) != 23 or set(remaining)&set(chosen):
        raise ValueError('Remaining confirmation reservation changed')
    write(args.output/'confirmation-reservation.json',dict(task_ids=remaining,n_tasks=23,
        source=full_ident, status='reserved_not_run_or_scored_in_026',
        exposure='Unused by 025 B4/R2 strategy selection, but public historical inference/upstream exposure exists; not blind.'))
    print(json.dumps(dict(task_ids=chosen,n_tasks=len(entries),n_queries=manifest['n_queries'],historical_outcomes=manifest['historical_outcome_counts'],challenges_sha256=manifest['input_artifact']['sha256']),indent=2))


if __name__=='__main__':
    main()
