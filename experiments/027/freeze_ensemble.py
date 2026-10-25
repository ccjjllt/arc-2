#!/usr/bin/env python3
"""Freeze one base/extra same-weight NVARC run before reading query answers.

The selector is pure stdlib and may be imported by a notebook final cell. The
CLI only copies predictions and exports candidate diagnostics; it never reads
gold, starts a model, or submits to Kaggle.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path


RULE = ('Keep base first. If extra has real candidates and its first grid is '
        'different, replace only base second with extra first; otherwise retain '
        'base second. Never scan extra second. If base has no real candidates, '
        'use extra two slots; if neither arm has candidates, use placeholders.')


def valid_grid(value):
    return (isinstance(value, list) and 1 <= len(value) <= 30
            and isinstance(value[0], list) and 1 <= len(value[0]) <= 30
            and all(isinstance(row, list) and len(row) == len(value[0])
                    and all(type(cell) is int and 0 <= cell <= 9 for cell in row)
                    for row in value))


def select_ensemble_query(base_slots, extra_slots, base_candidate_count,
                          extra_candidate_count):
    """Return (two-slot prediction, provenance), using real candidate counts.

    A literal [[0]] is a legal ARC grid. Only a zero candidate count makes it a
    placeholder. Inputs are never modified. The ordinary path reads only extra
    attempt_1; extra attempt_2 is used solely when base has no real candidate.
    """
    for count in (base_candidate_count, extra_candidate_count):
        if type(count) is not int or count < 0:
            raise ValueError('Candidate counts must be nonnegative integers')

    def copy_grid(grid):
        return [list(row) for row in grid]

    audit = dict(base_candidate_count=base_candidate_count,
                 extra_candidate_count=extra_candidate_count,
                 base_first_preserved=base_candidate_count > 0,
                 extra_second_scanned=False)
    if base_candidate_count == 0:
        if extra_candidate_count == 0:
            result = dict(attempt_1=[[0]], attempt_2=[[0]])
            audit.update(reason='no_real_candidates', slot_1_source='placeholder',
                         slot_2_source='placeholder')
        else:
            if not isinstance(extra_slots, dict) or any(
                    not valid_grid(extra_slots.get(key))
                    for key in ('attempt_1', 'attempt_2')):
                raise ValueError('Real extra candidates require valid exported slots')
            result = {key: copy_grid(extra_slots[key])
                      for key in ('attempt_1', 'attempt_2')}
            audit.update(reason='base_empty_use_extra_two',
                         slot_1_source='extra_B4.attempt_1',
                         slot_2_source='extra_B4.attempt_2',
                         extra_second_scanned=True)
        return result, audit

    if not isinstance(base_slots, dict) or any(
            not valid_grid(base_slots.get(key))
            for key in ('attempt_1', 'attempt_2')):
        raise ValueError('Real base candidates require valid exported slots')
    result = {key: copy_grid(base_slots[key])
              for key in ('attempt_1', 'attempt_2')}
    audit.update(slot_1_source='base_B4.attempt_1',
                 slot_2_source='base_B4.attempt_2')
    if extra_candidate_count == 0:
        audit['reason'] = 'extra_empty_keep_base'
    elif not isinstance(extra_slots, dict) or not valid_grid(extra_slots.get('attempt_1')):
        audit['reason'] = 'extra_first_invalid_keep_base'
    elif extra_slots['attempt_1'] == base_slots['attempt_1']:
        audit['reason'] = 'first_slots_agree_keep_base_second'
    else:
        result['attempt_2'] = copy_grid(extra_slots['attempt_1'])
        audit.update(reason='different_extra_first_replace_second',
                     slot_2_source='extra_B4.attempt_1')
    return result, audit


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def grid_sha(grid):
    return sha(json.dumps(grid, sort_keys=True, separators=(',', ':')).encode())


def read(path, lines=False):
    path = Path(path)
    raw = path.read_bytes()
    value = ([json.loads(line) for line in raw.decode().splitlines() if line.strip()]
             if lines else json.loads(raw))
    return value, dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True,
                               indent=2, allow_nan=False) + '\n')


def validate_predictions(prediction, query_rows):
    task_ids = {row['task_id'] for row in query_rows}
    if not isinstance(prediction, dict) or set(prediction) != task_ids:
        raise ValueError('Prediction task IDs differ from the complete panel')
    for task_id in task_ids:
        indices = [row['query_index'] for row in query_rows if row['task_id'] == task_id]
        if not isinstance(prediction[task_id], list) or len(prediction[task_id]) != len(indices):
            raise ValueError('Prediction query denominator differs')
        for index in indices:
            slots = prediction[task_id][index]
            if not isinstance(slots, dict) or set(slots) != {'attempt_1', 'attempt_2'}:
                raise ValueError('Prediction must contain exactly two named slots')
            if any(not valid_grid(slots[key]) for key in ('attempt_1', 'attempt_2')):
                raise ValueError('Invalid exported prediction grid')


def candidate_diagnostics(root, allowed, inputs):
    scored, identity = read(root / 'scored-candidates.jsonl', lines=True)
    inputs.append(identity)
    counts = Counter()
    scored_grids, raw_grids = defaultdict(dict), defaultdict(dict)
    for record in scored:
        key, grid = record['query_key'], record['solution']
        scores = record['score_aug']
        if key not in allowed or not valid_grid(grid):
            raise ValueError('Scored candidate grid/query is invalid')
        if not scores or any(type(x) not in (int, float) or not math.isfinite(x) for x in scores):
            raise ValueError('Scored candidate contains invalid scores')
        counts[key] += 1
        scored_grids[key][grid_sha(grid)] = grid
    raw_paths = sorted(root.glob('raw-candidates-rank*.jsonl'))
    if not raw_paths:
        raise ValueError('Raw observation files are required even for an empty extra arm')
    for path in raw_paths:
        records, identity = read(path, lines=True)
        inputs.append(identity)
        for record in records:
            key, grid = record['query_key'], record['solution']
            if key not in allowed or not valid_grid(grid):
                raise ValueError('Raw candidate grid/query is invalid')
            raw_grids[key][grid_sha(grid)] = grid
    return counts, scored_grids, raw_grids


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ensemble', type=Path,
                        help='Defaults to run-root/submission-ensemble.json; must match the pure rule.')
    args = parser.parse_args()
    manifest, manifest_id = read(args.manifest)
    state, state_id = read(args.run_root / 'run-state.json')
    if state.get('base', {}).get('return_code') != 0:
        raise ValueError('Base stage must return successfully before freezing the comparison')
    if state['input_identity']['challenges_sha256'] != manifest['input_artifact']['sha256']:
        raise ValueError('Runtime challenge identity differs from the panel')
    query_rows = sorted([dict(task_id=entry['task_id'], **query)
                         for entry in manifest['entries'] for query in entry['queries']],
                        key=lambda row: (row['task_id'], row['query_index']))
    for entry in manifest['entries']:
        indices = [row['query_index'] for row in query_rows if row['task_id'] == entry['task_id']]
        if indices != list(range(entry['n_test_queries'])):
            raise ValueError('Manifest query indices do not cover the task')
    if any(row['query_key'] != row['task_id'] + '_' + str(row['query_index']) for row in query_rows):
        raise ValueError('Manifest query keys differ from their task/index')
    allowed = {row['query_key'] for row in query_rows}
    if len(allowed) != manifest['n_queries'] or len(query_rows) != len(allowed):
        raise ValueError('Manifest query denominator/identities differ')
    inputs = [manifest_id, state_id]
    predictions, counts, scored, raw = {}, {}, {}, {}
    for arm in ('base', 'extra'):
        predictions[arm], identity = read(args.run_root / arm / 'submission-B4.json')
        inputs.append(identity)
        validate_predictions(predictions[arm], query_rows)
        counts[arm], scored[arm], raw[arm] = candidate_diagnostics(args.run_root / arm, allowed, inputs)
    ensemble = {entry['task_id']: [] for entry in manifest['entries']}
    decisions = []
    for row in query_rows:
        key, task_id, index = row['query_key'], row['task_id'], row['query_index']
        slots, audit = select_ensemble_query(predictions['base'][task_id][index],
            predictions['extra'][task_id][index], counts['base'][key], counts['extra'][key])
        ensemble[task_id].append(slots)
        decisions.append(dict(query_key=key, task_id=task_id, query_index=index,
            input_sha256=row['input_sha256'], **audit,
            slot_1_sha256=grid_sha(slots['attempt_1']), slot_2_sha256=grid_sha(slots['attempt_2'])))
    remote_path = args.ensemble or args.run_root / 'submission-ensemble.json'
    remote, remote_id = read(remote_path)
    inputs.append(remote_id)
    validate_predictions(remote, query_rows)
    if remote != ensemble:
        raise ValueError('Remote ensemble differs from the fixed pure selector')
    args.output.mkdir(parents=True, exist_ok=False)
    output_ids = {}
    for name, source in [('base_B4', args.run_root / 'base/submission-B4.json'),
                         ('extra_B4', args.run_root / 'extra/submission-B4.json'),
                         ('ensemble', remote_path)]:
        destination = args.output / ('submission-' + name + '.json')
        destination.write_bytes(source.read_bytes())
        output_ids[destination.name] = dict(bytes=destination.stat().st_size, sha256=sha(destination.read_bytes()))
    ledger = []
    for kind, pools in [('scored', scored), ('raw', raw)]:
        exported = {name: {} for name in ('base_B4', 'extra_B4', 'ensemble')}
        for key in sorted(allowed):
            union = {**pools['base'][key], **pools['extra'][key]}
            for arm in ('base', 'extra'):
                exported[arm + '_B4'][key] = [{'grid': grid} for _, grid in sorted(pools[arm][key].items())]
            exported['ensemble'][key] = [{'grid': grid} for _, grid in sorted(union.items())]
        write(args.output / ('candidate-pools-' + kind + '.json'), exported)
    for row in query_rows:
        key = row['query_key']
        for arm in ('base', 'extra'):
            ledger.append(dict(arm=arm, query_key=key, scored_records=counts[arm][key],
                scored_candidate_sha256=sorted(scored[arm][key]), raw_candidate_sha256=sorted(raw[arm][key])))
    for name, rows in [('ensemble-decisions.jsonl', decisions), ('candidate-ledger.jsonl', ledger)]:
        with (args.output / name).open('w') as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + '\n')
    for name in ('candidate-pools-scored.json', 'candidate-pools-raw.json',
                 'ensemble-decisions.jsonl', 'candidate-ledger.jsonl'):
        path = args.output / name
        output_ids[name] = dict(bytes=path.stat().st_size, sha256=sha(path.read_bytes()))
    report = dict(frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort=manifest['cohort'], n_tasks=manifest['n_tasks'], n_queries=len(allowed),
        gold_read=False, fixed_rule=RULE, selector_source_sha256=sha(Path(__file__).read_bytes()),
        remote_ensemble_exactly_reproduced=True,
        decision_counts=dict(Counter(row['reason'] for row in decisions)),
        inputs=inputs, outputs=output_ids,
        note='All three predictions frozen before scoring. Ensemble oracle pools are the union of both arms, not selected answers.')
    write(args.output / 'prediction-freeze.json', report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
