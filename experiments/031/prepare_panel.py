#!/usr/bin/env python3
"""Freeze a small exposed regression panel using input metadata only."""
import copy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SEED = '031:ttt64:quota20261027'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def identity(path):
    raw = path.read_bytes()
    return dict(path=str(path.resolve()), bytes=len(raw), sha256=sha(raw))


def main():
    source = ROOT/'experiments/030/panels-v1/initial40'
    metadata = json.loads((source/'manifest.json').read_bytes())
    challenges = json.loads((source/'challenges.json').read_bytes())
    pilot = json.loads((ROOT/'experiments/030/panels-v1/pilot20/manifest.json').read_bytes())
    if identity(source/'challenges.json')['sha256'] != '38df4c7126f160f9dd045b4b0c9fb37ccb4f24dfbb51c8d32aa7b50f132ebeb9':
        raise ValueError('Frozen input identity changed')
    entries = {row['task_id']: row for row in metadata['entries']}
    pressure = '2ba387bc'
    excluded = set(pilot['task_order'])
    eligible = sorted(set(challenges) - excluded - {pressure})
    if len(eligible) != 19 or pressure in excluded:
        raise ValueError('Expected19 remaining tasks plus one pressure task')
    groups = {}
    for key in eligible:
        row = entries[key]
        groups.setdefault(row['tier']+'_'+row['query_group'], []).append(key)
    allocation = {group: 7*len(keys)//19 for group, keys in groups.items()}
    remainders = sorted(groups, key=lambda group: (-(7*len(groups[group]) % 19), group))
    for group in remainders[:7-sum(allocation.values())]:
        allocation[group] += 1
    selected = [pressure]
    strata = {}
    for group, keys in sorted(groups.items()):
        ranked = sorted(keys, key=lambda key: (sha((SEED+':'+key).encode()), key))
        picked = ranked[:allocation[group]]
        selected.extend(picked)
        strata[group] = dict(eligible=ranked, n_selected=allocation[group], selected=sorted(picked))
    selected.sort()
    output = HERE/'panel-v1'
    output.mkdir(parents=True, exist_ok=False)
    subset = {key: challenges[key] for key in selected}
    if len(subset) != 8 or any(set(q) != {'input'} for task in subset.values() for q in task['test']):
        raise ValueError('Expected8 tasks without query answers')
    raw = (json.dumps(subset, ensure_ascii=False, sort_keys=True, indent=2)+'\n').encode()
    (output/'challenges.json').write_bytes(raw)
    frozen_entries = [copy.deepcopy(entries[key]) for key in selected]
    for row in frozen_entries:
        row['source_group'] = 'public_evaluation_NVARC031_exposed_regression'
    manifest = dict(created_at_utc=datetime.now(timezone.utc).isoformat(),
        cohort='031_ttt64_exposed_eight_task_regression', panel='panel8',
        status='input_panel_frozen_GPU_not_run', n_tasks=8,
        n_queries=sum(len(task['test']) for task in subset.values()),
        source_task_count=120, source_query_count=172,
        task_order=selected, entries=frozen_entries, source=metadata['source'],
        input_artifact=identity(output/'challenges.json'),
        sources=[identity(source/'manifest.json'), identity(source/'challenges.json'),
                 identity(ROOT/'experiments/030/panels-v1/pilot20/manifest.json')],
        selection_seed=SEED, strata=strata, forced_pressure_task=pressure,
        selection_rule=('Include2ba387bc as the previously known second-slot-only pressure task. '
            'Exclude all20 tasks from030 pilot to diversify public regression inputs. '
            'From the remaining19 allocate7 proportionally by inherited length tier crossed with '
            'single/multi-query; largest integer remainder, ties by stratum name. '
            'Within each stratum select ascending sha256(seed+colon+task_id), ties task_id. '
            'Retain every query and use original ascending task order.'),
        independent_confirmation=False, query_solutions_read=False, model_predictions_read=False,
        historical_pressure_annotation_used=True, other_outcomes_used_for_selection=False,
        limitations=['All eight tasks were already used in025 and019; this is exposed development evidence.',
            'The forced pressure task intentionally uses a historical outcome annotation.',
            'Input length strata do not establish runtime or semantic difficulty.',
            'All queries remain in the denominator after timeouts or missing candidates.',
            'A small positive result cannot guarantee a hidden-test score increase.'])
    (output/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(task_order=selected,n_tasks=8,n_queries=manifest['n_queries'],
        manifest=identity(output/'manifest.json'),challenges=identity(output/'challenges.json'),
        strata=strata),ensure_ascii=False,indent=2))


if __name__ == '__main__':
    main()
