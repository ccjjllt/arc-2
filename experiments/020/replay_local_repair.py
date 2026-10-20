#!/usr/bin/env python3
"""Manually replay fixed P0/R programs with bounded, demo-only CPU repair.

Input: {"tasks": {task_id: {"challenge": {"train": [...], "test": [{"input": ...}]},
                          "programs": [{"source": "P0", "code": "..."},
                                       {"source": "R", "code": "..."}]}},
        "metadata": {... optional provenance ...}}

Legacy row.code is accepted as a single P0 program. No model, GPU, query gold,
or near-program routing is used. The original program candidates and
all task/query IDs remain in the denominator even when admission/execution fails.
Each initial program gets at most 32 repair variants and 30 seconds of search.
This is a per-initial-program bound, not a per-task total. Baseline demo execution
and application of selected programs are separately bounded and accounted.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

HERE = Path(__file__).resolve().parent
if (HERE / 'submit').is_dir():
    sys.path.insert(0, str(HERE / 'submit'))

import solve_program as SP
import local_repair as LR
import submit_lib as SL


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def code_sha(code):
    return hashlib.sha256(code.encode()).hexdigest() if isinstance(code, str) else None


def query_execution(code, queries):
    started = time.monotonic()
    cutoff = started + 20.0
    results = []
    for query in queries:
        remaining = min(2.0, cutoff - time.monotonic())
        results.append(SP.execute_grid(code, query['input'], wall_seconds=remaining,
                                       memory_mb=512))
    return results, round(time.monotonic() - started, 6)


def ranked_programs(candidates):
    """Rank/deduplicate source bytes using demo evidence before any query use."""
    distinct = {}
    for candidate in candidates:
        candidate = dict(candidate)
        candidate['code_sha256'] = code_sha(candidate.get('code'))
        if not candidate['code_sha256']:
            continue
        digest = candidate['code_sha256']
        if digest not in distinct or SL.candidate_sort_key(candidate) < SL.candidate_sort_key(distinct[digest]):
            distinct[digest] = candidate
    return sorted(distinct.values(), key=SL.candidate_sort_key)[:2]


def task_replay(task_id, task, programs):
    started = time.monotonic()
    originals, variants, repairs = [], [], []
    demo_wall = search_wall = 0.0
    for initial_index, program in enumerate(programs):
        code, source = program['code'], program['source']
        if code:
            evaluation = SP.evaluate_program(code, task, include_queries=False)
        else:
            evaluation = {'code': code, 'status': 'missing_initial_program',
                          'demo_results': [], 'query_results': [],
                          'metrics': SL.demo_metrics(task['train'], []), 'execution_seconds': 0.0}
        evaluation.update(label=source, source=source, order=initial_index,
                          code_sha256=code_sha(code))
        originals.append(evaluation)
        demo_wall += evaluation.get('execution_seconds', 0.0)
        repair_started = time.monotonic()
        repair = LR.repair_program(code, task['train'], base_evaluation=evaluation,
                                   max_variants=32, max_seconds=30) if code else {
            'status': 'missing_initial_program', 'variants': [], 'variants_tried': 0,
            'variants_generated': 0, 'elapsed_s': 0.0, 'best': None}
        elapsed = time.monotonic() - repair_started
        search_wall += elapsed
        repair.update(initial_source=source, initial_code_sha256=code_sha(code),
                      search_wall_seconds=round(elapsed, 6))
        repairs.append(repair)
        for variant_index, variant in enumerate(repair.get('variants', [])):
            candidate = dict(variant)
            candidate.update(label=source + ':' + str(variant.get('label', variant_index)),
                             source=source + ':CPU', initial_source=source,
                             order=len(programs) + initial_index * 32 + variant_index)
            variants.append(candidate)

    # Both baseline and enhanced program choices are frozen before executing
    # query inputs. Cache shared programs across the two output policies.
    baseline_selected = ranked_programs(originals)
    cpu_selected = ranked_programs([*originals, *variants])
    query_cache = {}
    def apply(selected):
        applied, new_query_seconds = [], 0.0
        for candidate in selected:
            candidate = dict(candidate)
            digest = candidate['code_sha256']
            reused = digest in query_cache
            if not reused:
                results, elapsed = query_execution(candidate['code'], task['test'])
                query_cache[digest] = (results, elapsed)
                new_query_seconds += elapsed
            candidate.update(query_results=query_cache[digest][0],
                             query_execution_seconds=0.0 if reused else query_cache[digest][1],
                             query_results_reused=reused)
            applied.append(candidate)
        return applied, new_query_seconds
    baseline_applied, baseline_query_wall = apply(baseline_selected)
    cpu_applied, incremental_query_wall = apply(cpu_selected)
    return {'task_id': task_id, 'status': 'complete', 'n_queries': len(task['test']),
            'initial_program_count': len(programs), 'initial_programs': originals,
            'repairs': repairs, 'baseline_selected_programs': baseline_applied,
            'selected_programs': cpu_applied,
            'cost': {'initial_demo_execution_seconds': round(demo_wall, 6),
                     'repair_search_wall_seconds': round(search_wall, 6),
                     'baseline_query_execution_seconds': round(baseline_query_wall, 6),
                     'cpu_incremental_query_execution_seconds': round(incremental_query_wall, 6),
                     'query_programs_executed': len(query_cache),
                     'task_wall_seconds': round(time.monotonic() - started, 6)},
            'near_filter_used': False, 'query_gold_used': False}


def insert_candidates(task_id, evaluations, submission, query_map):
    pools = {}
    for order, evaluation in enumerate(evaluations):
        for index, result in enumerate(evaluation.get('query_results', [])):
            if result.get('status') != 'ok':
                continue
            try:
                grid = SL.valid_grid(result.get('grid'))
            except (ValueError, TypeError):
                continue
            key = '%s_%d' % (task_id, index)
            pools.setdefault(key, []).append({'grid': grid, 'metrics': evaluation['metrics'],
                'order': order, 'code_sha256': evaluation.get('code_sha256'),
                'source': evaluation.get('label', 'CPU')})
    for key, candidates in pools.items():
        first, second = SL.merge_candidate_pool(candidates)
        SL.update_slots(submission, query_map, {key: [first, second]})
    return list(pools)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', required=True)
    ap.add_argument('--output', required=True)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    if args.workers < 1:
        raise ValueError('--workers must be positive')
    source = Path(args.input)
    payload = json.loads(source.read_text())
    rows = payload['tasks']
    if not isinstance(rows, dict) or not rows:
        raise ValueError('input.tasks must be a nonempty task dictionary')
    challenges, programs = {}, {}
    for task_id, row in sorted(rows.items()):
        challenges[task_id] = SP.task_inputs_only(row['challenge'])
        initial = row.get('programs')
        if initial is None:
            initial = [{'source': 'P0', 'code': row.get('code')}]
        if not isinstance(initial, list):
            raise ValueError('programs must be a list: ' + task_id)
        programs[task_id] = []
        for index, program in enumerate(initial):
            code = program.get('code')
            source_name = program.get('source', 'initial%d' % index)
            if code is not None and not isinstance(code, str):
                raise ValueError('code must be a string or null: ' + task_id)
            if not isinstance(source_name, str) or not source_name:
                raise ValueError('program source must be a nonempty string: ' + task_id)
            programs[task_id].append({'source': source_name, 'code': code})

    output = Path(args.output)
    (output / 'per-task').mkdir(parents=True, exist_ok=True)
    baseline, query_map = SL.initial_submission(challenges)
    enhanced, _ = SL.initial_submission(challenges)
    baseline_keys, enhanced_keys = set(), set()
    started = time.monotonic()
    run = {'status': 'running', 'input_sha256': file_sha(source),
           'input_metadata': payload.get('metadata', {}),
           'source_sha256': {Path(mod.__file__).name: file_sha(mod.__file__) for mod in (SP, LR, SL)},
           'script_sha256': file_sha(__file__), 'python': sys.version,
           'platform': platform.platform(), 'cpu_count': os.cpu_count(),
           'workers_requested': args.workers, 'workers': min(args.workers, len(rows)),
           'n_tasks': len(challenges), 'n_queries': SL.count_queries(challenges),
           'task_ids': sorted(challenges), 'completed_task_ids': [], 'task_errors': {},
           'cost_by_task': {}, 'variants_tried_by_task': {},
           'near_filter_used': False, 'query_gold_used': False, 'gpu_used': False,
           'initial_program_count_by_task': {k: len(v) for k, v in programs.items()},
           'selection': 'Baseline ranks original P0/R programs; CPU ranks those originals plus all bounded variants. Each selects at most two distinct programs using demos before query execution. CPU final grid selection retains the executed baseline candidates plus selected CPU programs, at most four executed programs and exactly two final slots.',
           'limits': {'demo_seconds_per_initial_program': 20,
                      'repair_search_seconds_per_initial_program': 30,
                      'repair_variants_per_initial_program': 32,
                      'query_seconds_per_selected_program': 20, 'per_grid_seconds': 2}}
    def persist():
        SL.atomic_json(output / 'submission-baseline.json', baseline)
        SL.atomic_json(output / 'submission-cpu.json', enhanced)
        run['elapsed_seconds'] = round(time.monotonic() - started, 6)
        run['coverage'] = {
            'baseline': SL.coverage_report(baseline, query_map, updated_keys=baseline_keys),
            'cpu': SL.coverage_report(enhanced, query_map, updated_keys=enhanced_keys)}
        SL.atomic_json(output / 'run.json', run)
    persist()
    with ThreadPoolExecutor(max_workers=run['workers']) as executor:
        futures = {executor.submit(task_replay, task_id, task, programs[task_id]): task_id
                   for task_id, task in challenges.items()}
        for future in as_completed(futures):
            task_id = futures[future]
            try:
                result = future.result()
                SL.atomic_json(output / 'per-task' / (task_id + '.json'), result)
                baseline_keys.update(insert_candidates(task_id, result['baseline_selected_programs'], baseline, query_map))
                # Retain legal original candidates if a demo-promising repair
                # fails on a query; final ranking/deduplication still fills two slots.
                enhanced_keys.update(insert_candidates(task_id,
                    [*result['baseline_selected_programs'], *result['selected_programs']],
                    enhanced, query_map))
                run['cost_by_task'][task_id] = result['cost']
                run['variants_tried_by_task'][task_id] = sum(x.get('variants_tried', 0) for x in result['repairs'])
                run['completed_task_ids'].append(task_id)
                before = max((x['metrics']['demo_matches'] for x in result['baseline_selected_programs']), default=0)
                after = max((x['metrics']['demo_matches'] for x in result['selected_programs']), default=0)
                print('%s demo %d -> %d variants=%d wall=%.3fs' %
                      (task_id, before, after, run['variants_tried_by_task'][task_id],
                       result['cost']['task_wall_seconds']), flush=True)
            except Exception as exc:
                run['task_errors'][task_id] = {'error': '%s: %s' % (type(exc).__name__, exc),
                                               'traceback': traceback.format_exc()[-3000:]}
            persist()
    run['status'] = 'complete' if len(run['completed_task_ids']) == len(challenges) else 'partial'
    run['final_validate'] = {
        'baseline': SL.validate_submission(output / 'submission-baseline.json', baseline),
        'cpu': SL.validate_submission(output / 'submission-cpu.json', enhanced)}
    run['output_sha256'] = {name: file_sha(output / name)
                           for name in ('submission-baseline.json', 'submission-cpu.json')}
    run['task_cost_sum_not_wall_seconds'] = round(sum(x['task_wall_seconds'] for x in run['cost_by_task'].values()), 6)
    persist()
    return 0 if run['status'] == 'complete' and all(v[0] for v in run['final_validate'].values()) else 2


if __name__ == '__main__':
    raise SystemExit(main())
