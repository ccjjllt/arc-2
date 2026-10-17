"""014 R: B4 4B paired budget diagnosis (R0 vs R1, single pool, same task same card).

Normal save: the B4 cohort only (4 tasks / 6 queries), one pool on 4xL4 where
each card runs its task twice back-to-back (wave1 arm, then wave2 arm), 700s
per task per strategy, solve deadline = notebook start + 45min - 180s save.
Single model load (v4): two pools doubled the exposure to slow checkpoint
mounts and paid the multi-minute load twice. Per-wave candidate separation is
kept via wave1/wave2 output subdirs selected by pop position.
Hidden rerun: full challenge set with the 013-equivalent adaptive budget; the R
recording patch stays on but its ".raw." files never enter selection, so full-set
selection semantics are unchanged.

Reads b4_cohort.json shipped next to this file for task ids, wave order and caps.
Outputs per-wave submissions plus query_records.json (per strategy per query:
trained steps/seconds, raw/rescored counts, zero-candidate flags, stop reasons).
No solutions are read and noスコア-based selection happens here.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


RESEARCH_SESSION_CAP_SECONDS = 3600
NORMAL_TOTAL_SECONDS, NORMAL_SAVE_SECONDS = 2700, 180
RERUN_TOTAL_SECONDS, RERUN_SAVE_SECONDS = 43200, 600
WAVE_GRACE_SECONDS = 300


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    os.replace(temp, path)


def terminate_group(proc):
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    try:
        proc.wait(timeout=8)
    except subprocess.TimeoutExpired:
        return 'process_not_reaped_after_sigkill'
    return None


def read_events(folder):
    rows = []
    for file in sorted(Path(folder).glob('*.jsonl')):
        for line in file.read_text(errors='replace').splitlines():
            try:
                rows.append(json.loads(line))
            except (ValueError, TypeError):
                continue
    return rows


def rebuild(source, challenges, outputs, submission, report, timeout=45):
    cmd = [sys.executable, str(source / 'build_submission.py'),
           '--challenges', str(challenges), '--outputs', str(outputs),
           '--submission', str(submission), '--report', str(report)]
    completed = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    if completed.returncode:
        raise RuntimeError('Submission builder failed: ' + completed.stderr[-2000:])
    return json.loads(report.read_text())


def resolve_input(root, rerun):
    name = 'arc-agi_test_challenges.json' if rerun else 'arc-agi_evaluation_challenges.json'
    candidates = [root / 'competitions/arc-prize-2026-arc-agi-2' / name,
                  root / 'arc-prize-2026-arc-agi-2' / name]
    found = [p for p in candidates if p.is_file()]
    if len(found) != 1:
        raise RuntimeError('Expected one competition input, found: ' + repr(found))
    return found[0]


def resolve_model(root):
    candidates = [root / 'models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1',
                  root / 'qwen3_4b_grids15_sft139/transformers/bfloat16/1']
    found = [p for p in candidates if (p / 'config.json').is_file()]
    if len(found) != 1:
        raise RuntimeError('Expected one fixed Sorokin model version, found: ' + repr(found))
    return found[0]


def load_cohort(source):
    path = source / 'b4_cohort.json'
    cohort = json.loads(path.read_text())
    b4 = cohort['cohorts']['B4_4B_budget_diagnostic']
    assert b4['task_count'] == 4 and b4['query_count'] == 6, b4
    waves = b4['paired_protocol']['waves']
    assert len(waves) == 2 and all(len(w['assignments']) == 4 for w in waves), waves
    first = {a['task_id']: (a['gpu_index'], a['strategy']) for a in waves[0]['assignments']}
    second = {a['task_id']: (a['gpu_index'], a['strategy']) for a in waves[1]['assignments']}
    assert set(first) == set(second), (first, second)
    for task_id in first:
        assert first[task_id][0] == second[task_id][0], 'same task must stay on same card'
        assert first[task_id][1] != second[task_id][1], 'waves must swap strategies'
    ordered = sorted(first)
    assert [first[t][1] for t in ordered] == ['R0', 'R0', 'R1', 'R1'], first
    assert [second[t][1] for t in ordered] == ['R1', 'R1', 'R0', 'R0'], second
    policies = b4['paired_protocol']['strategy_policies']
    assert policies['R0']['task_wall_seconds'] == 700
    assert policies['R0']['train_allowance_seconds'] == 280
    r1 = policies['R1']
    assert r1['train_allowance_seconds'] == min(480, r1['task_wall_seconds'] - 220) == 480, r1
    return cohort, b4


def base_env(source, selected_path, model, outputs, events, markers):
    env = dict(os.environ)
    env.update({'PYTHONUNBUFFERED': '1', 'UNSLOTH_DISABLE_STATISTICS': '1',
                'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas', 'OMP_NUM_THREADS': '12',
                'ARC_CHALLENGES_PATH': str(selected_path), 'ARC_MODEL_PATH': str(model),
                'ARC_OUTPUT_DIR': str(outputs), 'ARC_EVENT_DIR': str(events),
                'ARC_MARKER_DIR': str(markers), 'ARC_RUN_TAG': 'base',
                'ARC_WORLD_SIZE': '4', 'ARC_NPROCS': '4', 'ARC_QUEUE_SENTINELS': '4',
                'ARC_TRAIN_AUG_N': '16', 'ARC_TRAIN_DATA_SEED': '1',
                'ARC_EVAL_SEED': '2', 'ARC_EVAL_AUG_N': '2',
                'ARC_MIN_TOKEN_PROB': '0.20', 'ARC_DFS_TIME_LIMIT': '540',
                'ARC_SCORE_SEED_OFFSET': '13', 'ARC_MAX_PUZZLE_SECONDS': '1200',
                'ARC_MAX_SCORED_BEAMS': '0', 'ARC_TRAIN_MAX_FRACTION': '0.4',
                'ARC_TRAIN_MAX_SECONDS': '480', 'ARC_INFERENCE_RESERVE_SECONDS': '60',
                'ARC_SCORING_RESERVE_SECONDS': '30'})
    return env


def run_pool(source, env, selected_path, assignments_path, deadline, worker_log, gpu_csv):
    proc = subprocess.Popen(
        [sys.executable, '-u', str(source / 'worker_pool.py'),
         '--challenges', str(selected_path), '--deadline', str(deadline),
         '--workers', '4'] + (['--assignments', str(assignments_path)]
                              if assignments_path is not None else []),
        cwd=source, env=env, stdout=worker_log, stderr=subprocess.STDOUT,
        start_new_session=True)
    return proc


def monitor_pool(proc, events, deadline, load_deadline, gpu_csv, state, run, tag):
    reason = None
    last_rebuild = last_sample = 0.0
    while proc.poll() is None:
        now = time.time()
        rows = read_events(events)
        ready = {row.get('rank') for row in rows if row.get('event') == 'model_loaded'}
        errors = [row for row in rows if row.get('event') in {'pool_worker_error', 'worker_error'}]
        if errors:
            reason = 'worker_error'
        elif now >= deadline:
            reason = 'solve_deadline'
        elif now >= load_deadline and len(ready) != 4:
            reason = 'load_deadline'
        if reason:
            terminate_group(proc)
            break
        if now - last_sample >= 10:
            with gpu_csv.open('a') as gpu_log:
                try:
                    subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                                    '--format=csv,noheader,nounits'], stdout=gpu_log,
                                   stderr=subprocess.DEVNULL, timeout=3)
                except (OSError, subprocess.TimeoutExpired):
                    pass
            last_sample = now
        time.sleep(1)
    return reason


def aggregate_query_records(b4, events):
    waves = b4['paired_protocol']['waves']
    strategy_of = {}
    gpu_of = {}
    for wave in waves:
        for entry in wave['assignments']:
            strategy_of[(wave['wave'], entry['task_id'])] = entry['strategy']
            gpu_of[entry['task_id']] = entry['gpu_index']
    queries = {(q['task_id'], q['query_index']): q['query_key'] for q in b4['queries']}
    rows = read_events(events)
    done = {}
    for row in rows:
        if row.get('event') == 'task_done' and row.get('task_key'):
            done[(row.get('wave', 1), row['task_key'])] = row
    records = []
    for wave_id in (1, 2):
        for (task_id, query_index), query_key in queries.items():
            strategy = strategy_of[(wave_id, task_id)]
            task_row = done.get((wave_id, task_id), {})
            stats = (task_row.get('query_stats') or {}).get(query_key, {})
            records.append({
                'wave': wave_id, 'task_id': task_id, 'query_index': query_index,
                'query_key': query_key, 'strategy': strategy, 'gpu_index': gpu_of[task_id],
                'task_status': task_row.get('status'), 'trained_steps': task_row.get('trained_steps'),
                'train_seconds_actual': task_row.get('train_seconds_actual'),
                'task_elapsed_s': task_row.get('elapsed_seconds'),
                'task_budget_s': task_row.get('task_budget_seconds'),
                'decode_batches': stats.get('decode_batches'),
                'raw_complete_beams': stats.get('raw_complete_beams'),
                'unparseable_beams': stats.get('unparseable_beams'),
                'empty_beam_batches': stats.get('empty_beam_batches'),
                'rescored_candidates': stats.get('rescored_candidates'),
                'rejected_empty_aug': stats.get('rejected_empty_aug'),
                'rejected_invalid_scores': stats.get('rejected_invalid_scores'),
                'first_generate_elapsed_s': stats.get('first_generate_elapsed_s'),
                'rescore_seconds': stats.get('rescore_seconds'),
                'zero_candidate': query_key in (task_row.get('zero_candidate_queries') or []),
            })
    records.sort(key=lambda r: (r['wave'], r['task_id'], r['query_index']))
    return records


def build_r_assignments(b4, task_ids, policies):
    """Build single-pool R assignments + per-wave strategy maps (pure data).

    Each card runs its task twice back-to-back: [wave1, wave2]. Returned
    assignments list is in per-rank wave order. CPU-testable before push.
    """
    waves = b4['paired_protocol']['waves']
    assert len(waves) == 2, waves
    per_task = {}
    for wave in waves:
        for entry in wave['assignments']:
            per_task.setdefault(entry['task_id'], {})[wave['wave']] = entry
    assert sorted(per_task) == sorted(task_ids), (sorted(per_task), sorted(task_ids))
    assert all(sorted(v) == [1, 2] for v in per_task.values())
    assignments = []
    wave1_map, wave2_map = {}, {}
    for rank, task_id in enumerate(sorted(per_task)):
        first, second = per_task[task_id][1], per_task[task_id][2]
        assert first['gpu_index'] == second['gpu_index'] == rank, (task_id, first, second)
        assert first['strategy'] != second['strategy'], (task_id, first, second)
        assignments += [{'gpu_index': rank, 'task_id': task_id},
                        {'gpu_index': rank, 'task_id': task_id}]
        wave1_map[task_id] = {'strategy': first['strategy'],
                              'train_max_seconds': policies[first['strategy']]}
        wave2_map[task_id] = {'strategy': second['strategy'],
                              'train_max_seconds': policies[second['strategy']]}
    return assignments, wave1_map, wave2_map


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook-start', required=True, type=float)
    parser.add_argument('--input-root', default='/kaggle/input')
    parser.add_argument('--work-root', default='/kaggle/working')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    work = Path(args.work_root)
    run = work / '014-r-b4-run'
    run.mkdir(parents=True, exist_ok=False)
    (run / 'markers').mkdir()
    rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').lower() in {'1', 'true', 'yes'}
    total_seconds, save_seconds = (RERUN_TOTAL_SECONDS, RERUN_SAVE_SECONDS) if rerun else (
        NORMAL_TOTAL_SECONDS, NORMAL_SAVE_SECONDS)
    deadline = args.notebook_start + total_seconds - save_seconds
    challenges = resolve_input(Path(args.input_root), rerun)
    all_data = json.loads(challenges.read_text())
    cohort, b4 = load_cohort(source)
    task_ids = sorted({q['task_id'] for q in b4['queries']})
    missing = [t for t in task_ids if t not in all_data]
    if not rerun and missing:
        raise RuntimeError('B4 tasks missing from competition input (no substitution allowed): '
                           + repr(missing))
    selected = all_data if rerun else {key: all_data[key] for key in task_ids}
    selected_path = run / 'selected-challenges.json'
    atomic_json(selected_path, selected)
    submission = work / 'submission.json'
    report = run / 'coverage.json'
    # Fallback is saved before dependency imports and model loading.
    outputs_init = run / 'candidates-init'
    outputs_init.mkdir()
    initial_report = rebuild(source, challenges, outputs_init, submission, report)
    model = resolve_model(Path(args.input_root))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.glob('*.py'))}
    manifest['b4_cohort.json'] = hashlib.sha256(
        (source / 'b4_cohort.json').read_bytes()).hexdigest()
    state = {'status': 'preparing', 'rerun': rerun, 'policy': 'r_b4_paired',
             'notebook_start': args.notebook_start, 'total_seconds': total_seconds,
             'save_seconds': save_seconds, 'solve_deadline': deadline,
             'research_session_cap_seconds': RESEARCH_SESSION_CAP_SECONDS,
             'selected_task_ids': sorted(selected),
             'selected_queries': sum(len(x['test']) for x in selected.values()),
             'input_sha256': hashlib.sha256(challenges.read_bytes()).hexdigest(),
             'selected_input_sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest(),
             'source_sha256': manifest, 'model_path': str(model),
             'initial_coverage': initial_report,
             'b4_task_ids': task_ids,
             'b4_query_count': b4['query_count'],
             'fixed_task_seconds': 700,
             'r0_train_seconds': 280,
             'r1_train_seconds': min(480, 700 - 220)}
    atomic_json(run / 'supervisor.json', state)

    def interrupted(signum, frame):
        raise RuntimeError('Supervisor received signal %s' % signum)
    signal.signal(signal.SIGTERM, interrupted)

    proc = None
    reason = None
    try:
        setup_budget = min(120, deadline - time.time() - 60)
        if setup_budget <= 0:
            raise RuntimeError('Insufficient setup budget')
        devices = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                 capture_output=True, text=True, timeout=10, check=True)
        state['gpu_names'] = devices.stdout.strip().splitlines()
        if len(state['gpu_names']) != 4 or any('L4' not in name for name in state['gpu_names']):
            raise RuntimeError('Expected exactly four L4 GPUs')
        with (run / 'setup.log').open('w') as setup_log:
            setup = subprocess.Popen([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow'],
                                     stdout=setup_log, stderr=subprocess.STDOUT,
                                     env=dict(os.environ), start_new_session=True)
            try:
                setup.wait(timeout=setup_budget)
            finally:
                setup_error = terminate_group(setup)
                if setup_error:
                    state['setup_termination_error'] = setup_error
        state['setup_returncode'] = setup.returncode
        if setup.returncode:
            raise RuntimeError('TensorFlow removal failed')

        if rerun:
            # Full-set hidden semantics: 013-equivalent adaptive budget, no pinning.
            events = run / 'events'
            events.mkdir()
            outputs = run / 'candidates'
            outputs.mkdir()
            env = base_env(source, selected_path, model, outputs, events, run / 'markers')
            state['environment_config'] = {k: v for k, v in env.items() if k.startswith('ARC_')}
            state['waves'] = [{'wave': 'full_rerun', 'assignments': 'shared_queue_013_equivalent'}]
            atomic_json(run / 'supervisor.json', state)
            with (run / 'worker.log').open('w') as worker_log:
                proc = run_pool(source, env, selected_path, None, deadline, worker_log,
                                run / 'gpu.csv')
                state.update(status='running', worker_started=time.time(), process_id=proc.pid)
                atomic_json(run / 'supervisor.json', state)
                load_deadline = min(deadline, time.time() + 600)
                reason = monitor_pool(proc, events, deadline, load_deadline,
                                      run / 'gpu.csv', state, run, 'full')
                state['worker_returncode'] = proc.poll()
            state['coverage'] = rebuild(source, challenges, outputs, submission, report)
        else:
            # Single pool, single model load (v4): each card runs its task
            # twice back-to-back (wave1 arm, then wave2 arm). v1/v3 showed two
            # pools double the exposure to slow checkpoint mounts (v3 wave1
            # stalled on shard loading past the 600s load line) and pay the
            # multi-minute load twice. Cross-arm isolation is per-task
            # (adapter reset + fresh Trainer + fixed seeds), matching upstream
            # hidden reruns where one worker serves many tasks sequentially.
            # Order balance is structural: ranks 0/1 run R0->R1, ranks 2/3 R1->R0.
            policies = {'R0': 280, 'R1': min(480, 700 - 220)}
            assignments, wave1_map, wave2_map = build_r_assignments(b4, task_ids, policies)
            map1_path = run / 'wave1-strategy-map.json'
            map2_path = run / 'wave2-strategy-map.json'
            atomic_json(map1_path, wave1_map)
            atomic_json(map2_path, wave2_map)
            assignments_path = run / 'assignments.json'
            atomic_json(assignments_path, assignments)
            events = run / 'events'
            events.mkdir()
            outputs = run / 'candidates'
            outputs.mkdir()
            env = base_env(source, selected_path, model, outputs, events,
                           run / 'markers')
            # R fixed budget shared by both arms; ONLY the per-task train cap differs.
            # Fraction 0.8 + reserve 220 keep the explicit caps binding:
            # R0 min(280, 560, 480)=280, R1 min(480, 560, 480)=480.
            # Single 900s load line: v3 stalled on slow checkpoint shards past
            # 600s; 900+1400=2300s still fits the 2520s solve window.
            env.update({'ARC_MAX_PUZZLE_SECONDS': '700',
                        'ARC_R_FIXED_TASK_SECONDS': '700',
                        'ARC_TRAIN_MAX_FRACTION': '0.8',
                        'ARC_INFERENCE_RESERVE_SECONDS': '220',
                        'ARC_TRAIN_MAX_SECONDS': '280',
                        'ARC_MODEL_LOAD_MAX_SECONDS': '900',
                        'ARC_R_STRATEGY_MAP_WAVE1': str(map1_path),
                        'ARC_R_STRATEGY_MAP_WAVE2': str(map2_path)})
            state['environment_config'] = {k: v for k, v in env.items()
                                           if k.startswith('ARC_')}
            state['waves'] = [
                {'wave': 1,
                 'strategy_map_sha256': hashlib.sha256(map1_path.read_bytes()).hexdigest(),
                 'assignments': assignments[::2]},
                {'wave': 2,
                 'strategy_map_sha256': hashlib.sha256(map2_path.read_bytes()).hexdigest(),
                 'assignments': assignments[1::2]}]
            atomic_json(run / 'supervisor.json', state)
            pool_start = time.time()
            with (run / 'worker.log').open('w') as worker_log:
                proc = run_pool(source, env, selected_path, assignments_path, deadline,
                                worker_log, run / 'gpu.csv')
                state.update(status='running', worker_started=pool_start,
                             process_id=proc.pid)
                atomic_json(run / 'supervisor.json', state)
                load_deadline = min(deadline, pool_start + 900)
                reason = monitor_pool(proc, events, deadline, load_deadline,
                                      run / 'gpu.csv', state, run, 'single')
                state['worker_returncode'] = proc.poll()
            proc = None
            wave_reports = {}
            for wave_id in (1, 2):
                wave_dir = outputs / f'wave{wave_id}'
                wave_submission = run / f'wave{wave_id}-submission.json'
                wave_report = run / f'wave{wave_id}-coverage.json'
                try:
                    if not wave_dir.is_dir():
                        raise RuntimeError(f'wave{wave_id} output dir missing (pool ended early)')
                    wave_reports[f'wave{wave_id}'] = rebuild(
                        source, challenges, wave_dir, wave_submission, wave_report)
                except BaseException as exc:
                    wave_reports[f'wave{wave_id}'] = {'status': 'rebuild_failed',
                                                     'error': repr(exc)}
            state['waves'][0]['coverage'] = wave_reports['wave1']
            state['waves'][1]['coverage'] = wave_reports['wave2']
            query_records = aggregate_query_records(b4, events)
            atomic_json(run / 'query_records.json', query_records)
            state['query_records'] = str(run / 'query_records.json')
            state['query_record_count'] = len(query_records)
            # Diagnostic run: per-wave submissions carry the arm pairing; the
            # working submission tracks wave1 (first arm on every card).
            state['coverage'] = rebuild(source, challenges, outputs / 'wave1',
                                        submission, report)
    except BaseException as exc:
        state['error'] = repr(exc)
        reason = reason or 'infrastructure_error'
    finally:
        if proc is not None:
            try:
                termination_error = terminate_group(proc)
                if termination_error:
                    state['termination_error'] = termination_error
            except BaseException as exc:
                state['termination_error'] = repr(exc)
        state['stop_reason'] = reason or 'workers_finished'
        try:
            if 'coverage' not in state:
                state['coverage'] = rebuild(source, challenges,
                                            run / 'candidates'
                                            if rerun else run / 'candidates' / 'wave1',
                                            submission, report)
        except BaseException as exc:
            state['final_save_error'] = repr(exc)
        state['elapsed_s'] = time.time() - args.notebook_start
        normal_exit = reason is None
        bounded_exit = reason == 'solve_deadline'
        try:
            worker_ok = state.get('worker_returncode', 0) in {0, None} or all(
                w.get('worker_returncode') == 0 for w in state.get('waves', [])
                if isinstance(w, dict) and 'worker_returncode' in w)
        except BaseException:
            worker_ok = False
        state['status'] = ('complete' if (normal_exit or bounded_exit) and worker_ok and
                           not any(state.get(key) for key in ['final_save_error', 'termination_error',
                                                              'setup_termination_error']) else 'failed')
        atomic_json(run / 'supervisor.json', state)
        print(json.dumps({'event': 'final', 'status': state['status'], 'reason': state['stop_reason'],
                          'elapsed_s': state['elapsed_s']}), flush=True)
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
