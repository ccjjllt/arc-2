"""Bound the whole candidate run and keep a complete submission on disk."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time


PILOT_IDS = ['0934a4d8', '36a08778', '981571dc', 'aa4ec2a5']


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
    # Reap descendants even if the pool leader exited first.
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook-start', required=True, type=float)
    parser.add_argument('--input-root', default='/kaggle/input')
    parser.add_argument('--work-root', default='/kaggle/working')
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    work = Path(args.work_root)
    run = work / '013-run'
    # Fresh saved/hidden execution only; never reuse candidate files by accident.
    run.mkdir(parents=True, exist_ok=False)
    outputs = run / 'candidates'
    outputs.mkdir()
    events = run / 'events'
    events.mkdir()
    (run / 'markers').mkdir()
    rerun = os.getenv('KAGGLE_IS_COMPETITION_RERUN', '').lower() in {'1', 'true', 'yes'}
    total_seconds, save_seconds = (43200, 600) if rerun else (2700, 180)
    deadline = args.notebook_start + total_seconds - save_seconds
    challenges = resolve_input(Path(args.input_root), rerun)
    all_data = json.loads(challenges.read_text())
    selected = all_data if rerun else {key: all_data[key] for key in PILOT_IDS}
    selected_path = run / 'selected-challenges.json'
    atomic_json(selected_path, selected)
    submission = work / 'submission.json'
    report = run / 'coverage.json'
    # Fallback is saved before dependency imports and model loading.
    initial_report = rebuild(source, challenges, outputs, submission, report)
    model = resolve_model(Path(args.input_root))
    manifest = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(source.glob('*.py'))}
    state = {'status': 'preparing', 'rerun': rerun, 'policy': 'base_only_kgmon',
             'notebook_start': args.notebook_start, 'total_seconds': total_seconds,
             'save_seconds': save_seconds, 'solve_deadline': deadline,
             'selected_task_ids': sorted(selected),
             'selected_queries': sum(len(x['test']) for x in selected.values()),
             'input_sha256': hashlib.sha256(challenges.read_bytes()).hexdigest(),
             'selected_input_sha256': hashlib.sha256(selected_path.read_bytes()).hexdigest(),
             'source_sha256': manifest, 'model_path': str(model),
             'initial_coverage': initial_report}
    atomic_json(run / 'supervisor.json', state)
    def interrupted(signum, frame):
        raise RuntimeError('Supervisor received signal %s' % signum)
    signal.signal(signal.SIGTERM, interrupted)
    env = dict(os.environ)
    env.update({'PYTHONUNBUFFERED': '1', 'UNSLOTH_DISABLE_STATISTICS': '1',
                'TRITON_PTXAS_PATH': '/usr/local/cuda/bin/ptxas', 'OMP_NUM_THREADS': '12',
                'ARC_CHALLENGES_PATH': str(selected_path), 'ARC_MODEL_PATH': str(model),
                'ARC_OUTPUT_DIR': str(outputs), 'ARC_EVENT_DIR': str(events),
                'ARC_MARKER_DIR': str(run / 'markers'), 'ARC_RUN_TAG': 'base',
                'ARC_WORLD_SIZE': '4', 'ARC_NPROCS': '4', 'ARC_QUEUE_SENTINELS': '4',
                'ARC_TRAIN_AUG_N': '16', 'ARC_TRAIN_DATA_SEED': '1',
                'ARC_EVAL_SEED': '2', 'ARC_EVAL_AUG_N': '2',
                'ARC_MIN_TOKEN_PROB': '0.20', 'ARC_DFS_TIME_LIMIT': '540',
                'ARC_SCORE_SEED_OFFSET': '13', 'ARC_MAX_PUZZLE_SECONDS': '1200',
                'ARC_MAX_SCORED_BEAMS': '0', 'ARC_TRAIN_MAX_FRACTION': '0.4',
                'ARC_TRAIN_MAX_SECONDS': '480', 'ARC_INFERENCE_RESERVE_SECONDS': '60',
                'ARC_SCORING_RESERVE_SECONDS': '30'})
    state['environment_config'] = {k: v for k, v in env.items() if k.startswith('ARC_')}
    proc = None
    reason = None
    last_rebuild = last_sample = 0.0
    try:
        # Match the verified upstream image: TensorFlow conflicts with Unsloth.
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
                                     env=env, start_new_session=True)
            try:
                setup.wait(timeout=setup_budget)
            finally:
                setup_error = terminate_group(setup)
                if setup_error:
                    state['setup_termination_error'] = setup_error
        state['setup_returncode'] = setup.returncode
        if setup.returncode:
            raise RuntimeError('TensorFlow removal failed')
        with (run / 'worker.log').open('w') as worker_log:
            proc = subprocess.Popen([sys.executable, '-u', str(source / 'worker_pool.py'),
                                     '--challenges', str(selected_path), '--deadline', str(deadline)],
                                    cwd=source, env=env, stdout=worker_log,
                                    stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', worker_started=time.time(), process_id=proc.pid)
            atomic_json(run / 'supervisor.json', state)
            load_deadline = min(deadline, time.time() + 600)
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
                if now - last_rebuild >= 60 and deadline - now > 60:
                    state['coverage'] = rebuild(source, challenges, outputs, submission, report)
                    last_rebuild = now
                    state.update(elapsed_s=now - args.notebook_start, loaded_workers=sorted(ready))
                    atomic_json(run / 'supervisor.json', state)
                    compact = {key: state['coverage'].get(key) for key in
                               ['generated_query_count', 'fallback_count', 'candidate_files', 'corrupt_count']}
                    print(json.dumps({'event': 'progress', 'elapsed_s': state['elapsed_s'],
                                      'loaded_workers': sorted(ready), 'coverage': compact}), flush=True)
                if now - last_sample >= 10:
                    with (run / 'gpu.csv').open('a') as gpu_log:
                        try:
                            subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw',
                                            '--format=csv,noheader,nounits'], stdout=gpu_log,
                                           stderr=subprocess.DEVNULL, timeout=3)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                    last_sample = now
                time.sleep(1)
            state['worker_returncode'] = proc.poll()
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
            state['coverage'] = rebuild(source, challenges, outputs, submission, report)
        except BaseException as exc:
            state['final_save_error'] = repr(exc)
        state['elapsed_s'] = time.time() - args.notebook_start
        normal_exit = reason is None and state.get('worker_returncode') == 0
        bounded_exit = reason == 'solve_deadline' and state.get('worker_returncode') in {0, -15, -9}
        state['status'] = ('complete' if (normal_exit or bounded_exit) and
                           not any(state.get(key) for key in ['final_save_error', 'termination_error',
                                                              'setup_termination_error']) else 'failed')
        atomic_json(run / 'supervisor.json', state)
        print(json.dumps({'event': 'final', 'status': state['status'], 'reason': state['stop_reason'],
                          'elapsed_s': state['elapsed_s']}), flush=True)
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
