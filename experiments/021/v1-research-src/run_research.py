"""Manually launched E20 V1 observation; no solutions are mounted or read here."""
import argparse
import bz2
import hashlib
import json
import math
import numbers
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time
import traceback


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False, indent=2) + '\n')
    os.replace(temporary, path)


def valid_grid(value):
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if not isinstance(value, list) or not 1 <= len(value) <= 30:
        return None
    width = None
    for row in value:
        if not isinstance(row, list) or not 1 <= len(row) <= 30:
            return None
        width = width or len(row)
        if len(row) != width or any(isinstance(v, bool) or not isinstance(v, numbers.Integral) or not 0 <= v <= 9 for v in row):
            return None
    return [[int(v) for v in row] for row in value]


def harvest(challenges, output_dir, run_dir, submission_path):
    # Rank using the actual V1 function; this retains its tie/insertion behaviour.
    from arc_decoder import score_kgmon
    submission = {task_id: [{'attempt_1': [[0]], 'attempt_2': [[0]]} for q in task['test']]
                  for task_id, task in challenges.items()}
    groups, errors, files = {}, [], 0
    for path in output_dir.iterdir():
        if path.name.endswith('.tmp'):
            continue
        key = path.name.split('.')[0]
        try:
            task_id, query = key.rsplit('_', 1)
            if task_id not in challenges or not 0 <= int(query) < len(challenges[task_id]['test']):
                raise ValueError('Candidate key does not belong to selected input')
            with bz2.BZ2File(path) as stream:
                candidates = pickle.load(stream)
            files += 1
            for index, candidate in enumerate(candidates):
                grid = valid_grid(candidate['solution'])
                scores = candidate['score_aug']
                if grid is None or not len(scores) or not all(math.isfinite(float(v)) for v in scores):
                    raise ValueError('Invalid grid or candidate score')
                if not math.isfinite(float(candidate['beam_score'])):
                    raise ValueError('Nonfinite beam score')
                groups.setdefault(key, {})[f'{path.name}.out{index}'] = candidate
        except Exception as exc:
            errors.append({'file': path.name, 'error': repr(exc)})
    pools = {}
    for key, candidates in groups.items():
        ranked = score_kgmon(candidates)
        task_id, query = key.rsplit('_', 1)
        for index, grid in enumerate(ranked[:2]):
            submission[task_id][int(query)]['attempt_%d' % (index + 1)] = valid_grid(grid)
        rows = []
        for grid in ranked:
            match = [value for value in candidates.values() if value['solution'].tolist() == grid.tolist()]
            aug_mean = sum(sum(float(v) for v in item['score_aug']) / len(item['score_aug']) for item in match) / len(match)
            rows.append({'grid': grid.tolist(), 'support': len(match), 'mean_aug_nll': aug_mean,
                         'kgmon': len(match) - aug_mean,
                         'beam_scores': [float(item['beam_score']) for item in match]})
        pools[key] = rows
    nquery = sum(len(task['test']) for task in challenges.values())
    report = {'tasks': len(challenges), 'queries': nquery, 'generated_queries': len(pools),
              'placeholder_queries': nquery - len(pools), 'candidate_files': files,
              'raw_candidates': sum(len(v) for v in groups.values()),
              'unique_candidates': sum(len(v) for v in pools.values()), 'errors': errors,
              'selection': 'unmodified V1 score_kgmon; original missing-slot [[0]]',
              'gold_read': False}
    atomic(submission_path, submission)
    atomic(run_dir / 'candidate-pools.json', pools)
    atomic(run_dir / 'coverage.json', report)
    return report


def terminate(process):
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=8)
    except subprocess.TimeoutExpired:
        pass
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.wait(timeout=8)


def model_path():
    roots = [Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1'),
             Path('/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1')]
    found = [path for path in roots if (path / 'config.json').is_file()]
    if len(found) != 1:
        raise RuntimeError('Expected exactly one fixed model mount: ' + repr(found))
    return found[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--notebook-start', type=float, required=True)
    parser.add_argument('--seconds', type=float, required=True)
    parser.add_argument('--task-seconds', type=float, required=True)
    parser.add_argument('--save-seconds', type=float, default=180)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent
    run = Path('/kaggle/working/021-run')
    run.mkdir(exist_ok=False)
    for name in ['candidates', 'events', 'markers']:
        (run / name).mkdir()
    challenge_path = source / 'challenges.json'
    challenges = json.loads(challenge_path.read_text())
    deadline = args.notebook_start + args.seconds - args.save_seconds
    state = {'status': 'starting', 'started': args.notebook_start, 'seconds': args.seconds,
             'save_seconds': args.save_seconds, 'solve_deadline': deadline,
             'task_seconds': args.task_seconds, 'selected_task_ids': sorted(challenges),
             'challenge_sha256': hashlib.sha256(challenge_path.read_bytes()).hexdigest(),
             'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in source.glob('*.py')},
             'policy': 'V1 training/search/ranking with bounded task/global deadlines',
             'gold_read': False}
    atomic(run / 'run.json', state)
    proc = None
    reason = None
    try:
        state['coverage'] = harvest(challenges, run / 'candidates', run, Path('/kaggle/working/submission.json'))
        gpu = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'], capture_output=True, text=True, check=True, timeout=10)
        state['gpu_names'] = gpu.stdout.strip().splitlines()
        if len(state['gpu_names']) != 4 or any('L4' not in name for name in state['gpu_names']):
            raise RuntimeError('Expected four L4 GPUs')
        fixed_model = model_path()
        state['model_path'] = str(fixed_model)
        state['model_config_sha256'] = hashlib.sha256((fixed_model / 'config.json').read_bytes()).hexdigest()
        env = dict(os.environ, ARC_MODEL_PATH=str(fixed_model), ARC_CHALLENGES_PATH=str(challenge_path),
                   ARC_OUTPUT_DIR=str(run / 'candidates'), ARC_EVENT_DIR=str(run / 'events'),
                   ARC_MARKER_DIR=str(run / 'markers'), ARC_TASK_SECONDS=str(args.task_seconds),
                   PYTHONUNBUFFERED='1', PYTHONHASHSEED='42', UNSLOTH_DISABLE_STATISTICS='1',
                   TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12')
        with (run / 'setup.log').open('w') as log:
            setup = subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', 'tensorflow'],
                                   env=env, stdout=log, stderr=subprocess.STDOUT,
                                   timeout=min(120, max(1, deadline - time.time())), check=True)
        if time.time() >= deadline:
            raise RuntimeError('No model execution time remains after setup')
        with (run / 'worker.log').open('w') as log:
            proc = subprocess.Popen([sys.executable, '-u', str(source / 'starter.py'), '--end-time', str(deadline)],
                                    env=env, cwd=source, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            state.update(status='running', worker_started=time.time())
            atomic(run / 'run.json', state)
            last_save = last_gpu = 0
            load_deadline = min(deadline, state['worker_started'] + 900)
            while proc.poll() is None:
                now = time.time()
                if now >= deadline:
                    reason = 'global_solve_deadline'
                    terminate(proc)
                    break
                errors = list((run / 'events').glob('pool-error-*.jsonl'))
                if errors:
                    reason = 'worker_error'
                    terminate(proc)
                    break
                if now >= load_deadline:
                    ready = set()
                    for event_file in (run / 'events').glob('worker-*.jsonl'):
                        for line in event_file.read_text(errors='replace').splitlines():
                            try:
                                event = json.loads(line)
                            except ValueError:
                                continue
                            if event.get('event') == 'model_loaded':
                                ready.add(event.get('rank'))
                    state['loaded_workers'] = sorted(ready)
                    if len(ready) != 4:
                        reason = 'model_load_deadline'
                        terminate(proc)
                        break
                if now - last_save >= 45:
                    state['coverage'] = harvest(challenges, run / 'candidates', run, Path('/kaggle/working/submission.json'))
                    state['elapsed_s'] = now - args.notebook_start
                    atomic(run / 'run.json', state)
                    print(json.dumps({'event': 'progress', 'elapsed_s': state['elapsed_s'], 'coverage': state['coverage']}), flush=True)
                    last_save = now
                if now - last_gpu >= 10:
                    with (run / 'gpu.csv').open('a') as gpu_log:
                        try:
                            subprocess.run(['nvidia-smi', '--query-gpu=timestamp,index,utilization.gpu,memory.used,power.draw', '--format=csv,noheader,nounits'],
                                           stdout=gpu_log, stderr=subprocess.DEVNULL, timeout=3)
                        except (OSError, subprocess.TimeoutExpired):
                            pass
                    last_gpu = now
                time.sleep(1)
            state['worker_returncode'] = proc.poll()
    except BaseException as exc:
        reason = reason or 'infrastructure_error'
        state.update(error=repr(exc), traceback=traceback.format_exc())
    finally:
        if proc is not None:
            try:
                terminate(proc)
            except BaseException as exc:
                state['termination_error'] = repr(exc)
        try:
            state['coverage'] = harvest(challenges, run / 'candidates', run, Path('/kaggle/working/submission.json'))
        except BaseException as exc:
            state['final_save_error'] = repr(exc)
        state.update(stop_reason=reason or 'workers_finished', elapsed_s=time.time() - args.notebook_start)
        normal = reason is None and state.get('worker_returncode') == 0
        bounded = reason == 'global_solve_deadline'
        state['status'] = 'complete' if (normal or bounded) and not state.get('termination_error') and not state.get('final_save_error') else 'failed'
        atomic(run / 'run.json', state)
        print(json.dumps({'event': 'final', 'status': state['status'], 'reason': state['stop_reason'], 'coverage': state.get('coverage')}), flush=True)
    return 0 if state['status'] == 'complete' else 2


if __name__ == '__main__':
    raise SystemExit(main())
