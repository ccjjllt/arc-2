#!/usr/bin/env python3
"""020 task-level program policies; no query gold enters this process.

paired: shared P0, M0=P0+R, M1=P0+bounded CPU repair+conditional F/R.
paired --p0-only: research cohort gets one P0; M0/M1 both retain that output.
submit: execute only the chosen policy. All callbacks execute/persist immediately.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

import run_reasoning_probe as P
import solve_program as SP
import submit_lib as SL
import local_repair as LR

VLLM_WHEEL_NAME = 'vllm-0.27.2rc1.dev91+g1f7427bc0-cp38-abi3-manylinux_2_28_x86_64.whl'
VLLM_WHEEL_SHA256 = '5cbeedeb8266362ef935775c4c33203e4dab74e8da1649a977bae833df62304a'
INSTALL_BUDGET_S = 1200
SAVE_MARGIN_S = 300
NEW_REQUEST_MARGIN_S = 600
SEEDS = {'P0': 5101, 'F': 5103, 'R': 5105}
GEN_MAX = 12288
THINK_MAX = 8192
DIRECT_CONTEXT_MARGIN = 128
STATE = {'proc': None, 'work': None, 'run': None, 'closed': False}
GPU_SAMPLER_STOP = threading.Event()


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def log(message):
    line = '[020 %s] %s' % (time.strftime('%H:%M:%S'), message)
    print(line, flush=True)
    if STATE.get('work'):
        with (STATE['work'] / 'driver-stdout.log').open('a') as stream:
            stream.write(line + '\n')


def save(work, run):
    SL.atomic_json(work / 'run.json', run)


def sh(cmd, timeout, cwd=None, env=None):
    setup_deadline = STATE.get('setup_deadline')
    if setup_deadline is not None:
        timeout = min(timeout, setup_deadline - time.monotonic())
        if timeout <= 0:
            raise subprocess.TimeoutExpired(cmd, 0)
    return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, cwd=cwd, env=env)


def finish(status, code=0, detail=None):
    if STATE['closed']:
        return code
    STATE['closed'] = True
    STATE['setup_deadline'] = None
    GPU_SAMPLER_STOP.set()
    run, work, proc = STATE.get('run'), STATE.get('work'), STATE.get('proc')
    if proc is not None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)
        except (OSError, subprocess.TimeoutExpired):
            pass
    if run is not None and work is not None:
        run['status'] = status
        run['elapsed_s'] = round(time.monotonic() - run['start_monotonic'], 3)
        if detail:
            run.setdefault('first_error', str(detail)[:2000])
        save(work, run)
        # Installed libraries are reproducible inputs, not useful notebook outputs.
        try:
            sh([sys.executable, '-c', 'import shutil,sys; shutil.rmtree(sys.argv[1],ignore_errors=True)',
                str(work / 'site')], timeout=45)
        except subprocess.TimeoutExpired:
            run['site_cleanup'] = 'timeout'
            save(work, run)
        log('EVIDENCE-EXIT ' + status)
    return code


def on_term(signum, _frame):
    finish('signal_%s' % signum, 1)
    raise SystemExit(1)


def compact(value):
    return json.dumps(value, ensure_ascii=False, separators=(',', ':'))


def build_program_prompt(task):
    task = SP.task_inputs_only(task)
    lines = ['Infer one rule that explains every demonstration. Write one reusable Python program.']
    for index, demo in enumerate(task['train']):
        lines.append('demo%d input=%s output=%s' % (index, compact(demo['input']), compact(demo['output'])))
    for index, query in enumerate(task['test']):
        lines.append('query%d input=%s' % (index, compact(query['input'])))
    lines.append('The FIRST line of the final response must be ```python. Return exactly one complete '
                 'transform(grid) program, closed by ``` on its own line, with no prose before or after. '
                 'Use the same rule for all queries; do not hard-code their answers. Keep code below 150 lines. '
                 'No imports. Available builtins and safe container methods are described in the system message. '
                 'Use literal integers instead of exponentiation.')
    return [{'role': 'system', 'content': SP.SYSTEM_PROMPT},
            {'role': 'user', 'content': '\n'.join(lines)}]


def build_direct_prompt(task):
    task = SP.task_inputs_only(task)
    lines = ['Infer the transformation rule from every demonstration and apply it to every query.']
    for index, demo in enumerate(task['train']):
        lines.append('demo%d input=%s output=%s' % (index, compact(demo['input']), compact(demo['output'])))
    for index, query in enumerate(task['test']):
        lines.append('query%d input=%s' % (index, compact(query['input'])))
    keys = ['query%d' % index for index in range(len(task['test']))]
    lines.append('Return exactly one JSON object with exactly these keys: %s. '
                 'Each value must be the complete output grid for that query: a rectangular array '
                 'of arrays of integers from 0 through 9, with height and width from 1 through 30. '
                 'Include every row and cell; no ellipses, code, comments, or prose.' % compact(keys))
    return [{'role': 'system', 'content': 'Solve the grid transformation puzzle using the demonstrations. '
             'Give the requested JSON grids as your final response.'},
            {'role': 'user', 'content': '\n'.join(lines)}]


def direct_context_budget(messages, requested_generation, requested_thinking, deadline):
    """Count the server's complete chat prompt; generation already includes thinking."""
    budget = {'requested_max_tokens': requested_generation,
              'requested_thinking_token_budget': requested_thinking,
              'prompt_tokens': None, 'max_model_len': P.MAX_MODEL_LEN,
              'context_margin_tokens': DIRECT_CONTEXT_MARGIN,
              'requested_final_output_space_tokens': requested_generation - requested_thinking,
              'effective_max_tokens': 0, 'effective_thinking_token_budget': 0,
              'status': 'tokenization_failed'}
    try:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError('start_deadline_before_tokenize')
        payload = {'model': 'qwen', 'messages': messages, 'add_generation_prompt': True,
                   'add_special_tokens': False,
                   'chat_template_kwargs': {'enable_thinking': True, 'reasoning_effort': 'xhigh'}}
        request = urllib.request.Request('http://127.0.0.1:%s/tokenize' % P.PORT,
                                         data=compact(payload).encode(),
                                         headers={'Content-Type': 'application/json'}, method='POST')
        with urllib.request.urlopen(request, timeout=min(30.0, remaining)) as response:
            tokenized = json.load(response)
        count, server_limit = tokenized.get('count'), tokenized.get('max_model_len')
        if type(count) is not int or count <= 0 or type(server_limit) is not int or server_limit <= 0:
            raise ValueError('invalid tokenize count/max_model_len')
        limit = min(P.MAX_MODEL_LEN, server_limit)
        available = limit - count - DIRECT_CONTEXT_MARGIN
        budget.update(prompt_tokens=count, max_model_len=limit,
                      available_generation_tokens=max(0, available))
        if available < 2:
            budget['status'] = 'no_context_room_for_generation'
            return budget
        generation = min(requested_generation, available)
        # Keep the requested final-grid space when clipping, then reduce thinking.
        # One thinking token remains for the unchanged enable_thinking template.
        thinking = min(requested_thinking, max(1, generation - (requested_generation - requested_thinking)))
        budget.update(effective_max_tokens=generation, effective_thinking_token_budget=thinking,
                      effective_final_output_space_tokens=generation - thinking,
                      status='trimmed' if generation < requested_generation else 'unchanged')
    except Exception as exc:
        # No guessed count: an uncounted prompt is never sent for generation.
        budget['error'] = '%s:%s' % (type(exc).__name__, str(exc)[:250])
    return budget


def parse_direct(rec, res, task):
    content = res.get('content') or ''
    completion_sha = hashlib.sha256(content.encode()).hexdigest()
    failure = {'status': 'direct_invalid', 'code': None, 'gen_ok': False,
               'query_results': [], 'completion_sha256': completion_sha}
    if (res.get('finish') not in ('stop', 'length') or rec.get('client_error') or
            rec.get('cancelled_by_client')):
        return dict(failure, status='generation_incomplete',
                    validation_error=rec.get('client_error') or str(res.get('finish')))

    def unique_object(pairs):
        value = {}
        for key, item in pairs:
            if key in value:
                raise ValueError('duplicate JSON key')
            value[key] = item
        return value

    def reject_constant(_value):
        raise ValueError('nonstandard JSON constant')

    try:
        text = content.strip()
        if text.startswith('```'):
            lines = text.splitlines()
            if len(lines) < 3 or lines[0].strip() not in ('```json', '```') or lines[-1].strip() != '```':
                raise ValueError('expected one complete JSON fence')
            text = '\n'.join(lines[1:-1])
        value = json.loads(text, object_pairs_hook=unique_object, parse_constant=reject_constant)
        keys = ['query%d' % index for index in range(len(task['test']))]
        if not isinstance(value, dict) or len(value) != len(keys) or set(value) != set(keys):
            raise ValueError('query key set mismatch')
        results = [{'status': 'ok', 'grid': SL.valid_grid(value[key]), 'execution_performed': False}
                   for key in keys]
        return {'status': 'length_with_complete_json' if res.get('finish') == 'length' else 'direct_ready',
                'code': None, 'gen_ok': True, 'query_results': results, 'completion_sha256': completion_sha}
    except (ValueError, TypeError) as exc:
        return dict(failure, validation_error=str(exc)[:300])


def parse_program(rec, res):
    if res.get('finish') == 'stop' and not rec.get('client_error') and not rec.get('cancelled_by_client'):
        parsed = SP.split_program_response(res.get('content') or '', thinking_mode='direct')
        parsed['gen_ok'] = parsed.get('status') == 'program_ready'
        return parsed
    return {'status': 'generation_incomplete', 'code': None, 'gen_ok': False,
            'validation_error': rec.get('client_error') or str(res.get('finish'))}


def near_program(evaluation):
    """Feedback-vs-restart rule after CPU search; not a CPU admission filter."""
    m = evaluation.get('metrics') or evaluation.get('demo_metrics') or {}
    if not evaluation.get('code') or m.get('all_demos_pass'):
        return False
    return (m.get('demo_exact_rate', 0) >= 0.5 or
            (m.get('shape_rate', 0) >= 0.75 and
             m.get('foreground_accuracy', 0) >= 0.80 and
             m.get('changed_accuracy', 0) >= 0.80))


def resolve_challenges(input_root, mode):
    if mode not in ('public_eval', 'competition_submit'):
        raise ValueError('unknown input mode')
    name = 'arc-agi_test_challenges.json' if mode == 'competition_submit' else 'arc-agi_evaluation_challenges.json'
    root = Path(input_root)
    # Search only the exact filename; any ambiguity is an error.
    hits = sorted(p for p in root.rglob(name) if p.is_file())
    if len(hits) != 1:
        raise ValueError('%s: expected exactly one %s, found %d' % (mode, name, len(hits)))
    return hits[0]


def rerun_mode(value):
    if value is None or value == '':
        return False
    normalized = value.strip().lower()
    if normalized in ('1', 'true', 'yes'):
        return True
    if normalized in ('0', 'false', 'no'):
        return False
    raise ValueError('unexpected nonempty KAGGLE_IS_COMPETITION_RERUN')


def _gpu_sampler(work):
    samples = []
    while not GPU_SAMPLER_STOP.is_set():
        try:
            r = sh(['nvidia-smi', '--query-gpu=index,memory.used,memory.total,utilization.gpu',
                    '--format=csv,noheader,nounits'], timeout=8)
            with (work / 'gpu-samples.jsonl').open('a') as stream:
                stream.write(compact({'time': time.time(), 'raw': r.stdout.strip()}) + '\n')
            rows = [[float(cell.strip()) for cell in line.split(',')] for line in r.stdout.splitlines() if line.strip()]
            if len(rows) == 4 and all(len(row) == 4 for row in rows):
                samples.append(rows)
                if len(samples) >= 12:
                    log('GPU runtime utilization means=%s memory-used-MiB=%s' % (
                        [round(sum(s[i][3] for s in samples) / len(samples), 1) for i in range(4)],
                        [round(sum(s[i][1] for s in samples) / len(samples)) for i in range(4)]))
                    samples.clear()
        except Exception as exc:
            log('GPU sampler ' + type(exc).__name__)
        GPU_SAMPLER_STOP.wait(5)


def main():
    global GEN_MAX, THINK_MAX
    ap = argparse.ArgumentParser()
    ap.add_argument('--work', default='/kaggle/working/vprobe')
    ap.add_argument('--challenges', required=True)
    ap.add_argument('--experiment', choices=['paired', 'submit'], default='paired')
    ap.add_argument('--policy', choices=['M0', 'M1'], default='M1')
    ap.add_argument('--session-seconds', type=float, default=10800)
    ap.add_argument('--session-start-monotonic', type=float)
    ap.add_argument('--session-start-wall', type=float)
    ap.add_argument('--pip-python')
    ap.add_argument('--submission', default='/kaggle/working/submission.json')
    ap.add_argument('--owner', required=True)
    ap.add_argument('--gen-max', type=int, default=GEN_MAX)
    ap.add_argument('--thinking-max', type=int, default=THINK_MAX)
    ap.add_argument('--mtp-tokens', type=int, choices=[0, 2], default=0)
    ap.add_argument('--generation-mode', choices=['program', 'direct'], default='program')
    ap.add_argument('--direct-attempts', type=int, choices=[1, 2], default=2)
    ap.add_argument('--line-complement', action='store_true',
                    help='Add the fixed demo-selected CPU line rule beside one direct prediction')
    ap.add_argument('--frozen-initial', help='Research-only initial programs; never read in competition mode')
    ap.add_argument('--cohort', help='Explicit research cohort input path')
    ap.add_argument('--p0-only', action='store_true', help='Research-only single P0 call per task (DeepP0)')
    args = ap.parse_args()
    if not 0 < args.thinking_max < args.gen_max <= 49152:
        raise ValueError('invalid generation/thinking budget')
    if args.frozen_initial and args.generation_mode != 'program':
        raise ValueError('frozen initial programs require program generation mode')
    if args.frozen_initial and args.experiment != 'paired':
        raise ValueError('frozen initial programs are research-only')
    if args.p0_only and (args.experiment != 'paired' or not args.cohort or args.frozen_initial):
        raise ValueError('p0-only requires a paired research cohort and excludes frozen-initial')
    if args.p0_only and Path(args.cohort).resolve() != Path(args.challenges).resolve():
        raise ValueError('p0-only cohort must be the supplied challenges')
    if args.line_complement and (args.generation_mode != 'direct' or args.direct_attempts != 1
                                or args.p0_only or args.frozen_initial):
        raise ValueError('line-complement requires direct-attempts=1, without p0-only or frozen-initial')
    GEN_MAX, THINK_MAX = args.gen_max, args.thinking_max
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    t0_mono = args.session_start_monotonic or time.monotonic()
    t0_wall = args.session_start_wall or time.time()
    session_deadline = t0_mono + args.session_seconds
    STATE['setup_deadline'] = min(t0_mono + INSTALL_BUDGET_S, session_deadline - NEW_REQUEST_MARGIN_S)
    run = {'status': 'preparing', 'start_monotonic': t0_mono, 'start_wall': t0_wall,
           'owner': args.owner, 'experiment': args.experiment, 'policy': args.policy,
           'p0_only': args.p0_only, 'cohort': args.cohort, 'generation_mode': args.generation_mode,
           'result_label': ('DeepDirect' if args.generation_mode == 'direct' else 'DeepP0') if args.p0_only else args.experiment,
           'config': {'tp': 4, 'concurrency': 8, 'generation_max': GEN_MAX, 'thinking_max': THINK_MAX,
                      'generation_mode': args.generation_mode,
                      'line_complement': args.line_complement,
                      'direct_attempts': args.direct_attempts,
                      'direct_context_margin_tokens': DIRECT_CONTEXT_MARGIN if args.generation_mode == 'direct' else None,
                      'effort': 'xhigh', 'sampling': P.SAMPLING, 'seeds': SEEDS,
                      'mtp_tokens': args.mtp_tokens,
                      'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2} if args.mtp_tokens else None,
                      'mamba_cache_mode': 'align' if args.mtp_tokens else None,
                      'near_rule': {'exact_min': 0.5, 'shape_min': 0.75, 'foreground_min': .8, 'changed_min': .8},
                      'cpu_scope': ('disabled for direct grids' if args.generation_mode == 'direct' else
                                    'M1 admitted P0/F/R with supported demo-only variants; no near filter'),
                      'cpu_max_variants': 32, 'cpu_max_seconds': 30,
                      'vllm_wheel_sha256': VLLM_WHEEL_SHA256},
           'budgets': {'session_s': args.session_seconds, 'stop_new_margin_s': 600, 'stop_generation_margin_s': 300},
           'source_sha256': {p.name: sha256_file(p) for p in Path(__file__).parent.glob('*.py')},
           'calls': [], 'tasks': {}, 'waves': []}
    STATE.update(work=work, run=run)
    signal.signal(signal.SIGTERM, on_term)
    signal.signal(signal.SIGINT, on_term)
    challenges = {k: SP.task_inputs_only(v) for k, v in json.loads(Path(args.challenges).read_text()).items()}
    run['challenges_sha256'] = sha256_file(args.challenges)
    run['n_tasks'] = len(challenges)
    run['n_queries'] = sum(len(t['test']) for t in challenges.values())
    policies = ['M0', 'M1'] if args.experiment == 'paired' else [args.policy]
    submissions, query_map = {}, None
    pools = {policy: {} for policy in policies}
    for policy in policies:
        submissions[policy], query_map = SL.initial_submission(challenges)
        SL.atomic_json(work / ('submission-%s.json' % policy), submissions[policy])
    SL.atomic_json(args.submission, submissions[args.policy if args.policy in policies else policies[0]])
    save(work, run)
    log('input tasks=%d queries=%d sha=%s' % (run['n_tasks'], run['n_queries'], run['challenges_sha256']))
    P.record_versions(run)
    if args.generation_mode == 'program':
        pre = SP.execute_grid('def transform(grid):\n    return [row[:] for row in grid]', [[1]],
                              wall_seconds=2.0, memory_mb=512)
        run['executor_preinstall'] = pre
        save(work, run)
        if pre.get('status') != 'ok' or pre.get('grid') != [[1]]:
            return finish('executor_infra_failed_before_install', 2)
    aa_fixture = Path(__file__).parent / 'aa-demo-preflight.json'
    if args.generation_mode == 'program' and aa_fixture.is_file():
        fixture = json.loads(aa_fixture.read_text())
        check = LR.repair_program(fixture['code'], fixture['demos'],
                                  deadline=STATE['setup_deadline'], max_variants=32, max_seconds=30)
        SL.atomic_json(work / 'aa-demo-mechanism.json', check)
        run['aa_demo_preflight'] = {'fixture_sha256': sha256_file(aa_fixture),
             'baseline_matches': check.get('baseline', {}).get('metrics', {}).get('demo_matches'),
             'best_matches': (check.get('best') or {}).get('metrics', {}).get('demo_matches'),
             'variants_tried': check.get('variants_tried'), 'elapsed_s': check.get('elapsed_s')}
        save(work, run)
        if run['aa_demo_preflight']['best_matches'] != len(fixture['demos']):
            return finish('cpu_repair_mechanism_failed_before_install', 2)
    py = sys.version_info
    if (py.major, py.minor) != (3, 12):
        return finish(f"env_incompatible_python_{py.major}_{py.minor}")
    try:
        model_root = P.one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        wh = P.one("/kaggle/input/**/qwen38-vllm0272-cu130-wheelhouse-v1/**/requirements-qwen38-runtime.txt").parent
    except FileNotFoundError as exc:
        run["mount_error"] = str(exc)[:200]
        return finish("mount_missing")
    site = work / "site"
    site.mkdir(exist_ok=True)
    wheels = wh / "wheels"
    req = wh / "requirements-qwen38-runtime.txt"
    wheel_path = wheels / VLLM_WHEEL_NAME
    run['model_root'] = str(model_root)
    run['model_small_files_sha256'] = {p.name: sha256_file(p) for p in
        (model_root / 'config.json', model_root / 'tokenizer_config.json', model_root / 'chat_template.jinja') if p.is_file()}
    run['runtime_requirements_sha256'] = sha256_file(req)
    if not wheels.is_dir() or not req.is_file() or not wheel_path.is_file():
        return finish("vllm_wheel_missing")
    STATE["phase"] = "wheelcheck"
    digest = sha256_file(wheel_path)
    run["wheel_hash"] = {"status": "match" if digest == VLLM_WHEEL_SHA256 else "mismatch",
                         "observed": digest[:16]}
    save(work, run)
    if digest != VLLM_WHEEL_SHA256:
        return finish("vllm_wheel_hash_mismatch")
    log("wheel hash match (sync)")

    pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    if pip_probe.returncode:
        ens = sh([sys.executable, "-m", "ensurepip", "--default-pip"], timeout=180)
        (work / "ensurepip.log").write_text((ens.stdout + "\n---STDERR---\n" + ens.stderr)[-4000:])
        pip_probe = sh([sys.executable, "-m", "pip", "--version"], timeout=60)
    use_fallback = pip_probe.returncode != 0
    if use_fallback and not args.pip_python:
        return finish("no_pip_for_312_and_no_fallback")

    def _base():
        if use_fallback:
            return [args.pip_python, "-m", "pip", "install", "--python-version", "3.12",
                    "--only-binary", ":all:", "--no-index"]
        return [sys.executable, "-m", "pip", "install", "--no-index"]
    STATE["phase"] = "pip-req"
    try:
        log("phase: requirements install start")
        ins = sh(_base() + ["--find-links", str(wheels), "--requirement", str(req),
                            "--target", str(site)], timeout=INSTALL_BUDGET_S)
        log(f"phase: requirements install rc={ins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("install_timeout_over_20min")
    (work / "pip-install.log").write_text(ins.stdout[-8000:] + "\n---STDERR---\n" + ins.stderr[-4000:])
    run["pip_requirements_rc"] = ins.returncode
    save(work, run)
    if ins.returncode:
        return finish("install_failed", detail=(ins.stderr or ins.stdout)[-400:])
    try:
        log("phase: vllm wheel install start")
        vins = sh(_base() + ["--no-deps", "--find-links", str(wheels), str(wheel_path),
                             "--target", str(site)], timeout=600)
        log(f"phase: vllm wheel install rc={vins.returncode}")
    except subprocess.TimeoutExpired:
        return finish("vllm_wheel_install_timeout")
    (work / "pip-vllm.log").write_text(vins.stdout[-4000:] + "\n---STDERR---\n" + vins.stderr[-4000:])
    run["pip_wheel_rc"] = vins.returncode
    save(work, run)
    if vins.returncode:
        return finish("vllm_wheel_install_failed")
    os.environ["PYTHONPATH"] = os.pathsep.join([str(site), os.environ.get("PYTHONPATH", "")]).strip(os.pathsep)
    STATE["phase"] = "import"
    imp = sh([sys.executable, "-c", "import vllm, torch; print(vllm.__version__)"], timeout=300,
             env=dict(os.environ))
    run["import_check"] = {"rc": imp.returncode, "out": (imp.stdout + imp.stderr)[-500:]}
    save(work, run)
    if imp.returncode:
        return finish("import_vllm_failed")
    run["install_s"] = round(time.time() - t0_wall, 1)

    install_deadline = min(t0_mono + INSTALL_BUDGET_S, session_deadline - NEW_REQUEST_MARGIN_S)
    if time.monotonic() >= install_deadline:
        return finish("install_load_over_20min")
    STATE["phase"] = "serve"
    server_extra = (['--speculative-config', compact({'method': 'mtp', 'num_speculative_tokens': 2}),
                     '--mamba-cache-mode', 'align'] if args.mtp_tokens else [])
    run['server_extra_args'] = server_extra
    try:
        proc = P.start_server(model_root, work, extra=server_extra, deadline=install_deadline)
    except Exception as exc:
        return finish("serve_failed", detail=f"{type(exc).__name__}:{exc}")
    STATE["proc"] = proc
    (work / "server-pgid.txt").write_text(str(proc.pid))
    run["install_load_s"] = round(time.time() - t0_wall, 1)
    save(work, run)
    STATE["phase"] = "preflight"
    if args.generation_mode == 'program':
        pre = SP.execute_grid("def transform(grid):\n    return [row[:] for row in grid]", [[1]],
                              wall_seconds=2.0, memory_mb=512)
        run["executor_preflight"] = {"status": pre.get("status"), "grid": pre.get("grid")}
        save(work, run)
        if pre.get("status") != "ok" or pre.get("grid") != [[1]]:
            return finish("executor_infra_failed")
    th = threading.Thread(target=_gpu_sampler, args=(work,), daemon=True)
    th.start()
    STATE['setup_deadline'] = None


    generation_deadline = session_deadline - SAVE_MARGIN_S
    start_deadline = session_deadline - NEW_REQUEST_MARGIN_S
    order = 0
    proposals = {task_id: {} for task_id in challenges}

    def generation_watchdog():
        if not GPU_SAMPLER_STOP.wait(max(0, generation_deadline - time.monotonic())):
            run['generation_watchdog_fired'] = True
            server = STATE.get('proc')
            if server:
                try:
                    os.killpg(server.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
    threading.Thread(target=generation_watchdog, daemon=True).start()

    def persist_policy(policy):
        SL.atomic_json(work / ('submission-%s.json' % policy), submissions[policy])
        if policy == args.policy or len(policies) == 1:
            SL.atomic_json(args.submission, submissions[policy])

    def add_proposal(task_id, name, evaluation, eligible):
        nonlocal order
        evaluation['arm'] = name
        proposals[task_id][name] = evaluation
        run['tasks'].setdefault(task_id, {})[name] = evaluation
        metrics = evaluation.get('metrics') or evaluation.get('demo_metrics') or {}
        for policy in eligible:
            if policy not in pools:
                continue
            for index, result in enumerate(evaluation.get('query_results', [])):
                if result.get('status') != 'ok':
                    continue
                key = '%s_%d' % (task_id, index)
                try:
                    grid = SL.valid_grid(result.get('grid'))
                except (ValueError, TypeError):
                    continue
                candidate = dict(metrics, grid=grid, metrics=metrics, source=name, order=order,
                                 code_sha256=(hashlib.sha256(evaluation['code'].encode()).hexdigest()
                                              if evaluation.get('code') else None))
                if evaluation.get('completion_sha256'):
                    candidate['completion_sha256'] = evaluation['completion_sha256']
                order += 1
                pools[policy].setdefault(key, []).append(candidate)
                a, b = SL.merge_candidate_pool(pools[policy][key])
                SL.update_slots(submissions[policy], query_map, {key: [a, b]})
            persist_policy(policy)
        SL.atomic_json(work / 'candidate-pools.json', pools)
        save(work, run)

    def execute_local_queries(task, variant, cpu_deadline):
        evaluation = dict(variant)
        evaluation['query_results'] = []
        spent = 0.0
        for q in task['test']:
            remaining = min(2.0, 20.0 - spent, cpu_deadline - time.monotonic())
            result = SP.execute_grid(variant['code'], q['input'], wall_seconds=remaining, memory_mb=512)
            spent += result.get('elapsed_s', 0)
            evaluation['query_results'].append(result)
        evaluation['query_execution_seconds'] = spent
        return evaluation

    def apply_local_repair(task_id, arm, base):
        """Cheap supported mutations after any M1 program, using demos only."""
        task = challenges[task_id]
        metrics = base.get('metrics') or base.get('demo_metrics') or {}
        if not base.get('code') or metrics.get('all_demos_pass'):
            return base
        cpu_deadline = min(generation_deadline, time.monotonic() + 30)
        demo_seconds = max(1.0, 30.0 - min(10.0, 4.0 * len(task['test'])))
        repaired = LR.repair_program(base['code'], task['train'], base_evaluation=base,
                                     deadline=cpu_deadline, max_variants=32, max_seconds=demo_seconds)
        run['tasks'][task_id].setdefault('cpu_repairs', {})[arm] = repaired
        ranked = sorted(repaired.get('variants', []), key=SL.candidate_sort_key)
        for index, variant in enumerate(ranked[:2]):
            candidate = execute_local_queries(task, variant, cpu_deadline)
            add_proposal(task_id, 'CPU_%s_%d' % (arm, index), candidate, ['M1'])
        if ranked and SL.candidate_sort_key(ranked[0]) < SL.candidate_sort_key(base):
            return ranked[0]
        return base

    def result_callback(_i, spec, slot):
        task_id, arm = spec['task_id'], spec['arm']
        task = challenges[task_id]
        rec, res = slot['rec'], slot['res']
        rec.update(task_id=task_id, arm=arm)
        if 'token_budget' in spec:
            rec['token_budget'] = spec['token_budget']
            if spec['token_budget']['status'] not in ('unchanged', 'trimmed'):
                rec['client_error'] = 'direct_context_budget:' + spec['token_budget']['status']
        raw_dir = work / 'programs'
        raw_dir.mkdir(exist_ok=True)
        SL.atomic_json(raw_dir / ('%s-%s.reply.json' % (arm, task_id)), {'rec': rec, 'res': res})
        parsed = parse_direct(rec, res, task) if args.generation_mode == 'direct' else parse_program(rec, res)
        rec['parse_status'] = parsed.get('status')
        if args.generation_mode == 'direct':
            rec['completion_sha256'] = parsed['completion_sha256']
            rec['demo_evidence'] = 'unknown'
        run['calls'].append(rec)
        save(work, run)  # Raw completion survives any subsequent executor failure.
        if args.generation_mode == 'direct':
            evaluation = {'code': None, 'status': parsed['status'], 'demo_results': [],
                          'query_results': parsed['query_results'], 'completion_sha256': parsed['completion_sha256'],
                          'execution_performed': False,
                          'metrics': {'demo_evidence': 'unknown', 'demo_total': len(task['train']),
                                      'demo_matches': None, 'all_demos_pass': None}}
            if parsed.get('validation_error'):
                evaluation['validation_error'] = parsed['validation_error']
        elif parsed.get('gen_ok'):
            code = parsed['code']
            (raw_dir / ('%s-%s.code.py' % (arm, task_id))).write_text(code)
            evaluation = SP.evaluate_program(code, task, deadline=generation_deadline)
        else:
            evaluation = {'code': None, 'status': parsed.get('status'), 'demo_results': [],
                          'query_results': [], 'metrics': SL.demo_metrics(task['train'], [])}
        eligible = []
        if arm == 'P0':
            eligible = policies
        elif arm == 'F':
            eligible = ['M1']
        elif arm == 'R':
            eligible = ['M0']
            if run['tasks'].get(task_id, {}).get('route_M1') == 'R':
                eligible.append('M1')
        add_proposal(task_id, arm, evaluation, eligible)
        m = evaluation.get('metrics') or evaluation.get('demo_metrics') or {}
        rec.update(gen_ok=bool(parsed.get('gen_ok')), demo_matches=m.get('demo_matches', 0),
                   all_demos_pass=m.get('all_demos_pass', False),
                   query_legal=sum(r.get('status') == 'ok' for r in evaluation.get('query_results', [])))
        repaired_base = evaluation
        if args.generation_mode == 'program' and 'M1' in eligible and not args.p0_only and parsed.get('gen_ok'):
            repaired_base = apply_local_repair(task_id, arm, evaluation)
        if arm == 'P0' and not args.p0_only and ('M1' in policies or args.generation_mode == 'direct'):
            base = repaired_base
            metrics = base.get('metrics') or base.get('demo_metrics') or {}
            route = (('stop' if args.direct_attempts == 1 else 'R') if args.generation_mode == 'direct' else
                     ('stop' if metrics.get('all_demos_pass') else ('F' if near_program(base) else 'R')))
            run['tasks'][task_id]['route_M1'] = route
            proposals[task_id]['feedback_base'] = base
        save(work, run)
        log('%s %s gen=%s demo=%s/%s query=%s route=%s' %
            (arm, task_id, parsed.get('gen_ok'), m.get('demo_matches', 0), len(task['train']),
             rec['query_legal'], run['tasks'].get(task_id, {}).get('route_M1')))
        if arm == 'P0' and not args.p0_only:
            if args.generation_mode == 'direct' and args.direct_attempts == 1:
                return None
            if args.generation_mode == 'direct' and spec['token_budget']['status'] not in ('unchanged', 'trimmed'):
                return None
            route = run['tasks'].get(task_id, {}).get('route_M1')
            arms = []
            if 'M0' in policies or route == 'R':
                arms.append('R')
            if 'M1' in policies and route == 'F':
                arms.append('F')
            if task_ids.index(task_id) % 2:
                arms.reverse()
            return [spec_for(task_id, followup) for followup in arms]
        return None

    def spec_for(task_id, arm):
        messages = (build_direct_prompt(challenges[task_id]) if args.generation_mode == 'direct' else
                    build_program_prompt(challenges[task_id]))
        if arm == 'F':
            base = proposals[task_id]['feedback_base']
            feedback = SP.demo_feedback(challenges[task_id]['train'], base['demo_results'], 3000)
            instruction = ('Correct the rule and return the complete program as exactly one closed Python block. '
                           'Query answers are unavailable. Use every demonstration.')
            if (args.frozen_initial and proposals[task_id].get('R', {}).get('reused_historical_generation') and
                    (base.get('metrics') or base.get('demo_metrics') or {}).get('all_demos_pass')):
                instruction = ('All demonstrations pass for this program. Examine remaining ambiguity and propose '
                               'another rule supported by every demonstration. Preserve the existing candidate; '
                               'the new program will be considered alongside it. Return the complete program as '
                               'exactly one closed Python block. Query answers are unavailable.')
            messages += [{'role': 'assistant', 'content': '```python\n' + base['code'] + '\n```'},
                         {'role': 'user', 'content': 'Real execution feedback for exactly the program above:\n' + feedback +
                          '\n' + instruction}]
        spec = {'task_id': task_id, 'arm': arm, 'messages': messages, 'effort': 'xhigh',
                'seed': SEEDS[arm], 'max_tokens': GEN_MAX, 'thinking_token_budget': THINK_MAX,
                'enable_thinking': True, 'request_id': '%s-%s' % (arm, task_id),
                'timeout_s': max(1, generation_deadline - time.monotonic()),
                'start_deadline': start_deadline, 'deadline': generation_deadline}
        if args.generation_mode == 'direct':
            budget = direct_context_budget(messages, GEN_MAX, THINK_MAX, start_deadline)
            spec.update(max_tokens=budget['effective_max_tokens'],
                        thinking_token_budget=budget['effective_thinking_token_budget'], token_budget=budget)
            if budget['status'] not in ('unchanged', 'trimmed'):
                spec['start_deadline'] = 0  # Existing transport records a skipped slot; no generation HTTP.
            run.setdefault('direct_token_budgets', {})[spec['request_id']] = budget
            save(work, run)
        return spec

    def wave(name, specs):
        if not specs or time.monotonic() >= start_deadline:
            run['waves'].append({'name': name, 'status': 'skipped_deadline', 'n': len(specs)})
            save(work, run)
            return
        result = P.chat_wave(specs, client_concurrency=8, on_slot_done=result_callback)
        callbacks_failed = []
        for spec, slot in zip(specs, result['calls']):
            if slot['rec'].get('callback_error'):
                callbacks_failed.append({'task_id': spec['task_id'], 'arm': spec['arm'],
                                         'error': slot['rec']['callback_error']})
        run['waves'].append({k: v for k, v in result.items() if k != 'calls'} | {'name': name, 'n': len(specs)})
        run.setdefault('callback_errors', []).extend(callbacks_failed)
        save(work, run)

    if args.line_complement:
        line_started = time.monotonic()
        line_report = {'enabled': True, 'status': 'running', 'tasks': {},
                       'admitted_task_count': 0, 'supported_query_count': 0,
                       'errors': [], 'query_gold_used': False}
        run['line_complement'] = line_report
        try:
            import aligned_line_solver as line_solver
            line_report['solver_source_sha256'] = sha256_file(line_solver.__file__)
            for task_id, task in challenges.items():
                try:
                    solved = line_solver.solve_task(task)
                    demo_results = [
                        {'status': 'ok' if result['supported'] else 'unsupported',
                         'grid': result['prediction']}
                        for result in solved['demo_results']]
                    metrics = SL.demo_metrics(task['train'], demo_results)
                    admitted = bool(solved['admitted'] and metrics['all_demos_pass'])
                    detail = {'demo_total': metrics['demo_total'],
                              'demo_matches': metrics['demo_matches'],
                              'changed_demo_count': solved['changed_demo_count'],
                              'admitted': admitted, 'supported_queries': 0}
                    line_report['tasks'][task_id] = detail
                    if not admitted:
                        continue
                    line_report['admitted_task_count'] += 1
                    query_results = [{'status': 'unsupported'} for _ in task['test']]
                    for query in solved['query_predictions']:
                        if query['supported']:
                            grid = SL.valid_grid(query['grid'])
                            query_results[query['query_index']] = {'status': 'ok', 'grid': grid}
                            detail['supported_queries'] += 1
                    line_report['supported_query_count'] += detail['supported_queries']
                    if detail['supported_queries']:
                        evaluation = {'code': solved['candidate_code'], 'status': 'cpu_line_ready',
                                      'demo_results': demo_results, 'query_results': query_results,
                                      'metrics': metrics, 'execution_performed': 'trusted_cpu_solver'}
                        # Exactly one CPU grid/query plus at most one direct P0 grid/query.
                        # Deduplication therefore leaves <=2 grids and cannot evict P0.
                        add_proposal(task_id, 'CPU_line', evaluation, policies)
                except Exception as exc:
                    line_report['errors'].append({'task_id': task_id,
                                                  'error': '%s: %s' % (type(exc).__name__, str(exc)[:400])})
            line_report['status'] = 'complete_with_errors' if line_report['errors'] else 'complete'
        except Exception as exc:
            line_report['status'] = 'unavailable'
            line_report['errors'].append({'error': '%s: %s' % (type(exc).__name__, str(exc)[:400])})
        line_report['elapsed_cpu_wall_seconds'] = round(time.monotonic() - line_started, 6)
        save(work, run)
        log('CPU line complement admitted=%d queries=%d seconds=%.3f status=%s' %
            (line_report['admitted_task_count'], line_report['supported_query_count'],
             line_report['elapsed_cpu_wall_seconds'], line_report['status']))

    # Shortest task inputs first, stable ID tie-break; all tasks get P0 before extras.
    task_ids = sorted(challenges, key=lambda k: (len(compact(challenges[k])), k))
    if args.frozen_initial:
        initial = json.loads(Path(args.frozen_initial).read_text())
        if set(initial) != set(task_ids):
            raise ValueError('frozen initial task set mismatch')
        run['frozen_initial_sha256'] = sha256_file(args.frozen_initial)
        run['frozen_initial_metadata'] = {
            k: {a: b for a, b in row.items() if a not in ('code', 'restart_code')}
            for k, row in initial.items()}
        run['historical_initial_completion_tokens'] = sum(row['initial_completion_tokens'] for row in initial.values())
        reused_restart_tasks = [task_id for task_id, row in initial.items() if 'restart_code' in row]
        run['historical_restart_tokens'] = sum(initial[k]['restart_completion_tokens'] for k in reused_restart_tasks)
        short_specs = []
        for index, task_id in enumerate(task_ids):
            base = SP.evaluate_program(initial[task_id]['code'], challenges[task_id], deadline=generation_deadline)
            add_proposal(task_id, 'P0', base, policies)
            run['tasks'][task_id]['route_M1'] = 'F'
            if task_id in reused_restart_tasks:
                restart_code = initial[task_id]['restart_code']
                if restart_code:
                    restart = SP.evaluate_program(restart_code, challenges[task_id], deadline=generation_deadline)
                else:
                    restart = {'code': None, 'status': 'reused_restart_without_program', 'demo_results': [],
                               'query_results': [], 'metrics': SL.demo_metrics(challenges[task_id]['train'], [])}
                restart['reused_historical_generation'] = True
                add_proposal(task_id, 'R', restart, ['M0'])
                base = apply_local_repair(task_id, 'P0', base)
                arms = ['F']
            else:
                arms = ['F', 'R'] if index % 2 == 0 else ['R', 'F']
            proposals[task_id]['feedback_base'] = base
            short_specs.extend(spec_for(task_id, arm) for arm in arms)
        wave('full_feedback_vs_reused_restart' if reused_restart_tasks else
             'short_mechanical_feedback_vs_restart', short_specs)
    else:
        # Followups are appended behind the original P0 queue, overlapping its tail.
        wave(('DeepDirect' if args.generation_mode == 'direct' else 'DeepP0') if args.p0_only else 'P0_and_secondary',
             [spec_for(k, 'P0') for k in task_ids])
    run['coverage'] = {policy: SL.coverage_report(submissions[policy], query_map, updated_keys=pools[policy].keys())
                       for policy in policies}
    run['final_validate'] = {policy: SL.validate_submission(work / ('submission-%s.json' % policy), submissions[policy])
                             for policy in policies}
    run['new_output_tokens_total'] = sum((rec.get('usage') or {}).get('completion_tokens', 0) for rec in run['calls'])
    run['calls_missing_usage'] = sum(not rec.get('usage') for rec in run['calls'])
    run['planned_p0_tasks'] = len(task_ids)
    p0_calls = [rec for rec in run['calls'] if rec.get('arm') == 'P0']
    run['p0_funnel'] = {
        'planned': len(task_ids), 'returned_slots': len(p0_calls),
        'http_issued': sum('issued_monotonic' in rec.get('events', {}) for rec in p0_calls),
        'generation_finished': sum(rec.get('finish_reason') in ('stop', 'length') and not rec.get('client_error') for rec in p0_calls),
        'admitted_programs': sum(bool(rec.get('gen_ok')) for rec in p0_calls) if args.generation_mode == 'program' else 0,
        'admitted_direct_predictions': sum(bool(rec.get('gen_ok')) for rec in p0_calls) if args.generation_mode == 'direct' else 0,
        'all_demos_pass': sum(bool(rec.get('all_demos_pass')) for rec in p0_calls)}
    run['completed_p0_tasks'] = run['p0_funnel']['generation_finished']
    run['policy_costs'] = {}
    for policy in policies:
        included = []
        for rec in run['calls']:
            arm, task_id = rec['arm'], rec['task_id']
            if arm == 'P0' or (policy == 'M0' and arm == 'R') or (
                policy == 'M1' and arm == run['tasks'].get(task_id, {}).get('route_M1')):
                included.append(rec)
        run['policy_costs'][policy] = {
            'logical_calls': len(included),
            'completion_tokens': sum((r.get('usage') or {}).get('completion_tokens', 0) for r in included),
            'request_seconds_sum_not_wall': sum(r.get('wall_s', 0) for r in included),
            'cpu_repair_seconds': sum(
                repair.get('elapsed_s', 0) for task in run['tasks'].values()
                for repair in task.get('cpu_repairs', {}).values()) if policy == 'M1' else 0}
        if args.frozen_initial:
            run['policy_costs'][policy]['historical_initial_tokens'] = run['historical_initial_completion_tokens']
            restart_tokens = run['historical_restart_tokens'] if policy == 'M0' else 0
            run['policy_costs'][policy]['historical_restart_tokens'] = restart_tokens
            run['policy_costs'][policy]['logical_calls_including_reused'] = (
                len(included) + len(initial) + (len(reused_restart_tasks) if policy == 'M0' else 0))
            run['policy_costs'][policy]['deployment_tokens_including_initial'] = (
                run['historical_initial_completion_tokens'] + restart_tokens +
                run['policy_costs'][policy]['completion_tokens'])
    run['query_gold_used'] = False
    ok = all(result[0] for result in run['final_validate'].values()) and not run.get('callback_errors')
    partial = ((not args.frozen_initial and run['completed_p0_tasks'] < len(task_ids)) or
               any(rec.get('client_error') for rec in run['calls']) or
               any(w.get('status') == 'skipped_deadline' and w.get('n') for w in run['waves']))
    run['finished_wall'] = time.time()
    return finish(('partial_with_evidence' if partial else 'complete_with_evidence') if ok else 'execution_incomplete',
                  0 if ok else 2)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except Exception as exc:
        if STATE.get('run'):
            STATE['run']['traceback'] = traceback.format_exc()[-6000:]
        finish('fatal', 2, '%s: %s' % (type(exc).__name__, exc))
        raise
