#!/usr/bin/env python3
"""Package a private manually launched NVARC run from an external local source directory.

This file contains orchestration only. Solver sources and generated notebooks stay
outside the repository. Packaging neither starts a GPU nor creates a competition submission.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib

ROOT = Path(__file__).resolve().parents[2]
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868'
DEFAULT_SOURCE = '/private/tmp/arc2-022-nvarc-work/src'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--mode', choices=['research', 'final'], default='research')
    parser.add_argument('--source-dir', default=DEFAULT_SOURCE)
    parser.add_argument('--out', required=True)
    parser.add_argument('--cohort', default='experiments/020/evaluation-e20/challenges.json')
    parser.add_argument('--session-seconds', type=int, default=6000)
    parser.add_argument('--task-seconds', type=int, default=660)
    parser.add_argument('--save-seconds', type=int, default=180)
    parser.add_argument('--ordinary-seconds', type=int, default=2700)
    parser.add_argument('--ordinary-task-seconds', type=int, default=1200)
    parser.add_argument('--ordinary-ids', nargs='+', default=['136b0064', '16de56c4', '4e34c42c', 'd8e07eb2'])
    parser.add_argument('--rerun-seconds', type=int, default=41400)
    parser.add_argument('--rerun-task-cap', type=int, default=1200)
    args = parser.parse_args()
    source = Path(args.source_dir).resolve()
    out = Path(args.out).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated notebook must stay outside repository; use the private temporary work directory')
    if args.mode == 'final' and not 1800 <= args.rerun_seconds <= 41400:
        raise ValueError('Final rerun total must be within 1800..41400 seconds')
    if args.session_seconds <= args.save_seconds + 60 or args.task_seconds <= 60:
        raise ValueError('Research budget leaves no solver/scoring time')
    out.mkdir(parents=True, exist_ok=False)
    blobs = {}
    for path in sorted(source.glob('*.py')):
        raw = path.read_bytes()
        ast.parse(raw, filename=str(path))
        blobs[path.name] = base64.b64encode(raw).decode()
    needed = {'arc_loader.py', 'arc_decoder.py', 'arc_solver.py', 'copy_model.py', 'run_research.py', 'starter.py'}
    if not needed <= set(blobs):
        raise ValueError('Missing solver source modules: ' + repr(sorted(needed - set(blobs))))
    data_identity = None
    if args.mode == 'research':
        cohort = (ROOT / args.cohort).resolve()
        raw = cohort.read_bytes()
        data = json.loads(raw)
        if not data or any('output' in q for task in data.values() for q in task['test']):
            raise ValueError('Research cohort must contain no query outputs')
        blobs['challenges.json'] = base64.b64encode(raw).decode()
        data_identity = {'path': str(cohort), 'sha256': digest(raw), 'tasks': len(data),
                         'queries': sum(len(task['test']) for task in data.values())}
    hashes = {name: digest(base64.b64decode(raw)) for name, raw in blobs.items()}
    payload = base64.b64encode(zlib.compress(json.dumps(blobs).encode(), 9)).decode()
    config = {'mode': args.mode, 'research_seconds': args.session_seconds,
              'research_task_seconds': args.task_seconds, 'research_save_seconds': args.save_seconds,
              'ordinary_seconds': args.ordinary_seconds, 'ordinary_task_seconds': args.ordinary_task_seconds,
              'ordinary_ids': args.ordinary_ids, 'rerun_seconds': args.rerun_seconds,
              'rerun_task_cap': args.rerun_task_cap}
    code = "import time\nnotebook_started = time.time()\nprint('NOTEBOOK_START', notebook_started, flush=True)\n"
    code += "import base64, hashlib, json, os, subprocess, sys, zlib\nfrom pathlib import Path\n"
    code += 'config = ' + repr(config) + '\nexpected = ' + repr(hashes) + '\npayload = ' + repr(payload) + '\n'
    code += '''source = Path('/kaggle/working/022-code')
source.mkdir(exist_ok=False)
blobs = json.loads(zlib.decompress(base64.b64decode(payload)))
for name, encoded in blobs.items():
    raw = base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest() != expected[name]:
        raise RuntimeError('Source identity mismatch: ' + name)
    (source / name).write_bytes(raw)
(source / 'source-manifest.json').write_text(json.dumps(expected, indent=2))
lookup_started = time.time()
known = Path('/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch')
roots = set()
if (known / 'unsloth/__init__.py').is_file():
    roots.add(str(known.resolve()))
else:
    def utility_name(path):
        return 'pip_install_unsloth_flash_patch' in str(path).replace('-', '_')
    for entry in sys.path:
        path = Path(entry or '.')
        if utility_name(path) and (path / 'unsloth/__init__.py').is_file():
            roots.add(str(path.resolve()))
    if not roots:
        for mount in [Path('/kaggle/usr/lib'), Path('/kaggle/input')]:
            if mount.is_dir():
                for path in mount.rglob('unsloth/__init__.py'):
                    if utility_name(path):
                        roots.add(str(path.parent.parent.resolve()))
print(json.dumps(dict(event='utility_lookup_done', elapsed_s=time.time()-lookup_started,
                      roots=sorted(roots))), flush=True)
if len(roots) != 1:
    raise RuntimeError('Expected exactly one original Unsloth utility root')
raw_rerun = os.environ.get('KAGGLE_IS_COMPETITION_RERUN', '').strip().lower()
if raw_rerun in {'1', 'true', 'yes'}:
    rerun = True
elif raw_rerun in {'', '0', 'false', 'no'}:
    rerun = False
else:
    raise RuntimeError('Unexpected competition rerun flag: ' + repr(raw_rerun))
if config['mode'] == 'research':
    if rerun:
        raise RuntimeError('Research package cannot be competition submitted')
    mode, seconds = 'research', config['research_seconds']
    task_seconds, save_seconds, task_policy = config['research_task_seconds'], config['research_save_seconds'], 'fixed'
elif rerun:
    mode, seconds = 'hidden', config['rerun_seconds']
    task_seconds, save_seconds, task_policy = config['rerun_task_cap'], 300, 'fair'
else:
    mode, seconds = 'ordinary', config['ordinary_seconds']
    task_seconds, save_seconds, task_policy = config['ordinary_task_seconds'], 180, 'fixed'
env = dict(os.environ)
env.update(PYTHONPATH=os.pathsep.join([next(iter(roots)), str(source), env.get('PYTHONPATH', '')]),
           PYTHONUNBUFFERED='1', PYTHONHASHSEED='42')
boot = dict(notebook_started=notebook_started, python=sys.version, mode=mode, seconds=seconds,
            task_seconds=task_seconds, task_policy=task_policy, save_seconds=save_seconds,
            utility_root=next(iter(roots)), source_sha256=expected,
            reference='Failed in AIMO V1 / scriptVersionId312063593')
(source / 'boot.json').write_text(json.dumps(boot, indent=2))
print(json.dumps(dict(event='boot', **boot)), flush=True)
command = [sys.executable, '-u', str(source / 'run_research.py'),
           '--notebook-start', str(notebook_started), '--seconds', str(seconds),
           '--task-seconds', str(task_seconds), '--save-seconds', str(save_seconds),
           '--mode', mode, '--task-policy', task_policy]
if mode == 'ordinary':
    command += ['--ordinary-ids'] + config['ordinary_ids']
remaining = notebook_started + seconds + 30 - time.time()
if remaining <= 0:
    raise RuntimeError('No time remains after notebook setup')
process = subprocess.Popen(command, env=env, start_new_session=True)
try:
    result = process.wait(timeout=remaining)
except subprocess.TimeoutExpired:
    import signal
    state_path = Path('/kaggle/working/022-run/run.json')
    if state_path.is_file():
        try:
            worker_group = json.loads(state_path.read_text()).get('worker_pgid')
            if worker_group:
                os.killpg(int(worker_group), signal.SIGKILL)
        except (OSError, ValueError, TypeError):
            pass
    os.killpg(process.pid, signal.SIGKILL)
    process.wait(timeout=15)
    raise RuntimeError('Outer notebook deadline killed the solver group')
print('NVARC_RETURN_CODE', result, flush=True)
if result:
    raise RuntimeError('Inspect 022-run/run.json and worker.log')
if not Path('/kaggle/working/submission.json').is_file():
    raise RuntimeError('Expected final submission.json')
'''
    ast.parse(code)
    title = 'ARC2 022 NVARC ' + ('E20' if args.mode == 'research' else 'full submission')
    filename = args.slug + '.ipynb'
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.11.13'},
                             'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False, 'accelerator': 'nvidiaL4'}},
                'cells': [{'id': 'scope', 'cell_type': 'markdown', 'metadata': {},
                           'source': ['# ' + title + '\n',
                                      'Private runtime derived from exact Failed in AIMO V1 (312063593).\n',
                                      'Original 4B grid formatter, r256 one-epoch TTT, augmentations, DFS and kgmon final two candidates.\n',
                                      'Local SSD checkpoint staging, serialized complete model loading and explicit wall budgets are recorded.\n',
                                      'Research: fixed E20. Final package: ordinary save actually solves four public tasks; rerun reads all real test tasks.\n']},
                          {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': code.splitlines(keepends=True)}]}
    metadata = {'id': args.owner + '/' + args.slug, 'title': title, 'code_file': filename,
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'competition_sources': ['arc-prize-2026-arc-agi-2'], 'dataset_sources': [],
                'kernel_sources': ['sorokin/pip-install-unsloth-flash-patch'],
                'model_sources': ['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'],
                'docker_image': IMAGE, 'machine_shape': 'NvidiaL4'}
    manifest = {'mode': args.mode, 'notebook_ref': metadata['id'], 'source_dir': str(source),
                'notebook_path': str(out / filename), 'reference_scriptVersionId': 312063593,
                'recovered_from_git_ref': '6a119e9', 'data': data_identity,
                'config': config, 'source_sha256': hashes,
                'status': 'packaged_not_launched', 'query_gold_read': False,
                'operational_changes': ['Fixed input/model identity and explicit rerun flag parsing.',
                    'Stage model once onto temporary local SSD with I/O hashes and timing.',
                    'Serialize complete model+PEFT loading, then release all four workers together.',
                    'Task/global deadlines and 60s scoring reserve; export actual TTT steps and raw/scored candidates.',
                    'Final rerun recomputes a capped fair task budget from remaining wall time and shared remaining-task count.',
                    'Original score_kgmon and distinct-grid final2; raw placeholder provenance is explicit.']}
    (out / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    manifest['notebook_sha256'] = digest((out / filename).read_bytes())
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'package-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
