"""Package one manually launched V1-derived E20 observation; never pushes a kernel."""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path
import zlib


IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868'


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', default='arc2-021-nvarc-v1-e20')
    parser.add_argument('--seconds', type=int, default=6000)
    parser.add_argument('--task-seconds', type=int, default=660)
    parser.add_argument('--save-seconds', type=int, default=180)
    parser.add_argument('--ids', nargs='*')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    input_path = root.parent / '020/evaluation-e20/challenges.json'
    data = json.loads(input_path.read_text())
    ids = args.ids or sorted(data)
    if not ids or len(ids) != len(set(ids)) or set(ids) - set(data):
        raise ValueError('Need unique IDs from the fixed E20 cohort')
    selected = {key: data[key] for key in ids}
    for task in selected.values():
        if any('output' in query for query in task['test']):
            raise ValueError('Research input must not contain query outputs')
    raw_input = json.dumps(selected, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()
    blobs = {}
    for path in sorted((root / 'v1-research-src').glob('*.py')):
        raw = path.read_bytes()
        ast.parse(raw, filename=path.name)
        blobs[path.name] = base64.b64encode(raw).decode()
    blobs['challenges.json'] = base64.b64encode(raw_input).decode()
    hashes = {name: sha(base64.b64decode(raw)) for name, raw in blobs.items()}
    payload = base64.b64encode(zlib.compress(json.dumps(blobs).encode(), 9)).decode()
    code = '''import base64, hashlib, json, os, subprocess, sys, time, zlib
from pathlib import Path
notebook_started = time.time()
source = Path('/kaggle/working/021-code')
source.mkdir(exist_ok=False)
blobs = json.loads(zlib.decompress(base64.b64decode(PAYLOAD)))
expected = HASHES
for name, encoded in blobs.items():
    raw = base64.b64decode(encoded)
    if hashlib.sha256(raw).hexdigest() != expected[name]:
        raise RuntimeError('Source hash mismatch: ' + name)
    (source / name).write_bytes(raw)
(source / 'source-manifest.json').write_text(json.dumps(expected, indent=2))
def utility_name(path):
    return 'pip_install_unsloth_flash_patch' in str(path).replace('-', '_')
found = set()
for entry in sys.path:
    candidate = Path(entry or '.')
    if utility_name(candidate) and (candidate / 'unsloth/__init__.py').is_file():
        found.add(str(candidate.resolve()))
for mount in [Path('/kaggle/usr/lib'), Path('/kaggle/input')]:
    if mount.is_dir():
        for path in mount.rglob('unsloth/__init__.py'):
            if utility_name(path):
                found.add(str(path.parent.parent.resolve()))
if len(found) != 1:
    raise RuntimeError('Expected one original Unsloth Utility Script: ' + repr(sorted(found)))
utility = next(iter(found))
env = dict(os.environ)
env.update(PYTHONPATH=os.pathsep.join([utility, str(source), env.get('PYTHONPATH', '')]),
           PYTHONUNBUFFERED='1', PYTHONHASHSEED='42')
boot = dict(notebook_started=notebook_started, python=sys.version, utility_root=utility,
            source_sha256=expected, total_seconds=SECONDS, task_seconds=TASK_SECONDS,
            save_seconds=SAVE_SECONDS, reference='Failed in AIMO V1 / 312063593',
            execution='public E20 research; not a competition submission')
(source / 'boot.json').write_text(json.dumps(boot, indent=2))
print(json.dumps(boot), flush=True)
result = subprocess.run([sys.executable, '-u', str(source / 'run_research.py'),
                         '--notebook-start', str(notebook_started), '--seconds', str(SECONDS),
                         '--task-seconds', str(TASK_SECONDS), '--save-seconds', str(SAVE_SECONDS)], env=env)
print('RESEARCH_RETURN_CODE', result.returncode, flush=True)
if result.returncode:
    raise RuntimeError('Research failed; inspect 021-run/run.json and worker.log')
'''
    # Placeholders are unique uppercase names, replaced before execution.
    for key, value in [('PAYLOAD', repr(payload)), ('HASHES', repr(hashes)),
                       ('TASK_SECONDS', str(args.task_seconds)), ('SAVE_SECONDS', str(args.save_seconds)),
                       ('SECONDS', str(args.seconds))]:
        code = code.replace(key, value)
    ast.parse(code)
    title = 'ARC2 021 NVARC V1 E20'
    filename = args.slug + '.ipynb'
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                             'language_info': {'name': 'python', 'version': '3.11.13'},
                             'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False, 'accelerator': 'nvidiaL4'}},
                'cells': [{'id': 'scope', 'cell_type': 'markdown', 'metadata': {},
                           'source': ['# ARC2 021 NVARC V1 E20\n',
                                      'Derived from exact Failed in AIMO V1 (312063593, Apache-2.0).\n',
                                      'Four persistent L4 workers; original one-epoch r256 TTT, 128 training augmentations, 16 inference views and kgmon ranking.\n',
                                      'Task/global deadlines may stop training/search early; actual steps and coverage are reported.\n',
                                      'Research cohort only; no hidden rerun or competition submission is created by this Notebook.\n']},
                          {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': code.splitlines(keepends=True)}]}
    metadata = {'id': args.owner + '/' + args.slug, 'title': title, 'code_file': filename,
                'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
                'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'competition_sources': ['arc-prize-2026-arc-agi-2'], 'dataset_sources': [],
                'kernel_sources': ['sorokin/pip-install-unsloth-flash-patch'],
                'model_sources': ['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'],
                'docker_image': IMAGE, 'machine_shape': 'NvidiaL4'}
    manifest = {'reference': 'koushikrudra/failed-in-aimo V1', 'scriptVersionId': 312063593,
                'reference_notebook_sha256': sha((root / 'reference/failed-in-aimo-v1.ipynb').read_bytes()),
                'input_source': str(input_path), 'input_source_sha256': sha(input_path.read_bytes()),
                'selected_input_sha256': sha(raw_input), 'task_ids': ids,
                'tasks': len(selected), 'queries': sum(len(task['test']) for task in selected.values()),
                'seconds': args.seconds, 'task_seconds': args.task_seconds, 'save_seconds': args.save_seconds,
                'source_sha256': hashes,
                'changes_from_reference': [
                    'Explicit fixed E20 input and dynamic verified model mount; no query outputs.',
                    'Original four-worker queue/import serialization retained; errors are recorded and propagated.',
                    'PYTHONHASHSEED=42 fixes the V1 scoring seed for this observation.',
                    'Global wall-clock includes setup/load; process group stopped before saving reserve.',
                    'Per-task deadline is applied to TTT via TrainerCallback and DFS recursively; actual steps are recorded.',
                    'Scoring checks the task/global deadline before candidates and between score batches; unscored candidates are not saved.',
                    'Candidate files are atomically saved and periodically ranked by the unmodified V1 score_kgmon.',
                    'Original placeholder [[0]] slots are retained; true candidate provenance is exported separately.',
                    'Adds TTT/DFS/scoring timing, model-load events, GPU sampling and per-query candidate pools.'
                ], 'research_status': 'packaged_not_launched', 'gold_read': False}
    (out / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'source-manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'out': str(out), 'metadata_id': metadata['id'], 'tasks': manifest['tasks'],
                      'queries': manifest['queries'], 'notebook_bytes': (out / filename).stat().st_size,
                      'notebook_sha256': sha((out / filename).read_bytes()), 'source_sha256': hashes}))


if __name__ == '__main__':
    main()
