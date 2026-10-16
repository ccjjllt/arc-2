"""Manually package a frozen baseline candidate; never publish or start jobs."""
import argparse
import base64
import hashlib
import json
from pathlib import Path


FILES = ['arc_loader.py', 'arc_solver.py', 'build_submission.py', 'worker_pool.py', 'supervise.py']
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    source = root / 'submission-src'
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    blobs = {name: base64.b64encode((source / name).read_bytes()).decode() for name in FILES}
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in FILES}
    packed = base64.b64encode(json.dumps(blobs, sort_keys=True).encode()).decode()
    code = '''import base64, hashlib, json, os, subprocess, sys, time
from pathlib import Path
notebook_started = time.time()
source_dir = Path('/kaggle/working/013-code')
source_dir.mkdir(exist_ok=False)
blobs = json.loads(base64.b64decode(PACKED))
expected = HASHES
for name, encoded in blobs.items():
    raw = base64.b64decode(encoded)
    assert hashlib.sha256(raw).hexdigest() == expected[name], name
    (source_dir / name).write_bytes(raw)
(source_dir / 'source-manifest.json').write_text(json.dumps(expected, indent=2))
# The frozen upstream Utility Script contains installed packages, not wheels.
# Find its actual mounted root and make its import path explicit for children.
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
utility_roots = sorted(found)
assert len(utility_roots) == 1, ('Expected one original Unsloth Utility Script', utility_roots)
utility = utility_roots[0]
env = dict(os.environ)
env['PYTHONPATH'] = os.pathsep.join([utility, str(source_dir), env.get('PYTHONPATH', '')])
env['PYTHONUNBUFFERED'] = '1'
print(json.dumps({'event': 'boot', 'source_sha256': expected, 'utility_root': utility,
                  'python': sys.version, 'notebook_started': notebook_started}), flush=True)
(source_dir / 'boot.json').write_text(json.dumps({'source_sha256': expected,
    'utility_root': utility, 'python': sys.version, 'notebook_started': notebook_started}, indent=2))
result = subprocess.run([sys.executable, '-u', str(source_dir / 'supervise.py'),
                         '--notebook-start', str(notebook_started)], env=env)
print('SUPERVISOR_RETURN_CODE', result.returncode, flush=True)
assert result.returncode == 0, 'Inspect 013-run/supervisor.json and worker.log'
assert Path('/kaggle/working/submission.json').is_file()
'''.replace('PACKED', repr(packed)).replace('HASHES', repr(hashes))
    nb = {'nbformat': 4, 'nbformat_minor': 5,
          'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
                       'language_info': {'name': 'python', 'version': '3.11.13'},
                       'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False,
                                  'accelerator': 'nvidiaL4'}},
          'cells': [{'id': 'scope', 'cell_type': 'markdown', 'metadata': {},
                     'source': ['# ARC2 013 bounded baseline\n',
                                'Base-only Sorokin 4B TTT, one coherent kgmon selection, bounded workers and incremental submission.\n',
                                'Normal save: four previously used diagnostic tasks, <=45 minutes including loading. Hidden rerun: all tasks, <=12 hours with 10-minute saving reserve.\n',
                                'This is a new research candidate derived from latest v21; it is not the exact historical 30.14 version. No score improvement is claimed.\n']},
                    {'id': 'run', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                     'outputs': [], 'source': code.splitlines(keepends=True)}]}
    filename = 'arc2-013-bounded-baseline.ipynb'
    (out / filename).write_text(json.dumps(nb, ensure_ascii=False, indent=2) + '\n')
    metadata = {'id': 'ccjjllt/arc2-013-bounded-baseline', 'title': 'ARC2 013 bounded baseline',
                'code_file': filename, 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'competition_sources': ['arc-prize-2026-arc-agi-2'],
                'dataset_sources': [], 'kernel_sources': ['sorokin/pip-install-unsloth-flash-patch'],
                'model_sources': ['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'],
                'docker_image': IMAGE, 'machine_shape': 'NvidiaL4'}
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'source-manifest.json').write_text(json.dumps(hashes, indent=2) + '\n')
    print(json.dumps({'out': str(out), 'source_sha256': hashes,
                      'notebook_sha256': hashlib.sha256((out / filename).read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
