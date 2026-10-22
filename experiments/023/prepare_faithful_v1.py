"""Manually package the frozen high-score V1 with minimal runtime repairs.

Generated third-party sources/notebooks stay in a private temporary directory.
This preparation command does not launch Kaggle or submit to the competition.
"""
import argparse
import ast
import base64
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REFERENCE = ROOT / 'experiments/021/reference/failed-in-aimo-v1.ipynb'
REFERENCE_SHA = '71416dd4d9885ecf2ae99b8b72ff0ef66d31fac0354b79d2d11071c49f84111d'
IMAGE = 'gcr.io/kaggle-private-byod/python@sha256:320043e14c68293f1c946585b9257123385205a58af4b94b17d31868cae4e868'


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--owner', required=True)
    parser.add_argument('--slug', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--copy-helper', required=True)
    args = parser.parse_args()
    out = Path(args.out).resolve()
    if out == ROOT or ROOT in out.parents:
        raise ValueError('Generated notebook must stay outside the repository')
    raw = REFERENCE.read_bytes()
    if digest(raw) != REFERENCE_SHA:
        raise ValueError('Frozen V1 identity mismatch')
    reference = json.loads(raw)
    helper = Path(args.copy_helper).read_bytes()
    ast.parse(helper)
    notebook = json.loads(raw)
    notebook['metadata'] = {
        'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'},
        'language_info': {'name': 'python', 'version': '3.11.13'},
        'kaggle': {'isGpuEnabled': True, 'isInternetEnabled': False, 'accelerator': 'nvidiaL4'},
    }
    for cell in notebook['cells']:
        cell['metadata'] = {}
        if cell['cell_type'] == 'code':
            cell['outputs'] = []
            cell['execution_count'] = None

    old_model = '/kaggle/input/qwen3_4b_grids15_sft139/transformers/bfloat16/1'
    local_model = '/kaggle/temp/arc023-model'
    solver = ''.join(notebook['cells'][4]['source'])
    if solver.count(old_model) != 1:
        raise ValueError('Unexpected V1 model reference')
    notebook['cells'][4]['source'] = solver.replace(old_model, local_model).splitlines(keepends=True)

    runtime = '''import base64, hashlib, json, os, subprocess, sys
from pathlib import Path
utility = Path('/kaggle/usr/lib/notebooks/sorokin/pip_install_unsloth_flash_patch')
if not (utility / 'unsloth/__init__.py').is_file():
    raise RuntimeError('Original mounted Unsloth utility is unavailable')
os.environ['PYTHONPATH'] = os.pathsep.join([str(utility), os.environ.get('PYTHONPATH', '')])
if str(utility) not in sys.path:
    sys.path.insert(0, str(utility))
model_source = Path('/kaggle/input/models/sorokin/qwen3_4b_grids15_sft139/transformers/bfloat16/1')
if not (model_source / 'config.json').is_file():
    raise RuntimeError('Expected fixed sft139 BF16 V1 checkpoint mount is unavailable')
boot = dict(reference_version=312063593, notebook_started=global_end_time-42600,
            global_end_time=global_end_time, raw_rerun_flag=os.getenv('KAGGLE_IS_COMPETITION_RERUN'),
            model_source=str(model_source), model_target='/kaggle/temp/arc023-model',
            model_config_sha256=hashlib.sha256((model_source/'config.json').read_bytes()).hexdigest(),
            utility_root=str(utility), python=sys.version,
            original_task_soft_limit=1200, original_global_seconds=42600,
            runtime_changes=['explicit GPU and current fixed model mount', 'stage identical checkpoint on local disk',
                             'original utility in child PYTHONPATH', 'propagate child failure'])
Path('023-boot.json').write_text(json.dumps(boot, indent=2))
print(json.dumps(dict(event='faithful_v1_boot', **boot)), flush=True)
'''
    runtime += "helper_bytes = base64.b64decode(" + repr(base64.b64encode(helper).decode()) + ")\n"
    runtime += "if hashlib.sha256(helper_bytes).hexdigest() != " + repr(digest(helper)) + ":\n    raise RuntimeError('Copy helper identity mismatch')\n"
    runtime += "Path('copy_model.py').write_bytes(helper_bytes)\n"
    runtime += "subprocess.run([sys.executable, 'copy_model.py', '--source', str(model_source), '--target', '/kaggle/temp/arc023-model', '--report', '/kaggle/working/023-model-copy.json', '--deadline', str(global_end_time)], check=True)\n"
    ast.parse(runtime)

    run = '''import subprocess
solver_env = dict(os.environ)
solver_env.update(UNSLOTH_DISABLE_STATISTICS='1', TRITON_PTXAS_PATH='/usr/local/cuda/bin/ptxas', OMP_NUM_THREADS='12')
subprocess.run([sys.executable, 'starter.py', '--end-time', str(global_end_time)], env=solver_env, check=True)
'''
    ast.parse(run)
    notebook['cells'][6]['source'] = run.splitlines(keepends=True)
    notebook['cells'].insert(2, {'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                               'outputs': [], 'source': runtime.splitlines(keepends=True)})

    source_hashes = {}
    differences = []
    for i, cell in enumerate(reference['cells']):
        if i in (4, 6):
            differences.append({'original_cell': i, 'change': 'model storage path only' if i == 4 else 'same child command/environment with checked return code'})
        else:
            current = notebook['cells'][i + (i >= 2)]
            if current['source'] != cell['source']:
                raise ValueError('Unexpected modification to original cell ' + str(i))
    for cell in notebook['cells']:
        source = ''.join(cell.get('source', []))
        if source.startswith('%%writefile '):
            first, body = source.split('\n', 1)
            name = first.split()[1]
            ast.parse(body, filename=name)
            source_hashes[name] = digest(body.encode())
        elif source and not source.startswith('!'):
            ast.parse(source)

    out.mkdir(parents=True, exist_ok=False)
    filename = args.slug + '.ipynb'
    metadata = {'id': args.owner + '/' + args.slug, 'title': 'ARC2 023 Failed in AIMO V1 faithful',
                'code_file': filename, 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
                'competition_sources': ['arc-prize-2026-arc-agi-2'], 'dataset_sources': [],
                'kernel_sources': ['sorokin/pip-install-unsloth-flash-patch'],
                'model_sources': ['sorokin/qwen3_4b_grids15_sft139/Transformers/bfloat16/1'],
                'docker_image': IMAGE, 'machine_shape': 'NvidiaL4'}
    notebook_raw = (json.dumps(notebook, ensure_ascii=False, indent=2) + '\n').encode()
    (out / filename).write_bytes(notebook_raw)
    (out / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    manifest = {'ref': metadata['id'], 'reference_script_version_id': 312063593,
                'reference_sha256': REFERENCE_SHA, 'notebook_sha256': digest(notebook_raw),
                'package_directory': str(out), 'runtime_source_sha256': source_hashes,
                'copy_helper_sha256': digest(helper), 'original_cell_differences': differences,
                'additional_runtime_cell': 'Current mounts, original utility path, local checkpoint copy and boot metadata',
                'preserved': ['42600s global clock starts in first original cell', 'original ordinary four tasks',
                              'original worker import/marker order', 'no all-model-ready barrier',
                              '1200s soft check', 'complete original one-epoch TTT', 'original DFS and scoring deadlines',
                              'original augmentation seeds, NLL and kgmon final2', 'no explicit PYTHONHASHSEED override'],
                'image_note': 'Image used by successful 022 execution; original historical image digest was not exposed in downloaded V1.',
                'status': 'prepared_not_launched'}
    (out / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
