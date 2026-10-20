#!/usr/bin/env python3
"""Manually invoked packaging of a frozen 020 research or submission notebook."""
import argparse
import base64
import hashlib
import json
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--owner', required=True)
    ap.add_argument('--slug', required=True)
    ap.add_argument('--cohort', help='Repo-relative challenge-only JSON; omitted for final notebook')
    ap.add_argument('--session-seconds', type=int, default=10800)
    ap.add_argument('--public-seconds', type=int, default=2700,
                    help='Final notebook ordinary Save & Run total wall budget, including installation')
    ap.add_argument('--ordinary-cpu-only', action='store_true',
                    help='Final notebook only: ordinary save uses the fixed CPU rule for at most 300s; '
                         'overrides public-seconds and leaves the model rerun unchanged')
    ap.add_argument('--rerun-seconds', type=int, default=39600,
                    help='Final competition rerun total wall budget, including installation')
    ap.add_argument('--runtime-kernel',
                    help='Private CPU carrier containing the identical recovered runtime wheelhouse')
    ap.add_argument('--policy', choices=['M0', 'M1'], default='M1')
    ap.add_argument('--gen-max', type=int, default=12288)
    ap.add_argument('--thinking-max', type=int, default=8192)
    ap.add_argument('--mtp-tokens', type=int, choices=[0, 2], default=0)
    ap.add_argument('--generation-mode', choices=['program', 'direct'], default='program')
    ap.add_argument('--direct-attempts', type=int, choices=[1, 2], default=2)
    ap.add_argument('--line-complement', action='store_true',
                    help='Embed the fixed CPU aligned-line complement for one direct attempt')
    ap.add_argument('--frozen-initial')
    ap.add_argument('--p0-only', action='store_true', help='Cohort-only single-call DeepP0 research')
    ap.add_argument('--output', required=True)
    args = ap.parse_args()
    if args.ordinary_cpu_only:
        if args.cohort:
            raise ValueError('ordinary-cpu-only requires a final notebook without a cohort')
        args.public_seconds = 300
    if not args.cohort and not ((args.ordinary_cpu_only or 1800 <= args.public_seconds <= 41400) and
                               1800 <= args.rerun_seconds <= 41400):
        raise ValueError('final model wall budgets must be between 1800 and 41400 seconds; '
                         'ordinary-cpu-only uses 300 seconds')
    if args.p0_only and (not args.cohort or args.frozen_initial):
        raise ValueError('p0-only requires a research cohort and excludes frozen-initial')
    if args.frozen_initial and args.generation_mode != 'program':
        raise ValueError('frozen initial programs require program generation mode')
    if args.line_complement and (args.generation_mode != 'direct' or args.direct_attempts != 1
                                or args.p0_only or args.frozen_initial):
        raise ValueError('line-complement requires direct-attempts=1, without p0-only or frozen-initial')
    blobs = {p.name: p.read_bytes() for p in sorted((ROOT / 'experiments/020/submit').glob('*.py'))}
    if args.line_complement:
        blobs['aligned_line_solver.py'] = (ROOT / 'experiments/020/aligned_line_solver.py').read_bytes()
    if args.cohort:
        blobs['cohort-challenges.json'] = (ROOT / args.cohort).read_bytes()
        obj = json.loads(blobs['cohort-challenges.json'])
        if any('output' in q for task in obj.values() for q in task['test']):
            raise ValueError('Query gold cannot be embedded')
    if args.frozen_initial:
        if not args.cohort:
            raise ValueError('frozen initial programs require a research cohort')
        blobs['frozen-initial.json'] = (ROOT / args.frozen_initial).read_bytes()
    expected = {k: hashlib.sha256(v).hexdigest() for k, v in blobs.items()}
    encoded = {k: base64.b64encode(v).decode() for k, v in blobs.items()}
    # First code cell owns the only session clock, before unpacking/install/load.
    first = '''import time, os
notebook_start_monotonic = time.monotonic()
notebook_start_wall = time.time()
'''
    second = '''import base64, hashlib, json, os, signal, subprocess, sys, zlib
from pathlib import Path
source_dir = Path('/kaggle/working/020-code')
source_dir.mkdir(parents=True, exist_ok=True)
blobs = json.loads(zlib.decompress(base64.b64decode(BLOBS_BASE64)))
expected = EXPECTED_SHAS
for name, value in blobs.items():
    raw = base64.b64decode(value)
    assert hashlib.sha256(raw).hexdigest() == expected[name], name
    (source_dir / name).write_bytes(raw)
sys.path.insert(0, str(source_dir))
import vdriver020_submit as driver
rerun = driver.rerun_mode(os.environ.get('KAGGLE_IS_COMPETITION_RERUN'))
IS_RESEARCH = RESEARCH_VALUE
ordinary_cpu = ORDINARY_CPU_ONLY_VALUE and not IS_RESEARCH and not rerun
if IS_RESEARCH:
    assert not rerun, 'Research notebooks must never be competition submitted'
    challenges = source_dir / 'cohort-challenges.json'
    experiment = 'paired'
    session_seconds = RESEARCH_SECONDS
else:
    challenges = driver.resolve_challenges('/kaggle/input', 'competition_submit' if rerun else 'public_eval')
    experiment = 'submit'
    session_seconds = RERUN_SECONDS if rerun else PUBLIC_SECONDS
PY312 = '/usr/bin/python3.12'
version = subprocess.run([PY312, '--version'], capture_output=True, text=True, check=True)
assert 'Python 3.12' in version.stdout + version.stderr
print(json.dumps({'event': 'boot', 'owner': OWNER_VALUE, 'rerun': rerun,
    'experiment': experiment, 'session_seconds': session_seconds,
    'ordinary_cpu_only': ordinary_cpu,
    'mtp_tokens': MTP_VALUE,
    'generation_mode': MODE_VALUE,
    'direct_attempts': DIRECT_ATTEMPTS_VALUE,
    'line_complement': LINE_COMPLEMENT_VALUE,
    'p0_only': P0_ONLY_VALUE,
    'result_label': 'ordinary_cpu_save' if ordinary_cpu else
        (('DeepDirect' if MODE_VALUE == 'direct' else 'DeepP0') if P0_ONLY_VALUE else experiment),
    'challenges_path': str(challenges), 'challenges_sha256': hashlib.sha256(challenges.read_bytes()).hexdigest(),
    'source_sha256': expected, 'notebook_start_wall': notebook_start_wall}), flush=True)
command = [PY312, '-u', str(source_dir / 'vdriver020_submit.py'),
    '--work', '/kaggle/working/vprobe', '--challenges', str(challenges),
    '--experiment', experiment, '--policy', POLICY_VALUE,
    '--owner', OWNER_VALUE, '--session-seconds', str(session_seconds),
    '--gen-max', str(GEN_VALUE), '--thinking-max', str(THINK_VALUE),
    '--mtp-tokens', str(MTP_VALUE),
    '--generation-mode', MODE_VALUE,
    '--direct-attempts', str(DIRECT_ATTEMPTS_VALUE),
    '--session-start-monotonic', str(notebook_start_monotonic),
    '--session-start-wall', str(notebook_start_wall), '--pip-python', sys.executable]
if FROZEN_VALUE:
    command += ['--frozen-initial', str(source_dir / 'frozen-initial.json')]
if IS_RESEARCH:
    command += ['--cohort', str(challenges)]
if P0_ONLY_VALUE:
    command += ['--p0-only']
if LINE_COMPLEMENT_VALUE:
    command += ['--line-complement']
if ordinary_cpu:
    command = [PY312, '-u', str(source_dir / 'ordinary_cpu.py'),
        '--challenges', str(challenges), '--submission', '/kaggle/working/submission.json',
        '--report', '/kaggle/working/ordinary-run.json',
        '--session-seconds', str(session_seconds),
        '--session-start-monotonic', str(notebook_start_monotonic)]
    if LINE_COMPLEMENT_VALUE:
        command += ['--line-complement']
proc = subprocess.Popen(command, cwd=str(source_dir), start_new_session=True)
deadline = notebook_start_monotonic + session_seconds
try:
    rc = proc.wait(timeout=max(1, deadline - time.monotonic() - 45))
except subprocess.TimeoutExpired:
    print('NOTEBOOK_PARENT_GUARD', flush=True)
    os.killpg(proc.pid, signal.SIGTERM)
    try:
        rc = proc.wait(timeout=max(1, min(30, deadline - time.monotonic() - 10)))
    except subprocess.TimeoutExpired:
        server_pid = Path('/kaggle/working/vprobe/server-pgid.txt')
        if server_pid.exists():
            try:
                os.killpg(int(server_pid.read_text()), signal.SIGKILL)
            except ProcessLookupError:
                pass
        os.killpg(proc.pid, signal.SIGKILL)
        rc = proc.wait(timeout=5)
print('VDRIVER_RETURN_CODE', rc, flush=True)
assert rc == 0, ('Inspect /kaggle/working/ordinary-run.json' if ordinary_cpu
                 else 'Inspect /kaggle/working/vprobe/run.json')
'''
    replacements = {'BLOBS_BASE64': repr(base64.b64encode(zlib.compress(json.dumps(encoded).encode(), 9)).decode()),
                    'EXPECTED_SHAS': repr(expected), 'RESEARCH_VALUE': repr(bool(args.cohort)),
                    'RESEARCH_SECONDS': str(args.session_seconds), 'OWNER_VALUE': repr(args.owner),
                    'PUBLIC_SECONDS': str(args.public_seconds), 'RERUN_SECONDS': str(args.rerun_seconds),
                    'ORDINARY_CPU_ONLY_VALUE': repr(args.ordinary_cpu_only),
                    'POLICY_VALUE': repr(args.policy), 'GEN_VALUE': str(args.gen_max),
                    'THINK_VALUE': str(args.thinking_max), 'MTP_VALUE': str(args.mtp_tokens),
                    'MODE_VALUE': repr(args.generation_mode),
                    'DIRECT_ATTEMPTS_VALUE': str(args.direct_attempts),
                    'LINE_COMPLEMENT_VALUE': repr(args.line_complement),
                    'FROZEN_VALUE': repr(bool(args.frozen_initial)),
                    'P0_ONLY_VALUE': repr(args.p0_only)}
    for key, value in replacements.items():
        second = second.replace(key, value)
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'cell_type': 'markdown', 'metadata': {}, 'source':
                           ['# ' + args.slug + '\n', 'Frozen 020 program policy. Query answers are excluded.']}]}
    for source in [first, second]:
        compile(source, '<notebook-cell>', 'exec')
        notebook['cells'].append({'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                                  'outputs': [], 'source': source.splitlines(keepends=True)})
    out = ROOT / args.output
    out.mkdir(parents=True, exist_ok=True)
    filename = args.slug + '.ipynb'
    (out / filename).write_text(json.dumps(notebook, ensure_ascii=False, indent=1) + '\n')
    meta = {'id': args.owner + '/' + args.slug, 'title': args.slug, 'code_file': filename,
            'language': 'python', 'kernel_type': 'notebook', 'is_private': True,
            'enable_gpu': True, 'enable_tpu': False, 'enable_internet': False,
            'machine_shape': 'NvidiaL4', 'competition_sources': ['arc-prize-2026-arc-agi-2'],
            'dataset_sources': [] if args.runtime_kernel else ['nick2187/qwen38-vllm0272-cu130-wheelhouse-v1/1'],
            'model_sources': ['foysalemonshanto/qwen3-8-27b-fp8-repacked-v1/PyTorch/hf-fp8/1'],
            'kernel_sources': [args.runtime_kernel] if args.runtime_kernel else []}
    (out / 'kernel-metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
    manifest = {'owner': args.owner, 'slug': args.slug, 'cohort': args.cohort,
                'p0_only': args.p0_only, 'generation_mode': args.generation_mode,
                'direct_attempts': args.direct_attempts,
                'line_complement': args.line_complement,
                'ordinary_cpu_only': args.ordinary_cpu_only,
                'result_label': ('DeepDirect' if args.generation_mode == 'direct' else 'DeepP0') if args.p0_only else ('paired' if args.cohort else 'submit'),
                'frozen_initial': args.frozen_initial, 'gen_max': args.gen_max, 'thinking_max': args.thinking_max,
                'mtp_tokens': args.mtp_tokens,
                'speculative_config': {'method': 'mtp', 'num_speculative_tokens': 2} if args.mtp_tokens else None,
                'mamba_cache_mode': 'align' if args.mtp_tokens else None,
                'policy': args.policy, 'source_sha256': expected,
                'asset_sources': {k: meta[k] for k in ('competition_sources', 'dataset_sources', 'model_sources', 'kernel_sources')},
                'notebook_sha256': hashlib.sha256((out / filename).read_bytes()).hexdigest(),
                'session_seconds': args.session_seconds if args.cohort else
                    {'public': args.public_seconds, 'rerun': args.rerun_seconds}}
    (out / 'package-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'output': str(out), 'notebook_sha256': manifest['notebook_sha256'],
                      'files': len(blobs), 'owner': args.owner, 'slug': args.slug}))


if __name__ == '__main__':
    main()
