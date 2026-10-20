#!/usr/bin/env python3
"""One-off manual recovery of the exact 014 wheelhouse; download bytes only.

Never resolves dependencies, installs packages, imports wheels, or substitutes
versions. A file is admitted only after its original byte count and SHA match.
"""
import argparse
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'experiments/014/public-sources/qwen38-vllm0272-manifest.json'
REQUIREMENTS = ROOT / 'experiments/014/public-sources/qwen38-vllm0272-requirements.txt'
REPORT = ROOT / 'experiments/020/execution/wheelhouse-restore-report.json'
DEFAULT_OUTPUT = Path('/private/tmp/arc2-020-wheelhouse-restore')


def sha_file(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2) + '\n')
    temporary.replace(path)


def request(url):
    return urllib.request.urlopen(urllib.request.Request(
        url, headers={'User-Agent': 'arc2-exact-wheelhouse-recovery/1'}), timeout=60)


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.extend(value for key, value in attrs if key == 'href' and value)


def index_candidates(index_url, filename, expected_sha):
    with request(index_url) as response:
        html = response.read().decode('utf-8')
    parser = Links()
    parser.feed(html)
    found = []
    for link in parser.links:
        url = urllib.parse.urljoin(index_url, link)
        parts = urllib.parse.urlsplit(url)
        if urllib.parse.unquote(parts.path.rsplit('/', 1)[-1]) != filename:
            continue
        hashes = urllib.parse.parse_qs(parts.fragment)
        if hashes.get('sha256') and hashes['sha256'] != [expected_sha]:
            continue
        if parts.scheme != 'https':
            continue
        found.append(urllib.parse.urlunsplit(parts._replace(fragment='')))
    return found


def exact_urls(entry, manifest):
    filename = Path(entry['path']).name
    package, version = filename.split('-')[:2]
    package = package.replace('_', '-').lower()
    errors, urls = [], []
    if filename == manifest['vllm_wheel']:
        return [manifest['vllm_wheel_url']], errors
    metadata_url = 'https://pypi.org/pypi/%s/%s/json' % (
        urllib.parse.quote(package, safe=''), urllib.parse.quote(version, safe=''))
    try:
        with request(metadata_url) as response:
            release = json.load(response)
        for row in release.get('urls', []):
            if row.get('filename') != filename:
                continue
            if row.get('digests', {}).get('sha256') != entry['sha256'] or row.get('size') != entry['bytes']:
                errors.append('PyPI exact filename has a different original hash/size; rejected')
                continue
            urls.append(row['url'])
        if not urls:
            errors.append('PyPI release has no exact filename+SHA+size match')
    except Exception as exc:
        errors.append('PyPI metadata: %s: %s' % (type(exc).__name__, str(exc)[:240]))
    if not urls:
        indexes = []
        if package.startswith(('torch', 'nvidia-', 'triton')):
            indexes.append('https://download.pytorch.org/whl/cu130/%s/' % package)
        if package.startswith('flashinfer-'):
            indexes.append('https://flashinfer.ai/whl/%s/' % package)
        for index_url in indexes:
            try:
                found = index_candidates(index_url, filename, entry['sha256'])
                urls.extend(found)
                if not found:
                    errors.append(index_url + ': no exact filename/hash link')
            except Exception as exc:
                errors.append(index_url + ': %s: %s' % (type(exc).__name__, str(exc)[:240]))
    return list(dict.fromkeys(urls)), errors


def recover(entry, manifest, output):
    started = time.monotonic()
    target = output / entry['path']
    result = dict(entry, status='failed', sources_tried=[], source_lookup_errors=[])
    if target.is_file() and target.stat().st_size == entry['bytes'] and sha_file(target) == entry['sha256']:
        return dict(result, status='verified_existing', elapsed_s=round(time.monotonic() - started, 3))
    urls, result['source_lookup_errors'] = exact_urls(entry, manifest)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(target.name + '.part')
    for url in urls:
        for attempt in range(1, 3):
            digest, size = hashlib.sha256(), 0
            attempt_record = {'url': url, 'attempt': attempt}
            result['sources_tried'].append(attempt_record)
            try:
                with request(url) as response, temporary.open('wb') as stream:
                    for chunk in iter(lambda: response.read(1 << 20), b''):
                        stream.write(chunk)
                        digest.update(chunk)
                        size += len(chunk)
                observed_sha = digest.hexdigest()
                attempt_record.update(observed_bytes=size, observed_sha256=observed_sha)
                if size != entry['bytes'] or observed_sha != entry['sha256']:
                    raise ValueError('original byte count or SHA mismatch; file rejected')
                temporary.replace(target)
                return dict(result, status='verified_downloaded', source_url=url,
                            elapsed_s=round(time.monotonic() - started, 3))
            except Exception as exc:
                attempt_record['error'] = '%s: %s' % (type(exc).__name__, str(exc)[:400])
                temporary.unlink(missing_ok=True)
                if isinstance(exc, ValueError):
                    break
    return dict(result, elapsed_s=round(time.monotonic() - started, 3))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--workers', type=int, default=8, choices=range(1, 9))
    args = parser.parse_args()
    manifest = json.loads(MANIFEST.read_text())
    wheels = [entry for entry in manifest['files'] if entry['path'].endswith('.whl')]
    if (len(wheels) != 193 or len(wheels) != manifest['wheel_count'] or
            sum(row['bytes'] for row in wheels) != manifest['total_wheel_bytes'] or
            len({row['path'] for row in wheels}) != len(wheels)):
        raise ValueError('original wheel manifest count/bytes/uniqueness mismatch')
    if any(Path(row['path']).parts[0] != 'wheels' or len(Path(row['path']).parts) != 2 for row in wheels):
        raise ValueError('unexpected original wheel path')
    requirement_entry = next(row for row in manifest['files'] if row['path'] == 'requirements-qwen38-runtime.txt')
    if REQUIREMENTS.stat().st_size != requirement_entry['bytes'] or sha_file(REQUIREMENTS) != requirement_entry['sha256']:
        raise ValueError('original requirements hash/size mismatch')
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(REQUIREMENTS, args.output / requirement_entry['path'])
    shutil.copyfile(MANIFEST, args.output / 'wheelhouse-manifest.json')
    report = {'status': 'downloading', 'started_wall': time.time(), 'output': str(args.output),
              'manifest_path': str(MANIFEST), 'manifest_sha256': sha_file(MANIFEST),
              'requirements_sha256': sha_file(REQUIREMENTS), 'expected_wheels': len(wheels),
              'expected_wheel_bytes': manifest['total_wheel_bytes'], 'workers': args.workers,
              'installed_or_executed_downloaded_code': False, 'GPU_used': False, 'uploaded': False,
              'results': []}
    def persist():
        verified = [row for row in report['results'] if row['status'].startswith('verified_')]
        report['verified_wheels'] = len(verified)
        report['verified_wheel_bytes'] = sum(row['bytes'] for row in verified)
        report['failed_wheels'] = sum(row['status'] == 'failed' for row in report['results'])
        report['elapsed_s'] = round(time.time() - report['started_wall'], 3)
        atomic_json(REPORT, report)
        atomic_json(args.output / 'restore-report.json', report)
    persist()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = {pool.submit(recover, entry, manifest, args.output): entry for entry in wheels}
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                entry = pending.pop(future)
                try:
                    result = future.result()
                except Exception as exc:
                    result = dict(entry, status='failed', error='%s: %s' % (type(exc).__name__, str(exc)[:400]))
                report['results'].append(result)
                print(json.dumps({'file': entry['path'], 'status': result['status'],
                                  'bytes': entry['bytes'], 'elapsed_s': result.get('elapsed_s'),
                                  'lookup_errors': result.get('source_lookup_errors') if result['status'] == 'failed' else None}), flush=True)
            persist()
            if not done:
                print(json.dumps({'event': 'progress', 'verified': report['verified_wheels'],
                                  'failed': report['failed_wheels'], 'pending': len(pending),
                                  'elapsed_s': report['elapsed_s']}), flush=True)
    report['status'] = 'complete_exact_original_wheels' if report['verified_wheels'] == len(wheels) else 'incomplete_missing_original_wheels'
    report['finished_wall'] = time.time()
    persist()
    print(json.dumps({key: report[key] for key in ('status', 'verified_wheels', 'failed_wheels', 'verified_wheel_bytes', 'elapsed_s')}), flush=True)
    return 0 if report['status'] == 'complete_exact_original_wheels' else 2


if __name__ == '__main__':
    raise SystemExit(main())
