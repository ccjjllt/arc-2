"""Manually record sanitized identity/quota before a research launch."""
import argparse
import datetime as dt
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--account', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    import requests
    original_send = requests.Session.send
    def bounded_send(session, request, **kwargs):
        kwargs.setdefault('timeout', (10, 20))
        return original_send(session, request, **kwargs)
    requests.Session.send = bounded_send
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()
    actual = api.get_config_value(api.CONFIG_NAME_USER)
    if actual != args.account:
        raise RuntimeError('Authenticated identity does not match requested account')
    q = api.quota_view()
    gpu = q.gpu_quota
    seconds = lambda x: x.total_seconds() if x is not None else None
    total, used, reserved = [seconds(getattr(gpu, k)) for k in
                             ('total_time_allowed', 'time_used', 'time_reserved')]
    record = {
        'checked_at_utc': dt.datetime.now(dt.timezone.utc).isoformat(),
        'account': actual,
        'quota_refresh_time_utc': q.quota_refresh_time.isoformat(),
        'gpu': {'total_seconds': total, 'used_seconds': used,
                'reserved_seconds': reserved,
                'available_seconds': max(0, total - used - reserved),
                'available_hours': max(0, total - used - reserved) / 3600},
        'scope': 'Read-only identity and quota; no credential values recorded, no job launch.',
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(record, ensure_ascii=False))


if __name__ == '__main__':
    main()
