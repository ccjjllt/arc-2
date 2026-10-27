#!/usr/bin/env python3
"""Manually record current Kaggle quota and team submission limits, read-only."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path


def seconds(value):
    return None if value is None else value.total_seconds()


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    from kaggle.api.kaggle_api_extended import KaggleApi
    api=KaggleApi();api.authenticate()
    quota=api.quota_view()
    limits=api.competition_get_submission_limits('arc-prize-2026-arc-agi-2')
    resources={}
    for name,q in [('gpu',quota.gpu_quota),('tpu',quota.tpu_quota)]:
        if q is None:
            resources[name]=None;continue
        used,total=seconds(q.time_used),seconds(q.total_time_allowed)
        resources[name]=dict(used_seconds=used,reserved_seconds=seconds(q.time_reserved),
            total_seconds=total,remaining_seconds=None if used is None or total is None else max(0,total-used),
            is_pay_to_scale_enabled=q.is_pay_to_scale_enabled)
    refresh=quota.quota_refresh_time
    record=dict(checked_at_utc=datetime.now(timezone.utc).isoformat(),
        source='Installed official Kaggle SDK, existing configured account; read-only',
        authenticated_account=api.get_config_value(api.CONFIG_NAME_USER),
        quota_refresh_time_utc=refresh.isoformat() if refresh else None,
        resources=resources,submission_limits=dict(num_today=limits.num_today,
            num_total=limits.num_total,num_allowed_now=limits.num_allowed_now,
            limited_by_total=limits.limited_by_total),
        gpu_started=False,competition_submission_created=False)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(record,indent=2)+'\n')
    print(json.dumps(record,indent=2))


if __name__=='__main__':main()
