"""Run the authorized Kaggle CLI once, recording bounded submission evidence."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import sys

destination = Path(sys.argv[1])
command = sys.argv[2:]
os.environ['KAGGLE_API_TOKEN'] = (Path.home()/'.kaggle/access_token_zhuating').read_text().strip()
from kagglesdk.competitions.services.competition_api_service import CompetitionApiClient

record = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(), account='zhuating',
    source='Response hooks inside official Kaggle CLI', command=command,
    code_submission_requests=[], code_submission_responses=[], submissions=[])
original_create = CompetitionApiClient.create_code_submission
original_list = CompetitionApiClient.list_submissions
original_limits = CompetitionApiClient.get_submission_limits


def create_submission(self, request=None):
    record['code_submission_requests'].append(dict(competition=request.competition_name,
        kernel_owner=request._kernel_owner, kernel_slug=request.kernel_slug,
        kernel_version=request.kernel_version, file_name=request.file_name,
        description=request.submission_description))
    response = original_create(self, request)
    record['code_submission_responses'].append(dict(ref=response.ref, message=response.message))
    return response


def list_submissions(self, request=None):
    response = original_list(self, request)
    for row in response.submissions:
        status = getattr(row, '_status', None)
        record['submissions'].append(dict(ref=row.ref, file_name=row.file_name,
            date=row.date.isoformat() if row.date else None,
            description=row.description, status=str(status) if status is not None else None,
            public_score=row.public_score or None, private_score=row.private_score or None,
            team_name=row.team_name, submitted_by=row.submitted_by,
            url=row.url, error_description=row.error_description or None))
    return response


def limits(self, request=None):
    response = original_limits(self, request)
    record['limits'] = dict(num_today=response.num_today, num_total=response.num_total,
        num_allowed_now=response.num_allowed_now, limited_by_total=response.limited_by_total)
    return response


CompetitionApiClient.create_code_submission = create_submission
CompetitionApiClient.list_submissions = list_submissions
CompetitionApiClient.get_submission_limits = limits
sys.argv = ['kaggle', *command]
try:
    from kaggle.cli import main
    main()
finally:
    destination.write_text(json.dumps(record, ensure_ascii=False, indent=2)+'\n')
