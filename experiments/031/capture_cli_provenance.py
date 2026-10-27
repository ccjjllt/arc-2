"""Record version identifiers while the ordinary authenticated CLI performs reads."""
from datetime import datetime, timezone
from pathlib import Path
import json
import os
import re
import sys
from urllib.parse import urlsplit

destination = Path(sys.argv[1])
command = sys.argv[2:]
os.environ['KAGGLE_API_TOKEN'] = (Path.home()/'.kaggle/access_token_zhuating').read_text().strip()
from kagglesdk.kernels.services.kernels_api_service import KernelsApiClient
record = dict(recorded_at_utc=datetime.now(timezone.utc).isoformat(),
              source='Read-only response hooks inside official Kaggle CLI',
              account='zhuating', command=command, kernels=[], output_files=[])
original_get = KernelsApiClient.get_kernel
original_list = KernelsApiClient.list_kernel_session_output
def get_kernel(self, request=None):
    response = original_get(self, request)
    metadata = response.metadata
    record['kernels'].append(dict(current_version_number=metadata.current_version_number,
                                  slug=getattr(metadata,'slug',None), id=getattr(metadata,'id',None)))
    return response
def list_output(self, request=None):
    response = original_list(self, request)
    for item in response.files:
        match = re.search(r'/kf/(\d+)/', urlsplit(item.url).path)
        record['output_files'].append(dict(file_name=item.file_name,
            script_version_id=int(match.group(1)) if match else None))
    return response
KernelsApiClient.get_kernel = get_kernel
KernelsApiClient.list_kernel_session_output = list_output
if command and command[0] == 'quota':
    from kaggle.api.kaggle_api_extended import KaggleApi
    original_quota = KaggleApi.quota_view
    def quota_view(self):
        response = original_quota(self)
        record['authenticated_account'] = self.get_config_value(self.CONFIG_NAME_USER)
        q = response.gpu_quota
        record['gpu_quota_seconds'] = dict(time_used=q.time_used.total_seconds(),
            time_reserved=q.time_reserved.total_seconds(), total_time_allowed=q.total_time_allowed.total_seconds(),
            is_pay_to_scale_enabled=q.is_pay_to_scale_enabled)
        record['quota_refresh_time_utc'] = response.quota_refresh_time.isoformat()
        return response
    KaggleApi.quota_view = quota_view
sys.argv = ['kaggle', *command]
try:
    from kaggle.cli import main
    main()
finally:
    destination.write_text(json.dumps(record,indent=2)+'\n')
