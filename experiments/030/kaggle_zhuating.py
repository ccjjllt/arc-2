"""Forward one manual Kaggle CLI call with the explicitly authorized account."""
import os
from pathlib import Path
import shutil
import subprocess
import sys

token = (Path.home()/'.kaggle/access_token_zhuating').read_text().strip()
if not token:
    raise RuntimeError('The authorized zhuating token file is empty')
environment = dict(os.environ)
environment.update(KAGGLE_API_TOKEN=token, PYTHONUNBUFFERED='1')
process = subprocess.Popen([shutil.which('kaggle'), *sys.argv[1:]], env=environment,
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                           text=True, bufsize=1)
for line in process.stdout:
    sys.stdout.write(line.replace(token, '[REDACTED]'))
    sys.stdout.flush()
raise SystemExit(process.wait())
