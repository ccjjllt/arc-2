"""Stage 012 inputs and build three local Kaggle packages. No push, no GPU.

Manual use (CPU, no network needed beyond local files):
  python3 experiments/012/prepare_notebooks.py --check-only   # in-memory only
  python3 experiments/012/prepare_notebooks.py --stage-only   # real inputs/ staging only
  python3 experiments/012/prepare_notebooks.py                # staging + three packages
  python3 experiments/012/prepare_notebooks.py --out-suffix -v2  # new dir kaggle-v2, keep history

Basis: docs/plan/012 sections 4 (rows 5-6), 5, 7.2, 10. Code deltas in section 4 are
requirements, not existing runnable features; missing new files are recorded as
UNIMPLEMENTED, never silently claimed.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import io
import json
import re
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/012"
SRC = ROOT / "experiments/011/kaggle-s1-pilots-v1/011/pilot-code"
PILOT_TASKS = ("db118e2a", "291dc1e1")

EXPECTED_INPUT_SHA = {
    "p2-challenges.json": "abef4b67006970e7a899e9a15f661d65ec56b6f3f4264bf5c667717f916804d3",
    "p2-queries.json": "6379a22f3581b32411b7547f187c57403dbd6ba07fa22875d3b6e4a63dbd8b61",
    "p2-scoring.json": "2efbc9bf7437fd332521571aa582fc2126b30b9ff8658280a8d31c331ed08daf",
}
REQUIRED_CORE = (
    "arc_episode_data.py", "grid_runtime.py", "run_budgeted_grid_experiment.py",
    "solve_grid.py", "solve_program.py", "train_grid_sft.py", "ttt_per_task.py",
)
OPTIONAL_NEW = ("scripts/probe_gdn_inference.py",)  # plan 012 section 4 row 1
WHEELS = [
    {"file": "causal_conv1d-1.7.0+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl",
     "version": "1.7.0+cu12torch2.10cxx11abiTRUE",
     "sha256": "8e81f8c76435ad31aa41edc6c0c9d26de971a293b190183d2816da71547614d7",
     "target": "cp312-linux_x86_64, CXX11 ABI TRUE, cu12/torch2.10",
     "source": "011 gdn-followup-evidence.md 2026-10-11; pinned, no version scan"},
    {"file": "fla_core-0.5.2-py3-none-any.whl",
     "version": "0.5.2",
     "sha256": "5e830c85bad3d0d34677f98ac7074d08687a3756f0f0499d95ceb96eb6920761",
     "target": "py3-none-any",
     "source": "011 gdn-followup-evidence.md 2026-10-11; pinned, no version scan"},
    {"file": "flash_linear_attention-0.5.2-py3-none-any.whl",
     "version": "0.5.2",
     "sha256": "dcf405d81f5426393b59037097aa700d0f4a841465d5028d5aa543f4502f2400",
     "target": "py3-none-any",
     "source": "011 gdn-followup-evidence.md 2026-10-11; pinned, no version scan"},
]
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
MODEL = "ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1"
WHEELHOUSE = "ccjjllt/qwen38-wheelhouse-py312"
K_WHEEL_DATASET = "ccjjllt/arc2-012-k-wheels"  # real published K-wheel Dataset (progress 012 S2)
P0_FROZEN_LIMITS = {"max_new_tokens": 3072, "max_call_seconds": 720,
                    "max_task_seconds": 760, "max_prompt_tokens": 8192, "seed": 20261011}
COMPETITION = "arc-prize-2026-arc-agi-2"
MODES = ("cpu-preflight", "kernel-probe", "program-completion")
SLUGS = {"cpu-preflight": "arc2-012-cpu-preflight", "kernel-probe": "arc2-012-kernel-probe",
         "program-completion": "arc2-012-program-completion"}
TITLES = {"cpu-preflight": "ARC2 012 CPU preflight", "kernel-probe": "ARC2 012 kernel probe",
          "program-completion": "ARC2 012 program completion"}


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def json_bytes(value) -> bytes:
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def wheel_rows():
    """SHA-verify local wheel bytes; content is never modified, only measured."""
    rows = []
    for want in WHEELS:
        path = BASE / "wheels" / want["file"]
        if not path.is_file():
            raise FileNotFoundError("missing local wheel for SHA check: " + str(path))
        digest = hashlib.sha256()
        with open(path, "rb") as handle:
            for block in iter(lambda: handle.read(2 * 1024 * 1024), b""):
                digest.update(block)
        got = digest.hexdigest()
        if got != want["sha256"]:
            raise ValueError("wheel SHA mismatch for %s: got %s" % (want["file"], got))
        rows.append({**want, "measured_sha256": got, "bytes": path.stat().st_size})
    return rows


def stage_inputs(write: bool, k_wheel_dataset=None):
    """Copy 011 P2 files byte-identical, verify SHA, split per-task queries."""
    staged, manifest = {}, {}
    for name, want in EXPECTED_INPUT_SHA.items():
        src = SRC / name
        if not src.is_file():
            raise FileNotFoundError("missing 011 source (K exclusion cannot be verified): " + str(src))
        data = src.read_bytes()
        got = sha_bytes(data)
        if got != want:
            raise ValueError("011 source SHA mismatch for %s: got %s want %s" % (name, got, want))
        staged[name] = data
        manifest[name] = {"source": str(src.relative_to(ROOT)), "sha256": got, "bytes": len(data)}
    queries = json.loads(staged["p2-queries.json"].decode())
    if queries != [{"task_id": t, "query_index": 0} for t in PILOT_TASKS]:
        raise ValueError("unexpected P2 query list: %r" % (queries,))
    challenges = json.loads(staged["p2-challenges.json"].decode())
    if list(challenges) != list(PILOT_TASKS):
        raise ValueError("unexpected P2 challenge tasks")
    for task in challenges.values():  # input-only guard: scoring stays separate
        assert all(set(q) == {"input"} for q in task["test"]), "challenge test must be input-only"
    scoring = json.loads(staged["p2-scoring.json"].decode())
    assert set(scoring) == set(PILOT_TASKS) and all(len(v) == 1 for v in scoring.values())
    per_task = {}
    for tid in PILOT_TASKS:
        blob = json_bytes([{"task_id": tid, "query_index": 0}])
        per_task["p2-queries-%s.json" % tid] = blob
        manifest["p2-queries-%s.json" % tid] = {"source": "split from p2-queries.json", "sha256": sha_bytes(blob),
                                                "bytes": len(blob), "task_id": tid, "query_index": 0}
    wheel_manifest = {"status": "READY" if k_wheel_dataset else "PINNED_NOT_UPLOADED",
                      "note": ("Local wheel bytes SHA-verified; published Dataset " + str(k_wheel_dataset)
                               + ". Content unmodified.")
                      if k_wheel_dataset else
                      "Exact files/SHA from plan 012 section 5 (checked 2026-10-11). "
                      "New wheel Dataset has no uploaded slug in this handoff; K push is blocked "
                      "until k_wheel_dataset is filled with the real published slug/version.",
                      "python": "3.12", "torch": "2.10.0+cu128", "transformers": "5.8.0",
                      "base_wheelhouse": WHEELHOUSE, "k_wheel_dataset": k_wheel_dataset or None,
                      "dataset_metadata_id": K_WHEEL_DATASET, "wheels": wheel_rows()}
    if write:
        inp = BASE / "inputs"
        inp.mkdir(parents=True, exist_ok=True)
        for name, data in {**staged, **per_task}.items():
            (inp / name).write_bytes(data)
        (inp / "inputs-manifest.json").write_bytes(json_bytes(manifest))
        (BASE / "wheel-manifest.json").write_bytes(json_bytes(wheel_manifest))
    return staged, per_task, manifest, wheel_manifest


def load_core():
    packed, missing = {}, []
    for name in REQUIRED_CORE:
        path = ROOT / "scripts" / name
        if not path.is_file():
            raise FileNotFoundError("missing required core: " + str(path))
        data = path.read_bytes()
        ast.parse(data.decode(), filename=name)
        packed[name] = data
    for rel in OPTIONAL_NEW:
        path = ROOT / rel
        if path.is_file():
            data = path.read_bytes()
            ast.parse(data.decode(), filename=path.name)
            packed["probe_gdn_inference.py"] = data
        else:
            missing.append(rel + " UNIMPLEMENTED (plan 012 sec 4 row 1)")
    core_manifest = {n: sha_bytes(d) for n, d in sorted(packed.items()) if n.endswith(".py")}
    core_sha = sha_bytes(json.dumps(core_manifest, sort_keys=True, separators=(",", ":")).encode())
    return packed, core_manifest, core_sha, missing


def payload_for(mode, core_packed, staged, per_task):
    if mode == "kernel-probe" and "p2-scoring.json" in staged:
        pass  # staged inputs keep scoring; the K *package* must exclude it (checked below)
    blob = dict(core_packed)
    blob["p2-challenges.json"] = staged["p2-challenges.json"]
    blob["p2-queries.json"] = staged["p2-queries.json"]
    blob.update(per_task)
    if mode == "program-completion":
        blob["p2-scoring.json"] = staged["p2-scoring.json"]
    if mode == "kernel-probe" and any("scoring" in n for n in blob):
        raise RuntimeError("K package must exclude scoring files")
    return blob


SETUP_TMPL = (
    "import base64, hashlib, io, json, pathlib, time, os\nimport zipfile\n"
    "os.environ['HF_DEACTIVATE_ASYNC_LOAD']='1'\n"
    "NOTEBOOK_START=time.monotonic()\nCODE=pathlib.Path('{code_dir}'); CODE.mkdir(parents=True,exist_ok=True)\n"
    "CODE_MANIFEST={manifest}\nCORE_MANIFEST={core}\nCORE_MANIFEST_SHA={sha}\n"
    "payload=base64.b64decode({b64})\n"
    "with zipfile.ZipFile(io.BytesIO(payload)) as z:\n"
    "    assert set(z.namelist())==set(CODE_MANIFEST)\n"
    "    for name in z.namelist():\n"
    "        data=z.read(name)\n"
    "        assert hashlib.sha256(data).hexdigest()==CODE_MANIFEST[name]\n"
    "        (CODE/name).write_bytes(data)\n"
    "(CODE/'source-manifest.json').write_text(json.dumps(CODE_MANIFEST,indent=2))\n"
    "(CODE/'core-manifest.json').write_text(json.dumps(CORE_MANIFEST,indent=2))\n"
    "print('012_CORE_SHA',CORE_MANIFEST_SHA)\n"
)

INSTALL_BASE = (
    "import glob, subprocess, sys, json\n"
    "wheel=glob.glob('/kaggle/input/**/qwen38-wheelhouse-py312',recursive=True)\n"
    "if len(wheel)!=1:\n    raise RuntimeError('Expected one frozen wheelhouse, found %d'%len(wheel))\n"
    "p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--find-links='+wheel[0],"
    "'transformers==5.8.0','accelerate==1.13.0','peft==0.19.1'],capture_output=True,text=True,timeout=max(1,min(240,PREP_DEADLINE-time.monotonic())))\n"
    "print((p.stdout or '')[-1200:])\n"
    "if p.returncode:\n    print((p.stderr or '')[-1200:]);raise RuntimeError('Offline base install failed')\n"
    "import transformers,torch,peft,accelerate\n"
    "versions=dict(python=sys.version.split()[0],transformers=transformers.__version__,"
    "torch=torch.__version__,peft=peft.__version__,accelerate=accelerate.__version__)\n"
    "print('VERSIONS',json.dumps(versions))\n"
    "assert transformers.__version__=='5.8.0'\n"
)
INSTALL_K_EXTRA = (
    "import glob,subprocess,sys,hashlib,pathlib,shutil,json\n"
    "cand=[p for p in glob.glob('/kaggle/input/**/*.whl',recursive=True) if 'qwen38' not in p]\n"
    "want={wheels}\n"
    "def _norm(n):\n    return n.replace('+','')\n"
    "byexact={{pathlib.Path(p).name:p for p in cand}}\n"
    "bynorm={{_norm(pathlib.Path(p).name):p for p in cand}}\n"
    "resolved={{}}\n"
    "for w in want:\n"
    "    if w['file'] in byexact:\n"
    "        resolved[w['file']]=byexact[w['file']]\n"
    "    elif _norm(w['file']) in bynorm:\n"
    "        resolved[w['file']]=bynorm[_norm(w['file'])]\n"
    "        print('WHEEL_NAME_SANITIZED',w['file'],'->',pathlib.Path(resolved[w['file']]).name)\n"
    "    else:\n"
    "        raise RuntimeError('Missing K wheels in new Dataset: '+w['file'])\n"
    "wdir=pathlib.Path('/kaggle/working/wheels');wdir.mkdir(parents=True,exist_ok=True)\n"
    "mapping={{}}\n"
    "for w in want:\n"
    "    h=hashlib.sha256(open(resolved[w['file']],'rb').read()).hexdigest()\n"
    "    assert h==w['sha256'],w['file']\n"
    "    print('WHEEL_SHA_OK',pathlib.Path(resolved[w['file']]).name,h[:16])\n"
    "    canon=wdir/w['file']\n"
    "    shutil.copyfile(resolved[w['file']],canon)\n"
    "    assert hashlib.sha256(open(canon,'rb').read()).hexdigest()==w['sha256']\n"
    "    mapping[pathlib.Path(resolved[w['file']]).name]=canon.name\n"
    "    print('WHEEL_RESTORED',canon.name)\n"
    "(CODE/'wheel-name-map.json').write_text(json.dumps(mapping,indent=2))\n"
    "print('WHEEL_NAME_MAP',json.dumps(mapping))\n"
    "p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps']"
    "+[str(wdir/w['file']) for w in want],capture_output=True,text=True,timeout=max(1,PREP_DEADLINE-time.monotonic()))\n"
    "print((p.stdout or '')[-1200:])\n"
    "if p.returncode:\n    print((p.stderr or '')[-1200:]);raise RuntimeError('K wheel install failed')\n"
)
RESOLVE = (
    "from pathlib import Path\nimport hashlib,json,sys\n"
    "def digest(p):\n    h=hashlib.sha256()\n"
    "    f=open(p,'rb')\n"
    "    b=f.read(2097152)\n"
    "    while b:\n        h.update(b);b=f.read(2097152)\n"
    "    f.close();return h.hexdigest()\n"
    "models=[p.parent for p in Path('/kaggle/input').rglob('config.json') if 'qwen3' in p.read_text().lower()]\n"
    "if len(models)!=1:\n    raise RuntimeError('Expected one Qwen3.8 model dir, found %d'%len(models))\n"
    "model_dir=str(models[0])\nprint('MODEL_DIR',model_dir)\n"
    "assert not list(Path('/kaggle/input').rglob('adapter_step_25/adapter_model.safetensors')),'012 mounts no S4 adapter'\n"
    "sys.path.insert(0,str(CODE))\n"
)
CONFIG_K = (
    "import json\n"
    "from transformers import AutoTokenizer\n"
    "import arc_episode_data as episodes\n"
    "import probe_gdn_inference as probe\n"
    "tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)\n"
    "tasks=json.loads((CODE/'p2-challenges.json').read_text())\n"
    "task=tasks['db118e2a']\n"
    "msgs=episodes.build_messages(task['train'],task['test'][0]['input'],formatter='compact')\n"
    "prompt_text=tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,enable_thinking=False)\n"
    "pinfo=probe.write_f1_prompt_with_tokenizer(tok,prompt_text,str(CODE/'db118e2a-query0-f1-prompt.txt'))\n"
    "print('PROMPT_RENDERED',json.dumps(pinfo))\n"
    "config={'notebook':'kernel-probe','task':'db118e2a/query0','prompt':'f1_full_nonhthinking',\n"
    " 'fixed_decode_tokens':64,'order':['A1','B1','B2','A2'],\n"
    " 'thresholds':{'a_rel_rms':0.0001,'b_rel_floor':0.01,'gain':0.15},\n"
    " 'notebook_seconds':1800,'load_limit_s':600,'save_window_s':120,\n"
    " 'model_dir':model_dir,'output_dir':'/kaggle/working/012/kernel-probe/output',\n"
    " 'prompt_file':'db118e2a-query0-f1-prompt.txt','prompt_sha256':pinfo['sha256'],\n"
    " 'prompt_tokens':pinfo['n_tokens'],'journal':'/kaggle/working/012/kernel-probe/journal.jsonl'}\n"
    "(CODE/'run-k.json').write_text(json.dumps(config,indent=2))\nprint(json.dumps(config,indent=2))\n"
    "assert not (CODE/'p2-scoring.json').exists(),'K must not stage scoring'\n"
    "assert (CODE/'probe_gdn_inference.py').is_file(),'packed probe entry missing'\n"
    "import subprocess,sys\n"
    "print('PROBE_IMPORTED',probe.__file__)\n"
    "p=subprocess.run([sys.executable,probe.__file__,'--dry-run-cpu','--prompt-file',pinfo['path']],"
    "capture_output=True,text=True,timeout=60)\n"
    "print((p.stdout or '')[-1500:])\n"
    "assert p.returncode==0,'probe --dry-run-cpu failed: '+(p.stderr or '')[-500:]\n"
    "print('PROBE_DRY_RUN_OK')\n"
)
CONFIG_P_TMPL = (
    "import json\n"
    "frozen={'max_new_tokens':3072,'max_call_seconds':720,'max_task_seconds':760,'max_prompt_tokens':8192,'seed':20261011}\n"
    "stages=[]\n"
    "for name,tid,think,budget in [('p0-db118e2a-D','db118e2a',False,4500),"
    "('p0-db118e2a-L','db118e2a',True,760),('p0-291dc1e1-L','291dc1e1',True,760),"
    "('p0-291dc1e1-D','291dc1e1',False,760)]:\n"
    "    stages.append({'name':name,'type':'program','query_file':str(CODE/('p2-queries-%s.json'%tid)),"
    "'policy':'p0_only','enable_thinking':think,'reasoning_effort':'low','preserve_thinking':True,"
    "'arms':['p0'],'limits':dict(frozen),'time_budget_s':budget})\n"
    "config={'model_dir':model_dir,'challenges':str(CODE/'p2-challenges.json'),"
    "'out_dir':'/kaggle/working/012/program-completion/output',"
    "'time_budget_s':4500,'load_limit_s':900,'notebook_start_monotonic':NOTEBOOK_START,"
    "'runtime':{'base_source':'012BASESOURCE','save_window_s':300,'deactivate_async_load':True},'stages':stages}\n"
    "(CODE/'run-p.json').write_text(json.dumps(config,indent=2))\nprint(json.dumps(config,indent=2))\n"
    "import run_budgeted_grid_experiment as runner\n"
    "checked=runner.read_config(str(CODE/'run-p.json'))\n"
    "print('RUNNER_SCHEMA_OK',[s['name'] for s in checked['stages']])\n"
    "import solve_program\n"
    "solve_program.ProgramLimits(**frozen).validate(policy='p0_only')\n"
    "gen={'status':'complete','ended_eos':True}\n"
    "code='```python\\ndef transform(grid):\\n    return [row[:] for row in grid]\\n```'\n"
    "d=solve_program.split_program_response(code,thinking_mode='direct',generation=gen)\n"
    "assert d['status']=='program_ready',d\n"
    "l=solve_program.split_program_response('draft</think>\\n'+code,thinking_mode='thinking',generation=gen)\n"
    "assert l['status']=='program_ready',l\n"
    "print('P0_ENTRY_OK')\n"
)
CONFIG_P = CONFIG_P_TMPL.replace("012BASESOURCE", MODEL)
RUN_K = (
    "import json,time\n"
    "elapsed=time.monotonic()-NOTEBOOK_START\nprint('HARD_LIMITS',{'notebook_seconds':1800,'already_elapsed_s':elapsed})\n"
    "cfg=json.loads((CODE/'run-k.json').read_text())\n"
    "assert cfg['task']=='db118e2a/query0' and cfg['order']==['A1','B1','B2','A2']\n"
    "assert cfg['fixed_decode_tokens']==64 and cfg['load_limit_s']==600\n"
    "import probe_gdn_inference as probe\n"
    "rc=probe.main(['--model-dir',cfg['model_dir'],'--output-dir',cfg['output_dir'],"
    "'--prompt-file',str(CODE/cfg['prompt_file']),'--journal',cfg['journal'],"
    "'--notebook-start-monotonic',str(NOTEBOOK_START)])\n"
    "print('K_RC',rc)\nassert rc==0\n"
)
RUN_P = (
    "import time\n"
    "elapsed=time.monotonic()-NOTEBOOK_START\nprint('HARD_LIMITS',{'notebook_seconds':4500,'already_elapsed_s':elapsed})\n"
    "import run_budgeted_grid_experiment as runner\n"
    "rc=runner.main(['--config',str(CODE/'run-p.json')])\n"
    "print('P_RC',rc)\nassert rc==0\n"
)
CPU_CHECKS = (
    "import json\n"
    "assert not __import__('torch').cuda.is_available(),'CPU preflight must not reserve GPU'\n"
    "tasks=json.loads((CODE/'p2-challenges.json').read_text())\n"
    "assert list(tasks)==['db118e2a','291dc1e1']\n"
    "print('CPU_INPUTS_OK',sorted(tasks))\n"
)
CPU_FULLPATH = (
    "import contextlib, json, time, pathlib\n"
    "import grid_runtime as runtime\n"
    "import solve_program\n"
    "import run_budgeted_grid_experiment as runner\n"
    "FIXED_D_RAW='```python\\ndef transform(grid):\\n    return [row[:] for row in grid]\\n```'\n"
    "FIXED_L_RAW='draft-thinking</think>\\n```python\\ndef transform(grid):\\n    return [row[:] for row in grid]\\n```'\n"
    "class FakeTensor:\n"
    "    def __init__(self, data):\n"
    "        self.data=data\n"
    "        self.device='cpu'\n"
    "    def to(self, dev):\n"
    "        return self\n"
    "    def tolist(self):\n"
    "        return self.data\n"
    "    def __getitem__(self, key):\n"
    "        return self\n"
    "class FakeSequences:\n"
    "    def __init__(self, data):\n"
    "        self.data=data\n"
    "    def __getitem__(self, key):\n"
    "        r, c=key\n"
    "        return FakeTensor(self.data[r][c])\n"
    "class FakeTokenizer:\n"
    "    chat_template='fake-cpu-template'\n"
    "    def __init__(self, queue):\n"
    "        self.queue=queue\n"
    "    def apply_chat_template(self, messages, **kw):\n"
    "        return 'FAKE-PROMPT|'+json.dumps(messages, sort_keys=True)[:200]\n"
    "    def __call__(self, text, **kw):\n"
    "        if kw.get('return_tensors')=='pt':\n"
    "            return {'input_ids': FakeTensor([[0]*64]), 'attention_mask': FakeTensor([[1]*64])}\n"
    "        return {'input_ids': list(range(64))}\n"
    "    def decode(self, ids, **kw):\n"
    "        if not self.queue:\n"
    "            raise RuntimeError('FakeModel queue exhausted')\n"
    "        return self.queue.pop(0)\n"
    "class FakeCuda:\n"
    "    @staticmethod\n"
    "    def is_available():\n"
    "        return False\n"
    "    @staticmethod\n"
    "    def device_count():\n"
    "        return 0\n"
    "    @staticmethod\n"
    "    def synchronize(i=None):\n"
    "        return None\n"
    "    @staticmethod\n"
    "    def manual_seed_all(s):\n"
    "        return None\n"
    "class FakeTorch:\n"
    "    cuda=FakeCuda()\n"
    "    @staticmethod\n"
    "    def manual_seed(s):\n"
    "        import random\n"
    "        random.seed(s)\n"
    "        return None\n"
    "    @staticmethod\n"
    "    def multinomial(*a, **k):\n"
    "        raise RuntimeError('FakeTorch multinomial must not run')\n"
    "    class Generator:\n"
    "        def __init__(self, device=None):\n"
    "            self.device=device\n"
    "        def manual_seed(self, s):\n"
    "            return self\n"
    "    @staticmethod\n"
    "    def inference_mode():\n"
    "        return contextlib.nullcontext()\n"
    "class FakeModel:\n"
    "    def __init__(self, queue, eos_id=999):\n"
    "        self.queue=queue\n"
    "        self.eos_id=eos_id\n"
    "        self.gen_calls=0\n"
    "        self.generation_config=None\n"
    "    def eval(self):\n"
    "        return self\n"
    "    def generate(self, **kw):\n"
    "        self.gen_calls+=1\n"
    "        inp=kw.get('input_ids')\n"
    "        try:\n"
    "            plen=len(inp.data[0])\n"
    "        except Exception:\n"
    "            plen=64\n"
    "        new_ids=[7, 8, self.eos_id]\n"
    "        full=[[0]*plen+new_ids]\n"
    "        return FakeSequences(full)\n"
    "QUEUE=[FIXED_D_RAW, FIXED_L_RAW, FIXED_D_RAW, FIXED_L_RAW, FIXED_L_RAW, FIXED_D_RAW]\n"
    "TOK=FakeTokenizer(list(QUEUE))\n"
    "FAKE_JPATH='/kaggle/working/012/cpu-preflight/journal-fullpath.jsonl'\n"
    "pathlib.Path(FAKE_JPATH).parent.mkdir(parents=True, exist_ok=True)\n"
    "_p=pathlib.Path(FAKE_JPATH)\n"
    "if _p.exists():\n"
    "    _p.unlink()\n"
    "JOUR=runtime.CandidateJournal(FAKE_JPATH)\n"
    "FAKE_SESSION=runtime.Session(model=FakeModel(TOK.queue), tokenizer=TOK, torch=FakeTorch(), in_dev='cpu', eos_id=999, pad_id=999, deadline=time.monotonic()+3600.0, model_dir='cpu-fake', base_identity={'fake': 'cpu'}, journal=JOUR, runtime_identity={'fake': 'cpu'}, active_identity={'kind': 'base'}, save_window_s=300.0, allow_cached=True)\n"
    "MSGS_D=[{'role': 'system', 'content': solve_program.SYSTEM_PROMPT}, {'role': 'user', 'content': 'direct-arm-check'}]\n"
    "gen_d=runtime.generate_text(FAKE_SESSION, MSGS_D, max_new_tokens=64, do_sample=False, enable_thinking=False, reasoning_effort='low', preserve_thinking=True, deadline=time.monotonic()+60.0, max_time_s=60.0, max_prompt_tokens=8192, identity={'task_id': 'cpu-check-D', 'program_arm': 'p0', 'mode': 'D'})\n"
    "assert gen_d.get('status')=='complete', gen_d\n"
    "assert gen_d.get('ended_eos') is True, gen_d\n"
    "parsed_d=solve_program.split_program_response(gen_d.get('raw_text', ''), thinking_mode='direct', generation=gen_d)\n"
    "assert parsed_d.get('status')=='program_ready', parsed_d\n"
    "MSGS_L=[{'role': 'system', 'content': solve_program.SYSTEM_PROMPT}, {'role': 'user', 'content': 'thinking-arm-check'}]\n"
    "gen_l=runtime.generate_text(FAKE_SESSION, MSGS_L, max_new_tokens=64, do_sample=False, enable_thinking=True, reasoning_effort='low', preserve_thinking=True, deadline=time.monotonic()+60.0, max_time_s=60.0, max_prompt_tokens=8192, identity={'task_id': 'cpu-check-L', 'program_arm': 'p0', 'mode': 'L'})\n"
    "assert gen_l.get('status')=='complete', gen_l\n"
    "parsed_l=solve_program.split_program_response(gen_l.get('raw_text', ''), thinking_mode='thinking', generation=gen_l)\n"
    "assert parsed_l.get('status')=='program_ready', parsed_l\n"
    "print('CPU_FULLPATH_PARSE_OK', parsed_d.get('status'), parsed_l.get('status'))\n"
    "TASKS=json.loads((CODE/'p2-challenges.json').read_text())\n"
    "FROZEN={'max_new_tokens': 3072, 'max_call_seconds': 720, 'max_task_seconds': 760, 'max_prompt_tokens': 8192, 'seed': 20261011}\n"
    "PLAN=[('db118e2a', False), ('db118e2a', True), ('291dc1e1', True), ('291dc1e1', False)]\n"
    "CFG_STAGES=[]\n"
    "for _tid, _think in PLAN:\n"
    "    CFG_STAGES.append({'name': 'p0-%s-%s' % (_tid, 'L' if _think else 'D'), 'type': 'program', 'query_file': str(CODE/('p2-queries-%s.json' % _tid)), 'policy': 'p0_only', 'enable_thinking': _think, 'reasoning_effort': 'low', 'preserve_thinking': True, 'arms': ['p0'], 'limits': dict(FROZEN)})\n"
    "plan_calls=runner._program_call_plan({'stages': CFG_STAGES})\n"
    "assert len(plan_calls)==4, plan_calls\n"
    "assert [c.get('mode') for c in plan_calls]==['D', 'L', 'L', 'D'], plan_calls\n"
    "RESULTS=[]\n"
    "for _idx, (_tid, _think) in enumerate(PLAN):\n"
    "    _lim=solve_program.ProgramLimits(**FROZEN)\n"
    "    _lim.validate(policy='p0_only')\n"
    "    _res=solve_program.run_program_task(FAKE_SESSION, _tid, TASKS[_tid], _lim, deadline=time.monotonic()+3000.0, task_index=_idx, journal=lambda rec: JOUR.append({'journal': 'p0-task'}, rec), policy='p0_only', enable_thinking=_think, reasoning_effort='low', preserve_thinking=True)\n"
    "    RESULTS.append(_res)\n"
    "    assert _res.get('physical_generation_calls')==1, _res\n"
    "    assert _res.get('repair_invoked_count')==0, _res\n"
    "    assert set(_res.get('proposals', {}).keys())=={'p0'}, _res\n"
    "assert sum(r.get('physical_generation_calls', 0) for r in RESULTS)==4, RESULTS\n"
    "assert sum(r.get('repair_invoked_count', 0) for r in RESULTS)==0, RESULTS\n"
    "print('CPU_FULLPATH_P0_OK', len(RESULTS))\n"
    "gen_before=FAKE_SESSION.model.gen_calls\n"
    "TINY_STAGE={'name': 'p0-db118e2a-D', 'type': 'program', 'query_file': str(CODE/'p2-queries-db118e2a.json'), 'policy': 'p0_only', 'enable_thinking': False, 'reasoning_effort': 'low', 'preserve_thinking': True, 'arms': ['p0'], 'limits': dict(FROZEN)}\n"
    "TINY_OUT=pathlib.Path('/kaggle/working/012/cpu-preflight/tiny-out')\n"
    "TINY_OUT.mkdir(parents=True, exist_ok=True)\n"
    "TINY_JPATH='/kaggle/working/012/cpu-preflight/journal-tiny.jsonl'\n"
    "_tp=pathlib.Path(TINY_JPATH)\n"
    "if _tp.exists():\n"
    "    _tp.unlink()\n"
    "TINY_JOUR=runtime.CandidateJournal(TINY_JPATH)\n"
    "TINY_SESSION=runtime.Session(model=FAKE_SESSION.model, tokenizer=TOK, torch=FakeTorch(), in_dev='cpu', eos_id=999, pad_id=999, deadline=time.monotonic()+100.0, model_dir='cpu-fake', base_identity={'fake': 'cpu'}, journal=TINY_JOUR, runtime_identity={'fake': 'cpu'}, active_identity={'kind': 'base'}, save_window_s=300.0, allow_cached=True)\n"
    "tiny_res=runner.run_program_stage(TINY_SESSION, TINY_STAGE, TASKS, deadline=time.monotonic()+100.0, out_dir=TINY_OUT, remaining_later_calls=3)\n"
    "assert all(t.get('status')=='not_run' for t in tiny_res.get('tasks', [])), tiny_res\n"
    "assert any(t.get('reason')=='global_admission_insufficient' for t in tiny_res.get('tasks', [])), tiny_res\n"
    "assert FAKE_SESSION.model.gen_calls==gen_before, (gen_before, FAKE_SESSION.model.gen_calls)\n"
    "print('CPU_FULLPATH_NOT_RUN_OK', len(tiny_res.get('tasks', [])))\n"
    "ID_CODE='def transform(grid):\\n    return [row[:] for row in grid]'\n"
    "r_id=solve_program.execute_grid(ID_CODE, [[1, 2], [3, 4]], wall_seconds=2.0, memory_mb=512)\n"
    "if r_id.get('status')=='executor_unavailable':\n"
    "    raise RuntimeError('executor_unavailable fail-closed identity')\n"
    "assert r_id.get('status')=='ok' and r_id.get('grid')==[[1, 2], [3, 4]], r_id\n"
    "OOB_CODE='def transform(grid):\\n    return [[10]]'\n"
    "r_oob=solve_program.execute_grid(OOB_CODE, [[1]], wall_seconds=2.0, memory_mb=512)\n"
    "if r_oob.get('status')=='executor_unavailable':\n"
    "    raise RuntimeError('executor_unavailable fail-closed oob')\n"
    "assert r_oob.get('status')!='ok', r_oob\n"
    "LOOP_CODE='def transform(grid):\\n    while True:\\n        pass\\n    return [row[:] for row in grid]'\n"
    "r_loop=solve_program.execute_grid(LOOP_CODE, [[1]], wall_seconds=0.5, memory_mb=512)\n"
    "if r_loop.get('status')=='executor_unavailable':\n"
    "    raise RuntimeError('executor_unavailable fail-closed loop')\n"
    "assert r_loop.get('status')=='execution_timeout', r_loop\n"
    "IMP_CODE='import os\\ndef transform(grid):\\n    return [row[:] for row in grid]'\n"
    "r_imp=solve_program.execute_grid(IMP_CODE, [[1]], wall_seconds=2.0, memory_mb=512)\n"
    "if r_imp.get('status')=='executor_unavailable':\n"
    "    raise RuntimeError('executor_unavailable fail-closed import')\n"
    "assert r_imp.get('status')=='program_rejected', r_imp\n"
    "print('CPU_FULLPATH_EXECUTOR_OK', r_id.get('status'), r_oob.get('status'), r_loop.get('status'), r_imp.get('status'))\n"
    "SUMMARY={'full_plan_calls': 4, 'modes': [c.get('mode') for c in plan_calls], 'per_stage_generations': [r.get('physical_generation_calls') for r in RESULTS], 'repair_total': sum(r.get('repair_invoked_count', 0) for r in RESULTS), 'not_run_group': len(tiny_res.get('tasks', [])), 'executor': [r_id.get('status'), r_oob.get('status'), r_loop.get('status'), r_imp.get('status')]}\n"
    "assert SUMMARY['full_plan_calls']==4 and SUMMARY['modes']==['D', 'L', 'L', 'D'], SUMMARY\n"
    "assert SUMMARY['per_stage_generations']==[1, 1, 1, 1], SUMMARY\n"
    "assert SUMMARY['repair_total']==0, SUMMARY\n"
    "assert all(t.get('status')=='not_run' for t in tiny_res.get('tasks', [])), SUMMARY\n"
    "pathlib.Path('/kaggle/working/012/cpu-preflight/fullpath-summary.json').write_text(json.dumps(SUMMARY, indent=2))\n"
    "print('CPU_FULLPATH_OK', json.dumps(SUMMARY))\n"
)


GPU_RECEIPT = r'''
receipts=[]
for p in pathlib.Path('/kaggle/input').rglob('receipt.json'):
    try:
        value=json.loads(p.read_text())
        if value.get('core_manifest_sha256')==CORE_MANIFEST_SHA and value.get('status')=='passed':
            receipts.append(str(p))
    except (ValueError,OSError):
        pass
assert receipts,'Matching full CPU receipt must be mounted before GPU work'
print('MATCHED_CPU_RECEIPT',receipts[0],flush=True)
'''

CPU_EXTRA = r'''
import hashlib, subprocess, sys
from transformers import AutoTokenizer
real_tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
prompt_meta=[]
for tid,think in PLAN:
    msgs=[{'role':'system','content':solve_program.SYSTEM_PROMPT},
          {'role':'user','content':json.dumps(solve_program.task_inputs_only(TASKS[tid]),separators=(',',':'))}]
    prompt=real_tok.apply_chat_template(msgs,tokenize=False,add_generation_prompt=True,
        enable_thinking=think,reasoning_effort='low',preserve_thinking=True)
    n=len(real_tok(prompt,add_special_tokens=False)['input_ids'])
    assert n<8192
    if think: assert prompt.endswith('<think>\n')
    prompt_meta.append({'task_id':tid,'mode':'L' if think else 'D','prompt_tokens':n,
                        'sha256':hashlib.sha256(prompt.encode()).hexdigest()})

# Exercise the real worker and four stages, including whole-group admission.
original_load=runtime.load_session
worker_results=[]
try:
    for label,work_window in [('full',4200),('insufficient',3000)]:
        q=[FIXED_D_RAW,FIXED_L_RAW,FIXED_L_RAW,FIXED_D_RAW]
        tok=FakeTokenizer(q)
        out=pathlib.Path('/kaggle/working/012/cpu-preflight')/('worker-'+label)
        out.mkdir(parents=True,exist_ok=True)
        deadline=time.monotonic()+work_window+300
        sess=runtime.Session(model=FakeModel(q),tokenizer=tok,torch=FakeTorch(),in_dev='cpu',
            eos_id=999,pad_id=999,deadline=deadline,model_dir='cpu-fake',base_identity={'fake':'cpu'},
            journal=runtime.CandidateJournal(str(out/'candidates.jsonl')),
            runtime_identity={'fake':'cpu'},active_identity={'kind':'base'},save_window_s=300,allow_cached=False)
        runtime.load_session=lambda *args,**kwargs:sess
        cfg=json.loads((CODE/'run-p.json').read_text())
        cfg['out_dir']=str(out)
        cfg['runtime'].pop('setup_script',None)
        assert runner.worker(cfg,deadline)==0
        saved=json.loads((out/'SUMMARY.json').read_text())
        if label=='full':
            assert sess.model.gen_calls==4 and saved['work_complete'],saved
        else:
            assert sess.model.gen_calls==0
            assert len(saved['stages'])==4
            assert all(t['status']=='not_run' for s in saved['stages'] for t in s['tasks'])
        worker_results.append({'case':label,'calls':sess.model.gen_calls,'stages':len(saved['stages'])})
finally:
    runtime.load_session=original_load

hang_out='/kaggle/working/012/cpu-preflight/k-hang'
p=subprocess.run([sys.executable,str(CODE/'probe_gdn_inference.py'),'--simulate-hang',
    '--prompt-file',str(CODE/'db118e2a-query0-f1-prompt.txt'),
    '--output-dir',hang_out,'--hang-budget-s','0.3','--hang-child-secs','5'],
    capture_output=True,text=True,timeout=10)
print(p.stdout,flush=True)
assert p.returncode==0,p.stderr

receipt={'status':'passed','core_manifest_sha256':CORE_MANIFEST_SHA,
    'model_weights_loaded':False,'gpu_used':False,'fullpath':SUMMARY,
    'worker_cases':worker_results,'real_tokenizer_prompts':prompt_meta,
    'kernel_silent_child_timeout_returncode':p.returncode,
    'limits':'CPU orchestration only; CUDA kernels, 27B outputs and speed unverified'}
pathlib.Path('/kaggle/working/012/cpu-preflight/receipt.json').write_text(json.dumps(receipt,indent=2))
print('CPU_RECEIPT',json.dumps(receipt),flush=True)
'''


def make_cell(kind, source, index):
    item = {"cell_type": kind, "metadata": {}, "id": "arc012-%02d" % index, "source": source.strip() + "\n"}
    if kind == "code":
        ast.parse(item["source"])
        item.update(execution_count=None, outputs=[])
    return item


def with_gpu_telemetry(source):
    import textwrap
    return (
        "import subprocess\n"
        "telemetry_file=open(CODE/'gpu-telemetry.csv','w')\n"
        "telemetry=subprocess.Popen(['nvidia-smi','--query-gpu=timestamp,index,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw','--format=csv','-l','10'],stdout=telemetry_file,stderr=subprocess.STDOUT)\n"
        "try:\n" + textwrap.indent(source, "    ") +
        "\nfinally:\n"
        "    telemetry.terminate()\n"
        "    try: telemetry.wait(timeout=5)\n"
        "    except subprocess.TimeoutExpired:\n"
        "        telemetry.kill(); telemetry.wait(timeout=5)\n"
        "    telemetry_file.close()\n"
    )


def notebook_for(mode, setup_src, install_src):
    slug, title = SLUGS[mode], TITLES[mode]
    assert re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") == slug, (title, slug)
    head = ("# " + title + "\n\nLocal preparation only; publication and execution are separate. "
            "Plan 012. Fixed Qwen3.8-27B BF16, Kaggle L4x4, no internet. "
            "K mounts the new wheel Dataset, P does not; no S4 adapter.")
    secs = [("markdown", head),
            ("markdown", "### 1. Restore hashed code and task packets (scoring excluded from K)"),
            ("code", setup_src)]
    if mode == "cpu-preflight":
        secs += [("markdown", "### 2. Offline CPU dependencies"),("code",install_src),
                 ("markdown", "### 3. Fixed model inputs and executable configuration"),("code",RESOLVE+CONFIG_P),
                 ("markdown", "### 4. CPU input guards (no GPU, no weights)"), ("code", CPU_CHECKS),
                 ("markdown", "### 5. CPU full-path P0-only (FakeModel, no weights)"), ("code", CPU_FULLPATH),
                 ("markdown", "### 6. Real templates, whole worker admission and silent-child timeout"),
                 ("code",CPU_EXTRA)]
    elif mode == "kernel-probe":
        # The probe parent starts before installation; the worker executes this
        # frozen setup under its install/env deadline, importing the stack once.
        preparation=install_src+RESOLVE+CONFIG_K
        launch=GPU_RECEIPT+"\nimport sys\nsys.path.insert(0,str(CODE))\n"+RESOLVE
        launch+="\nsetup_path=CODE/'setup-k.py'\nsetup_path.write_text("+repr(preparation)+")\n"
        launch+="import probe_gdn_inference as probe\n"
        launch+="rc=probe.main(['--model-dir',model_dir,'--output-dir',str(CODE/'output'),'--prompt-file',str(CODE/'db118e2a-query0-f1-prompt.txt'),'--journal',str(CODE/'journal.jsonl'),'--notebook-start-monotonic',str(NOTEBOOK_START),'--setup-script',str(setup_path)])\nassert rc==0,rc\n"
        secs += [("markdown", "### 2. Run installation and one-load probe under the 30-minute parent deadline"),
                 ("code",with_gpu_telemetry(launch))]
    else:
        launch=GPU_RECEIPT+RESOLVE+CONFIG_P
        launch+="setup_path=CODE/'setup-p.py'\nsetup_path.write_text("+repr(install_src)+")\n"
        launch+="config['runtime']['setup_script']=str(setup_path)\n(CODE/'run-p.json').write_text(json.dumps(config,indent=2))\n"
        launch+=RUN_P
        secs += [("markdown", "### 2. One supervised setup/load and four P0 calls within 75 minutes"),
                 ("code",with_gpu_telemetry(launch))]
    secs += [("markdown", "### Interpretation\n\nCPU/K/P evidence stays separate. "
              "P2 is development data, not unseen validation. No score is claimed here.")]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [make_cell(k, s, i) for i, (k, s) in enumerate(secs)]}


def verify_entries(staged, per_task):
    """Real entry checks on the exact notebook CONFIG cells (CPU-safe paths only).

    probe.main(['--dry-run-cpu']) runs in-process; the K config cell runs the
    real probe --dry-run-cpu CLI in a subprocess; the P config cell runs the
    real runner.read_config plus ProgramLimits/split_program_response; then the
    real runner --dry-run CLI validates the produced run-p.json. RUN_K/RUN_P
    (GPU/model loads) are wired but never executed here.
    """
    import subprocess
    import sys
    import tempfile
    import types
    sys.path.insert(0, str(ROOT / "scripts"))
    import probe_gdn_inference as probe
    assert probe.main(["--dry-run-cpu", "--prompt-file", "validate-only-prompt.txt"]) == 0
    import run_budgeted_grid_experiment as runner
    tmp = Path(tempfile.mkdtemp(prefix="012-packager-validate-"))
    for name, data in {**staged, **per_task}.items():
        (tmp / name).write_bytes(data)
    probe_src = ROOT / "scripts" / "probe_gdn_inference.py"
    (tmp / "probe_gdn_inference.py").write_bytes(probe_src.read_bytes())  # same bytes as packed core
    import time
    ns = {"CODE": tmp, "model_dir": "validate-only-not-a-checkpoint", "NOTEBOOK_START": time.monotonic()}
    exec(CONFIG_P, dict(ns))
    cfg = json.loads((tmp / "run-p.json").read_text())
    assert cfg["time_budget_s"] == 4500 and cfg["load_limit_s"] == 900
    assert cfg["runtime"] == {"base_source": MODEL, "save_window_s": 300, "deactivate_async_load": True}
    assert [(s["name"], s.get("enable_thinking")) for s in cfg["stages"]] == [
        ("p0-db118e2a-D", False), ("p0-db118e2a-L", True),
        ("p0-291dc1e1-L", True), ("p0-291dc1e1-D", False)]
    for stage in cfg["stages"]:
        assert stage["limits"] == P0_FROZEN_LIMITS, stage["name"]
    assert runner.main(["--config", str(tmp / "run-p.json"), "--dry-run"]) == 0
    # Local CPU has no 27B tokenizer; stub only the tokenizer backend so the exact
    # K config cell still executes end to end (real build_messages on real staged
    # inputs, real write_f1_prompt_with_tokenizer, real run-k.json, real probe
    # --dry-run-cpu subprocess). Kaggle runtime uses the real AutoTokenizer.
    fake_tf = types.ModuleType("transformers")
    class _StubTok:
        def encode(self, text, **kwargs):
            return list(range(64))
        def apply_chat_template(self, messages, **kw):
            assert kw.get("tokenize") is False and kw.get("add_generation_prompt") is True
            assert kw.get("enable_thinking") is False
            return "STUB-F1-PROMPT|" + json.dumps(messages)[:200]
    class _StubAuto:
        @staticmethod
        def from_pretrained(model_dir, **kw):
            assert kw.get("local_files_only") is True
            return _StubTok()
    fake_tf.AutoTokenizer = _StubAuto
    sys.modules["transformers"] = fake_tf
    try:
        (tmp / "p2-scoring.json").unlink()  # mirror the K payload, which excludes scoring
        exec(CONFIG_K, dict(ns))
    finally:
        sys.modules.pop("transformers", None)
    kcfg = json.loads((tmp / "run-k.json").read_text())
    assert kcfg["order"] == ["A1", "B1", "B2", "A2"] and kcfg["fixed_decode_tokens"] == 64
    prompt_on_disk = (tmp / "db118e2a-query0-f1-prompt.txt").read_text(encoding="utf-8")
    assert prompt_on_disk.startswith("STUB-F1-PROMPT|")
    assert kcfg["prompt_sha256"] == hashlib.sha256(prompt_on_disk.encode()).hexdigest()
    assert kcfg["prompt_tokens"] == 64
    proc = subprocess.run([sys.executable, str(ROOT / "scripts" / "probe_gdn_inference.py"),
                           "--help"], capture_output=True, text=True, timeout=60)
    assert proc.returncode == 0 and "--dry-run-cpu" in proc.stdout
    return {"probe_dry_run_cpu": 0, "runner_dry_run": 0,
            "stages": [s["name"] for s in cfg["stages"]]}


def build_all(cpu_version, k_wheel_dataset):
    staged, per_task, inputs_manifest, wheel_manifest = stage_inputs(
        write=False, k_wheel_dataset=k_wheel_dataset)
    core_packed, core_manifest, core_sha, missing = load_core()
    validation = verify_entries(staged, per_task)
    out, k_blocked = {}, None
    for mode in ("cpu-preflight", "program-completion", "kernel-probe"):
        if mode == "kernel-probe" and not k_wheel_dataset:
            k_blocked = ("K package blocked: --k-wheel-dataset is empty; publish wheels as "
                         + K_WHEEL_DATASET + " and re-run. CPU/P packages are unaffected.")
            continue
        blob = payload_for(mode, core_packed, staged, per_task)
        code_manifest = {n: sha_bytes(d) for n, d in sorted(blob.items())}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, d in sorted(blob.items()):
                z.writestr(n, d)
        setup = SETUP_TMPL.format(code_dir="/kaggle/working/012/" + ("code" if mode == "cpu-preflight" else mode),
                                  manifest=repr(code_manifest), core=repr(core_manifest),
                                  sha=repr(core_sha), b64=repr(base64.b64encode(buf.getvalue()).decode()))
        install = "import time\nPREP_DEADLINE=NOTEBOOK_START+%d\n" % (120 if mode == "kernel-probe" else 300)
        # Install all optional wheels before importing Transformers so its
        # availability caches cannot retain the pre-install backend state.
        install += (INSTALL_K_EXTRA.format(wheels=repr(WHEELS)) if mode == "kernel-probe" else "") + INSTALL_BASE
        nb = notebook_for(mode, setup, install)
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
        slug = SLUGS[mode]
        ds = [WHEELHOUSE] + ([k_wheel_dataset] if mode == "kernel-probe" and k_wheel_dataset else [])
        meta = {"id": "ccjjllt/" + slug, "title": TITLES[mode], "code_file": slug + ".ipynb",
                "language": "python", "kernel_type": "notebook", "is_private": True,
                "enable_gpu": mode != "cpu-preflight", "enable_tpu": False, "enable_internet": False,
                "competition_sources": [COMPETITION], "dataset_sources": ds, "model_sources": [MODEL],
                "kernel_sources": [], "docker_image": IMAGE}
        if mode != "cpu-preflight":
            meta["machine_shape"] = "NvidiaL4"
            meta["kernel_sources"] = ["ccjjllt/%s/%d" % (SLUGS["cpu-preflight"], cpu_version)]
        info = {"mode": mode, "slug": slug, "core_manifest_sha256": core_sha,
                "missing_implementations": missing, "entry_validation": validation,
                "push_performed": False, "gpu_started": False,
                "notebook_seconds": {"cpu-preflight": None, "kernel-probe": 1800, "program-completion": 4500}[mode]}
        out[mode] = (blob, code_manifest, buf.getvalue(), nb, meta, info)
    return staged, per_task, inputs_manifest, wheel_manifest, core_manifest, core_sha, missing, out, k_blocked


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true", help="Build in memory; write nothing.")
    ap.add_argument("--stage-only", action="store_true", help="Only stage experiments/012/inputs + wheel manifest.")
    ap.add_argument("--cpu-version", type=int, default=1, help="Future CPU Notebook version mounted by GPU metadata.")
    ap.add_argument("--k-wheel-dataset", default=K_WHEEL_DATASET,
                    help="Published new-wheel Dataset slug for K (empty = K blocked, CPU/P unaffected).")
    ap.add_argument("--out-suffix", default="",
                    help="Package dir suffix, e.g. -v2 writes experiments/012/kaggle-v2 (history kept).")
    args = ap.parse_args(argv)
    if args.cpu_version < 1:
        ap.error("--cpu-version must be positive")
    if args.out_suffix and not re.fullmatch(r"[-_A-Za-z0-9]+", args.out_suffix):
        ap.error("--out-suffix must match [-_A-Za-z0-9]+")
    out_root = BASE / ("kaggle" + args.out_suffix)
    if args.check_only and args.stage_only:
        ap.error("choose at most one of --check-only/--stage-only")
    if args.check_only:
        staged, per_task, inputs_manifest, wheel_manifest, core_manifest, core_sha, missing, out, k_blocked = build_all(
            args.cpu_version, args.k_wheel_dataset or None)
        if k_blocked:
            print(json.dumps({"status": "K_BLOCKED", "reason": k_blocked}, indent=2))
            raise RuntimeError(k_blocked)
        k_blob = out["kernel-probe"][0]
        assert "p2-scoring.json" not in k_blob, "K package must exclude scoring"
        assert "p2-scoring.json" in staged, "scoring staging missing; K exclusion unverifiable"
        print(json.dumps({"status": "CHECKED_IN_MEMORY", "core_manifest_sha256": core_sha,
                          "inputs": sorted(inputs_manifest), "k_excludes_scoring": True,
                          "wheel_manifest": wheel_manifest,
                          "entry_validation": out["program-completion"][5]["entry_validation"],
                          "missing": missing, "push_performed": False, "gpu_started": False}, indent=2))
        return
    staged, per_task, inputs_manifest, wheel_manifest = stage_inputs(
        write=True, k_wheel_dataset=args.k_wheel_dataset or None)
    if args.stage_only:
        k_note = "K package would exclude p2-scoring.json; staged inputs retain it separately"
        print(json.dumps({"status": "INPUTS_STAGED", "inputs_manifest": inputs_manifest,
                          "k_excludes_scoring": True, "note": k_note,
                          "push_performed": False, "gpu_started": False}, indent=2))
        return
    _, _, _, _, core_manifest, core_sha, missing, out, k_blocked = build_all(
        args.cpu_version, args.k_wheel_dataset or None)
    existing = [str(out_root / m) for m in MODES if (out_root / m).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing notebook packages: " + ", ".join(existing))
    # Re-verify live core bytes before committing either package.
    live, _, _, _ = load_core()
    assert {n: sha_bytes(d) for n, d in sorted(live.items()) if n.endswith(".py")} == core_manifest
    for mode in out:
        blob, code_manifest, _, nb, meta, info = out[mode]
        folder = out_root / mode
        folder.mkdir(parents=True)
        (folder / (SLUGS[mode] + ".ipynb")).write_bytes(json_bytes(nb))
        (folder / "kernel-metadata.json").write_bytes(json_bytes(meta))
        (folder / "code-manifest.json").write_bytes(json_bytes(code_manifest))
        (folder / "core-manifest.json").write_bytes(json_bytes(core_manifest))
        (folder / "package-info.json").write_bytes(json_bytes(info))
        run_cfg = {"mode": mode, "slug": SLUGS[mode], "model": MODEL, "docker": IMAGE,
                   "k_wheel_dataset": args.k_wheel_dataset or None,
                   "out_root": str(out_root.relative_to(ROOT)),
                   "core_manifest_sha256": core_sha, "missing": missing}
        (folder / "run.json").write_bytes(json_bytes(run_cfg))
    rel = str(out_root.relative_to(ROOT))
    print(json.dumps({"status": "LOCAL_PACKAGES_CREATED_NOT_RUN", "core_manifest_sha256": core_sha,
                      "outputs": [rel + "/" + m for m in out],
                      "k_blocked": k_blocked,
                      "k_excludes_scoring": True, "missing": missing,
                      "push_performed": False, "gpu_started": False}, indent=2))
    if k_blocked:
        raise RuntimeError(k_blocked + " CPU/P packages above are already written.")


if __name__ == "__main__":
    main()
