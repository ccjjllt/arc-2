"""Manually invoked packaging for the bounded 011 Kaggle experiments; no GPU here."""
import ast
import base64
import hashlib
import io
import json
from pathlib import Path
import zipfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/011"
IMAGE = "gcr.io/kaggle-private-byod/python@sha256:37c64f7dd9c54116ecd1bcc88817c5469b88387388fade02bfa8bf3fc647d461"
ADAPTER_SHA = "9282c46cbce02923afbea3ad041614f0bc6ff3d7bd97c5174adfa3dd3e485ff9"
PACK = {
    **{p.name: p for p in (ROOT / "scripts").glob("*.py") if p.name in {
        "grid_runtime.py", "run_budgeted_grid_experiment.py", "solve_grid.py",
        "arc_episode_data.py", "train_grid_sft.py", "ttt_per_task.py", "solve_program.py"}},
    "p12-challenges.json": ROOT / "experiments/010/e0-p12-challenges.json",
    "p12-scoring.json": ROOT / "experiments/010/kaggle-e0-v1/010/scoring.json",
    "p12-queries.json": BASE / "execution/p12-queries.json",
    "p6-queries.json": BASE / "execution/p6-queries.json",
    "research-manifest.json": BASE / "research-manifest.json",
}
INSTALL = r'''
import glob, os, subprocess, sys, time, json
wheel = glob.glob("/kaggle/input/**/qwen38-wheelhouse-py312", recursive=True)
if len(wheel) != 1:
    raise RuntimeError("Expected one frozen wheelhouse, found %d" % len(wheel))
subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"],
               stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT, timeout=90)
p = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index",
                    "--find-links=" + wheel[0], "transformers==5.8.0", "accelerate", "peft"],
                   capture_output=True, text=True, timeout=240)
print((p.stdout or "")[-1400:])
if p.returncode:
    print((p.stderr or "")[-1400:])
    raise RuntimeError("Offline environment preparation failed")
import transformers, torch, peft, accelerate
versions = dict(python=sys.version.split()[0], transformers=transformers.__version__,
                torch=torch.__version__, peft=peft.__version__, accelerate=accelerate.__version__)
print("VERSIONS", json.dumps(versions))
assert transformers.__version__ == "5.8.0"
'''
RESOLVE = r'''
from pathlib import Path
import hashlib
def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(2*1024*1024), b""):
            h.update(b)
    return h.hexdigest()
models = []
for p in Path("/kaggle/input").rglob("config.json"):
    try:
        c = json.loads(p.read_text())
    except Exception:
        continue
    if c.get("model_type") == "qwen3_5" and (p.parent/"model.safetensors.index.json").is_file():
        models.append(p.parent)
if len(models) != 1:
    raise RuntimeError("Expected one Qwen3.8 model directory, found %d" % len(models))
model_dir = str(models[0])
adapters = list(Path("/kaggle/input").rglob("adapter_step_25/adapter_model.safetensors"))
matching = [p.parent for p in adapters if digest(p) == EXPECTED_ADAPTER_SHA]
if len(matching) != 1:
    raise RuntimeError("Expected verified S4 adapter SHA, matching directories=%d" % len(matching))
adapter_dir = str(matching[0])
assert json.loads((matching[0]/"formatter.json").read_text())["formatter"] == "f0_space"
print("INPUT_IDENTITY", json.dumps(dict(model_dir=model_dir, adapter_dir=adapter_dir,
     adapter_sha256=EXPECTED_ADAPTER_SHA, model_config_sha256=digest(models[0]/"config.json"))))
sys.path.insert(0, str(CODE))
'''
CONFIG = r'''
config = {
 "model_dir": model_dir, "challenges": str(CODE/"p12-challenges.json"),
 "solutions": str(CODE/"p12-scoring.json"), "out_dir": "/kaggle/working/011/s0-shared",
 "time_budget_s": 7200, "load_limit_s": 1200,
 "notebook_start_monotonic": NOTEBOOK_START,
 "runtime": {"base_source":"kaggle:ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1",
             "deactivate_async_load":True, "save_window_s":300},
 "stages": [
  {"type":"grid","name":"s0_f0_25_p12","adapter":adapter_dir,"formatter":"f0_space",
   "query_file":str(CODE/"p12-queries.json"),"max_new_tokens":2048,
   "max_time_s":600,"estimate_s":240,
   "candidates":[{"view":"original","do_sample":False}]},
  {"type":"grid","name":"s1_base_f1_p6","adapter":None,"formatter":"f1_compact",
   "query_file":str(CODE/"p6-queries.json"),"max_new_tokens":960,
   "time_budget_s":1200,"max_time_s":300,"estimate_s":120,
   "candidates":[{"view":"original","do_sample":False},
                 {"view":"original","do_sample":True,"sample_index":1},
                 {"view":"transpose","do_sample":False}]}
 ]}
config_path=CODE/"run.json"
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
p=subprocess.run([sys.executable,str(CODE/"run_budgeted_grid_experiment.py"),
                  "--config",str(config_path),"--dry-run"],capture_output=True,text=True,timeout=60)
print(p.stdout)
if p.returncode:
    print(p.stderr[-1500:])
    raise RuntimeError("Configuration preflight failed")
'''
CPU = r'''
# Small manually run CPU diagnostics; no model weights or CUDA session are loaded.
from transformers import AutoTokenizer
from arc_episode_data import build_messages
import solve_program, ttt_per_task
from grid_runtime import CandidateJournal, canonical_hash
tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
tasks=json.loads((CODE/"p12-challenges.json").read_text())
sample=tasks["db118e2a"]
msg=build_messages(sample["train"],sample["test"][0]["input"],"f1_compact")
off=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,enable_thinking=False)
on=tok.apply_chat_template(msg,tokenize=False,add_generation_prompt=True,
                           enable_thinking=True,reasoning_effort="low",preserve_thinking=True)
assert off != on and on.endswith("<think>\n")
executor=solve_program.executor_preflight()
assert executor["status"]=="ok",executor
cases=[
 ("identity","def transform(grid):\n    return [row[:] for row in grid]",2.0,"ok"),
 ("exception","def transform(grid):\n    return grid[100]",2.0,"execution_error"),
 ("timeout","def transform(grid):\n    while True:\n        pass",0.2,"execution_timeout"),
 ("import","import os\ndef transform(grid):\n    return grid",2.0,"program_rejected"),
]
case_results={}
for name,code,wall,want in cases:
    result=solve_program.execute_grid(code,[[0,1],[2,3]],wall_seconds=wall,memory_mb=512)
    case_results[name]=result
    assert result["status"]==want,(name,result)
assert case_results["identity"]["grid"]==[[0,1],[2,3]]
assert not torch.cuda.is_available(),"CPU diagnostic must not reserve GPU"
report={"status":"CPU_PREFLIGHT_OK","model_weights_loaded":False,"gpu_count":torch.cuda.device_count(),
        "versions":versions,"code_manifest":CODE_MANIFEST,"adapter_sha256":EXPECTED_ADAPTER_SHA,
        "executor":executor,"case_results":case_results,
        "thinking_prompt_tokens":len(tok.encode(on,add_special_tokens=False)),
        "nonthinking_prompt_tokens":len(tok.encode(off,add_special_tokens=False)),
        "elapsed_s":time.monotonic()-NOTEBOOK_START}
Path("/kaggle/working/011/cpu-preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
'''
RUN = r'''
assert torch.cuda.device_count()==4,"Expected four L4 devices"
print("HARD_LIMITS",json.dumps({"notebook_seconds":7200,"load_seconds":1200,
      "already_elapsed_s":time.monotonic()-NOTEBOOK_START}))
remaining=7200-(time.monotonic()-NOTEBOOK_START)
if remaining < 900:
    raise RuntimeError("Insufficient budget after setup; refusing model load")
out=Path(config["out_dir"]);out.mkdir(parents=True,exist_ok=True)
telemetry_file=(out/"gpu-telemetry.csv").open("w")
telemetry=None
try:
    telemetry=subprocess.Popen(["nvidia-smi",
        "--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw",
        "--format=csv", "-l", "30"],stdout=telemetry_file,stderr=subprocess.DEVNULL)
    proc=subprocess.run([sys.executable,str(CODE/"run_budgeted_grid_experiment.py"),
                         "--config",str(config_path)],timeout=max(1,remaining+15))
finally:
    if telemetry is not None:
        telemetry.terminate()
        try:
            telemetry.wait(timeout=10)
        except subprocess.TimeoutExpired:
            telemetry.kill();telemetry.wait()
    telemetry_file.close()
print("EXPERIMENT_EXIT",proc.returncode)
for name in ["supervisor.json","SUMMARY.json"]:
    path=out/name
    if path.is_file():
        print(name,path.read_text()[:8000])
if proc.returncode not in (0,2):
    raise RuntimeError("Experiment infrastructure error, inspect worker.log")
'''

def make_cell(kind, source, index):
    item = {"cell_type":kind, "metadata":{}, "id":"arc011-%02d"%index, "source":source.strip()+"\n"}
    if kind == "code":
        ast.parse(item["source"])
        item.update(execution_count=None,outputs=[])
    return item

def main():
    assert set(PACK) >= {"grid_runtime.py","solve_program.py","ttt_per_task.py"}
    buf=io.BytesIO()
    manifest={}
    with zipfile.ZipFile(buf,"w",zipfile.ZIP_DEFLATED) as z:
        for name,path in sorted(PACK.items()):
            data=path.read_bytes()
            if name.endswith(".py"):
                ast.parse(data.decode())
            manifest[name]=hashlib.sha256(data).hexdigest()
            z.writestr(name,data)
    encoded=base64.b64encode(buf.getvalue()).decode()
    setup = (
      "import base64, hashlib, io, json, pathlib, time, zipfile\n"
      "NOTEBOOK_START=time.monotonic()\n"
      "CODE=pathlib.Path('/kaggle/working/011/code'); CODE.mkdir(parents=True,exist_ok=True)\n"
      "CODE_MANIFEST="+repr(manifest)+"\n"
      "EXPECTED_ADAPTER_SHA="+repr(ADAPTER_SHA)+"\n"
      "payload=base64.b64decode("+repr(encoded)+")\n"
      "with zipfile.ZipFile(io.BytesIO(payload)) as z:\n"
      "    for name in z.namelist():\n"
      "        if pathlib.PurePosixPath(name).name != name: raise ValueError('unsafe archive path')\n"
      "        data=z.read(name)\n"
      "        assert hashlib.sha256(data).hexdigest()==CODE_MANIFEST[name]\n"
      "        (CODE/name).write_bytes(data)\n"
      "(CODE/'source-manifest.json').write_text(json.dumps(CODE_MANIFEST,indent=2))\n"
      "print('PACKAGED_INPUTS',len(CODE_MANIFEST))\n"
    )
    for mode,slug,title in [
       ("cpu-preflight","arc2-011-cpu-preflight","ARC2 011 CPU preflight"),
       ("s0-shared","arc2-011-s0-shared-model","ARC2 011 S0 shared model")]:
        sections=[
          ("markdown","# "+title+"\n\nManual bounded research experiment, 2026-10-11. "
           "Fixed Qwen3.8-27B and verified S4 step25; no competition submission. "
           "Results below are produced by execution, not assumed. CPU preflight does not load model weights. "
           "GPU mode has a 120-minute notebook budget and a 20-minute model-load stop; it does not retry."),
          ("markdown","### 1. Restore frozen code and input-only task packets"),
          ("code",setup),
          ("markdown","### 2. Prepare the existing offline environment"),
          ("code",INSTALL),
          ("markdown","### 3. Verify model and adapter identities"),
          ("code",RESOLVE),
          ("markdown","### 4. Freeze the two development stages\n\nP12 F0-25 closes the previous comparison. "
           "P6 base F1 original/sample/transpose is a predeclared diagnostic. "
           "No confirmation tasks are loaded. Labels are available only to the post-generation scorer."),
          ("code",CONFIG),
          ("markdown","### 5. "+("CPU diagnostics" if mode=="cpu-preflight" else "Run within the hard budget")),
          ("code",CPU if mode=="cpu-preflight" else RUN),
          ("markdown","### Interpretation\n\nRead the generated JSON and worker journal. "
           "A successful CPU diagnostic establishes preparation only. "
           "GPU outcomes must include all requested queries, explicit incomplete cases, and actual timing. "
           "No result here is a competition score.")
        ]
        nb={"nbformat":4,"nbformat_minor":5,
            "metadata":{"kernelspec":{"display_name":"Python 3","language":"python","name":"python3"},
                        "language_info":{"name":"python"}},
            "cells":[make_cell(k,s,i) for i,(k,s) in enumerate(sections)]}
        folder=BASE/"kaggle"/mode;folder.mkdir(parents=True,exist_ok=True)
        filename=slug+".ipynb"
        (folder/filename).write_text(json.dumps(nb,ensure_ascii=False,indent=1)+"\n")
        metadata={"id":"ccjjllt/"+slug,"title":title,"code_file":filename,
           "language":"python","kernel_type":"notebook","is_private":True,
           "enable_gpu":mode!="cpu-preflight","enable_tpu":False,"enable_internet":False,
           "competition_sources":["arc-prize-2026-arc-agi-2"],
           "dataset_sources":["ccjjllt/qwen38-wheelhouse-py312"],
           "model_sources":["ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1"],
           "kernel_sources":["ccjjllt/s4-f0-train-010/1"],"docker_image":IMAGE}
        if mode!="cpu-preflight":
            metadata["machine_shape"]="NvidiaL4"
        (folder/"kernel-metadata.json").write_text(json.dumps(metadata,indent=2)+"\n")
        (folder/"code-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
        print(mode,"cells",len(nb["cells"]),"packed_bytes",len(buf.getvalue()),"notebook_bytes",(folder/filename).stat().st_size)
    (BASE/"execution"/"packaged-source-manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")

if __name__=="__main__":
    main()
