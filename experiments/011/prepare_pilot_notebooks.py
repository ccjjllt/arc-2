"""Build immutable local 011 P2 notebooks. Never pushes or starts a session.

This intentionally does not invoke prepare_execution_notebooks.main() or write
its output directories/manifests. CPU preflight validates the independent core
manifest, so the same report can authorize an S0 package with different inputs.
"""
from __future__ import annotations

import argparse
import ast
import base64
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import zipfile


ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "experiments/011"
PILOT_TASKS = ("db118e2a", "291dc1e1")
CORE_FILES = (
    "arc_episode_data.py", "grid_runtime.py", "run_budgeted_grid_experiment.py",
    "solve_grid.py", "solve_program.py", "train_grid_sft.py", "ttt_per_task.py",
)
FROZEN_FIXES = {
    "grid_runtime.py": "54b519deb5876f0f9f90008da69151b6ec4aa35b808b7c80d7d624f385c7d20f",
    "run_budgeted_grid_experiment.py": "af5ae4f2baca3f1c670f9b50fc94b98dc46c04eaeea1cf55ec34ea0497107e23",
}
CPU_SLUG = "arc2-011-cpu-pilot-preflight"
GPU_SLUG = "arc2-011-s1-pilots"

_spec = importlib.util.spec_from_file_location("arc011_pack_reference", BASE / "prepare_execution_notebooks.py")
_reference = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_reference)
INSTALL, RESOLVE, RUN = _reference.INSTALL, _reference.RESOLVE, _reference.RUN
IMAGE, ADAPTER_SHA, make_cell = _reference.IMAGE, _reference.ADAPTER_SHA, _reference.make_cell


CONFIG = r'''
config = {
 "model_dir": model_dir, "challenges": str(CODE/"p2-challenges.json"),
 "solutions": str(CODE/"p2-scoring.json"), "out_dir": "/kaggle/working/011/s1-pilots",
 "time_budget_s": 7200, "load_limit_s": 1200,
 "notebook_start_monotonic": NOTEBOOK_START,
 "runtime": {"base_source":"kaggle:ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1",
             "deactivate_async_load":True, "save_window_s":300},
 "stages": [
  {"type":"ttt","name":"s1_base_f1_ttt_p2","adapter":None,"formatter":"f1_compact",
   "ttt_adapter_name":"ttt_p2_base","query_file":str(CODE/"p2-queries.json"),
   "updates":4,"max_new_tokens":960,"seed":20261011,"time_budget_s":3600,
   "a4_adaptation_estimate_s":480,"a4_generation_reserve_s_per_query":480,
   "source_by_task":{"db118e2a":"H_new","291dc1e1":"E24"}},
  {"type":"program","name":"s2_base_program_p2","adapter":None,
   "query_file":str(CODE/"p2-queries.json"),"time_budget_s":2400,"pilot_tasks":2,
   "limits":{"max_new_tokens":1536,"max_prompt_tokens":8192,
             "max_call_seconds":360,"max_task_seconds":1080,
             "max_feedback_chars":3000,"per_grid_seconds":2,
             "per_program_seconds":20,"memory_mb":512,"seed":20261011}}
 ]}
config_path=CODE/"run.json"
config_path.write_text(json.dumps(config,ensure_ascii=False,indent=2))
p=subprocess.run([sys.executable,str(CODE/"run_budgeted_grid_experiment.py"),
                  "--config",str(config_path),"--dry-run"],capture_output=True,text=True,timeout=60)
print(p.stdout)
if p.returncode:
    print(p.stderr[-1500:])
    raise RuntimeError("Pilot configuration preflight failed")
'''


CPU = r'''
# Manual CPU diagnostics: real orchestration, synthetic generation, no 27B weights.
assert sys.platform.startswith("linux"),"Executor diagnostics require the Kaggle Linux image"
assert not torch.cuda.is_available(),"CPU preflight must not reserve a GPU"
from types import SimpleNamespace
from transformers import AutoTokenizer
import arc_episode_data, grid_runtime, run_budgeted_grid_experiment, solve_grid, solve_program, ttt_per_task, train_grid_sft
core_modules=[arc_episode_data,grid_runtime,run_budgeted_grid_experiment,solve_grid,
              solve_program,ttt_per_task,train_grid_sft]
for module in core_modules:
    path=Path(module.__file__).resolve()
    assert path.parent==CODE.resolve(),(module.__name__,str(path))
    assert digest(path)==CORE_MANIFEST[path.name],path.name
assert hasattr(run_budgeted_grid_experiment,"score_ttt_results")
assert hasattr(run_budgeted_grid_experiment,"stage_work_complete")
tasks=json.loads((CODE/"p2-challenges.json").read_text())
queries=json.loads((CODE/"p2-queries.json").read_text())
assert list(tasks)==list(PILOT_TASKS)
assert queries==[{"task_id":t,"query_index":0} for t in PILOT_TASKS]
assert all(set(q)=={"input"} for task in tasks.values() for q in task["test"])
assert all(stage.get("adapter") is None for stage in config["stages"])
tok=AutoTokenizer.from_pretrained(model_dir,local_files_only=True,trust_remote_code=False)
sample=tasks[PILOT_TASKS[0]]
messages=arc_episode_data.build_messages(sample["train"],sample["test"][0]["input"],"f1_compact")
off=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
on=tok.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,
                           enable_thinking=True,reasoning_effort="low",preserve_thinking=True)
assert off!=on and on.endswith("<think>\n")
loo=[]
for tid in PILOT_TASKS:
    pool,detail=ttt_per_task.build_ttt_episodes(tok,tasks[tid]["train"],tid,
                    formatter="f1_compact",eos_id=tok.eos_token_id,max_len=4096,seed=20261011)
    assert detail["status"]=="complete",detail
    for episode in pool:
        assert all(x==-100 for x in episode.labels[:episode.prompt_len])
        assert all(x!=-100 for x in episode.labels[episode.prompt_len:])
        assert episode.labels[-1]==tok.eos_token_id
    loo.append(detail)

# Exercise the real generate_grid -> deadline wrapper -> _run_one_candidate ->
# decode/parse -> candidate journal path. Only FakeModel.generate is synthetic.
# This catches event-name/"kind" collisions without downloading/loading a model.
class FakeModel:
    def __init__(self):
        self.config=SimpleNamespace(use_cache=True)
        self.generation_config=SimpleNamespace(cache_implementation=None)
        self.calls=0
        self.answer="12\n34"
    def eval(self):
        return self
    def gradient_checkpointing_disable(self):
        pass
    def generate(self,input_ids,**kwargs):
        self.calls+=1
        assert "stopping_criteria" in kwargs and len(kwargs["stopping_criteria"])>=1
        for stop in kwargs["stopping_criteria"]:
            assert not bool(stop(input_ids,None))
        if kwargs.get("do_sample"):
            torch.multinomial(torch.tensor([0.5,0.5]),1)
        tail=tok.encode(self.answer,add_special_tokens=False)+[tok.eos_token_id]
        assert len(tail)<kwargs["max_new_tokens"]
        return torch.cat([input_ids,torch.tensor([tail],dtype=input_ids.dtype,device=input_ids.device)],dim=1)
fake=FakeModel()
fixture_session=grid_runtime.Session(model=fake,tokenizer=tok,torch=torch,
    in_dev=torch.device("cpu"),eos_id=tok.eos_token_id,pad_id=tok.pad_token_id,
    deadline=time.monotonic()+120,model_dir="synthetic-no-weights",
    base_identity={"kind":"cpu_fixture"},
    journal=grid_runtime.CandidateJournal("/kaggle/working/011/cpu-pilot-candidates.jsonl"),
    active_identity={"kind":"cpu_fixture"},save_window_s=0,allow_cached=True)
fixture_demos=[{"input":[[0]],"output":[[0]]}]
fixture_rows=[]
for fmt,text in [("f0_space","1 2\n3 4"),("f1_compact","12\n34")]:
    fake.answer=text
    for view,sampling in [("original",False),("original",True),("transpose",False)]:
        record=grid_runtime.generate_grid(fixture_session,"cpu_fixture_"+fmt,0,
           fixture_demos,[[1,2],[3,4]],formatter=fmt,view=view,do_sample=sampling,
           sample_index=1 if sampling else 0,max_new_tokens=32,seed=20261011,
           deadline=fixture_session.deadline,run_tag="cpu-pilot-orchestration")
        assert record["status"]=="complete",record
        assert record["grid"]==([[1,3],[2,4]] if view=="transpose" else [[1,2],[3,4]])
        fixture_rows.append({k:record.get(k) for k in ("identity_key","view","kind","status","parse","cache_hit")})
calls_before=fake.calls
cached=grid_runtime.generate_grid(fixture_session,"cpu_fixture_f1_compact",0,
    fixture_demos,[[1,2],[3,4]],formatter="f1_compact",view="transpose",do_sample=False,
    max_new_tokens=32,seed=20261011,deadline=fixture_session.deadline,run_tag="cpu-pilot-orchestration")
assert cached["cache_hit"] and fake.calls==calls_before
expired=grid_runtime.generate_grid(fixture_session,"cpu_fixture_expired",0,
    fixture_demos,[[0]],formatter="f1_compact",max_new_tokens=32,
    deadline=time.monotonic()-1,run_tag="cpu-pilot-expired")
assert expired["status"]=="budget_interrupted" and fake.calls==calls_before

# Tiny randomly initialized HF model: exercise the actual PEFT/generate API on
# CPU for two tokens. This is an interface probe, not a Qwen3.8 forward/ARC score.
from transformers import Qwen3Config,Qwen3ForCausalLM
tiny_config=Qwen3Config(vocab_size=64,hidden_size=32,intermediate_size=64,
    num_hidden_layers=1,num_attention_heads=4,num_key_value_heads=2,head_dim=8,
    max_position_embeddings=64,bos_token_id=1,eos_token_id=2,pad_token_id=0)
tiny=Qwen3ForCausalLM(tiny_config).eval()
tiny,initial=ttt_per_task.ensure_ttt_adapter(tiny,adapter_name="cpu_ttt_interface",
                                           seed=20261011,torch_module=torch)
with torch.inference_mode():
    tiny_result=tiny.generate(input_ids=torch.tensor([[1,4,5]]),
        attention_mask=torch.ones((1,3),dtype=torch.long),max_new_tokens=2,
        do_sample=False,pad_token_id=0)
assert 3<int(tiny_result.shape[1])<=5
assert all(not p.requires_grad for p in tiny.parameters())
tiny_probe={"status":"ok","model":"random_tiny_Qwen3_not_Qwen3_8_27B",
            "device":"cpu","max_new_tokens":2,"generated_tokens":int(tiny_result.shape[1])-3,
            "lora_r":8,"lora_alpha":32,"initial_adapter_sha256":initial["adapter_sha256"]}
del tiny_result,tiny,initial

executor=solve_program.executor_preflight()
assert executor["status"]=="ok",executor
cases=[
 ("identity","def transform(grid):\n    return [row[:] for row in grid]",2.0,"ok"),
 ("exception","def transform(grid):\n    return grid[100]",2.0,"execution_error"),
 ("timeout","def transform(grid):\n    while True:\n        pass",0.2,"execution_timeout"),
 ("import","import os\ndef transform(grid):\n    return grid",2.0,"program_rejected")]
case_results={}
for name,code,wall,want in cases:
    result=solve_program.execute_grid(code,[[0,1],[2,3]],wall_seconds=wall,memory_mb=512)
    case_results[name]=result
    assert result["status"]==want,(name,result)
assert case_results["identity"]["grid"]==[[0,1],[2,3]]
report={"status":"CPU_PILOT_PREFLIGHT_OK","model_weights_loaded":False,
        "pretrained_model_weights_loaded":False,"synthetic_cpu_model_instantiated":True,
        "qwen38_27b_forward_ran":False,"gpu_count":torch.cuda.device_count(),
        "versions":versions,"core_manifest":CORE_MANIFEST,"core_manifest_sha256":CORE_MANIFEST_SHA,
        "payload_manifest":CODE_MANIFEST,"adapter_sha256":EXPECTED_ADAPTER_SHA,
        "pilot_tasks":list(PILOT_TASKS),"ttt_loo":loo,
        "config_validated":True,"configuration_runtime_gpu_verified":False,
        "candidate_orchestration":{"status":"ok","model_generate":"synthetic_stub",
            "rows":fixture_rows,"generate_calls":fake.calls,"journal_cache_hit":True,
            "expired_admission":"budget_interrupted"},
        "tiny_peft_generate":tiny_probe,"executor":executor,"case_results":case_results,
        "thinking_prompt_tokens":len(tok.encode(on,add_special_tokens=False)),
        "nonthinking_prompt_tokens":len(tok.encode(off,add_special_tokens=False)),
        "elapsed_s":time.monotonic()-NOTEBOOK_START}
Path("/kaggle/working/011/cpu-pilot-preflight.json").write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))
'''


CPU_REPORT_CHECK = r'''
# The preflight dependency is matched on core code only, not its task payload.
# This deliberately lets the same CPU report support a later S0 P12 package.
reports=[]
for path in Path("/kaggle/input").rglob("cpu-pilot-preflight.json"):
    try:
        report=json.loads(path.read_text())
    except Exception:
        continue
    if (report.get("status")=="CPU_PILOT_PREFLIGHT_OK"
            and report.get("core_manifest")==CORE_MANIFEST
            and report.get("core_manifest_sha256")==CORE_MANIFEST_SHA):
        reports.append((path,report))
if len(reports)!=1:
    raise RuntimeError("Expected one successful CPU pilot preflight with the identical core SHA set")
path,report=reports[0]
for key in ("python","transformers","torch","peft","accelerate"):
    if versions[key]!=report["versions"][key]:
        raise RuntimeError("CPU/GPU environment differs for "+key)
print("MATCHED_CPU_PREFLIGHT",json.dumps({"path":str(path),"core_manifest_sha256":CORE_MANIFEST_SHA,
      "payload_equality_required":False,"pretrained_weights_checked_by_cpu":False}))
'''


def digest_bytes(data):
    return hashlib.sha256(data).hexdigest()


def json_bytes(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + "\n").encode()


def build_payload():
    p6_path = BASE / "execution/p6-queries.json"
    source_queries = json.loads(p6_path.read_text())
    selected = source_queries[:2]
    assert selected == [{"task_id": tid, "query_index": 0} for tid in PILOT_TASKS]
    source_challenges = ROOT / "experiments/010/e0-p12-challenges.json"
    source_scoring = ROOT / "experiments/010/kaggle-e0-v1/010/scoring.json"
    task_map = json.loads(source_challenges.read_text())
    scoring_map = json.loads(source_scoring.read_text())
    tasks, scoring = {}, {}
    for tid in PILOT_TASKS:
        task = task_map[tid]
        if len(task["test"]) != 1 or len(scoring_map[tid]) != 1:
            raise ValueError("P2 assumes exactly one original query per task: " + tid)
        tasks[tid] = {"train": [{"input": pair["input"], "output": pair["output"]}
                                for pair in task["train"]],
                      "test": [{"input": task["test"][0]["input"]}]}
        scoring[tid] = [scoring_map[tid][0]]
    packed = {name: (ROOT / "scripts" / name).read_bytes() for name in CORE_FILES}
    for name, data in packed.items():
        ast.parse(data.decode(), filename=name)
    core_manifest = {name: digest_bytes(data) for name, data in sorted(packed.items())}
    for name, expected in FROZEN_FIXES.items():
        if core_manifest[name] != expected:
            raise RuntimeError("Core changed after the communicated freeze: " + name)
    packed.update({"p2-challenges.json": json_bytes(tasks), "p2-scoring.json": json_bytes(scoring),
                   "p2-queries.json": json_bytes(selected)})
    info = {"created_date": "2026-10-11", "status": "LOCAL_PREPARATION_NOT_RUN",
            "initialization": "base_default_pending_S0_final_operator_decision",
            "tasks": selected, "n_tasks": 2, "n_queries": 2,
            "sources": {str(path.relative_to(ROOT)): digest_bytes(path.read_bytes())
                        for path in (p6_path, source_challenges, source_scoring)},
            "selection": "First two interleaved P6 entries; unchanged task and query indices",
            "confirmation_data_included": False,
            "labels_use": "post-generation scorer only",
            "stage_limits_s": {"ttt_from_notebook_start_including_load": 3600,
                               "program_after_ttt": 2400, "notebook": 7200,
                               "load": 1200, "save_reserve": 300},
            "interpretation": "A0/A4 fixed O+T candidate comparison; not an equal-time TTT control"}
    packed["pilot-info.json"] = json_bytes(info)
    return packed, core_manifest, info


def build_setup(packed, core_manifest):
    manifest = {name: digest_bytes(data) for name, data in sorted(packed.items())}
    core_sha = digest_bytes(json.dumps(core_manifest, sort_keys=True, separators=(",", ":")).encode())
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as archive:
        for name, data in sorted(packed.items()):
            archive.writestr(name, data)
    encoded = base64.b64encode(buf.getvalue()).decode()
    setup = (
        "import base64, hashlib, io, json, pathlib, time, zipfile\n"
        "NOTEBOOK_START=time.monotonic()\n"
        "CODE=pathlib.Path('/kaggle/working/011/pilot-code'); CODE.mkdir(parents=True,exist_ok=True)\n"
        "CODE_MANIFEST=" + repr(manifest) + "\n"
        "CORE_MANIFEST=" + repr(core_manifest) + "\n"
        "CORE_MANIFEST_SHA=" + repr(core_sha) + "\n"
        "PILOT_TASKS=" + repr(PILOT_TASKS) + "\n"
        "EXPECTED_ADAPTER_SHA=" + repr(ADAPTER_SHA) + "\n"
        "payload=base64.b64decode(" + repr(encoded) + ")\n"
        "with zipfile.ZipFile(io.BytesIO(payload)) as z:\n"
        "    assert set(z.namelist())==set(CODE_MANIFEST)\n"
        "    for name in z.namelist():\n"
        "        if pathlib.PurePosixPath(name).name!=name: raise ValueError('unsafe archive path')\n"
        "        data=z.read(name)\n"
        "        assert hashlib.sha256(data).hexdigest()==CODE_MANIFEST[name]\n"
        "        (CODE/name).write_bytes(data)\n"
        "(CODE/'source-manifest.json').write_text(json.dumps(CODE_MANIFEST,indent=2))\n"
        "(CODE/'core-manifest.json').write_text(json.dumps(CORE_MANIFEST,indent=2))\n"
        "print('PILOT_CORE_SHA',CORE_MANIFEST_SHA)\n"
    )
    return setup, manifest, core_sha, len(buf.getvalue())


def notebook(mode, setup):
    cpu = mode == "cpu-pilot-preflight"
    slug = CPU_SLUG if cpu else GPU_SLUG
    title = "ARC2 011 CPU pilot preflight" if cpu else "ARC2 011 S1 pilots"
    assert re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") == slug
    sections = [
        ("markdown", "# " + title + "\n\nLocal preparation only; publication and execution are separate. "
         "2026-10-11. Fixed two development tasks; no confirmation tasks or competition submission. "
         "Base initialization is provisional pending the operator's S0 decision. "
         "CPU diagnostics load tokenizer assets and an optional tiny random interface model, never 27B weights."),
        ("markdown", "### 1. Restore independently hashed core code and P2-only inputs"),
        ("code", setup),
        ("markdown", "### 2. Prepare the frozen offline dependencies"), ("code", INSTALL),
        ("markdown", "### 3. Verify model and mounted S4 adapter identities\n\nThe S4 adapter is verified but neither pilot uses it."),
        ("code", RESOLVE),
        ("markdown", "### 4. Validate the bounded P2 configuration\n\nTTT: A0/A4, O+T, 4 updates × 4 accumulation. "
         "Its 3600-second stage limit includes initial setup/loading. Program: shared base, initial/restart/repair, "
         "1536 tokens / 360 seconds per call / 1080 seconds per task, stage limit 2400 seconds. "
         "The complete Notebook remains bounded by 7200 seconds, with 300 seconds reserved for saving."),
        ("code", CONFIG),
    ]
    if cpu:
        sections += [("markdown", "### 5. Linux CPU diagnostics, including the actual candidate orchestration path"),
                     ("code", CPU)]
    else:
        sections += [("markdown", "### 5. Match the successful CPU report by core SHA, independent of its input payload"),
                     ("code", CPU_REPORT_CHECK),
                     ("markdown", "### 6. Run only after the operator's final decision"), ("code", RUN)]
    sections += [("markdown", "### Interpretation\n\nA CPU pass is not a GPU gradient/throughput result. "
                  "TTT here tests fixed-candidate adaptation, not equal-time superiority. Program repair/restart "
                  "uses the same P0 and only demo feedback. Include incomplete work and actual costs; "
                  "a parsed grid or low training loss is not a solved task. These are development diagnostics, not competition scores.")]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [make_cell(kind, text, index) for index, (kind, text) in enumerate(sections)]}, slug, title


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true", help="Build and inspect in memory; write nothing.")
    parser.add_argument("--cpu-version", type=int, default=1,
                        help="Exact future successful CPU Notebook version mounted by GPU metadata; default 1.")
    args = parser.parse_args(argv)
    if args.cpu_version < 1:
        parser.error("--cpu-version must be positive")
    packed, core_manifest, info = build_payload()
    setup, payload_manifest, core_sha, packed_bytes = build_setup(packed, core_manifest)
    outputs = []
    for mode in ("cpu-pilot-preflight", "s1-pilots"):
        nb, slug, title = notebook(mode, setup)
        filename = slug + ".ipynb"
        metadata = {"id": "ccjjllt/" + slug, "title": title, "code_file": filename,
                    "language": "python", "kernel_type": "notebook", "is_private": True,
                    "enable_gpu": mode == "s1-pilots", "enable_tpu": False, "enable_internet": False,
                    "competition_sources": ["arc-prize-2026-arc-agi-2"],
                    "dataset_sources": ["ccjjllt/qwen38-wheelhouse-py312"],
                    "model_sources": ["ugvfpdcuwfnh/qwen3-8-27b/Transformers/transformers-bf16/1"],
                    "kernel_sources": ["ccjjllt/s4-f0-train-010/1"], "docker_image": IMAGE}
        if mode == "s1-pilots":
            metadata["machine_shape"] = "NvidiaL4"
            metadata["kernel_sources"].append("ccjjllt/%s/%d" % (CPU_SLUG, args.cpu_version))
        for cell in nb["cells"]:
            if cell["cell_type"] == "code":
                ast.parse(cell["source"])
        outputs.append((BASE / "kaggle" / mode, filename, nb, metadata))
    # Catch a concurrent core edit before committing either new package locally.
    assert core_manifest == {name: digest_bytes((ROOT / "scripts" / name).read_bytes()) for name in CORE_FILES}
    if not args.check_only:
        existing = [str(folder) for folder, *_ in outputs if folder.exists()]
        if existing:
            raise FileExistsError("Refusing to overwrite existing notebook packages: " + ", ".join(existing))
        for folder, filename, nb, metadata in outputs:
            folder.mkdir(parents=True)
            (folder / filename).write_bytes(json_bytes(nb))
            (folder / "kernel-metadata.json").write_bytes(json_bytes(metadata))
            (folder / "code-manifest.json").write_bytes(json_bytes(payload_manifest))
            (folder / "core-manifest.json").write_bytes(json_bytes(core_manifest))
            (folder / "package-info.json").write_bytes(json_bytes({**info,
                "core_manifest_sha256": core_sha, "cpu_report_match_scope": "core_only",
                "future_cpu_version_dependency": args.cpu_version}))
    print(json.dumps({"status": "CHECKED_IN_MEMORY" if args.check_only else "LOCAL_PACKAGES_CREATED_NOT_RUN",
        "core_manifest_sha256": core_sha, "core_manifest": core_manifest,
        "packed_bytes": packed_bytes, "packed_names": sorted(packed),
        "tasks": list(PILOT_TASKS), "outputs": [str(x[0].relative_to(ROOT)) for x in outputs],
        "push_performed": False, "gpu_started": False}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
