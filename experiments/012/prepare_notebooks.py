"""Stage 012 inputs and build three local Kaggle packages. No push, no GPU.

Manual use (CPU, no network needed beyond local files):
  python3 experiments/012/prepare_notebooks.py --check-only   # in-memory only
  python3 experiments/012/prepare_notebooks.py --stage-only   # real inputs/ staging only
  python3 experiments/012/prepare_notebooks.py                # staging + three packages

Basis: docs/plan/012 sections 4 (rows 5-6), 5, 10. Code deltas in section 4 are
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


def stage_inputs(write: bool):
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
    wheel_manifest = {"status": "PINNED_NOT_UPLOADED",
                      "note": "Exact files/SHA from plan 012 section 5 (checked 2026-10-11). "
                              "New wheel Dataset has no uploaded slug in this handoff; K push is blocked "
                              "until k_wheel_dataset is filled with the real published slug/version.",
                      "python": "3.12", "torch": "2.10.0+cu128", "transformers": "5.8.0",
                      "base_wheelhouse": WHEELHOUSE, "k_wheel_dataset": None, "wheels": WHEELS}
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
    "import base64, hashlib, io, json, pathlib, time\nimport zipfile\n"
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
    "'transformers==5.8.0','accelerate','peft'],capture_output=True,text=True,timeout=240)\n"
    "print((p.stdout or '')[-1200:])\n"
    "if p.returncode:\n    print((p.stderr or '')[-1200:]);raise RuntimeError('Offline base install failed')\n"
    "import transformers,torch,peft,accelerate\n"
    "versions=dict(python=sys.version.split()[0],transformers=transformers.__version__,"
    "torch=torch.__version__,peft=peft.__version__,accelerate=accelerate.__version__)\n"
    "print('VERSIONS',json.dumps(versions))\n"
    "assert transformers.__version__=='5.8.0'\n"
)
INSTALL_K_EXTRA = (
    "import glob,subprocess,sys,hashlib,pathlib\n"
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
    "for w in want:\n"
    "    h=hashlib.sha256(open(resolved[w['file']],'rb').read()).hexdigest()\n"
    "    assert h==w['sha256'],w['file']\n"
    "    print('WHEEL_SHA_OK',pathlib.Path(resolved[w['file']]).name,h[:16])\n"
    "p=subprocess.run([sys.executable,'-m','pip','install','--no-index','--no-deps']"
    "+[resolved[w['file']] for w in want],capture_output=True,text=True,timeout=120)\n"
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
    "config={'notebook':'kernel-probe','notebook_seconds':1800,'load_limit_s':600,'save_window_s':120,\n"
    " 'model_dir':model_dir,'task':'db118e2a/query0','prompt':'f1_full_nonhthinking',\n"
    " 'fixed_decode_tokens':64,'order':['A1','B1','B2','A2'],'thresholds':{'a_rel_rms':0.0001,'b_rel_floor':0.01,'gain':0.15}}\n"
    "(CODE/'run-k.json').write_text(json.dumps(config,indent=2))\nprint(json.dumps(config,indent=2))\n"
    "assert not (CODE/'p2-scoring.json').exists(),'K must not stage scoring'\n"
    "import probe_gdn_inference as probe\nprint('PROBE_ENTRY',hasattr(probe,'main'))\n"
)
CONFIG_P = (
    "import json\n"
    "arms=[{'task':'db118e2a','arm':'D','thinking':False},{'task':'db118e2a','arm':'L','thinking':True},\n"
    " {'task':'291dc1e1','arm':'L','thinking':True},{'task':'291dc1e1','arm':'D','thinking':False}]\n"
    "config={'notebook':'program-completion','notebook_seconds':4500,'load_limit_s':600,'save_window_s':300,\n"
    " 'model_dir':model_dir,'policy':'p0_only','max_new_tokens':3072,'max_call_seconds':720,\n"
    " 'max_task_seconds':760,'max_prompt_tokens':8192,'seed_base':20261011,'order':arms}\n"
    "(CODE/'run-p.json').write_text(json.dumps(config,indent=2))\nprint(json.dumps(config,indent=2))\n"
    "import solve_program,run_budgeted_grid_experiment\n"
    "print('P0_ENTRY',hasattr(solve_program,'run_program_task'))\n"
)
RUN_K = (
    "import time\n"
    "elapsed=time.monotonic()-NOTEBOOK_START\nprint('HARD_LIMITS',{'notebook_seconds':1800,'already_elapsed_s':elapsed})\n"
    "print('K_STATUS','UNIMPLEMENTED until scripts/probe_gdn_inference.py lands; keep session closed')\n"
)
RUN_P = (
    "import time\n"
    "elapsed=time.monotonic()-NOTEBOOK_START\nprint('HARD_LIMITS',{'notebook_seconds':4500,'already_elapsed_s':elapsed})\n"
    "print('P_STATUS','UNIMPLEMENTED until solve_program p0_only lands; keep session closed')\n"
)
CPU_CHECKS = (
    "import json\n"
    "assert not __import__('torch').cuda.is_available(),'CPU preflight must not reserve GPU'\n"
    "tasks=json.loads((CODE/'p2-challenges.json').read_text())\n"
    "assert list(tasks)==['db118e2a','291dc1e1']\n"
    "print('CPU_INPUTS_OK',sorted(tasks))\n"
)


def make_cell(kind, source, index):
    item = {"cell_type": kind, "metadata": {}, "id": "arc012-%02d" % index, "source": source.strip() + "\n"}
    if kind == "code":
        ast.parse(item["source"])
        item.update(execution_count=None, outputs=[])
    return item


def notebook_for(mode, setup_src, install_src):
    slug, title = SLUGS[mode], TITLES[mode]
    assert re.sub(r"[^a-z0-9]+", "-", title.lower()).strip("-") == slug, (title, slug)
    head = ("# " + title + "\n\nLocal preparation only; publication and execution are separate. "
            "Plan 012. Fixed Qwen3.8-27B BF16, Kaggle L4x4, no internet. "
            "K mounts the new wheel Dataset, P does not; no S4 adapter.")
    secs = [("markdown", head),
            ("markdown", "### 1. Restore hashed code and task packets (scoring excluded from K)"),
            ("code", setup_src),
            ("markdown", "### 2. Offline dependencies (pinned, --no-index, no upgrades)"),
            ("code", install_src),
            ("markdown", "### 3. Model identity (base only, no adapter)"),
            ("code", RESOLVE)]
    if mode == "cpu-preflight":
        secs += [("markdown", "### 4. CPU input guards (no GPU, no weights)"), ("code", CPU_CHECKS)]
    elif mode == "kernel-probe":
        secs += [("markdown", "### 4. K config: one load, fixed 64-token A1/B1/B2/A2"), ("code", CONFIG_K),
                 ("markdown", "### 5. Run inside the 30-minute envelope (parent deadline, no retry)"),
                 ("code", RUN_K)]
    else:
        secs += [("markdown", "### 4. P config: P0-only D/L x2 tasks, 4 calls, 75-minute envelope"),
                 ("code", CONFIG_P),
                 ("markdown", "### 5. Run four P0 calls only (no restart/repair)"), ("code", RUN_P)]
    secs += [("markdown", "### Interpretation\n\nCPU/K/P evidence stays separate. "
              "P2 is development data, not unseen validation. No score is claimed here.")]
    return {"nbformat": 4, "nbformat_minor": 5,
            "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                         "language_info": {"name": "python"}},
            "cells": [make_cell(k, s, i) for i, (k, s) in enumerate(secs)]}


def build_all(cpu_version, k_wheel_dataset):
    staged, per_task, inputs_manifest, wheel_manifest = stage_inputs(write=False)
    core_packed, core_manifest, core_sha, missing = load_core()
    wheel_manifest = dict(wheel_manifest, k_wheel_dataset=k_wheel_dataset or None,
                          status="READY" if k_wheel_dataset else "PINNED_NOT_UPLOADED")
    out = {}
    for mode in MODES:
        blob = payload_for(mode, core_packed, staged, per_task)
        code_manifest = {n: sha_bytes(d) for n, d in sorted(blob.items())}
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
            for n, d in sorted(blob.items()):
                z.writestr(n, d)
        setup = SETUP_TMPL.format(code_dir="/kaggle/working/012/" + ("code" if mode == "cpu-preflight" else mode),
                                  manifest=repr(code_manifest), core=repr(core_manifest),
                                  sha=repr(core_sha), b64=repr(base64.b64encode(buf.getvalue()).decode()))
        install = INSTALL_BASE + (INSTALL_K_EXTRA.format(wheels=repr(WHEELS)) if mode == "kernel-probe" else "")
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
                "missing_implementations": missing,
                "push_performed": False, "gpu_started": False,
                "notebook_seconds": {"cpu-preflight": None, "kernel-probe": 1800, "program-completion": 4500}[mode]}
        out[mode] = (blob, code_manifest, buf.getvalue(), nb, meta, info)
    return staged, per_task, inputs_manifest, wheel_manifest, core_manifest, core_sha, missing, out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check-only", action="store_true", help="Build in memory; write nothing.")
    ap.add_argument("--stage-only", action="store_true", help="Only stage experiments/012/inputs + wheel manifest.")
    ap.add_argument("--cpu-version", type=int, default=1, help="Future CPU Notebook version mounted by GPU metadata.")
    ap.add_argument("--k-wheel-dataset", default="", help="Published new-wheel Dataset slug/version for K (empty = blocked).")
    args = ap.parse_args(argv)
    if args.cpu_version < 1:
        ap.error("--cpu-version must be positive")
    if args.check_only and args.stage_only:
        ap.error("choose at most one of --check-only/--stage-only")
    if args.check_only:
        staged, per_task, inputs_manifest, wheel_manifest, core_manifest, core_sha, missing, out = build_all(
            args.cpu_version, args.k_wheel_dataset or None)
        k_blob = out["kernel-probe"][0]
        assert "p2-scoring.json" not in k_blob, "K package must exclude scoring"
        assert "p2-scoring.json" in staged, "scoring staging missing; K exclusion unverifiable"
        print(json.dumps({"status": "CHECKED_IN_MEMORY", "core_manifest_sha256": core_sha,
                          "inputs": sorted(inputs_manifest), "k_excludes_scoring": True,
                          "missing": missing, "push_performed": False, "gpu_started": False}, indent=2))
        return
    staged, per_task, inputs_manifest, wheel_manifest = stage_inputs(write=True)
    if args.stage_only:
        k_note = "K package would exclude p2-scoring.json; staged inputs retain it separately"
        print(json.dumps({"status": "INPUTS_STAGED", "inputs_manifest": inputs_manifest,
                          "k_excludes_scoring": True, "note": k_note,
                          "push_performed": False, "gpu_started": False}, indent=2))
        return
    _, _, _, _, core_manifest, core_sha, missing, out = build_all(args.cpu_version, args.k_wheel_dataset or None)
    existing = [str(BASE / "kaggle" / m) for m in MODES if (BASE / "kaggle" / m).exists()]
    if existing:
        raise FileExistsError("Refusing to overwrite existing notebook packages: " + ", ".join(existing))
    # Re-verify live core bytes before committing either package.
    live, _, _, _ = load_core()
    assert {n: sha_bytes(d) for n, d in sorted(live.items()) if n.endswith(".py")} == core_manifest
    for mode in MODES:
        blob, code_manifest, _, nb, meta, info = out[mode]
        folder = BASE / "kaggle" / mode
        folder.mkdir(parents=True)
        (folder / (SLUGS[mode] + ".ipynb")).write_bytes(json_bytes(nb))
        (folder / "kernel-metadata.json").write_bytes(json_bytes(meta))
        (folder / "code-manifest.json").write_bytes(json_bytes(code_manifest))
        (folder / "core-manifest.json").write_bytes(json_bytes(core_manifest))
        (folder / "package-info.json").write_bytes(json_bytes(info))
        run_cfg = {"mode": mode, "slug": SLUGS[mode], "model": MODEL, "docker": IMAGE,
                   "k_wheel_dataset": args.k_wheel_dataset or None,
                   "core_manifest_sha256": core_sha, "missing": missing}
        (folder / "run.json").write_bytes(json_bytes(run_cfg))
    print(json.dumps({"status": "LOCAL_PACKAGES_CREATED_NOT_RUN", "core_manifest_sha256": core_sha,
                      "outputs": ["experiments/012/kaggle/" + m for m in MODES],
                      "k_excludes_scoring": True, "missing": missing,
                      "push_performed": False, "gpu_started": False}, indent=2))


if __name__ == "__main__":
    main()
