# Archived executed Notebook code; not rerun during 007 review.
# %% cell 0
import glob, os
WH = glob.glob("/kaggle/input/qwen38-wheelhouse-py312")[0] if glob.glob("/kaggle/input/qwen38-wheelhouse-py312") else None
print("WHEELHOUSE:", WH)
if WH:
    print(sorted(os.listdir(WH)))
import subprocess, sys
r0 = subprocess.run([sys.executable, "-m", "pip", "uninstall", "-y", "torchao"], capture_output=True, text=True)
print(r0.stdout[-500:])
r = subprocess.run([sys.executable, "-m", "pip", "install", "--no-index", f"--find-links={WH}", "transformers==5.8.0", "accelerate", "peft"], capture_output=True, text=True)
print(r.stdout[-1500:])
print(r.stderr[-500:])
import transformers
print("TF_NOW:", transformers.__version__, hasattr(transformers, "Qwen3_5ForCausalLM"))


# %% cell 1
import gc, glob, json, os, sys, time
print("== R0L short LoRA probe v2 ==", flush=True)
R = {"final": "PENDING"}
model = opt = enc = out = None
try:
    import torch
    assert torch.cuda.is_available() and torch.cuda.device_count() >= 4, "need 4x GPU"
    import transformers
    assert hasattr(transformers, "Qwen3_5ForCausalLM"), "need tf>=5.8"
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    cands = sorted(glob.glob("/kaggle/input/**/config.json", recursive=True))
    MODEL_DIR = None
    for c in cands:
        try:
            cfg = json.load(open(c))
        except Exception:
            continue
        if cfg.get("model_type") in ("qwen3_5", "qwen3"):
            MODEL_DIR = os.path.dirname(c)
            break
    assert MODEL_DIR, "no qwen model dir"
    print("MODEL_DIR:", MODEL_DIR)
    base = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
    ch = json.load(open(glob.glob(base + "/arc-agi_training_challenges.json")[0]))
    so = json.load(open(glob.glob(base + "/arc-agi_training_solutions.json")[0]))
    print(f"tasks: {len(ch)}, solutions: {len(so)}")
    def g2t(g):
        return "\n".join(" ".join(str(c) for c in r) for r in g)
    def episode(tid):
        t, outs = ch[tid], so.get(tid, [])
        assert outs, f"no solutions for {tid}"
        L = []
        for i, pr in enumerate(t.get("train", [])):
            L += [f"[demo {i}] in:", g2t(pr["input"]), f"[demo {i}] out:", g2t(pr["output"])]
        ti = t.get("test", [])
        assert ti and len(outs) >= 1, f"no test/solution for {tid}"
        L += ["[query] in:", g2t(ti[0]["input"]), "[query] out:", g2t(outs[0])]
        return "\n".join(L)
    tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    eps, skipped = [], 0
    for tid in sorted(ch.keys()):
        try:
            ids = tok(episode(tid))["input_ids"]
        except Exception as e:
            skipped += 1
            if skipped <= 3:
                print(f"skip {tid}:", str(e)[:120])
            continue
        if len(ids) <= 1024:
            eps.append(ids)
        if len(eps) >= 6:
            break
    print(f"scanned_skipped={skipped} kept={len(eps)} lens={[len(e) for e in eps]}")
    assert len(eps) >= 3, f"only {len(eps)} short episodes"
    R["ep_lens"] = [len(e) for e in eps]
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_DIR, dtype=torch.bfloat16, device_map="balanced",
        low_cpu_mem_usage=True, trust_remote_code=False)
    try:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
        R["grad_ckpt"] = True
    except Exception as e:
        R["grad_ckpt"] = f"skip:{str(e)[:100]}"
    model.config.use_cache = False
    print("NOTE linear_attn-specific projs NOT covered by LoRA (probe only)")
    lcfg = LoraConfig(r=8, lora_alpha=32,
                      target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                                      "gate_proj", "up_proj", "down_proj"],
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()
    ntr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("trainable_params:", ntr)
    R["trainable"] = ntr
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=2e-5)
    try:
        in_dev = model.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(model.parameters()).device
    dm = getattr(model, "hf_device_map", None)
    if dm is None:
        try:
            dm = model.base_model.model.hf_device_map
        except Exception:
            dm = {}
    lm_dev = torch.device(dm.get("lm_head", str(in_dev))) if isinstance(dm, dict) else in_dev
    print("input_device:", in_dev, "lm_head_device:", lm_dev)
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    model.train()
    losses, times = [], []
    NSTEPS = 6
    for s in range(NSTEPS):
        ids = eps[s % len(eps)]
        x = torch.tensor([ids], dtype=torch.long).to(in_dev)
        y = torch.tensor([ids], dtype=torch.long).to(lm_dev)
        torch.cuda.synchronize()
        t0 = time.time()
        opt.zero_grad(set_to_none=True)
        out = model(input_ids=x, labels=y)
        loss = out.loss
        loss.backward()
        opt.step()
        torch.cuda.synchronize()
        dt = time.time() - t0
        losses.append(round(float(loss.detach()), 4))
        times.append(round(dt, 1))
        print(f"step {s}: loss={losses[-1]} {dt:.1f}s", flush=True)
        del out, loss
    peak_a, peak_r = {}, {}
    for i in range(torch.cuda.device_count()):
        peak_a[f"gpu{i}"] = round(torch.cuda.max_memory_allocated(i) / 1e9, 2)
        peak_r[f"gpu{i}"] = round(torch.cuda.max_memory_reserved(i) / 1e9, 2)
    print("train_peak_alloc_GB:", peak_a)
    print("train_peak_reserved_GB:", peak_r)
    R.update({"steps": NSTEPS, "losses": losses, "step_s": times,
              "peak_alloc_gb": peak_a, "peak_reserved_gb": peak_r, "final": "OK"})
    print("R0L_RESULT=OK")
except BaseException as e:
    import traceback as _tb
    msg = str(e)[:500] if str(e) else type(e).__name__
    tag = "FAILED_CKPT" if "checkpoint" in msg.lower() else "FAILED"
    R.update({"final": tag, "error": msg})
    print(f"R0L_RESULT={tag}", msg)
    print(_tb.format_exc()[-2000:])
finally:
    try:
        del model, opt, enc, out
    except Exception:
        pass
    try:
        gc.collect()
        import torch as _t
        if _t.cuda.is_available():
            _t.cuda.empty_cache()
            _t.cuda.ipc_collect()
    except Exception:
        pass
print("R0L_SUMMARY=" + json.dumps(R))
print("R0L_DONE")
print("R0L_COMPLETE")

