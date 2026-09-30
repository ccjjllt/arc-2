# Archived executed Notebook code; not rerun during 007 review.
# %% cell 0
import glob, os
hits = glob.glob("/kaggle/input/**/qwen38-wheelhouse-py312", recursive=True)
WH = hits[0] if hits else None
print("WHEELHOUSE:", WH)
import glob as _g
print("INPUT_ROOT:", sorted(_g.glob("/kaggle/input/*")))
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
print("== R1 no-TTT inference baseline v1 ==", flush=True)
R = {"final": "PENDING"}
model = tok = enc = out = None
T_START = time.time()
T_BUDGET = 3600
T_SAFE = 3300
MAX_NEW = 512
N_TASKS = 8
try:
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("SKIP_NO_GPU: cuda unavailable")
    ngpu = torch.cuda.device_count()
    print("GPUs:", ngpu, flush=True)
    if ngpu < 4:
        raise RuntimeError(f"SKIP_GPU_COUNT: need 4xL4, got {ngpu}")
    import transformers
    print("TF:", transformers.__version__, hasattr(transformers, "Qwen3_5ForCausalLM"), flush=True)
    if not hasattr(transformers, "Qwen3_5ForCausalLM"):
        raise RuntimeError("SKIP_NO_MODEL: need transformers>=5.8 with Qwen3_5")
    from transformers import AutoTokenizer, AutoModelForCausalLM
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
    if not MODEL_DIR:
        raise RuntimeError("SKIP_NO_MODEL: no qwen3.5/3 model dir")
    print("MODEL_DIR:", MODEL_DIR, flush=True)
    base = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
    ch_paths = glob.glob(base + "/arc-agi_training_challenges.json")
    so_paths = glob.glob(base + "/arc-agi_training_solutions.json")
    if not ch_paths or not so_paths:
        raise RuntimeError("SKIP_NO_DATA: missing challenges/solutions")
    ch = json.load(open(ch_paths[0]))
    so = json.load(open(so_paths[0]))
    print(f"tasks: {len(ch)}, solutions: {len(so)}", flush=True)
    if not ch or not so:
        raise RuntimeError("SKIP_NO_DATA: empty challenges/solutions")
    try:
        tok = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
    except Exception as e:
        raise RuntimeError(f"SKIP_TOKENIZER: {str(e)[:200]}")
    print("tokenizer ok:", type(tok).__name__, flush=True)
    def g2t(g):
        return "\n".join(" ".join(str(c) for c in r) for r in g)
    SYS = "You are an ARC grid solver. Output ONLY the answer grid as rows of digits 0-9 separated by single spaces, one row per line. No explanation."
    def build_user(tid):
        # INPUT ONLY: demos (in+out) + query input. Truth never enters prompt.
        t = ch[tid]
        L = []
        for i, pr in enumerate(t.get("train", [])):
            L += [f"[demo {i}] in:", g2t(pr["input"]), f"[demo {i}] out:", g2t(pr["output"])]
        ti = t.get("test", [])
        if not ti:
            raise ValueError("no test input")
        L += ["[query] in:", g2t(ti[0]["input"]), "[query] out:"]
        return "\n".join(L)
    tpl = getattr(tok, "chat_template", None)
    has_apply = callable(getattr(tok, "apply_chat_template", None))
    R["chat_template_present"] = bool(tpl)
    try:
        import inspect as _insp
        sig_params = list(_insp.signature(tok.apply_chat_template).parameters.keys()) if has_apply else []
    except Exception:
        sig_params = []
    R["apply_params"] = sig_params[:20]
    try:
        R["chat_template_head"] = str(tpl)[:300] if tpl else None
    except Exception:
        R["chat_template_head"] = None
    def render_prompt(user_text):
        # Probe template support; never assume thinking param name.
        # Unknown -> most conservative direct concat + short output.
        msgs = [{"role": "system", "content": SYS}, {"role": "user", "content": user_text}]
        if has_apply and tpl:
            if "enable_thinking" in sig_params:
                try:
                    t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                    return t, "chat_enable_thinking_false"
                except Exception:
                    pass
            try:
                t = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
                return t, "chat_default"
            except Exception:
                pass
        return SYS + "\n" + user_text, "direct_concat"
    scored = []
    skipped_tok = 0
    for tid in sorted(ch.keys()):
        outs = so.get(tid, [])
        t = ch[tid]
        if not outs or not t.get("test"):
            continue
        try:
            u = build_user(tid)
        except Exception:
            continue
        ptxt, pmode = render_prompt(u)
        try:
            n = len(tok(ptxt)["input_ids"])
        except Exception:
            skipped_tok += 1
            continue
        scored.append((n, tid, ptxt, pmode))
    scored.sort(key=lambda x: x[0])
    short = [s for s in scored if s[0] < 1500][:N_TASKS]
    if len(short) < N_TASKS:
        rest = [s for s in scored if 1500 <= s[0] <= 2048 and s not in short][:N_TASKS - len(short)]
        short = short + rest
    print(f"scored={len(scored)} skipped_tok={skipped_tok} short_kept={len(short)} lens={[s[0] for s in short]}", flush=True)
    R["prompt_mode_probe"] = short[0][3] if short else "unknown"
    if len(short) < N_TASKS:
        raise RuntimeError(f"SKIP_NO_SHORT: only {len(short)} short tasks")
    sel = short[:N_TASKS]
    R["task_ids"] = [t for _, t, _, _ in sel]
    R["prompt_lens"] = [n for n, _, _, _ in sel]
    try:
        tL0 = time.time()
        model = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="balanced", low_cpu_mem_usage=True, trust_remote_code=False)
        model.eval()
        torch.cuda.synchronize()
        R["load_s"] = round(time.time() - tL0, 1)
        print(f"model loaded in {R['load_s']}s", flush=True)
        try:
            print("device_map:", getattr(model, "hf_device_map", None), flush=True)
        except Exception:
            pass
    except BaseException as e:
        raise RuntimeError(f"FAILED_LOAD: {str(e)[:300]}")
    try:
        in_dev = model.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(model.parameters()).device
    print("input_device:", in_dev, flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id
    def parse_grid(text):
        # Parse candidate text back to grid (digits 0-9/space/newline only).
        # Fail -> PARSE_FAIL marker, never raise.
        try:
            lines = text.strip().splitlines()
            groups = []
            cur = []
            for ln in lines:
                s = ln.strip().replace(",", " ").replace("|", " ")
                if not s:
                    if cur:
                        groups.append(cur)
                        cur = []
                    continue
                parts = s.split()
                if len(parts) == 1 and len(parts[0]) > 1 and all(x.isdigit() for x in parts[0]):
                    row = [int(x) for x in parts[0]]
                    if all(0 <= v <= 9 for v in row):
                        cur.append(row)
                        continue
                    else:
                        if cur:
                            groups.append(cur)
                            cur = []
                        continue
                ok = True
                row = []
                for p in parts:
                    if len(p) == 1 and p.isdigit():
                        row.append(int(p))
                    else:
                        ok = False
                        break
                if ok and row:
                    cur.append(row)
                else:
                    if cur:
                        groups.append(cur)
                        cur = []
            if cur:
                groups.append(cur)
            if not groups:
                return None, "PARSE_FAIL_empty"
            for g in reversed(groups):
                if not g:
                    continue
                w = len(g[0])
                if w == 0:
                    continue
                if all(len(r) == w for r in g):
                    return g, "OK"
            return None, "PARSE_FAIL_shape"
        except Exception as e:
            return None, f"PARSE_FAIL_exc:{str(e)[:60]}"
    print(f"BUDGET: {N_TASKS} tasks x2 cand x{MAX_NEW} tok upper bound, T_SAFE={T_SAFE}s", flush=True)
    details = []
    n_oracle = 0
    n_greedy = 0
    n_sample = 0
    for ti, (plen, tid, ptxt, pmode) in enumerate(sel):
        elapsed = time.time() - T_START
        print(f"-- task {ti+1}/{len(sel)} {tid} prompt_tok={plen} mode={pmode} elapsed={elapsed:.1f}s", flush=True)
        if elapsed > T_SAFE:
            print(f"TIME_GUARD: elapsed {elapsed:.1f}s > SAFE {T_SAFE}s, stop early", flush=True)
            R["time_guard_stop"] = True
            break
        truth = so[tid][0]
        t0_task = time.time()
        cands_out = []
        for ci, do_sample in enumerate([False, True]):
            tag = "greedy" if not do_sample else "sample_t07"
            rec = {"kind": tag, "status": "PENDING"}
            try:
                torch.cuda.synchronize()
                t_pre = time.time()
                enc = tok(ptxt, return_tensors="pt")
                ids = enc["input_ids"].to(in_dev)
                if "attention_mask" in enc:
                    attn = enc["attention_mask"].to(in_dev)
                else:
                    attn = None
                torch.cuda.synchronize()
                prefill_s = time.time() - t_pre
                prompt_n = ids.shape[1]
                gen_kwargs = dict(input_ids=ids, attention_mask=attn, max_new_tokens=MAX_NEW, pad_token_id=pad_id, eos_token_id=eos_id)
                if do_sample:
                    gen_kwargs.update(dict(do_sample=True, temperature=0.7, top_p=0.9))
                else:
                    gen_kwargs.update(dict(do_sample=False))
                torch.cuda.synchronize()
                t_gen = time.time()
                with torch.inference_mode():
                    out = model.generate(**gen_kwargs)
                torch.cuda.synchronize()
                decode_s = time.time() - t_gen
                gen_ids = out[0][prompt_n:]
                gen_n = int(gen_ids.numel())
                text = tok.decode(gen_ids, skip_special_tokens=True)
                trunc = bool(gen_n >= MAX_NEW)
                pred, pstat = parse_grid(text)
                em = bool(pred is not None and pred == truth)
                rec.update({"status": "OK", "prefill_s": round(prefill_s, 3), "decode_s": round(decode_s, 3), "prompt_tok": int(prompt_n), "gen_tok": int(gen_n), "trunc": trunc, "parse": pstat, "em": em, "text_head": text[:500]})
                print(f"  cand{ci} {tag}: gen_tok={gen_n} trunc={trunc} parse={pstat} em={em} pre={prefill_s:.2f}s dec={decode_s:.2f}s", flush=True)
                del out, gen_ids
            except BaseException as e:
                rec.update({"status": f"GEN_FAIL:{str(e)[:120]}"})
                print(f"  cand{ci} {tag} GEN_FAIL: {str(e)[:200]}", flush=True)
            cands_out.append(rec)
        g_em = bool(cands_out[0].get("em")) if len(cands_out) > 0 else False
        s_em = bool(cands_out[1].get("em")) if len(cands_out) > 1 else False
        oracle = bool(g_em or s_em)
        n_greedy += int(g_em)
        n_sample += int(s_em)
        n_oracle += int(oracle)
        task_s = round(time.time() - t0_task, 1)
        thw = [len(truth), len(truth[0]) if truth else 0]
        details.append({"tid": tid, "prompt_tok": plen, "truth_hw": thw, "greedy_em": g_em, "sample_em": s_em, "oracle": oracle, "task_s": task_s, "cands": cands_out})
        print(f"  task {tid} oracle={oracle} task_s={task_s}s total_elapsed={time.time()-T_START:.1f}s", flush=True)
    R.update({"n_tasks": len(details), "n_oracle": n_oracle, "n_greedy": n_greedy, "n_sample": n_sample, "em_any_rate": round(n_oracle / max(1, len(details)), 4), "details": details, "total_s": round(time.time() - T_START, 1), "final": "OK"})
    print(f"R1_RESULT=OK oracle={n_oracle}/{len(details)} total={R['total_s']}s")
except BaseException as e:
    import traceback as _tb
    msg = str(e)[:500] if str(e) else type(e).__name__
    tag = "FAILED"
    for cand in ("SKIP_NO_GPU", "SKIP_GPU_COUNT", "SKIP_NO_MODEL", "SKIP_NO_DATA", "SKIP_TOKENIZER", "SKIP_NO_SHORT", "FAILED_LOAD"):
        if cand in msg:
            tag = cand
            break
    else:
        if "SKIP_" in msg:
            tag = "FAILED_SKIP_UNKNOWN"
        elif "FAILED_" in msg:
            tag = "FAILED_GEN"
    R.update({"final": tag, "error": msg})
    print(f"R1_RESULT={tag} {msg}")
    print(_tb.format_exc()[-2000:])
finally:
    try:
        del model, tok, enc, out
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
print("R1_SUMMARY=" + json.dumps(R, default=str))
print("R1_DONE")
print("R1_COMPLETE")

