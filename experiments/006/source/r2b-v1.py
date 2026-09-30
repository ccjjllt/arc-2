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
import gc, glob, json, os, random, sys, time
print("== R2b T8/T16补数探针 v1 ==", flush=True)
R = {"final": "PENDING", "exp": "R2b"}
base = tok = enc = out = opt = peft_m = None
T_START = time.time()
T_BUDGET = 3300
T_SAFE = 3000
MAX_NEW = 512
N_TASKS = 4
LR = 2e-5
STEPS8 = 8
STEPS16 = 16
SEED = 0
# 固定4题: 2个R1已解 + 2个R1未解(默认5582e5ca/d037b0a7)。
# R2输出缺失(本地无/tmp/r2*stdout,仅R2v1口头结论T0=4/T8=4零增减),按默认不替换。
R2B_TIDS = ["66e6c45b", "e9afcf9a", "5582e5ca", "d037b0a7"]
# T0不重跑:直接引用R1同tid exact-match@2结论hardcode。
# 来源:/tmp/r1_stdout.txt "== R1 no-TTT inference baseline v1 ==" R1_SUMMARY
#   66e6c45b cand0 greedy em=True / cand1 sample_t07 em=True / oracle=True
#   e9afcf9a cand0 greedy em=True / cand1 sample_t07 em=True / oracle=True
#   5582e5ca cand0 greedy em=False trunc=True / cand1 sample em=False trunc=True / oracle=False
#   d037b0a7 cand0 greedy em=False trunc=True / cand1 sample em=False trunc=True / oracle=False
R1_T0 = {
    "66e6c45b": {"greedy_em": True, "sample_em": True, "oracle": True},
    "e9afcf9a": {"greedy_em": True, "sample_em": True, "oracle": True},
    "5582e5ca": {"greedy_em": False, "sample_em": False, "oracle": False},
    "d037b0a7": {"greedy_em": False, "sample_em": False, "oracle": False},
}
print("R1_T0_SOURCE=/tmp/r1_stdout.txt R1 no-TTT baseline; hardcode, T0 not rerun", flush=True)
for _tid in R2B_TIDS:
    _v = R1_T0.get(_tid, {})
    print(f"R1_T0 tid={_tid} greedy={_v.get('greedy_em')} sample={_v.get('sample_em')} oracle={_v.get('oracle')}", flush=True)
# 随机种子说明(诚实注明):
#  能固定的:同一episode文本/同一token ids、LoRA同构同参、lr同值、gen同参(greedy+temp0.7),
#   每次建adapter/每次生成前seed_everything(SEED+task_idx)重置python/random/torch/cuda RNG。
#  不能固定的:多卡device_map分片并发、CUDA kernel非确定性、transformers generate采样未传显式
#   generator对象(依赖全局RNG,并发下仍可能抖动)。故T8/T16前8步loss轨迹可比但不承诺bit一致。
def seed_everything(s):
    try:
        random.seed(s)
    except Exception:
        pass
    try:
        import torch as _tt
        _tt.manual_seed(s)
        if _tt.cuda.is_available():
            _tt.cuda.manual_seed_all(s)
    except Exception:
        pass
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
    if not MODEL_DIR:
        raise RuntimeError("SKIP_NO_MODEL: no qwen3.5/3 model dir")
    print("MODEL_DIR:", MODEL_DIR, flush=True)
    base_comp = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2"
    ch_paths = glob.glob(base_comp + "/arc-agi_training_challenges.json")
    so_paths = glob.glob(base_comp + "/arc-agi_training_solutions.json")
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
    # --- R1/R2-verified helpers (沿用R2v1逻辑) ---
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
    def build_ttt_text(tid):
        # TTT TRAIN ONLY: known train demos (input+output). No test input, no test truth.
        # 真值(demo outputs)仅在此作训练目标;推理prompt永不接触;test真值仅事后统计。
        t = ch[tid]
        L = []
        for i, pr in enumerate(t.get("train", [])):
            L += [f"[demo {i}] in:", g2t(pr["input"]), f"[demo {i}] out:", g2t(pr["output"])]
        if not L:
            raise ValueError("no train demos for TTT")
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
        # Probe template; thinking off when supported (enable_thinking=False).
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
    def parse_grid(text):
        # R1-verified: parse candidate text back to grid; never raise.
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
    # --- task selection: 固定R2B 4题,缺失则回退short-first补齐(绝不ERROR) ---
    sel = []
    sel_mode = "fixed_r2b"
    try:
        fixed = []
        for tid in R2B_TIDS:
            if tid not in ch or tid not in so:
                raise ValueError(f"fixed tid missing: {tid}")
            t = ch[tid]
            outs = so.get(tid, [])
            if not outs or not t.get("test"):
                raise ValueError(f"fixed tid no test/solution: {tid}")
            u = build_user(tid)
            ptxt, pmode = render_prompt(u)
            n = len(tok(ptxt)["input_ids"])
            fixed.append((n, tid, ptxt, pmode))
        if len(fixed) != N_TASKS:
            raise ValueError("fixed count mismatch")
        sel = fixed
        sel_mode = "fixed_r2b"
    except Exception as e:
        print(f"fixed selection failed ({str(e)[:150]}), fallback to short-first", flush=True)
        sel_mode = "fallback_short"
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
        if len(short) < N_TASKS:
            raise RuntimeError(f"SKIP_NO_SHORT: only {len(short)} short tasks")
        sel = short[:N_TASKS]
    R["selection_mode"] = sel_mode
    R["prompt_mode_probe"] = sel[0][3] if sel else "unknown"
    R["task_ids"] = [t for _, t, _, _ in sel]
    R["prompt_lens"] = [n for n, _, _, _ in sel]
    R["r1_t0_ref"] = {k: dict(v) for k, v in R1_T0.items()}
    R["seed"] = SEED
    print(f"SEL mode={sel_mode} tids={R['task_ids']} lens={R['prompt_lens']}", flush=True)
    # --- load frozen BF16 base (single load, shared offline checkpoint) ---
    try:
        tL0 = time.time()
        base = AutoModelForCausalLM.from_pretrained(MODEL_DIR, dtype=torch.bfloat16, device_map="balanced", low_cpu_mem_usage=True, trust_remote_code=False)
        base.eval()
        try:
            base.gradient_checkpointing_enable()
            base.enable_input_require_grads()
            R["grad_ckpt"] = True
        except Exception as e:
            R["grad_ckpt"] = f"skip:{str(e)[:100]}"
        base.config.use_cache = True
        torch.cuda.synchronize()
        R["load_s"] = round(time.time() - tL0, 1)
        print(f"model loaded in {R['load_s']}s", flush=True)
        try:
            print("device_map:", getattr(base, "hf_device_map", None), flush=True)
        except Exception:
            pass
    except BaseException as e:
        raise RuntimeError(f"FAILED_LOAD: {str(e)[:300]}")
    try:
        in_dev = base.get_input_embeddings().weight.device
    except Exception:
        in_dev = next(base.parameters()).device
    dm = getattr(base, "hf_device_map", None)
    if dm is None:
        try:
            dm = base.base_model.model.hf_device_map
        except Exception:
            dm = {}
    try:
        lm_dev = torch.device(dm.get("lm_head", str(in_dev))) if isinstance(dm, dict) else in_dev
    except Exception:
        lm_dev = in_dev
    print("input_device:", in_dev, "lm_head_device:", lm_dev, flush=True)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    eos_id = tok.eos_token_id
    # --- 同一episode预计算:T8/T16共用同一ttt_ids(同一切片),绝不ERROR ---
    EP = {}
    for _ti, (_plen, _tid, _ptxt, _pmode) in enumerate(sel):
        try:
            _txt = build_ttt_text(_tid)
        except Exception as e:
            EP[_tid] = {"status": f"EPISODE_FAIL:{str(e)[:120]}", "ttt_ids": []}
            print(f"  EP {_tid} FAIL: {str(e)[:150]}", flush=True)
            continue
        try:
            _ids = tok(_txt)["input_ids"]
        except Exception as e:
            EP[_tid] = {"status": f"EPISODE_TOKENIZE_FAIL:{str(e)[:120]}", "ttt_ids": []}
            print(f"  EP {_tid} TOKENIZE_FAIL: {str(e)[:150]}", flush=True)
            continue
        if len(_ids) > 4096:
            print(f"  EP {_tid} too long ({len(_ids)}), trunc carve to 4096 (demos only)", flush=True)
            _ids = _ids[:4096]
        EP[_tid] = {"status": "OK", "ttt_ids": _ids}
        R.setdefault("ttt_ep_lens", {})[_tid] = len(_ids)
    print(f"EP ready: {[(k, R.get('ttt_ep_lens', {}).get(k, -1), v.get('status')) for k, v in EP.items()]}", flush=True)
    # --- helpers: generation (R1-verified) + TTT (R0L-verified) ---
    def gen_two(ptxt, gmodel, truth, seed):
        # Same 2 candidates every group: greedy + temp0.7 sampling, MAX_NEW<=512, thinking off via prompt.
        # Truth used only for post-hoc EM; never fed to model.
        recs = []
        for ci, do_sample in enumerate([False, True]):
            tag = "greedy" if not do_sample else "sample_t07"
            rec = {"kind": tag, "status": "PENDING"}
            try:
                seed_everything(seed + ci)
                torch.cuda.synchronize()
                t_pre = time.time()
                enc_l = tok(ptxt, return_tensors="pt")
                ids_l = enc_l["input_ids"].to(in_dev)
                attn_l = enc_l["attention_mask"].to(in_dev) if "attention_mask" in enc_l else None
                torch.cuda.synchronize()
                pre_s = time.time() - t_pre
                pn = int(ids_l.shape[1])
                gk = dict(input_ids=ids_l, attention_mask=attn_l, max_new_tokens=MAX_NEW, pad_token_id=pad_id, eos_token_id=eos_id)
                if do_sample:
                    gk.update(dict(do_sample=True, temperature=0.7, top_p=0.9))
                else:
                    gk.update(dict(do_sample=False))
                torch.cuda.synchronize()
                t_g = time.time()
                with torch.inference_mode():
                    out_l = gmodel.generate(**gk)
                torch.cuda.synchronize()
                dec_s = time.time() - t_g
                gen_ids = out_l[0][pn:]
                gn = int(gen_ids.numel())
                txt = tok.decode(gen_ids, skip_special_tokens=True)
                trunc = bool(gn >= MAX_NEW)
                pred, pstat = parse_grid(txt)
                em = bool(pred is not None and pred == truth)
                rec.update({"status": "OK", "prefill_s": round(pre_s, 3), "decode_s": round(dec_s, 3), "prompt_tok": pn, "gen_tok": gn, "trunc": trunc, "parse": pstat, "em": em, "text_head": txt[:300]})
                print(f"    cand{ci} {tag}: gen_tok={gn} trunc={trunc} parse={pstat} em={em} pre={pre_s:.2f}s dec={dec_s:.2f}s", flush=True)
                del out_l, gen_ids
                del enc_l, ids_l
                if attn_l is not None:
                    del attn_l
            except BaseException as e:
                rec.update({"status": f"GEN_FAIL:{str(e)[:120]}"})
                print(f"    cand{ci} {tag} GEN_FAIL: {str(e)[:200]}", flush=True)
            recs.append(rec)
        return recs
    def fresh_lora(seed):
        # Fresh adapter from same offline checkpoint; never reuse across tasks/groups.
        # seed固定保证T8/T16起点可比(仍受CUDA非确定性影响,见顶部注明)。
        seed_everything(seed)
        lcfg = LoraConfig(r=8, lora_alpha=32, target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], lora_dropout=0.0, bias="none", task_type="CAUSAL_LM")
        return get_peft_model(base, lcfg)
    def run_ttt(pm, ttt_ids, nsteps):
        # R0L-verified: single-episode batch, lr=2e-5, labels on lm_head device, sync timing.
        # Returns (ttt_s, losses, step_times, steps_done, status).
        losses, stimes = [], []
        ttt_s = 0.0
        status = "OK"
        try:
            tr_params = [p for p in pm.parameters() if p.requires_grad]
            if not tr_params:
                return 0.0, [], [], 0, "TTT_SKIP_no_trainable"
            o = torch.optim.AdamW(tr_params, lr=LR)
            try:
                for i in range(torch.cuda.device_count()):
                    torch.cuda.reset_peak_memory_stats(i)
            except Exception:
                pass
            pm.train()
            try:
                pm.config.use_cache = False
            except Exception:
                pass
            try:
                base.config.use_cache = False
            except Exception:
                pass
            t_all = time.time()
            done = 0
            for s in range(nsteps):
                if time.time() - T_START > T_SAFE:
                    status = "TTT_TRUNCATED_time"
                    break
                try:
                    x = torch.tensor([ttt_ids], dtype=torch.long).to(in_dev)
                    y = torch.tensor([ttt_ids], dtype=torch.long).to(lm_dev)
                    torch.cuda.synchronize()
                    t0 = time.time()
                    o.zero_grad(set_to_none=True)
                    oo = pm(input_ids=x, labels=y)
                    ls = oo.loss
                    ls.backward()
                    o.step()
                    torch.cuda.synchronize()
                    dt = time.time() - t0
                    losses.append(round(float(ls.detach()), 4))
                    stimes.append(round(dt, 1))
                    print(f"      ttt step {s+1}/{nsteps}: loss={losses[-1]} {dt:.1f}s", flush=True)
                    done += 1
                    del oo, ls, x, y
                except BaseException as e:
                    status = f"TTT_STEP_FAIL:{str(e)[:100]}"
                    print(f"      ttt step {s} FAIL: {str(e)[:200]}", flush=True)
                    break
            ttt_s = round(time.time() - t_all, 1)
            try:
                del o
            except Exception:
                pass
            try:
                pm.eval()
                pm.config.use_cache = True
            except Exception:
                pass
            try:
                base.config.use_cache = True
            except Exception:
                pass
            return ttt_s, losses, stimes, done, status
        except BaseException as e:
            return round(ttt_s, 1), losses, stimes, len(losses), f"TTT_FAIL:{str(e)[:120]}"
    print(f"BUDGET: {N_TASKS} tasks x groups[T8/T16] x2 cand x{MAX_NEW} tok, T_SAFE={T_SAFE}s (T0 hardcode, not rerun)", flush=True)
    print("NOTE: T8 then T16, shared episode+seed per tid; T16 lowest priority, may TRUNCATE.", flush=True)
    groups_cfg = [("T8", STEPS8), ("T16", STEPS16)]
    G = {}
    for gname, nsteps in groups_cfg:
        g_t0 = time.time()
        det = []
        truncated = False
        # T16最低优先级:预留120s,保证组内记账完整(TIME_GUARD沿用R2v1)。
        if gname == "T16" and (time.time() - T_START) > (T_SAFE - 120):
            truncated = True
            print(f"TIME_GUARD: skip {gname} entirely, elapsed={time.time()-T_START:.1f}s", flush=True)
            G[gname] = {"n_steps_cfg": nsteps, "details": [], "truncated": True, "skipped_entirely": True}
            R["t16_truncated"] = True
            continue
        for ti, (plen, tid, ptxt, pmode) in enumerate(sel):
            el = time.time() - T_START
            if el > T_SAFE:
                truncated = True
                print(f"TIME_GUARD: {gname} stop before task {tid} ({ti+1}/{len(sel)}), elapsed={el:.1f}s", flush=True)
                break
            print(f"-- [{gname}] task {ti+1}/{len(sel)} {tid} prompt_tok={plen} mode={pmode} elapsed={el:.1f}s", flush=True)
            truth = None
            try:
                truth = so[tid][0]
            except Exception as e:
                print(f"  no truth for {tid}: {str(e)[:120]}", flush=True)
                det.append({"tid": tid, "status": "NO_TRUTH", "ttt_s": 0.0, "gen_s": 0.0, "oracle": False})
                continue
            t_task = time.time()
            ttt_s = 0.0
            ttt_losses, ttt_times, ttt_done, ttt_status = [], [], 0, "NA"
            cands_out = []
            gmodel = base
            peft_active = False
            # TTT组:每题独立adapter重置,同episode同seed逻辑。
            try:
                ep = EP.get(tid, {})
                ttt_ids = ep.get("ttt_ids", [])
                if ep.get("status") != "OK" or not ttt_ids:
                    raise RuntimeError(f"TTT_EPISODE_FAIL: {str(ep.get('status'))[:150]}")
                try:
                    peft_m = fresh_lora(SEED + ti)
                    peft_active = True
                except BaseException as e:
                    raise RuntimeError(f"TTT_ADAPTER_FAIL: {str(e)[:150]}")
                ttt_s, ttt_losses, ttt_times, ttt_done, ttt_status = run_ttt(peft_m, ttt_ids, nsteps)
                print(f"  TTT {gname} {tid}: done={ttt_done}/{nsteps} ttt_s={ttt_s}s status={ttt_status}", flush=True)
                try:
                    t_g0 = time.time()
                    cands_out = gen_two(ptxt, peft_m, truth, SEED + ti)
                    gen_s = round(time.time() - t_g0, 1)
                except BaseException as e:
                    cands_out = [{"kind": "greedy", "status": f"GEN_FAIL:{str(e)[:100]}"}, {"kind": "sample_t07", "status": "GEN_FAIL:skipped"}]
                    gen_s = 0.0
            except BaseException as e:
                msg_e = str(e)[:200]
                print(f"  TTT path FAIL {tid}: {msg_e}", flush=True)
                ttt_status = msg_e[:150]
                # Fallback:仍尝试frozen生成,组间对比优雅降级(绝不中断)。
                try:
                    base.eval()
                    t_g0 = time.time()
                    cands_out = gen_two(ptxt, base, truth, SEED + ti)
                    gen_s = round(time.time() - t_g0, 1)
                except Exception:
                    cands_out = [{"kind": "greedy", "status": "GEN_FAIL"}, {"kind": "sample_t07", "status": "GEN_FAIL"}]
                    gen_s = 0.0
            finally:
                # Reset adapter: unload so next task/group starts from same offline checkpoint.
                if peft_active:
                    try:
                        tmp = peft_m.unload()
                        try:
                            base = tmp
                        except Exception:
                            pass
                    except Exception:
                        try:
                            peft_m.delete_adapter("default")
                        except Exception:
                            pass
                    try:
                        del peft_m
                    except Exception:
                        pass
                    peft_m = None
                try:
                    base.eval()
                    base.config.use_cache = True
                except Exception:
                    pass
                gc.collect()
            g_em = bool(cands_out[0].get("em")) if len(cands_out) > 0 else False
            s_em = bool(cands_out[1].get("em")) if len(cands_out) > 1 else False
            oracle = bool(g_em or s_em)
            task_s = round(time.time() - t_task, 1)
            try:
                thw = [len(truth), len(truth[0]) if truth else 0]
            except Exception:
                thw = [0, 0]
            # 相对T0新增/丢失(该组视角,T0为hardcode引用)。
            t0_or = bool(R1_T0.get(tid, {}).get("oracle", False))
            det.append({"tid": tid, "prompt_tok": plen, "truth_hw": thw, "t0_oracle_ref": t0_or, "ttt_s": round(float(ttt_s), 1), "ttt_done": int(ttt_done), "ttt_status": str(ttt_status)[:150], "ttt_losses": ttt_losses, "ttt_step_s": ttt_times, "gen_s": round(float(gen_s), 1), "greedy_em": bool(g_em), "sample_em": bool(s_em), "oracle": bool(oracle), "new_vs_T0": bool(oracle and not t0_or), "lost_vs_T0": bool(t0_or and not oracle), "task_s": task_s, "cands": cands_out})
            print(f"  [{gname}] task {tid} oracle={oracle} new_vs_T0={bool(oracle and not t0_or)} lost_vs_T0={bool(t0_or and not oracle)} ttt_s={ttt_s}s gen_s={gen_s}s task_s={task_s}s total_elapsed={time.time()-T_START:.1f}s", flush=True)
            # free per-task tensors
            try:
                del enc, out
            except Exception:
                pass
        # group summary
        try:
            n_or = sum(1 for d in det if d.get("oracle"))
            n_gr = sum(1 for d in det if d.get("cands", [{}])[0].get("em"))
            n_sa = sum(1 for d in det if len(d.get("cands", [])) > 1 and d["cands"][1].get("em"))
            ttt_sum = round(sum(float(d.get("ttt_s", 0.0)) for d in det), 1)
            gen_sum = round(sum(float(d.get("gen_s", 0.0)) for d in det), 1)
        except Exception:
            n_or, n_gr, n_sa, ttt_sum, gen_sum = 0, 0, 0, 0.0, 0.0
        G[gname] = {"n_steps_cfg": nsteps, "details": det, "n_tasks_done": len(det), "n_oracle": n_or, "n_greedy": n_gr, "n_sample": n_sa, "ttt_sum_s": ttt_sum, "gen_sum_s": gen_sum, "group_s": round(time.time() - g_t0, 1), "truncated": bool(truncated)}
        print(f"== [{gname}] done: oracle={n_or}/{len(det)} greedy={n_gr} sample={n_sa} ttt_sum={ttt_sum}s gen_sum={gen_sum}s group_s={G[gname]['group_s']}s truncated={truncated} ==", flush=True)
        if gname == "T16":
            R["t16_truncated"] = bool(truncated)
        if gname == "T8":
            R["t8_truncated"] = bool(truncated)
    # --- net-gain vs T0(hardcode引用) ---
    def solved_set(gname):
        try:
            return {d["tid"] for d in G.get(gname, {}).get("details", []) if d.get("oracle")}
        except Exception:
            return set()
    try:
        s0 = {tid for tid, v in R1_T0.items() if v.get("oracle")}
    except Exception:
        s0 = set()
    s8, s16 = solved_set("T8"), solved_set("T16")
    new8, lost8 = sorted(s8 - s0), sorted(s0 - s8)
    new16, lost16 = sorted(s16 - s0), sorted(s0 - s16)
    R.update({"groups": {k: {"n_steps_cfg": v.get("n_steps_cfg"), "n_tasks_done": v.get("n_tasks_done"), "n_oracle": v.get("n_oracle"), "n_greedy": v.get("n_greedy"), "n_sample": v.get("n_sample"), "ttt_sum_s": v.get("ttt_sum_s"), "gen_sum_s": v.get("gen_sum_s"), "group_s": v.get("group_s"), "truncated": v.get("truncated")} for k, v in G.items()}, "gain_vs_T0": {"T0_solved_ref": sorted(s0), "T8_solved": sorted(s8), "T16_solved": sorted(s16), "T8_new_vs_T0": new8, "T8_lost_vs_T0": lost8, "T16_new_vs_T0": new16, "T16_lost_vs_T0": lost16}, "total_s": round(time.time() - T_START, 1), "final": "OK"})
    # keep full per-task details but cap text heads already capped; store compact
    try:
        R["details"] = {k: v.get("details", []) for k, v in G.items()}
    except Exception:
        pass
    print(f"R2B_RESULT=OK T0ref={len(s0)} T8={len(s8)} T16={len(s16)} T8_new={new8} T8_lost={lost8} T16_new={new16} T16_lost={lost16} total={R['total_s']}s t16_trunc={R.get('t16_truncated')}")
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
    print(f"R2B_RESULT={tag} {msg}")
    print(_tb.format_exc()[-2000:])
finally:
    try:
        del base, tok, enc, out, opt, peft_m
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
print("R2B_SUMMARY=" + json.dumps(R, default=str))
print("R2B_DONE")
print("R2B_COMPLETE")

