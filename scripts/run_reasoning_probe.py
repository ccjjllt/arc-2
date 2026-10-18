#!/usr/bin/env python3
"""V先导最小探针 (014计划第5节V,未在GPU运行).

只做:离线安装->TP4服务->单请求->小并发,记录reasoning/final区分、
token数/finish_reason/TTFT/吞吐/OOM-超时,并只从完成final解析网格。
stdlib only;不引入agent脚手架。借用公开源码位置见文件头注释。

借用(行号指冻结快照):
- 安装 pip --no-index --find-links: experiments/014/public-sources/madarshbb-b98k-v1.source.py L1931-1950
- cuda_link_dir: 同上 L1953-1967
- 服务启动/就绪轮询/进程组kill+nvidia-smi排空: 同上 L1970-2020;
  experiments/013/qwen-public/russcore.source.py L1441-1477, L1528-1596
- 请求体 chat_template_kwargs(enable_thinking+reasoning_effort),seed,温度系:
  russcore.source.py L1658-1676; madarshbb-b98k-v1 L994-1008(顶层effort仅gpt-oss用,Qwen不用)
- 响应 reasoning_content/reasoning双字段、finish_reason、usage:
  madarshbb-b98k-v1 L1037-1058; russcore.source.py L743-749, L1701-1717
- 截断只认闭合fence: russcore.source.py L1711-1712 require_closed_fence思想

不照抄:131K context/C16/98K思考、多级server_ladder(MTP/FP8-KV/--reasoning-config)、
旧0.19+TF5.8混装、顶层reasoning_effort字段。
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# ---- V固定配置(014-5),GPU上不得擅自放大 ----
TP_SIZE = 4
MAX_MODEL_LEN = 65536
MAX_NUM_SEQS = 4  # =客户端并发上限
GPU_MEM_UTIL = 0.90
PORT = 1234
SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 20}
SEED_XHIGH = 1001
SEED_LOW = 2001
SEED_LOW_B = 2002
INSTALL_BUDGET_S = 20 * 60
PROTOCOL_BUDGET_S = 5 * 60

EFFORTS = ("xhigh", "low")


def log(msg: str) -> None:
    print(f"[probe {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def one(pattern: str) -> Path:
    hits = glob.glob(pattern, recursive=True)
    if not hits:
        raise FileNotFoundError(f"not mounted: {pattern}")
    if len(hits) > 1:
        log(f"WARN multiple hits for {pattern}: {hits[:3]}")
    return Path(hits[0])


def record_versions(out: dict) -> None:
    try:
        smi = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
             "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
    except FileNotFoundError:
        smi = "nvidia-smi not present (CPU)"
    out["host"] = {"python": sys.version.split()[0], "nvidia_smi": smi}
    try:
        import importlib.metadata as md
        out["host"]["pip_pkgs"] = {p: md.version(p) for p in
                                   ("vllm", "torch", "transformers", "tokenizers") if _has(md, p)}
    except Exception as exc:
        out["host"]["pip_error"] = f"{type(exc).__name__}:{exc}"[:200]


def _has(md, pkg: str) -> bool:
    try:
        md.version(pkg)
        return True
    except Exception:
        return False


def cuda_link_dir(work: Path) -> str:
    """借 madarshbb-b98k-v1 L1953-1967:给FlashInfer JIT补libcuda.so。"""
    d = work / "cuda-link"
    d.mkdir(parents=True, exist_ok=True)
    hits = []
    for pat in ("/usr/local/nvidia/lib64/libcuda.so*", "/usr/lib/x86_64-linux-gnu/libcuda.so*",
                "/usr/lib64/libcuda.so*", "/usr/local/cuda/compat/libcuda.so*"):
        hits += sorted(glob.glob(pat))
    link = d / "libcuda.so"
    if hits and not os.path.lexists(link):
        os.symlink(next((h for h in hits if h.endswith(".so.1")), hits[0]), link)
    log(f"libcuda hits: {hits[:4]}")
    return str(d)


def start_server(model_root: Path, work: Path, extra: list[str],
                 deadline: float | None = None) -> subprocess.Popen:
    """V首配置:TP4/65536/C4/FP8(模型自带quant config,不传--quantization)、KV默认dtype、无MTP。"""
    env = dict(os.environ, HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
               VLLM_NO_USAGE_STATS="1",
               LIBRARY_PATH=":".join([cuda_link_dir(work), "/usr/local/nvidia/lib64",
                                      os.environ.get("LIBRARY_PATH", "")]))
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server",
           "--model", str(model_root), "--served-model-name", "qwen",
           "--host", "127.0.0.1", "--port", str(PORT),
           "--tensor-parallel-size", str(TP_SIZE),
           "--max-model-len", str(MAX_MODEL_LEN),
           "--gpu-memory-utilization", str(GPU_MEM_UTIL),
           "--max-num-seqs", str(MAX_NUM_SEQS),
           "--enable-prefix-caching",
           "--reasoning-parser", "qwen3",
           "--language-model-only",
           *extra]
    log("serve: " + " ".join(cmd))
    slog = work / "vllm-server.log"
    proc = subprocess.Popen(cmd, env=env, stdout=open(slog, "w"),
                            stderr=subprocess.STDOUT, start_new_session=True)
    t = time.time()
    limit = time.monotonic() + 1500 if deadline is None else float(deadline)
    while time.monotonic() < limit:
        if proc.poll() is not None:
            break
        try:
            with socket.create_connection(("127.0.0.1", PORT), timeout=3):
                with urllib.request.urlopen(f"http://127.0.0.1:{PORT}/v1/models",
                                            timeout=5) as r:
                    if r.status < 500:
                        log(f"server ready after {time.time() - t:.0f}s")
                        return proc
        except Exception:
            time.sleep(min(10, max(0.0, limit - time.monotonic())))
    tail = "\n".join(slog.read_text(errors="replace").splitlines()[-30:]) if slog.exists() else ""
    kill_server(proc, work)
    raise RuntimeError(f"server failed to start; tail:\n{tail}")


def kill_server(proc: subprocess.Popen | None, work: Path) -> None:
    """借 madarshbb kill:杀进程组后等nvidia-smi排空,防重启OOM。"""
    if proc is None:
        return
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        proc.kill()
    try:
        proc.wait(timeout=60)
    except Exception:
        pass
    for _ in range(36):
        try:
            used = subprocess.run(["nvidia-smi", "--query-gpu=memory.used",
                                   "--format=csv,noheader,nounits"],
                                  capture_output=True, text=True).stdout.split()
        except Exception:
            return
        if used and all(int(u) < 1500 for u in used if u.strip().isdigit()):
            return
        time.sleep(5)


def chat_once(messages: list[dict], effort: str, seed: int, max_tokens: int,
              timeout_s: float, out: dict, deadline: float | None = None) -> dict:
    """官方模板kwargs传参。请求流式usage,先读usage再读choices。
    timeout_s与deadline取较早者作为本次墙钟;socket超时只限制一次静默。"""
    payload = {"model": "qwen", "messages": messages, "stream": True,
               "max_tokens": max_tokens, "seed": seed,
               "stream_options": {"include_usage": True},
               "chat_template_kwargs": {"enable_thinking": True, "reasoning_effort": effort},
               **SAMPLING}
    out["request"] = {"effort": effort, "seed": seed, "max_tokens": max_tokens,
                      "sampling": dict(SAMPLING),
                      "stream_options": payload["stream_options"],
                      "chat_template_kwargs": payload["chat_template_kwargs"]}
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json",
                                          "Accept": "text/event-stream"})
    limit = time.monotonic() + float(timeout_s)
    if deadline is not None:
        limit = min(limit, float(deadline))
    t0 = time.time()
    first_tok_t = None
    texts, reasons, finish, usage = [], [], None, {}
    try:
        remaining = limit - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("deadline_before_request")
        # Prefill can sit silent; 180s bounds a dead socket without cutting a long prompt.
        stall_s = min(180.0, max(1.0, remaining))
        with urllib.request.urlopen(req, timeout=stall_s) as r:
            while True:
                if time.monotonic() >= limit:
                    if finish is None:
                        out["client_error"] = "deadline_exceeded"
                        out["cancelled_by_client"] = True
                    break
                raw = r.readline()
                if not raw:
                    break
                line = raw.decode(errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except Exception:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                ch = choices[0]
                d = ch.get("delta") or {}
                piece_c = d.get("content") or ""
                piece_r = d.get("reasoning_content") or d.get("reasoning") or ""
                if first_tok_t is None and (piece_c or piece_r):
                    first_tok_t = time.time()
                texts.append(piece_c)
                reasons.append(piece_r)
                if ch.get("finish_reason"):
                    finish = ch["finish_reason"]
    except Exception as exc:
        if "client_error" not in out:
            out["client_error"] = f"{type(exc).__name__}:{exc}"[:300]
            out["cancelled_by_client"] = True
    wall = time.time() - t0
    content, reasoning = "".join(texts), "".join(reasons)
    out.update({"content_chars": len(content), "reasoning_chars": len(reasoning),
                "finish_reason": finish, "usage": usage, "wall_s": round(wall, 1),
                "ttft_s": round(first_tok_t - t0, 2) if first_tok_t else None})
    ntok = usage.get("completion_tokens") if isinstance(usage, dict) else None
    out["agg_tok_s"] = round(ntok / wall, 2) if wall > 0 and ntok else None
    out["usage_present"] = ntok is not None
    return {"content": content, "reasoning": reasoning, "finish": finish}


def chat_wave(specs: list[dict]) -> dict:
    """One concurrent wave, at most MAX_NUM_SEQS in flight. Aggregate tok/s uses the wave wall."""
    n = len(specs)
    workers = min(MAX_NUM_SEQS, n) if n else 1
    slots: list = [None] * n
    t0 = time.monotonic()

    def run_one(i, spec):
        rec: dict = {}
        res = chat_once(spec["messages"], spec["effort"], spec["seed"], spec["max_tokens"],
                        spec.get("timeout_s", 600.0), rec, deadline=spec.get("deadline"))
        return i, rec, res

    if n:
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = [ex.submit(run_one, i, spec) for i, spec in enumerate(specs)]
            for fut in as_completed(futs):
                i, rec, res = fut.result()
                slots[i] = {"rec": rec, "res": res}
    wall = time.monotonic() - t0
    missing = 0
    ntok = 0
    for slot in slots:
        usage = slot["rec"].get("usage") or {}
        got = usage.get("completion_tokens") if isinstance(usage, dict) else None
        if got is None:
            missing += 1
        else:
            ntok += got
    return {"calls": slots, "wave_wall_s": round(wall, 2), "concurrency": min(MAX_NUM_SEQS, n),
            "completion_tokens_sum": ntok, "usage_missing": missing,
            "agg_tok_s": None if wall <= 0 or missing else round(ntok / wall, 2)}


def extract_final_grid(content: str, finish: str | None) -> list[list[int]] | None:
    """只从完成final解析网格;未闭合think尾部矩形不算答案。"""
    if finish != "stop":
        return None
    fences = []
    cur, in_f = [], False
    for ln in content.splitlines():
        s = ln.strip()
        if s.startswith("```"):
            if in_f:
                fences.append(cur)
                cur, in_f = [], False
            else:
                in_f, cur = True, []
        elif in_f:
            cur.append(s)
    if in_f:
        return None  # 有未闭合fence:不算答案
    for block in reversed(fences):
        grid = []
        for ln in block:
            if not ln:
                continue
            cells = ln.replace(",", " ").split()
            row, bad = [], not cells
            for c in cells:
                if not c.isdigit() or int(c) > 9:
                    bad = True
                    break
                row.append(int(c))
            if bad:
                grid = []
                break
            grid.append(row)
        if (grid and 1 <= len(grid) <= 30 and 1 <= len(grid[0]) <= 30
                and all(len(r) == len(grid[0]) for r in grid)):
            return grid
    return None


def classify_generation(rec: dict, content: str, finish: str | None) -> dict:
    """Separate tokens seen, a finished protocol, and a legal ARC grid."""
    usage = rec.get("usage") or {}
    ntok = usage.get("completion_tokens") if isinstance(usage, dict) else None
    generated = bool((ntok or 0) > 0 or rec.get("content_chars") or rec.get("reasoning_chars")
                     or rec.get("ttft_s") is not None)
    protocol_complete = finish == "stop" and not rec.get("client_error")
    grid = extract_final_grid(content, finish) if protocol_complete else None
    return {"generation_observed": generated, "protocol_complete": bool(protocol_complete),
            "valid_answer": grid is not None, "usage_present": ntok is not None, "grid": grid}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", default="/kaggle/working/vprobe")
    ap.add_argument("--prompt-file", default=None,
                    help="ARC prompt文本;不给则只做模板渲染核对+hello探测,不跑题")
    ap.add_argument("--max-tokens", type=int, default=4096,
                    help="短协议验证预算,不是T阶段32768")
    ap.add_argument("--timeout-s", type=float, default=300.0)
    ap.add_argument("--skip-serve", action="store_true",
                    help="CPU核对:只渲染模板两档并比对,不启动服务")
    args = ap.parse_args()
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    run = {"config": {"tp": TP_SIZE, "max_model_len": MAX_MODEL_LEN,
                      "max_num_seqs": MAX_NUM_SEQS, "gpu_mem_util": GPU_MEM_UTIL,
                      "kv_dtype": "default", "mtp": False},
           "budgets": {"install_s": INSTALL_BUDGET_S, "protocol_s": PROTOCOL_BUDGET_S}}
    record_versions(run)

    # 0) 模板两档渲染核对(014-5要求先保存渲染prompt核实确实不同)。
    try:
        from jinja2 import Environment, BaseLoader
        pat = os.environ.get(
            "PROBE_TEMPLATE",
            "/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/chat_template.jinja")
        src = one(pat)
        tpl_src = Path(src).read_text()
    except FileNotFoundError:
        tpl_src = None
        run["template_render"] = {"status": "model_not_mounted_cpu_fallback"}
        log("模型未挂载(CPU/草案阶段),模板核对留待GPU开机首步")
    if tpl_src:
        import json as _j

        def _exc(s):
            raise RuntimeError(s)
        env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
        env.globals["raise_exception"] = _exc
        env.filters["tojson"] = lambda v: _j.dumps(v)
        tpl = env.from_string(tpl_src)
        renders = {}
        for ef in EFFORTS:
            r = tpl.render(messages=[{"role": "user", "content": "hello"}],
                           tools=None, documents=None, add_generation_prompt=True,
                           enable_thinking=True, reasoning_effort=ef)
            renders[ef] = {"chars": len(r),
                           "sha256": hashlib.sha256(r.encode()).hexdigest(), "tail": r[-200:]}
            (work / f"rendered-prompt-{ef}.txt").write_text(r)
        run["template_render"] = {
            "status": "d differ" if renders["xhigh"]["sha256"] != renders["low"]["sha256"] else "IDENTICAL-FAIL",
            "renders": renders}
        log(f"render xhigh={renders['xhigh']['chars']} low={renders['low']['chars']} "
            f"differ={renders['xhigh']['sha256'] != renders['low']['sha256']}")
        if renders["xhigh"]["sha256"] == renders["low"]["sha256"]:
            (work / "run.json").write_text(json.dumps(run, indent=1))
            log("EVIDENCE-FAIL:两档渲染相同,按证据退出,不进P2")
            return 3

    if args.skip_serve:
        (work / "run.json").write_text(json.dumps(run, indent=1))
        log("CPU核对完成(未启动服务)")
        return 0

    t_boot = time.time()
    proc = None
    try:
        model_root = one("/kaggle/input/**/qwen3-8-27b-fp8-repacked-v1/**/config.json").parent
        proc = start_server(model_root, work, extra=[])
        if time.time() - t_boot > INSTALL_BUDGET_S:
            run["budget"] = "install_over_20min_EVIDENCE_FAIL"
            return 4
        run["install_load_s"] = round(time.time() - t_boot, 1)
        prompt = Path(args.prompt_file).read_text() if args.prompt_file else "Return [[1]]."
        t_proto = time.time()
        run["probes"] = []
        for ef, seed in (("xhigh", SEED_XHIGH), ("low", SEED_LOW)):
            if time.time() - t_proto > PROTOCOL_BUDGET_S:
                run["probes"].append({"effort": ef, "status": "protocol_budget_exceeded"})
                break
            rec: dict = {}
            res = chat_once([{"role": "user", "content": prompt}], ef, seed,
                            args.max_tokens, args.timeout_s, rec)
            flags = classify_generation(rec, res["content"], res["finish"])
            rec.update({k: flags[k] for k in (
                "generation_observed", "protocol_complete", "valid_answer", "usage_present")})
            rec["grid"] = flags["grid"]
            rec["answered"] = flags["valid_answer"]
            (work / f"reply-{ef}.json").write_text(json.dumps(
                {"request": rec.get("request"), "finish_reason": rec.get("finish_reason"),
                 "usage": rec.get("usage"), "wall_s": rec.get("wall_s"),
                 "ttft_s": rec.get("ttft_s"), "content": res["content"],
                 "reasoning": res["reasoning"]}, indent=1))
            run["probes"].append(rec)
            log(f"{ef}: finish={rec.get('finish_reason')} wall={rec.get('wall_s')} "
                f"ttft={rec.get('ttft_s')} tok/s={rec.get('agg_tok_s')} "
                f"answered={rec['answered']} err={rec.get('client_error')}")
        run["protocol_s"] = round(time.time() - t_proto, 1)
    except Exception as exc:
        import traceback
        run["fatal"] = f"{type(exc).__name__}:{exc}"[:500]
        run["traceback_tail"] = traceback.format_exc()[-1500:]
        log(f"EVIDENCE-FAIL: {run['fatal']}")
        return 2
    finally:
        kill_server(proc, work)
        (work / "run.json").write_text(json.dumps(run, indent=1))
    # 成功标准:真实生成+reasoning/final区分可见,不是"端口已打开"。
    ok = [p for p in run.get("probes", [])
          if p.get("generation_observed") and p.get("usage_present") and not p.get("client_error")]
    if not ok:
        log("EVIDENCE-FAIL:无带usage的真实生成,不进P2")
        return 5
    log("短协议通过;是否进P2由T准入(实测R)决定")
    return 0


if __name__ == "__main__":
    sys.exit(main())
