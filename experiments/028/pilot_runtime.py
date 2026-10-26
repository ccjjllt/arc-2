"""One manual same-adapter NVARC DFS mechanism comparison; no gold access."""
import gc
import hashlib
import inspect
import json
import os
import platform
import time
from pathlib import Path

import torch

from dfs_path_reuse import inference_path_dfs


def digest(value):
    raw = json.dumps(value, sort_keys=True, separators=(',', ':')).encode()
    return hashlib.sha256(raw).hexdigest()


class CountedModel:
    def __init__(self, model):
        self.model = model
        self.calls = 0
        self.tokens = 0
        self.shapes = {}
        self.prefill_returned_at = None

    def __getattr__(self, name):
        return getattr(self.model, name)

    def __call__(self, *args, **kwargs):
        tokens = kwargs.get('input_ids', args[0] if args else None)
        self.calls += 1
        self.tokens += tokens.numel()
        shape = str(list(tokens.shape))
        self.shapes[shape] = self.shapes.get(shape, 0) + 1
        output = self.model(*args, **kwargs)
        if self.calls == 1:
            self.prefill_returned_at = time.time()
        return output


def run_task_pilot(model, tokenizer, formatter, puzzle_ds_multi, eval_ds, batches,
                   max_new_tokens, max_score, end_time, rank, task_id,
                   original_inference):
    root = Path('/kaggle/working/028/pilot')
    root.mkdir(exist_ok=True)
    path = root / (task_id + '.json')
    query_key = sorted({s.split('.')[0] for b in batches for s in b})[0]
    selected = [b for b in batches if all(s.split('.')[0] == query_key for s in b)][:2]
    if len(selected) != 2:
        raise ValueError('Need two original batches for first query')
    report = dict(task_id=task_id, query_key=query_key, rank=rank,
                  status='running', scope='Mechanism pilot on first query and first two original batches only; no accuracy score.',
                  all_task_queries=len(puzzle_ds_multi.keys), measured_queries=1,
                  model_object_id=id(model), ttt_adapter_reused=True,
                  probability_cutoff=0.2, max_score=float(max_score),
                  max_new_tokens=max_new_tokens, hash_seed=os.environ.get('PYTHONHASHSEED'),
                  order=['seed_original', 'original_A', 'path_A', 'path_B', 'original_B'],
                  precompute_tokens=16, records=[], python=platform.python_version(),
                  torch_version=torch.__version__, cuda=torch.version.cuda,
                  gpu=torch.cuda.get_device_name(), gold_read=False,
                  planned_mechanism='One full prefill of prompt plus at most16 known draft tokens; original DFS callback and original traversal unchanged.')

    def save():
        temporary = path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, path)

    def serialise(result, subkeys):
        rows = []
        for row, beams in result:
            for score, tokens in beams:
                grid = formatter.convert_tokens_to_array(tokens)
                solution = None if grid is None else puzzle_ds_multi.invert_mod(grid, subkeys[row], inv_perm=True).tolist()
                rows.append(dict(row=int(row), subkey=subkeys[row], nll=float(score),
                                 tokens=[int(v) for v in tokens], solution=solution))
        return rows

    def execute(label, subkeys, drafts=None):
        if time.time()+20 >= end_time:
            raise TimeoutError('Insufficient shared pilot budget for next run')
        prompts = [tokenizer.encode(eval_ds.get(s, formatter)['input']) for s in subkeys]
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        proxy = CountedModel(model)
        started_at = time.time()
        call_deadline = min(end_time, started_at+560)
        started = time.perf_counter()
        try:
            if label.startswith('path'):
                result, extra = inference_path_dfs(proxy, prompts, max_new_tokens, max_score,
                    call_deadline, draft_tokens=drafts,
                    original_inference=original_inference, precompute_tokens=16)
            else:
                result = original_inference(proxy, prompts, max_new_tokens, max_score,
                                            call_deadline)
                extra = dict(policy='original', fallback=False)
        except Exception as error:
            report.update(status='error',failed_label=label,error_type=type(error).__name__,
                          error=str(error)[:500],mechanism_error=getattr(error,'arc028_metrics',None))
            save()
            raise
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-started
        finished_at = time.time()
        rows = serialise(result, subkeys)
        record = dict(label=label, elapsed_s=elapsed, started_at=started_at,finished_at=finished_at,
            call_deadline=call_deadline,prefill_returned_at=proxy.prefill_returned_at,
            cutoff_possible=elapsed>=540 or finished_at>=call_deadline-0.1,
            model_forward_calls=proxy.calls, model_input_tokens=proxy.tokens,
            input_shapes=proxy.shapes, peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            prompt_tokens_sha256=digest(prompts), prompt_lengths=[len(p) for p in prompts],
            subkeys=subkeys, draft_tokens_sha256=digest(drafts) if drafts else None,
            mechanism=extra, candidates=rows,
            returned_sequences=len(rows), legal_candidates=sum(r['solution'] is not None for r in rows))
        report['records'].append(record)
        save()
        print(json.dumps(dict(event='pilot_call_complete', task_id=task_id,
            label=label, elapsed_s=elapsed, forward_calls=proxy.calls,
            returned_sequences=len(rows), mechanism=extra), allow_nan=False), flush=True)
        return rows

    save()
    # Save actual loaded forward source identity and paged state names for audit.
    attention = next(m for m in model.modules() if 'Attention' in type(m).__name__)
    report['attention_class'] = type(attention).__module__+'.'+type(attention).__name__
    report['attention_paged_state_names_before'] = [name for name in vars(attention) if 'paged' in name]
    try:
        source = inspect.getsource(type(attention).forward)
        report['attention_forward_source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
        # Third-party runtime source is an output artifact, kept outside repository on download.
        (root/(task_id+'-attention-forward.py')).write_text(source)
    except (OSError, TypeError):
        report['attention_forward_source_sha256'] = None
    seed = execute('seed_original', selected[0])
    first_legal = next((row for row in seed if row['solution'] is not None), None)
    drafts = None
    if first_legal is not None:
        drafts = [tokenizer.encode(formatter.fmt_reply([
            puzzle_ds_multi.forward_mod(first_legal['solution'], subkey).tolist()]))
            for subkey in selected[1]]
        report['draft_source'] = dict(seed_subkey=first_legal['subkey'],
            seed_tokens_sha256=digest(first_legal['tokens']),
            original_grid_sha256=digest(first_legal['solution']),
            target_subkeys=selected[1], draft_lengths=[len(x) for x in drafts],
            rule='First legal result in original returned order from earlier seed batch, no gold or score-based selection.')
    else:
        report['draft_source'] = None
    save()
    for label in report['order'][1:]:
        execute(label, selected[1], drafts)
    records = {r['label']: r for r in report['records']}
    def mapping(label):
        return {(r['row'], tuple(r['tokens'])):r['nll'] for r in records[label]['candidates']}
    reference = mapping('original_A')
    comparisons = {}
    for label in ('original_B','path_A','path_B'):
        other = mapping(label)
        common = reference.keys() & other.keys()
        comparisons[label] = dict(reference_sequences=len(reference), sequences=len(other),
            same_sequence_set=reference.keys()==other.keys(),
            same_returned_sequence_order=[(r['row'],r['tokens']) for r in records['original_A']['candidates']]==[(r['row'],r['tokens']) for r in records[label]['candidates']],
            extra_sequences=len(other.keys()-reference.keys()),
            missing_sequences=len(reference.keys()-other.keys()),
            max_common_nll_difference=max((abs(reference[k]-other[k]) for k in common), default=None),
            speed_ratio_vs_original_A=records['original_A']['elapsed_s']/records[label]['elapsed_s'])
    report.update(status='complete', comparisons=comparisons,
                  no_promotion_decision=True,
                  note='Timing includes precompute, KV cloning and synchronization at call boundaries. First call is cold; original_A/path_A/path_B/original_B are fixed ABBA. Partial-deadline outputs cannot establish complete tree equivalence.')
    save()
    print(json.dumps(dict(event='task_pilot_complete',task_id=task_id,
                         comparisons=comparisons),allow_nan=False),flush=True)
