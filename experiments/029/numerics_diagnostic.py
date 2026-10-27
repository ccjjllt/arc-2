"""One manual, same-adapter diagnosis of the frozen028 numerical difference.

No gold, scoring, parameter search, training, or original DFS replacement.
Large KV tensors stay in GPU memory and are compared in sequence chunks.
"""
import gc
import hashlib
import importlib
import inspect
import json
import marshal
import os
import time
from pathlib import Path

import torch

from dfs_path_reuse import ARC_TOKENS, EOS_ID, _PAGED_ATTRIBUTES


class DiagnosticUnsupported(RuntimeError):
    pass


def _sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _validate_cache(cache, batch, length):
    if not isinstance(cache, (tuple, list)) or not cache:
        raise DiagnosticUnsupported('Expected nonempty legacy per-layer KV cache')
    for layer in cache:
        if not isinstance(layer, (tuple, list)) or len(layer) != 2:
            raise DiagnosticUnsupported('Expected exactly K,V per cache layer')
        if any(not isinstance(t, torch.Tensor) or t.ndim != 4 or
               t.shape[0] != batch or t.shape[-2] != length for t in layer):
            raise DiagnosticUnsupported('Unexpected KV tensor shape or sequence length')


def _clone_cache(cache, batch, length):
    _validate_cache(cache, batch, length)
    return tuple(tuple(t.detach().clone() for t in layer) for layer in cache)


def _small_logits(logits):
    return logits.detach().float().cpu().clone()


def _logits_record(logits, tokens=None):
    if logits.ndim == 2:
        logits = logits[:, None, :]
    probabilities = logits.log_softmax(-1)
    order = probabilities[..., list(ARC_TOKENS)].argsort(dim=-1, descending=True)
    record = dict(logits=logits.tolist(), arc_token_order=[
        [[int(ARC_TOKENS[index]) for index in position] for position in row]
        for row in order.tolist()])
    if tokens is not None:
        nll = -probabilities[:, :tokens.shape[1]].gather(-1, tokens[..., None]).squeeze(-1)
        record.update(token_nll=nll.tolist(), cumulative_nll=nll.cumsum(-1).tolist())
    return record


def _logits_diff(left, right):
    if left.shape != right.shape:
        raise DiagnosticUnsupported('Compared logit shapes differ')
    delta = (left-right).abs()
    log_delta = (left.log_softmax(-1)-right.log_softmax(-1)).abs()
    return dict(shape=list(left.shape), equal=torch.equal(left, right),
                max_abs=float(delta.max()), mean_abs=float(delta.mean()),
                max_log_probability_abs=float(log_delta.max()),
                per_row_position_max_abs=delta.max(dim=-1).values.tolist())


def run_task_diagnostic(model, tokenizer, formatter, puzzle_ds_multi, eval_ds,
                        batches, max_new_tokens, max_score, end_time, rank,
                        task_id, original_inference):
    root = Path('/kaggle/working/029/diagnostic')
    root.mkdir(parents=True, exist_ok=True)
    path = root/(task_id+'.json')
    report = dict(task_id=task_id, rank=rank, status='running', stage='initializing',
        started_at=time.time(), model_object_id=id(model), same_adapter=True,
        torch_version=torch.__version__, cuda=torch.version.cuda,
        gpu=torch.cuda.get_device_name(), model_training=bool(model.training),
        inference_mode_enabled=torch.is_inference_mode_enabled(),
        grad_enabled=torch.is_grad_enabled(), gold_read=False, B4_run=False,
        accuracy_measured=False, restore_depths=[0, 1, 13], draft_limit=16,
        hypothesis='Separate prefill shape, sequential decode, and snapshot restore differences; no causal conclusion predetermined.',
        calls=[], stages={})

    def save():
        temporary = path.with_suffix('.json.partial')
        temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
        os.replace(temporary, path)

    def budget():
        if time.time()+10 >= end_time:
            raise TimeoutError('Diagnostic shared deadline is near')

    attention = tuple(m for m in model.modules() if all(hasattr(m, name)
        for name in ('q_proj', 'k_proj', 'v_proj', 'o_proj')))

    def clear_paged():
        count = 0
        for module in attention:
            present = False
            for name in _PAGED_ATTRIBUTES:
                if hasattr(module, name):
                    delattr(module, name)
                    present = True
            count += int(present)
        return count

    def forward(label, ids, cache=None, position=None):
        budget()
        arguments = dict(input_ids=ids, return_dict=True, use_cache=True)
        if cache is not None:
            arguments.update(past_key_values=cache,
                position_ids=torch.full((ids.shape[0], 1), position, device=model.device))
        torch.cuda.synchronize()
        started = time.time()
        output = model(**arguments)
        torch.cuda.synchronize()
        report['calls'].append(dict(label=label, started_at=started,
            elapsed_s=time.time()-started, input_shape=list(ids.shape),
            cached=cache is not None, position=position,
            output_cache_type=type(output.past_key_values).__name__,
            output_logits_shape=list(output.logits.shape)))
        return output

    def compare_cache(label, left, right, length):
        _validate_cache(left, batch_size, length)
        _validate_cache(right, batch_size, right[0][0].shape[-2])
        if right[0][0].shape[-2] < length or len(left) != len(right):
            raise DiagnosticUnsupported('Compared cache layer count/length differs')
        layers = []
        for index, (a_layer, b_layer) in enumerate(zip(left, right)):
            budget()
            layer = dict(layer=index)
            for kind, a, b in zip(('K', 'V'), a_layer, b_layer):
                maximum = total = 0.0
                changed = count = 0
                for start in range(0, length, 64):
                    delta = (a[:, :, start:min(start+64, length)].float()-
                             b[:, :, start:min(start+64, length)].float()).abs()
                    if not bool(torch.isfinite(delta).all()):
                        raise DiagnosticUnsupported('Nonfinite KV difference')
                    maximum = max(maximum, float(delta.max()))
                    total += float(delta.sum())
                    changed += int(torch.count_nonzero(delta))
                    count += delta.numel()
                layer[kind] = dict(max_abs=maximum, mean_abs=total/count,
                                   nonzero_elements=changed, elements=count)
            layers.append(layer)
        return dict(label=label, prefix_length=length, layers=layers,
            all_equal=all(layer[kind]['nonzero_elements']==0 for layer in layers for kind in ('K', 'V')),
            max_abs=max(layer[kind]['max_abs'] for layer in layers for kind in ('K', 'V')))

    def announce(stage):
        save()
        print(json.dumps(dict(event='diagnostic_stage_complete',task_id=task_id,stage=stage)), flush=True)

    def copied_paged_prefix(snapshot, cache_length):
        if len(attention) != len(snapshot):
            return dict(status='unconfirmed', reason='Attention module/cache layer count differs')
        pages, shapes = [], []
        for module, layer in zip(attention, snapshot):
            pair = []
            for name, reference in zip(('paged_attention_K', 'paged_attention_V'), layer):
                value = getattr(module, name, None)
                if not isinstance(value, torch.Tensor) or value.ndim != 4:
                    return dict(status='unconfirmed', reason='No recognized four-dimensional '+name)
                shapes.append(list(value.shape))
                if tuple(value.shape[:2]) == tuple(reference.shape[:2]):
                    pair.append(value)
                elif tuple(value.shape[1:3]) == tuple(reference.shape[:2]):
                    pair.append(value.permute(1, 2, 0, 3))
                else:
                    return dict(status='unconfirmed', reason='Paged buffer layout not recognized', shapes=shapes)
            pages.append(tuple(pair))
        return dict(status='observed', paged_shapes=shapes,
            snapshot_effective_length=cache_length,
            comparison=compare_cache('snapshot_vs_internal_paged_prefix', snapshot, tuple(pages), cache_length),
            length_note='Paged shapes are allocation capacity. Effective prefix length comes from passed snapshot and returned legacy cache, not an independent paged active-length counter.')

    save()
    try:
        if not report['inference_mode_enabled'] or report['grad_enabled'] or not attention:
            raise DiagnosticUnsupported('Requires original inference_mode and Qwen-style attention')
        function = getattr(importlib.import_module('unsloth.models.qwen3'),
                           'Qwen3Attention_fast_forward_inference')
        function = inspect.unwrap(function)
        identity = dict(module=function.__module__, name=function.__name__)
        identity['code_sha256'] = hashlib.sha256(marshal.dumps(function.__code__)).hexdigest()
        try:
            identity['source_sha256'] = hashlib.sha256(inspect.getsource(function).encode()).hexdigest()
        except (OSError, TypeError):
            identity['source_sha256'] = None
        report['actual_fastdecode_function'] = identity
        query_key = sorted({s.split('.')[0] for b in batches for s in b})[0]
        selected = [b for b in batches if all(s.split('.')[0]==query_key for s in b)][:2]
        if len(selected) != 2:
            raise DiagnosticUnsupported('Need first query and its first two original batches')
        report.update(query_key=query_key, seed_subkeys=selected[0], target_subkeys=selected[1],
                      attention_modules=len(attention), stage='seed_original')
        seed_prefix = [tokenizer.encode(eval_ds.get(s, formatter)['input']) for s in selected[0]]
        budget()
        started = time.time()
        seed = original_inference(model, seed_prefix, max_new_tokens, max_score, min(end_time, started+560))
        report['seed_elapsed_s'] = time.time()-started
        first = None
        for row, beams in seed:
            for score, tokens in beams:
                array = formatter.convert_tokens_to_array(tokens)
                if array is not None:
                    first = dict(row=int(row), subkey=selected[0][row], nll=float(score),
                        tokens=list(map(int, tokens)), grid=puzzle_ds_multi.invert_mod(array, selected[0][row], inv_perm=True).tolist())
                    break
            if first is not None:
                break
        if first is None:
            raise DiagnosticUnsupported('Frozen seed rule produced no legal draft; no substitute selected')
        prefixes = [tokenizer.encode(eval_ds.get(s, formatter)['input']) for s in selected[1]]
        drafts = [tokenizer.encode(formatter.fmt_reply([
            puzzle_ds_multi.forward_mod(first['grid'], s).tolist()])) for s in selected[1]]
        count = min([16, max_new_tokens-1]+[d.index(EOS_ID) if EOS_ID in d else len(d) for d in drafts])
        if count != 16 or any(t not in ARC_TOKENS for d in drafts for t in d):
            raise DiagnosticUnsupported('This fixed16-token diagnosis requires16 pre-EOS ARC tokens')
        prefix = torch.tensor(prefixes, device=model.device, dtype=torch.long)
        draft = torch.tensor([d[:16] for d in drafts], device=model.device, dtype=torch.long)
        draft_cpu = draft.cpu()
        batch_size, length = prefix.shape
        report.update(stage='prefill_shape', prefix_tokens=prefixes,
            draft_tokens=draft_cpu.tolist(), prefix_tokens_sha256=_sha(prefixes),
            draft_tokens_sha256=_sha(draft_cpu.tolist()), seed_first_legal=first,
            prefix_length=length, batch_size=batch_size)
        announce('seed_and_inputs_frozen')

        # A: repeated prompt-only control versus a longer causal prefill.
        clear_paged()
        output = forward('prompt_A', prefix)
        baseline_logits = _small_logits(output.logits[:, -1])
        baseline_cache = _clone_cache(output.past_key_values, batch_size, length)
        del output
        clear_paged()
        output = forward('prompt_B', prefix)
        repeat_logits = _small_logits(output.logits[:, -1])
        repeat_cache = _clone_cache(output.past_key_values, batch_size, length)
        report['stages']['prefill_shape'] = dict(prompt_A=_logits_record(baseline_logits),
            prompt_B=_logits_record(repeat_logits), repeat_logits=_logits_diff(baseline_logits, repeat_logits),
            repeat_KV=compare_cache('prompt_A_vs_prompt_B', baseline_cache, repeat_cache, length))
        del repeat_cache, output
        save()
        clear_paged()
        output = forward('prompt_plus16', torch.cat((prefix, draft), dim=1))
        if output.logits.shape[1] != length+16:
            raise DiagnosticUnsupported('Full prefill did not return every position logit')
        full_logits = _small_logits(output.logits[:, length-1:length+16])
        full_cache = _clone_cache(output.past_key_values, batch_size, length+16)
        full_prefix = tuple((k[:, :, :length], v[:, :, :length]) for k, v in full_cache)
        report['stages']['prefill_shape'].update(full_root=_logits_record(full_logits[:, 0]),
            extended_root_logits=_logits_diff(baseline_logits, full_logits[:, 0]),
            extended_prefix_KV=compare_cache('prompt_A_vs_extended_prefix', baseline_cache, full_prefix, length))
        del baseline_cache, output, full_prefix
        announce('prefill_shape')

        # B: two original sequential paths versus the same fixed causal path.
        report['stage'] = 'sequential_vs_prefill'
        def sequential(label):
            clear_paged()
            output = forward(label+'_root', prefix)
            cache = output.past_key_values
            values = [_small_logits(output.logits[:, -1])]
            for depth in range(16):
                output = forward(label+'_'+str(depth), draft[:, depth:depth+1], cache, length+depth)
                cache = output.past_key_values
                values.append(_small_logits(output.logits[:, -1]))
            return torch.stack(values, dim=1), _clone_cache(cache, batch_size, length+16)
        seq_a, cache_a = sequential('sequential_A')
        report['stages']['sequential_vs_prefill'] = dict(sequential_A=_logits_record(seq_a, draft_cpu),
            full_prefill=_logits_record(full_logits, draft_cpu),
            precomputed_logits_difference=_logits_diff(seq_a, full_logits))
        save()
        seq_b, cache_b = sequential('sequential_B')
        report['stages']['sequential_vs_prefill'].update(sequential_B=_logits_record(seq_b, draft_cpu),
            repeat_logits=_logits_diff(seq_a, seq_b),
            repeat_KV=compare_cache('sequential_A_vs_B', cache_a, cache_b, length+16),
            full_prefill_KV=compare_cache('sequential_A_vs_fullprefill', cache_a, full_cache, length+16))
        del cache_a, cache_b, full_cache
        gc.collect()
        announce('sequential_vs_prefill')

        # C: original-path snapshots, with skip fully absent from the calls.
        report['stage'] = 'snapshot_restore'
        report['stages']['snapshot_restore'] = []
        for depth in (0, 1, 13):
            clear_paged()
            output = forward('restore_build_'+str(depth), prefix)
            cache = output.past_key_values
            for step in range(depth):
                output = forward('restore_build_'+str(depth)+'_'+str(step),
                    draft[:, step:step+1], cache, length+step)
                cache = output.past_key_values
            snapshot = _clone_cache(cache, batch_size, length+depth)
            point = dict(depth=depth, position=length+depth, skipped_forwards=0,
                depth_scope='Predeclared mechanism depths, not the unrecorded three actual028 restore depths',
                snapshot_source='Original sequential target calls; never full-prefill cache', results={})
            report['stages']['snapshot_restore'].append(point)
            saved = {}
            for label in ('inplace_A', 'inplace_B', 'restored_A', 'restored_B'):
                if label.startswith('restored'):
                    point['cleared_modules_'+label] = clear_paged()
                output = forward(str(depth)+'_'+label, draft[:, depth:depth+1],
                                 snapshot if label.startswith('restored') else cache, length+depth)
                value = _small_logits(output.logits[:, -1])
                cloned = _clone_cache(output.past_key_values, batch_size, length+depth+1)
                saved[label] = (value, cloned)
                point['results'][label] = _logits_record(value)
                if label.startswith('restored'):
                    point['paged_prefix_'+label] = copied_paged_prefix(snapshot, length+depth)
                    point['paged_prefix_'+label]['returned_cache_effective_length'] = output.past_key_values[0][0].shape[-2]
                save()
            for label in ('inplace_B', 'restored_A', 'restored_B'):
                point[label+'_vs_inplace_A'] = dict(logits=_logits_diff(saved['inplace_A'][0], saved[label][0]),
                    KV=compare_cache(str(depth)+'_'+label, saved['inplace_A'][1], saved[label][1], length+depth+1))
            point['restored_repeat'] = _logits_diff(saved['restored_A'][0], saved['restored_B'][0])
            del snapshot, saved, output, cloned, cache
            gc.collect()
            announce('snapshot_restore_'+str(depth))
        clear_paged()
        report.update(status='complete', stage='complete', finished_at=time.time(),
            elapsed_s=time.time()-report['started_at'], peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            note='Per-position and KV diagnostic only. No automatic causal verdict, throughput claim, accuracy or promotion.')
        announce('complete')
    except Exception as error:
        report.update(status='error', error_type=type(error).__name__, error=str(error)[:1000],
                      finished_at=time.time())
        save()
        raise
