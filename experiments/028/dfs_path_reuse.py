"""Reuse a short, already generated path while calling the unchanged NVARC DFS.

This module contains only the proxy and cache handling added for experiment028.
The caller supplies the original ``inference_turbo_dfs`` from the frozen source;
candidate order, strict probability cutoff, EOS and time limits stay there.

Mechanism reference: Daniel Franzen and Jan Disselhoff's Apache-2.0 2024
``training_code/inference_tools.py`` (commit d3ac3f6ebf6fb609bfdc782561ee99977ca35d95).
This is a separate implementation for the patched Unsloth2025.9.7/Qwen3 path,
not a copy of their non-batched DFS. In that runtime cached forward accepts
one token only, and each layer keeps mutable ``paged_attention`` buffers.

Only the initial <=16 draft tokens are precomputed, together with the prompt,
without passing a cache. All batch rows, including inactive PAD rows, must
match before a forward can be skipped. The first mismatch permanently ends
reuse for that invocation. Returning to an immutable snapshot always clears
the mutable paged buffers before a real forward copies that prefix into them.
Numerical agreement and throughput still require the manual GPU comparison.
"""

from types import SimpleNamespace

import torch


ARC_TOKENS = tuple(range(11)) + (15,)
PAD_ID = 13
EOS_ID = 15
_PAGED_ATTRIBUTES = (
    "paged_attention_K", "paged_attention_V", "paged_attention",
    "temp_QA", "temp_KV", "RH_Q", "attention", "temp_O",
    "scalar", "half_head_dim",
)


class PathReuseUnsupported(RuntimeError):
    """A checked runtime assumption was not satisfied; no equivalence claim."""


class _PathProxy:
    def __init__(self, model, draft_tokens, max_new_tokens, metrics):
        self.target = model
        self.metrics = metrics
        self.first_call = True
        self.enabled = False
        self.snapshot = None
        self.path_logits = None
        self.draft = None
        self.prefix_length = None
        self.draft_length = 0
        self.draft_tokens = draft_tokens
        self.max_new_tokens = max_new_tokens
        self.attention_modules = tuple(
            module for module in model.modules()
            if all(hasattr(module, name) for name in ("q_proj", "k_proj", "v_proj", "o_proj"))
        )
        metrics["attention_module_count"] = len(self.attention_modules)

    def __getattr__(self, name):
        return getattr(self.target, name)

    def _clear_paged(self):
        cleared = 0
        for attention in self.attention_modules:
            present = False
            for name in _PAGED_ATTRIBUTES:
                if hasattr(attention, name):
                    delattr(attention, name)
                    present = True
            cleared += int(present)
        self.metrics["paged_module_clear_count"] += cleared

    def _real_forward(self, *args, **kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.metrics["model_forward_calls"] += 1
        if isinstance(ids, torch.Tensor):
            self.metrics["model_forward_input_tokens"] += ids.numel()
            self.metrics["max_model_forward_sequence"] = max(
                self.metrics["max_model_forward_sequence"], int(ids.shape[-1]))
        return self.target(*args, **kwargs)

    def _prepare_draft(self, batch_size):
        if self.draft_tokens is None:
            self.metrics["fallback_reason"] = "no_draft"
            return None
        try:
            rows = [list(map(int, row)) for row in self.draft_tokens]
        except (TypeError, ValueError):
            self.metrics["fallback_reason"] = "draft_is_not_token_rows"
            return None
        if len(rows) != batch_size:
            self.metrics["fallback_reason"] = "draft_batch_size_mismatch"
            return None
        if any(token not in ARC_TOKENS for row in rows for token in row):
            self.metrics["fallback_reason"] = "invalid_draft_token"
            return None
        lengths = [row.index(EOS_ID) if EOS_ID in row else len(row) for row in rows]
        length = min([16, self.max_new_tokens - 1] + lengths)
        if length <= 0:
            self.metrics["fallback_reason"] = "empty_draft_or_early_eos"
            return None
        self.draft_length = length
        return [row[:length] for row in rows]

    def _snapshot_slice(self, depth):
        length = self.prefix_length + depth
        return tuple((key[:, :, :length, :], value[:, :, :length, :])
                     for key, value in self.snapshot)

    def _snapshot_depth(self, cache):
        """Recognize prefix views by storage, offset, layout and sequence length."""
        if self.snapshot is None or not isinstance(cache, (tuple, list)):
            return None
        if len(cache) != len(self.snapshot):
            return None
        depth = None
        for layer, immutable in zip(cache, self.snapshot):
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                return None
            for tensor, base in zip(layer, immutable):
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    return None
                current = tensor.shape[-2] - self.prefix_length
                if not 0 <= current <= self.draft_length:
                    return None
                if depth is None:
                    depth = current
                if current != depth or tensor.shape[:2] != base.shape[:2] or tensor.shape[-1] != base.shape[-1]:
                    return None
                if tensor.stride() != base.stride() or tensor.storage_offset() != base.storage_offset():
                    return None
                if tensor.untyped_storage().data_ptr() != base.untyped_storage().data_ptr():
                    return None
        return depth

    def _prefill(self, input_ids, kwargs):
        rows = self._prepare_draft(input_ids.shape[0])
        if rows is None:
            return self._real_forward(**kwargs)
        if not self.attention_modules:
            raise PathReuseUnsupported("No Qwen-style attention modules found for paged-cache reset")
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise PathReuseUnsupported("Expected a nonempty two-dimensional prompt batch")
        if kwargs.get("position_ids") is not None or kwargs.get("attention_mask") is not None:
            raise PathReuseUnsupported("Pilot prefill requires the original unpadded, implicit-position batch")
        self.prefix_length = input_ids.shape[1]
        self.draft = torch.tensor(rows, device=input_ids.device, dtype=input_ids.dtype)
        combined = torch.cat((input_ids, self.draft), dim=1)
        combined_kwargs = dict(kwargs, input_ids=combined)
        output = self._real_forward(**combined_kwargs)
        cache = output.past_key_values
        if not isinstance(cache, (tuple, list)) or not cache:
            raise PathReuseUnsupported("Expected Unsloth legacy per-layer (K,V) cache")
        full_length = combined.shape[1]
        for layer in cache:
            if not isinstance(layer, (tuple, list)) or len(layer) != 2:
                raise PathReuseUnsupported("Expected exactly two tensors in each attention cache layer")
            for tensor in layer:
                if not isinstance(tensor, torch.Tensor) or tensor.ndim != 4:
                    raise PathReuseUnsupported("Expected four-dimensional legacy KV tensors")
                if tensor.shape[0] != input_ids.shape[0] or tensor.shape[-2] != full_length:
                    raise PathReuseUnsupported("KV batch or sequence length does not match the prefill")
        if output.logits.ndim != 3 or output.logits.shape[1] != full_length:
            raise PathReuseUnsupported("Prefill did not return logits for all prompt and draft positions")
        self.metrics["prefill_cache_type"] = type(cache).__module__ + "." + type(cache).__qualname__
        self.metrics["first_layer_key_shape"] = list(cache[0][0].shape)
        self.snapshot = tuple((key.detach().clone(), value.detach().clone()) for key, value in cache)
        self.path_logits = output.logits[:, self.prefix_length - 1:full_length].detach().clone()
        self.metrics["immutable_snapshot_bytes"] = sum(t.numel() * t.element_size()
                                                      for layer in self.snapshot for t in layer)
        self.metrics["precomputed_draft_tokens_per_row"] = self.draft_length
        self.metrics["precomputed_draft_tokens_total"] = self.draft.numel()
        self.enabled = True
        self.metrics["reuse_was_enabled"] = True
        # This output corresponds to the prompt alone. Future causal positions
        # are retained only in the immutable snapshot and path-logit table.
        return SimpleNamespace(logits=self.path_logits[:, :1], past_key_values=self._snapshot_slice(0))

    def __call__(self, *args, **kwargs):
        if args:
            raise PathReuseUnsupported("The frozen original DFS must call the proxy with keyword arguments")
        ids = kwargs.get("input_ids")
        cache = kwargs.get("past_key_values")
        if self.first_call:
            self.first_call = False
            if cache is not None or not isinstance(ids, torch.Tensor):
                raise PathReuseUnsupported("First original DFS call must be an uncached tensor prefill")
            return self._prefill(ids, kwargs)

        depth = self._snapshot_depth(cache)
        if self.enabled:
            reason = None
            if depth is None:
                reason = "cache_no_longer_on_immutable_path"
            elif depth >= self.draft_length:
                reason = "draft_exhausted"
            elif not isinstance(ids, torch.Tensor) or tuple(ids.shape) != (self.draft.shape[0], 1):
                reason = "non_single_token_decode"
            else:
                positions = kwargs.get("position_ids")
                if positions is None or not bool(torch.all(positions == self.prefix_length + depth)):
                    reason = "decode_position_mismatch"
                elif not torch.equal(ids[:, 0], self.draft[:, depth]):
                    reason = "batch_token_mismatch"
            if reason is None:
                self.metrics["skipped_single_token_forwards"] += 1
                self.metrics["skipped_input_tokens"] += ids.numel()
                return SimpleNamespace(logits=self.path_logits[:, depth+1:depth+2],
                                       past_key_values=self._snapshot_slice(depth+1))
            self.enabled = False
            self.metrics["reuse_stop_reason"] = reason

        if depth is not None:
            # A snapshot can reappear when the unchanged recursive DFS returns
            # to an ancestor. Unsloth otherwise uses stale mutable paged data
            # and reads only the passed cache's length, not its tensor contents.
            self._clear_paged()
            self.metrics["immutable_snapshot_restores"] += 1
        return self._real_forward(**kwargs)


@torch.no_grad()
def inference_path_dfs(model, prefix_tokens, max_new_tokens, max_score, end_time,
                       draft_tokens=None, *, original_inference, precompute_tokens=16):
    """Call frozen DFS through a conservative path-reuse proxy; return metrics.

    ``original_inference`` must be the original five-argument NVARC inference
    function. The proxy does not implement its traversal or modify its 540s
    search limit. Timing synchronization belongs in the manual caller. On an
    unsupported cache/runtime, ``PathReuseUnsupported`` is raised with an
    ``arc028_metrics`` attribute; other original/model exceptions propagate.
    """
    metrics = dict(status="running", model_forward_calls=0, model_forward_input_tokens=0,
                   max_model_forward_sequence=0, reuse_was_enabled=False,
                   precomputed_draft_tokens_per_row=0, precomputed_draft_tokens_total=0,
                   skipped_single_token_forwards=0, skipped_input_tokens=0,
                   immutable_snapshot_bytes=0, immutable_snapshot_restores=0,
                   paged_module_clear_count=0, fallback_reason=None, reuse_stop_reason=None,
                   precompute_limit=16, source_runtime="patched Unsloth2025.9.7/Qwen3 legacy KV",
                   traversal="unchanged original_inference callback",
                   numerical_equivalence="unconfirmed until paired GPU measurement")
    try:
        if precompute_tokens != 16:
            raise PathReuseUnsupported("This frozen pilot supports only precompute_tokens=16")
        if not callable(original_inference):
            raise PathReuseUnsupported("original_inference must be the frozen NVARC callable")
        proxy = _PathProxy(model, draft_tokens, int(max_new_tokens), metrics)
        result = original_inference(proxy, prefix_tokens, max_new_tokens, max_score, end_time)
        metrics["status"] = "completed"
        return result, metrics
    except Exception as error:
        metrics["status"] = "unsupported" if isinstance(error, PathReuseUnsupported) else "error"
        metrics["error_type"] = type(error).__name__
        error.arc028_metrics = metrics
        raise
