# -*- coding: utf-8 -*-
"""G3: copy 16 Qwen3.8 token rows into an untied grid vocabulary.

This does not train, does not replace the backbone, does not swap GDN blocks
for attention, does not tie the input embedding to lm_head, and does not apply
an old LoRA. ``--self-check`` uses a fake tokenizer and in-memory matrices.
``write_converted_model`` is the Kaggle entry and is not called from the
self-check.

Untied embed + lm_head shrink by ``2 * (V_old - 16) * hidden`` parameters.
BF16 bytes removed = ``2 * (V_old - 16) * hidden * 2``.
For the official V=248320, d=5120 that is 5085265920 bytes (about 5.085GB
decimal, about 4.736GiB). Compute the figure from the loaded tensor shapes.
Never replace an actual measurement with 5.085.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
import sys


# Digits, one newline character (not the two-character backslash-n), then the
# role words and the three Qwen control strings. New ids are 0..15 in this order.
GRID_SYMBOLS: list[str] = [
    "0",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "\n",
    "user",
    "assistant",
    "<|endoftext|>",
    "<|im_start|>",
    "<|im_end|>",
]

EOS_SYMBOL = "<|im_end|>"
BOS_SYMBOL = "<|endoftext|>"
PAD_SYMBOL = "<|im_end|>"

_WRITE_CALLS = 0


class GridSymbolError(ValueError):
    """A grid symbol is not exactly one old token. No row was averaged."""

    def __init__(self, symbol: str, detail: str):
        self.symbol = symbol
        self.detail = detail
        payload = {
            "symbol": symbol,
            "repr": repr(symbol),
            "detail": detail,
            "averaged": False,
        }
        super().__init__(
            "grid symbol is not a single old token: "
            + json.dumps(payload, ensure_ascii=False)
        )


def _jsonable(value):
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable(item())
        except Exception:
            pass
    return str(value)


def _as_int(value):
    if isinstance(value, bool) or value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _is_unk_fallback(tokenizer, symbol: str, token_id: int) -> bool:
    """True when convert_tokens_to_ids fell back to unk, not when the symbol is unk."""
    unk_id = _as_int(getattr(tokenizer, "unk_token_id", None))
    if unk_id is None or token_id != unk_id:
        return False
    unk_token = getattr(tokenizer, "unk_token", None)
    if isinstance(unk_token, str) and unk_token == symbol:
        return False
    content = getattr(unk_token, "content", None)
    if isinstance(content, str) and content == symbol:
        return False
    reverse = getattr(tokenizer, "convert_ids_to_tokens", None)
    if callable(reverse):
        try:
            back = reverse(token_id)
        except Exception:
            back = None
        if back == symbol or getattr(back, "content", None) == symbol:
            return False
    get_vocab = getattr(tokenizer, "get_vocab", None)
    if callable(get_vocab):
        try:
            vocab = get_vocab()
        except Exception:
            vocab = None
        if isinstance(vocab, dict) and _as_int(vocab.get(symbol)) == token_id:
            return False
    return True


def _convert_id(tokenizer, symbol: str):
    fn = getattr(tokenizer, "convert_tokens_to_ids", None)
    if not callable(fn):
        return None
    try:
        raw = fn(symbol)
    except Exception:
        return None
    if isinstance(raw, tuple):
        raw = list(raw)
    if isinstance(raw, list):
        if len(raw) != 1:
            return None
        raw = raw[0]
    token_id = _as_int(raw)
    if token_id is None or token_id < 0:
        return None
    if _is_unk_fallback(tokenizer, symbol, token_id):
        return None
    return token_id


def _coerce_id_list(raw):
    if hasattr(raw, "ids") and not isinstance(raw, (str, bytes, list, tuple)):
        raw = raw.ids
    elif hasattr(raw, "tolist") and not isinstance(raw, (str, bytes, list, tuple)):
        raw = raw.tolist()
    if isinstance(raw, tuple):
        raw = list(raw)
    if not isinstance(raw, list):
        raise ValueError("encode did not return a token-id list: %r" % (raw,))
    return raw


def _encode_one(tokenizer, symbol: str) -> int:
    fn = getattr(tokenizer, "encode", None)
    if not callable(fn):
        raise GridSymbolError(symbol, "tokenizer has no encode()")
    try:
        raw = fn(symbol, add_special_tokens=False)
    except TypeError:
        raw = fn(symbol)
    ids = _coerce_id_list(raw)
    if len(ids) != 1:
        detail = "encode returned %d ids %s; want exactly 1." % (len(ids), ids)
        if symbol in "0123456789":
            detail += (
                " Tried the bare digit only. Refusing to average embeddings"
                " or guess another surface form."
            )
        else:
            detail += " Refusing to average embeddings."
        raise GridSymbolError(symbol, detail)
    token_id = _as_int(ids[0])
    if token_id is None or token_id < 0:
        raise GridSymbolError(symbol, "encode returned a non-id %r" % (ids[0],))
    return token_id


def resolve_old_ids(tokenizer) -> list[int]:
    """One old id per GRID_SYMBOLS entry. Stop at the first symbol that is not one token."""
    old_ids: list[int] = []
    for symbol in GRID_SYMBOLS:
        token_id = _convert_id(tokenizer, symbol)
        if token_id is None:
            # Digits use the bare character only. A multi-id encode stops the convert.
            token_id = _encode_one(tokenizer, symbol)
        old_ids.append(token_id)
    if len(set(old_ids)) != len(old_ids):
        raise GridSymbolError(
            "",
            "old ids are not unique: %s. Refusing to alias two symbols to one row."
            % (old_ids,),
        )
    return old_ids


def build_maps(old_ids) -> tuple[dict, dict]:
    ids = [_as_int(item) for item in old_ids]
    if len(ids) != len(GRID_SYMBOLS) or any(item is None or item < 0 for item in ids):
        raise ValueError("old_ids must be %d non-negative ints" % len(GRID_SYMBOLS))
    old_to_new: dict = {}
    new_to_old: dict = {}
    for new_id, old_id in enumerate(ids):
        if old_id in old_to_new:
            raise ValueError("duplicate old id %s" % old_id)
        old_to_new[old_id] = new_id
        new_to_old[new_id] = old_id
    return old_to_new, new_to_old


def _is_torch_tensor(obj) -> bool:
    module = type(obj).__module__
    if module != "torch" and not module.startswith("torch."):
        return False
    return hasattr(obj, "shape") and hasattr(obj, "index_select") and hasattr(obj, "dim")


def _is_numpy_array(obj) -> bool:
    return type(obj).__module__ == "numpy" and hasattr(obj, "shape")


def _shares_storage(left, right) -> bool:
    if left is right:
        return True
    if _is_torch_tensor(left) and _is_torch_tensor(right):
        try:
            left_storage = left.untyped_storage()
            right_storage = right.untyped_storage()
            if int(left_storage.size()) > 0 and left_storage.data_ptr() == right_storage.data_ptr():
                return True
        except Exception:
            try:
                if left.storage().data_ptr() == right.storage().data_ptr():
                    return True
            except Exception:
                pass
        try:
            if int(left.numel()) > 0 and left.data_ptr() == right.data_ptr():
                return True
        except Exception:
            return False
        return False
    if _is_numpy_array(left) and _is_numpy_array(right):
        import numpy as np

        return bool(np.shares_memory(left, right))
    return False


def _matrix_shape(rows):
    if _is_torch_tensor(rows) or _is_numpy_array(rows):
        shape = tuple(int(dim) for dim in rows.shape)
        if len(shape) != 2:
            raise ValueError("weight must be rank 2 [vocab, hidden], got %s" % (shape,))
        return shape
    if isinstance(rows, (str, bytes)) or not hasattr(rows, "__len__") or len(rows) == 0:
        raise ValueError("weight must be a non-empty matrix")
    width = None
    for row in rows:
        if isinstance(row, (str, bytes)) or not hasattr(row, "__len__"):
            raise ValueError("weight row is not a numeric sequence")
        if width is None:
            width = len(row)
        elif len(row) != width:
            raise ValueError("ragged weight matrix")
    return len(rows), int(width)


def select_rows(rows, ids):
    """Independent copies of rows[id]. Torch and numpy are optional; lists always work."""
    wanted = []
    for item in ids:
        token_id = _as_int(item)
        if token_id is None or token_id < 0:
            raise ValueError("bad row id %r" % (item,))
        wanted.append(token_id)
    height, width = _matrix_shape(rows)
    if any(token_id >= height for token_id in wanted):
        raise ValueError(
            "old id outside dim 0 (size %d, width %d). "
            "Qwen stores lm_head as [vocab, hidden]; refusing to transpose or average."
            % (height, width)
        )
    if _is_torch_tensor(rows):
        import torch

        index = torch.tensor(wanted, dtype=torch.long, device=rows.device)
        picked = rows.index_select(0, index).clone()
        if _shares_storage(picked, rows):
            picked = picked.contiguous().clone()
        if _shares_storage(picked, rows):
            raise RuntimeError("selected rows still share storage with the source matrix")
        return picked
    if _is_numpy_array(rows):
        import numpy as np

        index = np.asarray(wanted, dtype=np.int64)
        picked = np.take(np.asarray(rows), index, axis=0)
        return np.array(picked, copy=True)
    copied = []
    for token_id in wanted:
        row = rows[token_id]
        if hasattr(row, "tolist") and not isinstance(row, (list, tuple)):
            copied.append([item for item in row.tolist()])
        else:
            copied.append([item for item in row])
    return copied


def convert_state(embed_weight, lm_head_weight, old_ids):
    """Copy selected rows from untied [vocab, hidden] matrices. Does not tie them."""
    if len(list(old_ids)) != len(GRID_SYMBOLS):
        raise ValueError("convert_state expects %d old ids" % len(GRID_SYMBOLS))
    if _shares_storage(embed_weight, lm_head_weight):
        raise RuntimeError(
            "embed_weight and lm_head_weight share storage; refusing to tie or convert"
        )
    embed_shape = _matrix_shape(embed_weight)
    head_shape = _matrix_shape(lm_head_weight)
    if embed_shape != head_shape:
        raise ValueError(
            "embed shape %s and lm_head shape %s differ; refusing to broadcast or average"
            % (embed_shape, head_shape)
        )
    new_embed = select_rows(embed_weight, old_ids)
    new_head = select_rows(lm_head_weight, old_ids)
    if _shares_storage(new_embed, new_head) or _shares_storage(new_embed, embed_weight):
        raise RuntimeError("cloned vocab rows still share storage")
    if _shares_storage(new_head, lm_head_weight):
        raise RuntimeError("cloned lm_head rows still share storage")
    return new_embed, new_head


def vocab_matrix_shrink(v_old: int, hidden: int) -> dict:
    # Untied embed + lm_head shrink by 2 * (V_old - 16) * hidden parameters.
    # BF16 bytes removed = 2 * (V_old - 16) * hidden * 2.
    # Official V=248320, d=5120 -> 5085265920 bytes (~5.085GB decimal, ~4.736GiB).
    # Callers must pass the actual tensor shapes. Do not substitute 5.085.
    v_old = int(v_old)
    hidden = int(hidden)
    if v_old < len(GRID_SYMBOLS):
        raise ValueError("V_old %s is smaller than the grid vocab" % v_old)
    if hidden < 1:
        raise ValueError("hidden must be positive")
    removed_parameters = 2 * (v_old - len(GRID_SYMBOLS)) * hidden
    bf16_bytes_removed = removed_parameters * 2
    return {
        "V_old": v_old,
        "hidden": hidden,
        "vocab_after": len(GRID_SYMBOLS),
        "removed_parameters": removed_parameters,
        "bf16_bytes_removed": bf16_bytes_removed,
        "formula_parameters": "2 * (V_old - 16) * hidden",
        "formula_bf16_bytes": "2 * (V_old - 16) * hidden * 2",
    }


def encode_grid_string(text: str) -> list[int]:
    """Longest-match encode. No byte fallback and no digit merges."""
    symbols = sorted(GRID_SYMBOLS, key=len, reverse=True)
    ids = []
    index = 0
    length = len(text)
    while index < length:
        matched = None
        for symbol in symbols:
            if text.startswith(symbol, index):
                matched = symbol
                break
        if matched is None:
            raise ValueError(
                "no single grid token at %d (%r); refusing byte fallback"
                % (index, text[index : index + 16])
            )
        ids.append(GRID_SYMBOLS.index(matched))
        index += len(matched)
    return ids


def build_formatter() -> dict:
    return {
        "name": "G3",
        "chat_template": None,
        "think_tokens": False,
        "no_chat_template": True,
        "no_think_tokens": True,
        "demo": "user\n<grid>\nassistant\n<grid>\n",
        "query": "user\n<grid>\nassistant\n",
        "supervised_target": "<grid>\n<|im_end|>",
        "supervised_target_note": (
            "The supervised target is <grid> followed by the newline token and then <|im_end|>."
        ),
        "grid_encoding": (
            "Grids are digits with no spaces, rows separated by the newline token."
        ),
        "distinct_from_g1": (
            "This is a different format from G1 space-separated digits. "
            "Do not reuse G1 adapters or G1 token ids."
        ),
        "newline_token": "\n",
        "newline_is_single_character": True,
        "placeholder_grid_is_not_a_token": True,
        "symbols": list(GRID_SYMBOLS),
        "tokenizer_loader": "grid_tokenizer.GridTokenizer.from_pretrained(out_dir)",
        "training_loop": None,
        "notes": [
            "no chat template and no think tokens",
            "a demo is user\\n<grid>\\nassistant\\n<grid>\\n using the single tokens",
            "the query is user\\n<grid>\\nassistant\\n",
            "the supervised target is <grid>\\n plus <|im_end|>",
            "grids are digits with no spaces, rows separated by the newline token",
            "This is a different format from G1 space-separated digits. Do not reuse G1 adapters or G1 token ids.",
        ],
    }


def _wrapper_source() -> str:
    symbols = json.dumps(GRID_SYMBOLS, ensure_ascii=False)
    return (
        "# -*- coding: utf-8 -*-\n"
        "# Generated by scripts/convert_grid_vocab.py. Do not point this at a G1 tokenizer.\n"
        '"""G3 16-symbol wrapper. No chat template, no think tokens, no byte fallback."""\n'
        "import json\n"
        "import os\n"
        "\n"
        "GRID_SYMBOLS = %s\n"
        "\n"
        "\n"
        "def _encode(text):\n"
        "    symbols = sorted(GRID_SYMBOLS, key=len, reverse=True)\n"
        "    ids = []\n"
        "    index = 0\n"
        "    while index < len(text):\n"
        "        matched = None\n"
        "        for symbol in symbols:\n"
        "            if text.startswith(symbol, index):\n"
        "                matched = symbol\n"
        "                break\n"
        "        if matched is None:\n"
        "            raise ValueError(\n"
        "                'no single grid token at %%d (%%r); refusing byte fallback'\n"
        "                %% (index, text[index:index + 16])\n"
        "            )\n"
        "        ids.append(GRID_SYMBOLS.index(matched))\n"
        "        index += len(matched)\n"
        "    return ids\n"
        "\n"
        "\n"
        "class GridTokenizer(object):\n"
        "    def __init__(self, symbols=None):\n"
        "        if symbols is None:\n"
        "            symbols = list(GRID_SYMBOLS)\n"
        "        if list(symbols) != list(GRID_SYMBOLS):\n"
        "            raise ValueError('vocab must be exactly the 16 G3 symbols in order')\n"
        "        self._symbols = list(symbols)\n"
        "        self._to_id = {}\n"
        "        for index, symbol in enumerate(self._symbols):\n"
        "            self._to_id[symbol] = index\n"
        "        self.vocab_size = len(self._symbols)\n"
        "        self.bos_token = '<|endoftext|>'\n"
        "        self.eos_token = '<|im_end|>'\n"
        "        self.pad_token = '<|im_end|>'\n"
        "        self.unk_token = None\n"
        "        self.bos_token_id = self._to_id[self.bos_token]\n"
        "        self.eos_token_id = self._to_id[self.eos_token]\n"
        "        self.pad_token_id = self._to_id[self.pad_token]\n"
        "        self.unk_token_id = None\n"
        "        self.chat_template = None\n"
        "        self.model_max_length = 262144\n"
        "        self.padding_side = 'right'\n"
        "        self.add_bos_token = False\n"
        "        self.add_eos_token = False\n"
        "\n"
        "    @classmethod\n"
        "    def from_pretrained(cls, path):\n"
        "        with open(os.path.join(path, 'vocab.json'), 'r', encoding='utf-8') as handle:\n"
        "            vocab = json.load(handle)\n"
        "        if len(vocab) != len(GRID_SYMBOLS):\n"
        "            raise ValueError('vocab.json does not contain exactly 16 symbols')\n"
        "        symbols = [None] * len(GRID_SYMBOLS)\n"
        "        for token, index in vocab.items():\n"
        "            symbols[int(index)] = token\n"
        "        if symbols != GRID_SYMBOLS:\n"
        "            raise ValueError('vocab.json symbols do not match the G3 order')\n"
        "        tokenizer = cls(symbols)\n"
        "        config_path = os.path.join(path, 'tokenizer_config.json')\n"
        "        if os.path.isfile(config_path):\n"
        "            with open(config_path, 'r', encoding='utf-8') as handle:\n"
        "                config = json.load(handle)\n"
        "            tokenizer.chat_template = config.get('chat_template', None)\n"
        "            if tokenizer.chat_template:\n"
        "                raise ValueError('G3 tokenizer refuses a chat template')\n"
        "            max_length = config.get('model_max_length')\n"
        "            if isinstance(max_length, int) and not isinstance(max_length, bool):\n"
        "                tokenizer.model_max_length = max_length\n"
        "        return tokenizer\n"
        "\n"
        "    def convert_tokens_to_ids(self, token):\n"
        "        if isinstance(token, (list, tuple)):\n"
        "            return [self.convert_tokens_to_ids(item) for item in token]\n"
        "        return self._to_id.get(token)\n"
        "\n"
        "    def convert_ids_to_tokens(self, index):\n"
        "        if isinstance(index, (list, tuple)):\n"
        "            return [self.convert_ids_to_tokens(item) for item in index]\n"
        "        return self._symbols[int(index)]\n"
        "\n"
        "    def encode(self, text, add_special_tokens=False):\n"
        "        del add_special_tokens\n"
        "        return _encode(text)\n"
        "\n"
        "    def decode(self, ids, skip_special_tokens=False):\n"
        "        specials = set(['<|endoftext|>', '<|im_start|>', '<|im_end|>'])\n"
        "        parts = []\n"
        "        for index in ids:\n"
        "            token = self._symbols[int(index)]\n"
        "            if skip_special_tokens and token in specials:\n"
        "                continue\n"
        "            parts.append(token)\n"
        "        return ''.join(parts)\n"
        "\n"
        "    def __call__(self, text, add_special_tokens=False, return_tensors=None):\n"
        "        ids = self.encode(text, add_special_tokens=add_special_tokens)\n"
        "        mask = [1] * len(ids)\n"
        "        if return_tensors is None:\n"
        "            return {'input_ids': ids, 'attention_mask': mask}\n"
        "        if return_tensors == 'pt':\n"
        "            import torch\n"
        "            return {\n"
        "                'input_ids': torch.tensor([ids], dtype=torch.long),\n"
        "                'attention_mask': torch.tensor([mask], dtype=torch.long),\n"
        "            }\n"
        "        raise ValueError('unsupported return_tensors')\n"
        "\n"
        "    def __len__(self):\n"
        "        return self.vocab_size\n"
        "\n"
        "    def get_vocab(self):\n"
        "        return dict(self._to_id)\n"
        "\n"
        "    def apply_chat_template(self, *args, **kwargs):\n"
        "        del args, kwargs\n"
        "        raise RuntimeError('G3 has no chat template and no think tokens')\n"
        % symbols
    )


def _string_key_map(mapping: dict) -> dict:
    return {str(key): int(value) for key, value in mapping.items()}


def _special_ids() -> dict:
    return {
        "eos_token_id": GRID_SYMBOLS.index(EOS_SYMBOL),
        "bos_token_id": GRID_SYMBOLS.index(BOS_SYMBOL),
        "pad_token_id": GRID_SYMBOLS.index(PAD_SYMBOL),
    }


def _model_max_length(tokenizer):
    value = getattr(tokenizer, "model_max_length", None)
    if isinstance(value, bool) or not isinstance(value, int):
        return 262144, "default_262144_not_on_tokenizer"
    if value <= 0 or value > 10**12:
        return 262144, "default_262144_replaced_placeholder"
    return value, "source_tokenizer"


def _write_json(path: str, payload) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")


def _manual_tokenizer_json(symbols: list[str]) -> dict:
    vocab = {symbol: index for index, symbol in enumerate(symbols)}
    # Longest-first alternation. json.dumps escapes the newline and backslashes.
    parts = []
    for symbol in sorted(symbols, key=len, reverse=True):
        if symbol == "\n":
            parts.append(r"\n")
        else:
            parts.append(
                "".join("\\" + ch if ch in r"\|()[]{}?*+.^$" else ch for ch in symbol)
            )
    pattern = "|".join(parts)
    added = []
    for symbol in (BOS_SYMBOL, "<|im_start|>", EOS_SYMBOL):
        added.append(
            {
                "id": vocab[symbol],
                "content": symbol,
                "single_word": False,
                "lstrip": False,
                "rstrip": False,
                "normalized": False,
                "special": True,
            }
        )
    return {
        "version": "1.0",
        "truncation": None,
        "padding": None,
        "added_tokens": added,
        "normalizer": None,
        "pre_tokenizer": {
            "type": "Split",
            "pattern": {"Regex": pattern},
            "behavior": "Isolated",
            "invert": False,
        },
        "post_processor": None,
        "decoder": {"type": "Fuse"},
        "model": {
            "type": "WordLevel",
            "vocab": vocab,
            "unk_token": BOS_SYMBOL,
        },
    }


def _try_write_tokenizers_lib(path: str) -> None:
    import re

    from tokenizers import AddedToken, Regex, Tokenizer
    from tokenizers.decoders import Fuse
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Split

    vocab = {symbol: index for index, symbol in enumerate(GRID_SYMBOLS)}
    tokenizer = Tokenizer(WordLevel(vocab=vocab, unk_token=BOS_SYMBOL))
    parts = []
    for symbol in sorted(GRID_SYMBOLS, key=len, reverse=True):
        if symbol == "\n":
            parts.append(r"\n")
        else:
            parts.append(re.escape(symbol))
    tokenizer.pre_tokenizer = Split(Regex("|".join(parts)), behavior="isolated")
    tokenizer.decoder = Fuse()
    tokenizer.add_special_tokens(
        [
            AddedToken(symbol, special=True, normalized=False)
            for symbol in (BOS_SYMBOL, "<|im_start|>", EOS_SYMBOL)
        ]
    )
    for symbol, index in vocab.items():
        got = tokenizer.token_to_id(symbol)
        if got != index:
            raise RuntimeError("tokenizers reassigned %r to %s" % (symbol, got))
    sample = "user\n01\nassistant\n10\n<|im_end|>"
    if list(tokenizer.encode(sample).ids) != encode_grid_string(sample):
        raise RuntimeError("tokenizers encode does not match the G3 wrapper")
    if list(tokenizer.encode("890").ids) != [8, 9, 0]:
        raise RuntimeError("tokenizers merged or byte-fallbacked digits")
    tokenizer.save(path)


def _write_tokenizer_files(out_dir: str, tokenizer, old_ids, old_to_new, new_to_old) -> None:
    vocab = {symbol: index for index, symbol in enumerate(GRID_SYMBOLS)}
    _write_json(os.path.join(out_dir, "vocab.json"), vocab)
    tokenizer_path = os.path.join(out_dir, "tokenizer.json")
    try:
        _try_write_tokenizers_lib(tokenizer_path)
    except Exception:
        _write_json(tokenizer_path, _manual_tokenizer_json(GRID_SYMBOLS))
    max_length, max_length_source = _model_max_length(tokenizer)
    specials = _special_ids()
    added_decoder = {}
    added_list = []
    for symbol in (BOS_SYMBOL, "<|im_start|>", EOS_SYMBOL):
        index = GRID_SYMBOLS.index(symbol)
        record = {
            "content": symbol,
            "lstrip": False,
            "normalized": False,
            "rstrip": False,
            "single_word": False,
            "special": True,
        }
        added_decoder[str(index)] = record
        listed = {"id": index}
        listed.update(record)
        added_list.append(listed)
    tokenizer_config = {
        "tokenizer_class": "GridTokenizer",
        "auto_map": {"AutoTokenizer": ["grid_tokenizer.GridTokenizer", None]},
        "model_max_length": max_length,
        "model_max_length_source": max_length_source,
        "padding_side": "right",
        "truncation_side": "right",
        "clean_up_tokenization_spaces": False,
        "add_bos_token": False,
        "add_eos_token": False,
        "bos_token": BOS_SYMBOL,
        "eos_token": EOS_SYMBOL,
        "pad_token": PAD_SYMBOL,
        "unk_token": None,
        "chat_template": None,
        "model_input_names": ["input_ids", "attention_mask"],
        "added_tokens_decoder": added_decoder,
        "additional_special_tokens": ["<|im_start|>"],
        "bos_token_id": specials["bos_token_id"],
        "eos_token_id": specials["eos_token_id"],
        "pad_token_id": specials["pad_token_id"],
    }
    _write_json(os.path.join(out_dir, "tokenizer_config.json"), tokenizer_config)
    _write_json(
        os.path.join(out_dir, "special_tokens_map.json"),
        {
            "bos_token": BOS_SYMBOL,
            "eos_token": EOS_SYMBOL,
            "pad_token": PAD_SYMBOL,
            "additional_special_tokens": ["<|im_start|>"],
        },
    )
    _write_json(os.path.join(out_dir, "added_tokens.json"), added_list)
    _write_json(
        os.path.join(out_dir, "old_id_map.json"),
        {
            "symbols": list(GRID_SYMBOLS),
            "old_ids": [int(item) for item in old_ids],
            "old_to_new": _string_key_map(old_to_new),
            "new_to_old": _string_key_map(new_to_old),
        },
    )
    with open(os.path.join(out_dir, "grid_tokenizer.py"), "w", encoding="utf-8") as handle:
        handle.write(_wrapper_source())
    _write_json(os.path.join(out_dir, "formatter.json"), build_formatter())


def _remap_id_value(value, old_to_new):
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        if value in old_to_new:
            return int(old_to_new[value])
        return None
    if isinstance(value, (list, tuple)) and value:
        mapped = []
        for item in value:
            if isinstance(item, bool) or not isinstance(item, int) or item not in old_to_new:
                return None
            mapped.append(int(old_to_new[item]))
        return mapped
    return None


def _set_field(cfg, name: str, value) -> None:
    if isinstance(cfg, dict):
        cfg[name] = value
    else:
        setattr(cfg, name, value)


def _field(cfg, name: str):
    if isinstance(cfg, dict):
        return cfg.get(name), name in cfg
    try:
        exists = hasattr(cfg, name)
    except Exception:
        return None, False
    if not exists:
        return None, False
    try:
        return getattr(cfg, name), True
    except Exception:
        return None, False


def _token_id_names(cfg) -> list[str]:
    names = set()
    if isinstance(cfg, dict):
        pool = list(cfg.keys())
    else:
        pool = [name for name in dir(cfg) if not name.startswith("_")]
    for name in pool:
        if name.endswith("_token_id") or name.endswith("_token_ids"):
            names.add(name)
    return sorted(names)


def _update_live_config(cfg, where: str, old_to_new: dict, changes: list, set_vocab: bool) -> None:
    specials = _special_ids()
    if set_vocab:
        old_vocab, _has_vocab = _field(cfg, "vocab_size")
        _set_field(cfg, "vocab_size", len(GRID_SYMBOLS))
        changes.append(
            {
                "where": where,
                "field": "vocab_size",
                "old": _jsonable(old_vocab),
                "new": len(GRID_SYMBOLS),
            }
        )
        old_tie, _has_tie = _field(cfg, "tie_word_embeddings")
        _set_field(cfg, "tie_word_embeddings", False)
        changes.append(
            {
                "where": where,
                "field": "tie_word_embeddings",
                "old": _jsonable(old_tie),
                "new": False,
            }
        )
    for name, new_id in specials.items():
        old_value, existed = _field(cfg, name)
        if not existed and not set_vocab and where != "generation_config":
            continue
        _set_field(cfg, name, new_id)
        changes.append(
            {
                "where": where,
                "field": name,
                "old": _jsonable(old_value),
                "new": new_id,
                "existed": existed,
            }
        )
    explicit = set(specials)
    for name in _token_id_names(cfg):
        if name in explicit:
            continue
        old_value, existed = _field(cfg, name)
        if not existed:
            continue
        mapped = _remap_id_value(old_value, old_to_new)
        if mapped is None:
            changes.append(
                {
                    "where": where,
                    "field": name,
                    "old": _jsonable(old_value),
                    "action": "left_unchanged_unmapped",
                }
            )
            continue
        _set_field(cfg, name, mapped)
        changes.append(
            {
                "where": where,
                "field": name,
                "old": _jsonable(old_value),
                "new": mapped,
                "action": "remapped",
            }
        )


def _child_config(cfg, name: str):
    if isinstance(cfg, dict):
        child = cfg.get(name)
    else:
        child = getattr(cfg, name, None) if hasattr(cfg, name) else None
    if child is None or child is cfg:
        return None
    if isinstance(child, (str, int, float, bool)):
        return None
    return child


def _patch_saved_tree(node, v_old: int, hidden: int, specials: dict, generation_root: bool) -> None:
    if isinstance(node, list):
        for item in node:
            _patch_saved_tree(item, v_old, hidden, specials, False)
        return
    if not isinstance(node, dict):
        return
    vocab = node.get("vocab_size")
    hidden_here = node.get("hidden_size")
    # Only the text config that owns this vocab. Do not rewrite vision or unrelated sizes.
    textish = isinstance(vocab, int) and (
        vocab == v_old or (hidden_here == hidden and "num_hidden_layers" in node and vocab >= 16)
    )
    if textish:
        node["vocab_size"] = len(GRID_SYMBOLS)
        node["tie_word_embeddings"] = False
        node["eos_token_id"] = specials["eos_token_id"]
        node["bos_token_id"] = specials["bos_token_id"]
        node["pad_token_id"] = specials["pad_token_id"]
    elif "tie_word_embeddings" in node:
        node["tie_word_embeddings"] = False
    if generation_root or (
        "eos_token_id" in node and "hidden_size" not in node and "vocab_size" not in node
    ):
        node["eos_token_id"] = specials["eos_token_id"]
        node["bos_token_id"] = specials["bos_token_id"]
        node["pad_token_id"] = specials["pad_token_id"]
    for key, value in list(node.items()):
        if key == "vision_config":
            continue
        _patch_saved_tree(value, v_old, hidden, specials, False)


def _patch_saved_configs(out_dir: str, v_old: int, hidden: int) -> None:
    specials = _special_ids()
    for dirpath, _dirs, files in os.walk(out_dir):
        for name in files:
            if name not in ("config.json", "generation_config.json"):
                continue
            path = os.path.join(dirpath, name)
            with open(path, "r", encoding="utf-8") as handle:
                payload = json.load(handle)
            _patch_saved_tree(
                payload,
                v_old,
                hidden,
                specials,
                generation_root=(name == "generation_config.json"),
            )
            _write_json(path, payload)


def _safetensors_shapes(path: str) -> dict:
    with open(path, "rb") as handle:
        header_raw = handle.read(8)
        if len(header_raw) != 8:
            raise RuntimeError("short safetensors header: %s" % path)
        (header_len,) = struct.unpack("<Q", header_raw)
        header = handle.read(header_len)
    meta = json.loads(header.decode("utf-8"))
    shapes = {}
    for key, info in meta.items():
        if key == "__metadata__":
            continue
        shapes[key] = (info.get("dtype"), list(info.get("shape", [])))
    return shapes


def _saved_tensor_shapes(out_dir: str) -> dict:
    shapes = {}
    for dirpath, _dirs, files in os.walk(out_dir):
        for name in files:
            if name.endswith(".safetensors"):
                shapes.update(_safetensors_shapes(os.path.join(dirpath, name)))
    return shapes


def _param_name(model, target) -> str:
    for name, param in model.named_parameters():
        if param is target:
            return name
    shape = tuple(int(dim) for dim in target.shape)
    hits = []
    for name, param in model.named_parameters():
        if tuple(int(dim) for dim in param.shape) != shape:
            continue
        try:
            same = param.data_ptr() == target.data_ptr()
        except Exception:
            same = False
        if same:
            hits.append(name)
    if len(hits) == 1:
        return hits[0]
    raise RuntimeError("could not name a vocab parameter, hits=%s" % (hits,))


def _install_weight(module, cpu_bf16) -> None:
    import torch

    if hasattr(module, "bias") and module.bias is not None:
        raise RuntimeError("vocab matrix has a bias; refusing to drop or resize it")
    old = module.weight
    data = cpu_bf16.detach().to(device=old.device, dtype=torch.bfloat16).contiguous()
    module.weight = torch.nn.Parameter(data, requires_grad=False)
    rows = int(data.shape[0])
    width = int(data.shape[1])
    if hasattr(module, "num_embeddings"):
        module.num_embeddings = rows
    if hasattr(module, "embedding_dim"):
        module.embedding_dim = width
    if hasattr(module, "out_features"):
        module.out_features = rows
    # padding_idx zeros or freezes that row. Pad id 15 is also <|im_end|>, so leave
    # the copied vector trainable instead of pointing padding_idx at it.
    if getattr(module, "padding_idx", None) is not None:
        module.padding_idx = None


def _backbone_marker(model, embed_weight, head_weight):
    for name, param in model.named_parameters():
        if param is embed_weight or param is head_weight:
            continue
        if name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"):
            continue
        return name, param.data_ptr(), tuple(int(dim) for dim in param.shape)
    raise RuntimeError("model has no backbone parameter")


def _read_layer_count(config):
    value = getattr(config, "num_hidden_layers", None)
    if value is not None:
        return value
    text = _child_config(config, "text_config")
    if text is None:
        return None
    return getattr(text, "num_hidden_layers", None) if not isinstance(text, dict) else text.get(
        "num_hidden_layers"
    )


def _to_cpu_float(tensor):
    import torch

    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().clone()


def write_converted_model(model_dir, out_dir, tokenizer):
    """Load the trainer's causal LM, copy 16 untied rows, and save a G3 directory.

    Not used by ``--self-check``. Does not download when ``model_dir`` is not a
    local directory, and does not copy an optimizer or an old adapter.
    """
    global _WRITE_CALLS
    _WRITE_CALLS += 1
    model_dir = os.path.abspath(model_dir)
    out_dir = os.path.abspath(out_dir)
    if not os.path.isdir(model_dir) or not os.path.isfile(os.path.join(model_dir, "config.json")):
        raise FileNotFoundError(
            "model_dir must be a local model directory with config.json; refusing to download weights"
        )
    if os.path.exists(out_dir):
        if not os.path.isdir(out_dir):
            raise NotADirectoryError(out_dir)
        if os.listdir(out_dir):
            raise FileExistsError(
                "out_dir must be empty so old shards, adapters, and optimizer state are not mixed in"
            )
    else:
        os.makedirs(out_dir)

    old_ids = resolve_old_ids(tokenizer)
    old_to_new, new_to_old = build_maps(old_ids)

    import torch
    from transformers import AutoModelForCausalLM

    # Same arguments as experiments/006 trainers, plus local_files_only so a
    # partial directory cannot fetch Qwen weights.
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        dtype=torch.bfloat16,
        device_map="balanced",
        low_cpu_mem_usage=True,
        trust_remote_code=False,
        local_files_only=True,
    )
    embed_module = model.get_input_embeddings()
    head_module = model.get_output_embeddings()
    if embed_module is None or head_module is None:
        raise RuntimeError("model is missing input embeddings or lm_head")
    embed_weight = embed_module.weight
    head_weight = head_module.weight
    if _shares_storage(embed_weight, head_weight):
        raise RuntimeError(
            "loaded embed and lm_head share storage; refusing to tie or convert"
        )
    embed_name = _param_name(model, embed_weight)
    head_name = _param_name(model, head_weight)
    if embed_name == head_name:
        raise RuntimeError("embed and lm_head resolved to the same parameter name")
    backbone_name, backbone_ptr, backbone_shape = _backbone_marker(
        model, embed_weight, head_weight
    )
    layers_before = _read_layer_count(model.config)

    # Tie stays false. Do not call tie_weights and do not reuse one tensor for both.
    changes = []
    _set_field(model.config, "tie_word_embeddings", False)
    text_config = _child_config(model.config, "text_config")
    if text_config is not None:
        _set_field(text_config, "tie_word_embeddings", False)

    v_old, hidden = _matrix_shape(embed_weight)
    head_shape = _matrix_shape(head_weight)
    if head_shape != (v_old, hidden):
        raise ValueError("lm_head shape %s != embed shape %s" % (head_shape, (v_old, hidden)))
    embed_dtype = str(embed_weight.dtype)
    head_dtype = str(head_weight.dtype)
    new_embed, new_head = convert_state(embed_weight.detach(), head_weight.detach(), old_ids)
    embed_fp32 = _to_cpu_float(new_embed)
    head_fp32 = _to_cpu_float(new_head)
    if _shares_storage(embed_fp32, head_fp32):
        raise RuntimeError("CPU float32 copies share storage")
    embed_bf16 = embed_fp32.to(dtype=torch.bfloat16).contiguous()
    head_bf16 = head_fp32.to(dtype=torch.bfloat16).contiguous()
    embed_padding_before = getattr(embed_module, "padding_idx", None)
    _install_weight(embed_module, embed_bf16)
    _install_weight(head_module, head_bf16)
    _, top_has_vocab = _field(model.config, "vocab_size")
    _update_live_config(
        model.config,
        "model.config",
        old_to_new,
        changes,
        set_vocab=bool(top_has_vocab or text_config is None),
    )
    if text_config is not None:
        _update_live_config(
            text_config, "model.config.text_config", old_to_new, changes, set_vocab=True
        )
    generation = getattr(model, "generation_config", None)
    if generation is not None:
        _update_live_config(
            generation, "generation_config", old_to_new, changes, set_vocab=False
        )
        if hasattr(generation, "tie_word_embeddings"):
            generation.tie_word_embeddings = False

    found = None
    for name, param in model.named_parameters():
        if name == backbone_name:
            found = param
            break
    if found is None or found.data_ptr() != backbone_ptr or tuple(int(dim) for dim in found.shape) != backbone_shape:
        raise RuntimeError("backbone parameter changed during vocab conversion")
    if _read_layer_count(model.config) != layers_before:
        raise RuntimeError("num_hidden_layers changed during vocab conversion")

    model.save_pretrained(out_dir, safe_serialization=True)
    _patch_saved_configs(out_dir, v_old, hidden)
    shapes = _saved_tensor_shapes(out_dir)
    embed_keys = sorted(key for key in shapes if key.endswith("embed_tokens.weight"))
    head_keys = sorted(key for key in shapes if key.endswith("lm_head.weight"))
    if not embed_keys or not head_keys:
        raise RuntimeError(
            "saved model dropped embed or lm_head (embed=%s lm_head=%s)"
            % (embed_keys, head_keys)
        )
    for key in embed_keys + head_keys:
        dtype, shape = shapes[key]
        if shape != [len(GRID_SYMBOLS), hidden]:
            raise RuntimeError("saved %s has shape %s, want [16, %s]" % (key, shape, hidden))
        if str(dtype).upper() not in ("BF16", "BFLOAT16"):
            raise RuntimeError("saved %s dtype %s, expected BF16" % (key, dtype))
    forbidden = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
        "optimizer.bin",
        "optimizer.pt",
        "scheduler.pt",
        "trainer_state.json",
        "rng_state.pth",
    )
    present = set(os.listdir(out_dir))
    leaked = sorted(name for name in forbidden if name in present)
    if leaked:
        raise RuntimeError("refusing optimizer or adapter files in out_dir: %s" % (leaked,))

    _write_tokenizer_files(out_dir, tokenizer, old_ids, old_to_new, new_to_old)
    shrink = vocab_matrix_shrink(v_old, hidden)
    shrink["bf16_gb_decimal"] = shrink["bf16_bytes_removed"] / 1000000000.0
    shrink["bf16_gib"] = shrink["bf16_bytes_removed"] / float(1024**3)
    summary = {
        "source_model_dir": model_dir,
        "out_dir": out_dir,
        "old_ids": [int(item) for item in old_ids],
        "symbols": list(GRID_SYMBOLS),
        "old_to_new": _string_key_map(old_to_new),
        "new_to_old": _string_key_map(new_to_old),
        "shapes_before": {"embed": [v_old, hidden], "lm_head": [v_old, hidden]},
        "shapes_after": {
            "embed": [len(GRID_SYMBOLS), hidden],
            "lm_head": [len(GRID_SYMBOLS), hidden],
        },
        "embed_parameter": embed_name,
        "lm_head_parameter": head_name,
        "embed_dtype_before": embed_dtype,
        "lm_head_dtype_before": head_dtype,
        "saved_dtype": "bfloat16",
        "copy_path": "selected rows copied on CPU in float32, then saved as bfloat16",
        "parameter_counts_removed": shrink["removed_parameters"],
        "bf16_bytes_removed": shrink["bf16_bytes_removed"],
        "shrink": shrink,
        "tie_word_embeddings": False,
        "inputs_shared_storage": False,
        "embedding_padding_idx_before": _jsonable(embed_padding_before),
        "embedding_padding_idx_after": _jsonable(getattr(embed_module, "padding_idx", None)),
        "special_id_choice": {
            "eos_token_id": {"symbol": EOS_SYMBOL, "new_id": _special_ids()["eos_token_id"]},
            "bos_token_id": {"symbol": BOS_SYMBOL, "new_id": _special_ids()["bos_token_id"]},
            "pad_token_id": {"symbol": PAD_SYMBOL, "new_id": _special_ids()["pad_token_id"]},
        },
        "config_updates": changes,
        "saved_embed_keys": embed_keys,
        "saved_lm_head_keys": head_keys,
        "num_hidden_layers_unchanged": _jsonable(layers_before),
        "backbone_parameter_unchanged": {
            "name": backbone_name,
            "shape": list(backbone_shape),
        },
        "backbone_note": (
            "The backbone was not modified. GDN blocks were not swapped for attention. "
            "The input embedding and lm_head were copied as separate row selections and were not tied. "
            "No optimizer state and no old LoRA adapter were written."
        ),
    }
    _write_json(os.path.join(out_dir, "conversion.json"), summary)
    print("CONVERT_SUMMARY=" + json.dumps(summary, ensure_ascii=False))
    print("CONVERT_DONE")
    return summary


class _FakeTokenizer(object):
    def __init__(self, broken=None, prefer_encode=False, unk_is_eot=False):
        self.broken = broken
        self.prefer_encode = prefer_encode
        self.unk_is_eot = unk_is_eot
        self.ids = {}
        for index, symbol in enumerate(GRID_SYMBOLS):
            self.ids[symbol] = 1000 + index
        if unk_is_eot:
            self.unk_token = "<|endoftext|>"
            self.unk_token_id = self.ids["<|endoftext|>"]
        else:
            self.unk_token = "<unk>"
            self.unk_token_id = 99991

    def convert_tokens_to_ids(self, symbol):
        if symbol == self.broken:
            return [41, 42]
        if self.prefer_encode:
            return self.unk_token_id
        return self.ids[symbol]

    def encode(self, symbol, add_special_tokens=False):
        if add_special_tokens:
            raise RuntimeError("fake tokenizer was asked to add special tokens")
        if symbol == self.broken or (self.unk_is_eot and symbol == "<|endoftext|>"):
            return [41, 42]
        return [self.ids[symbol]]


def _expect(condition, message: str) -> None:
    if not condition:
        raise SystemExit(message)


def _self_check() -> None:
    writes_before = _WRITE_CALLS
    _expect(len(GRID_SYMBOLS) == 16, "expected 16 symbols")
    _expect(GRID_SYMBOLS[:10] == [str(digit) for digit in range(10)], "digit order")
    _expect(GRID_SYMBOLS[10] == "\n" and len(GRID_SYMBOLS[10]) == 1, "newline is one character")
    _expect("\\" not in GRID_SYMBOLS[10], "newline must not be a backslash escape")
    _expect(
        GRID_SYMBOLS[11:] == ["user", "assistant", "<|endoftext|>", "<|im_start|>", "<|im_end|>"],
        "control symbol order",
    )

    good = _FakeTokenizer()
    old_ids = resolve_old_ids(good)
    _expect(old_ids == [1000 + index for index in range(16)], "unique fake ids")
    old_to_new, new_to_old = build_maps(old_ids)
    _expect(len(old_to_new) == 16 and len(new_to_old) == 16, "map sizes")
    for new_id, old_id in enumerate(old_ids):
        _expect(new_to_old[new_id] == old_id, "new_to_old")
        _expect(old_to_new[old_id] == new_id, "old_to_new")

    encoded = _FakeTokenizer(prefer_encode=True)
    encoded_ids = resolve_old_ids(encoded)
    _expect(encoded_ids == old_ids, "encode fallback ids")
    eot_unk = _FakeTokenizer(unk_is_eot=True)
    eot_ids = resolve_old_ids(eot_unk)
    _expect(eot_ids[GRID_SYMBOLS.index("<|endoftext|>")] == 1000 + GRID_SYMBOLS.index("<|endoftext|>"), "eot unk id")

    import array

    width = 4
    height = max(old_ids) + 3
    embed = []
    head = []
    for row_index in range(height):
        embed.append(array.array("f", [float(row_index * 10 + col) for col in range(width)]))
        head.append(
            array.array("f", [float(5000 + row_index * 10 + col) for col in range(width)])
        )
    new_embed, new_head = convert_state(embed, head, old_ids)
    _expect(len(new_embed) == 16 and len(new_head) == 16, "selected row count")
    _expect(len(new_embed[0]) == width and len(new_head[0]) == width, "selected width")
    for new_id, old_id in enumerate(old_ids):
        _expect(list(new_embed[new_id]) == list(embed[old_id]), "embed row value")
        _expect(list(new_head[new_id]) == list(head[old_id]), "lm_head row value")
        _expect(new_embed[new_id] is not embed[old_id], "embed row is not a view")
        _expect(new_head[new_id] is not head[old_id], "lm_head row is not a view")
        _expect(new_embed[new_id] is not new_head[new_id], "matrices were reused")
    source_embed = list(embed[old_ids[0]])
    source_head = list(head[old_ids[0]])
    kept_embed = list(new_embed[1])
    new_embed[0][0] = -123.0
    new_head[0][0] = -456.0
    embed[old_ids[1]][1] = float(embed[old_ids[1]][1] + 50.0)
    _expect(list(embed[old_ids[0]]) == source_embed, "source embed changed")
    _expect(list(head[old_ids[0]]) == source_head, "source lm_head changed")
    _expect(new_head[0][0] != -123.0, "writing embed changed lm_head")
    _expect(new_embed[0][0] != -456.0, "writing lm_head changed embed")
    _expect(list(new_embed[1]) == kept_embed, "selected embed is a view of the source")

    try:
        convert_state(embed, embed, old_ids)
    except RuntimeError:
        pass
    else:
        raise SystemExit("shared storage was not refused")

    broken = _FakeTokenizer(broken="7")
    try:
        resolve_old_ids(broken)
    except (SystemExit, ValueError) as exc:
        text = str(exc)
        _expect('"symbol": "7"' in text or "'7'" in text, "failure was not recorded: %s" % text)
        _expect("average" in text, "digit failure did not refuse averaging: %s" % text)
    else:
        raise SystemExit("broken symbol did not raise")

    _expect(encode_grid_string("0123456789") == list(range(10)), "digits stay split")
    _expect(encode_grid_string("\n") == [10], "newline id")
    _expect(encode_grid_string("user") == [11], "user id")
    _expect(encode_grid_string("assistant") == [12], "assistant id")
    _expect(encode_grid_string("<|im_end|>") == [15], "im_end is one token")
    sample = "user\n01\nassistant\n10\n"
    _expect(encode_grid_string(sample) == [11, 10, 0, 1, 10, 12, 10, 1, 0, 10], "demo sample")
    try:
        encode_grid_string("0 1")
    except ValueError:
        pass
    else:
        raise SystemExit("space was accepted")

    namespace = {}
    exec(compile(_wrapper_source(), "grid_tokenizer.py", "exec"), namespace)
    wrapper = namespace["GridTokenizer"]()
    _expect(wrapper.encode(sample, add_special_tokens=False) == encode_grid_string(sample), "wrapper")
    _expect(wrapper.encode("890") == [8, 9, 0], "wrapper digits")
    _expect(wrapper.chat_template is None, "wrapper chat template")
    try:
        wrapper.apply_chat_template([])
    except RuntimeError:
        pass
    else:
        raise SystemExit("wrapper accepted a chat template")

    formatter = build_formatter()
    _expect(formatter["chat_template"] is None and formatter["think_tokens"] is False, "formatter flags")
    _expect(formatter["demo"] == "user\n<grid>\nassistant\n<grid>\n", "demo string")
    _expect(formatter["query"] == "user\n<grid>\nassistant\n", "query string")
    _expect(formatter["supervised_target"] == "<grid>\n<|im_end|>", "target string")
    _expect("\\n" not in formatter["demo"], "demo escaped the newline")
    _expect("G1 space-separated digits" in formatter["distinct_from_g1"], "G1 warning")
    _expect(formatter["training_loop"] is None, "formatter started a training loop")

    official = vocab_matrix_shrink(248320, 5120)
    _expect(official["removed_parameters"] == 2542632960, "official parameter shrink")
    _expect(official["bf16_bytes_removed"] == 5085265920, "official bf16 bytes")
    tiny = vocab_matrix_shrink(20, 4)
    _expect(tiny["removed_parameters"] == 32 and tiny["bf16_bytes_removed"] == 64, "tiny shrink")
    _expect(_WRITE_CALLS == writes_before, "self-check called write_converted_model")
    print("SELF_CHECK_OK")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G3 grid vocab converter")
    parser.add_argument("--self-check", action="store_true")
    args = parser.parse_args(argv)
    if not args.self_check:
        raise SystemExit(
            "only supported invocation: python3 scripts/convert_grid_vocab.py --self-check"
        )
    _self_check()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
