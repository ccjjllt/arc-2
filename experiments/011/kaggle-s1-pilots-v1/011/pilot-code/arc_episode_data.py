"""007 episode pools.

main() builds the G1 official-train split and a 10_000-episode pool. It does
not read ARC-GEN. build_g2() writes the separate G2 pool. No torch or
transformers. prompt_tokens and target_tokens stay null; the Kaggle tokenizer
fills them later.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import io
import json
import random
import shutil
import signal
import subprocess
import sys
import time
import zipfile
from collections import deque
from pathlib import Path


SYSTEM_PROMPT = (
    "Output only the answer grid. Each row is digits 0-9 separated by "
    "single spaces. Put one row on each line. Do not write an explanation."
)

SYSTEM_PROMPT_COMPACT = (
    "Output only the answer grid. Each row is digits 0-9 with no spaces. "
    "Put one row on each line. Do not write an explanation."
)

# Formatter versions (009 plan 3.2). F0 keeps the original "0 0 2" space form;
# F1 is compact: continuous single digits per row plus newline ("00222").
# Full vocab, embeddings/head and chat control tokens are unchanged; the
# tokenizer still sees one token per digit, so no new token ids are needed.
FORMATTER_F0 = "space"
FORMATTER_F1 = "compact"
FORMATTERS = (FORMATTER_F0, FORMATTER_F1)


def normalize_formatter(value) -> str:
    """Map formatter aliases to 'space' or 'compact'. Raises on unknown."""
    text = str(value or "").strip().lower().replace("-", "_")
    if text in ("space", "f0", "f0_space"):
        return FORMATTER_F0
    if text in ("compact", "f1", "f1_compact"):
        return FORMATTER_F1
    raise ValueError("unknown formatter: %r" % (value,))

SOURCE = "official_train"
REVISION = "kaggle:arc-prize-2026-arc-agi-2"
DEFAULT_SEED = 20260930
DEFAULT_HOLDOUT_N = 100
DEFAULT_N_EPISODES = 10000
# D2 (008 plan S4) defaults. Sampling order source->task->base->query->aug.
# source start 40% official / 60% gen is a starting point, not a fixed mixture.
D2_DEFAULT_PER_BASE_K = 2
D2_DEFAULT_OFFICIAL_FRAC = 0.4
D2_TOKEN_BUCKET_NAMES = ("<=1024", "1025-2048", "2049-4096", "4097-8192", ">8192")
ORIGIN_VALUES = ("arc1_train", "arc1_eval", "non_arc1")
CREATED = "2026-09-30"
ARCGEN_COMMIT = "a15cbdb44c776610aeeb9f487a06af875d3d0878"
ARCGEN_SOURCE = "arcgen_v2"
ARCGEN_REVISION = "arc-gen:" + ARCGEN_COMMIT
G2_OFFICIAL_N = 12000
G2_ARCGEN_N = 18000
G2_CREATED = "2026-10-01"
# A generate() call that does not return is a failed draw, not a grid.
ARCGEN_DRAW_TIMEOUT_SECONDS = 10.0

# 006 diagnostic tasks. Out of the train pool and out of the fresh holdout.
DIAG_TASK_IDS = (
    "833966f4",
    "6150a2bd",
    "e9afcf9a",
    "5582e5ca",
    "d037b0a7",
    "d631b094",
    "66e6c45b",
    "00576224",
)

GEOMETRIES = ("ident", "rot90", "rot180", "rot270", "flipud", "fliplr")
N_COLORS = 10
AUGMENTATIONS_PER_BASE = len(GEOMETRIES) * 4  # identity color + 3 perms


def grid_to_text(grid) -> str:
    return "\n".join(" ".join(str(int(c)) for c in row) for row in grid)


def text_to_grid(text) -> list | None:
    """Parse one rectangle of single digits. None unless the stripped text is exact."""
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped == "":
        return None
    rows = []
    width = None
    for line in stripped.split("\n"):
        if line == "" or line != line.strip():
            return None
        cells = line.split(" ")
        parsed = []
        for cell in cells:
            if len(cell) != 1 or cell < "0" or cell > "9":
                return None
            parsed.append(ord(cell) - 48)
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            return None
        rows.append(parsed)
    if width is None or width < 1:
        return None
    return rows


def target_text(grid, formatter=FORMATTER_F0) -> str:
    return serialize_grid(grid, formatter)


def serialize_grid(grid, formatter=FORMATTER_F0) -> str:
    """Serialize one grid under the given formatter. F0 keeps spaces, F1 is compact."""
    formatter = normalize_formatter(formatter)
    if formatter == FORMATTER_F1:
        return "\n".join("".join(str(int(c)) for c in row) for row in grid)
    return grid_to_text(grid)


def parse_grid_text(text, formatter=FORMATTER_F0):
    """Parse one rectangle of single digits. None unless the stripped text is exact.

    F0 is the original space form (delegates to text_to_grid). F1 is compact:
    each row is continuous 0-9 chars and any space rejects the text. Every
    char is one integer cell (multi-digit sequences split per char), matching
    the tokenizer's one-token-per-digit behavior.
    """
    formatter = normalize_formatter(formatter)
    if formatter == FORMATTER_F0:
        return text_to_grid(text)
    if not isinstance(text, str):
        return None
    stripped = text.strip()
    if stripped == "":
        return None
    rows = []
    width = None
    for line in stripped.split("\n"):
        if line == "" or line != line.strip():
            return None
        if " " in line:
            return None
        parsed = []
        for char in line:
            if char < "0" or char > "9":
                return None
            parsed.append(ord(char) - 48)
        if not parsed:
            return None
        if width is None:
            width = len(parsed)
        elif len(parsed) != width:
            return None
        rows.append(parsed)
    if width is None or width < 1:
        return None
    return rows


def apply_geometry(grid, name) -> list:
    """Return a new grid. rot90 is clockwise. flipud reverses rows. fliplr reverses each row."""
    if name == "ident":
        return [row[:] for row in grid]
    if name == "rot180":
        return [row[::-1] for row in reversed(grid)]
    if name == "flipud":
        return [row[:] for row in reversed(grid)]
    if name == "fliplr":
        return [row[::-1] for row in grid]
    if name not in ("rot90", "rot270"):
        raise ValueError("unknown geometry: %s" % name)
    if not grid or not grid[0]:
        return []
    height = len(grid)
    width = len(grid[0])
    if name == "rot90":
        # new[c][height - 1 - r] = old[r][c]
        return [[grid[height - 1 - r][c] for r in range(height)] for c in range(width)]
    # rot270 clockwise: new[width - 1 - c][r] = old[r][c]
    return [[grid[r][width - 1 - c] for r in range(height)] for c in range(width)]


def apply_color(grid, perm) -> list:
    """One lookup table. Do not walk colors and overwrite the grid in place."""
    if len(perm) != N_COLORS or sorted(perm) != list(range(N_COLORS)):
        raise ValueError("color perm must be a permutation of 0..9")
    return [[perm[c] for c in row] for row in grid]


def build_messages(demos, query_input, formatter=FORMATTER_F0) -> list:
    """demos is a list of {input, output} grids. One train pair may pass demos=[]."""
    formatter = normalize_formatter(formatter)
    system = SYSTEM_PROMPT if formatter == FORMATTER_F0 else SYSTEM_PROMPT_COMPACT
    parts = []
    for i, pair in enumerate(demos):
        parts.append("[demo %d] in:\n%s" % (i, serialize_grid(pair["input"], formatter)))
        parts.append("[demo %d] out:\n%s" % (i, serialize_grid(pair["output"], formatter)))
    parts.append("[query] in:\n%s" % serialize_grid(query_input, formatter))
    parts.append("[query] out:")
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": "\n".join(parts)},
    ]


def derive_color_perms(seed: int) -> list:
    """Identity is separate. Three more perms from Random(seed), shared by every task.

    Each draw shuffles a copy of range(10). Identity and duplicates are rejected
    and another shuffle is drawn, so rejected draws still consume RNG state.
    """
    rng = random.Random(seed)
    identity = list(range(N_COLORS))
    perms = []
    draws = 0
    while len(perms) < 3:
        trial = identity[:]
        rng.shuffle(trial)
        draws += 1
        if draws > 10000:
            raise RuntimeError("could not draw 3 non-identity color permutations")
        if trial == identity or trial in perms:
            continue
        perms.append(trial[:])
    return perms


def exclude_arcgen_families(generator_task_ids, holdout_ids, diag_ids):
    """Return generator ids whose family is not in the holdout or DIAG sets.

    Official tasks use family_id == task_id. Each ARC-GEN generator id is
    treated as that family id, so an id listed in ``holdout_ids`` or
    ``diag_ids`` is removed. Input order is kept, including duplicates of
    ids that remain. This function does not read ARC-GEN and does not build
    the 30k pool; G2 calls it later.
    """
    banned = set(holdout_ids) | set(diag_ids)
    return [gid for gid in generator_task_ids if gid not in banned]


def classify_origin(task_id, arc1_train_ids=None, arc1_eval_ids=None):
    """Historical source label only; not a contamination claim.

    Returns one of arc1_train / arc1_eval / non_arc1 when id sets are
    given, else 'unknown' (caller did not supply ARC1 lists). V2 generator
    version is not consulted here; V2 != new-task is stated in the coverage
    report, not in this label.
    """
    if arc1_train_ids is not None and task_id in arc1_train_ids:
        return "arc1_train"
    if arc1_eval_ids is not None and task_id in arc1_eval_ids:
        return "arc1_eval"
    if arc1_train_ids is None and arc1_eval_ids is None:
        return "unknown"
    return "non_arc1"


def suspect_same_as_demo(target, demos) -> bool:
    """True when the target grid equals any demo output grid.

    Flag only. Equality alone is not a leak proof: different inputs may
    legally share an answer. Callers decide whether to drop (old opt-in
    flag) or keep with this mark (D2 default).
    """
    return any(demo.get("output") == target for demo in demos)


def make_base_instance_id(task_id, query_kind, query_index) -> str:
    return "%s|%s|%d" % (task_id, query_kind, query_index)


def task_order_key(task_id: str, seed: int) -> str:
    payload = ("%d:%s" % (seed, task_id)).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def ordering_rule(seed: int, holdout_n: int) -> str:
    return (
        "Eligible task ids are present in both challenges and solutions and are not in diag. "
        "Order eligible by sha256(utf-8 of '%d:'+task_id) hex digest ascending, ties broken by task_id. "
        "holdout is the first %d of that order, then sorted lexicographically in this file. "
        "train is the remaining eligible ids, sorted lexicographically. "
        "diag is the fixed 006 exclusion list in its given order, excluded from the train pool and the holdout. "
        "family_id = task_id for official tasks. "
        "Round-robin sampling walks lexicographically sorted train ids, one episode per task per turn. "
        "Within a task, test bases stay in test-index order and loo bases stay in train-index order. "
        "The primary pair is the first test base, if any, then the first loo base, if any. "
        "Front augmentations are geometry outer (ident, rot90, rot180, rot270, flipud, fliplr) "
        "and color inner (ident, then perm1). "
        "For each front augmentation, yield each primary base once. "
        "After that prefix, yield every remaining geometry, color (ident, perm1, perm2, perm3), "
        "and interleaved base (test0, loo0, test1, loo1, ...), skipping episode ids already yielded. "
        "A 10k cap therefore mixes ident and perm1 on test0 and loo0 before later geometries, perm2, or perm3."
        % (seed, holdout_n)
    )


def make_split(challenge_ids, solution_ids, seed: int, holdout_n: int, diag_ids=DIAG_TASK_IDS):
    diag = [tid for tid in diag_ids]
    diag_set = set(diag)
    both = set(challenge_ids) & set(solution_ids)
    eligible = [tid for tid in both if tid not in diag_set]
    eligible.sort(key=lambda tid: (task_order_key(tid, seed), tid))
    if holdout_n < 0:
        raise ValueError("holdout_n must be >= 0")
    picked = eligible[:holdout_n]
    picked_set = set(picked)
    holdout = sorted(picked)
    train = sorted(tid for tid in eligible if tid not in picked_set)
    if not set(diag).isdisjoint(holdout) or not set(diag).isdisjoint(train):
        raise RuntimeError("diag leaked into holdout or train")
    if not set(holdout).isdisjoint(train):
        raise RuntimeError("holdout and train overlap")
    return diag, holdout, train


def grid_problem(grid) -> str | None:
    """None when grid is a rectangular 1..30 grid of ints 0-9. Does not mutate."""
    if not isinstance(grid, list) or len(grid) < 1:
        return "not_rectangular"
    if len(grid) > 30:
        return "height_out_of_range"
    width = None
    for row in grid:
        if not isinstance(row, list):
            return "not_rectangular"
        if len(row) < 1:
            return "width_out_of_range"
        if len(row) > 30:
            return "width_out_of_range"
        if width is None:
            width = len(row)
        elif len(row) != width:
            return "not_rectangular"
        for cell in row:
            if isinstance(cell, bool) or not isinstance(cell, int) or cell < 0 or cell > 9:
                return "value_out_of_range"
    return None


def _note_skip(skipped, reason, task_id, query_kind, query_index):
    skipped["n"] += 1
    by_kind = skipped["reasons"].setdefault(reason, {})
    by_kind[query_kind] = by_kind.get(query_kind, 0) + 1
    examples = skipped["examples"].setdefault(reason, [])
    if len(examples) < 8:
        examples.append(
            {
                "task_id": task_id,
                "query_kind": query_kind,
                "query_index": query_index,
            }
        )


def _pair_problem(pair) -> str | None:
    if not isinstance(pair, dict):
        return "pair_not_object"
    return grid_problem(pair.get("input")) or grid_problem(pair.get("output"))


def collect_bases(
    task_id,
    task,
    solution,
    skipped,
    drop_target_equals_demo=False,
    min_loo_demos=2,
    arc1_train_ids=None,
    arc1_eval_ids=None,
):
    """Valid base episodes for one train task. Grids are the loaded objects, not copies.

    D2 (008 plan S4): enumerates every test query plus every legal LOO
    candidate. A LOO candidate is legal only when the remaining demo count
    >= min_loo_demos (default 2). suspect_same_as_demo is flag-only; the
    old 'drop when target equals a demo output' rule runs only when
    drop_target_equals_demo=True (007 reproduction; D2 default False).
    target_isolated=True records that the target pair is excluded from demos
    by construction. origin/base_instance_id are stamped per base.
    """
    bases = []
    origin = classify_origin(task_id, arc1_train_ids, arc1_eval_ids)
    if not isinstance(task, dict):
        _note_skip(skipped, "task_not_object", task_id, "test", -1)
        return bases
    train = task.get("train", [])
    test = task.get("test", [])
    if not isinstance(train, list):
        _note_skip(skipped, "train_not_list", task_id, "loo", -1)
        train = []
    if not isinstance(test, list):
        _note_skip(skipped, "test_not_list", task_id, "test", -1)
        test = []
    sols = solution if isinstance(solution, list) else None

    for index, item in enumerate(test):
        if not isinstance(item, dict):
            _note_skip(skipped, "query_not_object", task_id, "test", index)
            continue
        if sols is None or index >= len(sols):
            _note_skip(skipped, "solution_missing", task_id, "test", index)
            continue
        reason = grid_problem(item.get("input")) or grid_problem(sols[index])
        if reason:
            _note_skip(skipped, reason, task_id, "test", index)
            continue
        demos = []
        demo_reason = None
        for pair in train:
            demo_reason = _pair_problem(pair)
            if demo_reason:
                break
            demos.append(pair)
        if demo_reason:
            _note_skip(skipped, demo_reason, task_id, "test", index)
            continue
        # Keep the official test item even when its gold grid equals a demo output.
        # That answer is not copied into the prompt; dropping it would discard a real query.
        target = sols[index]
        suspect = suspect_same_as_demo(target, demos)
        if drop_target_equals_demo and suspect:
            _note_skip(skipped, "target_equals_demo_output", task_id, "test", index)
            continue
        bases.append(
            {
                "query_kind": "test",
                "query_index": index,
                "demos": demos,
                "query_input": item["input"],
                "target": target,
                "origin": origin,
                "base_instance_id": make_base_instance_id(task_id, "test", index),
                "suspect_same_as_demo": bool(suspect),
                "target_isolated": True,
            }
        )

    for index, pair in enumerate(train):
        reason = _pair_problem(pair)
        if reason:
            _note_skip(skipped, reason, task_id, "loo", index)
            continue
        demos = []
        demo_reason = None
        for other_index, other in enumerate(train):
            if other_index == index:
                continue
            demo_reason = _pair_problem(other)
            if demo_reason:
                break
            demos.append(other)
        if demo_reason:
            _note_skip(skipped, demo_reason, task_id, "loo", index)
            continue
        if len(demos) < min_loo_demos:
            _note_skip(skipped, "loo_demos_lt2", task_id, "loo", index)
            continue
        target = pair["output"]
        suspect = suspect_same_as_demo(target, demos)
        if drop_target_equals_demo and suspect:
            _note_skip(skipped, "target_equals_demo_output", task_id, "loo", index)
            continue
        bases.append(
            {
                "query_kind": "loo",
                "query_index": index,
                "demos": demos,
                "query_input": pair["input"],
                "target": target,
                "origin": origin,
                "base_instance_id": make_base_instance_id(task_id, "loo", index),
                "suspect_same_as_demo": bool(suspect),
                "target_isolated": True,
            }
        )
    return bases


def _color_tables(perms):
    tables = [("ident", list(range(N_COLORS)))]
    for index, perm in enumerate(perms, start=1):
        tables.append(("perm%d" % index, list(perm)))
    return tables


def _make_episode(
    task_id, base, geometry, color_name, perm, seed, source=SOURCE, revision=REVISION,
    formatter=FORMATTER_F0, drop_suspect=False,
):
    formatter = normalize_formatter(formatter)
    demos = []
    for pair in base["demos"]:
        demos.append(
            {
                "input": apply_color(apply_geometry(pair["input"], geometry), perm),
                "output": apply_color(apply_geometry(pair["output"], geometry), perm),
            }
        )
    query = apply_color(apply_geometry(base["query_input"], geometry), perm)
    target = apply_color(apply_geometry(base["target"], geometry), perm)
    target_grid_text = target_text(target, formatter)
    if parse_grid_text(target_grid_text, formatter) != target:
        raise RuntimeError("target text did not round-trip: %s" % task_id)
    messages = build_messages(demos, query, formatter)
    # Loo targets must not be shown as a remaining demo output. Test gold may match
    # a demo grid; that official item is kept and is not a prompt-construction leak.
    # Unified suspect policy (009 plan S4 handoff): flag-only by default, the row is
    # kept with suspect_same_as_demo=True. Only an explicit drop_suspect=True
    # discards it (007 G1/G2 reproduction, which also drops upstream in
    # collect_bases/_arcgen_instance, so the raise below is unreachable there).
    suspect = bool(base.get("suspect_same_as_demo", False))
    if base["query_kind"] == "loo":
        for block in _demo_output_texts(messages[1]["content"]):
            if block == target_grid_text:
                suspect = True
                if drop_suspect:
                    raise RuntimeError("loo target leaked into a demo output: %s" % task_id)
                break
    return {
        "episode_id": "%s|%s|%d|%s|%s"
        % (task_id, base["query_kind"], base["query_index"], geometry, color_name),
        "task_id": task_id,
        # family_id == task_id is task-ID isolation only, not semantic
        # rule dedup; base-model pretraining exposure is unknown.
        "family_id": task_id,
        "source": source,
        "revision": revision,
        "seed": seed,
        "formatter": formatter,
        "augmentation": "%s|%s" % (geometry, color_name),
        "query_kind": base["query_kind"],
        "query_index": base["query_index"],
        "origin": base.get("origin", "unknown"),
        "base_instance_id": base.get(
            "base_instance_id",
            make_base_instance_id(task_id, base["query_kind"], base["query_index"]),
        ),
        "suspect_same_as_demo": suspect,
        "target_isolated": bool(base.get("target_isolated", True)),
        "split": "train",
        "prompt_tokens": None,
        "target_tokens": None,
        "color_perm": list(perm),
        "geometry": geometry,
        "messages": messages,
        "target_grid_text": target_grid_text,
    }


def _interleaved_bases(bases):
    test_bases = [base for base in bases if base["query_kind"] == "test"]
    loo_bases = [base for base in bases if base["query_kind"] == "loo"]
    interleaved = []
    for index in range(max(len(test_bases), len(loo_bases))):
        if index < len(test_bases):
            interleaved.append(test_bases[index])
        if index < len(loo_bases):
            interleaved.append(loo_bases[index])
    primary = []
    if test_bases:
        primary.append(test_bases[0])
    if loo_bases:
        primary.append(loo_bases[0])
    return primary, interleaved


def iter_augmented(task_id, bases, color_tables, seed, formatter=FORMATTER_F0, drop_suspect=False):
    # Front: each geometry with ident then perm1, on the first test and first loo only.
    # A 10k round-robin keeps ~12 episodes per task, so this prefix is what the file contains.
    primary, interleaved = _interleaved_bases(bases)
    perm_by_name = {name: perm for name, perm in color_tables}
    seen = set()

    def emit(base, geometry, color_name):
        episode = _make_episode(task_id, base, geometry, color_name, perm_by_name[color_name], seed,
                                formatter=formatter, drop_suspect=drop_suspect)
        if episode["episode_id"] in seen:
            return None
        seen.add(episode["episode_id"])
        return episode

    front_colors = [name for name in ("ident", "perm1") if name in perm_by_name]
    for geometry in GEOMETRIES:
        for color_name in front_colors:
            for base in primary:
                episode = emit(base, geometry, color_name)
                if episode is not None:
                    yield episode
    for geometry in GEOMETRIES:
        for color_name, _perm in color_tables:
            for base in interleaved:
                episode = emit(base, geometry, color_name)
                if episode is not None:
                    yield episode


def _demo_output_texts(user_text):
    lines = user_text.split("\n")
    index = 0
    found = []
    while index < len(lines):
        line = lines[index]
        if line.startswith("[demo ") and line.endswith("] out:"):
            label = line[len("[demo ") : -len("] out:")]
            if label.isdigit():
                index += 1
                block = []
                while index < len(lines) and not (
                    lines[index].startswith("[demo ") or lines[index].startswith("[query] ")
                ):
                    block.append(lines[index])
                    index += 1
                found.append("\n".join(block))
                continue
        index += 1
    return found


def iter_round_robin(task_bases, color_tables, seed, limit, formatter=FORMATTER_F0, drop_suspect=False):
    pending = deque()
    for task_id, bases in task_bases:
        if bases:
            pending.append((task_id, iter_augmented(task_id, bases, color_tables, seed,
                                                    formatter=formatter, drop_suspect=drop_suspect)))
    seen = set()
    produced = 0
    while pending and produced < limit:
        task_id, stream = pending.popleft()
        try:
            episode = next(stream)
        except StopIteration:
            continue
        if episode["episode_id"] in seen:
            raise RuntimeError("duplicate episode_id %s" % episode["episode_id"])
        seen.add(episode["episode_id"])
        produced += 1
        yield episode
        pending.append((task_id, stream))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _extract_json(zip_path: Path, dest_json: Path) -> Path:
    with zipfile.ZipFile(zip_path) as archive:
        members = [
            name
            for name in archive.namelist()
            if name.lower().endswith(".json") and not name.endswith("/")
        ]
        if not members:
            raise FileNotFoundError("no json member in %s" % zip_path)
        chosen = None
        for name in members:
            if Path(name).name == dest_json.name:
                chosen = name
                break
        if chosen is None:
            if len(members) != 1:
                raise FileNotFoundError("multiple json members in %s: %s" % (zip_path, members))
            chosen = members[0]
        if dest_json.resolve() == zip_path.resolve():
            dest_json = zip_path.with_suffix("")
            if dest_json.suffix != ".json":
                dest_json = dest_json.parent / (dest_json.name + ".json")
        dest_json.parent.mkdir(parents=True, exist_ok=True)
        # Member names are untrusted. Write only the destination basename.
        with archive.open(chosen) as src, dest_json.open("wb") as dst:
            shutil.copyfileobj(src, dst)
    print("unzipped %s from %s -> %s" % (Path(chosen).name, zip_path, dest_json))
    return dest_json


def ensure_json(path: Path) -> Path:
    """Use a JSON file, or unzip a sibling/same-path zip into that JSON path first."""
    path = Path(path)
    if path.is_file() and not zipfile.is_zipfile(path):
        return path
    zip_path = None
    dest = path
    if path.is_file() and zipfile.is_zipfile(path):
        zip_path = path
    else:
        candidates = [Path(str(path) + ".zip"), path.with_suffix(path.suffix + ".zip")]
        for candidate in candidates:
            if candidate.is_file():
                zip_path = candidate
                break
    if zip_path is None:
        raise FileNotFoundError(path)
    return _extract_json(zip_path, dest)


def load_task_map(path: Path):
    resolved = ensure_json(path)
    with resolved.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise ValueError("%s is not a JSON object" % resolved)
    return data, resolved


def _file_record(path: Path) -> dict:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256_file(path),
    }


def _bump(counter, key):
    counter[key] = counter.get(key, 0) + 1


def build(challenges_path, solutions_path, out_dir, n_episodes, seed, holdout_n):
    challenges, challenges_file = load_task_map(Path(challenges_path))
    solutions, solutions_file = load_task_map(Path(solutions_path))
    diag, holdout, train = make_split(challenges.keys(), solutions.keys(), seed, holdout_n)
    holdout_set = set(holdout)
    diag_set = set(diag)
    perms = derive_color_perms(seed)
    color_tables = _color_tables(perms)

    skipped = {"n": 0, "reasons": {}, "examples": {}}
    valid_base = {"test": 0, "loo": 0}
    task_bases = []
    for task_id in train:
        # 007 G1 semantics: keep the old deletion rule for reproducibility.
        # D2 callers use the default drop_target_equals_demo=False (flag only).
        bases = collect_bases(
            task_id, challenges[task_id], solutions.get(task_id), skipped,
            drop_target_equals_demo=True,
        )
        for base in bases:
            valid_base[base["query_kind"]] += 1
        task_bases.append((task_id, bases))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_path = out_dir / "split.json"
    episodes_path = out_dir / "episodes.jsonl"
    manifest_path = out_dir / "manifest.json"

    split_doc = {
        "seed": seed,
        "ordering_rule": ordering_rule(seed, holdout_n),
        "diag": diag,
        "holdout": holdout,
        "train": train,
    }
    with split_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(split_doc, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    geometry_counts = {}
    color_counts = {}
    query_kind_counts = {}
    used_tasks = set()
    used_families = set()
    written = 0
    with episodes_path.open("w", encoding="utf-8", newline="\n") as handle:
        # 007 G1 reproduction: explicit drop keeps the old deletion rule end to end.
        for episode in iter_round_robin(task_bases, color_tables, seed, n_episodes, drop_suspect=True):
            if episode["task_id"] in holdout_set or episode["task_id"] in diag_set:
                raise RuntimeError("refusing to write held-out task %s" % episode["task_id"])
            handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
            written += 1
            used_tasks.add(episode["task_id"])
            used_families.add(episode["family_id"])
            _bump(geometry_counts, episode["geometry"])
            color_name = episode["augmentation"].split("|", 1)[1]
            _bump(color_counts, color_name)
            _bump(query_kind_counts, episode["query_kind"])

    n_valid = valid_base["test"] + valid_base["loo"]
    manifest = {
        "created": CREATED,
        "seed": seed,
        "n_episodes": written,
        "n_episodes_requested": n_episodes,
        "n_train_tasks_used": len(used_tasks),
        "n_train_tasks_available": len(train),
        "n_train_tasks_with_base": sum(1 for _, bases in task_bases if bases),
        "n_holdout": len(holdout),
        "n_diag": len(diag),
        "n_skipped": skipped["n"],
        "n_skipped_note": (
            "Train-split base episodes (one test index or one loo index) not expanded. "
            "Each skipped base drops all %d geometry x color variants. "
            "A base is skipped when a grid is missing, non-rectangular, or outside a 1..30 "
            "grid of ints 0-9, or when the test solution is missing. "
            "A loo base is also skipped when a remaining demonstration has the same output "
            "grid as the held-out target. Test episodes are not skipped for that reason: "
            "the test answer is not placed in the prompt, and a matching grid is a real official item. "
            "Cells are never dropped or truncated to fit a length."
            % AUGMENTATIONS_PER_BASE
        ),
        "skip_reasons": skipped["reasons"],
        "skip_examples": skipped["examples"],
        "geometry_counts": geometry_counts,
        "color_counts": color_counts,
        "query_kind_counts": query_kind_counts,
        "unique_family_count": len(used_families),
        "source": SOURCE,
        "revision": REVISION,
        "color_perms": {
            "ident": list(range(N_COLORS)),
            "perm1": list(perms[0]),
            "perm2": list(perms[1]),
            "perm3": list(perms[2]),
        },
        "prompt_tokens_note": (
            "prompt_tokens and target_tokens are null because tokenization happens "
            "on the Kaggle tokenizer."
        ),
        "episodes_sha256": _sha256_file(episodes_path),
        "arcgen_pool": "not_generated",
        "n_challenges": len(challenges),
        "n_solutions": len(solutions),
        "n_tasks_with_solutions": len(solutions),
        "n_eligible": len(train) + len(holdout),
        "n_valid_base": valid_base,
        "n_augmented_pool": n_valid * AUGMENTATIONS_PER_BASE,
        "inputs": {
            "challenges": _file_record(challenges_file),
            "solutions": _file_record(solutions_file),
        },
        "python": sys.version.split()[0],
        "sampling": ordering_rule(seed, holdout_n),
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")

    print("challenges=%d solutions=%d" % (len(challenges), len(solutions)))
    print("diag=%d holdout=%d train=%d" % (len(diag), len(holdout), len(train)))
    print(
        "valid_base test=%d loo=%d skipped_base=%d augmented_pool=%d"
        % (valid_base["test"], valid_base["loo"], skipped["n"], manifest["n_augmented_pool"])
    )
    print("skip_reasons=%s" % json.dumps(skipped["reasons"], sort_keys=True))
    print("episodes=%d tasks_used=%d families=%d" % (written, len(used_tasks), len(used_families)))
    print("geometry_counts=%s" % json.dumps(geometry_counts, sort_keys=True))
    print("color_counts=%s" % json.dumps(color_counts, sort_keys=True))
    print("query_kind_counts=%s" % json.dumps(query_kind_counts, sort_keys=True))
    print("color_perms=%s" % json.dumps(manifest["color_perms"]))
    print("episodes_sha256=%s" % manifest["episodes_sha256"])
    print("inputs challenges_bytes=%d solutions_bytes=%d" % (
        manifest["inputs"]["challenges"]["bytes"],
        manifest["inputs"]["solutions"]["bytes"],
    ))
    print("wrote %s" % split_path)
    print("wrote %s" % episodes_path)
    print("wrote %s" % manifest_path)
    return manifest


def _geometry_color_cycle(color_tables):
    """Six geometries under one color table, then the next table. 24 pairs, then repeat."""
    cycle = []
    for color_name, perm in color_tables:
        for geometry in GEOMETRIES:
            cycle.append((geometry, color_name, list(perm)))
    return cycle


def _target_in_demos(episode) -> bool:
    target = episode["target_grid_text"]
    for block in _demo_output_texts(episode["messages"][1]["content"]):
        if block == target:
            return True
    return False


def arcgen_v2_ids(task_list_path):
    """Generator ids imported in task_list.py after the '# V2 Tasks' comment."""
    lines = Path(task_list_path).read_text(encoding="utf-8").splitlines()
    start = None
    for index, line in enumerate(lines):
        if line.strip() == "# V2 Tasks":
            start = index + 1
            break
    if start is None:
        raise RuntimeError("V2 Tasks comment not found in %s" % task_list_path)
    ids = []
    for line in lines[start:]:
        stripped = line.strip()
        if stripped.startswith("from tasks import task_"):
            ids.append(stripped.split("task_", 1)[1])
        elif stripped.startswith("def "):
            break
    if not ids or ids[0] != "00576224":
        raise RuntimeError("V2 generator ids did not start at 00576224")
    return ids


class _GenerateTimeout(Exception):
    pass


def _on_generate_timeout(signum, frame):
    raise _GenerateTimeout()


_TIMEOUT_HANDLER_READY = False


def _ensure_generate_timeout():
    global _TIMEOUT_HANDLER_READY
    if not _TIMEOUT_HANDLER_READY:
        signal.signal(signal.SIGALRM, _on_generate_timeout)
        _TIMEOUT_HANDLER_READY = True


def _arcgen_seed_int(task_id, instance, slot, attempt) -> int:
    """sha256 of the material, first 8 bytes, little-endian.

    Attempt 0 uses 20260930:task:instance:slot. A failed draw appends :attempt.
    """
    material = "20260930:%s:%s:%s" % (task_id, instance, slot)
    if attempt:
        material = "%s:%d" % (material, attempt)
    digest = hashlib.sha256(material.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little")


def _draw_arcgen_pair(generate, task_id, instance, slot, attempt_reasons, stdout_families, throw_examples):
    last_reason = "not_drawn"
    _ensure_generate_timeout()
    for attempt in range(20):
        random.seed(_arcgen_seed_int(task_id, instance, slot, attempt))
        buffer = io.StringIO()
        try:
            signal.setitimer(signal.ITIMER_REAL, ARCGEN_DRAW_TIMEOUT_SECONDS)
            try:
                with contextlib.redirect_stdout(buffer):
                    pair = generate()
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0.0)
        except _GenerateTimeout:
            last_reason = "generate_timeout"
            _bump(attempt_reasons, last_reason)
            if buffer.getvalue():
                stdout_families.add(task_id)
            if attempt == 0:
                print(
                    "generate_timeout task=%s instance=%s slot=%s" % (task_id, instance, slot),
                    flush=True,
                )
            continue
        except Exception as exc:
            last_reason = "generate_throw:%s" % type(exc).__name__
            _bump(attempt_reasons, last_reason)
            if buffer.getvalue():
                stdout_families.add(task_id)
            if len(throw_examples) < 8:
                message = "%s: %s" % (type(exc).__name__, exc)
                throw_examples.append(
                    {
                        "task_id": task_id,
                        "instance": instance,
                        "slot": slot,
                        "attempt": attempt,
                        "error": message[:200],
                    }
                )
            continue
        if buffer.getvalue():
            stdout_families.add(task_id)
        if not isinstance(pair, dict):
            last_reason = "pair_not_object"
            _bump(attempt_reasons, last_reason)
            continue
        reason = grid_problem(pair.get("input")) or grid_problem(pair.get("output"))
        if reason:
            last_reason = reason
            _bump(attempt_reasons, last_reason)
            continue
        return (
            {
                "input": copy.deepcopy(pair["input"]),
                "output": copy.deepcopy(pair["output"]),
            },
            None,
        )
    return None, last_reason


def _arcgen_instance(
    generate,
    task_id,
    instance,
    skipped,
    attempt_reasons,
    stdout_families,
    throw_examples,
    drop_target_equals_demo=False,
    arc1_train_ids=None,
    arc1_eval_ids=None,
    gen_validator=None,
):
    """Four seeded draws. Slots 0-2 are demos and slot 3 is the query.

    D2 default keeps suspect bases with suspect_same_as_demo=True; pass
    drop_target_equals_demo=True only to reproduce the 007 deletion rule.
    gen_validator is an ARC-GEN-cooperation anchor: optional callable
    pairs4 -> reason|None checking cross-slot shared vars / validate()
    semantics. When None only grid-shape checks run; no grid cells are
    hardcoded here (see code_gaps_followup.json).
    """
    pairs = []
    for slot in range(4):
        pair, reason = _draw_arcgen_pair(
            generate, task_id, instance, slot, attempt_reasons, stdout_families, throw_examples
        )
        if pair is None:
            _note_skip(skipped, reason or "slot_failed", task_id, "gen", instance)
            return None
        pairs.append(pair)
    if gen_validator is not None:
        try:
            reason = gen_validator(pairs)
        except Exception as exc:
            reason = "gen_validator_throw:%s" % type(exc).__name__
        if reason:
            _note_skip(skipped, reason, task_id, "gen", instance)
            return None
    target = pairs[3]["output"]
    suspect = any(pair["output"] == target for pair in pairs[:3])
    if drop_target_equals_demo and suspect:
        _note_skip(skipped, "target_equals_demo_output", task_id, "gen", instance)
        return None
    return {
        "task_id": task_id,
        "instance": instance,
        "demos": pairs[:3],
        "query_input": pairs[3]["input"],
        "target": target,
        "origin": classify_origin(task_id, arc1_train_ids, arc1_eval_ids),
        "base_instance_id": make_base_instance_id(task_id, "gen", instance),
        "suspect_same_as_demo": bool(suspect),
        "target_isolated": True,
    }


def _collect_arcgen_bases(
    allowed,
    generators,
    need,
    skipped,
    attempt_reasons,
    stdout_families,
    throw_examples,
    drop_target_equals_demo=False,
    arc1_train_ids=None,
    arc1_eval_ids=None,
    gen_validator=None,
):
    """One kept base per family before any family receives a second."""
    state = {}
    usable = []
    for task_id in allowed:
        info = generators.get(task_id)
        generate = None
        if isinstance(info, (list, tuple)) and info:
            generate = info[0]
        if not callable(generate):
            _note_skip(skipped, "generator_missing", task_id, "gen", -1)
            continue
        state[task_id] = {
            "alive": True,
            "n_kept": 0,
            "next_instance": 0,
            "fails": 0,
            "generate": generate,
        }
        usable.append(task_id)
    kept = []
    idle_passes = 0
    while len(kept) < need and any(state[task_id]["alive"] for task_id in usable):
        min_kept = min(state[task_id]["n_kept"] for task_id in usable if state[task_id]["alive"])
        progress = False
        for task_id in usable:
            if len(kept) >= need:
                break
            item = state[task_id]
            if not item["alive"] or item["n_kept"] != min_kept:
                continue
            instance = item["next_instance"]
            item["next_instance"] += 1
            if instance >= 32:
                item["alive"] = False
                _note_skip(skipped, "instance_cap", task_id, "gen", instance)
                continue
            base = _arcgen_instance(
                item["generate"],
                task_id,
                instance,
                skipped,
                attempt_reasons,
                stdout_families,
                throw_examples,
                drop_target_equals_demo=drop_target_equals_demo,
                arc1_train_ids=arc1_train_ids,
                arc1_eval_ids=arc1_eval_ids,
                gen_validator=gen_validator,
            )
            if base is None:
                item["fails"] += 1
                if item["fails"] >= 8:
                    item["alive"] = False
                continue
            item["fails"] = 0
            item["n_kept"] += 1
            kept.append(base)
            progress = True
            if len(kept) % 50 == 0:
                print(
                    "arcgen_bases_kept=%d last=%s instance=%d" % (len(kept), task_id, instance),
                    flush=True,
                )
        if progress:
            idle_passes = 0
        else:
            idle_passes += 1
            if idle_passes > 32:
                break
    alive = sum(1 for task_id in usable if state[task_id]["alive"])
    return kept, alive


def _new_source_bucket(color_tables):
    return {
        "n": 0,
        "geometry": {name: 0 for name in GEOMETRIES},
        "color": {name: 0 for name, _perm in color_tables},
        "query_kind": {},
        "families": set(),
    }


def _tally_episode(episode, geometry_counts, color_counts, query_kind_counts, bucket):
    color_name = episode["augmentation"].split("|", 1)[1]
    _bump(geometry_counts, episode["geometry"])
    _bump(color_counts, color_name)
    _bump(query_kind_counts, episode["query_kind"])
    _bump(bucket["geometry"], episode["geometry"])
    _bump(bucket["color"], color_name)
    _bump(bucket["query_kind"], episode["query_kind"])
    bucket["n"] += 1
    bucket["families"].add(episode["family_id"])


def _write_episode(handle, episode):
    handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
    handle.write("\n")


def _iter_official_g2(task_bases, cycle, seed, limit, skipped, formatter=FORMATTER_F0, drop_suspect=False):
    pending = deque()
    for task_id, bases in task_bases:
        if bases:
            pending.append((task_id, bases, 0))
    produced = 0
    cycle_len = len(cycle)
    while pending and produced < limit:
        task_id, bases, step = pending.popleft()
        base = bases[step // cycle_len]
        geometry, color_name, perm = cycle[step % cycle_len]
        try:
            episode = _make_episode(task_id, base, geometry, color_name, perm, seed,
                                    formatter=formatter, drop_suspect=drop_suspect)
        except RuntimeError as exc:
            _note_skip(skipped, "episode_build_failed", task_id, base["query_kind"], base["query_index"])
            if len(skipped["examples"].setdefault("episode_build_failed", [])) <= 8:
                examples = skipped["examples"]["episode_build_failed"]
                if examples and examples[-1].get("error") is None and len(examples) <= 8:
                    examples[-1]["error"] = str(exc)[:200]
            step += 1
            if step < len(bases) * cycle_len:
                pending.append((task_id, bases, step))
            continue
        produced += 1
        yield episode
        step += 1
        if step < len(bases) * cycle_len:
            pending.append((task_id, bases, step))


def _git_head(root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def build_g2(
    challenges_path,
    solutions_path,
    split_path,
    out_dir,
    arcgen_root,
    official_n=G2_OFFICIAL_N,
    arcgen_n=G2_ARCGEN_N,
    seed=DEFAULT_SEED,
):
    """Write the G2 pool. Does not modify the G1 split, episodes, or adapters.

    Official rows are a round-robin over split train families. Inside a family
    the 6 geometries cycle under each of the 4 color tables before the next
    base repeats that cycle. ARC-GEN keeps one base per allowed V2 family
    before any family gets a second, then expands each kept base across that
    same cycle. A short side is not filled from the other side.
    """
    started = time.time()
    if official_n < 0 or arcgen_n < 0:
        raise ValueError("episode caps must be >= 0")
    arcgen_root = Path(arcgen_root)
    commit = _git_head(arcgen_root)
    if commit != ARCGEN_COMMIT:
        raise RuntimeError("ARC-GEN commit %s != %s" % (commit, ARCGEN_COMMIT))

    split_path = Path(split_path)
    with split_path.open("r", encoding="utf-8") as handle:
        split_doc = json.load(handle)
    diag = list(split_doc["diag"])
    holdout = list(split_doc["holdout"])
    train = list(split_doc["train"])
    banned = set(diag) | set(holdout)
    if not banned.isdisjoint(train):
        raise RuntimeError("split train overlaps diag or holdout")

    challenges, challenges_file = load_task_map(Path(challenges_path))
    solutions, solutions_file = load_task_map(Path(solutions_path))
    perms = derive_color_perms(seed)
    color_tables = _color_tables(perms)
    cycle = _geometry_color_cycle(color_tables)

    skipped_official = {"n": 0, "reasons": {}, "examples": {}}
    task_bases = []
    for task_id in train:
        # 007 G2 semantics: old deletion rule on (reproduces filed 30k pool).
        bases = collect_bases(
            task_id,
            challenges.get(task_id),
            solutions.get(task_id),
            skipped_official,
            drop_target_equals_demo=True,
        )
        task_bases.append((task_id, bases))
    families_with_base = sum(1 for _task_id, bases in task_bases if bases)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    episodes_path = out_dir / "episodes.jsonl"
    manifest_path = out_dir / "manifest.json"

    geometry_counts = {name: 0 for name in GEOMETRIES}
    color_counts = {name: 0 for name, _perm in color_tables}
    query_kind_counts = {}
    by_source = {
        SOURCE: _new_source_bucket(color_tables),
        ARCGEN_SOURCE: _new_source_bucket(color_tables),
    }
    seen = set()
    leak_count = 0
    duplicate_skipped = 0
    arcgen_row_skips = {"n": 0, "reasons": {}, "examples": {}}
    written = {"official": 0, "arcgen": 0}

    def accept(episode, bucket_name):
        nonlocal leak_count, duplicate_skipped
        if episode["task_id"] in banned or episode["family_id"] in banned:
            leak_count += 1
            return False
        if episode["episode_id"] in seen:
            duplicate_skipped += 1
            return False
        if episode["prompt_tokens"] is not None or episode["target_tokens"] is not None:
            raise RuntimeError("token fields must stay null")
        seen.add(episode["episode_id"])
        _write_episode(handle, episode)
        _tally_episode(
            episode,
            geometry_counts,
            color_counts,
            query_kind_counts,
            by_source[bucket_name],
        )
        return True

    with episodes_path.open("w", encoding="utf-8", newline="\n") as handle:
        # 007 G2 reproduction: explicit drop keeps the filed 30k-pool deletion rule.
        for episode in _iter_official_g2(task_bases, cycle, seed, official_n, skipped_official, drop_suspect=True):
            if episode["source"] != SOURCE or episode["revision"] != REVISION:
                raise RuntimeError("official row lost its source or revision")
            if accept(episode, SOURCE):
                written["official"] += 1
                if written["official"] % 2000 == 0:
                    print("official_rows=%d" % written["official"], flush=True)
        print(
            "official_done rows=%d families_with_base=%d elapsed=%.1f"
            % (written["official"], families_with_base, time.time() - started),
            flush=True,
        )
        del task_bases
        del challenges
        del solutions

        attempt_reasons = {}
        stdout_families = set()
        throw_examples = []
        skipped_arcgen = {"n": 0, "reasons": {}, "examples": {}}
        kept = []
        generators = {}
        allowed = []
        v2_ids = []
        try:
            v2_ids = arcgen_v2_ids(arcgen_root / "task_list.py")
            allowed = exclude_arcgen_families(v2_ids, holdout, diag)
            if any(task_id in banned for task_id in allowed):
                raise RuntimeError("exclude_arcgen_families left a banned id")
            root_text = str(arcgen_root.resolve())
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            import task_list

            generators = task_list.task_list()
            need = (arcgen_n + len(cycle) - 1) // len(cycle) if arcgen_n else 0
            print(
                "arcgen_generators v2=%d allowed=%d bases_needed=%d"
                % (len(v2_ids), len(allowed), need),
                flush=True,
            )
            kept, _alive = _collect_arcgen_bases(
                allowed,
                generators,
                need,
                skipped_arcgen,
                attempt_reasons,
                stdout_families,
                throw_examples,
                drop_target_equals_demo=True,
            )
            print(
                "arcgen_bases_kept=%d elapsed=%.1f" % (len(kept), time.time() - started),
                flush=True,
            )
            for geometry, color_name, perm in cycle:
                if written["arcgen"] >= arcgen_n:
                    break
                for base in kept:
                    if written["arcgen"] >= arcgen_n:
                        break
                    episode_base = {
                        "query_kind": "gen",
                        "query_index": base["instance"],
                        "demos": base["demos"],
                        "query_input": base["query_input"],
                        "target": base["target"],
                    }
                    try:
                        episode = _make_episode(
                            base["task_id"],
                            episode_base,
                            geometry,
                            color_name,
                            perm,
                            seed,
                            source=ARCGEN_SOURCE,
                            revision=ARCGEN_REVISION,
                            drop_suspect=True,
                        )
                    except Exception as exc:
                        _note_skip(
                            arcgen_row_skips,
                            "episode_build_failed",
                            base["task_id"],
                            "gen",
                            base["instance"],
                        )
                        if len(throw_examples) < 8:
                            throw_examples.append(
                                {
                                    "task_id": base["task_id"],
                                    "instance": base["instance"],
                                    "slot": None,
                                    "attempt": None,
                                    "error": str(exc)[:200],
                                }
                            )
                        continue
                    if _target_in_demos(episode):
                        _note_skip(
                            arcgen_row_skips,
                            "target_equals_demo_output",
                            base["task_id"],
                            "gen",
                            base["instance"],
                        )
                        continue
                    if episode["source"] != ARCGEN_SOURCE or episode["revision"] != ARCGEN_REVISION:
                        raise RuntimeError("arcgen row lost its source or revision")
                    if episode["query_kind"] != "gen":
                        raise RuntimeError("arcgen query_kind")
                    if accept(episode, ARCGEN_SOURCE):
                        written["arcgen"] += 1
                        if written["arcgen"] % 2000 == 0:
                            print("arcgen_rows=%d" % written["arcgen"], flush=True)
        except Exception as exc:
            _note_skip(skipped_arcgen, "arcgen_aborted", "_pool", "gen", -1)
            if len(throw_examples) < 8:
                throw_examples.append(
                    {
                        "task_id": "_pool",
                        "instance": None,
                        "slot": None,
                        "attempt": None,
                        "error": "%s: %s" % (type(exc).__name__, exc)[:200],
                    }
                )
            print("arcgen_aborted %s: %s" % (type(exc).__name__, exc), flush=True)

    official_families = by_source[SOURCE]["families"]
    arcgen_families = by_source[ARCGEN_SOURCE]["families"]
    instance_counts = {}
    for base in kept:
        key = str(base["instance"])
        instance_counts[key] = instance_counts.get(key, 0) + 1
    manifest = {
        "created": G2_CREATED,
        "seed": seed,
        "trained": False,
        "commit": commit,
        "n_episodes": written["official"] + written["arcgen"],
        "n_episodes_requested": official_n + arcgen_n,
        "row_counts_by_source": {
            SOURCE: written["official"],
            ARCGEN_SOURCE: written["arcgen"],
        },
        "official_requested": official_n,
        "arcgen_requested": arcgen_n,
        "official_shortfall": official_n - written["official"],
        "arcgen_shortfall": arcgen_n - written["arcgen"],
        "family_counts": {
            SOURCE: len(official_families),
            ARCGEN_SOURCE: len(arcgen_families),
            "unique": len(official_families | arcgen_families),
            "official_train_available": len(train),
            "official_train_with_base": families_with_base,
            "arcgen_v2_generators": len(v2_ids),
            "arcgen_v2_allowed": len(allowed),
            "arcgen_v2_excluded": len(v2_ids) - len(allowed) if v2_ids else None,
            "arcgen_v2_with_base": len({base["task_id"] for base in kept}),
        },
        "geometry_counts": geometry_counts,
        "color_counts": color_counts,
        "geometry_counts_by_source": {
            SOURCE: by_source[SOURCE]["geometry"],
            ARCGEN_SOURCE: by_source[ARCGEN_SOURCE]["geometry"],
        },
        "color_counts_by_source": {
            SOURCE: by_source[SOURCE]["color"],
            ARCGEN_SOURCE: by_source[ARCGEN_SOURCE]["color"],
        },
        "query_kind_counts": query_kind_counts,
        "diag_holdout_leak_count": leak_count,
        "duplicate_episode_id_skipped": duplicate_skipped,
        "episodes_sha256": _sha256_file(episodes_path),
        "skip_reasons": {
            "official_base": skipped_official["reasons"],
            "arcgen_instance": skipped_arcgen["reasons"],
            "arcgen_attempt": attempt_reasons,
            "arcgen_row": arcgen_row_skips["reasons"],
        },
        "skip_examples": {
            "official_base": skipped_official["examples"],
            "arcgen_instance": skipped_arcgen["examples"],
            "arcgen_row": arcgen_row_skips["examples"],
            "arcgen_throw": throw_examples,
        },
        "arcgen_bases_kept": len(kept),
        "arcgen_bases_needed_for_full_cycle": (arcgen_n + len(cycle) - 1) // len(cycle) if arcgen_n else 0,
        "arcgen_instance_counts": instance_counts,
        "arcgen_stdout_family_count": len(stdout_families),
        "arcgen_draw_timeout_seconds": ARCGEN_DRAW_TIMEOUT_SECONDS,
        "source_revisions": {
            SOURCE: REVISION,
            ARCGEN_SOURCE: ARCGEN_REVISION,
        },
        "color_perms": {
            "ident": list(range(N_COLORS)),
            "perm1": list(perms[0]),
            "perm2": list(perms[1]),
            "perm3": list(perms[2]),
        },
        "augmentation_cycle": ["%s|%s" % (geometry, color_name) for geometry, color_name, _perm in cycle],
        "sampling": (
            "Official: round-robin over split.json train ids in file order, one row per family per turn. "
            "Within a family, bases stay in collect_bases order (test index, then loo index). "
            "Each base is expanded by color table outer (ident, perm1, perm2, perm3) and geometry inner "
            "(ident, rot90, rot180, rot270, flipud, fliplr) before the next base repeats that 24-step cycle. "
            "iter_augmented()'s ident/perm1 front prefix is not used. Cap %d. "
            "ARC-GEN: V2 import order after '# V2 Tasks', minus diag and holdout via exclude_arcgen_families. "
            "Each allowed family receives one kept instance before any family receives a second. "
            "An instance is four generate() calls, slots 0-2 demos and slot 3 query, seeded by sha256 "
            "of '20260930:'+task_id+':'+instance+':'+slot (first 8 bytes, little-endian) into random.seed. "
            "A rejected slot appends ':'+attempt and is retried up to 20 times. "
            "A generate() call that does not return within %d seconds is a failed attempt "
            "(generate_timeout), not a stored grid. "
            "Kept bases are then expanded across the same 24-step cycle, one augmentation at a time "
            "across bases, until %d rows. Neither side borrows rows from the other."
            % (official_n, int(ARCGEN_DRAW_TIMEOUT_SECONDS), arcgen_n)
        ),
        "prompt_tokens_note": (
            "prompt_tokens and target_tokens are null because tokenization happens "
            "on the Kaggle tokenizer."
        ),
        "elapsed_seconds": round(time.time() - started, 1),
        "inputs": {
            "challenges": _file_record(challenges_file),
            "solutions": _file_record(solutions_file),
            "split": _file_record(split_path),
            "arcgen_root": str(arcgen_root.resolve()),
        },
        "python": sys.version.split()[0],
    }
    with manifest_path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print("official=%d arcgen=%d leak=%d" % (written["official"], written["arcgen"], leak_count))
    print("family_counts=%s" % json.dumps(manifest["family_counts"], sort_keys=True))
    print("geometry_counts=%s" % json.dumps(geometry_counts, sort_keys=True))
    print("color_counts=%s" % json.dumps(color_counts, sort_keys=True))
    print("query_kind_counts=%s" % json.dumps(query_kind_counts, sort_keys=True))
    print("skip_reasons=%s" % json.dumps(manifest["skip_reasons"], sort_keys=True))
    print("episodes_sha256=%s" % manifest["episodes_sha256"])
    print("elapsed_seconds=%s" % manifest["elapsed_seconds"])
    print("wrote %s" % episodes_path)
    print("wrote %s" % manifest_path)
    return manifest


# ---- D2 (008 plan S4): independent-base-first index, true-token buckets, coverage ----

def d2_first_k_augmentations(color_tables, k=D2_DEFAULT_PER_BASE_K):
    """First K consistent augments of the shared 24-step cycle (008 S4.3).

    Each entry applies one geometry+color to every demo, the query and the
    target together. Default K=2. No per-task RNG; order is the fixed cycle.
    """
    if k < 1:
        raise ValueError("per-base K must be >= 1")
    return _geometry_color_cycle(color_tables)[:k]


def d2_sort_key(row):
    return (
        str(row.get("task_id", "")),
        str(row.get("base_instance_id", "")),
        str(row.get("query_kind", "")),
        int(row.get("query_index", 0)),
        str(row.get("augmentation", "")),
    )


def d2_order_episodes(official_rows, gen_rows, official_frac=D2_DEFAULT_OFFICIAL_FRAC):
    """Fixed order source -> task -> base -> query -> aug (008 S4.4).

    Within each source, rows are sorted by task/base/query/aug. Sources are
    interleaved by a deterministic 40/60 quota (default 2 official per 5
    slots; configurable via official_frac). An exhausted side does not borrow
    rows from the other side; the remainder of the live side is appended.
    """
    if not 0 < official_frac < 1:
        raise ValueError("official_frac must be in (0,1)")
    off = sorted(official_rows, key=d2_sort_key)
    gen = sorted(gen_rows, key=d2_sort_key)
    # Quota pattern over 20 slots keeps 0.4/0.6 exact without float drift.
    n_off = round(official_frac * 20)
    pattern = (["off"] * n_off + ["gen"] * (20 - n_off)) or ["off", "gen"]
    out, io, ig, slot = [], 0, 0, 0
    while io < len(off) or ig < len(gen):
        want = pattern[slot % len(pattern)]
        slot += 1
        if want == "off" and io < len(off):
            out.append(off[io])
            io += 1
        elif want == "gen" and ig < len(gen):
            out.append(gen[ig])
            ig += 1
        elif io < len(off):
            out.append(off[io])
            io += 1
        elif ig < len(gen):
            out.append(gen[ig])
            ig += 1
    return out


def _d2_bucket_name(n):
    if n <= 1024:
        return "<=1024"
    if n <= 2048:
        return "1025-2048"
    if n <= 4096:
        return "2049-4096"
    if n <= 8192:
        return "4097-8192"
    return ">8192"


def _d2_encode(tokenizer, text):
    if hasattr(tokenizer, "encode") and callable(tokenizer.encode):
        return tokenizer.encode(text)
    if callable(tokenizer):
        return tokenizer(text)
    raise TypeError("tokenizer must expose encode(text) or be callable")


def d2_token_buckets(episodes, tokenizer, all_task_ids=None):
    """True-token buckets over full prompt+target+EOS (008 S4.7).

    tokenizer is an external object (no weight download here). Each episode
    counts len(enc(prompt_text)) + len(enc(target_text)) + 1 EOS. prompt_text
    is system + user content; target_text is target_grid_text. No truncation:
    >8192 rows are only reported, never cut to fill a long bucket, and this
    function never reads max_len/overflow_len.
    Returns per-bucket {tasks, independent_bases, supervised_tokens,
    sources} plus lost_tasks (train ids with zero rows when all_task_ids
    is given).
    """
    buckets = {
        name: {"tasks": set(), "independent_bases": set(), "supervised_tokens": 0, "sources": {}}
        for name in D2_TOKEN_BUCKET_NAMES
    }
    for ep in episodes:
        msgs = ep.get("messages", [])
        sys_text = msgs[0].get("content", "") if len(msgs) > 0 else ""
        user_text = msgs[1].get("content", "") if len(msgs) > 1 else ""
        prompt_text = "%s\n%s" % (sys_text, user_text)
        target_text_ = ep.get("target_grid_text", "")
        n = len(_d2_encode(tokenizer, prompt_text)) + len(_d2_encode(tokenizer, target_text_)) + 1
        name = _d2_bucket_name(n)
        b = buckets[name]
        b["tasks"].add(ep.get("task_id"))
        b["independent_bases"].add(ep.get("base_instance_id"))
        b["supervised_tokens"] += n
        src = ep.get("source", "?")
        b["sources"][src] = b["sources"].get(src, 0) + 1
    out = {}
    for name in D2_TOKEN_BUCKET_NAMES:
        b = buckets[name]
        out[name] = {
            "n_tasks": len(b["tasks"]),
            "n_independent_bases": len(b["independent_bases"]),
            "supervised_tokens": b["supervised_tokens"],
            "sources": dict(b["sources"]),
        }
    covered = {ep.get("task_id") for ep in episodes}
    lost = sorted(set(all_task_ids or []) - covered)
    return {
        "buckets": out,
        "lost_tasks": lost,
        "n_episodes": len(episodes),
        "note": "Full prompt+target+EOS per episode; overlong only reported, never truncated to fake a long bucket.",
    }


def d2_coverage_report(
    train_ids,
    origin_of,
    v2_allowed_ids,
    official_query_cover,
    indexed_task_ids,
):
    """Coverage gaps for 008 S4 (numbers computed, not hardcoded).

    official_query_cover: {task_id: {'test': [idx...], 'loo': [idx...]}} as
    enumerated by collect_bases (suspect kept). indexed_task_ids: tasks with
    >=1 row in the first-round index. Returns non-ARC1 coverage, the no-V2
    55-equivalent subset covered by official only, the V2-is-not-new-task
    statement, and multi-query gaps (test1+ / loo beyond first base missing).
    """
    train_ids = list(train_ids)
    v2_allowed = set(v2_allowed_ids or [])
    indexed = set(indexed_task_ids or [])
    non_arc1 = [t for t in train_ids if origin_of(t) == "non_arc1"]
    non_arc1_in_index = [t for t in non_arc1 if t in indexed]
    no_v2 = [t for t in non_arc1 if t not in v2_allowed]
    no_v2_official_only = [t for t in no_v2 if t in indexed]
    multi_gap = {}
    for task_id, cov in (official_query_cover or {}).items():
        tests = sorted((cov or {}).get("test", []))
        loos = sorted((cov or {}).get("loo", []))
        if len(tests) > 1 or len(loos) > 1:
            multi_gap[task_id] = {
                "n_test_queries": len(tests),
                "n_loo_queries": len(loos),
                "note": "First-round index takes the first base only; test1+ / later LOO are enumerated but not yet indexed.",
            }
    return {
        "non_arc1_train": {
            "n_total": len(non_arc1),
            "n_in_index": len(non_arc1_in_index),
            "missing": sorted(set(non_arc1) - indexed),
        },
        "non_arc1_without_v2_official_only": {
            "n_total": len(no_v2),
            "n_covered_by_official": len(no_v2_official_only),
            "missing": sorted(set(no_v2) - indexed),
            "gen_coverage": 0,
        },
        "v2_is_not_new_task": (
            "ARC-GEN V2 is a generator-version label, not a new-task label. "
            "Allowed V2 families are a subset of official train families; "
            "unique families do not grow. V2 adds new instances, not unseen rules."
        ),
        "multi_query_gap": multi_gap,
        "n_train": len(train_ids),
        "n_indexed_tasks": len(indexed),
    }


def _load_id_set(path):
    if not path:
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    ids = data if isinstance(data, list) else data.get("ids", data)
    return set(ids)


class _D2DebugTokenizer:
    """Whitespace proxy for --d2-dry-run interface check only; NOT true tokens."""

    def encode(self, text):
        return text.split()


def d2_dry_run(
    challenges_path,
    solutions_path,
    max_tasks=3,
    per_base_k=D2_DEFAULT_PER_BASE_K,
    official_frac=D2_DEFAULT_OFFICIAL_FRAC,
    arc1_train_ids_path=None,
    arc1_eval_ids_path=None,
    seed=DEFAULT_SEED,
):
    """Micro index+bucket check on the first N official tasks (no model, no writes)."""
    arc1_train = _load_id_set(arc1_train_ids_path)
    arc1_eval = _load_id_set(arc1_eval_ids_path)
    challenges, _ = load_task_map(Path(challenges_path))
    solutions, _ = load_task_map(Path(solutions_path))
    task_ids = sorted(challenges.keys())[:max(1, max_tasks)]
    perms = derive_color_perms(seed)
    color_tables = _color_tables(perms)
    augs = d2_first_k_augmentations(color_tables, per_base_k)
    skipped = {"n": 0, "reasons": {}, "examples": {}}
    episodes, query_cover = [], {}
    for task_id in task_ids:
        bases = collect_bases(
            task_id,
            challenges[task_id],
            solutions.get(task_id),
            skipped,
            drop_target_equals_demo=False,
            min_loo_demos=2,
            arc1_train_ids=arc1_train,
            arc1_eval_ids=arc1_eval,
        )
        query_cover[task_id] = {
            "test": sorted({b["query_index"] for b in bases if b["query_kind"] == "test"}),
            "loo": sorted({b["query_index"] for b in bases if b["query_kind"] == "loo"}),
        }
        for base in bases[:2]:
            for geometry, color_name, perm in augs:
                episodes.append(_make_episode(task_id, base, geometry, color_name, perm, seed))
    ordered = d2_order_episodes(episodes, [], official_frac=official_frac)
    buckets = d2_token_buckets(ordered, _D2DebugTokenizer(), all_task_ids=task_ids)
    coverage = d2_coverage_report(
        task_ids,
        lambda t: classify_origin(t, arc1_train, arc1_eval),
        set(),
        query_cover,
        {ep["task_id"] for ep in ordered},
    )
    return {
        "tasks": task_ids,
        "n_bases": sum(len(v["test"]) + len(v["loo"]) for v in query_cover.values()),
        "query_cover": query_cover,
        "n_episodes": len(ordered),
        "skip_reasons": skipped["reasons"],
        "buckets_proxy_only": buckets,
        "coverage_micro": coverage,
        "tokenizer": "debug-whitespace-proxy (interface check; not 27B true tokens)",
    }


# ---- 009 handoff: materialize index + raw newbases + repair pair ----

MATERIALIZE_DEFAULT_OUT = "experiments/008/data/d2_materialized_episodes.jsonl"


def _parse_augmentation(aug):
    geometry, sep, color_name = str(aug).partition("|")
    if not sep or geometry not in GEOMETRIES:
        raise ValueError("bad augmentation: %r" % (aug,))
    return geometry, color_name


def _demo_blocks(user_text):
    """Yield (kind, text) for each [demo i] in/out grid block in user content."""
    lines = user_text.split("\n")
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.startswith("[demo ") and (line.endswith("] in:") or line.endswith("] out:")):
            kind = "in" if line.endswith("] in:") else "out"
            index += 1
            block = []
            while index < len(lines) and not (
                lines[index].startswith("[demo ") or lines[index].startswith("[query] ")
            ):
                block.append(lines[index])
                index += 1
            yield kind, "\n".join(block)
            continue
        index += 1


def _query_input_text(user_text):
    if not isinstance(user_text, str):
        return None
    lines = user_text.split("\n")
    for i, line in enumerate(lines):
        if line == "[query] in:":
            block = []
            j = i + 1
            while j < len(lines) and lines[j] != "[query] out:":
                block.append(lines[j])
                j += 1
            if j >= len(lines):
                return None
            return "\n".join(block)
    return None


def verify_episode_texts(episode):
    """Round-trip check: target, every demo block and the query input re-parse.

    Uses the episode's own formatter field (default space). Raises
    RuntimeError naming the first bad block; never guesses across formats.
    """
    formatter = normalize_formatter(episode.get("formatter", FORMATTER_F0))
    if parse_grid_text(episode.get("target_grid_text", ""), formatter) is None:
        raise RuntimeError("unparseable target_grid_text in %s" % episode.get("episode_id"))
    msgs = episode.get("messages", [])
    user_text = msgs[1].get("content", "") if len(msgs) > 1 else ""
    for kind, text in _demo_blocks(user_text):
        if parse_grid_text(text, formatter) is None:
            raise RuntimeError("unparseable demo %s block in %s" % (kind, episode.get("episode_id")))
    if parse_grid_text(_query_input_text(user_text) or "", formatter) is None:
        raise RuntimeError("unparseable query input in %s" % episode.get("episode_id"))


def convert_episode_formatter(episode, formatter):
    """Rebuild messages/target_grid_text under another formatter.

    Returns a new episode dict, or None when any grid block does not parse in
    the source format (rows without a formatter field read as F0 space).
    """
    target_fmt = normalize_formatter(formatter)
    src_fmt = normalize_formatter(episode.get("formatter", FORMATTER_F0))
    if src_fmt == target_fmt:
        out = dict(episode)
        out["formatter"] = target_fmt
        try:
            verify_episode_texts(out)
        except RuntimeError:
            return None
        return out
    msgs = episode.get("messages", [])
    user_text = msgs[1].get("content", "") if len(msgs) > 1 else ""
    target_grid = parse_grid_text(episode.get("target_grid_text", ""), src_fmt)
    query_grid = parse_grid_text(_query_input_text(user_text) or "", src_fmt)
    if target_grid is None or query_grid is None:
        return None
    demos = []
    pending_in = None
    for kind, text in _demo_blocks(user_text):
        grid = parse_grid_text(text, src_fmt)
        if grid is None:
            return None
        if kind == "in":
            pending_in = grid
        else:
            if pending_in is None:
                return None
            demos.append({"input": pending_in, "output": grid})
            pending_in = None
    if pending_in is not None:
        return None
    out = dict(episode)
    out["messages"] = build_messages(demos, query_grid, target_fmt)
    out["target_grid_text"] = serialize_grid(target_grid, target_fmt)
    out["formatter"] = target_fmt
    try:
        verify_episode_texts(out)
    except RuntimeError:
        return None
    return out


def _materialize_count(summary, key, example=None):
    summary["counts"][key] = summary["counts"].get(key, 0) + 1
    if example is not None:
        examples = summary["examples"].setdefault(key, [])
        if len(examples) < 8:
            examples.append(example)


def materialize_episodes(
    out_path,
    challenges_path=None,
    solutions_path=None,
    index_path=None,
    newbases_path=None,
    repair_path=None,
    g2_episodes_path=None,
    formatter=FORMATTER_F0,
    drop_suspect=False,
    seed=DEFAULT_SEED,
    first_n_tasks=None,
    max_rows=None,
    skip_index=False,
    skip_newbases=False,
    skip_repair=False,
):
    """Build trainer-ready episodes (messages/target_grid_text) from 008/009素材.

    Sources: D2 index rows (official rebuilt from challenges/solutions, gen
    resolved via ref_g2_episode_id in a G2 episodes file), raw newbases
    (planned_enhances expanded per base), and the 009 repair pilot
    (original + repaired). Index rows that violate the current
    min_loo_demos=2 rule are excluded and counted; input files are only read.
    Every emitted row passes verify_episode_texts. Never overwrites an
    existing file and never writes inside experiments/007.
    """
    formatter = normalize_formatter(formatter)
    out = Path(out_path)
    if out.exists():
        raise ValueError("refusing to overwrite existing file: %s (pass a new --out path)" % out)
    parts = out.resolve().parts
    if any(a == "experiments" and b == "007" for a, b in zip(parts, parts[1:])):
        raise ValueError("refusing to write inside experiments/007: %s" % out)
    perms = derive_color_perms(seed)
    perm_by_name = {name: list(perm) for name, perm in _color_tables(perms)}
    ident_perm = list(range(N_COLORS))
    summary = {
        "formatter": formatter,
        "drop_suspect": bool(drop_suspect),
        "seed": seed,
        "out": str(out),
        "counts": {},
        "examples": {},
        "inputs": {},
        "note": "Token buckets not assigned here; true-token counting stays with the tokenizer owner.",
    }

    challenges, solutions = None, None
    bases_cache = {}

    def get_bases(task_id):
        if task_id not in bases_cache:
            skipped = {"n": 0, "reasons": {}, "examples": {}}
            task = challenges.get(task_id) if isinstance(challenges, dict) else None
            sol = solutions.get(task_id) if isinstance(solutions, dict) else None
            bases_cache[task_id] = collect_bases(
                task_id, task, sol, skipped,
                drop_target_equals_demo=drop_suspect,
                min_loo_demos=2,
            )
        return bases_cache[task_id]

    emitted = []
    seen_ids = set()
    allow = None

    def capped():
        return max_rows is not None and len(emitted) >= max_rows

    def accept(episode, example):
        if episode["episode_id"] in seen_ids:
            _materialize_count(summary, "duplicate_episode_id_skipped", example)
            return False
        try:
            verify_episode_texts(episode)
        except RuntimeError as exc:
            _materialize_count(summary, "verify_failed", {"episode_id": example, "error": str(exc)[:200]})
            return False
        if drop_suspect and episode.get("suspect_same_as_demo"):
            _materialize_count(summary, "suspect_dropped", example)
            return False
        seen_ids.add(episode["episode_id"])
        emitted.append(episode)
        if episode.get("suspect_same_as_demo"):
            _materialize_count(summary, "suspect_kept", example)
        return True

    def base_to_episode(task_id, base, geometry, color_name, source, revision, example):
        perm = perm_by_name.get(color_name, ident_perm)
        try:
            return _make_episode(task_id, base, geometry, color_name, perm, seed,
                                 source=source, revision=revision,
                                 formatter=formatter, drop_suspect=drop_suspect)
        except RuntimeError as exc:
            _materialize_count(summary, "suspect_dropped", {"episode_id": example, "error": str(exc)[:200]})
            return None

    # ---- D2 index rows ----
    if not skip_index and index_path:
        summary["inputs"]["index"] = str(index_path)
        rows = [json.loads(line) for line in Path(index_path).read_text(encoding="utf-8").splitlines() if line.strip()]
        if first_n_tasks is not None:
            wanted = []
            for row in rows:
                tid = row.get("task_id")
                if tid not in wanted:
                    wanted.append(tid)
                if len(wanted) >= first_n_tasks:
                    break
            allow = set(wanted)
            summary["first_n_tasks"] = wanted
        gen_need = {}
        for row in rows:
            if row.get("source") == "arcgen_v2" and (allow is None or row.get("task_id") in allow):
                ref = row.get("ref_g2_episode_id") or "%s|%s|%d|%s|%s" % (
                    row.get("task_id"), row.get("query_kind"), int(row.get("query_index", 0)),
                    row.get("geometry"), row.get("color_name"))
                gen_need.setdefault(ref, row)
        g2_lookup = {}
        if gen_need and g2_episodes_path:
            summary["inputs"]["g2_episodes"] = str(g2_episodes_path)
            with Path(g2_episodes_path).open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    try:
                        ep = json.loads(line)
                    except ValueError:
                        continue
                    eid = ep.get("episode_id")
                    if eid in gen_need and eid not in g2_lookup:
                        g2_lookup[eid] = ep
                        if len(g2_lookup) >= len(gen_need):
                            break
        for row in rows:
            if capped():
                break
            task_id = row.get("task_id")
            if allow is not None and task_id not in allow:
                continue
            if row.get("source") == "arcgen_v2":
                ref = row.get("ref_g2_episode_id") or "%s|%s|%d|%s|%s" % (
                    task_id, row.get("query_kind"), int(row.get("query_index", 0)),
                    row.get("geometry"), row.get("color_name"))
                src = g2_lookup.get(ref)
                if src is None:
                    _materialize_count(summary, "gen_skipped_no_g2_lookup", row.get("sample_id"))
                    continue
                converted = convert_episode_formatter(src, formatter)
                if converted is None:
                    _materialize_count(summary, "gen_convert_unparseable", row.get("sample_id"))
                    continue
                if accept(converted, row.get("sample_id")):
                    _materialize_count(summary, "index_gen_materialized")
                continue
            # Official rows are rebuilt from challenges/solutions under current rules.
            if challenges is None:
                challenges, _ = load_task_map(Path(challenges_path))
                solutions, _ = load_task_map(Path(solutions_path))
                summary["inputs"]["challenges"] = str(challenges_path)
                summary["inputs"]["solutions"] = str(solutions_path)
            bases = get_bases(task_id)
            match = None
            for base in bases:
                if base["query_kind"] == row.get("query_kind") and base["query_index"] == int(row.get("query_index", -1)):
                    match = base
                    break
            if match is None:
                if row.get("query_kind") == "loo" and int(row.get("n_demos", 0) or 0) < 2:
                    _materialize_count(summary, "loo_demos_lt2_excluded", row.get("sample_id"))
                else:
                    _materialize_count(summary, "index_official_base_missing", row.get("sample_id"))
                continue
            try:
                geometry, color_name = _parse_augmentation(row.get("augmentation"))
            except ValueError:
                _materialize_count(summary, "index_official_base_missing", row.get("sample_id"))
                continue
            episode = base_to_episode(task_id, match, geometry, color_name, SOURCE, REVISION, row.get("sample_id"))
            if episode is None:
                continue
            if accept(episode, row.get("sample_id")):
                _materialize_count(summary, "index_official_materialized")

    # ---- raw newbases: planned enhances expanded per base ----
    if not skip_newbases and newbases_path and not capped():
        summary["inputs"]["newbases"] = str(newbases_path)
        for line in Path(newbases_path).read_text(encoding="utf-8").splitlines():
            if capped():
                break
            if not line.strip():
                continue
            base = json.loads(line)
            task_id = base.get("task_id")
            if allow is not None and task_id not in allow:
                continue
            try:
                instance = int(base.get("instance"))
            except (TypeError, ValueError):
                _materialize_count(summary, "newbase_bad_instance", base.get("base_instance_id"))
                continue
            enhances = base.get("planned_enhances") or ["ident|ident"]
            for aug in enhances:
                if capped():
                    break
                try:
                    geometry, color_name = _parse_augmentation(aug)
                except ValueError:
                    _materialize_count(summary, "newbase_bad_augmentation", base.get("base_instance_id"))
                    continue
                episode_base = {
                    "query_kind": "gen",
                    "query_index": instance,
                    "demos": base.get("demos", []),
                    "query_input": base.get("query_input"),
                    "target": base.get("target"),
                    "origin": base.get("origin", "unknown"),
                    "base_instance_id": base.get("base_instance_id", make_base_instance_id(task_id, "gen", instance)),
                    "suspect_same_as_demo": bool(suspect_same_as_demo(base.get("target"), base.get("demos", []))),
                    "target_isolated": True,
                }
                episode = base_to_episode(task_id, episode_base, geometry, color_name,
                                          base.get("source", ARCGEN_SOURCE), base.get("revision", ARCGEN_REVISION),
                                          base.get("base_instance_id"))
                if episode is None:
                    continue
                if accept(episode, base.get("base_instance_id")):
                    _materialize_count(summary, "newbase_materialized")

    # ---- 009 repair pilot: original + repaired ----
    if not skip_repair and repair_path and not capped():
        summary["inputs"]["repair"] = str(repair_path)
        for line in Path(repair_path).read_text(encoding="utf-8").splitlines():
            if capped():
                break
            if not line.strip():
                continue
            pair = json.loads(line)
            for side in ("original", "repaired"):
                if capped():
                    break
                base = pair.get(side)
                if not isinstance(base, dict):
                    continue
                task_id = base.get("task_id")
                if allow is not None and task_id not in allow:
                    continue
                try:
                    instance = int(base.get("instance"))
                except (TypeError, ValueError):
                    _materialize_count(summary, "repair_bad_instance", base.get("base_instance_id"))
                    continue
                enhances = base.get("planned_enhances") or ["ident|ident"]
                for aug in enhances:
                    if capped():
                        break
                    try:
                        geometry, color_name = _parse_augmentation(aug)
                    except ValueError:
                        continue
                    episode_base = {
                        "query_kind": "gen",
                        "query_index": instance,
                        "demos": base.get("demos", []),
                        "query_input": base.get("query_input"),
                        "target": base.get("target"),
                        "origin": base.get("origin", "unknown"),
                        "base_instance_id": base.get("base_instance_id", make_base_instance_id(task_id, "gen", instance)),
                        "suspect_same_as_demo": bool(suspect_same_as_demo(base.get("target"), base.get("demos", []))),
                        "target_isolated": True,
                    }
                    episode = base_to_episode(task_id, episode_base, geometry, color_name,
                                              base.get("source", ARCGEN_SOURCE), base.get("revision", ARCGEN_REVISION),
                                              base.get("base_instance_id"))
                    if episode is None:
                        continue
                    for extra in ("repair_status", "parent_base_instance_id"):
                        if base.get(extra) is not None:
                            episode[extra] = base[extra]
                    if accept(episode, base.get("base_instance_id")):
                        _materialize_count(summary, "repair_materialized_%s" % side)

    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="\n") as handle:
        for episode in emitted:
            handle.write(json.dumps(episode, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")
    summary["counts"]["n_written"] = len(emitted)
    summary["sha256"] = _sha256_file(out)
    print("materialize formatter=%s written=%d out=%s" % (formatter, len(emitted), out))
    print("materialize counts=%s" % json.dumps(summary["counts"], sort_keys=True))
    print("materialize sha256=%s" % summary["sha256"])
    print("wrote %s" % out)
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build the 007 G1 split and episode pool; D2 index helpers for 008 S4; --materialize builds trainer-ready episodes (009 handoff).")
    parser.add_argument("--challenges", required=True)
    parser.add_argument("--solutions", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--holdout-n", type=int, default=DEFAULT_HOLDOUT_N)
    parser.add_argument("--drop-suspect", action="store_true",
                        help="007 reproduction only: drop loo/gen bases whose target equals a demo output. D2 default keeps with suspect flag.")
    parser.add_argument("--per-base-k", type=int, default=D2_DEFAULT_PER_BASE_K,
                        help="D2: first K consistent augments per independent base (default 2).")
    parser.add_argument("--official-frac", type=float, default=D2_DEFAULT_OFFICIAL_FRAC,
                        help="D2: source quota start, official share (default 0.4).")
    parser.add_argument("--arc1-train-ids", default=None, help="JSON list of ARC1 train ids for origin labels.")
    parser.add_argument("--arc1-eval-ids", default=None, help="JSON list of ARC1 eval ids for origin labels.")
    parser.add_argument("--gen-cross-slot-check", action="store_true",
                        help="Anchor only: request ARC-GEN cross-slot shared-vars/validate() check (needs generator cooperation; not implemented, recorded in code_gaps_followup.json).")
    parser.add_argument("--d2-dry-run", action="store_true",
                        help="Micro check on first N official tasks: index+bucket logic only, no model, no writes.")
    parser.add_argument("--d2-dry-run-tasks", type=int, default=3)
    parser.add_argument("--materialize", action="store_true",
                        help="009 handoff: materialize D2 index + raw newbases + repair pilot into trainer-ready episodes (messages/target_grid_text). Never overwrites; never writes inside experiments/007.")
    parser.add_argument("--index", default="experiments/008/data/d2_sample_index.jsonl",
                        help="D2 metadata index to materialize (read-only).")
    parser.add_argument("--newbases", default="experiments/008/data/gen_newbases.jsonl",
                        help="Raw new independent bases with planned_enhances (read-only).")
    parser.add_argument("--repair", default="experiments/009/episode-repair-pilot.jsonl",
                        help="Original/repaired episode pair (read-only).")
    parser.add_argument("--g2-episodes", default=None,
                        help="G2 episodes.jsonl for gen index rows via ref_g2_episode_id (read-only). Without it, gen rows are skipped and counted.")
    parser.add_argument("--out", default=MATERIALIZE_DEFAULT_OUT,
                        help="New output JSONL path (default under experiments/008/data/). Refuses to overwrite.")
    parser.add_argument("--formatter", default=FORMATTER_F0,
                        help="Episode grid format: space (F0) or compact (F1).")
    parser.add_argument("--materialize-tasks", type=int, default=None,
                        help="Dry-run cap: only the first N distinct tasks in index order.")
    parser.add_argument("--materialize-max-rows", type=int, default=None,
                        help="Dry-run cap: stop after N emitted episode rows.")
    parser.add_argument("--skip-index", action="store_true", help="Materialize without D2 index rows.")
    parser.add_argument("--skip-newbases", action="store_true", help="Materialize without raw newbases.")
    parser.add_argument("--skip-repair", action="store_true", help="Materialize without the repair pilot.")
    args = parser.parse_args(argv)
    if args.n_episodes < 0:
        parser.error("--n-episodes must be >= 0")
    if args.gen_cross_slot_check:
        parser.error("--gen-cross-slot-check is an anchor only; generator cooperation recorded in experiments/008/code_gaps_followup.json, not implemented here")
    if args.d2_dry_run:
        summary = d2_dry_run(
            args.challenges,
            args.solutions,
            max_tasks=args.d2_dry_run_tasks,
            per_base_k=args.per_base_k,
            official_frac=args.official_frac,
            arc1_train_ids_path=args.arc1_train_ids,
            arc1_eval_ids_path=args.arc1_eval_ids,
            seed=args.seed,
        )
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return summary
    if args.materialize:
        try:
            formatter = normalize_formatter(args.formatter)
        except ValueError:
            parser.error("--formatter must be space (F0) or compact (F1)")
        if args.materialize_tasks is not None and args.materialize_tasks < 1:
            parser.error("--materialize-tasks must be >= 1")
        if args.materialize_max_rows is not None and args.materialize_max_rows < 1:
            parser.error("--materialize-max-rows must be >= 1")
        try:
            summary = materialize_episodes(
                args.out,
                challenges_path=args.challenges,
                solutions_path=args.solutions,
                index_path=None if args.skip_index else args.index,
                newbases_path=None if args.skip_newbases else args.newbases,
                repair_path=None if args.skip_repair else args.repair,
                g2_episodes_path=args.g2_episodes,
                formatter=formatter,
                drop_suspect=args.drop_suspect,
                seed=args.seed,
                first_n_tasks=args.materialize_tasks,
                max_rows=args.materialize_max_rows,
                skip_index=args.skip_index,
                skip_newbases=args.skip_newbases,
                skip_repair=args.skip_repair,
            )
        except ValueError as exc:
            parser.error(str(exc))
        print(json.dumps(summary, ensure_ascii=False, sort_keys=True, indent=2))
        return summary
    build(
        args.challenges,
        args.solutions,
        args.out_dir,
        args.n_episodes,
        args.seed,
        args.holdout_n,
    )


if __name__ == "__main__":
    main()
