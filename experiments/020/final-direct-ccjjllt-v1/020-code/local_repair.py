"""020 bounded local semantic changes, selected exclusively by demo evidence.

Public API:
  repair_variants(code, max_variants=32) -> [{code, label, code_sha256}, ...]
  repair_program(code, demos, *, deadline=None, max_variants=32,
                 max_seconds=30, base_evaluation=None) -> result

No query or query gold is accepted. Every execution uses solve_program's
restricted subprocess. This is a small mutation family, not arbitrary search.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import time

import solve_program as SP
import submit_lib as SL


CARDINAL = ((-1, 0), (0, -1), (0, 1), (1, 0))
DIAGONAL = ((-1, -1), (-1, 1), (1, -1), (1, 1))


def _affine_coordinate(node):
    """Accept only name, name +/- literal one, or integer offset literal."""
    if isinstance(node, ast.Name):
        return node.id, 0
    value = SP._signed_integer_literal(node)
    if value is not None and -1 <= value <= 1:
        return None, value
    if (isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub))
            and isinstance(node.left, ast.Name)):
        value = SP._signed_integer_literal(node.right)
        if value is not None and -1 <= value <= 1:
            return node.left.id, value if isinstance(node.op, ast.Add) else -value
    return None


def _neighborhood(node):
    if not isinstance(node, (ast.Tuple, ast.List)) or len(node.elts) not in (4, 8):
        return None
    bases, offsets = None, []
    for pair in node.elts:
        if not isinstance(pair, (ast.Tuple, ast.List)) or len(pair.elts) != 2:
            return None
        coords = [_affine_coordinate(value) for value in pair.elts]
        if any(value is None for value in coords):
            return None
        pair_bases = tuple(value[0] for value in coords)
        if bases is None:
            bases = pair_bases
        if pair_bases != bases:
            return None
        offsets.append(tuple(value[1] for value in coords))
    if len(set(offsets)) != len(offsets):
        return None
    points = set(offsets)
    if points == set(CARDINAL):
        return bases, offsets, offsets + list(DIAGONAL), "neighbors4_to8"
    if points == set(CARDINAL + DIAGONAL):
        return bases, offsets, [point for point in offsets if point in CARDINAL], "neighbors8_to4"
    return None


def _coordinate_node(base, offset):
    if base is None:
        return ast.Constant(value=offset)
    name = ast.Name(id=base, ctx=ast.Load())
    if offset == 0:
        return name
    return ast.BinOp(left=name, op=ast.Add() if offset > 0 else ast.Sub(),
                     right=ast.Constant(value=abs(offset)))


def _named_method(statement, method):
    call = statement.value if isinstance(statement, ast.Expr) else None
    if (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            and isinstance(call.func.value, ast.Name) and call.func.attr == method
            and len(call.args) == 1 and not call.keywords):
        return call.func.value.id, call.args[0]
    return None


def _deferred_segment_marking(loop, following):
    """Recognize only adjacent collect/mark statements plus a length-confirmed record."""
    if (not isinstance(loop, ast.While) or loop.orelse
            or not isinstance(following, ast.If) or following.orelse
            or len(following.body) != 1):
        return None
    test = following.test
    if (not isinstance(test, ast.Compare) or len(test.ops) != 1
            or not isinstance(test.ops[0], ast.GtE) or len(test.comparators) != 1
            or SP._signed_integer_literal(test.comparators[0]) != 2
            or not isinstance(test.left, ast.Call) or not isinstance(test.left.func, ast.Name)
            or test.left.func.id != 'len' or len(test.left.args) != 1 or test.left.keywords
            or not isinstance(test.left.args[0], ast.Name)):
        return None
    cells = test.left.args[0].id
    record = _named_method(following.body[0], 'append')
    if (not record or record[0] == cells
            or not any(isinstance(n, ast.Name) and n.id == cells for n in ast.walk(record[1]))):
        return None
    for index in range(len(loop.body) - 1):
        collect = _named_method(loop.body[index], 'append')
        marking = _named_method(loop.body[index + 1], 'add')
        if not collect or not marking or collect[0] != cells:
            continue
        marked, coordinate = marking
        if marked in (cells, record[0]) or ast.dump(collect[1]) != ast.dump(coordinate):
            continue
        if not (isinstance(coordinate, ast.Name) or
                isinstance(coordinate, ast.Tuple) and len(coordinate.elts) == 2):
            continue
        # The same set must guard traversal. Reject additional mutations or
        # assignments to either container; this family moves one marking site.
        guards = [n for n in ast.walk(loop.test) if isinstance(n, ast.Compare)
                  and len(n.ops) == 1 and isinstance(n.ops[0], ast.NotIn)
                  and len(n.comparators) == 1 and isinstance(n.comparators[0], ast.Name)
                  and n.comparators[0].id == marked and ast.dump(n.left) == ast.dump(coordinate)]
        calls = [n for n in ast.walk(loop) if isinstance(n, ast.Call)
                 and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                 and n.func.value.id in (cells, marked)]
        assigned = any(isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)
                       and n.id in (cells, marked) for n in ast.walk(loop))
        if guards and len(calls) == 2 and not assigned:
            return {'mark_index': index + 1, 'cells': cells, 'marked': marked}
    return None


def repair_variants(code, max_variants=32):
    """One local edit per variant, deterministic AST order, no combinations.

    Supported changes:
      * Exact four/eight-neighbor coordinate or offset literals, one site.
      * An existing literal reverse=True/False argument on sorted()/list.sort().
      * Defer marking collected coordinates until an adjacent len>=2 segment
        confirmation records them, one loop site, with no variable-name rules.
    Only existing source structures can trigger a change. No constants learned
    from query answers or task IDs are used.
    """
    limit = min(32, max(0, int(max_variants)))
    if not limit:
        return []
    tree = SP.validate_code(code)
    nodes = list(ast.walk(tree))
    node_indices = {id(node): index for index, node in enumerate(nodes)}
    following_ifs = {}
    for parent in nodes:
        for _, value in ast.iter_fields(parent):
            if isinstance(value, list):
                for before, after in zip(value, value[1:]):
                    if isinstance(before, ast.While) and isinstance(after, ast.If):
                        following_ifs[id(before)] = after
    variants, seen = [], {hashlib.sha256(code.encode()).hexdigest()}
    for index, node in enumerate(nodes):
        neighborhood = _neighborhood(node)
        following = following_ifs.get(id(node))
        deferred = _deferred_segment_marking(node, following)
        change = None
        if neighborhood is not None:
            change = "neighborhood"
        elif isinstance(node, ast.Call):
            sort_call = ((isinstance(node.func, ast.Name) and node.func.id == "sorted")
                         or (isinstance(node.func, ast.Attribute) and node.func.attr == "sort"))
            if sort_call:
                for kw in node.keywords:
                    if kw.arg == "reverse" and isinstance(kw.value, ast.Constant) and type(kw.value.value) is bool:
                        change = "reverse"
                        break
        elif deferred is not None:
            change = "deferred_segment_marking"
        if change is None:
            continue
        modified = copy.deepcopy(tree)
        modified_nodes = list(ast.walk(modified))
        target = modified_nodes[index]
        if change == "neighborhood":
            bases, before, after, kind = neighborhood
            target.elts = [
                ast.Tuple(elts=[_coordinate_node(base, offset)
                                for base, offset in zip(bases, pair)], ctx=ast.Load())
                for pair in after
            ]
            label = "%s:L%d:C%d" % (kind, node.lineno, node.col_offset)
        elif change == "reverse":
            kw = next(k for k in target.keywords if k.arg == "reverse")
            kw.value = ast.Constant(value=not kw.value.value)
            label = "reverse_sort:L%d:C%d" % (node.lineno, node.col_offset)
        else:
            confirmed = modified_nodes[node_indices[id(following)]]
            del target.body[deferred['mark_index']]
            confirmed.body.insert(0, ast.Expr(value=ast.Call(
                func=ast.Attribute(value=ast.Name(id=deferred['marked'], ctx=ast.Load()),
                                   attr='update', ctx=ast.Load()),
                args=[ast.Name(id=deferred['cells'], ctx=ast.Load())], keywords=[])))
            label = "defer_mark_until_segment_confirmed:L%d:C%d" % (node.lineno, node.col_offset)
        ast.fix_missing_locations(modified)
        rendered = ast.unparse(modified) + "\n"
        try:
            SP.validate_code(rendered)
        except (ValueError, SyntaxError, RecursionError):
            continue
        digest = hashlib.sha256(rendered.encode()).hexdigest()
        if digest in seen:
            continue
        seen.add(digest)
        variants.append({"code": rendered, "label": label, "code_sha256": digest})
        if len(variants) >= limit:
            break
    return variants


def _evaluate_demos(code, demos, cutoff):
    results = []
    program_end = min(cutoff, time.monotonic() + 20.0)
    for demo in demos:
        remaining = min(2.0, program_end - time.monotonic())
        results.append(SP.execute_grid(code, demo["input"],
                                       wall_seconds=remaining, memory_mb=512))
    return results


def repair_program(code, demos, *, deadline=None, max_variants=32,
                   max_seconds=30, base_evaluation=None):
    """Evaluate bounded mutations against ALL demos; never accepts queries.

    `deadline` is absolute time.monotonic. The <=30-second wall-clock budget
    includes a baseline execution when none is supplied and all subprocess
    startup/execution. At most 32 mutation candidates are executed. A candidate
    is attempted only if baseline has a valid same-shape but non-exact demo;
    execution failures alone do not justify semantic neighborhood changes.
    `best` may be the original baseline; callers should retain original P0.
    """
    started = time.monotonic()
    seconds = min(30.0, max(0.0, float(max_seconds)))
    cutoff = min(started + seconds, deadline if deadline is not None else float("inf"))
    clean = [{"input": SP.checked_grid(d["input"]), "output": SP.checked_grid(d["output"])}
             for d in demos]
    result = {"variants": [], "variants_tried": 0, "variants_generated": 0,
              "elapsed_s": 0.0, "status": "not_started", "best": None,
              "max_variants": min(32, max(0, int(max_variants))),
              "max_seconds": seconds, "query_results_computed": False}
    def finish(status):
        result["status"] = status
        result["elapsed_s"] = round(time.monotonic() - started, 6)
        return result
    if not clean:
        return finish("ineligible_no_demos")
    if time.monotonic() >= cutoff:
        return finish("budget_interrupted")
    try:
        SP.validate_code(code)
    except (ValueError, SyntaxError, RecursionError) as exc:
        result["error"] = str(exc)[:400]
        return finish("ineligible_program_rejected")
    # Only reuse evaluations explicitly attached to these exact source bytes;
    # recompute metrics from demo grids, never trust a caller's summary score.
    if (isinstance(base_evaluation, dict) and base_evaluation.get("code") == code
            and isinstance(base_evaluation.get("demo_results"), list)
            and len(base_evaluation["demo_results"]) == len(clean)):
        baseline_results = base_evaluation["demo_results"]
    else:
        baseline_results = _evaluate_demos(code, clean, cutoff)
    metrics = SL.demo_metrics(clean, baseline_results)
    baseline = {"code": code, "label": "original", "order": 0,
                "code_sha256": hashlib.sha256(code.encode()).hexdigest(),
                "demo_results": baseline_results, "metrics": metrics}
    result["baseline"] = baseline
    result["best"] = baseline
    if metrics["all_demos_pass"]:
        return finish("already_all_demos_pass")
    if not any(d["shape_match"] and not d["exact"] for d in metrics["per_demo"]):
        return finish("ineligible_no_local_demo_evidence")
    variants = repair_variants(code, result["max_variants"])
    result["variants_generated"] = len(variants)
    if not variants:
        return finish("no_supported_variants")
    for index, variant in enumerate(variants):
        if time.monotonic() >= cutoff:
            return finish("budget_interrupted")
        demo_results = _evaluate_demos(variant["code"], clean, cutoff)
        row = {**variant, "order": index + 1, "demo_results": demo_results,
               "metrics": SL.demo_metrics(clean, demo_results)}
        result["variants"].append(row)
        result["variants_tried"] += 1
        if SL.candidate_sort_key(row) < SL.candidate_sort_key(result["best"]):
            result["best"] = row
        if row["metrics"]["all_demos_pass"]:
            return finish("all_demos_pass")
    return finish("complete")
