import os
import time
import json
import hashlib
from fractions import Fraction
import torch
import argparse
import torch.multiprocessing as mp


def local_worker(rank, queue, end_time):
    
    os.environ["CUDA_VISIBLE_DEVICES"] = str(rank)

    torch.set_default_device("cpu")

    # Fix Unsloth patching issue
    if rank > 0:
        while not os.path.exists(f"/kaggle/worker{rank-1}"):
            time.sleep(5)
    
    from arc_solver import worker

    with open(f"/kaggle/worker{rank}", "w") as f:
        f.write("Ok")
    
    print(f"[Rank {rank}] start!")
    
    worker(rank, queue, end_time)
    
    print(f"[Rank {rank}] done!")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--end-time", type=float, default=0.0)
    args = parser.parse_args()

    rerun_mode = os.getenv("KAGGLE_IS_COMPETITION_RERUN")

    if rerun_mode:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_test_challenges.json"
    else:
        test_path = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_challenges.json"

    with open(test_path, "rb") as f:
        source_raw = f.read()
    data = json.loads(source_raw)

    # Visible inputs/demos only. Never read a test-query output.
    def grid_cells(grid):
        return sum(len(row) for row in grid)

    priority_rows = []
    for key in data:
        if not rerun_mode and key not in ['2ba387bc', 'd35bdbdc']:
            continue
        task = data[key]
        n_queries = len(task['test'])
        if n_queries < 1:
            raise ValueError('Task has no test query: '+key)
        visible_cells = sum(grid_cells(pair['input']) + grid_cells(pair['output'])
                            for pair in task['train'])
        visible_cells += sum(grid_cells(query['input']) for query in task['test'])
        priority = Fraction(visible_cells, n_queries)
        priority_rows.append(dict(task_id=key, visible_cell_cost=visible_cells,
            n_test_queries=n_queries, priority_numerator=priority.numerator,
            priority_denominator=priority.denominator))
    priority_rows.sort(key=lambda row: (Fraction(row['priority_numerator'],
        row['priority_denominator']), row['task_id']))
    actual_order = [row['task_id'] for row in priority_rows]
    expected_ids = set(data) if rerun_mode else {'2ba387bc', 'd35bdbdc'}
    if len(actual_order) != len(set(actual_order)) or set(actual_order) != expected_ids:
        raise RuntimeError('Queue membership must preserve every selected task exactly once')
    queue_manifest = dict(mode='hidden_test' if rerun_mode else 'ordinary_order_smoke',
        source_path=test_path, source_sha256=hashlib.sha256(source_raw).hexdigest(),
        priority_rule='visible_cells(train inputs+outputs and test inputs) / n_test_queries; exact Fraction, task_id ties',
        original_reference='033 V2/335137615', actual_order=actual_order,
        n_tasks=len(actual_order), n_queries=sum(row['n_test_queries'] for row in priority_rows),
        entries=priority_rows, query_outputs_read=False)
    with open('queue-manifest.json', 'w') as f:
        json.dump(queue_manifest, f, indent=2)
    print(json.dumps(dict(event='queue_priority_manifest', **queue_manifest)), flush=True)

    queue = mp.Manager().Queue()
    for key in actual_order:
        queue.put(key)
    for _ in range(4):
        queue.put(None)
    
    mp.spawn(local_worker, args=(queue, args.end_time), nprocs=4)
