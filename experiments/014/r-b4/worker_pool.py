"""One persistent model per L4; launched inside a supervised process group."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback


def parse_assignments(data, workers, assignments):
    """Validate pinned R assignments; return per-rank ordered task lists.

    Pure-data helper (no multiprocessing) so the pairing can be checked on
    CPU before push. Each rank must hold one task twice: [wave1, wave2].
    """
    if not isinstance(assignments, list) or not assignments:
        raise ValueError('Assignments must be a nonempty list')
    per_rank = [[] for _ in range(workers)]
    for entry in assignments:
        rank = entry['gpu_index']
        task_id = entry['task_id']
        if not isinstance(rank, int) or not 0 <= rank < workers:
            raise ValueError('gpu_index out of range: %r' % (entry,))
        if task_id not in data:
            raise ValueError('Assigned task missing from challenges: %r' % (task_id,))
        per_rank[rank].append(task_id)
    flat = sorted(task for tasks in per_rank for task in tasks)
    if flat != sorted(list(data) * 2):
        raise ValueError('R assignments must cover every challenge exactly twice '
                         '(wave1 then wave2): assigned=%r challenges=%r'
                         % (flat, sorted(data)))
    for rank, tasks in enumerate(per_rank):
        if len(tasks) != 2 or tasks[0] != tasks[1]:
            raise ValueError('R rank queue must hold one task twice '
                             '([wave1, wave2]): rank=%d tasks=%r' % (rank, tasks))
    return per_rank


def local_worker(rank, queue, deadline):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    marker_dir = Path(os.environ['ARC_MARKER_DIR'])
    marker_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Preserve the upstream serialized Unsloth import to avoid concurrent
        # first-import patching. The supervisor bounds this wait and imports.
        if rank:
            previous = marker_dir / str(rank - 1)
            while not previous.exists():
                if time.time() >= deadline:
                    return
                time.sleep(0.5)
        if time.time() >= deadline:
            return
        import torch
        torch.set_default_device('cpu')
        from arc_solver import worker
        (marker_dir / str(rank)).write_text('imported\n')
        worker(rank, queue, deadline)
    except BaseException as exc:
        row = {'event': 'pool_worker_error', 'rank': rank,
               'time': time.time(), 'error': repr(exc),
               'traceback': traceback.format_exc()}
        events = Path(os.environ['ARC_EVENT_DIR'])
        events.mkdir(parents=True, exist_ok=True)
        with (events / ('pool-error-%d.jsonl' % rank)).open('a') as handle:
            handle.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--challenges', required=True)
    parser.add_argument('--deadline', required=True, type=float)
    parser.add_argument('--workers', type=int, default=4)
    # R paired protocol (014): optional pinned task->GPU assignment. JSON list of
    # {"gpu_index": int, "task_id": str} in per-rank wave order (wave1 entry
    # first, wave2 entry second); each rank consumes its own queue in file
    # order, so the same task stays on the same card across waves. Every
    # challenge appears exactly twice (once per wave). Without this flag the
    # 013 shared-queue behavior is unchanged.
    parser.add_argument('--assignments', default=None)
    args = parser.parse_args()
    data = json.loads(Path(args.challenges).read_text())
    ctx = mp.get_context('spawn')
    if args.assignments is None:
        queue = ctx.Queue()
        for key in sorted(data):
            queue.put(key)
        for _ in range(args.workers):
            queue.put(None)
        queues = [queue] * args.workers
    else:
        assignments = json.loads(Path(args.assignments).read_text())
        per_rank = parse_assignments(data, args.workers, assignments)
        queues = []
        for tasks in per_rank:
            queue = ctx.Queue()
            for key in tasks:
                queue.put(key)
            queue.put(None)
            queues.append(queue)
    children = [ctx.Process(target=local_worker, args=(rank, queues[rank], args.deadline))
                for rank in range(args.workers)]
    for child in children:
        child.start()
    for child in children:
        child.join()
    codes = [child.exitcode for child in children]
    print(json.dumps({'event': 'pool_done', 'exit_codes': codes}), flush=True)
    return 0 if all(code == 0 for code in codes) else 2


if __name__ == '__main__':
    raise SystemExit(main())
