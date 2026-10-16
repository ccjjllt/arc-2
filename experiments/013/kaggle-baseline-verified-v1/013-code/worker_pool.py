"""One persistent model per L4; launched inside a supervised process group."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback


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
    args = parser.parse_args()
    data = json.loads(Path(args.challenges).read_text())
    ctx = mp.get_context('spawn')
    queue = ctx.Queue()
    for key in sorted(data):
        queue.put(key)
    for _ in range(args.workers):
        queue.put(None)
    children = [ctx.Process(target=local_worker, args=(rank, queue, args.deadline))
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
