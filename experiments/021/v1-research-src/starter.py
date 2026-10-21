"""Four persistent V1 models, one independent task queue per shared process pool."""
import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
import traceback


def local_worker(rank, queue, end_time):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(rank)
    markers = Path(os.environ['ARC_MARKER_DIR'])
    try:
        if rank:
            while not (markers / str(rank - 1)).exists():
                if time.time() >= end_time:
                    return
                time.sleep(0.5)
        import torch
        torch.set_default_device('cpu')
        from arc_solver import worker
        (markers / str(rank)).write_text('imported\n')
        worker(rank, queue, end_time)
    except BaseException as exc:
        row = {'event': 'pool_worker_error', 'rank': rank, 'time': time.time(),
               'error': repr(exc), 'traceback': traceback.format_exc()}
        with (Path(os.environ['ARC_EVENT_DIR']) / ('pool-error-%d.jsonl' % rank)).open('a') as f:
            f.write(json.dumps(row) + '\n')
        print(json.dumps(row), flush=True)
        raise


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--end-time', type=float, required=True)
    args = parser.parse_args()
    data = json.loads(Path(os.environ['ARC_CHALLENGES_PATH']).read_text())
    context = mp.get_context('spawn')
    queue = context.Queue()
    for task_id in sorted(data):
        queue.put(task_id)
    for _ in range(4):
        queue.put(None)
    children = [context.Process(target=local_worker, args=(rank, queue, args.end_time)) for rank in range(4)]
    for child in children:
        child.start()
    for child in children:
        child.join()
    codes = [child.exitcode for child in children]
    print(json.dumps({'event': 'pool_finished', 'exit_codes': codes}), flush=True)
    return 0 if all(code == 0 for code in codes) else 2


if __name__ == '__main__':
    raise SystemExit(main())
