"""Manual offline analysis of a downloaded 013 baseline pilot. Never starts jobs."""
import argparse
import hashlib
import json
from pathlib import Path


def read_json(path):
    return json.loads(Path(path).read_text())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--download', required=True)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    download = Path(args.download)
    run = download / '013-run'
    supervisor = read_json(run / 'supervisor.json')
    coverage = read_json(run / 'coverage.json')
    submission = read_json(download / 'submission.json')
    selected = read_json(run / 'selected-challenges.json')
    expected = read_json(root / 'kaggle-baseline-v3/source-manifest.json')
    source_matches = {name: hashlib.sha256((download / '013-code' / name).read_bytes()).hexdigest() == sha
                      for name, sha in expected.items()}
    events = []
    for path in sorted((run / 'events').glob('*.jsonl')):
        for line in path.read_text().splitlines():
            events.append(json.loads(line))
    loaded = [e for e in events if e.get('event') == 'model_loaded']
    trained = [e for e in events if e.get('event') == 'train_done']
    tasks = [e for e in events if e.get('event') == 'task_done']
    errors = [e for e in events if e.get('event') in {'pool_worker_error', 'worker_error', 'task_error'}]
    selected_keys = {f'{key}_{i}' for key, task in selected.items() for i in range(len(task['test']))}
    selected_rows = [row for row in coverage['rows'] if row['key'] in selected_keys]
    gold = read_json(root / 'execution/pilot-scoring-excluding-reserve.json')
    reserved = read_json(root / '../011/research-manifest.json')
    reserved_ids = {row['task_id'] for field in ['confirmation_C6', 'confirmation_C6_reserve']
                    for row in reserved[field]['entries']}
    assert not set(gold) & reserved_ids, 'Scoring must not access confirmation answers'
    scores = []
    for task_id, answers in gold.items():
        assert task_id in selected
        assert len(answers) == len(submission[task_id])
        for query, answer in enumerate(answers):
            prediction = submission[task_id][query]
            scores.append({'task_id': task_id, 'query': query,
                           'attempt_1_correct': prediction['attempt_1'] == answer,
                           'attempt_2_correct': prediction['attempt_2'] == answer,
                           'pass2_correct': any(prediction[k] == answer for k in ['attempt_1', 'attempt_2'])})
    ready_checks = {
        'supervisor_complete': supervisor['status'] == 'complete',
        'natural_worker_exit': supervisor.get('worker_returncode') == 0 and supervisor.get('stop_reason') == 'workers_finished',
        'within_pilot_wall': supervisor['elapsed_s'] <= 2700,
        'four_l4_loaded': len({e['rank'] for e in loaded}) == 4 and all('L4' in e['gpu_name'] for e in loaded),
        'four_real_adaptations': len(trained) == 4 and all(e.get('trained_steps', 0) > 0 for e in trained),
        'all_selected_queries_generated': len(selected_rows) == len(selected_keys) and all(not row['fallback_only'] for row in selected_rows),
        'no_worker_errors': not errors,
        'no_corrupt_candidates': coverage['corrupt_count'] == 0,
        'frozen_code_matches': all(source_matches.values()),
    }
    result = {
        'scope': 'Public diagnostic pilot, not a competition score or unseen confirmation; baseline score improvement unproven.',
        'download': str(download), 'source_matches': source_matches,
        'supervisor': supervisor, 'model_loaded_events': loaded,
        'train_done_events': trained, 'task_done_events': tasks, 'errors': errors,
        'selected_query_coverage': selected_rows,
        'operational_readiness_checks': ready_checks,
        'operationally_ready': all(ready_checks.values()),
        'diagnostic_score': {'tasks': len(gold), 'queries': len(scores),
                             'pass2_correct': sum(row['pass2_correct'] for row in scores), 'rows': scores,
                             'excluded_from_scoring': sorted(set(selected) & reserved_ids)},
        'warning': 'Even a passing pilot does not guarantee full hidden coverage, deadline behavior on every task, or improved leaderboard score.',
    }
    Path(args.out).write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'operationally_ready': result['operationally_ready'], 'checks': ready_checks,
                       'elapsed_s': supervisor['elapsed_s'], 'diagnostic_score': result['diagnostic_score']}))


if __name__ == '__main__':
    main()
