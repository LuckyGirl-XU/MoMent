"""Default-first, single-seed reproduction in isolated output directories."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import shutil
import subprocess
import sys
import threading
import time
import traceback

MODEL_SHA256 = 'eb182f1205566957d548ace13d93965b7618eb05a559e81735d266248c38ccc0'
CONFIG_LOCK = threading.Lock()


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False))
    temporary.replace(path)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def compare(result, target):
    metrics = {}
    for metric in ['average_precision', 'roc_auc']:
        actual = float(result['new node test metrics'][metric])
        assert 0 <= actual <= 1
        delta = actual - target[metric]
        metrics[metric] = {'value': actual, 'paper': target[metric], 'difference_pp': delta * 100,
                          'relative_difference_percent': delta / target[metric] * 100,
                          'within_2_percentage_points': abs(delta) <= .02 + 1e-12}
    return {'metrics': metrics, 'within_2_percentage_points': all(item['within_2_percentage_points'] for item in metrics.values())}


def validation_score(result):
    metrics = result['new node validate metrics']
    return (float(metrics['average_precision']) + float(metrics['roc_auc'])) / 2


def check_data(directory):
    import numpy as np
    import pandas as pd
    frame = pd.read_csv(directory/'edge_list.csv')
    assert frame.ts.is_monotonic_increasing, 'Input events are not chronological'
    assert np.isfinite(frame.ts.to_numpy()).all()
    result = {'rows': len(frame), 'max_node_id': int(max(frame.u.max(), frame.i.max())),
              'max_relation_id': int(frame.r.max()), 'time_cutoffs': np.quantile(frame.ts, [.7, .85]).tolist(), 'files': {}}
    for name, maximum in [('e_feat.npy', result['max_node_id']), ('r_feat.npy', result['max_relation_id'])]:
        array = np.load(directory/name, mmap_mode='r', allow_pickle=False)
        assert array.shape == (maximum+1, 768), (name, array.shape, maximum)
        assert np.all(array[0] == 0), name + ' padding row is nonzero'
        for start in range(0, len(array), 8192):
            assert np.isfinite(array[start:start+8192]).all(), name + ' contains non-finite features'
        result['files'][name] = {'shape': list(array.shape), 'dtype': str(array.dtype)}
    for name in ['edge_list.csv', 'entity_text.csv', 'relation_text.csv', 'e_feat.npy', 'r_feat.npy']:
        path = directory/name
        before = path.stat()
        result['files'].setdefault(name, {}).update(bytes=before.st_size, sha256=digest(path), mtime_ns=before.st_mtime_ns)
        after = path.stat()
        assert (before.st_size, before.st_mtime_ns) == (after.st_size, after.st_mtime_ns), name
    return result


def source_files(source):
    names = ['train_link_prediction.py', 'evaluate_models_utils.py']
    for folder in ['models', 'utils']:
        names += [str(path.relative_to(source)) for path in (source/folder).glob('*.py')]
    return names


def copy_attempt(source, destination, data_root, dataset, overrides):
    destination.mkdir(parents=True, exist_ok=False)
    for name in source_files(source):
        target = destination/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source/name, target)
    with CONFIG_LOCK:
        table = json.loads((source/'configs/moment_link_prediction.json').read_text())
    table['datasets'][dataset] = overrides
    write_json(destination/'configs/moment_link_prediction.json', table)
    (destination/'DyLink_Datasets').symlink_to(data_root, target_is_directory=True)
    assert digest(destination/'models/MoMent.py') == MODEL_SHA256
    manifest = {name: digest(destination/name) for name in source_files(destination)}
    manifest['configs/moment_link_prediction.json'] = digest(destination/'configs/moment_link_prediction.json')
    write_json(destination/'source_manifest.json', manifest)


def run_attempt(options, dataset, gpu, name, overrides, state, save):
    root = options.output/dataset/name
    result_file = root/'saved_results/MoMent'/dataset/'MoMent_seed0Bert.json'
    if root.exists():
        # Completed runs are immutable and can be collected again after a coordinator restart.
        table = json.loads((root/'configs/moment_link_prediction.json').read_text())
        assert table['datasets'][dataset] == overrides, 'Existing run has different parameters'
        assert digest(root/'models/MoMent.py') == MODEL_SHA256
        if result_file.exists():
            return json.loads(result_file.read_text()), root
        raise RuntimeError('Incomplete existing run; inspect before resuming: ' + str(root))
    copy_attempt(options.source, root, options.data_root, dataset, overrides)
    command = [options.python, '-u', 'train_link_prediction.py', '--dataset_name', dataset,
               '--model_name', 'MoMent', '--best-configs', '--num_runs', '1', '--gpu', str(gpu)]
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1', OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
    code = "import json; from utils.load_configs import get_link_prediction_args; print(json.dumps(vars(get_link_prediction_args())))"
    checked = subprocess.run([options.python, '-c', code] + command[3:], cwd=root, env=env,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, universal_newlines=True)
    effective = json.loads(checked.stdout)
    assert effective['dataset_name'] == dataset and effective['num_runs'] == 1
    assert effective['use_feature'] == 'Bert' and effective['time_gap'] == 2000
    write_json(root/'effective_config.json', effective)
    with (root/'stdout.log').open('wb') as output:
        child = subprocess.Popen(command, cwd=root, env=env, stdin=subprocess.DEVNULL,
                                 stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
        state['active_run'] = {'attempt': name, 'pid': child.pid, 'root': str(root), 'gpu': gpu,
                               'command': command, 'effective_config': effective, 'started_at': now()}
        save()
        returncode = child.wait()
    state['active_run'].update(returncode=returncode, finished_at=now())
    save()
    assert returncode == 0 and result_file.exists(), 'Training failed: ' + str(root)
    assert digest(root/'models/MoMent.py') == MODEL_SHA256
    return json.loads(result_file.read_text()), root


def process_dataset(options, dataset, gpu):
    directory = options.output/dataset
    directory.mkdir(parents=True, exist_ok=True)
    state_path = directory/'status.json'
    if state_path.exists():
        existing = json.loads(state_path.read_text())
        if existing['status'] in ['passed', 'outside_tolerance', 'user_accepted']:
            return existing
    state = {'dataset': dataset, 'gpu': gpu, 'status': 'preparing_data', 'started_at': now(), 'attempts': []}

    def save():
        state['updated_at'] = now()
        write_json(state_path, state)

    save()
    try:
        if dataset == 'Yelp' and not all((options.data_root/dataset/name).exists() for name in ['edge_list.csv', 'e_feat.npy', 'r_feat.npy']):
            state['status'] = 'waiting_for_yelp_transfer'
            save()
            while not options.yelp_zip.exists():
                partial = options.yelp_zip.with_name(options.yelp_zip.name + '.partial')
                if (partial.exists() and options.yelp_zip_bytes and options.yelp_zip_sha256
                        and partial.stat().st_size == options.yelp_zip_bytes
                        and digest(partial) == options.yelp_zip_sha256):
                    partial.replace(options.yelp_zip)
                    break
                time.sleep(30)
            state['status'] = 'preparing_yelp_features'
            save()
            with (directory/'preparation.log').open('ab') as output:
                child = subprocess.Popen([options.python, str(options.source/'scripts/prepare_yelp_features.py'),
                    '--data-root', str(options.data_root), '--zip-file', str(options.yelp_zip),
                    '--bert-model', str(options.bert_model), '--gpu', str(gpu)],
                    stdout=output, stderr=subprocess.STDOUT)
                state['preparation_pid'] = child.pid
                save()
                assert child.wait() == 0, 'Yelp feature preparation failed; see preparation.log'
        state['status'] = 'checking_data'
        save()
        data = check_data(options.data_root/dataset)
        write_json(directory/'data_manifest.json', data)
        with CONFIG_LOCK:
            table = json.loads((options.source/'configs/moment_link_prediction.json').read_text())
        initial = dict(table['datasets'][dataset])
        target = table['paper_targets'][dataset]
        candidates = [('default', initial)]
        paper_override = dict(initial, time_feat_dim=128, loss_weight=.2)
        fast_override = dict(initial, learning_rate=3e-5)
        for name, parameters in [('paper128', paper_override), ('learning_rate_3e5', fast_override)]:
            if parameters not in [item[1] for item in candidates]:
                candidates.append((name, parameters))
        selected = None
        for name, parameters in candidates[:options.max_attempts]:
            state['status'] = 'training'
            save()
            result, root = run_attempt(options, dataset, gpu, name, parameters, state, save)
            entry = {'name': name, 'parameters': parameters, 'root': str(root), 'result': result,
                     'validation_score': validation_score(result), 'comparison': compare(result, target)}
            state['attempts'].append(entry)
            state.pop('active_run', None)
            save()
            if name == 'default' and entry['comparison']['within_2_percentage_points']:
                selected = entry
                break
        if selected is None:
            selected = max(state['attempts'], key=lambda item: item['validation_score'])
        state.update(selected=selected, status='passed' if selected['comparison']['within_2_percentage_points'] else 'outside_tolerance',
                     completed_at=now(), num_runs_per_attempt=1,
                     selection_policy='Prefer default if it meets tolerance. Otherwise rank the predeclared fallbacks by new-node validation AP/AUC mean, with fixed seed 0.')
        with CONFIG_LOCK:
            path = options.source/'configs/moment_link_prediction.json'
            latest = json.loads(path.read_text())
            latest['datasets'][dataset] = selected['parameters']
            write_json(path, latest)
            verification_path = options.source/'configs/moment_verification.json'
            verification = json.loads(verification_path.read_text())
            verification['datasets'][dataset] = {'status': state['status'], 'parameters': selected['parameters'],
                'num_runs': 1, 'seed': 0, 'comparison': selected['comparison'], 'evidence': str(directory/'status.json')}
            write_json(verification_path, verification)
        write_json(directory/'report.json', state)
        lines = [dataset + '：' + state['status'], 'seed=0；每组参数仅执行一次；原版 MoMent。',
                 '最终参数覆盖：' + json.dumps(selected['parameters']), '选中实验：' + selected['root']]
        for metric, values in selected['comparison']['metrics'].items():
            lines.append('%s: %.4f，论文 %.4f，差异 %+.2f 个百分点' %
                         (metric, values['value'], values['paper'], values['difference_pp']))
        (directory/'report.txt').write_text('\n'.join(lines)+'\n')
        save()
    except Exception:
        state.update(status='failed', error=traceback.format_exc())
        save()
    return state


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--datasets', nargs='+', default=['ICEWS1819', 'Googlemap_CT', 'Stack_ubuntu', 'Amazon_movies', 'Yelp'])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])
    parser.add_argument('--python', default=sys.executable)
    parser.add_argument('--max-attempts', type=int, choices=[1, 2, 3], default=3)
    parser.add_argument('--yelp-zip', type=Path, required=True)
    parser.add_argument('--yelp-zip-bytes', type=int, default=0)
    parser.add_argument('--yelp-zip-sha256', default='')
    parser.add_argument('--bert-model', type=Path, required=True)
    options = parser.parse_args()
    options.source = options.source.resolve()
    options.data_root = options.data_root.resolve()
    options.output = options.output.resolve()
    options.output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (options.output/'coordinator.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    assert digest(options.source/'models/MoMent.py') == MODEL_SHA256
    assert len(set(options.datasets)) == len(options.datasets)
    assert len(set(options.gpus)) == len(options.gpus)
    jobs = queue.Queue()
    for dataset in options.datasets:
        jobs.put(dataset)
    results = {}

    def worker(gpu):
        while True:
            try:
                dataset = jobs.get_nowait()
            except queue.Empty:
                return
            results[dataset] = process_dataset(options, dataset, gpu)['status']
            jobs.task_done()

    write_json(options.output/'coordinator_status.json', {'status': 'running', 'pid': os.getpid(),
               'started_at': now(), 'datasets': options.datasets, 'gpus': options.gpus,
               'max_attempts': options.max_attempts, 'num_runs_per_attempt': 1})
    with ThreadPoolExecutor(max_workers=len(options.gpus)) as pool:
        futures = [pool.submit(worker, gpu) for gpu in options.gpus]
        for future in futures:
            future.result()
    write_json(options.output/'coordinator_status.json', {'status': 'completed', 'pid': os.getpid(),
               'completed_at': now(), 'datasets': results})


if __name__ == '__main__':
    main()
