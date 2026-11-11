"""Prepare official Yelp CSVs and the original BERT pooler features without changing MoMent."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def prepare(data_root, zip_path, model_path, gpu, batch_size=64):
    import numpy as np
    import pandas as pd
    import torch
    from transformers import AutoModel, AutoTokenizer

    target = data_root / 'Yelp'
    target.mkdir(parents=True, exist_ok=True)
    status_file = target / 'preparation_status.json'
    state = {'status': 'extracting', 'source_zip': str(zip_path), 'bert_model': str(model_path),
             'gpu': gpu, 'batch_size': batch_size, 'max_length': 512,
             'embedding': 'bert-base-uncased pooler_output, rounded to 8 decimal places'}
    write_json(status_file, state)
    required = ['edge_list.csv', 'entity_text.csv', 'relation_text.csv']
    with zipfile.ZipFile(zip_path) as archive:
        for name in required:
            choices = [item for item in archive.infolist() if Path(item.filename).name == name
                       and '__MACOSX' not in Path(item.filename).parts]
            assert len(choices) == 1, 'Ambiguous or missing ZIP member: ' + name
            entry = choices[0]
            output = target / name
            if output.exists():
                assert output.stat().st_size == entry.file_size, 'Existing CSV size differs: ' + name
                continue
            temporary = output.with_suffix('.csv.partial')
            with archive.open(entry) as source, temporary.open('wb') as destination:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            temporary.replace(output)

    frame = pd.read_csv(target / 'edge_list.csv', usecols=['u', 'i', 'r'])
    node_count = int(max(frame.u.max(), frame.i.max())) + 1
    relation_count = int(frame.r.max()) + 1
    del frame
    model_hash = digest(model_path / 'pytorch_model.bin')
    state.update(status='loading_bert', node_rows=node_count, relation_rows=relation_count,
                 bert_weights_sha256=model_hash)
    write_json(status_file, state)
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    model = AutoModel.from_pretrained(str(model_path), local_files_only=True).to('cuda:%d' % gpu).eval()
    assert model.config.hidden_size == 768

    for csv_name, feature_name, count in [('entity_text.csv', 'e_feat.npy', node_count),
                                           ('relation_text.csv', 'r_feat.npy', relation_count)]:
        output = target / feature_name
        if output.exists():
            existing = np.load(output, mmap_mode='r')
            assert existing.shape == (count, 768) and np.all(existing[0] == 0)
            continue
        temporary = target / (feature_name + '.partial')
        progress_file = target / (feature_name + '.progress.json')
        if progress_file.exists():
            progress = json.loads(progress_file.read_text())
            assert progress['bert_weights_sha256'] == model_hash
            completed = progress['completed_id']
            embeddings = np.lib.format.open_memmap(temporary, mode='r+')
            assert embeddings.shape == (count, 768)
        else:
            completed = 0
            embeddings = np.lib.format.open_memmap(temporary, mode='w+', dtype=np.float32, shape=(count, 768))
            embeddings[0] = 0
        expected = completed + 1
        state.update(status='embedding', feature=feature_name, completed_id=completed, total_rows=count)
        write_json(status_file, state)
        batch_number = 0
        for chunk in pd.read_csv(target / csv_name, chunksize=batch_size):
            chunk = chunk[chunk['i'] >= expected]
            if chunk.empty:
                continue
            ids = chunk['i'].to_numpy(dtype=np.int64)
            assert np.array_equal(ids, np.arange(expected, expected + len(ids))), 'Non-contiguous CSV IDs'
            text = [value if isinstance(value, str) else 'NULL' for value in chunk['text'].tolist()]
            encoded = tokenizer(text, padding=True, truncation=True, max_length=512, return_tensors='pt')
            with torch.no_grad():
                features = model(**encoded.to('cuda:%d' % gpu))[1].cpu().numpy()
            embeddings[ids] = np.around(features, 8).astype(np.float32)
            expected += len(ids)
            batch_number += 1
            if batch_number % 100 == 0:
                embeddings.flush()
                write_json(progress_file, {'completed_id': expected - 1, 'bert_weights_sha256': model_hash})
                state.update(completed_id=expected - 1, updated_at=time.time())
                write_json(status_file, state)
                print(feature_name, expected - 1, '/', count - 1, flush=True)
        assert expected == count, 'CSV did not cover the expected feature IDs'
        embeddings.flush()
        del embeddings
        temporary.replace(output)
        write_json(progress_file, {'completed_id': count - 1, 'bert_weights_sha256': model_hash, 'complete': True})
    state.update(status='ready', completed_at=time.time(), output_dtype='float32 (same dtype consumed by MoMent)')
    write_json(status_file, state)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--zip-file', type=Path, required=True)
    parser.add_argument('--bert-model', type=Path, required=True)
    parser.add_argument('--gpu', type=int, required=True)
    parser.add_argument('--batch-size', type=int, default=64)
    args = parser.parse_args()
    prepare(args.data_root, args.zip_file, args.bert_model, args.gpu, args.batch_size)
