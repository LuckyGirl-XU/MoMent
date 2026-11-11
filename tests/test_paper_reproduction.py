import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT/'scripts'/filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = load('paper_runner', 'run_paper_reproduction.py')
preparation = load('yelp_preparation', 'prepare_yelp_features.py')


def result(validation, test):
    return {group: {'average_precision': str(test if 'test' in group else validation),
                    'roc_auc': str(test if 'test' in group else validation)}
            for group in ['validate metrics', 'new node validate metrics', 'test metrics', 'new node test metrics']}


class ReproductionTests(unittest.TestCase):
    def test_tolerance_uses_both_metrics_and_both_directions(self):
        self.assertTrue(runner.compare(result(.9, .92), {'average_precision': .9, 'roc_auc': .9})['within_2_percentage_points'])
        self.assertFalse(runner.compare(result(.9, .93), {'average_precision': .9, 'roc_auc': .9})['within_2_percentage_points'])
        self.assertFalse(runner.compare(result(.9, .87), {'average_precision': .9, 'roc_auc': .9})['within_2_percentage_points'])

    def test_copy_isolates_outputs_and_preserves_original_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = Path(temporary)/'attempt'
            runner.copy_attempt(ROOT, target, Path(temporary)/'data', 'ICEWS1819', {'time_feat_dim': 128})
            self.assertEqual(runner.digest(target/'models/MoMent.py'), runner.MODEL_SHA256)
            copied = json.loads((target/'configs/moment_link_prediction.json').read_text())
            original = json.loads((ROOT/'configs/moment_link_prediction.json').read_text())
            self.assertEqual(copied['datasets']['ICEWS1819'], {'time_feat_dim': 128})
            self.assertEqual(copied['datasets']['Stack_elec'], original['datasets']['Stack_elec'])
            with self.assertRaises(FileExistsError):
                runner.copy_attempt(ROOT, target, Path(temporary)/'data', 'ICEWS1819', {})

    def test_default_pass_runs_once_and_fallbacks_rank_validation(self):
        for default_pass in [True, False]:
            with self.subTest(default_pass=default_pass), tempfile.TemporaryDirectory() as temporary:
                base = Path(temporary)
                source = base/'source'
                (source/'configs').mkdir(parents=True)
                for name in ['moment_link_prediction.json', 'moment_verification.json']:
                    (source/'configs'/name).write_bytes((ROOT/'configs'/name).read_bytes())
                options = SimpleNamespace(source=source, output=base/'output', data_root=base/'data', max_attempts=3)
                calls = []

                def fake_attempt(options, dataset, gpu, name, overrides, state, save):
                    calls.append(name)
                    values = {'default': (.90, .99 if default_pass else .80),
                              'paper128': (.96, .99), 'learning_rate_3e5': (.94, .999)}
                    validation, test = values[name]
                    return result(validation, test), base/name

                with patch.object(runner, 'check_data', return_value={}), patch.object(runner, 'run_attempt', side_effect=fake_attempt):
                    state = runner.process_dataset(options, 'ICEWS1819', 0)
                self.assertEqual(state['status'], 'passed')
                self.assertEqual(calls, ['default'] if default_pass else ['default', 'paper128', 'learning_rate_3e5'])
                self.assertEqual(state['selected']['name'], 'default' if default_pass else 'paper128')

    def test_yelp_pooler_features_keep_ids_and_zero_padding(self):
        with tempfile.TemporaryDirectory() as temporary:
            base = Path(temporary)
            model = base/'bert'
            model.mkdir()
            (model/'pytorch_model.bin').write_bytes(b'unit-test fixture, no model weights')
            archive = base/'Yelp.zip'
            with zipfile.ZipFile(archive, 'w') as package:
                package.writestr('Yelp/edge_list.csv', 'u,i,r,ts,label\n1,2,1,1,0\n2,3,2,2,1\n3,1,3,3,0\n')
                for name in ['entity_text.csv', 'relation_text.csv']:
                    package.writestr('Yelp/'+name, 'i,text\n0,padding 0\n1,item 1\n2,item 2\n3,item 3\n')

            class Encoded(dict):
                def to(self, device):
                    return self

            class Tokenizer:
                def __call__(self, text, **kwargs):
                    return Encoded(input_ids=torch.tensor([[int(value.split()[-1])] for value in text], dtype=torch.float32))

            class Model:
                config = SimpleNamespace(hidden_size=768)

                def to(self, device):
                    return self

                def eval(self):
                    return self

                def __call__(self, input_ids):
                    return None, input_ids.repeat(1, 768)

            with patch('transformers.AutoTokenizer.from_pretrained', return_value=Tokenizer()), \
                 patch('transformers.AutoModel.from_pretrained', return_value=Model()):
                preparation.prepare(base/'data', archive, model, 0, batch_size=2)
            for name in ['e_feat.npy', 'r_feat.npy']:
                features = np.load(base/'data/Yelp'/name)
                self.assertEqual(features.shape, (4, 768))
                self.assertTrue(np.all(features[0] == 0))
                for index in [1, 2, 3]:
                    self.assertTrue(np.all(features[index] == index))
            status = json.loads((base/'data/Yelp/preparation_status.json').read_text())
            self.assertEqual(status['status'], 'ready')


if __name__ == '__main__':
    unittest.main()
