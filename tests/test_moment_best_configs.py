import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

from utils.load_configs import get_link_prediction_args


ROOT = Path(__file__).resolve().parents[1]


def parse(*arguments):
    with patch.object(sys, 'argv', ['train_link_prediction.py', '--gpu', '-1'] + list(arguments)):
        return get_link_prediction_args()


class DatasetConfigurationTests(unittest.TestCase):
    def test_original_model_is_restored_exactly(self):
        self.assertEqual(hashlib.sha256((ROOT/'models/MoMent.py').read_bytes()).hexdigest(),
                         'eb182f1205566957d548ace13d93965b7618eb05a559e81735d266248c38ccc0')

    def test_all_parser_datasets_have_an_explicit_configuration(self):
        table = json.loads((ROOT/'configs/moment_link_prediction.json').read_text())
        self.assertEqual(len(table['paper_targets']), 7)
        for dataset in table['datasets']:
            with self.subTest(dataset=dataset):
                args = parse('--dataset_name', dataset)
                expected = dict(table['defaults'], **table['datasets'][dataset])
                for name, value in expected.items():
                    self.assertEqual(getattr(args, name), value)
                self.assertEqual(args.num_runs, 1)
                self.assertEqual(args.num_epochs, 100)

    def test_accepted_enron_prefers_repository_defaults(self):
        args = parse('--dataset_name', 'Enron')
        self.assertEqual((args.time_feat_dim, args.loss_weight), (100, .2))
        self.assertEqual(vars(parse()), vars(args))

    def test_accepted_stack_configuration_is_preserved(self):
        args = parse('--dataset_name', 'Stack_elec')
        self.assertEqual((args.time_feat_dim, args.loss_weight), (100, .1))

    def test_ubuntu_uses_original_dataset_loss_weight(self):
        evidence = json.loads((ROOT/'configs/moment_verification.json').read_text())['datasets']['Stack_ubuntu']
        expected = .5 if evidence['status'] == 'pending_single_seed' else evidence['parameters'].get('loss_weight', .2)
        self.assertEqual(parse('--dataset_name', 'Stack_ubuntu').loss_weight, expected)

    def test_original_flag_and_short_alias_are_equivalent(self):
        self.assertEqual(vars(parse('--dataset_name', 'Yelp', '--load_best_configs')),
                         vars(parse('--dataset_name', 'Yelp', '--best-configs')))

    def test_explicit_run_count_is_not_overwritten(self):
        self.assertEqual(parse('--dataset_name', 'ICEWS1819', '--num_runs', '3').num_runs, 3)

    def test_manual_parameters_can_bypass_dataset_overrides(self):
        args = parse('--dataset_name', 'Stack_elec', '--no_best_configs',
                     '--time_feat_dim', '64', '--learning_rate', '.0003', '--loss_weight', '.7')
        self.assertEqual((args.time_feat_dim, args.learning_rate, args.loss_weight), (64, .0003, .7))

    def test_help_exits_successfully(self):
        with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse('--help')
        self.assertEqual(error.exception.code, 0)

    def test_invalid_dataset_is_not_silently_swallowed(self):
        with contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            parse('--dataset_name', 'missing_dataset')
        self.assertEqual(error.exception.code, 2)

    def test_other_models_keep_their_model_specific_parameters(self):
        args = parse('--dataset_name', 'Enron', '--model_name', 'TGAT')
        self.assertEqual((args.num_layers, args.num_neighbors), (2, 20))
        self.assertEqual(args.use_feature, '')

    def test_loaded_parameters_match_recorded_seed0_experiments(self):
        recorded = json.loads((ROOT/'reproduction/seed0_results.json').read_text())
        for dataset, evidence in recorded['results'].items():
            for flags in [[], ['--best-configs'], ['--load_best_configs']]:
                with self.subTest(dataset=dataset, flags=flags):
                    args = parse('--dataset_name', dataset, *flags)
                    for name, value in evidence['fixed_parameters'].items():
                        self.assertEqual(getattr(args, name), value)
                    self.assertEqual(args.num_runs, recorded['num_runs_per_group'])

    def test_recorded_comparisons_do_not_claim_yelp_passed(self):
        table = json.loads((ROOT/'configs/moment_link_prediction.json').read_text())
        recorded = json.loads((ROOT/'reproduction/seed0_results.json').read_text())
        passed = []
        for dataset, evidence in recorded['results'].items():
            target = table['paper_targets'][dataset]
            within = all(abs(values['value'] - target[metric]) <= .02 + 1e-12
                         for metric, values in evidence['comparison'].items())
            self.assertEqual(evidence['within_2_percentage_points'], within)
            if within:
                passed.append(dataset)
        self.assertEqual(len(passed), 6)
        self.assertNotIn('Yelp', passed)

    def test_selected_parameters_follow_recorded_validation_scores(self):
        table = json.loads((ROOT/'configs/moment_link_prediction.json').read_text())
        recorded = json.loads((ROOT/'reproduction/seed0_results.json').read_text())
        for dataset, evidence in recorded['results'].items():
            if 'attempts' not in evidence:
                continue
            attempts = evidence['attempts']
            expected = attempts[0] if attempts[0]['comparison']['within_2_percentage_points'] else max(
                attempts, key=lambda attempt: attempt['validation_score'])
            with self.subTest(dataset=dataset):
                self.assertEqual(evidence['selected_attempt'], expected['name'])
                self.assertEqual(table['datasets'][dataset], expected['parameters'])

    def test_training_evaluation_and_data_sources_remain_original(self):
        provenance = json.loads((ROOT/'reproduction/source_provenance.json').read_text())
        for name, expected in provenance['unchanged_source_sha256'].items():
            with self.subTest(source=name):
                self.assertEqual(hashlib.sha256((ROOT/name).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
