import json
from pathlib import Path
import tempfile
import unittest

from reproduce.main_simulation import load_compatible_results


class SimulationResumeTest(unittest.TestCase):
    def test_resume_rejects_mixed_or_legacy_outputs_and_duplicate_seeds(self):
        cfg = {'activation': 'ELU', 'n_bootstrap': 100, 'source_sha256': 'new',
               'quantile_method': 'linear',
               'training_policy': 'full_batch_lbfgs_exact_gamma_v1'}
        valid = [{'seed': 0, 'run_configuration': cfg}]
        invalid = [
            [{'seed': 0}],
            [{'seed': 0, 'run_configuration': dict(cfg, activation='Tanh')}],
            [{'seed': 0, 'run_configuration': dict(cfg, n_bootstrap=49)}],
            [{'seed': 0, 'run_configuration': dict(cfg, quantile_method='weibull')}],
            [{'seed': 0, 'run_configuration': dict(cfg, source_sha256='old')}],
            [{'seed': 0, 'run_configuration': dict(cfg, training_policy='early_stopping')}],
            [{'seed': 0, 'run_configuration': dict(cfg, training_policy='full_budget_validation_checkpoint_exact_gamma_v1')}],
            valid + valid,
        ]
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'results.json'
            path.write_text(json.dumps(valid))
            self.assertEqual(load_compatible_results(path, cfg), valid)
            for rows in invalid:
                with self.subTest(rows=rows):
                    path.write_text(json.dumps(rows))
                    with self.assertRaises(ValueError):
                        load_compatible_results(path, cfg)
                    self.assertEqual(json.loads(path.read_text()), rows)


if __name__ == '__main__':
    unittest.main()
