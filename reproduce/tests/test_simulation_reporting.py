import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from reproduce.main_simulation import atomic_json, main, result_lock
from reproduce.result_io import load_results, compute_summaries, curves, true_g
from neuralplsi.simulation import beta, gamma


class SimulationReportingTest(unittest.TestCase):
    def test_checkpoint_survives_serialization_error_and_replicate_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'checkpoint.json'
            atomic_json(path, [{'seed': 0}])
            with self.assertRaises(ValueError):
                atomic_json(path, [{'seed': float('nan')}])
            self.assertEqual(json.loads(path.read_text()), [{'seed': 0}])
            def replicate(args, model, seed, configuration):
                if seed == 1:
                    raise FloatingPointError('deliberate failure')
                return dict(seed=seed, run_configuration=configuration)
            with patch('reproduce.main_simulation.run_replicate', side_effect=replicate):
                with self.assertRaisesRegex(RuntimeError, 'seed 1'):
                    main(['--models', 'NeuralPLSI', '--out', temp, '--n_replicates', '3', '--save_every', '10'])
            files = list(Path(temp).glob('simulation+*.json'))
            self.assertEqual(len(files), 1)
            self.assertEqual([r['seed'] for r in json.loads(files[0].read_text())], [0])

    def test_concurrent_writer_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'results.json'
            with result_lock(path):
                with self.assertRaisesRegex(RuntimeError, 'Another job'):
                    with result_lock(path):
                        pass

    def fixture(self, seed=0, n=1000):
        grid = [-2., -.4, .7, 3.]
        return dict(model='NeuralPLSI', n=n, g_fn='linear', outcome='cox', x_dist='normal',
            seed=seed, beta_est=beta.tolist(), gamma_est=(gamma+.1).tolist(),
            beta_se=[.1]*8, gamma_se=[.2]*3,
            beta_lb=(beta-.2).tolist(), beta_ub=(beta+.2).tolist(),
            gamma_lb=(gamma-.2).tolist(), gamma_ub=(gamma+.3).tolist(),
            # Stale precomputed coverage must not override the stored endpoints.
            beta_cov=[0]*8, gamma_cov=[0]*3,
            g_grid=grid, g_pred=(np.array(grid)+3).tolist(),
            run_configuration=dict(source_sha256='test', n_bootstrap=100, quantile_method='linear', n=n))

    def test_actual_intervals_grid_and_sample_sizes(self):
        with tempfile.TemporaryDirectory() as temp:
            for n in (1000, 2000):
                Path(temp, f'simulation+{n}.json').write_text(json.dumps([self.fixture(0,n), self.fixture(1,n)]))
            records = load_results(temp)
            table = compute_summaries(records)
            self.assertEqual(set(table['n']), {'1000', '2000'})
            np.testing.assert_allclose(table.coverage, 1.)
            self.assertEqual(len(table), 22)
            grid, values = curves(records[0])
            np.testing.assert_allclose(values[0], grid, atol=1e-15)
            np.testing.assert_allclose(true_g(grid, 'linear'), grid)

    def test_mixed_configs_and_duplicate_scenarios_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp, 'simulation+one.json')
            rows = [self.fixture(), self.fixture(1)]
            rows[1]['run_configuration']['quantile_method'] = 'weibull'
            path.write_text(json.dumps(rows))
            with self.assertRaisesRegex(ValueError, 'Mixed run'):
                load_results(temp)
            path.write_text(json.dumps([self.fixture()]))
            Path(temp, 'simulation+two.json').write_text(path.read_text())
            with self.assertRaisesRegex(ValueError, 'Multiple runs'):
                load_results(temp)


if __name__ == '__main__':
    unittest.main()
