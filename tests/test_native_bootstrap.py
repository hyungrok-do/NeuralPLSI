import contextlib
import os
import unittest
from unittest.mock import patch

import numpy as np

from models import NeuralPLSI
from models.native import NativeModel, bootstrap_indices, bootstrap_options, library
from simulation import simulate_data


class NativeBootstrapTest(unittest.TestCase):
    def test_draws_match_explicit_refits_and_do_not_depend_on_worker_count(self):
        for family in ('continuous', 'binary', 'cox'):
            x, z, y, *_ = simulate_data(120, outcome=family, seed=5)
            if family == 'cox':
                # Different resamples must rebuild cached tied risk sets.
                y[:, 0] = np.ceil(y[:, 0]*2)
            for initial in (False, True):
                with self.subTest(family=family, initial=initial):
                    model = NeuralPLSI(family=family, max_iter=31, hidden_units=5,
                        n_hidden_layers=3, initial=initial,
                        add_intercept=family!='cox').fit(x, z, y, random_state=4)
                    before = model.net.to_bytes()
                    grid = np.linspace(-2, 2, 7)
                    a = model.inference_bootstrap(x, z, y, n_samples=4, random_state=9, n_jobs=1, g_grid=grid)
                    b = model.inference_bootstrap(x, z, y, n_samples=4, random_state=9, n_jobs=3, g_grid=grid)
                    for key in ('beta_samples', 'gamma_samples', 'g_samples', 'beta_se', 'g_lb'):
                        np.testing.assert_array_equal(a[key], b[key])
                    # Independently fit each actual resampled dataset through the
                    # single-fit API to check seeds, initialization and orientation.
                    for r in range(4):
                        rows = bootstrap_indices(len(x), 9, r)
                        fitted = NativeModel.from_bytes(before)
                        fitted.fit(x[rows], z[rows], y[rows], random_state=9+1337+r)
                        self.assertTrue(0 < fitted.fit_diagnostics['iterations'] <= 31)
                        self.assertGreater(fitted.fit_diagnostics['gamma_iterations'],0)
                        beta, gamma, intercept = fitted.coefficients()
                        np.testing.assert_array_equal(a['beta_samples'][r], beta)
                        np.testing.assert_array_equal(a['gamma_samples'][r], gamma)
                        np.testing.assert_array_equal(a['g_samples'][r], fitted.g_function(grid))
                        if intercept is not None:
                            np.testing.assert_array_equal(a['intercept_samples'][r], intercept)
                    self.assertEqual(before, model.net.to_bytes())
                    self.assertEqual(a['n_workers'], 1)
                    self.assertEqual(b['n_workers'], library().nplsi_bootstrap_workers(3, 4))

    def test_bootstrap_has_no_python_refits_rng_statistics_or_worker_creation(self):
        x, z, y, *_ = simulate_data(80, seed=3)
        model = NeuralPLSI(max_iter=2, hidden_units=4).fit(x, z, y)
        # The shared library is already loaded. Any of these calls would mean
        # that the bootstrap left the single native invocation.
        forbidden = ('models.base.run_parallel_bootstrap', 'joblib.Parallel',
                     'subprocess.Popen', 'multiprocessing.Process.start', 'threading.Thread.start',
                     'numpy.random.default_rng', 'numpy.percentile', 'numpy.quantile')
        with contextlib.ExitStack() as stack:
            for name in forbidden:
                stack.enter_context(patch(name, side_effect=AssertionError(name)))
            for cls, method in ((NeuralPLSI, 'fit'), (NativeModel, 'fit'), (NativeModel, 'from_bytes')):
                stack.enter_context(patch.object(cls, method, side_effect=AssertionError(method)))
            native = stack.enter_context(patch.object(library(), 'nplsi_bootstrap', wraps=library().nplsi_bootstrap))
            result = model.inference_bootstrap(x, z, y, n_samples=4, n_jobs=2, g_grid=[-1, 0, 1])
            self.assertEqual(native.call_count, 1)
        self.assertEqual(result['bootstrap_engine'], 'cpp')
        self.assertEqual(result['bootstrap_rng'], 'splitmix64_bootstrap_v1')
        self.assertTrue(np.isfinite(result['gamma_se']).all())

    def test_native_rng_golden_draw_prefix_and_numpy_state(self):
        np.testing.assert_array_equal(bootstrap_indices(10, 0, 0), [8, 5, 0, 2, 4, 2, 7, 1, 1, 7])
        x, z, y, *_ = simulate_data(80, seed=3)
        model = NeuralPLSI(max_iter=2, hidden_units=4).fit(x, z, y)
        np.random.seed(44)
        before = np.random.get_state()
        a = model.inference_bootstrap(x, z, y, n_samples=2, n_jobs=1)
        b = model.inference_bootstrap(x, z, y, n_samples=5, n_jobs=2)
        after = np.random.get_state()
        np.testing.assert_array_equal(before[1], after[1])
        self.assertEqual(before[2:], after[2:])
        for block in ('beta', 'gamma'):
            np.testing.assert_array_equal(a[block+'_samples'], b[block+'_samples'][:2])

    def test_worker_limits_seed_validation_and_failure_propagation(self):
        with patch.dict(os.environ, {'SLURM_CPUS_PER_TASK': '1'}):
            for jobs in (-1, -2, 1, 32):
                self.assertEqual(library().nplsi_bootstrap_workers(jobs, 5), 1)
        for options in [(2, 2**64-1, .95, 1, 'linear'), (2, 0, .95, 0, 'linear'),
                        (2, 0, 1., 1, 'linear'), (1, 0, .95, 1, 'linear')]:
            with self.assertRaises(ValueError): bootstrap_options(*options)
        # Cold linear initialization cannot identify an exposure direction on
        # all-zero X. Fail visibly inside a worker instead of omitting the draw.
        model = NeuralPLSI(max_iter=0, initial=True)
        model.net = NativeModel(2, 0, max_iter=0, initial=True)
        state = model.net.to_bytes()
        x, z, y = np.zeros((20, 2)), np.empty((20, 0)), np.arange(20.)
        for jobs in (1, 3):
            model.beta_se = np.ones(2)
            with self.assertRaisesRegex(RuntimeError, r'Bootstrap replicate [0-9]+ \(fit seed [0-9]+\) failed:'):
                model.inference_bootstrap(x, z, y, n_samples=4, n_jobs=jobs)
            self.assertEqual(model.net.to_bytes(), state)
            self.assertFalse(hasattr(model, 'beta_se'))

    def test_zero_covariates_and_optional_intercept(self):
        rng = np.random.default_rng(1)
        x, z, y = rng.normal(size=(60, 2)), np.empty((60, 0)), rng.normal(size=60)
        for intercept in (False, True):
            model = NeuralPLSI(max_iter=2, hidden_units=3, add_intercept=intercept).fit(x, z, y)
            result = model.inference_bootstrap(x, z, y, n_samples=3, n_jobs=2)
            self.assertEqual(result['gamma_samples'].shape, (3, 0))
            self.assertEqual(result['gamma_se'].shape, (0,))
            self.assertEqual('intercept_samples' in result, intercept)


if __name__ == '__main__':
    unittest.main()
