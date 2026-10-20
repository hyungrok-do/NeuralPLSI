import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.NeuralPLSI import NeuralPLSI, _nPLSInet


class BootstrapQuantilesTest(unittest.TestCase):
    def test_interpolation_applies_to_all_blocks_without_changing_draws(self):
        """For ordered draws 1..100, endpoints have hand-computable ranks."""
        model = NeuralPLSI(add_intercept=True)
        model.net = _nPLSInet(2, 1, add_intercept=True).to(model.device)
        sample = np.arange(1., 101.)
        draws = [(np.array([v, -v]), np.array([2*v]), np.array([3*v]),
                  np.array([4*v, 5*v])) for v in sample]
        X, Z, y = np.zeros((5, 2)), np.zeros((5, 1)), np.zeros(5)
        with patch('models.NeuralPLSI.run_parallel_bootstrap', return_value=draws):
            default = model.inference_bootstrap(X, Z, y, n_samples=100,
                                                n_jobs=1, g_grid=[0, 1])
            weibull = model.inference_bootstrap(X, Z, y, n_samples=100,
                n_jobs=1, g_grid=[0, 1], quantile_method='weibull')
        # One-based ranks: 1+(B-1)p versus (B+1)p.
        np.testing.assert_allclose(default['beta_lb'], [3.475, -97.525])
        np.testing.assert_allclose(default['beta_ub'], [97.525, -3.475])
        np.testing.assert_allclose(weibull['beta_lb'], [2.525, -98.475])
        np.testing.assert_allclose(weibull['beta_ub'], [98.475, -2.525])
        for block, factors in [('gamma', [2]), ('intercept', [3]), ('g', [4, 5])]:
            np.testing.assert_allclose(weibull[block+'_lb'], 2.525*np.array(factors))
            np.testing.assert_allclose(weibull[block+'_ub'], 98.475*np.array(factors))
        for key in ['beta_samples','gamma_samples','intercept_samples','beta_se',
                    'gamma_se','intercept_se','g_se','beta_hat','gamma_hat']:
            np.testing.assert_array_equal(default[key], weibull[key])
        self.assertEqual(default['quantile_method'], 'linear')
        self.assertEqual(weibull['quantile_method'], 'weibull')

    def test_invalid_interpolation_fails_before_fitting(self):
        model = NeuralPLSI()
        with self.assertRaisesRegex(ValueError, 'quantile_method'):
            model.inference_bootstrap(None, None, None, quantile_method='typo')
        self.assertIsNone(model.net)


if __name__ == '__main__':
    unittest.main()
