import unittest
import numpy as np
from models import NeuralPLSI
from models.native import bootstrap_summary


class BootstrapQuantilesTest(unittest.TestCase):
    def test_native_interpolation_has_hand_computable_ranks(self):
        sample = np.arange(1., 101.)
        draws = sample[:, None] * np.array([1., -1., 2., 3., 4., 5.])
        default = bootstrap_summary(draws)
        weibull = bootstrap_summary(draws, quantile_method='weibull')
        np.testing.assert_allclose(default[2, :2], [3.475, -97.525])
        np.testing.assert_allclose(default[3, :2], [97.525, -3.475])
        np.testing.assert_allclose(weibull[2, :2], [2.525, -98.475])
        np.testing.assert_allclose(weibull[3, :2], [98.475, -2.525])
        np.testing.assert_allclose(weibull[2, 2:], 2.525*np.array([2., 3., 4., 5.]))
        np.testing.assert_allclose(weibull[3, 2:], 98.475*np.array([2., 3., 4., 5.]))
        np.testing.assert_array_equal(default[:2], weibull[:2])
        np.testing.assert_allclose(default[0], draws.mean(axis=0))
        np.testing.assert_allclose(default[1], draws.std(axis=0, ddof=1))

    def test_native_summaries_match_numpy_with_small_B_ties_and_extreme_values(self):
        rng = np.random.default_rng(17)
        for B in (2, 3, 7, 100):
            draws = rng.integers(-9, 10, size=(B, 4)).astype(float)
            draws[:, 2] = 1e30
            draws[:, 3] *= 1e30
            for ci in (.01, .8, .95, .999):
                for method in ('linear', 'weibull'):
                    with self.subTest(B=B, ci=ci, method=method):
                        result = bootstrap_summary(draws, ci=ci, quantile_method=method)
                        # Compare on the scale of the inputs: a nearly zero
                        # mean can have a large relative cancellation error.
                        scale = np.maximum(1., np.max(np.abs(draws), axis=0))
                        np.testing.assert_allclose(result[0]/scale, draws.mean(axis=0)/scale,
                                                   rtol=0, atol=8*np.finfo(float).eps)
                        # A constant column has exactly zero native variance;
                        # NumPy's unsorted summation can have roundoff at 1e30.
                        np.testing.assert_allclose(result[1, [0, 1, 3]], draws.std(axis=0, ddof=1)[[0, 1, 3]], rtol=2e-15)
                        self.assertEqual(result[1, 2], 0.)
                        expected = np.quantile(draws, [(1-ci)/2, (1+ci)/2], axis=0, method=method)
                        np.testing.assert_allclose(result[2:], expected, rtol=1e-13, atol=1e-14)

    def test_interpolation_applies_to_all_model_blocks_without_changing_draws(self):
        rng = np.random.default_rng(4)
        x, z, y = rng.normal(size=(80, 2)), rng.normal(size=(80, 1)), rng.normal(size=80)
        model = NeuralPLSI(max_epoch=3, hidden_units=4, add_intercept=True).fit(x, z, y)
        outputs = [model.inference_bootstrap(x, z, y, n_samples=8, random_state=9,
                    g_grid=[-1, 0, 1], n_jobs=2, quantile_method=method)
                   for method in ('linear', 'weibull')]
        for block in ('beta', 'gamma', 'intercept', 'g'):
            np.testing.assert_array_equal(outputs[0][block+'_samples'], outputs[1][block+'_samples'])
            np.testing.assert_array_equal(outputs[0][block+'_se'], outputs[1][block+'_se'])
            for result in outputs:
                expected = np.quantile(result[block+'_samples'], [.025, .975], axis=0,
                                       method=result['quantile_method'])
                np.testing.assert_allclose(result[block+'_lb'], expected[0], rtol=1e-12, atol=1e-14)
                np.testing.assert_allclose(result[block+'_ub'], expected[1], rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(getattr(model, ('g_grid' if block=='g' else block)+'_lb'), outputs[1][block+'_lb'])

    def test_invalid_interpolation_fails_before_fitting(self):
        model = NeuralPLSI()
        with self.assertRaisesRegex(ValueError, 'quantile_method'):
            model.inference_bootstrap(None, None, None, quantile_method='typo')
        self.assertIsNone(model.net)


if __name__ == '__main__':
    unittest.main()
