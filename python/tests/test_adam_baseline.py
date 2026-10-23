import pickle
import tempfile
import unittest
from pathlib import Path

import numpy as np

from neuralplsi import NeuralPLSI
from neuralplsi.native import NativeModel, bootstrap_indices, outcome_loss_gradient
from neuralplsi.simulation import simulate_data


def baseline(family, **options):
    settings = dict(optimizer='adam', max_iter=60, batch_size=16, patience=3, hidden_units=4,
                    n_hidden_layers=1, add_intercept=family != 'cox')
    settings.update(options)
    return NeuralPLSI(family=family, **settings)


class AdamBaselineTest(unittest.TestCase):
    def test_early_stopping_restores_the_best_epoch_without_a_gamma_fit(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(150, outcome=family, seed=8)
                stopped = baseline(family).fit(x, z, y, random_state=2)
                d = stopped.fit_diagnostics
                self.assertEqual(d['gamma_iterations'], 0)
                self.assertTrue(0 < d['best_epoch'] <= d['iterations'] <= 60)
                if d['iterations'] < 60:
                    self.assertEqual(d['iterations']-d['best_epoch'], 3)
                # Training only to the selected epoch follows the same trajectory,
                # so its last epoch is also its best checkpoint.
                truncated = baseline(family, max_iter=d['best_epoch'], patience=1000).fit(x, z, y, random_state=2)
                np.testing.assert_array_equal(stopped.net.weights, truncated.net.weights)
                self.assertEqual(d['objective'], truncated.fit_diagnostics['objective'])
                self.assertEqual(truncated.fit_diagnostics['best_epoch'], d['best_epoch'])

    def test_initialization_only_and_settings_round_trip(self):
        x, z, y, *_ = simulate_data(80, outcome='continuous', seed=3)
        model = baseline('continuous', max_iter=0).fit(x, z, y, random_state=5)
        initial = NativeModel(x.shape[1], z.shape[1], hidden_units=4, n_hidden_layers=1,
                              add_intercept=True, random_state=5)
        initial.resolve_sign_ambiguity()
        np.testing.assert_array_equal(model.net.weights, initial.weights)
        self.assertEqual((model.fit_diagnostics['iterations'], model.fit_diagnostics['best_epoch']), (0, 0))
        model = baseline('continuous', learning_rate=3e-3, weight_decay=1e-2).fit(x, z, y, random_state=5)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'model.nplsi'
            model.save(path)
            restored = NeuralPLSI.load(path)
        # The native state stores the rates in float32.
        params, loaded = model.get_params(), restored.get_params()
        for name in ('learning_rate', 'weight_decay'):
            self.assertEqual(np.float32(loaded.pop(name)), np.float32(params.pop(name)))
        self.assertEqual(loaded, params)
        self.assertEqual(restored.fit_diagnostics, model.fit_diagnostics)
        np.testing.assert_array_equal(restored.predict(x, z), model.predict(x, z))
        np.testing.assert_array_equal(pickle.loads(pickle.dumps(model)).predict(x, z), model.predict(x, z))

    def test_bootstrap_refits_use_the_same_baseline_procedure(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(120, outcome=family, seed=5)
                model = baseline(family).fit(x, z, y, random_state=4)
                before = model.net.to_bytes()
                a = model.inference_bootstrap(x, z, y, n_samples=3, random_state=9, n_jobs=1, g_grid=[-1., 0., 1.])
                b = model.inference_bootstrap(x, z, y, n_samples=3, random_state=9, n_jobs=3, g_grid=[-1., 0., 1.])
                for key in ('beta_samples', 'gamma_samples', 'g_samples'):
                    np.testing.assert_array_equal(a[key], b[key])
                for r in range(3):
                    rows = bootstrap_indices(len(x), 9, r)
                    fitted = NativeModel.from_bytes(before)
                    fitted.fit(x[rows], z[rows], y[rows], random_state=9+1337+r)
                    self.assertEqual(fitted.fit_diagnostics['gamma_iterations'], 0)
                    beta, gamma, _ = fitted.coefficients()
                    np.testing.assert_array_equal(a['beta_samples'][r], beta)
                    np.testing.assert_array_equal(a['gamma_samples'][r], gamma)

    def test_exact_gamma_option_refits_only_gamma_on_all_rows(self):
        grid = np.linspace(-2, 2, 9)
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(200, outcome=family, seed=4)
                x, z, y = (a.astype('float32') for a in (x, z, y))
                base = baseline(family).fit(x, z, y, random_state=2)
                exact = baseline(family, optimizer='adam_exact_gamma').fit(x, z, y, random_state=2)
                # The AdamW fit is shared: beta, the link and the selected epoch are identical.
                np.testing.assert_array_equal(exact.beta, base.beta)
                np.testing.assert_array_equal(exact.g_function(grid), base.g_function(grid))
                d, e = base.fit_diagnostics, exact.fit_diagnostics
                self.assertEqual((e['iterations'], e['best_epoch'], e['objective']),
                                 (d['iterations'], d['best_epoch'], d['objective']))
                self.assertEqual(d['gamma_iterations'], 0)
                self.assertGreater(e['gamma_iterations'], 0)
                self.assertFalse(np.array_equal(exact.gamma, base.gamma))
                # Gamma and the intercept are the exact fit on all rows, given the index and link.
                g = exact.predict_gxb(x).astype(float)
                coefficients = np.append(exact.gamma, exact.intercept_val if family != 'cox' else [])
                if family == 'continuous':
                    design = np.column_stack([z, np.ones(len(x))])
                    expected = np.linalg.lstsq(design, y-g, rcond=None)[0]
                    np.testing.assert_allclose(coefficients, expected, atol=1e-5)
                else:
                    _, delta = outcome_loss_gradient(exact.predict(x, z), y, family)
                    design = z if family == 'cox' else np.column_stack([z, np.ones(len(x))])
                    np.testing.assert_allclose(design.T@delta.astype(float), 0., atol=1e-5)
                # Bootstrap refits follow the same procedure.
                before = exact.net.to_bytes()
                draws = exact.inference_bootstrap(x, z, y, n_samples=3, random_state=9, n_jobs=2)
                for r in range(3):
                    rows = bootstrap_indices(len(x), 9, r)
                    fitted = NativeModel.from_bytes(before)
                    fitted.fit(x[rows], z[rows], y[rows], random_state=9+1337+r)
                    self.assertGreater(fitted.fit_diagnostics['gamma_iterations'], 0)
                    np.testing.assert_array_equal(draws['gamma_samples'][r], fitted.coefficients()[1])

    def test_invalid_settings(self):
        for options in (dict(optimizer='sgd'), dict(optimizer='adam_gamma'), dict(learning_rate=0.), dict(patience=0), dict(batch_size=0)):
            with self.subTest(**options):
                with self.assertRaises(ValueError):
                    baseline('continuous', **options)


if __name__ == '__main__':
    unittest.main()
