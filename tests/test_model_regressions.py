import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from models import NeuralPLSI, SplinePLSI
from models.native import outcome_loss_gradient, training_validation_indices
from models.PLSI import _cox_loss_and_grad, _binary_fit
from simulation import simulate_data


class ModelRegressionTest(unittest.TestCase):
    def test_extreme_tied_cox_scores_and_gradients(self):
        risk = np.array([-1000., -900., 900., 1000., 999.])
        time, event = np.array([5., 4., 3., 3., 1.]), np.array([1., 1., 0., 1., 1.])
        expected = sum(logsumexp(risk[time >= t])-r for t, e, r in zip(time, event, risk) if e)
        ends = np.array([0, 1, 3, 4])
        value, grad = _cox_loss_and_grad(np.eye(5), event, risk, 0., group_ends=ends)
        np.testing.assert_allclose(value, expected)
        eps = 1e-4
        numeric = []
        for j in range(5):
            shift = np.eye(5)[j]*eps
            f = lambda w: _cox_loss_and_grad(np.eye(5), event, w, 0., group_ends=ends)[0]
            numeric.append((f(risk+shift)-f(risk-shift))/(2*eps))
        np.testing.assert_allclose(grad, numeric, atol=2e-8)
        loss, gradient = outcome_loss_gradient(risk, np.column_stack([time, event]), 'cox')
        np.testing.assert_allclose(loss, expected/event.sum())
        np.testing.assert_allclose(gradient, grad/event.sum(), atol=1e-7)

    def test_final_spline_components_use_returned_index(self):
        x, z, y, *_ = simulate_data(60, g_type='linear', seed=10)
        model = SplinePLSI(max_iter=1)
        start = np.array([0., 1., 0., 0., 0., 0., 0., 0.])
        proposed = np.array([1., .7, -.5, .5, .3, -.1, 0., 0.])
        proposed /= np.linalg.norm(proposed)
        with patch.object(model, '_optimize_beta', return_value=proposed):
            model.fit(x, z, y, beta_init=start)
        np.testing.assert_allclose(model.beta, proposed)
        gamma, spline, knots, loss = model._fit_linear_part(x, z, y, model.beta)
        np.testing.assert_allclose(model.gamma, gamma)
        np.testing.assert_allclose(model.spline_coeffs, spline)
        np.testing.assert_allclose(model.knot_vector, knots)
        self.assertAlmostEqual(model.loss_, loss)

    def test_binary_profile_and_final_fit_use_same_penalty(self):
        x, z, y, *_ = simulate_data(70, outcome='binary', seed=3)
        model = SplinePLSI(family='binary', max_iter=1, alpha=.003).fit(x, z, y, random_state=3)
        self.assertAlmostEqual(model._objective_factory(x, z, y)(model.beta), model.loss_, places=8)
        basis, _ = model._construct_spline_basis(x@model.beta)
        coefficient, loss = _binary_fit(np.column_stack([z, basis]), y, model.alpha)
        np.testing.assert_allclose(coefficient, np.r_[model.gamma, model.spline_coeffs])

    def test_serial_parallel_bootstrap_and_refit_clear_uncertainty(self):
        x, z, y, *_ = simulate_data(50, seed=1)
        for model in (SplinePLSI(max_iter=1), NeuralPLSI(max_epoch=2, hidden_units=4)):
            with self.subTest(model=type(model).__name__, backend=getattr(model, 'backend', None)):
                model.fit(x, z, y, random_state=5)
                serial = model.inference_bootstrap(x, z, y, n_samples=3, n_jobs=1, random_state=13, g_grid=[-1., 1.])
                parallel = model.inference_bootstrap(x, z, y, n_samples=3, n_jobs=2, random_state=13, g_grid=[-1., 1.])
                for key in ('beta_samples', 'gamma_samples', 'g_samples'):
                    np.testing.assert_array_equal(serial[key], parallel[key])
                model.fit(x, z, y, random_state=5)
                self.assertFalse(hasattr(model, 'beta_se'))
                self.assertFalse(hasattr(model, 'g_grid'))

    def test_linear_initialization_preserves_linear_predictor(self):
        from sklearn.linear_model import LinearRegression
        x, z, y, *_ = simulate_data(50, seed=2)
        for intercept in (False, True):
            model = NeuralPLSI(initial=True, max_epoch=0, add_intercept=intercept).fit(x, z, y, random_state=4)
            train, _ = training_validation_indices(len(x), 4)
            design = np.column_stack([x, z]).astype('float32').astype(float)
            reference = LinearRegression(fit_intercept=intercept).fit(design[train], y[train].astype('float32'))
            np.testing.assert_allclose(model.predict(x,z), reference.predict(design), rtol=1e-5, atol=2e-6)

    def test_binary_and_cox_initialization_match_independent_optimization(self):
        from scipy.optimize import minimize
        from scipy.special import expit
        rng = np.random.default_rng(24)
        x, z = rng.normal(size=(160, 3)), rng.normal(size=(160, 2))
        x, z = x.astype('float32'), z.astype('float32')
        train, _ = training_validation_indices(len(x), 4)
        for family, intercept in [('binary', False), ('binary', True), ('cox', False)]:
            with self.subTest(family=family, intercept=intercept):
                y = rng.binomial(1, expit(.5*x[:, 0] + .3*z[:, 0]))
                if family == 'cox':
                    y = np.column_stack([rng.integers(1, 20, len(x)), rng.binomial(1, .7, len(x))])
                design = np.column_stack([x, z]).astype(float)
                if intercept:
                    design = np.column_stack([design, np.ones(len(x))])
                d, target = design[train], y[train]
                penalized = np.ones(design.shape[1])
                if intercept:
                    penalized[-1] = 0

                def objective(coefficient):
                    eta = d @ coefficient
                    value = .5e-6 * np.dot(penalized * coefficient, coefficient)
                    score = np.zeros(len(eta))
                    if family == 'binary':
                        value += np.sum(np.logaddexp(0., eta) - target*eta)
                        score = expit(eta) - target
                    else:
                        # Direct complete risk sets, independent of the native
                        # sorted cumulative-moment Newton implementation.
                        for i in np.flatnonzero(target[:, 1]):
                            risk = target[:, 0] >= target[i, 0]
                            denominator = logsumexp(eta[risk])
                            value += denominator - eta[i]
                            score[risk] += np.exp(eta[risk] - denominator)
                            score[i] -= 1.
                    return value, d.T @ score + 1e-6 * penalized * coefficient

                reference = minimize(objective, np.zeros(design.shape[1]), jac=True,
                                     method='BFGS', options={'gtol': 1e-7})
                self.assertLess(np.max(np.abs(objective(reference.x)[1])), 1e-5)
                model = NeuralPLSI(family=family, initial=True, max_epoch=0,
                                   add_intercept=intercept).fit(x, z, y, random_state=4)
                np.testing.assert_allclose(model.predict(x, z), design @ reference.x,
                                           rtol=1e-5, atol=2e-6)

    def test_simulation_does_not_mutate_global_rng(self):
        np.random.seed(123)
        state = copy.deepcopy(np.random.get_state())
        for family in ('continuous', 'binary', 'cox'):
            simulate_data(30, outcome=family, seed=9)
        actual = np.random.rand(10)
        np.random.set_state(state)
        np.testing.assert_array_equal(actual, np.random.rand(10))

    def test_invalid_data_and_unidentified_intercept_fail(self):
        with self.assertRaisesRegex(ValueError, 'intercept'):
            NeuralPLSI(family='cox', add_intercept=True)
        x, z, y, *_ = simulate_data(20)
        x[0, 0] = np.nan
        for model in (NeuralPLSI(max_epoch=1), SplinePLSI(max_iter=1)):
            with self.assertRaisesRegex(ValueError, 'finite'):
                model.fit(x, z, y)


if __name__ == '__main__':
    unittest.main()
