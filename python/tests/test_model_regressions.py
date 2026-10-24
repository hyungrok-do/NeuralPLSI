import copy
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import logsumexp

from neuralplsi import NeuralPLSI, SplinePLSI
from neuralplsi.native import bootstrap_indices, outcome_loss_gradient
from neuralplsi.simulation import simulate_data
from native_reference import cox_loss_and_grad


class ModelRegressionTest(unittest.TestCase):
    def test_extreme_tied_cox_scores_and_gradients(self):
        risk = np.array([-1000., -900., 900., 1000., 999.])
        time, event = np.array([5., 4., 3., 3., 1.]), np.array([1., 1., 0., 1., 1.])
        expected = sum(logsumexp(risk[time >= t])-r for t, e, r in zip(time, event, risk) if e)
        ends = np.array([0, 1, 3, 4])
        value, grad = cox_loss_and_grad(np.eye(5), event, risk, 0., group_ends=ends)
        np.testing.assert_allclose(value, expected)
        eps = 1e-4
        numeric = []
        for j in range(5):
            shift = np.eye(5)[j]*eps
            f = lambda w: cox_loss_and_grad(np.eye(5), event, w, 0., group_ends=ends)[0]
            numeric.append((f(risk+shift)-f(risk-shift))/(2*eps))
        np.testing.assert_allclose(grad, numeric, atol=2e-8)
        loss, gradient = outcome_loss_gradient(risk, np.column_stack([time, event]), 'cox')
        np.testing.assert_allclose(loss, expected/event.sum())
        np.testing.assert_allclose(gradient, grad/event.sum(), atol=1e-7)

    def test_final_spline_components_use_returned_index(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(200, outcome=family, g_type='nlrelu', seed=10)
                # A start with beta[0] < 0 exercises the final sign identification.
                start = -np.array([1., .7, -.5, .5, .3, -.1, 0., 0.])
                model = SplinePLSI(family=family).fit(x, z, y, beta_init=start)
                self.assertGreaterEqual(model.beta[0], 0.)
                self.assertAlmostEqual(np.linalg.norm(model.beta), 1., places=12)
                gamma, spline, knots, loss = model.profile(x, z, y, model.beta)
                np.testing.assert_array_equal(model.gamma, gamma)
                np.testing.assert_array_equal(model.spline_coeffs, spline)
                np.testing.assert_array_equal(model.knot_vector, knots)
                self.assertEqual(model.loss_, loss)

    def test_spline_knots_and_basis_match_independent_references(self):
        from scipy.interpolate import BSpline
        x, z, y, *_ = simulate_data(101, g_type='nlrelu', seed=6)
        model = SplinePLSI().fit(x, z, y, random_state=1)
        eta = np.sort(x @ model.beta)
        knots = eta[np.linspace(0, len(eta)-1, 5).astype(np.int64)]
        np.testing.assert_allclose(model.knot_vector, np.r_[[knots[0]]*3, knots, [knots[-1]]*3],
                                   rtol=0, atol=1e-12)
        t = model.knot_vector
        points = np.r_[np.linspace(t[0]-1, t[-1]+1, 41), t]
        clipped = np.clip(points, t[0], t[-1])
        expected = BSpline.design_matrix(clipped, model.knot_vector, 3).toarray()
        np.testing.assert_allclose(model.basis(points), expected, rtol=0, atol=1e-12)
        np.testing.assert_allclose(model.basis(points).sum(axis=1), 1., atol=1e-12)

    def test_spline_profile_is_the_exact_penalized_fit(self):
        from scipy.optimize import minimize
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(300, outcome=family, g_type='sigmoid', seed=8)
                model = SplinePLSI(family=family, alpha=.003)
                beta = np.array([1., .7, -.5, .5, .3, -.1, 0., 0.]); beta /= np.linalg.norm(beta)
                gamma, spline, knots, loss = model.profile(x, z, y, beta)
                model.beta, model.gamma, model.spline_coeffs, model.knot_vector = beta, gamma, spline, knots
                design = np.column_stack([z, model.basis(x @ beta)])
                coefficient = np.r_[gamma, spline]
                if family == 'continuous':
                    expected = np.linalg.solve(design.T @ design + model.alpha*np.eye(design.shape[1]), design.T @ y)
                    np.testing.assert_allclose(coefficient, expected, rtol=1e-9, atol=1e-11)
                    residual = y - design @ expected
                    self.assertAlmostEqual(loss, residual @ residual + model.alpha*expected @ expected, places=8)
                elif family == 'binary':
                    def objective(w):
                        eta = design @ w
                        return (np.logaddexp(0., eta).sum() - y @ eta + .5*model.alpha*w @ w,
                                design.T @ (1/(1+np.exp(-eta)) - y) + model.alpha*w)
                    expected = minimize(objective, np.zeros(design.shape[1]), jac=True, method='BFGS',
                                        options={'gtol': 1e-10}).x
                    np.testing.assert_allclose(coefficient, expected, rtol=1e-5, atol=1e-6)
                    self.assertAlmostEqual(loss, objective(expected)[0], places=7)
                else:
                    # g is shifted to g(0)=0 after the penalized fit. The basis sums
                    # to one, so a common shift of the spline coefficients leaves the
                    # partial likelihood unchanged, and the ridge solution is the
                    # shift with zero mean spline coefficients.
                    self.assertAlmostEqual(float(model.g_function([0.])[0]), 0., places=12)
                    order = np.argsort(-y[:, 0], kind='stable')
                    times = y[order, 0]
                    ends = np.r_[np.flatnonzero(times[:-1] != times[1:]), len(times)-1]
                    ridge = np.r_[gamma, spline-spline.mean()]
                    value, gradient = cox_loss_and_grad(design[order], y[order, 1], ridge, model.alpha, ends)
                    self.assertAlmostEqual(loss, value, places=7)
                    self.assertLess(np.max(np.abs(gradient)), 1e-6)

    def test_spline_index_is_a_local_minimum_of_the_profile_loss(self):
        rng = np.random.default_rng(0)
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(400, outcome=family, g_type='nlrelu', seed=12)
                model = SplinePLSI(family=family).fit(x, z, y, random_state=2)
                self.assertTrue(model.converged_)
                for _ in range(6):
                    u = rng.normal(size=len(model.beta)); u -= (u @ model.beta)*model.beta
                    u /= np.linalg.norm(u)
                    for sign in (-1, 1):
                        moved = model.beta + sign*.05*u; moved /= np.linalg.norm(moved)
                        self.assertGreater(model.profile(x, z, y, moved)[3], model.loss_)

    def test_spline_bootstrap_draws_match_explicit_refits(self):
        grid = np.linspace(-2, 2, 7)
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y, *_ = simulate_data(150, outcome=family, seed=5)
                if family == 'cox':
                    y[:, 0] = np.ceil(y[:, 0]*2)  # tied times
                model = SplinePLSI(family=family).fit(x, z, y, random_state=4)
                result = model.inference_bootstrap(x, z, y, n_samples=4, random_state=9, n_jobs=3, g_grid=grid)
                for r in range(4):
                    rows = bootstrap_indices(len(x), 9, r)
                    refit = SplinePLSI(family=family).fit(x[rows], z[rows], y[rows], random_state=9+1337+r)
                    np.testing.assert_array_equal(result['beta_samples'][r], refit.beta)
                    np.testing.assert_array_equal(result['gamma_samples'][r], refit.gamma)
                    np.testing.assert_array_equal(result['spline_samples'][r], refit.spline_coeffs)
                    np.testing.assert_allclose(result['g_samples'][r], refit.g_function(grid),
                                               rtol=1e-12, atol=1e-12)
                np.testing.assert_allclose(result['beta_se'], result['beta_samples'].std(axis=0, ddof=1))

    def test_serial_parallel_bootstrap_and_refit_clear_uncertainty(self):
        x, z, y, *_ = simulate_data(50, seed=1)
        for model in (SplinePLSI(max_iter=1), NeuralPLSI(max_iter=2, hidden_units=4)):
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
        x, z, y, *_ = simulate_data(50, seed=2)
        for intercept in (False, True):
            model = NeuralPLSI(initial=True, max_iter=0, add_intercept=intercept).fit(x, z, y, random_state=4)
            design = np.column_stack([x, z]).astype('float32').astype(float)
            if intercept:
                design = np.column_stack([design, np.ones(len(design))])
            coefficients = np.linalg.lstsq(design, y.astype('float32').astype(float), rcond=None)[0]
            np.testing.assert_allclose(model.predict(x,z), design @ coefficients, rtol=1e-5, atol=2e-6)

    def test_binary_and_cox_initialization_match_independent_optimization(self):
        from scipy.optimize import minimize
        from scipy.special import expit
        rng = np.random.default_rng(24)
        x, z = rng.normal(size=(160, 3)), rng.normal(size=(160, 2))
        x, z = x.astype('float32'), z.astype('float32')
        for family, intercept in [('binary', False), ('binary', True), ('cox', False)]:
            with self.subTest(family=family, intercept=intercept):
                y = rng.binomial(1, expit(.5*x[:, 0] + .3*z[:, 0]))
                if family == 'cox':
                    y = np.column_stack([rng.integers(1, 20, len(x)), rng.binomial(1, .7, len(x))])
                design = np.column_stack([x, z]).astype(float)
                if intercept:
                    design = np.column_stack([design, np.ones(len(x))])
                d, target = design, y
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
                model = NeuralPLSI(family=family, initial=True, max_iter=0,
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
        for model in (NeuralPLSI(max_iter=1), SplinePLSI(max_iter=1)):
            with self.assertRaisesRegex(ValueError, 'finite'):
                model.fit(x, z, y)


if __name__ == '__main__':
    unittest.main()
