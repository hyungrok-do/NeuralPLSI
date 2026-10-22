"""Independent geometry checks and fitting-policy tests for alternating training."""
import unittest

import numpy as np
from scipy.special import expit, softmax

from models.native import NativeModel, checked, library, DIAGNOSTICS_SIZE, outcome_loss_gradient
from native_reference import forward, loss
from test_native_workspaces import epoch_orders


class AlternatingOptimizerTest(unittest.TestCase):
    def data(self, family, n=73):
        rng = np.random.default_rng(84)
        x, z = rng.normal(size=(n, 3)).astype('float32'), rng.normal(size=(n, 2)).astype('float32')
        y = x[:, 0] + z[:, 0] + rng.normal(size=n)
        if family == 'binary':
            y = (y > 0).astype(float)
        elif family == 'cox':
            y = np.column_stack([rng.integers(1, 7, n), rng.binomial(1, .7, n)])
        return x, z, y.astype('float32')

    def test_sphere_score_and_curvature_against_independent_finite_differences(self):
        for family in ('continuous', 'binary', 'cox'):
            for activation in ('ELU', 'LeakyReLU', 'ReLU', 'Tanh'):
                with self.subTest(family=family, activation=activation):
                    x, z, y = self.data(family)
                    net = NativeModel(3, 2, hidden_units=4, n_hidden_layers=3,
                        family=family, activation=activation, random_state=13)
                    net.set_weights(net.weights, sign=-1.)
                    w = net.weights.astype(float)
                    w[:3] /= np.linalg.norm(w[:3])
                    net.set_weights(w)
                    value, q, score, curvature = np.empty(1), np.empty((3, 2)), np.empty(2), np.empty((2, 2))
                    checked(library().nplsi_beta_geometry(net.config, net.hyper, net.weights,
                        x, z, y, len(x), value, q, score, curvature))
                    np.testing.assert_allclose(q.T @ q, np.eye(2), atol=1e-12)
                    np.testing.assert_allclose(q.T @ net.coefficients()[0], 0., atol=1e-7)
                    numeric, jacobian, eps = np.empty(2), np.empty((len(x), 2)), 1e-5
                    for j in range(2):
                        plus, minus = w.copy(), w.copy()
                        plus[:3] += eps*q[:, j]
                        minus[:3] -= eps*q[:, j]
                        numeric[j] = (loss(net.config, plus, x, z, y, -1., False) -
                                      loss(net.config, minus, x, z, y, -1., False))/(2*eps)
                        jacobian[:, j] = (forward(net.config, plus, x, z, -1.)[0] -
                                          forward(net.config, minus, x, z, -1.)[0])/(2*eps)
                    eta = forward(net.config, w, x, z, -1.)[0]
                    if family == 'continuous':
                        expected = 2*jacobian.T @ jacobian/len(x)
                    elif family == 'binary':
                        p = expit(eta)
                        expected = jacobian.T @ ((p*(1-p))[:, None]*jacobian)/len(x)
                    else:
                        # Explicit global risk sets, deliberately including ties.
                        expected = np.zeros((2, 2))
                        for i in np.flatnonzero(y[:, 1]):
                            risk = y[:, 0] >= y[i, 0]
                            probability = softmax(eta[risk])
                            design = jacobian[risk]
                            mean = probability @ design
                            expected += design.T @ (probability[:, None]*design)-np.outer(mean, mean)
                        expected /= y[:, 1].sum()
                    np.testing.assert_allclose(score, numeric, rtol=2e-5, atol=2e-7)
                    np.testing.assert_allclose(curvature, expected, rtol=2e-5, atol=2e-7)
                    np.testing.assert_allclose(value[0], loss(net.config, w, x, z, y, -1., False), rtol=2e-6)

    def test_beta_step_decreases_loss_preserves_network_and_handles_rank_deficiency(self):
        for family in ('continuous', 'binary', 'cox'):
            x, z, y = self.data(family)
            x[:, 2] = x[:, 1]
            net = NativeModel(3, 2, family=family, random_state=6, beta_max_step=.08)
            weights, before = net.weights, net.weights
            values = np.empty(2)
            checked(library().nplsi_beta_step(net.config, net.hyper, weights, x, z, y, len(x), values))
            self.assertLess(values[1], values[0])
            np.testing.assert_array_equal(weights[3:], before[3:])
            self.assertAlmostEqual(np.linalg.norm(weights[:3]), 1., places=6)
            angle = np.arccos(np.clip(np.dot(weights[:3], before[:3])/
                                     np.linalg.norm(before[:3]), -1., 1.))
            self.assertLessEqual(angle, np.arctan(.08)+1e-5)

    def test_link_adam_ignores_coefficient_gradients_and_decay(self):
        net = NativeModel(3, 2, hidden_units=4, add_intercept=True, weight_decay_beta=100.)
        rng = np.random.default_rng(4)
        gradient = rng.normal(size=len(net.weights)).astype('float32')
        distorted = gradient.copy()
        frozen = np.zeros(len(gradient), dtype=bool)
        frozen[:5] = True
        frozen[-1] = True
        distorted[frozen] *= 1e8
        a, b = net.weights, net.weights
        for w, g in ((a, gradient), (b, distorted)):
            checked(library().nplsi_adam_step(net.config, net.hyper, w, g, 2))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[frozen], net.weights[frozen])
        self.assertTrue(np.any(a[~frozen] != net.weights[~frozen]))

    def test_full_data_schedule_including_partial_final_cycle(self):
        for family in ('continuous', 'binary', 'cox'):
            x, z, y = self.data(family)
            options = dict(family=family, hidden_units=4, max_epoch=7, link_epochs=3, batch_size=19)
            actual = NativeModel(3, 2, **options).fit(x, z, y, random_state=13)
            expected = NativeModel(3, 2, **options, random_state=13)
            weights, diagnostics, empty = expected.weights, np.empty(DIAGNOSTICS_SIZE), np.empty(0, 'float32')
            checked(library().nplsi_train(expected.config, expected.hyper, weights,
                x, z, y, len(x), empty, empty, empty, 0, epoch_orders(len(x), 7, 13), diagnostics))
            expected.set_weights(weights)
            expected.resolve_sign_ambiguity()
            np.testing.assert_array_equal(actual.weights, expected.weights)
            self.assertEqual(actual.fit_diagnostics['training_rows'], len(x))
            self.assertEqual(actual.fit_diagnostics['validation_rows'], 0)
            self.assertIsNone(actual.fit_diagnostics['validation_loss'])
            self.assertIsNone(actual.fit_diagnostics['checkpoint_epoch'])
            self.assertEqual(actual.fit_diagnostics['joint_epochs'], 7)
            self.assertEqual(diagnostics[3]+diagnostics[9], 3)

    def test_validation_cannot_enter_coefficient_fit_and_checkpoint_is_recorded(self):
        x, z, y = self.data('continuous')
        states = []
        for vy in (y, -10*y):
            net = NativeModel(3, 2, hidden_units=4, max_epoch=1, validation_fraction=.2)
            weights, diagnostics = net.weights, np.empty(DIAGNOSTICS_SIZE)
            checked(library().nplsi_train(net.config, net.hyper, weights,
                x, z, y, len(x), x, z, vy, len(x), epoch_orders(len(x), 1, 0), diagnostics))
            states.append(weights)
            self.assertEqual(diagnostics[5], 1)
            net.set_weights(weights)
            value, _ = outcome_loss_gradient(net.predict(x, z), vy, 'continuous')
            self.assertAlmostEqual(diagnostics[6], value+float(net.g_function([0])[0])**2, places=6)
        np.testing.assert_array_equal(*states)

    def test_version_two_models_preserve_legacy_predictions_and_settings(self):
        x, z, y = self.data('binary')
        net = NativeModel(3, 2, family='binary', optimizer='joint', validation_fraction=.2,
                          max_epoch=3).fit(x, z, y)
        lines = net.to_bytes().decode().splitlines()
        legacy = '\n'.join(['NEURALPLSI_NATIVE 2', ' '.join(lines[1].split()[:11]),
                           ' '.join(lines[2].split()[:5]), lines[3], ' '.join(lines[4].split()[:3])])+'\n'
        loaded = NativeModel.from_bytes(legacy.encode())
        np.testing.assert_array_equal(net.predict(x, z), loaded.predict(x, z))
        self.assertEqual(loaded.config[11], 0)
        self.assertAlmostEqual(loaded.hyper[5], .2)
        self.assertIsNone(loaded.fit_diagnostics['training_loss'])
        self.assertIsNone(loaded.fit_diagnostics['training_rows'])
        self.assertIsNone(loaded.fit_diagnostics['validation_rows'])
        np.testing.assert_array_equal(loaded.predict(x, z), NativeModel.from_bytes(loaded.to_bytes()).predict(x, z))

    def test_alternating_restores_best_checkpoint_without_ending_training(self):
        target = np.tile(np.array([-1., 1.], dtype='float32'), 4)
        x, z, fits = target[:, None], np.empty((8, 0), dtype='float32'), []
        for epochs in (1, 30):
            net = NativeModel(1, 0, hidden_units=1, n_hidden_layers=1, max_epoch=epochs,
                              validation_fraction=.2, batch_size=8, weight_decay=0.)
            weights = np.zeros_like(net.weights)
            weights[0] = 1.
            diagnostics = np.empty(DIAGNOSTICS_SIZE)
            checked(library().nplsi_train(net.config, net.hyper, weights, x, z, target, 8,
                x, z, -target, 8, epoch_orders(8, epochs, 0), diagnostics))
            self.assertEqual(diagnostics[0], epochs)
            self.assertEqual(diagnostics[5], 1)
            fits.append(weights)
        np.testing.assert_array_equal(*fits)

    def test_flat_link_and_single_index_coordinate(self):
        x, z, y = self.data('continuous')
        for p in (1, 3):
            net = NativeModel(p, 2, hidden_units=2)
            weights = np.zeros_like(net.weights)
            weights[0] = 1.
            losses = np.empty(2)
            before = weights.copy()
            checked(library().nplsi_beta_step(net.config, net.hyper, weights,
                np.ascontiguousarray(x[:, :p]), z, y, len(x), losses))
            np.testing.assert_array_equal(weights, before)
            self.assertEqual(losses[0], losses[1])

    def test_final_gamma_score_uses_the_current_link_after_accepted_or_skipped_beta(self):
        # Cross the forward chunk boundary, include tied Cox times, and exercise
        # the p=1 path where beta is always skipped. A stale/rejected link cache
        # would leave gamma profiled against the wrong offset.
        for family in ('continuous', 'binary', 'cox'):
            x, z, y = self.data(family, n=517)
            for p in (1, 3):
                with self.subTest(family=family, p=p):
                    net = NativeModel(p, 2, hidden_units=5, n_hidden_layers=3,
                        family=family, activation='ELU', max_epoch=7,
                        batch_size=73, add_intercept=family!='cox').fit(x[:, :p], z, y)
                    eta = net.predict(x[:, :p], z).astype(float)
                    _, gamma, intercept = net.coefficients()
                    design = z.astype(float)
                    coefficients = gamma.astype(float)
                    if intercept is not None:
                        design = np.column_stack([design, np.ones(len(x))])
                        coefficients = np.r_[coefficients, intercept]
                    if family == 'continuous':
                        score = design.T @ (eta-y)
                    elif family == 'binary':
                        score = design.T @ (expit(eta)-y)
                    else:
                        score = np.zeros(design.shape[1])
                        for i in np.flatnonzero(y[:, 1]):
                            risk = y[:, 0] >= y[i, 0]
                            score += softmax(eta[risk]) @ design[risk]-design[i]
                    score[:2] += (1e-10 if family=='continuous' else 1e-6)*coefficients[:2]
                    np.testing.assert_allclose(score/len(x), 0., atol=1e-6)
                    self.assertEqual(net.fit_diagnostics['joint_epochs'], 7)
                    self.assertEqual(net.fit_diagnostics['beta_steps_skipped'] if p==1 else
                                     net.fit_diagnostics['beta_steps'], 7)


if __name__ == '__main__':
    unittest.main()
