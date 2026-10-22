"""Independent checks of frozen-link coefficient geometry and fitting policy."""
import unittest

import numpy as np
from scipy.special import expit, logsumexp

from models import NeuralPLSI
from models.native import NativeModel, checked, library, outcome_loss_gradient, training_validation_indices
from native_reference import forward


class AlternatingTrainingTest(unittest.TestCase):
    def data(self, family, n=43):
        rng = np.random.default_rng(31)
        x, z = rng.normal(size=(n, 3)).astype('float32'), rng.normal(size=(n, 2)).astype('float32')
        eta = np.tanh(x[:, 0]-x[:, 1]) + z @ [1., -.5]
        y = eta + rng.normal(size=n)
        if family == 'binary':
            y = rng.binomial(1, expit(eta))
        elif family == 'cox':
            y = np.column_stack([rng.integers(1, 9, n), rng.binomial(1, .8, n)])
        return x, z, y.astype('float32')

    def test_full_data_gradient_and_fisher_match_independent_jacobian(self):
        # Numeric differentiation of a separate float64 forward implementation.
        # Cox reference explicitly enumerates every complete tied risk set.
        for family in ('continuous', 'binary', 'cox'):
            for activation in ('ELU', 'LeakyReLU', 'ReLU', 'Tanh'):
                with self.subTest(family=family, activation=activation):
                    x, z, y = self.data(family)
                    net = NativeModel(3, 2, family=family, activation=activation, hidden_units=4,
                        n_hidden_layers=3, add_intercept=family!='cox', random_state=9)
                    w = net.weights
                    w[:3] /= np.linalg.norm(w[:3].astype(float))
                    net.set_weights(w, sign=-1.)
                    d = 5+int(net.config[4])
                    value, gradient, hessian = np.empty(1), np.empty(d), np.empty((d, d))
                    checked(library().nplsi_coefficient_system(net.config, net.hyper, w, x, z, y,
                                                             len(x), value, gradient, hessian))
                    eta, _ = forward(net.config, w, x, z, -1.)
                    jac = np.zeros((len(x), d))
                    beta = w[:3].astype(float)/np.linalg.norm(w[:3].astype(float))
                    for j in range(d):
                        plus, minus = w.astype(float), w.astype(float)
                        index = j if j < 5 else len(w)-1
                        if j < 3:
                            direction = np.eye(3)[j]-beta[j]*beta
                            plus[:3] += 1e-5*direction
                            minus[:3] -= 1e-5*direction
                            plus[:3] /= np.linalg.norm(plus[:3])
                            minus[:3] /= np.linalg.norm(minus[:3])
                        else:
                            plus[index] += 1e-5
                            minus[index] -= 1e-5
                        jac[:, j] = (forward(net.config, plus, x, z, -1.)[0]
                                     - forward(net.config, minus, x, z, -1.)[0])/2e-5
                    if family == 'continuous':
                        expected_value = np.mean((eta-y)**2)
                        g = jac.T @ (2*(eta-y)/len(x))
                        h = 2*jac.T @ jac/len(x)
                        scale = 2/len(x)
                    elif family == 'binary':
                        p = expit(eta)
                        expected_value = np.mean(np.logaddexp(0, eta)-y*eta)
                        g = jac.T @ ((p-y)/len(x))
                        h = jac.T @ ((p*(1-p))[:, None]*jac)/len(x)
                        scale = 1/len(x)
                    else:
                        g, h, expected_value = np.zeros(d), np.zeros((d, d)), 0.
                        for i in np.flatnonzero(y[:, 1]):
                            risk = y[:, 0] >= y[i, 0]
                            logden = logsumexp(eta[risk])
                            p = np.exp(eta[risk]-logden)
                            mean = p @ jac[risk]
                            expected_value += logden-eta[i]
                            g += mean-jac[i]
                            h += jac[risk].T @ (p[:, None]*jac[risk])-np.outer(mean, mean)
                        scale = 1/y[:, 1].sum()
                        g *= scale
                        h *= scale
                        expected_value *= scale
                    ridge = (1e-10 if family == 'continuous' else 1e-6)*scale
                    expected_value += .5*ridge*np.dot(w[3:5], w[3:5])
                    g[3:5] += ridge*w[3:5]
                    h[3:5, 3:5] += ridge*np.eye(2)
                    np.testing.assert_allclose(value[0], expected_value, rtol=2e-6, atol=2e-7)
                    np.testing.assert_allclose(gradient, g, rtol=2e-5, atol=2e-6)
                    np.testing.assert_allclose(hessian, h, rtol=3e-5, atol=2e-6)
                    self.assertLess(abs(beta @ gradient[:3]), 1e-10)
                    self.assertGreater(np.linalg.eigvalsh(hessian).min(), -1e-10)

    def test_coefficient_step_decreases_global_loss_and_keeps_link_fixed(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y = self.data(family, 517)  # crosses derivative chunk boundaries
                net = NativeModel(3, 2, family=family, hidden_units=5, n_hidden_layers=2,
                                  add_intercept=family!='cox', random_state=8)
                w = net.weights
                w[:3] /= np.linalg.norm(w[:3].astype(float))
                link = w[5:len(w)-int(net.config[4])].copy()
                for _ in range(4):
                    previous = w[:3].copy()
                    stats = np.empty(4)
                    checked(library().nplsi_coefficient_step(net.config, net.hyper, w, x, z, y, len(x), stats))
                    self.assertLessEqual(stats[1], stats[0])
                    np.testing.assert_array_equal(w[5:len(w)-int(net.config[4])], link)
                    self.assertAlmostEqual(np.linalg.norm(w[:3]), 1., places=6)
                    angle = np.arccos(np.clip(np.dot(previous, w[:3]), -1., 1.))
                    self.assertLessEqual(angle, np.arctan(.25)+1e-5)

    def test_link_adam_ignores_coefficient_gradients_decay_and_moments(self):
        net = NativeModel(3, 2, hidden_units=4, n_hidden_layers=2, add_intercept=True)
        rng = np.random.default_rng(5)
        gradients = rng.normal(size=(5, len(net.weights))).astype('float32')
        fixed = np.zeros(len(net.weights), dtype=bool)
        fixed[:5] = True
        fixed[-1] = True
        a, b = net.weights, net.weights
        distorted = gradients.copy()
        distorted[:, fixed] *= 1e10
        hyper = net.hyper.copy()
        hyper[2] = 1000.  # beta decay must have no effect in a link-only step
        for w, g in ((a, gradients), (b, distorted)):
            checked(library().nplsi_adam_updates(net.config, hyper, w, g, len(g), 2))
        np.testing.assert_array_equal(a, b)
        np.testing.assert_array_equal(a[fixed], net.weights[fixed])
        self.assertTrue(np.any(a[~fixed] != net.weights[~fixed]))

    def test_full_data_default_and_optional_checkpoint_are_complete_models(self):
        for fraction in (0., .2):
            with self.subTest(fraction=fraction):
                x, z, y = self.data('continuous', 143)
                options = dict(hidden_units=4, n_hidden_layers=2, add_intercept=True,
                               link_epochs=3, validation_fraction=fraction)
                model = NeuralPLSI(max_epoch=8, **options).fit(x, z, y, random_state=9)
                diag = model.fit_diagnostics
                restored = NativeModel.from_bytes(model.net.to_bytes())
                self.assertEqual(restored.optimizer, 'alternating')
                self.assertEqual(restored.link_epochs, 3)
                self.assertEqual(restored.validation_fraction, fraction)
                restored.fit(x, z, y, random_state=9)
                np.testing.assert_array_equal(model.net.weights, restored.weights)
                train, valid = training_validation_indices(len(x), 9, fraction)
                self.assertEqual(diag['epochs'], 8)
                self.assertEqual(diag['coefficient_cycles'], 3)
                self.assertEqual(diag['training_rows'], len(train))
                self.assertEqual(diag['validation_rows'], len(valid))
                g = model.predict_gxb(x[train]).astype(float)
                design = np.column_stack([z[train], np.ones(len(train))])
                expected = np.linalg.lstsq(design, y[train]-g, rcond=None)[0]
                np.testing.assert_allclose(np.r_[model.gamma, model.intercept_val], expected, atol=2e-6)
                if fraction:
                    selected = diag['checkpoint_epoch']
                    self.assertIn(selected, (3, 6, 8))
                    short = NeuralPLSI(max_epoch=selected, **options).fit(x, z, y, random_state=9)
                    np.testing.assert_array_equal(model.net.weights, short.net.weights)
                    loss, _ = outcome_loss_gradient(model.predict(x[valid], z[valid]), y[valid], 'continuous')
                    g0 = float(model.g_function([0])[0])
                    self.assertAlmostEqual(diag['checkpoint_loss'], loss+g0*g0, places=6)
                    self.assertAlmostEqual(diag['validation_loss'], loss, places=6)
                else:
                    self.assertIsNone(diag['checkpoint_epoch'])
                    self.assertIsNone(diag['validation_loss'])

    def test_full_data_initialization_and_v2_state_compatibility(self):
        x, z, y = self.data('continuous', 100)
        m = NeuralPLSI(initial=True, max_epoch=0, add_intercept=True).fit(x, z, y)
        design = np.column_stack([x, z, np.ones(len(x))])
        reference = np.linalg.lstsq(design, y, rcond=None)[0]
        np.testing.assert_allclose(m.predict(x, z), design @ reference, atol=2e-6)
        self.assertEqual(m.fit_diagnostics['coefficient_cycles'], 0)
        old = NativeModel(3, 2, optimizer='joint', max_epoch=1).fit(x, z, y)
        lines = old.to_bytes().decode().splitlines()
        v2 = '\n'.join(['NEURALPLSI_NATIVE 2', lines[1], lines[2], lines[4], lines[5]])+'\n'
        restored = NativeModel.from_bytes(v2.encode())
        self.assertEqual(restored.optimizer, 'joint')
        self.assertEqual(restored.validation_fraction, .2)
        np.testing.assert_array_equal(restored.predict(x, z), old.predict(x, z))

    def test_single_exposure_without_covariates_has_no_free_index_direction(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, _, y = self.data(family, 90)
                x, z = x[:, :1].copy(), np.empty((len(x), 0), dtype='float32')
                model = NeuralPLSI(family=family, hidden_units=4, max_epoch=3).fit(x, z, y)
                self.assertAlmostEqual(float(model.beta[0]), 1., places=5)
                self.assertEqual(model.fit_diagnostics['coefficient_steps'], 0)
                self.assertTrue(np.isfinite(model.predict(x, z)).all())


if __name__ == '__main__':
    unittest.main()
