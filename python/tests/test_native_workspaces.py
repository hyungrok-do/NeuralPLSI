"""Numerical and memory-scaling regressions for reusable native workspaces."""
import unittest

import numpy as np

from neuralplsi.native import NativeModel, outcome_loss_gradient


class NativeWorkspacesTest(unittest.TestCase):
    def data(self, family, n):
        rng = np.random.default_rng(32)
        x = rng.normal(size=(n, 3)).astype('float32')
        z = rng.normal(size=(n, 2)).astype('float32')
        y = x[:, 0]+z[:, 0]+rng.normal(size=n)
        if family == 'binary':
            y = (y > 0).astype(float)
        elif family == 'cox':
            y = np.column_stack([rng.integers(1, 17, n), rng.binomial(1, .75, n)])
        return x, z, y.astype('float32')

    def test_objective_matches_chunked_prediction_loss_including_cox_ties(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                # The data cross the internal 256-row inference boundary.
                x, z, y = self.data(family, 1303)
                model = NativeModel(3, 2, hidden_units=5, n_hidden_layers=3, family=family,
                    max_iter=8, add_intercept=family!='cox').fit(x, z, y, random_state=13)
                value, _ = outcome_loss_gradient(model.predict(x, z), y, family)
                value += float(model.g_function([0.])[0])**2
                decay, weights = float(model.hyper[0]), model.weights.astype(float)
                for w in weights[5:len(weights)-int(model.config[4])]:
                    value += .5*decay*w*w
                self.assertEqual(value, model.fit_diagnostics['objective'])

    def test_prediction_and_link_chunk_boundaries(self):
        x, z, _ = self.data('continuous', 517)
        grid = np.linspace(-3, 3, 513)
        for depth, activation in enumerate(('ELU', 'LeakyReLU', 'ReLU', 'Tanh'), 1):
            with self.subTest(activation=activation):
                model = NativeModel(3, 2, hidden_units=depth+3, n_hidden_layers=depth,
                                    activation=activation, add_intercept=True)
                model.set_weights(model.weights, sign=-1.)
                for mode in (0, 3):
                    expected = model.predict(x, z, batch_size=1, mode=mode)
                    for batch in (17, 256, 4096):
                        np.testing.assert_array_equal(expected, model.predict(x, z, batch_size=batch, mode=mode))
                expected = np.concatenate([model.g_function([v]) for v in grid])
                np.testing.assert_array_equal(expected, model.g_function(grid))


if __name__ == '__main__':
    unittest.main()
