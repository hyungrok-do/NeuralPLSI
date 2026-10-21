"""Numerical and memory-scaling regressions for reusable native workspaces."""
import unittest

import numpy as np

from models.native import (NativeModel, checked, library, outcome_loss_gradient,
                           training_validation_indices)


def epoch_orders(n, count, seed):
    """Independent implementation of the published splitmix64_v1 schedule."""
    mask = (1 << 64)-1
    state = seed ^ 0x4241544348

    def below(bound):
        nonlocal state
        threshold = (1 << 64) % bound
        while True:
            state = (state+0x9e3779b97f4a7c15) & mask
            value = state
            value = ((value ^ (value >> 30))*0xbf58476d1ce4e5b9) & mask
            value = ((value ^ (value >> 27))*0x94d049bb133111eb) & mask
            value ^= value >> 31
            if value >= threshold:
                return value % bound

    result = np.empty((count, n), dtype=np.int32)
    for row in result:
        row[:] = np.arange(n)
        for remaining in range(n, 1, -1):
            j = below(remaining)
            row[remaining-1], row[j] = row[j], row[remaining-1]
    return result


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

    def test_lazy_schedule_matches_explicit_orders_through_early_stopping(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y = self.data(family, 113)
                options = dict(hidden_units=4, n_hidden_layers=3, max_epoch=200,
                               batch_size=17, family=family, add_intercept=family!='cox')
                actual = NativeModel(3, 2, **options).fit(x, z, y, random_state=13)
                expected = NativeModel(3, 2, **options, random_state=13)
                train, valid = training_validation_indices(len(x), 13)
                arrays = [np.ascontiguousarray(a[rows]) for rows in (train, valid) for a in (x, z, y)]
                tx, tz, ty, vx, vz, vy = arrays
                orders = epoch_orders(len(train), 250, 13)
                weights, diagnostics = expected.weights, np.empty(3)
                checked(library().nplsi_train(expected.config, expected.hyper, weights,
                    tx, tz, ty, len(train), vx, vz, vy, len(valid), orders, diagnostics))
                expected.set_weights(weights)
                expected.resolve_sign_ambiguity()
                np.testing.assert_array_equal(actual.weights, expected.weights)
                np.testing.assert_array_equal(actual.hyper, expected.hyper)
                np.testing.assert_array_equal(list(actual.fit_diagnostics.values()), diagnostics)
                self.assertLess(diagnostics[0], 200)

    def test_large_epoch_limit_uses_only_executed_epochs(self):
        # An upfront schedule would exceed the old 250-million-index limit.
        # Zero learning rate gives exact, short stopping times for either cap.
        x, z, y = self.data('continuous', 801)
        models = [NativeModel(3, 2, hidden_units=2, max_epoch=cap, learning_rate=0.)
                  .fit(x, z, y, random_state=19) for cap in (40, 1_000_000)]
        np.testing.assert_array_equal(models[0].weights, models[1].weights)
        self.assertEqual(models[0].fit_diagnostics, models[1].fit_diagnostics)
        self.assertEqual(models[1].fit_diagnostics['joint_epochs'], 6)
        self.assertEqual(models[1].fit_diagnostics['gamma_epochs'], 21)

    def test_loss_only_validation_matches_full_outcome_including_cox_ties(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                # Validation crosses the internal 256-row inference boundary.
                x, z, y = self.data(family, 1303)
                model = NativeModel(3, 2, hidden_units=5, n_hidden_layers=3,
                    family=family, max_epoch=8, batch_size=47,
                    add_intercept=family!='cox').fit(x, z, y, random_state=13)
                _, valid = training_validation_indices(len(x), 13)
                risk = model.predict(x[valid], z[valid])
                value, _ = outcome_loss_gradient(risk, y[valid], family)
                self.assertEqual(value, model.fit_diagnostics['validation_loss'])

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
