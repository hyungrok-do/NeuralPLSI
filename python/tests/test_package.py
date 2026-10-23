import importlib.metadata
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import neuralplsi
from neuralplsi import NeuralPLSI, SplinePLSI
from neuralplsi.native import LIBRARY_NAMES, library, library_path
from neuralplsi.simulation import simulate_data


def without_library_override():
    return {key: value for key, value in os.environ.items() if key != 'NEURALPLSI_LIBRARY'}


class PackageTest(unittest.TestCase):
    def test_installed_metadata_matches_package_and_engine_versions(self):
        self.assertEqual(library().nplsi_version().decode(), neuralplsi.__version__)
        try:
            installed = importlib.metadata.version('neuralplsi')
        except importlib.metadata.PackageNotFoundError:
            self.skipTest('neuralplsi is not installed; the tests use the checkout')
        self.assertEqual(installed, neuralplsi.__version__)

    def test_installed_library_is_found_in_every_package_directory(self):
        # An editable install keeps the sources in the checkout and the library in site-packages.
        package = sys.modules['neuralplsi']
        with tempfile.TemporaryDirectory() as sources, tempfile.TemporaryDirectory() as installed:
            expected = Path(installed)/'_lib'/LIBRARY_NAMES[0]
            expected.parent.mkdir()
            expected.touch()
            with patch.dict(os.environ, without_library_override(), clear=True), \
                    patch.object(package, '__path__', [sources, installed]):
                self.assertEqual(library_path(), expected)

    def test_library_override_must_exist(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = os.path.join(directory, 'libneuralplsi.so')
            with patch.dict(os.environ, {'NEURALPLSI_LIBRARY': missing}):
                with self.assertRaisesRegex(RuntimeError, 'NEURALPLSI_LIBRARY=.* does not exist'):
                    library_path()

    def test_library_override_must_be_a_neuralplsi_engine(self):
        import _ctypes
        other = getattr(_ctypes, '__file__', None)
        if not other:
            self.skipTest('no shared library without nplsi_version to load')
        library.cache_clear()
        try:
            with patch.dict(os.environ, {'NEURALPLSI_LIBRARY': other}):
                with self.assertRaisesRegex(RuntimeError, 'is not a NeuralPLSI .* engine library'):
                    library()
        finally:
            library.cache_clear()

    def test_none_covariates_equal_an_empty_matrix(self):
        grid = np.linspace(-1, 1, 5)
        for family in ('continuous', 'binary', 'cox'):
            X, _, y, *_ = simulate_data(150, outcome=family, g_type='sigmoid', seed=4)
            empty = np.empty((len(X), 0))
            for name, estimator in (('NeuralPLSI', lambda: NeuralPLSI(family=family, max_iter=30)),
                                    ('SplinePLSI', lambda: SplinePLSI(family=family, max_iter=20))):
                with self.subTest(family=family, estimator=name):
                    without = estimator().fit(X, None, y, random_state=0)
                    reference = estimator().fit(X, empty, y, random_state=0)
                    self.assertEqual(without.gamma.shape, (0,))
                    np.testing.assert_array_equal(without.beta, reference.beta)
                    np.testing.assert_array_equal(without.predict(X, None), reference.predict(X, empty))
                    draws = without.inference_bootstrap(X, None, y, n_samples=3, random_state=1,
                                                        g_grid=grid, n_jobs=1)
                    expected = reference.inference_bootstrap(X, empty, y, n_samples=3, random_state=1,
                                                             g_grid=grid, n_jobs=1)
                    self.assertEqual(draws['gamma_samples'].shape, (3, 0))
                    np.testing.assert_array_equal(draws['beta_samples'], expected['beta_samples'])
                    np.testing.assert_array_equal(draws['g_samples'], expected['g_samples'])

    def test_models_with_covariates_reject_missing_covariates(self):
        X, Z, y, *_ = simulate_data(100, seed=5)
        for model in (NeuralPLSI(max_iter=5).fit(X, Z, y), SplinePLSI(max_iter=5).fit(X, Z, y, random_state=0)):
            with self.subTest(estimator=type(model).__name__):
                with self.assertRaisesRegex(ValueError, r'Z must .*\(100, 3\)'):
                    model.predict(X, None)


if __name__ == '__main__':
    unittest.main()
