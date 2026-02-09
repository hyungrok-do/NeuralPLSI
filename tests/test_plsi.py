import sys
import numpy as np
import pytest
from unittest.mock import MagicMock, patch

@pytest.fixture(scope="function")
def plsi_class():
    """
    Fixture that returns the SplinePLSI class.
    If dependencies are missing, it mocks them so the test can verify logic flow.
    """
    try:
        # Try importing real model
        from models.PLSI import SplinePLSI
        yield SplinePLSI
    except ImportError:
        # Mock dependencies
        with patch.dict(sys.modules):
            # Helper to create package mocks
            def create_package_mock():
                m = MagicMock()
                m.__path__ = []
                return m

            # Create mocks
            m_sklearn = create_package_mock()
            m_scipy = create_package_mock()
            m_torch = create_package_mock()
            m_pandas = create_package_mock()
            m_lifelines = create_package_mock()

            # Setup sklearn
            class MockRidge:
                def __init__(self, alpha=1.0, fit_intercept=False):
                    self.alpha = alpha
                    self.fit_intercept = fit_intercept
                    self.coef_ = None
                def fit(self, X, y):
                    d = X.shape[1]
                    self.coef_ = np.ones(d) * 0.1
                    return self
                def predict(self, X):
                    return X @ self.coef_

            class MockLogisticRegression:
                def __init__(self, penalty='l2', C=1.0, fit_intercept=False, solver='lbfgs', max_iter=100):
                    self.C = C
                    self.coef_ = None
                def fit(self, X, y):
                    d = X.shape[1]
                    self.coef_ = np.ones((1, d)) * 0.1
                    return self
                def predict_proba(self, X):
                    return np.ones((X.shape[0], 2)) * 0.5

            m_linear_model = create_package_mock()
            m_linear_model.Ridge = MockRidge
            m_linear_model.LogisticRegression = MockLogisticRegression
            m_sklearn.linear_model = m_linear_model

            # Setup scipy.optimize
            m_optimize = create_package_mock()
            def mock_minimize(fun, x0, **kwargs):
                res = MagicMock()
                res.x = np.array(x0)
                return res
            m_optimize.minimize = mock_minimize
            m_scipy.optimize = m_optimize

            # Setup torch (structure required for import)
            m_torch.nn = create_package_mock()
            m_torch.nn.functional = MagicMock()
            m_torch.optim = create_package_mock()
            m_torch.utils = create_package_mock()
            m_torch.utils.data = MagicMock()

            # Assign to sys.modules
            sys.modules['sklearn'] = m_sklearn
            sys.modules['sklearn.linear_model'] = m_linear_model
            sys.modules['sklearn.model_selection'] = MagicMock()
            sys.modules['sklearn.metrics'] = MagicMock()

            sys.modules['scipy'] = m_scipy
            sys.modules['scipy.optimize'] = m_optimize

            sys.modules['torch'] = m_torch
            sys.modules['torch.nn'] = m_torch.nn
            sys.modules['torch.nn.functional'] = m_torch.nn.functional
            sys.modules['torch.optim'] = m_torch.optim
            sys.modules['torch.utils'] = m_torch.utils
            sys.modules['torch.utils.data'] = m_torch.utils.data

            sys.modules['pandas'] = m_pandas
            sys.modules['lifelines'] = m_lifelines
            sys.modules['lifelines.utils'] = MagicMock()

            # Ensure models.PLSI is reloaded if it was already imported (partially)
            to_remove = [k for k in sys.modules if k.startswith('models')]
            for k in to_remove:
                del sys.modules[k]

            import models.PLSI
            # Force assignment of minimize in case of import caching issues
            models.PLSI.opt.minimize = mock_minimize

            yield models.PLSI.SplinePLSI

class TestSplinePLSI:
    def test_fit_continuous_linear(self, plsi_class):
        # Synthetic data
        np.random.seed(42)
        n = 100
        p = 5
        q = 3
        X = np.random.randn(n, p)
        Z = np.random.randn(n, q)
        beta_true = np.array([1, 0, 0, 0, 0])
        beta_true = beta_true / np.linalg.norm(beta_true)
        gamma_true = np.array([0.5, -0.5, 0.2])

        y = (X @ beta_true) + (Z @ gamma_true) + 0.01 * np.random.randn(n)

        # Instantiate model using the fixture class
        model = plsi_class(family='continuous', num_knots=5, spline_degree=3, alpha=1e-6)

        model.fit(X, Z, y)

        # Check if fitted
        assert model.beta is not None
        assert model.gamma is not None
        assert model.spline_coeffs is not None
        assert model.knot_vector is not None

        # Verify dimensions (logic check)
        assert len(model.beta) == p
        assert len(model.gamma) == q
        expected_spline_coeffs = 5 + 3 - 1
        assert len(model.spline_coeffs) == expected_spline_coeffs

        # Check predictions run
        y_pred = model.predict(X, Z)
        assert y_pred.shape == (n,)

    def test_fit_continuous_nonlinear(self, plsi_class):
        # Synthetic data
        np.random.seed(42)
        n = 50
        p = 5
        q = 3
        X = np.random.randn(n, p)
        Z = np.random.randn(n, q)
        y = np.random.randn(n)

        model = plsi_class(family='continuous', num_knots=5, spline_degree=3)
        model.fit(X, Z, y)

        assert model.beta is not None

        # Check predictions
        y_pred = model.predict(X, Z)
        assert y_pred.shape == (n,)
