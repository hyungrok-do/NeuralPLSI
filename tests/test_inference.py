
import pytest
import numpy as np
import torch
from models.NeuralPLSI import NeuralPLSI

def test_hessian_correctness_smoke():
    """
    Smoke test to ensure Hessian inference runs and produces plausible results.
    We don't have exact ground truth here without saving artifacts,
    but we can check for shapes and non-NaN values.
    """
    np.random.seed(42)
    torch.manual_seed(42)

    n = 100
    p = 5
    q = 2
    X = np.random.randn(n, p)
    Z = np.random.randn(n, q)
    y = np.random.randn(n)

    model = NeuralPLSI(family='continuous', max_epoch=2, hidden_units=4, n_hidden_layers=1)
    model.fit(X, Z, y)

    # Compute covariance
    res = model.inference_hessian(X, Z, y)

    cov_beta_gamma = res['cov_beta_gamma']
    assert cov_beta_gamma.shape == (p + q, p + q)
    assert not np.isnan(cov_beta_gamma).any()
    # Diagonal elements should be positive (variances)
    assert np.all(np.diag(cov_beta_gamma) >= 0)

    # Test hessian_g_bands
    g_grid = np.linspace(-1, 1, 10)
    res_g = model.inference_hessian_g(X, Z, y, g_grid=g_grid)
    g_se = res_g['g_se']
    assert len(g_se) == 10
    assert not np.isnan(g_se).any()
    assert np.all(g_se >= 0)

def test_vmap_available():
    """Ensure vmap is available in the environment as we rely on it for performance."""
    try:
        if hasattr(torch, 'func') and hasattr(torch.func, 'vmap'):
            pass
        elif hasattr(torch, 'vmap'):
            pass
        else:
            pytest.fail("vmap not found in torch")
    except ImportError:
        pytest.fail("Could not import torch")
