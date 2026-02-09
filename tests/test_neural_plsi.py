import sys
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

# --- MOCKING DEPENDENCIES ---
# We must mock torch and sklearn before importing models because they are not installed
mock_torch = MagicMock()
mock_torch.nn.Module = object # Allow inheritance
sys.modules['torch'] = mock_torch
sys.modules['torch.nn'] = mock_torch.nn
sys.modules['torch.nn.functional'] = MagicMock()
sys.modules['torch.utils'] = MagicMock()
sys.modules['torch.utils.data'] = MagicMock()
sys.modules['torch.optim'] = MagicMock()

mock_sklearn = MagicMock()
sys.modules['sklearn'] = mock_sklearn
sys.modules['sklearn.linear_model'] = MagicMock()
sys.modules['sklearn.model_selection'] = MagicMock()
sys.modules['sklearn.metrics'] = MagicMock()
sys.modules['sklearn.kernel_ridge'] = MagicMock()
sys.modules['sklearn.base'] = MagicMock()
sys.modules['sklearn.utils'] = MagicMock()
sys.modules['sklearn.utils.validation'] = MagicMock()

# Mock pandas and joblib
sys.modules['pandas'] = MagicMock()
sys.modules['joblib'] = MagicMock()

# Need to handle relative imports inside models if we are running from tests/
# conftest.py adds project root to sys.path, so 'models' is importable.

from models.NeuralPLSI import NeuralPLSI

def test_predict_unfitted_model():
    """Test that predicting with an unfitted model raises an error."""
    model = NeuralPLSI()
    X = np.array([[1]])
    Z = np.array([[1]])

    # Currently checking for the AttributeError that occurs when self.net is None
    with pytest.raises(ValueError, match="Model has not been fitted yet."):
        model.predict(X, Z)

def test_predict_forward_call():
    """Test that predict calls _batched_forward correctly."""
    model = NeuralPLSI()
    model.net = MagicMock() # Mock the network so it thinks it's fitted if needed
    model._net_infer = MagicMock() # Mock infer net to avoid calling eval on None

    X = np.array([[1]])
    Z = np.array([[1]])

    with patch.object(model, '_batched_forward', return_value=np.array([1, 2])) as mock_forward:
        result = model.predict(X, Z, batch_size=50)

        assert np.array_equal(result, np.array([1, 2]))
        mock_forward.assert_called_once_with(X, Z, batch_size=50)

def test_predict_proba_forward_call():
    """Test that predict_proba calls _batched_forward with sigmoid."""
    model = NeuralPLSI()
    X = np.array([[1]])
    Z = np.array([[1]])

    with patch.object(model, '_batched_forward') as mock_forward:
        model.predict_proba(X, Z, batch_size=32)
        mock_forward.assert_called_once()
        args, kwargs = mock_forward.call_args
        # Mock torch objects are magic mocks, so we can check if it passed the mock
        assert kwargs['transform'] == mock_torch.sigmoid
        assert kwargs['batch_size'] == 32

def test_predict_partial_hazard_forward_call():
    """Test that predict_partial_hazard calls _batched_forward with exp."""
    model = NeuralPLSI()
    X = np.array([[1]])
    Z = np.array([[1]])

    with patch.object(model, '_batched_forward') as mock_forward:
        model.predict_partial_hazard(X, Z, batch_size=32)
        mock_forward.assert_called_once()
        args, kwargs = mock_forward.call_args
        assert kwargs['transform'] == mock_torch.exp
