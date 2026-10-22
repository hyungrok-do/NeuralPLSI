"""Neural partial-linear single-index models, with a C++ engine.

NeuralPLSI fits g(X @ beta) + Z @ gamma with a neural-network link g; SplinePLSI
is the B-spline comparator. Both fit, predict and bootstrap in C++.
"""
from ._version import __version__
from .neural import NeuralPLSI
from .spline import SplinePLSI
from .simulation import simulate_data

__all__ = ['NeuralPLSI', 'SplinePLSI', 'simulate_data', '__version__']
