import numpy as np
import warnings
warnings.filterwarnings('ignore')
from models.PLSI import SplinePLSI
from models.NeuralPLSI import NeuralPLSI

np.random.seed(42)
n = 1000
p = 5
q = 3

X = np.random.randn(n, p)
Z = np.random.randn(n, q)

beta_true = np.array([1.0, -0.5, 0.5, -0.5, 0.5])
beta_true /= np.linalg.norm(beta_true)
gamma_true = np.array([0.5, -0.5, 0.5])

eta = X @ beta_true
g_true = 2 * (1 / (1 + np.exp(-eta)) - 0.5)

risk = g_true + Z @ gamma_true
T = np.random.exponential(np.exp(-risk))
C = np.random.exponential(1.5, size=n)

def build_cox_y(t, c):
    time = np.minimum(t, c)
    event = (t <= c).astype(int)
    return np.column_stack([time, event])

y = build_cox_y(T, C)

print("Starting SplinePLSI (Cox)...")
spline_model = SplinePLSI(family='cox', alpha=1e-2)
spline_model.fit(X, Z, y)

print("Spline Beta bias:", np.mean(spline_model.beta - beta_true))
print("Spline Gamma bias:", np.mean(spline_model.gamma - gamma_true))

print("Starting NeuralPLSI (Cox)...")
neural_model = NeuralPLSI(family='cox', max_epoch=200)
neural_model.fit(X, Z, y)

print("Neural Beta bias:", np.mean(neural_model.beta - beta_true))
print("Neural Gamma bias:", np.mean(neural_model.gamma - gamma_true))
