import numpy as np
from scipy.stats import multivariate_normal, bernoulli, expon, norm, t

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.linalg.norm(beta)
gamma = np.array([1, -0.5, 0.5])

def simulate_data(n, outcome='continuous', g_type='sigmoid', censoring_rate=0.3, seed=0, x_dist='normal'):
    np.random.seed(seed)

    g_dict = {
        'linear': lambda x, a=1: a * x,
        'sigmoid': lambda x, a=2: (1 / (1 + np.exp(-a * x)) - 0.5) * 5,
        'sfun': lambda x: (2 / (1 + np.exp(-x)) - 0.2 * x - 1) * 10,
    }
    if g_type not in g_dict:
        raise ValueError(f"Invalid g_type '{g_type}'. Choose from {list(g_dict.keys())}.")
    g_fn = g_dict[g_type]

    p = len(beta)
    Sigma = np.full((p, p), 0.3)
    np.fill_diagonal(Sigma, 1)

    U = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n)
    if U.ndim == 1:
        U = U.reshape(-1, p)

    V = norm.cdf(U)

    if x_dist == 'normal':
        X = U
    elif x_dist == 'uniform':
        X = 5 * V - 2.5
    elif x_dist == 't':
        X = t.ppf(V, df=5)
    else:
        raise ValueError(f"Invalid x_dist '{x_dist}'. Choose 'normal', 'uniform', or 't'.")

    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    z3 = np.random.binomial(1, 0.5, size=n) * 2 - 1
    Z = np.column_stack([z1, z2, z3])

    xb = X @ beta
    gxb = g_fn(xb)

    if outcome == 'continuous':
        y = gxb + Z @ gamma + np.random.normal(size=n)
    elif outcome == 'binary':
        logits = gxb + Z @ gamma
        prob = 1 / (1 + np.exp(-logits))
        y = bernoulli.rvs(prob)
    elif outcome == 'cox':
        lin_pred = gxb + Z @ gamma
        baseline_lambda = 1.0
        true_lambda = baseline_lambda * np.exp(lin_pred)
        T = expon.rvs(scale=1 / true_lambda)
        C = expon.rvs(scale=np.median(1 / true_lambda) / censoring_rate, size=n)
        time = np.minimum(T, C)
        event = (T <= C).astype(int)
        y = np.column_stack([time, event])
    else:
        raise ValueError(f"Invalid outcome '{outcome}'.")

    return X, Z, y, xb, gxb, g_fn
