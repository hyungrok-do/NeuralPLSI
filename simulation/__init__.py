import numpy as np
from scipy.stats import multivariate_normal, bernoulli, expon, norm, t

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.linalg.norm(beta)
gamma = np.array([1, -0.5, 0.5])

def simulate_data(
    n,
    outcome='continuous',
    g_type='sigmoid',
    censoring_rate=0.3,
    seed=0,
    x_dist='normal'
):
    """
    Simulates data for partial-linear single-index models using a copula-based
    correlation structure for X.

    Parameters
    ----------
    n : int
        Number of observations.
    outcome : {'continuous','binary','cox'}
        Outcome type.
    g_type : {'linear','sigmoid','sfun'}
        Nonlinear transformation g(x).
    censoring_rate : float
        Censoring fraction for Cox outcome.
    seed : int
        Random seed.
    x_dist : {'normal','uniform','t1'}
        Marginal distribution for X:
        - 'normal'  : multivariate normal N(0, Σ)
        - 'uniform' : Gaussian copula → Uniform[-1,1] margins
        - 't'      : Gaussian copula → Student-t(df=1) margins

    Returns
    -------
    x, z, y, xb, gxb, g_fn : see original docstring
    """
    np.random.seed(seed)

    # --- Define nonlinear transformations ---
    g_dict = {
        'linear': lambda x, a=1: a * x,
        'sigmoid': lambda x, a=2: (1 / (1 + np.exp(-a * x)) - 0.5) * 5,
        'sfun': lambda x: (2 / (1 + np.exp(-x)) - 0.2 * x - 1) * 10,
    }
    if g_type not in g_dict:
        raise ValueError(f"Invalid g_type '{g_type}'. Choose from {list(g_dict.keys())}.")
    g_fn = g_dict[g_type]

    # --- High-dimensional exposures X (copula-based) ---
    p = len(beta)
    Sigma = np.full((p, p), 0.3)
    np.fill_diagonal(Sigma, 1)

    # Step 1: draw correlated Gaussian
    U = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n)
    if U.ndim == 1:
        U = U.reshape(-1, p)

    # Step 2: convert to uniform(0,1) margins via Gaussian CDF
    V = norm.cdf(U)

    # Step 3: transform to desired marginals
    if x_dist == 'normal':
        X = U  # already correct
    elif x_dist == 'uniform':
        X = 2 * V - 1  # Uniform[-1,1]
    elif x_dist == 't':
        # heavy-tailed t(1) margins (Cauchy-like)
        X = t.ppf(V, df=1)
    else:
        raise ValueError(f"Invalid x_dist '{x_dist}'. Choose 'normal', 'uniform', or 't'.")

    # --- Low-dimensional covariates Z ---
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    z3 = np.random.binomial(1, 0.5, size=n) * 2 - 1  # {-1, +1}
    Z = np.column_stack([z1, z2, z3])

    # --- Single index and nonlinear transformation ---
    xb = X @ beta
    gxb = g_fn(xb)

    # --- Outcome generation ---
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
        C = expon.rvs(scale=true_lambda / (1 - censoring_rate), size=n)
        time = np.minimum(T, C)
        event = (T <= C).astype(int)
        y = np.column_stack([time, event])

    else:
        raise ValueError(f"Invalid outcome '{outcome}'.")

    return X, Z, y, xb, gxb, g_fn
