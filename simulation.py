import numpy as np
from scipy.stats import multivariate_normal, bernoulli, expon

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.linalg.norm(beta)
gamma = np.array([1, -0.5, 0.5])

def simulate_data(n, outcome='continuous', g_type='sigmoid', censoring_rate=0.3, seed=0):
    """
    Simulates data for partial-linear single-index models with a specified outcome type.

    Parameters:
    ----------
    n : int
        Number of observations.

    outcome : str
        Outcome type. One of:
        - 'continuous': Continuous outcome (default)
        - 'binary' : Binary outcome generated via logistic model
        - 'cox'  : Time-to-event outcome from exponential survival with censoring

    g_type : str
        Nonlinear transformation function g(x) applied to the index x @ beta. One of:
        - 'linear'    : Linear function, g(x) = x
        - 'sigmoid'   : Sigmoidal transformation, g(x) = (1 / (1 + exp(-2x)) - 0.5) * 5
        - 'sfun'      : Smooth nonlinear function combining sigmoid and linear terms
        - 'logsquare' : Log(1 + x^2)

    censoring_rate : float, optional
        Proportion of censoring in Cox model (only used if outcome='cox').

    seed : int, optional
        Random seed for reproducibility.

    Returns:
    -------
    x : ndarray (n, p)
        High-dimensional covariates.

    z : ndarray (n, q)
        Low-dimensional linear covariates.

    y : ndarray
        Outcome:
        - Continuous vector for 'continuous'
        - Binary vector for 'binary'
        - (n, 2) array [time, event] for 'cox'

    xb : ndarray (n,)
        Linear index x @ beta.

    gxb : ndarray (n,)
        Nonlinear index g(x @ beta).

    g_fn : callable
        The nonlinear function g(x) used in the simulation.
    """
    np.random.seed(seed)

    # Define nonlinear transformations
    g_dict = {
        'linear': lambda x, a=1: a * x,
        'sigmoid': lambda x, a=2: (1 / (1 + np.exp(-a * x)) - 0.5) * 5,
        'sfun': lambda x: (2 / (1 + np.exp(-x)) - 0.2 * x - 1) * 10,
        'logsquare': lambda x: np.log(1 + x**2)
    }
    if g_type not in g_dict:
        raise ValueError(f"Invalid g_type '{g_type}'. Choose from {list(g_dict.keys())}.")
    g_fn = g_dict[g_type]

    # High-dimensional exposures X ~ N(0, Σ) with correlation 0.3
    p = len(beta)
    Sigma = np.full((p, p), 0.3)
    np.fill_diagonal(Sigma, 1)
    x = multivariate_normal.rvs(mean=np.zeros(p), cov=Sigma, size=n)

    # Low-dimensional covariates Z
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    z3 = np.random.binomial(1, 0.5, size=n) * 2 - 1 # to standardize
    z = np.column_stack([z1, z2, z3])

    # Single index and nonlinear transformation
    xb = x @ beta
    gxb = g_fn(xb)

    # Outcome generation
    if outcome == 'continuous':
        y = gxb + z @ gamma + np.random.normal(size=n)

    elif outcome == 'binary':
        logits = gxb + z @ gamma
        prob = 1 / (1 + np.exp(-logits))
        y = bernoulli.rvs(prob)

    elif outcome == 'cox':
        lin_pred = gxb + z @ gamma
        baseline_lambda = 1.0
        true_lambda = baseline_lambda * np.exp(lin_pred)
        T = expon.rvs(scale=1 / true_lambda)

        # Censoring
        C = expon.rvs(scale= true_lambda / (1 - censoring_rate), size=n)
        time = np.minimum(T, C)
        event = (T <= C).astype(int)
        y = np.column_stack([time, event])

    else:
        raise ValueError(f"Invalid outcome '{outcome}'. Choose from 'continuous', 'binary', or 'cox'.")

    return x, z, y, xb, gxb, g_fn
