import numpy as np
from scipy.stats import multivariate_normal

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.sqrt(np.sum(beta**2))
gamma = np.array([1, -0.5, 0.5])

def simulate_data(n, g_type='sigmoid', seed=0):
    g_dict = {
        'linear': lambda x, a=1: a*x,
        'sigmoid': lambda x, a=2: (1/(1+np.exp(-a*x))-0.5)*5,
        'sfun': lambda x: (2/(1+np.exp(-x))-0.2*x-1)*10,
        'logsquare': lambda x: np.log(1 + x**2)
    }

    g_fn = g_dict[g_type]

    ##Exposures and covaraites 
    #correlation matrix for each group
    p = len(beta)
    mat1 = np.full((p, p), 0.3)
    np.fill_diagonal(mat1, 1)

    np.random.seed(seed)
    # Generate multivariate normal data
    x = multivariate_normal.rvs(mean=np.zeros(p), cov=mat1, size=n)

    # Generate additional variables
    z1 = np.random.normal(size=n)
    z2 = np.random.normal(size=n)
    z3 = np.random.binomial(1, 0.5, size=n)

    # Combine variables
    z = np.column_stack([z1, z2, z3])

    xb = x @ beta
    gxb = g_fn(xb)
    y = gxb + z @ gamma + np.random.normal(size=n)
    return x, z, y, xb, gxb, g_fn

