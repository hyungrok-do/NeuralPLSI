import numpy as np
from scipy.stats import multivariate_normal

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.BKMR import BKMR
from models.PLSI import KernelPLSI, SplinePLSI
from models.nPLSI import neuralPLSI

from time import perf_counter

n = 1000

def simulate_data(n, g_type='sigmoid', seed=0):
    g_dict = {
        'linear': lambda x, a=1: a*x,
        'sigmoid': lambda x, a=2: (1/(1+np.exp(-a*x))-0.5)*5,
        'sfun': lambda x: (2/(1+np.exp(-x))-0.2*x-1)*10,
        'logsquare': lambda x: np.log(1 + x**2)
    }

    g_fn = g_dict[g_type]
    
    beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
    beta = beta / np.sqrt(np.sum(beta**2))
    gamma = np.array([1, -0.5, 0.5])

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

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.sqrt(np.sum(beta**2))
gamma = np.array([1, -0.5, 0.5])

res = {
    'n': [],
    'g_fn': [],
    'model': [],
    'seed': [],
    'time': [],
    'pred_mse': [],
    'g_pred': [],
    'beta_bias': [],
    'gamma_bias': []
}

models = {
    'PLSI (3)': SplinePLSI,
    'PLSI (5)': SplinePLSI,
    'nPLSI': neuralPLSI,
    #'BKMR': BKMR
}

g_grid = np.linspace(-3, 3, 1000)
for i, g_fn in enumerate(['linear', 'logsquare', 'sfun', 'sigmoid']):
    for seed in range(1):
        n = 2000
        X, Z, y, xb, gxb, true_g_fn = simulate_data(n*2, g_type=g_fn, seed=seed)

        X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=42)
        
        model = neuralPLSI()
        model.fit(X_train, Z_train, y_train)
        print(model.summary())
            
        '''
        bootstrap_beta, bootstrap_gamma = [], []
        for _ in range(100):
            bootstrap_idx = np.random.choice(range(n), size=n, replace=True)
            X_bootstrap = X_train[bootstrap_idx]
            Z_bootstrap = Z_train[bootstrap_idx]
            y_bootstrap = y_train[bootstrap_idx]

            model = neuralPLSI()
            model.fit(X_bootstrap, Z_bootstrap, y_bootstrap)
            bootstrap_beta.append(model.beta)
            bootstrap_gamma.append(model.gamma)

        bootstrap_beta = np.stack(bootstrap_beta, axis=0)
        bootstrap_gamma = np.stack(bootstrap_gamma, axis=0)

        bootstrap_beta_mean = np.mean(bootstrap_beta, axis=0)
        bootstrap_gamma_mean = np.mean(bootstrap_gamma, axis=0)
        bootstrap_beta_se = np.std(bootstrap_beta, axis=0)
        bootstrap_gamma_se = np.std(bootstrap_gamma, axis=0)
        bootstrap_beta_ci_ub = np.quantile(bootstrap_beta, 0.975, axis=0)
        bootstrap_beta_ci_lb = np.quantile(bootstrap_beta, 0.025, axis=0)
        bootstrap_gamma_ci_ub = np.quantile(bootstrap_gamma, 0.975, axis=0)
        bootstrap_gamma_ci_lb = np.quantile(bootstrap_gamma, 0.025, axis=0)

        bootstrap_summary = pd.DataFrame(
            {
                'Parameter': [f'beta_{i:02d}' for i in range(bootstrap_beta.shape[1])] + [f'gamma_{i:02d}' for i in range(bootstrap_gamma.shape[1])],
                'Coefficients': bootstrap_beta_mean.tolist() + bootstrap_gamma_mean.tolist(),
                'Standard Error': bootstrap_beta_se.tolist() + bootstrap_gamma_se.tolist(),
                '95% CI Lower Bound': bootstrap_beta_ci_lb.tolist() + bootstrap_gamma_ci_lb.tolist(),
                '95% CI Upper Bound': bootstrap_beta_ci_ub.tolist() + bootstrap_gamma_ci_ub.tolist()
            }
        )

        print(bootstrap_summary)
      '''
        