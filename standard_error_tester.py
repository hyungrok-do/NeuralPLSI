import numpy as np
from scipy.stats import multivariate_normal

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from models.PLSI import SplinePLSI
from models.nPLSI import neuralPLSI
from simulation import simulate_data, beta, gamma

items = ['n', 'g_fn', 'model', 'seed', 'pred_mse', 'g_pred', 'beta_bias', 'gamma_bias', 'beta', 'gamma',
         'beta_se_sandwich', 'gamma_se_sandwich', 'beta_coverage_sandwich', 'gamma_coverage_sandwich',
         'beta_se_bootstrap', 'gamma_se_bootstrap', 'beta_coverage_bootstrap', 'gamma_coverage_bootstrap']
res = {}
for item in items:
    res[item] = []

g_grid = np.linspace(-3, 3, 1000)
for i, g_fn in enumerate(['linear', 'logsquare', 'sfun', 'sigmoid']):
    for seed in range(10):
        n = 2000
        X, Z, y, xb, gxb, true_g_fn = simulate_data(n*2, g_type=g_fn, seed=seed)

        X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=42)
        
        model = neuralPLSI()
        model.fit(X_train, Z_train, y_train)
        preds = model.predict(X_test, Z_test)

        res['n'].append(n)
        res['seed'].append(seed)
        res['model'].append('nPLSI')
        res['g_fn'].append(g_fn)
        
        res['g_pred'].append(model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None]*len(g_grid))
        
        res['pred_mse'].append(np.mean((preds - y_test)**2))

        res['beta_bias'].append((beta - model.beta).tolist() if hasattr(model, 'beta') else [None]*len(beta))
        res['gamma_bias'].append((gamma - model.gamma).tolist())

        res['beta'].append(model.beta.tolist())
        res['gamma'].append(model.gamma.tolist())

        model.inference_sandwich(X_train, Z_train, y_train)
        summary = model.summary()

        res['beta_se_sandwich'].append(model.beta_se.tolist())
        res['gamma_se_sandwich'].append(model.gamma_se.tolist())
        res['beta_coverage_sandwich'].append(((model.beta_lb <= beta) & (beta <= model.beta_ub)).astype(int).tolist())
        res['gamma_coverage_sandwich'].append(((model.gamma_lb <= gamma) & (gamma <= model.gamma_ub)).astype(int).tolist())

        model.inference_bootstrap(X_train, Z_train, y_train)
        summary = model.summary()

        res['beta_se_bootstrap'].append(model.beta_se.tolist())
        res['gamma_se_bootstrap'].append(model.gamma_se.tolist())
        res['beta_coverage_bootstrap'].append(((model.beta_lb <= beta) & (beta <= model.beta_ub)).astype(int).tolist())
        res['gamma_coverage_bootstrap'].append(((model.gamma_lb <= gamma) & (gamma <= model.gamma_ub)).astype(int).tolist())


import json
with open('output/standard_error_results.json', 'w') as f:
    json.dump(res, f)