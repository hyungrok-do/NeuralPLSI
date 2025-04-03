import json
import numpy as np

from time import perf_counter
from sklearn.model_selection import train_test_split

from models.PLSI import SplinePLSI
from models.nPLSI import neuralPLSI
from simulation import simulate_data, beta, gamma

res = {
    'n': [],
    'g_fn': [],
    'model': [],
    'seed': [],
    'pred_mse': [],
    'g_pred': [],
    'beta': [],
    'gamma': [],
    'std_err': [],
    'coverage': []
}

models = {
    'PLSI (3)': SplinePLSI,
    #'PLSI (5)': SplinePLSI,
    'nPLSI': neuralPLSI,
}

g_grid = np.linspace(-3, 3, 1000)
for i, g_fn in enumerate(['linear', 'logsquare', 'sfun', 'sigmoid']):
    for j, n in enumerate([500, 2000]):
        for seed in range(5):
            X, Z, y, xb, gxb, true_g_fn = simulate_data(n*2, g_type=g_fn, seed=seed)

            X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=42)

            np.random.seed(seed)
            for model_name, model_class in models.items():
                
                if 'PLSI' in model_name:
                    if '3' in model_name:
                        model = model_class(X_train.shape[1], 3)
                    elif '5' in model_name:
                        model = model_class(X_train.shape[1], 5)
                    else:
                        model = model_class()        

            model.fit(X_train, Z_train, y_train)

            res['n'].append(n)
            res['g_fn'].append(g_fn)
            res['model'].append(model_name)
            res['seed'].append(seed)
            res['pred_mse'].append(np.mean((preds - y_test)**2))
            res['beta_bias'].append((beta - model.beta).tolist() if hasattr(model, 'beta') else [None]*len(beta))
            res['gamma_bias'].append((gamma - model.gamma).tolist())
            res['g_pred'].append(model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None]*len(g_grid))

            for _ in range(100):
                bootstrap_idx = np.random.choice(range(n), size=n, replace=True)
                X_bootstrap = X_train[bootstrap_idx]
                Z_bootstrap = Z_train[bootstrap_idx]
                y_bootstrap = y_train[bootstrap_idx]

                model.fit(X_train, Z_train, y_train)
                preds = model.predict(X_test, Z_test)

                
            
            print(g_fn, n, seed, model_name, hasattr(model, 'g_function'), end - start)


with open('output/simulator_results.json', 'w') as f:
    json.dump(res, f)