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
    'time': [],
    'pred_mse': [],
    'g_pred': [],
    'beta_bias': [],
    'gamma_bias': []
}

models = {
    'PLSI': SplinePLSI,
    'nPLSI': neuralPLSI,
}

g_grid = np.linspace(-3, 3, 1000)
for i, g_fn in enumerate(['linear', 'logsquare', 'sfun', 'sigmoid']):
    for j, n in enumerate([500, 2000]):
        for seed in range(5):
            X, Z, y, xb, gxb, true_g_fn = simulate_data(n*2, g_type=g_fn, seed=seed)

            X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=42)

            for model_name, model_class in models.items():
                np.random.seed(seed)

                if model_name == "PLSI":
                    model = model_class(X_train.shape[1], 3)
                else:
                    model = model_class()        

                start = perf_counter()
                model.fit(X_train, Z_train, y_train)
                end = perf_counter()
                preds = model.predict(X_test, Z_test)

                res['n'].append(n)
                res['g_fn'].append(g_fn)
                res['model'].append(model_name)
                res['seed'].append(seed)
                res['pred_mse'].append(np.mean((preds - y_test)**2))
                res['beta_bias'].append((beta - model.beta).tolist() if hasattr(model, 'beta') else [None]*len(beta))
                res['gamma_bias'].append((gamma - model.gamma).tolist())
                res['g_pred'].append(model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None]*len(g_grid))
                res['time'].append(end - start)
            
                print(g_fn, n, seed, model_name, hasattr(model, 'g_function'), end - start)

with open('output/simulator_results.json', 'w') as f:
    json.dump(res, f)