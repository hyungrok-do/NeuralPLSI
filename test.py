import json
import numpy as np

from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.PLSI import SplinePLSI
from models.nPLSI import neuralPLSI
from simulation import simulate_data, beta, gamma
from lifelines.utils import concordance_index

res = {
    'n': [],
    'g_fn': [],
    'model': [],
    'seed': [],
    'time': [],
    'performance': [],
    'g_pred': [],
    'beta_bias': [],
    'gamma_bias': []
}

models = {
    'PLSI': SplinePLSI,
    #'NeuralPLSI': neuralPLSI,
}

g_grid = np.linspace(-3, 3, 1000)
for i, g_fn in enumerate(['linear', 'logsquare', 'sfun', 'sigmoid']):
    for outcome in ['continuous', 'binary', 'cox']:
        n = 500
        X, Z, y, xb, gxb, true_g_fn = simulate_data(n*2, outcome=outcome, g_type=g_fn, seed=0)

        X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=42)

        for model_name, model_class in models.items():
            np.random.seed(0)
            model = model_class(family=outcome)

            start = perf_counter()
            model.fit(X_train, Z_train, y_train)
            end = perf_counter()

            res['n'].append(n)
            res['g_fn'].append(g_fn)
            res['model'].append(model_name)
            
            if outcome == 'continuous':
                preds = model.predict(X_test, Z_test)
                res['performance'].append(np.mean((preds - y_test)**2))
                #res['pred_mse'].append(np.mean((preds - y_test)**2))
            elif outcome == 'binary':
                preds = model.predict_proba(X_test, Z_test)
                res['performance'].append(roc_auc_score(y_test, preds))
                #res['pred_mse'].append(np.mean((preds - y_test)**2))
            elif outcome == 'cox':
                preds = model.predict_partial_hazard(X_test, Z_test)
                res['performance'].append(concordance_index(y_test[:, 0], -preds, y_test[:, 1]))

            res['beta_bias'].append((beta - model.beta).tolist() if hasattr(model, 'beta') else [None]*len(beta))
            res['gamma_bias'].append((gamma - model.gamma).tolist())
            res['g_pred'].append(model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None]*len(g_grid))
            res['time'].append(end - start)
        
            print(g_fn, n, model_name, res['performance'][-1], end - start)
