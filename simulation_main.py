import argparse
import json
import numpy as np

from time import perf_counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from models.PLSI import SplinePLSI
from models.nPLSI import neuralPLSI
from simulation import simulate_data, beta, gamma
from lifelines.utils import concordance_index


# === Initialize result dictionary ===
res = {
    'n': [],
    'g_fn': [],
    'model': [],
    'seed': [],
    'performance': [],
    'g_pred': [],
    'beta_estimate': [],
    'beta_bootstrap': [],
    'gamma_estimate': [],
    'gamma_bootstrap': [],
    'time': []
}

models = {
    'PLSI': SplinePLSI,
    'NeuralPLSI': neuralPLSI,
}

# === Parse arguments ===
parser = argparse.ArgumentParser(description='Run repeated simulations for PLSI and NeuralPLSI models.')
parser.add_argument('--n_instances', type=int, default=500, help='Number of observations for each simulation.')
parser.add_argument('--n_replicates', type=int, default=100, help='Number of simulation replicates.')
parser.add_argument('--n_bootstrap', type=int, default=100, help='Number of bootstrap samples.')
parser.add_argument('--g_fn', type=str, default='sigmoid', choices=['linear', 'sfun', 'sigmoid'], help='Nonlinear function g(x) to use in the simulation.')
parser.add_argument('--outcome', type=str, default='continuous', choices=['continuous', 'binary', 'cox'], help='Type of outcome variable.')
parser.add_argument('--model', type=str, default='NeuralPLSI', choices=['PLSI', 'NeuralPLSI'], help='Model to use for the simulation.')
args = parser.parse_args()

n = args.n_instances
g_fn = args.g_fn
outcome = args.outcome

g_grid = np.linspace(-3, 3, 1000)
model_name = args.model
model_class = models[model_name]
# === Main simulation loop ===
for seed in range(args.n_replicates):
    X, Z, y, xb, gxb, true_g_fn = simulate_data(n * 2, outcome=outcome, g_type=g_fn, seed=seed)
    X_train, X_test, Z_train, Z_test, y_train, y_test = train_test_split(X, Z, y, test_size=n, random_state=seed)

    np.random.seed(seed)
    model = model_class(family=outcome)

    # === Fit on original training data ===
    start = perf_counter()
    model.fit(X_train, Z_train, y_train)
    end = perf_counter()

    # === Evaluate model ===
    if outcome == 'continuous':
        preds = model.predict(X_test, Z_test)
        res['performance'].append(np.mean((preds - y_test) ** 2))
    elif outcome == 'binary':
        preds = model.predict_proba(X_test, Z_test)
        res['performance'].append(roc_auc_score(y_test, preds))
    elif outcome == 'cox':
        preds = model.predict_partial_hazard(X_test, Z_test)
        res['performance'].append(concordance_index(y_test[:, 0], -preds, y_test[:, 1]))

    # === Store original parameter estimates ===
    orig_beta = model.beta.tolist() if hasattr(model, 'beta') else [None] * len(beta)
    orig_gamma = model.gamma.tolist()

    beta_boot = []
    gamma_boot = []

    for _ in range(args.n_bootstrap):
        bootstrap_idx = np.random.choice(range(len(X_train)), size=n, replace=True)
        X_bootstrap = X_train[bootstrap_idx]
        Z_bootstrap = Z_train[bootstrap_idx]
        y_bootstrap = y_train[bootstrap_idx]

        model_b = model_class(family=outcome)
        model_b.fit(X_bootstrap, Z_bootstrap, y_bootstrap)

        beta_boot.append(model_b.beta.tolist())
        gamma_boot.append(model_b.gamma.tolist())

    res['n'].append(n)
    res['g_fn'].append(g_fn)
    res['model'].append(model_name)
    res['seed'].append(seed)
    res['beta_estimate'].append(orig_beta)
    res['gamma_estimate'].append(orig_gamma)
    res['beta_bootstrap'].append(beta_boot)
    res['gamma_bootstrap'].append(gamma_boot)
    res['g_pred'].append(model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None] * len(g_grid))
    res['time'].append(end - start)

output_path = f'output/simulation+{model_name}+{n}+{g_fn}+{outcome}.json'
with open(output_path, 'w') as f:
    json.dump(res, f, indent=4)
