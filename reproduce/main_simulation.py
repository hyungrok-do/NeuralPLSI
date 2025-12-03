#!/usr/bin/env python3
import argparse
import json
import os
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import sys

try:
    from models import SplinePLSI, NeuralPLSI
    from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA
except ImportError:
    import sys
    # Add root directory to path to allow imports
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)

    # Local modules
    from models import SplinePLSI, NeuralPLSI
    from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA


MODELS = {
    'PLSI': SplinePLSI,
    'NeuralPLSI': NeuralPLSI,
}


def _parse_models(arg_models):
    """Return canonical list of models to run."""
    if arg_models is None:
        return ['NeuralPLSI']
    return _parse_models_list(arg_models)


def _parse_models_list(arg_models):
    if len(arg_models) == 1 and arg_models[0].lower() in {'all', 'both'}:
        return ['PLSI', 'NeuralPLSI']
    out = []
    for m in arg_models:
        ml = m.strip().lower()
        if ml in ('plsi', 'splineplsi'):
            key = 'PLSI'
        elif ml in ('neuralplsi', 'nplsi'):
            key = 'NeuralPLSI'
        else:
            raise ValueError(f"Unknown model name: {m}. Use PLSI, NeuralPLSI, or all.")
        if key not in out:
            out.append(key)
    return out


def _output_path_for_model(base_out, model_name, n, g_fn, outcome, x_dist):
    """
    Resolve per-model JSON output path.
    Includes x_dist in filename for clarity.
    """
    default_name = f"simulation+{model_name}+{n}+{g_fn}+{outcome}+{x_dist}.json"
    if base_out is None:
        os.makedirs("output", exist_ok=True)
        return os.path.join("output", default_name)

    if base_out.lower().endswith(".json"):
        root, ext = os.path.splitext(base_out)
        return f"{root}+{model_name}+{x_dist}{ext}"
    else:
        os.makedirs(base_out, exist_ok=True)
        return os.path.join(base_out, default_name)


def main():
    parser = argparse.ArgumentParser(
        description="Repeated simulations for PLSI / NeuralPLSI with efficient bootstrap (per-model output files)."
    )
    parser.add_argument('--n_instances', type=int, default=500,
                        help='Number of observations per replicate (train/test split will use n for test).')
    
    parser.add_argument('--n_replicates', type=int, default=50,
                        help='Number of simulation replicates.')
    
    parser.add_argument('--n_bootstrap', type=int, default=500,
                        help='Bootstrap refits per replicate (per model).')
    
    parser.add_argument('--g_fn', type=str, default='sigmoid',
                            choices=['linear', 'sfun', 'sigmoid'],
                        help='Nonlinear g(x) used in simulation.')
    
    parser.add_argument('--outcome', type=str, default='continuous',
                        choices=['continuous', 'binary', 'cox'],
                        help='Outcome family.')
    
    parser.add_argument('--exposure_dist', type=str, default='normal',
                        choices=['normal', 'uniform', 't'],
                        help="Distribution of exposures X : normal, uniform, or student-t (heavy-tailed).")
    
    parser.add_argument('--models', nargs='+', default=['all'],
                        help="Space-separated list of models to run, e.g., --models PLSI NeuralPLSI, or --models all.")

    parser.add_argument('--seed0', type=int, default=0,
                        help='Base random seed for replicates.')
    
    parser.add_argument('--save_every', type=int, default=1,
                        help='Write results to disk every K replicates.')
    
    parser.add_argument('--out', type=str, default=None,
                        help='Output path or directory. If multiple models, per-model files are created.')
    
    parser.add_argument('--g_grid_min', type=float, default=-3.0,
                        help='Lower bound for g(x) diagnostic grid.')
    
    parser.add_argument('--g_grid_max', type=float, default=3.0,
                        help='Upper bound for g(x) diagnostic grid.')
    
    parser.add_argument('--g_grid_n', type=int, default=1000,
                        help='Number of points in g(x) diagnostic grid (also used for spline g bootstrap).')
    
    args = parser.parse_args()

    # Determine models to run
    model_list = _parse_models(args.models)

    n = args.n_instances
    g_fn = args.g_fn
    outcome = args.outcome
    x_dist = args.exposure_dist

    # Build per-model result holders and output paths
    res_by_model = {}
    out_path_by_model = {}
    for mname in model_list:
        res_by_model[mname] = {
            'n': [], 'g_fn': [], 'x_dist': [], 'outcome': [], 'model': [], 'seed': [],
            'performance': [], 'g_pred': [],
            'beta_estimate': [], 'beta_bootstrap': [],
            'gamma_estimate': [], 'gamma_bootstrap': [],
            'time': [], 'time_bootstrap': []
        }
        out_path_by_model[mname] = _output_path_for_model(args.out, mname, n, g_fn, outcome, x_dist)

    # g diagnostic grid
    g_grid = np.linspace(args.g_grid_min, args.g_grid_max, args.g_grid_n, dtype=float)

    # Main simulation loop
    for rep in range(args.n_replicates):
        seed = args.seed0 + rep

        # Simulate once per replicate (shared for all models)
        X, Z, y, xb, gxb, true_g_fn = simulate_data(
            n * 2, outcome=outcome, g_type=g_fn, seed=seed, x_dist=x_dist
        )
        X_tr, X_te, Z_tr, Z_te, y_tr, y_te = train_test_split(
            X, Z, y, test_size=n, random_state=seed
        )

        for model_name in model_list:
            Model = MODELS[model_name]
            res = res_by_model[model_name]

            np.random.seed(seed)
            model = Model(family=outcome)

            # Fit
            t0 = perf_counter()
            model.fit(X_tr, Z_tr, y_tr)
            t1 = perf_counter()

            # Evaluate
            if outcome == 'continuous':
                preds = model.predict(X_te, Z_te)
                perf = float(np.mean((preds - y_te) ** 2))
            elif outcome == 'binary':
                preds = model.predict_proba(X_te, Z_te)
                perf = float(roc_auc_score(y_te, preds))
            elif outcome == 'cox':
                preds = model.predict_partial_hazard(X_te, Z_te)
                perf = float(concordance_index(y_te[:, 0], -preds, y_te[:, 1]))
            else:
                raise ValueError("Unexpected outcome")

            # Point estimates
            beta_hat = model.beta.tolist()
            gamma_hat = model.gamma.tolist()
            g_pred = model.g_function(g_grid).tolist() if hasattr(model, 'g_function') else [None] * len(g_grid)

            # Bootstrap
            tb0 = perf_counter()
            boot = model.inference_bootstrap(
                X_tr, Z_tr, y_tr,
                n_samples=args.n_bootstrap,
                random_state=seed,
                ci=0.95,
                g_grid=g_grid
            )
            tb1 = perf_counter()

            beta_samples = boot.get('beta_samples')
            gamma_samples = boot.get('gamma_samples')

            # Append results
            res['n'].append(n)
            res['g_fn'].append(g_fn)
            res['x_dist'].append(x_dist)
            res['outcome'].append(outcome)
            res['model'].append(model_name)
            res['seed'].append(seed)
            res['performance'].append(perf)
            res['g_pred'].append(g_pred)
            res['beta_estimate'].append(beta_hat)
            res['gamma_estimate'].append(gamma_hat)
            res['beta_bootstrap'].append(beta_samples.tolist() if beta_samples is not None else [])
            res['gamma_bootstrap'].append(gamma_samples.tolist() if gamma_samples is not None else [])
            res['time'].append(float(t1 - t0))
            res['time_bootstrap'].append(float(tb1 - tb0))

            print(f"[rep {rep+1}/{args.n_replicates}] {model_name:12s} "
                  f"performance={perf:.4f} | fit={res['time'][-1]:.3f}s | bootstrap={res['time_bootstrap'][-1]:.3f}s")

            # Checkpoint
            if (rep + 1) % args.save_every == 0:
                out_path = out_path_by_model[model_name]
                with open(out_path, 'w') as f:
                    json.dump(res, f, indent=2)

    # Final write
    for model_name, res in res_by_model.items():
        out_path = out_path_by_model[model_name]
        with open(out_path, 'w') as f:
            json.dump(res, f, indent=2)
        print(f"Saved results to {out_path}")


if __name__ == "__main__":
    main()
