#!/usr/bin/env python3
"""
Simulation study comparing SplinePLSI and NeuralPLSI with bootstrap inference.

For each replicate the script records:
  - point estimates  (beta, gamma, g-function)
  - standard errors  and 95 % confidence intervals
  - computation time  for fitting and bootstrap inference
  - predictive performance  (MSE / AUC / C-index)
"""

import argparse
import json
import os
import sys
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SplinePLSI, NeuralPLSI
from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA


def to_json(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj


def evaluate(model, X_te, Z_te, y_te, outcome):
    """Compute predictive performance on the test set."""
    if outcome == 'continuous':
        preds = model.predict(X_te, Z_te)
        return float(np.mean((preds - y_te) ** 2))
    elif outcome == 'binary':
        preds = model.predict_proba(X_te, Z_te)
        return float(roc_auc_score(y_te, preds))
    elif outcome == 'cox':
        preds = model.predict_partial_hazard(X_te, Z_te)
        return float(concordance_index(y_te[:, 0], -preds, y_te[:, 1]))
    raise ValueError(f"Unknown outcome: {outcome}")


def parse_models(arg):
    """Parse model names from CLI into canonical keys."""
    alias = {'plsi': 'PLSI', 'splineplsi': 'PLSI',
             'neuralplsi': 'NeuralPLSI', 'nplsi': 'NeuralPLSI'}
    if len(arg) == 1 and arg[0].lower() in ('all', 'both'):
        return ['PLSI', 'NeuralPLSI']
    out = []
    for m in arg:
        key = alias.get(m.strip().lower())
        if key is None:
            raise ValueError(f"Unknown model: {m}. Use PLSI, NeuralPLSI, or all.")
        if key not in out:
            out.append(key)
    return out


def output_path(base, model, n, g_fn, outcome, x_dist, initial=False):
    """Resolve the JSON output path for a model run."""
    init_tag = 'init1' if initial else 'init0'
    name = f"simulation+{model}+{n}+{g_fn}+{outcome}+{x_dist}+{init_tag}.json"
    if base is None:
        os.makedirs("output/simulation", exist_ok=True)
        return os.path.join("output/simulation", name)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


MODEL_CLS = {'PLSI': SplinePLSI, 'NeuralPLSI': NeuralPLSI}


def main():
    ap = argparse.ArgumentParser(description="Simulation study: PLSI vs NeuralPLSI inference comparison.")
    ap.add_argument('--n_instances',    type=int,   default=500)
    ap.add_argument('--n_replicates',   type=int,   default=500)
    ap.add_argument('--n_bootstrap',    type=int,   default=100)
    ap.add_argument('--g_fn',           type=str,   default='sigmoid', choices=['linear', 'sfun', 'sigmoid'])
    ap.add_argument('--outcome',        type=str,   default='continuous', choices=['continuous', 'binary', 'cox'])
    ap.add_argument('--exposure_dist',  type=str,   default='normal', choices=['normal', 'uniform', 't'])
    ap.add_argument('--models',         nargs='+',  default=['all'])
    ap.add_argument('--seed0',          type=int,   default=0)
    ap.add_argument('--save_every',     type=int,   default=1)
    ap.add_argument('--out',            type=str,   default=None)
    ap.add_argument('--g_grid_min',     type=float, default=-3.0)
    ap.add_argument('--g_grid_max',     type=float, default=3.0)
    ap.add_argument('--g_grid_n',       type=int,   default=1000)
    ap.add_argument('--activation',     type=str,   default='ELU')
    args = ap.parse_args()

    model_list = parse_models(args.models)
    n          = args.n_instances
    g_fn       = args.g_fn
    outcome    = args.outcome
    x_dist     = args.exposure_dist
    g_grid     = np.linspace(args.g_grid_min, args.g_grid_max, args.g_grid_n)

    results     = {m: [] for m in model_list}
    out_paths   = {m: output_path(args.out, m, n, g_fn, outcome, x_dist) for m in model_list}

    header = (f"Simulation: n={n}, g_fn={g_fn}, outcome={outcome}, "
              f"x_dist={x_dist}, models={model_list}")
    print(header)
    print("=" * len(header))

    for mname in model_list:
        def _run_single_rep(rep):
            seed = args.seed0 + rep
            try:
                X, Z, y, _, _, _ = simulate_data(n * 2, outcome=outcome, g_type=g_fn, seed=seed, x_dist=x_dist)
                X_tr, X_te, Z_tr, Z_te, y_tr, y_te = train_test_split(X, Z, y, test_size=n, random_state=seed)

                np.random.seed(seed)

                if mname == 'NeuralPLSI':
                    model = NeuralPLSI(family=outcome, activation=args.activation)
                else:
                    model = SplinePLSI(family=outcome)

                # --- Fit ---
                t0 = perf_counter()
                model.fit(X_tr, Z_tr, y_tr)
                time_fit = perf_counter() - t0

                # --- Point estimates ---
                perf  = evaluate(model, X_te, Z_te, y_te, outcome)
                g_est = model.g_function(g_grid).tolist()

                # --- Bootstrap inference ---
                t0 = perf_counter()
                boot = model.inference_bootstrap(
                    X_tr, Z_tr, y_tr,
                    n_samples=args.n_bootstrap,
                    random_state=seed,
                    ci=0.95,
                    g_grid=g_grid,
                )
                time_boot = perf_counter() - t0

                # --- Evaluate Metrics Directly ---
                b_hat = model.beta
                g_hat = model.gamma
                
                b_lb, b_ub = np.array(boot.get('beta_lb', b_hat)), np.array(boot.get('beta_ub', b_hat))
                g_lb, g_ub = np.array(boot.get('gamma_lb', g_hat)), np.array(boot.get('gamma_ub', g_hat))
                
                b_cov = ((b_lb <= TRUE_BETA) & (TRUE_BETA <= b_ub)).astype(int).tolist()
                g_cov = ((g_lb <= TRUE_GAMMA) & (TRUE_GAMMA <= g_ub)).astype(int).tolist()

                entry = {
                    'seed':           seed,
                    'n':              n,
                    'g_fn':           g_fn,
                    'x_dist':         x_dist,
                    'outcome':        outcome,
                    'model':          mname,
                    'performance':    perf,
                    'time_fit':       round(time_fit, 4),
                    'time_bootstrap': round(time_boot, 4),
                    
                    # Direct Metrics
                    'beta_est':       b_hat.tolist(),
                    'gamma_est':      g_hat.tolist(),
                    'beta_bias':      (b_hat - TRUE_BETA).tolist(),
                    'gamma_bias':     (g_hat - TRUE_GAMMA).tolist(),
                    'beta_se':        boot.get('beta_se', np.zeros_like(b_hat)).tolist(),
                    'gamma_se':       boot.get('gamma_se', np.zeros_like(g_hat)).tolist(),
                    'beta_cov':       b_cov,
                    'gamma_cov':      g_cov,
                    
                    # Used for empirical g-function plots
                    'g_pred':         g_est,
                }
                
                parts = [f"[{rep+1:4d}/{args.n_replicates}] {mname:12s}",
                         f"perf={perf:.4f}",
                         f"fit={time_fit:.2f}s",
                         f"boot={time_boot:.2f}s"]
                print(" | ".join(parts))
                
                return entry
            except Exception as e:
                print(f"[{rep+1:4d}/{args.n_replicates}] ERROR for {mname}: {e}")
                return None

        # Execute replicates sequentially (model internals handle parallel bootstrapping)
        print(f"--- Running {args.n_replicates} replicates for {mname} ---")
        
        # Load existing progress if file exists to allow resuming
        rep_results = []
        if os.path.exists(out_paths[mname]):
            try:
                with open(out_paths[mname], 'r') as f:
                    rep_results = json.load(f)
                print(f"Loaded {len(rep_results)} existing results for {mname}.")
            except Exception as e:
                print(f"Could not load existing file {out_paths[mname]}: {e}")

        completed_seeds = {r['seed'] for r in rep_results}
        
        for rep in range(args.n_replicates):
            seed = args.seed0 + rep
            if seed in completed_seeds:
                continue
                
            res = _run_single_rep(rep)
            if res is not None:
                rep_results.append(res)
                
                # Checkpointing
                if len(rep_results) % args.save_every == 0:
                    with open(out_paths[mname], 'w') as f:
                        json.dump(rep_results, f)
        
        # Final flush
        with open(out_paths[mname], 'w') as f:
            json.dump(rep_results, f)
        print(f"Saved {len(rep_results)} entries → {out_paths[mname]}")


if __name__ == "__main__":
    main()
len(rep_results)} entries → {out_paths[mname]}")


if __name__ == "__main__":
    main()
