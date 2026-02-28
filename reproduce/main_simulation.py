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


def output_path(base, model, n, g_fn, outcome, x_dist, warmstart, initial=False):
    """Resolve the JSON output path for a model run."""
    ws_tag = 'ws1' if warmstart else 'ws0'
    init_tag = 'init1' if initial else 'init0'
    name = f"simulation+{model}+{n}+{g_fn}+{outcome}+{x_dist}+{ws_tag}+{init_tag}.json"
    if base is None:
        os.makedirs("output", exist_ok=True)
        return os.path.join("output", name)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


MODEL_CLS = {'PLSI': SplinePLSI, 'NeuralPLSI': NeuralPLSI}


def main():
    ap = argparse.ArgumentParser(description="Simulation study: PLSI vs NeuralPLSI inference comparison.")
    ap.add_argument('--n_instances',    type=int,   default=500)
    ap.add_argument('--n_replicates',   type=int,   default=1000)
    ap.add_argument('--n_bootstrap',    type=int,   default=100)
    ap.add_argument('--g_fn',           type=str,   default='sigmoid', choices=['linear', 'sfun', 'sigmoid'])
    ap.add_argument('--outcome',        type=str,   default='continuous', choices=['continuous', 'binary', 'cox'])
    ap.add_argument('--exposure_dist',  type=str,   default='normal', choices=['normal', 'uniform', 't'])
    ap.add_argument('--models',         nargs='+',  default=['all'])
    ap.add_argument('--warmstart',      type=int,   default=1, choices=[0, 1],
                    help='GLM warm-start for NeuralPLSI (1=on, 0=off)')
    ap.add_argument('--seed0',          type=int,   default=0)
    ap.add_argument('--save_every',     type=int,   default=1)
    ap.add_argument('--out',            type=str,   default=None)
    ap.add_argument('--initial',        type=int,   default=0, choices=[0, 1],
                    help='GLM initial solution for NeuralPLSI beta+g (1=on, 0=off)')
    ap.add_argument('--g_grid_min',     type=float, default=-3.0)
    ap.add_argument('--g_grid_max',     type=float, default=3.0)
    ap.add_argument('--g_grid_n',       type=int,   default=1000)
    ap.add_argument('--activation',     type=str,   default='Tanh')
    args = ap.parse_args()

    model_list = parse_models(args.models)
    n          = args.n_instances
    g_fn       = args.g_fn
    outcome    = args.outcome
    x_dist     = args.exposure_dist
    warmstart  = bool(args.warmstart)
    initial    = bool(args.initial)
    g_grid     = np.linspace(args.g_grid_min, args.g_grid_max, args.g_grid_n)

    results     = {m: [] for m in model_list}
    out_paths   = {m: output_path(args.out, m, n, g_fn, outcome, x_dist, warmstart, initial) for m in model_list}

    header = (f"Simulation: n={n}, g_fn={g_fn}, outcome={outcome}, "
              f"x_dist={x_dist}, warmstart={warmstart}, initial={initial}, models={model_list}")
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
                    model = NeuralPLSI(family=outcome, activation=args.activation, warmstart=warmstart, initial=initial)
                else:
                    model = SplinePLSI(family=outcome)

                # --- Fit ---
                t0 = perf_counter()
                model.fit(X_tr, Z_tr, y_tr)
                time_fit = perf_counter() - t0

                # --- Point estimates ---
                perf  = evaluate(model, X_te, Z_te, y_te, outcome)
                g_est = model.g_function(g_grid).tolist()

                entry = {
                    'seed':           seed,
                    'n':              n,
                    'g_fn':           g_fn,
                    'x_dist':         x_dist,
                    'outcome':        outcome,
                    'model':          mname,
                    'performance':    perf,
                    'beta_estimate':  model.beta.tolist(),
                    'gamma_estimate': model.gamma.tolist(),
                    'g_pred':         g_est,
                    'time_fit':       round(time_fit, 4),
                }

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

                entry['bootstrap_summary'] = to_json({
                    k: boot[k] for k in ('beta_hat', 'beta_se', 'beta_lb', 'beta_ub',
                                          'gamma_hat', 'gamma_se', 'gamma_lb', 'gamma_ub')
                    if k in boot
                })
                entry['bootstrap_g'] = to_json({
                    k: boot[k] for k in ('g_mean', 'g_se', 'g_lb', 'g_ub')
                    if k in boot
                })
                entry['time_bootstrap'] = round(time_boot, 4)
                
                parts = [f"[{rep+1:4d}/{args.n_replicates}] {mname:12s}",
                         f"perf={perf:.4f}",
                         f"fit={time_fit:.2f}s",
                         f"boot={time_boot:.2f}s"]
                print(" | ".join(parts))
                
                return entry
            except Exception as e:
                print(f"[{rep+1:4d}/{args.n_replicates}] ERROR for {mname}: {e}")
                return None

        # Execute replicates in parallel for the current model
        print(f"--- Running {args.n_replicates} replicates for {mname} in parallel ---")
        rep_results = Parallel(n_jobs=-1)(
            delayed(_run_single_rep)(rep) for rep in range(args.n_replicates)
        )
        rep_results = [r for r in rep_results if r is not None]
        results[mname] = rep_results

    # Final save
    for mname in model_list:
        with open(out_paths[mname], 'w') as f:
            json.dump(results[mname], f)
        print(f"Saved {len(results[mname])} entries → {out_paths[mname]}")


if __name__ == "__main__":
    main()
