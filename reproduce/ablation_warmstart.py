#!/usr/bin/env python3
"""
Ablation study: 4-way NeuralPLSI (initial × warmstart) + PLSI (ws=0 only).

Reports per-combination:
  - MAB(β)  : mean absolute bias of β
  - MAB(γ)  : mean absolute bias of γ
  - L2(g)   : integrated L2 norm ‖g_true − g_hat‖₂  (on a grid)
"""
import argparse
import json
import os, sys, warnings
import numpy as np
from time import perf_counter
from joblib import Parallel, delayed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA
from models import NeuralPLSI, SplinePLSI

G_GRID = np.linspace(-3, 3, 500)


def l2_norm_g(g_true_fn, g_est_vals, grid):
    g_true_vals = g_true_fn(grid)
    diff = g_true_vals - g_est_vals
    dx = grid[1] - grid[0]
    return np.sqrt(np.sum(diff**2) * dx)


def run_nplsi(outcome, g_type, seed, warmstart, initial, n):
    X, Z, y, xb, gxb, g_fn = simulate_data(n, outcome=outcome, g_type=g_type, seed=seed)
    model = NeuralPLSI(family=outcome, max_epoch=200, warmstart=warmstart, initial=initial)
    t0 = perf_counter()
    model.fit(X, Z, y, random_state=seed)
    elapsed = perf_counter() - t0
    g_est = model.g_function(G_GRID)
    return model.beta, model.gamma, g_est, g_fn, elapsed


def run_plsi(outcome, g_type, seed, n):
    X, Z, y, xb, gxb, g_fn = simulate_data(n, outcome=outcome, g_type=g_type, seed=seed)
    model = SplinePLSI(family=outcome)
    t0 = perf_counter()
    model.fit(X, Z, y)
    elapsed = perf_counter() - t0
    return model.beta, model.gamma, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--n_reps', type=int, default=20)
    ap.add_argument('--outcome', type=str, default=None)
    ap.add_argument('--g_fn', type=str, default=None)
    ap.add_argument('--out', type=str, default=None)
    ap.add_argument('--ws', type=int, default=None, help='If set, only run this warmstart value (0 or 1)')
    ap.add_argument('--init', type=int, default=None, help='If set, only run this initial value (0 or 1)')
    args = ap.parse_args()

    outcomes = [args.outcome] if args.outcome else ['continuous', 'binary', 'cox']
    g_types  = [args.g_fn] if args.g_fn else ['linear', 'sigmoid', 'sfun']

    # Build combo list (default: full 4-way grid)
    if args.ws is not None and args.init is not None:
        combos = [(bool(args.ws), bool(args.init))]
    else:
        combos = [(ws, init) for ws in [False, True] for init in [False, True]]

    header = f"{'model':<12} {'outcome':<12} {'g_type':<10} {'ws':<4} {'init':<5} {'MAB(β)':<10} {'MAB(γ)':<10} {'L2(g)':<10} {'time(s)':<8}"
    print(header)
    print("=" * len(header))

    rows = []
    for outcome in outcomes:
        for g_type in g_types:
            # --- NeuralPLSI ablation ---
            for ws, init in combos:
                    def _run_single_nplsi(rep):
                        seed = rep * 100
                        try:
                            b, g, g_est, g_fn, t = run_nplsi(outcome, g_type, seed, ws, init, args.n)
                            return b, g, l2_norm_g(g_fn, g_est, G_GRID), t
                        except Exception as e:
                            print(f"  SKIP NeuralPLSI/{outcome}/{g_type}/ws={ws}/init={init}/rep={rep}: {e}")
                            return None

                    results = []
                    for rep in range(args.n_reps):
                        res = _run_single_nplsi(rep)
                        if res is not None:
                            results.append(res)

                    if len(results) == 0:
                        continue
                    
                    betas = np.array([r[0] for r in results])
                    gammas = np.array([r[1] for r in results])
                    l2s = [r[2] for r in results]
                    times = [r[3] for r in results]

                    mab_beta  = np.mean(np.abs(betas - TRUE_BETA))
                    mab_gamma = np.mean(np.abs(gammas - TRUE_GAMMA))
                    mean_l2   = np.mean(l2s); mean_t = np.mean(times)
                    ws_tag = "Y" if ws else "N"; init_tag = "Y" if init else "N"
                    print(f"{'NeuralPLSI':<12} {outcome:<12} {g_type:<10} {ws_tag:<4} {init_tag:<5} {mab_beta:<10.4f} {mab_gamma:<10.4f} {mean_l2:<10.4f} {mean_t:<8.2f}")
                    rows.append({
                        'model': 'NeuralPLSI', 'outcome': outcome, 'g_type': g_type,
                        'warmstart': ws, 'initial': init,
                        'mab_beta': round(mab_beta, 5), 'mab_gamma': round(mab_gamma, 5),
                        'l2_g': round(mean_l2, 5), 'time_s': round(mean_t, 3),
                        'n_reps': len(betas),
                        'beta_bias': (np.mean(betas, axis=0) - TRUE_BETA).tolist(),
                        'gamma_bias': (np.mean(gammas, axis=0) - TRUE_GAMMA).tolist(),
                    })

            # --- PLSI: ws=0 only (skip if running a specific combo) ---
            if args.ws is None and args.init is None:
                def _run_single_plsi(rep):
                    seed = rep * 100
                    try:
                        return run_plsi(outcome, g_type, seed, args.n)
                    except Exception as e:
                        print(f"  SKIP PLSI/{outcome}/{g_type}/rep={rep}: {e}")
                        return None
                        
                results = []
                for rep in range(args.n_reps):
                    res = _run_single_plsi(rep)
                    if res is not None:
                        results.append(res)

                if len(results) > 0:
                    betas = np.array([r[0] for r in results])
                    gammas = np.array([r[1] for r in results])
                    times = [r[2] for r in results]

                    mab_beta  = np.mean(np.abs(betas - TRUE_BETA))
                    mab_gamma = np.mean(np.abs(gammas - TRUE_GAMMA))
                    mean_t    = np.mean(times)
                    print(f"{'PLSI':<12} {outcome:<12} {g_type:<10} {'N':<4} {'N':<5} {mab_beta:<10.4f} {mab_gamma:<10.4f} {'—':<10} {mean_t:<8.2f}")
                    rows.append({
                        'model': 'PLSI', 'outcome': outcome, 'g_type': g_type,
                        'warmstart': False, 'initial': False,
                        'mab_beta': round(mab_beta, 5), 'mab_gamma': round(mab_gamma, 5),
                        'time_s': round(mean_t, 3), 'n_reps': len(betas),
                        'beta_bias': (np.mean(betas, axis=0) - TRUE_BETA).tolist(),
                        'gamma_bias': (np.mean(gammas, axis=0) - TRUE_GAMMA).tolist(),
                    })

    out_dir = args.out or os.path.join(ROOT, "reproduce", "output")
    os.makedirs(out_dir, exist_ok=True)
    suffix = ""
    if args.outcome: suffix += f"_{args.outcome}"
    if args.g_fn: suffix += f"_{args.g_fn}"
    if args.ws is not None and args.init is not None:
        suffix += f"_ws{args.ws}_init{args.init}"
    out_path = os.path.join(out_dir, f"ablation_init{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
