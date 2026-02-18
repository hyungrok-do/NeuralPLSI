#!/usr/bin/env python3
"""
Ablation study: GLM warm-start vs random init across all 9 outcome × g-function combos.

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

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

warnings.filterwarnings("ignore")

from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA
from models import NeuralPLSI

# ---------- config ----------
G_GRID = np.linspace(-3, 3, 500)


def l2_norm_g(g_true_fn, g_est_vals, grid):
    """Integrated L2 norm of (g_true - g_est) on a discrete grid."""
    g_true_vals = g_true_fn(grid)
    diff = g_true_vals - g_est_vals
    dx = grid[1] - grid[0]
    return np.sqrt(np.sum(diff**2) * dx)


def run_one(outcome, g_type, seed, warmstart, n):
    """Fit a single replicate. Returns (beta, gamma, g_est, g_true_fn, time)."""
    X, Z, y, xb, gxb, g_fn = simulate_data(n, outcome=outcome, g_type=g_type, seed=seed)

    model = NeuralPLSI(family=outcome, max_epoch=200, warmstart=warmstart)

    t0 = perf_counter()
    model.fit(X, Z, y, random_state=seed)
    elapsed = perf_counter() - t0

    g_est = model.g_function(G_GRID)

    return model.beta, model.gamma, g_est, g_fn, elapsed


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=500)
    ap.add_argument('--n_reps', type=int, default=20)
    ap.add_argument('--outcome', type=str, default=None, help='If set, run only this outcome')
    ap.add_argument('--g_fn', type=str, default=None, help='If set, run only this g_fn')
    ap.add_argument('--out', type=str, default=None)
    args = ap.parse_args()

    outcomes = [args.outcome] if args.outcome else ['continuous', 'binary', 'cox']
    g_types  = [args.g_fn] if args.g_fn else ['linear', 'sigmoid', 'sfun']

    header = f"{'outcome':<12} {'g_type':<10} {'warmstart':<10} {'MAB(β)':<10} {'MAB(γ)':<10} {'L2(g)':<10} {'time(s)':<8}"
    print(header)
    print("=" * len(header))

    rows = []
    for outcome in outcomes:
        for g_type in g_types:
            for ws in [False, True]:
                betas, gammas, l2s, times = [], [], [], []
                for rep in range(args.n_reps):
                    seed = rep * 100
                    try:
                        b, g, g_est, g_fn, t = run_one(outcome, g_type, seed, ws, args.n)
                        betas.append(b)
                        gammas.append(g)
                        l2s.append(l2_norm_g(g_fn, g_est, G_GRID))
                        times.append(t)
                    except Exception as e:
                        print(f"  SKIP {outcome}/{g_type}/ws={ws}/rep={rep}: {e}")

                if len(betas) == 0:
                    continue

                betas = np.array(betas)
                gammas = np.array(gammas)
                mab_beta  = np.mean(np.abs(betas - TRUE_BETA))
                mab_gamma = np.mean(np.abs(gammas - TRUE_GAMMA))
                mean_l2   = np.mean(l2s)
                mean_t    = np.mean(times)

                tag = "YES" if ws else "NO"
                print(f"{outcome:<12} {g_type:<10} {tag:<10} {mab_beta:<10.4f} {mab_gamma:<10.4f} {mean_l2:<10.4f} {mean_t:<8.2f}")
                rows.append({
                    'outcome': outcome, 'g_type': g_type, 'warmstart': ws,
                    'mab_beta': round(mab_beta, 5), 'mab_gamma': round(mab_gamma, 5),
                    'l2_g': round(mean_l2, 5), 'time_s': round(mean_t, 3),
                    'n_reps': len(betas),
                    'beta_bias': (np.mean(betas, axis=0) - TRUE_BETA).tolist(),
                    'gamma_bias': (np.mean(gammas, axis=0) - TRUE_GAMMA).tolist(),
                })

    # Save results
    out_dir = args.out or os.path.join(ROOT, "reproduce", "output")
    os.makedirs(out_dir, exist_ok=True)
    suffix = ""
    if args.outcome:
        suffix += f"_{args.outcome}"
    if args.g_fn:
        suffix += f"_{args.g_fn}"
    out_path = os.path.join(out_dir, f"ablation_warmstart{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
