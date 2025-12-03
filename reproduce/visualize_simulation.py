#!/usr/bin/env python3
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
import sys

try:
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data
except ImportError:
    # Add root directory to path to allow imports
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

sns.set_style("whitegrid")

def true_g(x, g_fn):
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)

out_dir = Path("output")
files = sorted([f for f in out_dir.glob("simulation+*.json") if f.is_file()])

if not files:
    raise FileNotFoundError("No simulation+*.json files found in output/")

print(f"Found {len(files)} result files:")
for f in files:
    print("  -", f.name)

records = []
for f in files:
    with open(f, "r") as fh:
        rec = json.load(fh)
    rec["_file"] = f
    try:
        rec["_model"] = rec.get("model", ["Unknown"])[0]
        rec["_n"] = str(rec.get("n", ["?"])[0])
        rec["_g_fn"] = rec.get("g_fn", ["?"])[0]
        rec["_x_dist"] = rec.get("x_dist", ["normal"])[0]
        
        if "outcome" in rec and len(rec["outcome"]) > 0:
            rec["_outcome"] = rec["outcome"][0]
        else:
            parts = f.stem.split("+")
            if len(parts) >= 5:
                rec["_outcome"] = parts[4]
            else:
                rec["_outcome"] = "?"
                
    except IndexError:
        print(f"Warning: Could not parse metadata from {f.name}")
        rec["_model"] = "Error"
        rec["_n"] = "?"
        rec["_g_fn"] = "?"
        rec["_outcome"] = "?"
        rec["_x_dist"] = "?"

    records.append(rec)

summaries = []
for rec in records:
    model = rec["_model"]
    n = rec["_n"]
    g_fn = rec["_g_fn"]
    outcome = rec["_outcome"]
    exposure_dist = rec["_x_dist"]

    beta_est = np.array([np.array(b) for b in rec["beta_estimate"]])
    gamma_est = np.array([np.array(g) for g in rec["gamma_estimate"]])
    beta_boot = [np.array(b) for b in rec["beta_bootstrap"]]
    gamma_boot = [np.array(g) for g in rec["gamma_bootstrap"]]

    beta_bias = beta_est.mean(axis=0) - TRUE_BETA
    gamma_bias = gamma_est.mean(axis=0) - TRUE_GAMMA
    beta_sd = beta_est.std(axis=0, ddof=1)
    gamma_sd = gamma_est.std(axis=0, ddof=1)
    beta_se_boot = np.mean([b.std(axis=0, ddof=1) for b in beta_boot], axis=0)
    gamma_se_boot = np.mean([g.std(axis=0, ddof=1) for g in gamma_boot], axis=0)

    def cov(estimates, boots, true):
        coverage_list = []
        for est, boot in zip(estimates, boots):
            if boot.size == 0: 
                continue
            l, h = np.percentile(boot, [2.5, 97.5], axis=0)
            covered = (l <= true) & (true <= h)
            coverage_list.append(covered)
        if not coverage_list:
            return np.full_like(true, np.nan), 0
        return np.mean(coverage_list, axis=0), len(coverage_list)

    beta_cov, n_rep_beta = cov(beta_est, beta_boot, TRUE_BETA)
    gamma_cov, n_rep_gamma = cov(gamma_est, gamma_boot, TRUE_GAMMA)

    n_replicates = max(n_rep_beta, n_rep_gamma) if n_rep_beta > 0 or n_rep_gamma > 0 else 0
    
    df = pd.DataFrame({
        "param": [f"beta_{i}" for i in range(len(TRUE_BETA))] +
                 [f"gamma_{i}" for i in range(len(TRUE_GAMMA))],
        "bias": np.concatenate([beta_bias, gamma_bias]),
        "empirical_sd": np.concatenate([beta_sd, gamma_sd]),
        "bootstrap_se": np.concatenate([beta_se_boot, gamma_se_boot]),
        "coverage": np.concatenate([beta_cov, gamma_cov]),
        "n_replicates": n_replicates,
        "model": model,
        "n": n, "g_fn": g_fn, "outcome": outcome, "exposure_dist": exposure_dist
    })
    summaries.append(df)

summary = pd.concat(summaries, ignore_index=True)
summary = summary[summary["model"] == "NeuralPLSI"]

wide_tables = []
for (n, g_fn, outcome, exposure_dist), grp in summary.groupby(["n", "g_fn", "outcome", "exposure_dist"]):
    models = grp["model"].unique()
    pivot = grp.pivot(index="param", columns="model", values=["bias","empirical_sd","bootstrap_se","coverage", "n_replicates"])
    pivot.columns = [f"{m}_{c}" for c, m in pivot.columns]
    pivot.reset_index(inplace=True)
    pivot.insert(0, "exposure_dist", exposure_dist)
    pivot.insert(0, "outcome", outcome)
    pivot.insert(0, "g_fn", g_fn)
    pivot.insert(0, "n", n)
    wide_tables.append(pivot)

summary_table = pd.concat(wide_tables, ignore_index=True)
summary_table.to_csv(out_dir / "summary_table.csv", index=False)
print(f"\nSaved side-by-side table: output/summary_table.csv")

latex_path = out_dir / "summary_table.tex"
summary_table.to_latex(latex_path, index=False, float_format="%.4f")
print(f"Saved LaTeX table: {latex_path}")

outcomes = set(r["_outcome"] for r in records)
exposure_dists = set(r["_x_dist"] for r in records)
g_map = {"linear": "Linear", "sfun": "s-Shaped", "sigmoid": "Sigmoid"}
g_functions = ["linear", "sfun", "sigmoid"]

def plot_panels(outcome=None, exposure_dist=None, true_only=False):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    
    for i, g_fn in enumerate(g_functions):
        ax = axes[i]
        x = np.linspace(-3, 3, 1000)
        g_true = true_g(x, g_fn)
        ax.plot(x, g_true, color="black", lw=2, ls="--", label="True")
        
        if not true_only:
            styles = [("500", "cornflowerblue", "NeuralPLSI (N=500)"), ("2000", "darkblue", "NeuralPLSI (N=2000)")]
            for n_val, color, label in styles:
                recs = [r for r in records if r["_model"] == "NeuralPLSI" and str(r["_n"]) == n_val and r["_g_fn"] == g_fn and r["_outcome"] == outcome and r["_x_dist"] == exposure_dist]
                if not recs: continue
                rec = recs[0]
                g_preds = [np.array(g) for g in rec["g_pred"] if len(g) > 0]
                if not g_preds: continue
                G = np.vstack(g_preds)
                mean = G.mean(axis=0)
                lb, ub = np.percentile(G, [2.5, 97.5], axis=0)
                ax.fill_between(x, lb, ub, color=color, alpha=0.3)
                ax.plot(x, mean, color=color, lw=2, label=label)

        ax.set_ylim(-4.5, 4.5)
        ax.set_title(g_map.get(g_fn, g_fn))
        ax.set_xlabel("Index")
        if i == 0: ax.set_ylabel("g(Index)")
        ax.legend()

    plt.tight_layout()
    if true_only:
        fname = "gplot_true.png"
    else:
        fname = f"gplot_panels+{outcome}+{exposure_dist}.png"
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print("Saved:", fname)

for outcome in outcomes:
    for exposure_dist in exposure_dists:
        plot_panels(outcome, exposure_dist, true_only=False)

plot_panels(true_only=True)
