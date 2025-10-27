#!/usr/bin/env python3
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt, seaborn as sns
from pathlib import Path
from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

sns.set_style("whitegrid")

# -------------------------------------------------------------------
# Utility: True g(x)
# -------------------------------------------------------------------
def true_g(x, g_fn):
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)

# -------------------------------------------------------------------
# Load all simulation JSONs
# -------------------------------------------------------------------
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
    # Parse file name: simulation+Model+N+g_fn+outcome+x_dist.json
    parts = f.stem.split("+")
    if len(parts) >= 6:
        rec["_model"] = parts[1]
        rec["_n"] = parts[2]
        rec["_g_fn"] = parts[3]
        rec["_outcome"] = parts[4]
        rec["_exposure_dist"] = parts[5]
    elif len(parts) >= 5:
        # Backward compatibility for old filenames without x_dist
        rec["_model"] = parts[1]
        rec["_n"] = parts[2]
        rec["_g_fn"] = parts[3]
        rec["_outcome"] = parts[4]
        rec["_exposure_dist"] = "normal"
    else:
        rec["_model"] = "Unknown"
        rec["_n"] = "?"
        rec["_g_fn"] = "?"
        rec["_outcome"] = "?"
        rec["_exposure_dist"] = "?"
    records.append(rec)

# -------------------------------------------------------------------
# Compute parameter summaries per model file
# -------------------------------------------------------------------
summaries = []
for rec in records:
    model = rec["_model"]
    n = rec["_n"]
    g_fn = rec["_g_fn"]
    outcome = rec["_outcome"]
    exposure_dist = rec["_exposure_dist"]

    beta_est = np.array([np.array(b) for b in rec["beta_estimate"]])
    gamma_est = np.array([np.array(g) for g in rec["gamma_estimate"]])
    beta_boot = [np.array(b) for b in rec["beta_bootstrap"]]
    gamma_boot = [np.array(g) for g in rec["gamma_bootstrap"]]

    # Compute stats
    beta_bias = beta_est.mean(axis=0) - TRUE_BETA
    gamma_bias = gamma_est.mean(axis=0) - TRUE_GAMMA
    beta_sd = beta_est.std(axis=0, ddof=1)
    gamma_sd = gamma_est.std(axis=0, ddof=1)
    beta_se_boot = np.mean([b.std(axis=0, ddof=1) for b in beta_boot], axis=0)
    gamma_se_boot = np.mean([g.std(axis=0, ddof=1) for g in gamma_boot], axis=0)

    # Coverage via percentile CI
    def cov(estimates, boots, true):
        """
        For each replicate, check if the bootstrap CI covers the true value.
        Returns the proportion of replicates with coverage (empirical coverage rate).
        """
        coverage_list = []
        for est, boot in zip(estimates, boots):
            if boot.size == 0: 
                continue
            # Bootstrap CI for this replicate
            l, h = np.percentile(boot, [2.5, 97.5], axis=0)
            # Check if true value is within CI for each parameter
            covered = (l <= true) & (true <= h)
            coverage_list.append(covered)
        if not coverage_list:
            return np.full_like(true, np.nan), 0
        # Average coverage across replicates
        return np.mean(coverage_list, axis=0), len(coverage_list)
    beta_cov, n_rep_beta = cov(beta_est, beta_boot, TRUE_BETA)
    gamma_cov, n_rep_gamma = cov(gamma_est, gamma_boot, TRUE_GAMMA)

    # Get number of replicates (should be same for beta and gamma)
    n_replicates = max(n_rep_beta, n_rep_gamma) if n_rep_beta > 0 or n_rep_gamma > 0 else 0
    
    # Combine
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

# -------------------------------------------------------------------
# Pivot side-by-side PLSI vs NeuralPLSI
# -------------------------------------------------------------------
wide_tables = []
for (n, g_fn, outcome, exposure_dist), grp in summary.groupby(["n", "g_fn", "outcome", "exposure_dist"]):
    models = grp["model"].unique()
    # Get n_replicates (should be same for all params in this group)
    n_replicates = grp["n_replicates"].iloc[0]
    pivot = grp.pivot(index="param", columns="model", values=["bias","empirical_sd","bootstrap_se","coverage"])
    pivot.columns = [f"{m}_{c}" for c, m in pivot.columns]
    pivot.reset_index(inplace=True)
    pivot.insert(0, "n_replicates", n_replicates)
    pivot.insert(0, "exposure_dist", exposure_dist)
    pivot.insert(0, "outcome", outcome)
    pivot.insert(0, "g_fn", g_fn)
    pivot.insert(0, "n", n)
    wide_tables.append(pivot)

summary_table = pd.concat(wide_tables, ignore_index=True)
summary_table.to_csv(out_dir / "summary_table.csv", index=False)
print(f"\nSaved side-by-side table: output/summary_table.csv")

# -------------------------------------------------------------------
# g(x) overlay plots
# -------------------------------------------------------------------
for (n, g_fn, outcome, exposure_dist), grp in summary.groupby(["n", "g_fn", "outcome", "exposure_dist"]):
    plt.figure(figsize=(6, 5))
    x = np.linspace(-3, 3, 1000)
    g_true = true_g(x, g_fn)
    plt.plot(x, g_true, color="black", lw=2, ls="--", label="True")

    for model in ["NeuralPLSI", "PLSI"]:
        recs = [r for r in records if r["_model"] == model and r["_n"] == n and r["_g_fn"] == g_fn and r["_outcome"] == outcome and r["_exposure_dist"] == exposure_dist]
        if not recs:
            continue
        g_preds = [np.array(g) for g in recs[0]["g_pred"] if len(g)]
        if not g_preds:
            continue
        G = np.vstack(g_preds)
        mean = G.mean(axis=0)
        lb, ub = np.percentile(G, [2.5, 97.5], axis=0)
        color = sns.color_palette("pastel")[['NeuralPLSI', 'PLSI'].index(model)]
        plt.fill_between(x, lb, ub, color=color, alpha=0.3, label=f"{model}")
        plt.plot(x, mean, color=color, lw=2)

    #plt.title(f"g(x) overlay — n={n}, g_fn={g_fn}, outcome={outcome}, x_dist={exposure_dist}")
    plt.xlabel("x")
    plt.ylabel("g(x)")
    plt.legend()
    plt.tight_layout()
    fname = f"gplot+{n}+{g_fn}+{outcome}+{exposure_dist}.png"
    plt.savefig(out_dir / fname, dpi=150)
    plt.close()
    print("Saved:", fname)

