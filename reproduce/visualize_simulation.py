#!/usr/bin/env python3
"""
Simulation Visualization Script

Restructured to produce:
1. Summary tables (CSV + LaTeX) with bias, SD, SE, and coverage
2. Bar charts comparing estimates across scenarios
3. Bootstrap vs Hessian inference comparison for NeuralPLSI
4. g-function recovery plots with both NeuralPLSI and PLSI overlay
"""
import os, json, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

try:
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data
except ImportError:
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)
    from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 150

# ============================================================================
# Data Loading
# ============================================================================
out_dir = Path("output")
files = sorted([f for f in out_dir.glob("simulation+*.json") if f.is_file()])

if not files:
    raise FileNotFoundError("No simulation+*.json files found in output/")

print(f"Found {len(files)} result files")

def load_results():
    """Load all simulation result files."""
    records = []
    for f in files:
        try:
            with open(f, "r") as fh:
                rec = json.load(fh)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  Skipping {f.name}: {e}")
            continue
        rec["_file"] = str(f)
        rec["_model"] = rec.get("model", ["Unknown"])[0]
        rec["_n"] = str(rec.get("n", ["?"])[0])
        rec["_g_fn"] = rec.get("g_fn", ["?"])[0]
        rec["_x_dist"] = rec.get("x_dist", ["normal"])[0]
        if "outcome" in rec and len(rec["outcome"]) > 0:
            rec["_outcome"] = rec["outcome"][0]
        else:
            parts = Path(f).stem.split("+")
            rec["_outcome"] = parts[4] if len(parts) >= 5 else "?"
        records.append(rec)
    return records

records = load_results()

# ============================================================================
# Helper Functions
# ============================================================================
def true_g(x, g_fn):
    """Get true g-function values at x."""
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)

def compute_coverage(estimates, lb_list, ub_list, true_val):
    """Compute coverage from lower/upper bounds."""
    coverage = []
    for est, lb, ub in zip(estimates, lb_list, ub_list):
        if lb is None or ub is None:
            continue
        lb, ub = np.array(lb), np.array(ub)
        covered = (lb <= true_val) & (true_val <= ub)
        coverage.append(covered)
    if not coverage:
        return np.full_like(true_val, np.nan, dtype=float)
    return np.mean(coverage, axis=0)

def extract_inference(rec, inf_type):
    """
    Extract inference summary from record.
    inf_type: 'hessian' or 'bootstrap'
    Handles both old format (inference_summary) and new format (hessian_summary/bootstrap_summary).
    """
    # New format
    if f"{inf_type}_summary" in rec:
        return rec[f"{inf_type}_summary"]
    # Old format fallback
    if inf_type == "hessian" and "inference_summary" in rec:
        return rec["inference_summary"]
    if inf_type == "bootstrap" and "inference_summary" in rec:
        return rec["inference_summary"]
    return []

# ============================================================================
# 1. Summary Statistics Computation
# ============================================================================
def compute_summaries(records):
    """Compute summary statistics for all scenarios, both Hessian and Bootstrap."""
    summaries = []
    
    for rec in records:
        model = rec["_model"]
        n = rec["_n"]
        g_fn = rec["_g_fn"]
        outcome = rec["_outcome"]
        x_dist = rec["_x_dist"]
        n_reps = len(rec.get("seed", []))
        
        beta_est = np.array([np.array(b) for b in rec.get("beta_estimate", [])])
        gamma_est = np.array([np.array(g) for g in rec.get("gamma_estimate", [])])
        
        if beta_est.size == 0 or gamma_est.size == 0:
            continue
        
        beta_bias = beta_est.mean(axis=0) - TRUE_BETA
        gamma_bias = gamma_est.mean(axis=0) - TRUE_GAMMA
        beta_emp_sd = beta_est.std(axis=0, ddof=1)
        gamma_emp_sd = gamma_est.std(axis=0, ddof=1)
        
        # Process both inference types
        for inf_type in ["hessian", "bootstrap"]:
            inf_summaries = extract_inference(rec, inf_type)
            
            # Skip if no data for this inference type
            if not inf_summaries or all(not s for s in inf_summaries):
                continue
            
            beta_se_list, gamma_se_list = [], []
            beta_lb_list, beta_ub_list = [], []
            gamma_lb_list, gamma_ub_list = [], []
            
            for inf in inf_summaries:
                if not inf:
                    continue
                beta_se_list.append(inf.get("beta_se"))
                gamma_se_list.append(inf.get("gamma_se"))
                beta_lb_list.append(inf.get("beta_lb"))
                beta_ub_list.append(inf.get("beta_ub"))
                gamma_lb_list.append(inf.get("gamma_lb"))
                gamma_ub_list.append(inf.get("gamma_ub"))
            
            valid_beta_se = [s for s in beta_se_list if s is not None]
            valid_gamma_se = [s for s in gamma_se_list if s is not None]
            
            if not valid_beta_se and not valid_gamma_se:
                continue
            
            beta_se_avg = np.mean(valid_beta_se, axis=0) if valid_beta_se else np.full(len(TRUE_BETA), np.nan)
            gamma_se_avg = np.mean(valid_gamma_se, axis=0) if valid_gamma_se else np.full(len(TRUE_GAMMA), np.nan)
            
            beta_cov = compute_coverage(beta_est, beta_lb_list, beta_ub_list, TRUE_BETA)
            gamma_cov = compute_coverage(gamma_est, gamma_lb_list, gamma_ub_list, TRUE_GAMMA)
            
            for i, (bias, emp_sd, se, cov) in enumerate(zip(beta_bias, beta_emp_sd, beta_se_avg, beta_cov)):
                summaries.append({
                    "model": model, "n": n, "g_fn": g_fn, "outcome": outcome, "x_dist": x_dist,
                    "param": f"beta_{i}", "true_val": TRUE_BETA[i],
                    "bias": bias, "emp_sd": emp_sd, "se": se, "coverage": cov,
                    "n_reps": n_reps, "inference_type": inf_type.capitalize()
                })
            
            for i, (bias, emp_sd, se, cov) in enumerate(zip(gamma_bias, gamma_emp_sd, gamma_se_avg, gamma_cov)):
                summaries.append({
                    "model": model, "n": n, "g_fn": g_fn, "outcome": outcome, "x_dist": x_dist,
                    "param": f"gamma_{i}", "true_val": TRUE_GAMMA[i],
                    "bias": bias, "emp_sd": emp_sd, "se": se, "coverage": cov,
                    "n_reps": n_reps, "inference_type": inf_type.capitalize()
                })
    
    return pd.DataFrame(summaries)

print("Computing summaries...")
summary_df = compute_summaries(records)

# ============================================================================
# 2. Save Summary Tables
# ============================================================================
def save_summary_tables(df):
    """Save summary tables in CSV and LaTeX format."""
    if df.empty:
        print("No summary data to save.")
        return
    
    df.to_csv(out_dir / "summary_table.csv", index=False)
    print(f"Saved: output/summary_table.csv")
    
    agg_df = df.groupby(["model", "g_fn", "outcome", "x_dist", "inference_type"]).agg({
        "bias": lambda x: np.mean(np.abs(x)),
        "emp_sd": "mean",
        "se": "mean", 
        "coverage": "mean",
        "n_reps": "first"
    }).reset_index()
    agg_df.columns = ["Model", "g_fn", "Outcome", "X_dist", "Inference", "MAB", "Emp_SD", "SE", "Coverage", "N_reps"]
    agg_df.to_csv(out_dir / "summary_aggregated.csv", index=False)
    print(f"Saved: output/summary_aggregated.csv")
    
    latex_df = agg_df.copy()
    latex_df["MAB"] = latex_df["MAB"].apply(lambda x: f"{x:.4f}")
    latex_df["Emp_SD"] = latex_df["Emp_SD"].apply(lambda x: f"{x:.4f}")
    latex_df["SE"] = latex_df["SE"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
    latex_df["Coverage"] = latex_df["Coverage"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
    latex_df.to_latex(out_dir / "summary_table.tex", index=False, escape=False)
    print(f"Saved: output/summary_table.tex")

save_summary_tables(summary_df)

# ============================================================================
# 2b. Per-Setting Tables (parameters as rows, Bias/SE/SD/Coverage as columns)
# ============================================================================
def save_per_setting_tables(df):
    """Save per-setting tables with parameters as rows and metrics as columns."""
    if df.empty:
        print("No data for per-setting tables.")
        return
    
    # Create output directory for per-setting tables
    tables_dir = out_dir / "tables"
    tables_dir.mkdir(exist_ok=True)
    
    # Group by setting
    groupby_cols = ["model", "g_fn", "outcome", "x_dist", "inference_type"]
    
    for setting, grp in df.groupby(groupby_cols):
        model, g_fn, outcome, x_dist, inf_type = setting
        
        # Create table with parameters as rows
        table_df = grp[["param", "bias", "se", "emp_sd", "coverage"]].copy()
        table_df = table_df.rename(columns={
            "param": "Parameter",
            "bias": "Bias",
            "se": "SE",
            "emp_sd": "Emp_SD",
            "coverage": "Coverage"
        })
        table_df = table_df.set_index("Parameter")
        
        # Format values
        fmt_df = table_df.copy()
        fmt_df["Bias"] = fmt_df["Bias"].apply(lambda x: f"{x:.4f}")
        fmt_df["SE"] = fmt_df["SE"].apply(lambda x: f"{x:.4f}" if pd.notna(x) else "-")
        fmt_df["Emp_SD"] = fmt_df["Emp_SD"].apply(lambda x: f"{x:.4f}")
        fmt_df["Coverage"] = fmt_df["Coverage"].apply(lambda x: f"{x:.2%}" if pd.notna(x) else "-")
        
        # Filename
        fname_base = f"table+{model}+{g_fn}+{outcome}+{x_dist}+{inf_type}"
        
        # Save CSV (raw values)
        table_df.to_csv(tables_dir / f"{fname_base}.csv")
        
        # Save LaTeX (formatted)
        fmt_df.to_latex(tables_dir / f"{fname_base}.tex", escape=False)
    
    print(f"Saved: {len(df.groupby(groupby_cols))} per-setting tables to output/tables/")

save_per_setting_tables(summary_df)

# ============================================================================
# 3. Bar Charts: Bias Comparison
# ============================================================================
def plot_bias_bars(df):
    """Create bar charts comparing bias across models and scenarios."""
    if df.empty:
        return
    
    beta_df = df[df["param"].str.startswith("beta")].copy()
    
    for outcome in beta_df["outcome"].unique():
        for x_dist in beta_df["x_dist"].unique():
            subset = beta_df[(beta_df["outcome"] == outcome) & (beta_df["x_dist"] == x_dist)]
            if subset.empty:
                continue
            
            g_fns = subset["g_fn"].unique()
            models = subset["model"].unique()
            
            fig, axes = plt.subplots(1, len(g_fns), figsize=(5*len(g_fns), 5), sharey=True)
            if len(g_fns) == 1:
                axes = [axes]
            
            for ax, g_fn in zip(axes, g_fns):
                g_subset = subset[subset["g_fn"] == g_fn]
                pivot = g_subset.groupby(["param", "model"])["bias"].mean().unstack()
                
                x = np.arange(len(pivot.index))
                width = 0.35
                
                for i, model in enumerate(models):
                    if model in pivot.columns:
                        offset = (i - len(models)/2 + 0.5) * width
                        color = "steelblue" if model == "NeuralPLSI" else "coral"
                        ax.bar(x + offset, pivot[model], width, label=model, color=color, alpha=0.8)
                
                ax.axhline(y=0, color='black', linestyle='--', linewidth=0.8)
                ax.set_xticks(x)
                ax.set_xticklabels(pivot.index, rotation=45, ha='right')
                ax.set_title(f"g_fn = {g_fn}")
                ax.set_ylabel("Bias" if ax == axes[0] else "")
                ax.legend()
            
            fig.suptitle(f"Parameter Bias: {outcome}, X~{x_dist}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"bar_bias+{outcome}+{x_dist}.png"
            plt.savefig(out_dir / fname)
            plt.close()
            print(f"Saved: {fname}")

plot_bias_bars(summary_df)

# ============================================================================
# 4. SE Comparison: Hessian vs Bootstrap (for NeuralPLSI)
# ============================================================================
def plot_hessian_vs_bootstrap(df):
    """Compare Hessian vs Bootstrap SE for NeuralPLSI."""
    if df.empty:
        return
    
    neural_df = df[df["model"] == "NeuralPLSI"].copy()
    if neural_df.empty:
        return
    
    beta_df = neural_df[neural_df["param"].str.startswith("beta")]
    
    for outcome in beta_df["outcome"].unique():
        for x_dist in beta_df["x_dist"].unique():
            subset = beta_df[(beta_df["outcome"] == outcome) & (beta_df["x_dist"] == x_dist)]
            if subset.empty:
                continue
            
            g_fns = subset["g_fn"].unique()
            
            fig, axes = plt.subplots(1, len(g_fns), figsize=(5*len(g_fns), 5), sharey=True)
            if len(g_fns) == 1:
                axes = [axes]
            
            for ax, g_fn in zip(axes, g_fns):
                g_subset = subset[subset["g_fn"] == g_fn]
                
                hess_data = g_subset[g_subset["inference_type"] == "Hessian"]
                boot_data = g_subset[g_subset["inference_type"] == "Bootstrap"]
                
                if hess_data.empty or boot_data.empty:
                    ax.text(0.5, 0.5, "Missing data", ha='center', va='center', transform=ax.transAxes)
                    ax.set_title(f"g_fn = {g_fn}")
                    continue
                
                params = hess_data["param"].unique()
                x = np.arange(len(params))
                width = 0.35
                
                hess_se = [hess_data[hess_data["param"] == p]["se"].values[0] for p in params]
                boot_se = [boot_data[boot_data["param"] == p]["se"].values[0] if p in boot_data["param"].values else np.nan for p in params]
                
                ax.bar(x - width/2, hess_se, width, label="Hessian", color="steelblue", alpha=0.8)
                ax.bar(x + width/2, boot_se, width, label="Bootstrap", color="coral", alpha=0.8)
                
                ax.set_xticks(x)
                ax.set_xticklabels(params, rotation=45, ha='right')
                ax.set_title(f"g_fn = {g_fn}")
                if ax == axes[0]:
                    ax.set_ylabel("Standard Error")
                ax.legend()
            
            fig.suptitle(f"NeuralPLSI: Hessian vs Bootstrap SE ({outcome}, X~{x_dist})", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"bar_hess_vs_boot+{outcome}+{x_dist}.png"
            plt.savefig(out_dir / fname)
            plt.close()
            print(f"Saved: {fname}")

plot_hessian_vs_bootstrap(summary_df)

# ============================================================================
# 5. Coverage Comparison: Hessian vs Bootstrap
# ============================================================================
def plot_coverage_comparison(df):
    """Compare coverage between Hessian and Bootstrap."""
    if df.empty:
        return
    
    cov_df = df.groupby(["model", "g_fn", "outcome", "x_dist", "inference_type"]).agg({
        "coverage": "mean"
    }).reset_index()
    
    for outcome in cov_df["outcome"].unique():
        subset = cov_df[cov_df["outcome"] == outcome]
        if subset.empty:
            continue
        
        fig, ax = plt.subplots(figsize=(14, 5))
        
        x_dists = sorted(subset["x_dist"].unique())
        g_fns = sorted(subset["g_fn"].unique())
        
        groups = [(g, d) for d in x_dists for g in g_fns]
        x = np.arange(len(groups))
        width = 0.2
        
        combos = subset[["model", "inference_type"]].drop_duplicates().values.tolist()
        colors = {"NeuralPLSI": {"Hessian": "steelblue", "Bootstrap": "skyblue"},
                  "PLSI": {"Hessian": "coral", "Bootstrap": "lightsalmon"}}
        
        for i, (model, inf_type) in enumerate(combos):
            vals = []
            for g, d in groups:
                row = subset[(subset["g_fn"] == g) & (subset["x_dist"] == d) & 
                            (subset["model"] == model) & (subset["inference_type"] == inf_type)]
                vals.append(row["coverage"].values[0] if len(row) > 0 else np.nan)
            
            offset = (i - len(combos)/2 + 0.5) * width
            color = colors.get(model, {}).get(inf_type, "gray")
            ax.bar(x + offset, vals, width, label=f"{model} ({inf_type})", color=color, alpha=0.8)
        
        ax.axhline(y=0.95, color='red', linestyle='--', linewidth=1.5, label="Nominal 95%")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{g}+{d}" for g, d in groups], rotation=45, ha='right')
        ax.set_ylabel("Coverage Rate")
        ax.set_ylim(0, 1.05)
        ax.set_title(f"Coverage Rate by Inference Method: {outcome}")
        ax.legend(loc='lower right', fontsize=8)
        
        plt.tight_layout()
        fname = f"bar_coverage+{outcome}.png"
        plt.savefig(out_dir / fname)
        plt.close()
        print(f"Saved: {fname}")

plot_coverage_comparison(summary_df)

# ============================================================================
# 6. Inference Time Comparison
# ============================================================================
def plot_inference_time(records):
    """Compare inference time between Hessian and Bootstrap."""
    time_data = []
    for rec in records:
        model = rec["_model"]
        outcome = rec["_outcome"]
        g_fn = rec["_g_fn"]
        x_dist = rec["_x_dist"]
        
        # New format
        if "time_hessian" in rec:
            for t in rec.get("time_hessian", []):
                if t > 0:
                    time_data.append({"model": model, "outcome": outcome, "g_fn": g_fn, 
                                     "x_dist": x_dist, "inference_type": "Hessian", "time": t})
        if "time_bootstrap" in rec:
            for t in rec.get("time_bootstrap", []):
                if t > 0:
                    time_data.append({"model": model, "outcome": outcome, "g_fn": g_fn, 
                                     "x_dist": x_dist, "inference_type": "Bootstrap", "time": t})
        # Old format fallback
        if "time_inference" in rec and "time_hessian" not in rec:
            inf_type = "Hessian" if model == "NeuralPLSI" else "Bootstrap"
            for t in rec.get("time_inference", []):
                time_data.append({"model": model, "outcome": outcome, "g_fn": g_fn, 
                                 "x_dist": x_dist, "inference_type": inf_type, "time": t})
    
    if not time_data:
        return
    
    time_df = pd.DataFrame(time_data)
    
    fig, ax = plt.subplots(figsize=(12, 5))
    sns.boxplot(data=time_df, x="outcome", y="time", hue="inference_type", ax=ax)
    ax.set_ylabel("Inference Time (seconds)")
    ax.set_xlabel("Outcome Type")
    ax.set_title("Inference Time: Hessian vs Bootstrap")
    ax.legend(title="Method")
    
    plt.tight_layout()
    plt.savefig(out_dir / "boxplot_inference_time.png")
    plt.close()
    print("Saved: boxplot_inference_time.png")

plot_inference_time(records)

# ============================================================================
# 7. g-Function Recovery Plots (NeuralPLSI + PLSI overlay)
# ============================================================================
def plot_g_panels():
    """Plot g-function recovery with both NeuralPLSI and PLSI."""
    g_map = {"linear": "Linear", "sfun": "S-Shaped", "sigmoid": "Sigmoid"}
    g_functions = ["linear", "sfun", "sigmoid"]
    x = np.linspace(-3, 3, 1000)
    
    outcomes = set(r["_outcome"] for r in records)
    x_dists = set(r["_x_dist"] for r in records)
    
    for outcome in outcomes:
        for x_dist in x_dists:
            fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
            
            for i, g_fn in enumerate(g_functions):
                ax = axes[i]
                g_true = true_g(x, g_fn)
                ax.plot(x, g_true, color="black", lw=2, ls="--", label="True")
                
                # Plot NeuralPLSI
                neural_recs = [r for r in records 
                              if r["_model"] == "NeuralPLSI" 
                              and r["_g_fn"] == g_fn 
                              and r["_outcome"] == outcome 
                              and r["_x_dist"] == x_dist]
                
                if neural_recs:
                    rec = neural_recs[0]
                    g_preds = [np.array(g) for g in rec.get("g_pred", []) if len(g) > 0]
                    if g_preds:
                        G = np.vstack(g_preds)
                        mean = G.mean(axis=0)
                        lb, ub = np.percentile(G, [2.5, 97.5], axis=0)
                        ax.fill_between(x, lb, ub, color="steelblue", alpha=0.2)
                        ax.plot(x, mean, color="steelblue", lw=2, label="NeuralPLSI")
                
                # Plot PLSI (overlay)
                plsi_recs = [r for r in records 
                            if r["_model"] == "PLSI" 
                            and r["_g_fn"] == g_fn 
                            and r["_outcome"] == outcome 
                            and r["_x_dist"] == x_dist]
                
                if plsi_recs:
                    rec = plsi_recs[0]
                    g_preds = [np.array(g) for g in rec.get("g_pred", []) if len(g) > 0]
                    if g_preds:
                        G = np.vstack(g_preds)
                        mean = G.mean(axis=0)
                        lb, ub = np.percentile(G, [2.5, 97.5], axis=0)
                        ax.fill_between(x, lb, ub, color="coral", alpha=0.2)
                        ax.plot(x, mean, color="coral", lw=2, label="PLSI")
                
                ax.set_ylim(-4.5, 4.5)
                ax.set_title(g_map.get(g_fn, g_fn))
                ax.set_xlabel("Index (X'β)")
                if i == 0:
                    ax.set_ylabel("g(Index)")
                ax.legend()
            
            plt.suptitle(f"g-Function Recovery: {outcome}, X~{x_dist}", fontsize=12, fontweight='bold')
            plt.tight_layout()
            fname = f"gplot_panels+{outcome}+{x_dist}.png"
            plt.savefig(out_dir / fname)
            plt.close()
            print(f"Saved: {fname}")
    
    # True g-functions only
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for i, g_fn in enumerate(g_functions):
        ax = axes[i]
        ax.plot(x, true_g(x, g_fn), color="black", lw=2, label="True g(x)")
        ax.set_ylim(-4.5, 4.5)
        ax.set_title(g_map.get(g_fn, g_fn))
        ax.set_xlabel("Index (X'β)")
        if i == 0:
            ax.set_ylabel("g(Index)")
        ax.legend()
    plt.suptitle("True g-Functions", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(out_dir / "gplot_true.png")
    plt.close()
    print("Saved: gplot_true.png")

plot_g_panels()

# ============================================================================
# 8. Hessian vs Bootstrap Detailed Comparison Table
# ============================================================================
def create_inference_comparison_table(df):
    """Create detailed comparison between Hessian and Bootstrap for NeuralPLSI."""
    if df.empty:
        return
    
    neural_df = df[df["model"] == "NeuralPLSI"]
    if neural_df.empty:
        return
    
    comparison_rows = []
    
    for (g_fn, outcome, x_dist), grp in neural_df.groupby(["g_fn", "outcome", "x_dist"]):
        hess = grp[grp["inference_type"] == "Hessian"]
        boot = grp[grp["inference_type"] == "Bootstrap"]
        
        if hess.empty or boot.empty:
            continue
        
        for param in hess["param"].unique():
            h_row = hess[hess["param"] == param]
            b_row = boot[boot["param"] == param]
            if h_row.empty or b_row.empty:
                continue
            h_row, b_row = h_row.iloc[0], b_row.iloc[0]
            
            comparison_rows.append({
                "g_fn": g_fn, "outcome": outcome, "x_dist": x_dist, "param": param,
                "Hessian_SE": h_row["se"], "Bootstrap_SE": b_row["se"],
                "Hessian_Coverage": h_row["coverage"], "Bootstrap_Coverage": b_row["coverage"],
                "Emp_SD": h_row["emp_sd"], "Bias": h_row["bias"]
            })
    
    if comparison_rows:
        comp_df = pd.DataFrame(comparison_rows)
        comp_df.to_csv(out_dir / "inference_comparison.csv", index=False)
        print("Saved: output/inference_comparison.csv")
        
        summary = comp_df.groupby(["g_fn", "outcome", "x_dist"]).agg({
            "Hessian_SE": "mean", "Bootstrap_SE": "mean",
            "Hessian_Coverage": "mean", "Bootstrap_Coverage": "mean",
            "Emp_SD": "mean"
        }).reset_index()
        summary.to_csv(out_dir / "inference_comparison_summary.csv", index=False)
        print("Saved: output/inference_comparison_summary.csv")

create_inference_comparison_table(summary_df)

print("\n✓ Visualization complete!")
