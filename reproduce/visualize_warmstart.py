#!/usr/bin/env python3
"""
Visualization for NeuralPLSI Simulation Study with Warm-Start Comparison.

Produces:
  1. Summary CSV + LaTeX tables: MAB(β), MAB(γ), SE, SD, Coverage
  2. Boxplot panels: β-error by model × g_fn × outcome × warmstart
  3. Warm-start comparison: Δ from ws=0 → ws=1
  4. g-Function recovery panels
  5. Bootstrap coverage summary
"""
import json, os, sys, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)

root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root))
from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 200,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DATA_DIR = root / "output"
OUT_DIR = root / "logs" / "figures"
TABLE_DIR = root / "logs" / "tables"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load & Parse
# ---------------------------------------------------------------------------
def parse_filename(fname):
    """Parse simulation+MODEL+N+GFN+OUTCOME+XDIST+wsX.json"""
    stem = Path(fname).stem
    parts = stem.split("+")
    # format: simulation+MODEL+N+GFN+OUTCOME+XDIST+wsX
    if len(parts) == 7:
        _, model, n, g_fn, outcome, x_dist, ws_tag = parts
        warmstart = int(ws_tag.replace("ws", ""))
    else:
        # old format without warmstart
        _, model, n, g_fn, outcome, x_dist = parts[:6]
        warmstart = -1  # unknown
    return model, int(n), g_fn, outcome, x_dist, warmstart


def load_all():
    """Load all ws0/ws1 JSON files into a list of flat records."""
    rows = []
    files = sorted(DATA_DIR.glob("simulation+*+ws*.json"))
    print(f"Found {len(files)} result files with warmstart tag")
    for f in files:
        model, n, g_fn, outcome, x_dist, ws = parse_filename(f.name)
        try:
            with open(f) as fh:
                reps = json.load(fh)
        except (json.JSONDecodeError, ValueError) as e:
            print(f"  SKIP {f.name}: {e}")
            continue
        for rep in reps:
            beta_est = np.array(rep["beta_estimate"])
            gamma_est = np.array(rep["gamma_estimate"])
            beta_err = np.mean(np.abs(beta_est - TRUE_BETA))
            gamma_err = np.mean(np.abs(gamma_est - TRUE_GAMMA))

            # bootstrap coverage
            bs = rep.get("bootstrap_summary", {})
            beta_lb = np.array(bs.get("beta_lb", []))
            beta_ub = np.array(bs.get("beta_ub", []))
            gamma_lb = np.array(bs.get("gamma_lb", []))
            gamma_ub = np.array(bs.get("gamma_ub", []))
            if len(beta_lb) == len(TRUE_BETA):
                beta_cov = np.mean((TRUE_BETA >= beta_lb) & (TRUE_BETA <= beta_ub))
            else:
                beta_cov = np.nan
            if len(gamma_lb) == len(TRUE_GAMMA):
                gamma_cov = np.mean((TRUE_GAMMA >= gamma_lb) & (TRUE_GAMMA <= gamma_ub))
            else:
                gamma_cov = np.nan

            rows.append({
                "model": model,
                "n": n,
                "g_fn": g_fn,
                "outcome": outcome,
                "warmstart": ws,
                "ws_label": "WS" if ws else "No-WS",
                "seed": rep["seed"],
                "MAB_beta": beta_err,
                "MAB_gamma": gamma_err,
                "perf": rep["performance"],
                "time_fit": rep["time_fit"],
                "time_boot": rep.get("time_bootstrap", np.nan),
                "beta_cov": beta_cov,
                "gamma_cov": gamma_cov,
            })
    df = pd.DataFrame(rows)
    print(f"Loaded {len(df)} replicate rows ({df['model'].nunique()} models, "
          f"n ∈ {sorted(df['n'].unique())}, g_fn ∈ {sorted(df['g_fn'].unique())})")
    return df


df_all = load_all()

# ---------------------------------------------------------------------------
# 2. Summary Tables
# ---------------------------------------------------------------------------
def summary_table(df):
    grp = df.groupby(["model", "n", "g_fn", "outcome", "warmstart"])
    tbl = grp.agg(
        MAB_beta_mean=("MAB_beta", "mean"),
        MAB_beta_sd=("MAB_beta", "std"),
        MAB_gamma_mean=("MAB_gamma", "mean"),
        MAB_gamma_sd=("MAB_gamma", "std"),
        perf_mean=("perf", "mean"),
        perf_sd=("perf", "std"),
        beta_cov=("beta_cov", "mean"),
        gamma_cov=("gamma_cov", "mean"),
        time_fit=("time_fit", "mean"),
        time_boot=("time_boot", "mean"),
        n_reps=("seed", "count"),
    ).reset_index()
    return tbl


summary = summary_table(df_all)
summary.to_csv(TABLE_DIR / "summary_warmstart.csv", index=False)
print(f"\nSaved summary CSV → {TABLE_DIR / 'summary_warmstart.csv'}")
print(summary.to_string(index=False, max_rows=40))

# ---------------------------------------------------------------------------
# 3. LaTeX per-setting tables
# ---------------------------------------------------------------------------
def make_latex_tables(df_summary):
    for (model, n, ws), sub in df_summary.groupby(["model", "n", "warmstart"]):
        ws_tag = f"ws{ws}"
        rows_latex = []
        for _, r in sub.iterrows():
            rows_latex.append(
                f"  {r['g_fn']} & {r['outcome']} & "
                f"{r['MAB_beta_mean']:.4f} ({r['MAB_beta_sd']:.4f}) & "
                f"{r['MAB_gamma_mean']:.4f} ({r['MAB_gamma_sd']:.4f}) & "
                f"{r['beta_cov']:.2f} & {r['gamma_cov']:.2f} & "
                f"{r['time_fit']:.1f} \\\\"
            )
        body = "\n".join(rows_latex)
        tex = (
            f"% {model}, n={n}, warmstart={ws}\n"
            "\\begin{tabular}{ll rr cc r}\n"
            "\\toprule\n"
            "g\\_fn & Outcome & MAB($\\beta$) & MAB($\\gamma$) & "
            "Cov($\\beta$) & Cov($\\gamma$) & Time(s) \\\\\n"
            "\\midrule\n"
            f"{body}\n"
            "\\bottomrule\n"
            "\\end{tabular}\n"
        )
        fname = TABLE_DIR / f"table+{model}+{n}+{ws_tag}.tex"
        with open(fname, "w") as f:
            f.write(tex)
    print(f"Saved LaTeX tables → {TABLE_DIR}")


make_latex_tables(summary)

# ---------------------------------------------------------------------------
# 4. Box-plot panels: MAB(β) by model × warmstart
# ---------------------------------------------------------------------------
def plot_bias_boxplots(df):
    """3×3 grid (g_fn × outcome) with boxplots for model × ws."""
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    n_vals = sorted(df["n"].unique())

    for n_val in n_vals:
        sub = df[df["n"] == n_val]
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey="row")
        fig.suptitle(f"MAB(β)  —  n = {n_val}", fontsize=14, fontweight="bold")

        for i, gfn in enumerate(g_fns):
            for j, oc in enumerate(outcomes):
                ax = axes[i, j]
                mask = (sub["g_fn"] == gfn) & (sub["outcome"] == oc)
                d = sub[mask]
                if d.empty:
                    ax.set_visible(False)
                    continue
                sns.boxplot(
                    data=d, x="model", y="MAB_beta", hue="ws_label",
                    ax=ax, palette="Set2", showfliers=False,
                    linewidth=0.8, width=0.6,
                )
                ax.set_title(f"g={gfn}, y={oc}")
                ax.set_xlabel("")
                ax.set_ylabel("MAB(β)" if j == 0 else "")
                if i == 0 and j == 2:
                    ax.legend(title="", loc="upper right", fontsize=8)
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                ax.tick_params(axis="x", labelsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT_DIR / f"boxplot_MAB_beta_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")

    # Same for MAB(γ)
    for n_val in n_vals:
        sub = df[df["n"] == n_val]
        fig, axes = plt.subplots(3, 3, figsize=(14, 12), sharey="row")
        fig.suptitle(f"MAB(γ)  —  n = {n_val}", fontsize=14, fontweight="bold")

        for i, gfn in enumerate(g_fns):
            for j, oc in enumerate(outcomes):
                ax = axes[i, j]
                mask = (sub["g_fn"] == gfn) & (sub["outcome"] == oc)
                d = sub[mask]
                if d.empty:
                    ax.set_visible(False)
                    continue
                sns.boxplot(
                    data=d, x="model", y="MAB_gamma", hue="ws_label",
                    ax=ax, palette="Set2", showfliers=False,
                    linewidth=0.8, width=0.6,
                )
                ax.set_title(f"g={gfn}, y={oc}")
                ax.set_xlabel("")
                ax.set_ylabel("MAB(γ)" if j == 0 else "")
                if i == 0 and j == 2:
                    ax.legend(title="", loc="upper right", fontsize=8)
                else:
                    ax.get_legend().remove() if ax.get_legend() else None
                ax.tick_params(axis="x", labelsize=9)

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT_DIR / f"boxplot_MAB_gamma_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_bias_boxplots(df_all)

# ---------------------------------------------------------------------------
# 5. Warm-start Δ heatmap (NeuralPLSI ws1 vs ws0)
# ---------------------------------------------------------------------------
def plot_warmstart_delta(df):
    """Heatmap of Δ MAB(β) = ws1 − ws0 for NeuralPLSI (negative = WS better)."""
    nplsi = df[df["model"] == "NeuralPLSI"]
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    n_vals = sorted(nplsi["n"].unique())

    for n_val in n_vals:
        sub = nplsi[nplsi["n"] == n_val]
        mat_beta = np.full((3, 3), np.nan)
        mat_gamma = np.full((3, 3), np.nan)
        for i, gfn in enumerate(g_fns):
            for j, oc in enumerate(outcomes):
                ws0 = sub[(sub["g_fn"] == gfn) & (sub["outcome"] == oc) & (sub["warmstart"] == 0)]
                ws1 = sub[(sub["g_fn"] == gfn) & (sub["outcome"] == oc) & (sub["warmstart"] == 1)]
                if len(ws0) > 0 and len(ws1) > 0:
                    mat_beta[i, j] = ws1["MAB_beta"].mean() - ws0["MAB_beta"].mean()
                    mat_gamma[i, j] = ws1["MAB_gamma"].mean() - ws0["MAB_gamma"].mean()

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(f"Warm-Start Effect (NeuralPLSI, n={n_val})\n"
                     "Negative = WS reduces bias", fontsize=13, fontweight="bold")
        data_beta = pd.DataFrame(mat_beta, index=g_fns, columns=outcomes)
        data_gamma = pd.DataFrame(mat_gamma, index=g_fns, columns=outcomes)

        vmax = max(abs(np.nanmax(mat_beta)), abs(np.nanmin(mat_beta)), 0.01)
        sns.heatmap(data_beta, annot=True, fmt=".4f", cmap="RdBu_r",
                    center=0, vmin=-vmax, vmax=vmax, ax=ax1, linewidths=0.5)
        ax1.set_title("Δ MAB(β)")
        ax1.set_ylabel("g-function")

        vmax2 = max(abs(np.nanmax(mat_gamma)), abs(np.nanmin(mat_gamma)), 0.01)
        sns.heatmap(data_gamma, annot=True, fmt=".4f", cmap="RdBu_r",
                    center=0, vmin=-vmax2, vmax=vmax2, ax=ax2, linewidths=0.5)
        ax2.set_title("Δ MAB(γ)")
        ax2.set_ylabel("")

        plt.tight_layout(rect=[0, 0, 1, 0.92])
        out = OUT_DIR / f"heatmap_warmstart_delta_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_warmstart_delta(df_all)

# ---------------------------------------------------------------------------
# 6. g-Function Recovery Panels
# ---------------------------------------------------------------------------
def true_g(x, g_fn):
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)


def plot_g_panels(data_dir=DATA_DIR):
    """Plot g-function recovery: 3 g_fn × 3 outcome, overlay NeuralPLSI + PLSI."""
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    n_vals = [200, 1000]
    grid = np.linspace(-3, 3, 1000)

    for n_val in n_vals:
        for ws in [0, 1]:
            fig, axes = plt.subplots(3, 3, figsize=(15, 13))
            ws_label = "WS" if ws else "No-WS"
            fig.suptitle(f"g-Function Recovery  —  n={n_val}, {ws_label}",
                         fontsize=14, fontweight="bold")

            for i, gfn in enumerate(g_fns):
                g_truth = true_g(grid, gfn)
                for j, oc in enumerate(outcomes):
                    ax = axes[i, j]
                    ax.plot(grid, g_truth, "k-", lw=2, label="True g", zorder=10)

                    for model, color, alpha in [
                        ("NeuralPLSI", "#1f77b4", 0.04),
                        ("PLSI", "#ff7f0e", 0.04),
                    ]:
                        fname = data_dir / f"simulation+{model}+{n_val}+{gfn}+{oc}+normal+ws{ws}.json"
                        if not fname.exists():
                            continue
                        try:
                            with open(fname) as fh:
                                reps = json.load(fh)
                        except (json.JSONDecodeError, ValueError):
                            continue
                        gs = np.array([r["g_pred"] for r in reps])
                        for k in range(min(50, len(gs))):
                            ax.plot(grid, gs[k], color=color, alpha=alpha, lw=0.4)
                        g_mean = gs.mean(axis=0)
                        ax.plot(grid, g_mean, color=color, lw=1.5,
                                label=f"{model} (mean)")

                    ax.set_title(f"g={gfn}, y={oc}", fontsize=10)
                    ax.set_xlim(-3, 3)
                    if i == 0 and j == 0:
                        ax.legend(fontsize=7, loc="best")

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out = OUT_DIR / f"g_recovery_n{n_val}_ws{ws}.png"
            fig.savefig(out)
            plt.close(fig)
            print(f"Saved {out}")


plot_g_panels()

# ---------------------------------------------------------------------------
# 7. Coverage bar chart
# ---------------------------------------------------------------------------
def plot_coverage(df):
    """Grouped bar: coverage by model × g_fn × outcome."""
    nplsi = df[df["model"] == "NeuralPLSI"]
    n_vals = sorted(nplsi["n"].unique())
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]

    for n_val in n_vals:
        sub = nplsi[nplsi["n"] == n_val]
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle(f"Bootstrap Coverage  —  NeuralPLSI, n={n_val}",
                     fontsize=13, fontweight="bold")

        for idx, (metric, label) in enumerate([("beta_cov", "β"), ("gamma_cov", "γ")]):
            ax = axes[idx]
            bars = sub.groupby(["g_fn", "outcome", "ws_label"])[metric].mean().reset_index()
            bars["setting"] = bars["g_fn"] + "\n" + bars["outcome"]
            sns.barplot(data=bars, x="setting", y=metric, hue="ws_label",
                        ax=ax, palette="Set2")
            ax.axhline(0.95, color="red", ls="--", lw=1, label="Nominal 95%")
            ax.set_title(f"Coverage({label})")
            ax.set_ylabel("Coverage")
            ax.set_xlabel("")
            ax.set_ylim(0, 1.05)
            ax.tick_params(axis="x", labelsize=8, rotation=45)
            if idx == 0:
                ax.legend(fontsize=8)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = OUT_DIR / f"coverage_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_coverage(df_all)

# ---------------------------------------------------------------------------
# 8. NeuralPLSI vs PLSI comparison bar chart
# ---------------------------------------------------------------------------
def plot_model_comparison(df):
    """Compare NeuralPLSI vs PLSI MAB(β) across settings."""
    n_vals = sorted(df["n"].unique())
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]

    for n_val in n_vals:
        sub = df[(df["n"] == n_val) & (df["warmstart"] == 1)]  # use ws=1 for comparison
        fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
        fig.suptitle(f"NeuralPLSI vs PLSI  —  n={n_val}, WS=1",
                     fontsize=13, fontweight="bold")

        for j, oc in enumerate(outcomes):
            ax = axes[j]
            d = sub[sub["outcome"] == oc]
            if d.empty:
                continue
            sns.barplot(data=d, x="g_fn", y="MAB_beta", hue="model",
                        ax=ax, palette="Set1", ci="sd", order=g_fns)
            ax.set_title(f"Outcome: {oc}")
            ax.set_xlabel("g-function")
            ax.set_ylabel("MAB(β)" if j == 0 else "")
            if j == 2:
                ax.legend(fontsize=9)
            else:
                ax.get_legend().remove() if ax.get_legend() else None

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = OUT_DIR / f"model_comparison_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_model_comparison(df_all)

# ---------------------------------------------------------------------------
# 9. Per-Parameter Signed Bias Plots
#    Three methods: PLSI, NeuralPLSI, NeuralPLSI (WS)
# ---------------------------------------------------------------------------
BIAS_YLIM = (-0.3, 0.3)
METHOD_ORDER = ["PLSI", "NeuralPLSI", "NeuralPLSI (WS)"]
METHOD_PALETTE = {"PLSI": "#8da0cb", "NeuralPLSI": "#66c2a5", "NeuralPLSI (WS)": "#fc8d62"}


def load_per_param_bias():
    """Load per-parameter signed bias for each replicate."""
    rows = []
    files = sorted(DATA_DIR.glob("simulation+*+ws*.json"))
    for f in files:
        model, n, g_fn, outcome, x_dist, ws = parse_filename(f.name)
        try:
            with open(f) as fh:
                reps = json.load(fh)
        except (json.JSONDecodeError, ValueError):
            continue
        # Assign method label:
        #   PLSI ws0 → "PLSI"  (skip PLSI ws1 — identical results)
        #   NeuralPLSI ws0  → "NeuralPLSI"
        #   NeuralPLSI ws1  → "NeuralPLSI (WS)"
        if model == "PLSI" and ws == 1:
            continue  # WS doesn't change PLSI, skip duplicates
        if model == "PLSI":
            method = "PLSI"
        elif model == "NeuralPLSI" and ws == 0:
            method = "NeuralPLSI"
        else:
            method = "NeuralPLSI (WS)"

        for rep in reps:
            beta_est = np.array(rep["beta_estimate"])
            gamma_est = np.array(rep["gamma_estimate"])
            beta_bias = beta_est - TRUE_BETA
            gamma_bias = gamma_est - TRUE_GAMMA
            base = {
                "method": method, "model": model, "n": n, "g_fn": g_fn,
                "outcome": outcome, "warmstart": ws, "seed": rep["seed"],
            }
            for k, b in enumerate(beta_bias):
                row = base.copy()
                row["param"] = f"β{k}"
                row["bias"] = b
                row["param_type"] = "β"
                rows.append(row)
            for k, g in enumerate(gamma_bias):
                row = base.copy()
                row["param"] = f"γ{k}"
                row["bias"] = g
                row["param_type"] = "γ"
                rows.append(row)
    return pd.DataFrame(rows)


print("\nLoading per-parameter bias data...")
df_bias = load_per_param_bias()
print(f"Loaded {len(df_bias)} per-parameter bias rows")


def plot_per_param_bias(df):
    """Per-parameter signed bias violin plots: PLSI vs NeuralPLSI vs NeuralPLSI (WS)."""
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    n_vals = sorted(df["n"].unique())

    for n_val in n_vals:
        for param_type, param_label in [("β", "beta"), ("γ", "gamma")]:
            sub = df[(df["n"] == n_val) & (df["param_type"] == param_type)]
            if sub.empty:
                continue

            fig, axes = plt.subplots(3, 3, figsize=(18, 14))
            fig.suptitle(f"Per-Parameter {param_type} Bias  —  n={n_val}",
                         fontsize=14, fontweight="bold")
            for i, gfn in enumerate(g_fns):
                for j, oc in enumerate(outcomes):
                    ax = axes[i, j]
                    d = sub[(sub["g_fn"] == gfn) & (sub["outcome"] == oc)]
                    if d.empty:
                        ax.set_visible(False)
                        continue
                    sns.violinplot(
                        data=d, x="param", y="bias", hue="method",
                        hue_order=METHOD_ORDER,
                        ax=ax, palette=METHOD_PALETTE, inner="quartile",
                        linewidth=0.8, cut=0, density_norm="width",
                    )
                    ax.axhline(0, color="red", ls="--", lw=0.8, alpha=0.7)
                    ax.set_ylim(BIAS_YLIM)
                    ax.set_title(f"g={gfn}, y={oc}", fontsize=10)
                    ax.set_xlabel("")
                    ax.set_ylabel("Signed Bias" if j == 0 else "")
                    if i == 0 and j == 2:
                        ax.legend(fontsize=7, loc="upper right")
                    else:
                        leg = ax.get_legend()
                        if leg:
                            leg.remove()
                    ax.tick_params(axis="x", labelsize=8)

            plt.tight_layout(rect=[0, 0, 1, 0.96])
            out = OUT_DIR / f"bias_{param_label}_violin_n{n_val}.png"
            fig.savefig(out)
            plt.close(fig)
            print(f"Saved {out}")


plot_per_param_bias(df_bias)

# ---------------------------------------------------------------------------
# 10. Per-Parameter Bias Forest Plots (Mean Bias with 95% CI)
#     Parameters on y-axis, bias on x-axis, "X" markers + capped CI lines
# ---------------------------------------------------------------------------
def plot_bias_forest(df):
    """Forest plot of mean signed bias per parameter: 3 methods, vertical layout."""
    n_vals = sorted(df["n"].unique())
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    params_order = [f"β{k}" for k in range(len(TRUE_BETA))] + \
                   [f"γ{k}" for k in range(len(TRUE_GAMMA))]

    for n_val in n_vals:
        sub = df[df["n"] == n_val]
        if sub.empty:
            continue

        fig, axes = plt.subplots(3, 3, figsize=(18, 16))
        fig.suptitle(f"Per-Parameter Bias (Mean ± 95% CI)  —  n={n_val}",
                     fontsize=14, fontweight="bold")

        for i, gfn in enumerate(g_fns):
            for j, oc in enumerate(outcomes):
                ax = axes[i, j]
                d = sub[(sub["g_fn"] == gfn) & (sub["outcome"] == oc)]
                if d.empty:
                    ax.set_visible(False)
                    continue

                agg = d.groupby(["param", "method"])["bias"].agg(
                    ["mean", "std", "count"]
                ).reset_index()
                agg["se"] = agg["std"] / np.sqrt(agg["count"])
                agg["ci95"] = 1.96 * agg["se"]
                agg["param"] = pd.Categorical(agg["param"],
                                              categories=params_order, ordered=True)
                agg = agg.sort_values("param")

                n_methods = len(METHOD_ORDER)
                y_base = np.arange(len(params_order))
                offset_step = 0.25
                
                for m_idx, method in enumerate(METHOD_ORDER):
                    m_data = agg[agg["method"] == method]
                    if m_data.empty:
                        continue
                    # Map params to y-positions
                    y_pos = np.array([params_order.index(p) for p in m_data["param"]])
                    offset = (m_idx - (n_methods - 1) / 2) * offset_step
                    
                    ax.errorbar(
                        m_data["mean"].values, y_pos + offset,
                        xerr=m_data["ci95"].values,
                        fmt="x", markersize=7, markeredgewidth=2,
                        color=METHOD_PALETTE[method],
                        ecolor=METHOD_PALETTE[method],
                        elinewidth=1.5, capsize=4, capthick=1.5,
                        label=method, alpha=0.9,
                    )

                ax.axvline(0, color="red", ls="--", lw=0.8, alpha=0.7)
                ax.set_yticks(y_base)
                ax.set_yticklabels(params_order, fontsize=9)
                ax.invert_yaxis()
                ax.set_xlim(BIAS_YLIM)
                ax.set_title(f"g={gfn}, y={oc}", fontsize=10)
                ax.set_xlabel("Mean Bias" if i == 2 else "")
                if i == 0 and j == 2:
                    ax.legend(fontsize=7, loc="lower right")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        out = OUT_DIR / f"bias_forest_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_bias_forest(df_bias)

# ---------------------------------------------------------------------------
# 11. Bias Heatmap: Mean Absolute Bias per parameter × setting
#     Three methods side-by-side
# ---------------------------------------------------------------------------
def plot_bias_heatmap(df):
    """Heatmap of MAB per parameter for all three methods."""
    g_fns = ["linear", "sfun", "sigmoid"]
    outcomes = ["continuous", "binary", "cox"]
    n_vals = sorted(df["n"].unique())
    params = [f"β{k}" for k in range(len(TRUE_BETA))] + \
             [f"γ{k}" for k in range(len(TRUE_GAMMA))]

    for n_val in n_vals:
        fig, axes = plt.subplots(1, 3, figsize=(20, 7))
        fig.suptitle(f"Per-Parameter MAB — n={n_val}", fontsize=14, fontweight="bold")

        for m_idx, method in enumerate(METHOD_ORDER):
            ax = axes[m_idx]
            sub = df[(df["n"] == n_val) & (df["method"] == method)]
            if sub.empty:
                ax.set_visible(False)
                continue

            settings = []
            mat = []
            for gfn in g_fns:
                for oc in outcomes:
                    d = sub[(sub["g_fn"] == gfn) & (sub["outcome"] == oc)]
                    if d.empty:
                        continue
                    settings.append(f"{gfn}\n{oc}")
                    row = []
                    for p in params:
                        pd_p = d[d["param"] == p]
                        row.append(pd_p["bias"].abs().mean() if len(pd_p) > 0 else np.nan)
                    mat.append(row)

            if not mat:
                ax.set_visible(False)
                continue

            mat_arr = np.array(mat)
            sns.heatmap(
                pd.DataFrame(mat_arr, index=settings, columns=params),
                annot=True, fmt=".3f", cmap="YlOrRd", ax=ax,
                linewidths=0.5, vmin=0, vmax=0.3,
                cbar_kws={"label": "MAB"} if m_idx == 2 else {"label": ""},
            )
            ax.set_title(method, fontsize=12, fontweight="bold")
            ax.set_ylabel("Setting" if m_idx == 0 else "")
            ax.set_xlabel("Parameter")

        plt.tight_layout(rect=[0, 0, 1, 0.94])
        out = OUT_DIR / f"bias_heatmap_n{n_val}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"Saved {out}")


plot_bias_heatmap(df_bias)

# ---------------------------------------------------------------------------
print(f"\n✓ All visualizations saved to {OUT_DIR}")
print(f"✓ All tables saved to {TABLE_DIR}")
