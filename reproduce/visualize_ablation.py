#!/usr/bin/env python3
"""
Ablation Study Visualization: 5-Way NeuralPLSI Initialization Comparison

Produces single-panel figures with all 9 scenarios (3 outcomes × 3 g-functions):
1. MAB(β) — all scenarios in one panel
2. MAB(γ) — all scenarios in one panel
3. L2(g) — all scenarios in one panel (NeuralPLSI variants only)
4. MSE + Timing boxplots
5. Average summary bars
6. Per-element bias (one panel per scenario)
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import defaultdict
from pathlib import Path



# ── Config ──────────────────────────────────────────────────────────────────
import sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

data_dir = Path(__file__).parent / "output"
out_dir  = Path(__file__).parent / "figures"
out_dir.mkdir(exist_ok=True)

# Imports for local simulation (for CIs and g-curves)
import joblib
from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA
from models import NeuralPLSI, SplinePLSI

G_GRID = np.linspace(-3, 3, 500)

def _worker_sim(outcome, g_type, seed, model_name, ws, init, n=500, return_g=False):
    """Run a single simulation rep and return parameters (and optionally g-curve)."""
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        try:
            data = simulate_data(n, outcome=outcome, g_type=g_type, seed=seed)
            if data is None:
                 return None
            
            X, Z, y, _, _, _ = data
            
            # Check for NaNs/Infs in data
            if not np.isfinite(X).all() or not np.isfinite(Z).all():
                 return None
                 
            if model_name == 'PLSI':
                model = SplinePLSI(family=outcome)
                model.fit(X, Z, y)
                b_est = model.beta
                g_est = model.gamma
                g_curve = model.g_function(G_GRID) if return_g else None
            else:
                # NeuralPLSI
                model = NeuralPLSI(family=outcome, max_epoch=200, warmstart=ws, initial=init)
                model.fit(X, Z, y, random_state=seed)
                b_est = model.beta
                g_est = model.gamma
                g_curve = model.g_function(G_GRID) if return_g else None
            
            # Discard non-finite results
            if not np.isfinite(b_est).all() or not np.isfinite(g_est).all():
                return None
            if g_curve is not None and not np.isfinite(g_curve).all():
                g_curve = None  # keep b/g estimates but drop bad g-curve
            
            return b_est, g_est, g_curve
        except Exception as e:
            return None


# Load data
files = sorted(data_dir.glob("ablation_init_*.json"))
if not files:
    raise FileNotFoundError("No ablation_init_*.json files found in output/")

all_rows = []
for f in files:
    with open(f) as fh:
        all_rows.extend(json.load(fh))

print(f"Loaded {len(all_rows)} rows from {len(files)} files")

# ── Method labels & colors ──────────────────────────────────────────────────
def method_label(r):
    if r['model'] == 'PLSI':
        return 'PLSI'
    ws   = r.get('warmstart', False)
    init = r.get('initial', False)
    if not ws and not init: return 'NPLSI\n(random)'
    elif ws and not init:   return 'NPLSI\n(ws)'
    elif not ws and init:   return 'NPLSI\n(init)'
    else:                   return 'NPLSI\n(ws+init)'

METHODS = ['PLSI', 'NPLSI\n(random)', 'NPLSI\n(ws)', 'NPLSI\n(init)', 'NPLSI\n(ws+init)']
COLORS  = ['#7f8c8d', '#e74c3c', '#3498db', '#2ecc71', '#9b59b6']
METHOD_COLORS = dict(zip(METHODS, COLORS))

OUTCOMES = ['continuous', 'binary', 'cox']
G_TYPES  = ['linear', 'sigmoid', 'sfun']

# Build the 9 scenarios in order
SCENARIOS = [(o, g) for o in OUTCOMES for g in G_TYPES]
SCENARIO_LABELS = [f"{o.capitalize()}\n{g.capitalize()}" for o, g in SCENARIOS]

# Build lookup table
table = defaultdict(dict)
for r in all_rows:
    key = (r['outcome'], r['g_type'])
    method = method_label(r)
    table[key][method] = r

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
})


# ============================================================================
# 1 & 2. SINGLE-PANEL BAR PLOTS: MAB(β) and MAB(γ) — all 9 scenarios
# ============================================================================

def plot_bias_panel():
    """One figure per metric, all 9 scenarios side-by-side."""
    for metric, label, fname in [
        ('mab_beta', r'MAB($\beta$)', 'ablation_mab_beta.png'),
        ('mab_gamma', r'MAB($\gamma$)', 'ablation_mab_gamma.png'),
    ]:
        n_groups = len(SCENARIOS)
        n_methods = len(METHODS)
        bar_width = 0.8 / n_methods
        x = np.arange(n_groups)

        fig, ax = plt.subplots(figsize=(18, 5))
        fig.suptitle(f'{label} — All 9 Scenarios',
                     fontsize=14, fontweight='bold', y=0.98)

        for i, method in enumerate(METHODS):
            vals = []
            for key in SCENARIOS:
                if key in table and method in table[key]:
                    vals.append(table[key][method][metric])
                else:
                    vals.append(0)
            offset = (i - n_methods / 2 + 0.5) * bar_width
            bars = ax.bar(x + offset, vals, bar_width * 0.9,
                         color=METHOD_COLORS[method], label=method,
                         edgecolor='white', linewidth=0.5)
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                            f'{v:.3f}', ha='center', va='bottom', fontsize=6,
                            rotation=90)

        # Best markers
        for j, key in enumerate(SCENARIOS):
            vals_dict = {m: table[key][m][metric] for m in METHODS
                        if key in table and m in table[key]}
            if vals_dict:
                best = min(vals_dict.values())
                for i, method in enumerate(METHODS):
                    if method in vals_dict and abs(vals_dict[method] - best) < 1e-6:
                        offset = (i - n_methods / 2 + 0.5) * bar_width
                        ax.scatter(j + offset, best - 0.003, marker='*',
                                  color='gold', s=60, zorder=5, edgecolors='black',
                                  linewidth=0.5)

        # Outcome group separators
        for sep in [2.5, 5.5]:
            ax.axvline(sep, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(SCENARIO_LABELS, fontsize=9)
        ax.set_ylabel(label, fontsize=12)
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        handles, labels_ = ax.get_legend_handles_labels()
        ax.legend(handles, labels_, loc='best', ncol=5, fontsize=9,
                  frameon=True, fancybox=True, shadow=True)

        plt.tight_layout()
        path = out_dir / fname
        fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        print(f"  Saved: {path}")


# ============================================================================
# 3. SINGLE-PANEL L2(g) — all 9 scenarios, NeuralPLSI variants only
# ============================================================================

def plot_l2g_panel():
    """All 9 scenarios in one panel, NeuralPLSI variants only."""
    nplsi_methods = [m for m in METHODS if m != 'PLSI']
    nplsi_colors  = [METHOD_COLORS[m] for m in nplsi_methods]

    n_groups = len(SCENARIOS)
    n_methods = len(nplsi_methods)
    bar_width = 0.8 / n_methods
    x = np.arange(n_groups)

    fig, ax = plt.subplots(figsize=(18, 5))
    fig.suptitle(r'L2 Error of $g(\cdot)$ — NeuralPLSI Variants (All 9 Scenarios)',
                 fontsize=14, fontweight='bold', y=0.98)

    for i, method in enumerate(nplsi_methods):
        vals = []
        for key in SCENARIOS:
            if key in table and method in table[key] and 'l2_g' in table[key][method]:
                vals.append(table[key][method]['l2_g'])
            else:
                vals.append(0)
        offset = (i - n_methods / 2 + 0.5) * bar_width
        bars = ax.bar(x + offset, vals, bar_width * 0.9,
                     color=METHOD_COLORS[method], label=method,
                     edgecolor='white', linewidth=0.5)
        for bar, v in zip(bars, vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{v:.2f}', ha='center', va='bottom', fontsize=6.5,
                        rotation=90)

    # Best markers
    for j, key in enumerate(SCENARIOS):
        vals_dict = {m: table[key][m]['l2_g'] for m in nplsi_methods
                    if key in table and m in table[key] and 'l2_g' in table[key][m]}
        if vals_dict:
            best = min(vals_dict.values())
            for i, method in enumerate(nplsi_methods):
                if method in vals_dict and abs(vals_dict[method] - best) < 1e-6:
                    offset = (i - n_methods / 2 + 0.5) * bar_width
                    ax.scatter(j + offset, best - 0.02, marker='*',
                              color='gold', s=60, zorder=5, edgecolors='black',
                              linewidth=0.5)

    # Outcome group separators
    for sep in [2.5, 5.5]:
        ax.axvline(sep, color='grey', linewidth=0.8, linestyle=':', alpha=0.5)

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS, fontsize=9)
    ax.set_ylabel(r'$L_2(g)$', fontsize=12)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    handles, labels_ = ax.get_legend_handles_labels()
    ax.legend(handles, labels_, loc='best', ncol=4, fontsize=9,
              frameon=True, fancybox=True, shadow=True)

    plt.tight_layout()
    path = out_dir / "ablation_l2g.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 4. BOXPLOTS: MSE (beta²+gamma²) and Timing across scenarios
# ============================================================================

def plot_mse_and_timing_boxplots():
    """Side-by-side boxplots of per-element squared biases and timing."""

    mse_data = {m: [] for m in METHODS}
    time_data = {m: [] for m in METHODS}

    for key in table:
        for method in METHODS:
            if method not in table[key]:
                continue
            r = table[key][method]
            beta_biases = np.array(r.get('beta_bias', []))
            gamma_biases = np.array(r.get('gamma_bias', []))
            all_biases = np.concatenate([beta_biases, gamma_biases])
            sq_biases = all_biases ** 2
            mse_data[method].extend(sq_biases.tolist())
            time_data[method].append(r.get('time_s', 0))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('MSE and Runtime — 5-Way Comparison',
                 fontsize=14, fontweight='bold', y=1.02)

    # MSE boxplot
    ax = axes[0]
    bp_data = [mse_data[m] for m in METHODS]
    bp = ax.boxplot(bp_data, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_ylabel('Squared Bias (per element)', fontsize=11)
    ax.set_title('MSE Distribution', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, m in enumerate(METHODS):
        median = np.median(bp_data[i])
        ax.text(i + 1, median, f'{median:.4f}', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='darkblue')

    # Timing boxplot
    ax = axes[1]
    bp_data_t = [time_data[m] for m in METHODS]
    bp = ax.boxplot(bp_data_t, patch_artist=True, widths=0.6,
                    medianprops=dict(color='black', linewidth=2),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2),
                    flierprops=dict(marker='o', markersize=3, alpha=0.5))
    for patch, color in zip(bp['boxes'], COLORS):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
    ax.set_xticklabels(METHODS, fontsize=9)
    ax.set_ylabel('Time (seconds)', fontsize=11)
    ax.set_title('Runtime', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    for i, m in enumerate(METHODS):
        median = np.median(bp_data_t[i])
        ax.text(i + 1, median, f'{median:.1f}s', ha='center', va='bottom',
                fontsize=8, fontweight='bold', color='darkblue')

    plt.tight_layout()
    path = out_dir / "ablation_mse_timing.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 5. COMBINED SUMMARY: Avg MAB bar chart
# ============================================================================

def plot_avg_summary():
    """Single bar chart of average MAB(β), MAB(γ), L2(g) across all 9 scenarios."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    fig.suptitle('Average Performance — 5-Way Comparison (9 Scenarios)',
                 fontsize=14, fontweight='bold', y=1.02)

    for ax_idx, (metric, label, include_plsi) in enumerate([
        ('mab_beta', r'Mean MAB($\beta$)', True),
        ('mab_gamma', r'Mean MAB($\gamma$)', True),
        ('l2_g', r'Mean $L_2(g)$', False),
    ]):
        ax = axes[ax_idx]
        methods_to_plot = METHODS if include_plsi else [m for m in METHODS if m != 'PLSI']
        colors_to_plot = [METHOD_COLORS[m] for m in methods_to_plot]

        avgs = []
        for m in methods_to_plot:
            vals = []
            for key in table:
                if m in table[key] and metric in table[key][m]:
                    vals.append(table[key][m][metric])
            avgs.append(np.mean(vals) if vals else 0)

        bars = ax.bar(range(len(methods_to_plot)), avgs,
                     color=colors_to_plot, edgecolor='white', linewidth=0.5)

        best_idx = np.argmin(avgs)
        bars[best_idx].set_edgecolor('gold')
        bars[best_idx].set_linewidth(3)

        for bar, v in zip(bars, avgs):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                    f'{v:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        ax.set_xticks(range(len(methods_to_plot)))
        ax.set_xticklabels(methods_to_plot, fontsize=9)
        ax.set_ylabel(label, fontsize=11)
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_title(label, fontsize=11, fontweight='bold')

    plt.tight_layout()
    path = out_dir / "ablation_avg_summary.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")




# ============================================================================
# 6. PER-ELEMENT BIAS FOREST PLOT: all 9 scenarios in a 3×3 grid
#    Parameters on y-axis, bias on x-axis, "X" markers at point estimate
# ============================================================================

def plot_per_element_bias_grid(n_reps=20, n=2000):
    """
    3×3 grid: rows = g-type, cols = outcome.
    Plots bias for each element of Beta (8) and Gamma (3) with 95% CI.
    Re-runs simulations locally to get variance.
    """
    rows_g = ['linear', 'sigmoid', 'sfun']
    cols_outcome = ['continuous', 'binary', 'cox']
    
    # Define methods to run: (name, ws, init, label, color, marker)
    # Markers: 'o' = no warmstart, 'D' = warmstart
    # Colors:  random-start → warm reds, GLM-init → cool blues
    methods_config = [
        ('PLSI',       False, False, 'PLSI',            '#7f8c8d', 'o'),
        ('NeuralPLSI', False, False, 'NPLSI\n(random)', '#e74c3c', 'o'),
        ('NeuralPLSI', True,  False, 'NPLSI\n(ws)',     '#c0392b', 'D'),
        ('NeuralPLSI', False, True,  'NPLSI\n(init)',   '#2980b9', 'o'),
        ('NeuralPLSI', True,  True,  'NPLSI\n(ws+init)','#1a5276', 'D'),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle(f"Per-Element Bias (Mean ± 95% CI) — n={n}, {n_reps} reps", 
                 fontsize=14, fontweight='bold', y=0.98)

    # Pre-calculate jobs
    tasks = []
    print(f"  Queueing {len(rows_g)*len(cols_outcome)*len(methods_config)*n_reps} simulations for Bias Grid...")
    
    # We will run them and organize results
    # Use joblib to run all
    # To parse results easily, we'll iterate and run standard loops, 
    #   or just run per-panel to keep memory low and logic simple.
    
    for row_idx, g_type in enumerate(rows_g):
        for col_idx, outcome in enumerate(cols_outcome):
            ax = axes[row_idx, col_idx]
            
            # Param indices
            p_beta = len(TRUE_BETA)
            p_gamma = len(TRUE_GAMMA)
            
            # Y-positions
            y_centers = np.arange(p_beta + p_gamma)
            bar_height = 0.15
            
            # Plot reference line
            ax.axvline(0, color='black', lw=0.8, alpha=0.5)
            
            # Run methods for this panel
            for m_idx, (m_name, ws, init, label, color, mkr) in enumerate(methods_config):
                
                # Run parallel reps
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(_worker_sim)(
                        outcome, g_type, rep*100, m_name, ws, init, n, return_g=False
                    ) for rep in range(n_reps)
                )
                
                # Filter None
                results = [r for r in results if r is not None]
                if not results: continue
                
                betas = np.array([r[0] for r in results], dtype=float)
                gammas = np.array([r[1] for r in results], dtype=float)
                
                # Compute bias stats (handled for NaNs)
                # Count valid samples per parameter
                n_valid_b = np.sum(np.isfinite(betas), axis=0)
                n_valid_g = np.sum(np.isfinite(gammas), axis=0)
                
                # Avoid division by zero
                n_valid_b[n_valid_b == 0] = 1
                n_valid_g[n_valid_g == 0] = 1
                
                mean_b = np.nanmean(betas, axis=0) - TRUE_BETA
                se_b   = np.nanstd(betas, axis=0) / np.sqrt(n_valid_b)
                err_b  = 1.96 * se_b
                
                mean_g = np.nanmean(gammas, axis=0) - TRUE_GAMMA
                se_g   = np.nanstd(gammas, axis=0) / np.sqrt(n_valid_g)
                err_g  = 1.96 * se_g
                
                # Debug print for first panel to verify
                # if row_idx==0 and col_idx==0:
                #      print(f"    {m_name}: valid_b={n_valid_b.min()}-{n_valid_b.max()}, bias_b_range=[{np.nanmin(mean_b):.3f}, {np.nanmax(mean_b):.3f}]")

                # Concatenate
                means = np.concatenate([mean_b, mean_g])
                errs  = np.concatenate([err_b, err_g])
                
                # Plot
                offset = (m_idx - len(methods_config)/2) * bar_height
                ys = y_centers + offset
                
                # Error bars first
                ax.errorbar(means, ys, xerr=errs, fmt='none', 
                            ecolor=color, elinewidth=1.0, capsize=2, alpha=0.6)
                # Points
                ax.scatter(means, ys, s=20, color=color, marker=mkr, 
                           edgecolors='white', linewidths=0.3,
                           label=label if (row_idx==0 and col_idx==0) else "",
                           alpha=0.9, zorder=3)

            # Titles/Labels
            if row_idx == 0:
                ax.set_title(f"{outcome.capitalize()}\n({g_type})", fontsize=11)
            else:
                ax.set_title(f"{g_type}", fontsize=10)
                
            # Y-ticks
            if col_idx == 0:
                ytick_labels = [f"β{k+1}" for k in range(p_beta)] + [f"γ{k+1}" for k in range(p_gamma)]
                ax.set_yticks(y_centers)
                ax.set_yticklabels(ytick_labels, fontsize=8)
            else:
                ax.set_yticks([])
                
            # X-limit (relaxed, auto-scale by default, or wider)
            # ax.set_xlim(-0.5, 0.5) # Let matplotlib auto-scale to show outliers
            # Add grid
            ax.grid(axis='x', linestyle=':', alpha=0.4)

    # Legend
    handles, labels = axes[0, 0].get_legend_handles_labels()
    # Deduplicate
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc='lower center', 
               ncol=5, bbox_to_anchor=(0.5, 0.02), fontsize=9)

    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    path = out_dir / "ablation_bias_elements_grid.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# 7. g-Function Recovery Panels (PLSI vs NPLSI ws+init)
# ============================================================================

def _get_true_g(x, g_fn):
    """Evaluate the true g-function on a grid."""
    _, _, _, _, _, g_true = simulate_data(5, outcome="continuous", g_type=g_fn, seed=0)
    return np.vectorize(g_true)(x)

def plot_g_recovery_panel(n_reps=20, n=2000):
    """
    3×3 grid: rows = g-function (linear, sigmoid, sfun),
              cols = outcome   (continuous, binary, cox).
    Each panel: true g (black solid) + PLSI (orange) + NPLSI ws+init (blue).
    g-curves are centered (mean-subtracted) to handle identifiability offsets.
    """
    g_fns = ['linear', 'sigmoid', 'sfun']
    outcomes_list = ['continuous', 'binary', 'cox']

    # Only PLSI and NPLSI (ws+init)
    methods_g = [
        # (model_name, ws, init, label, color, linestyle, fill_alpha)
        ('PLSI',       False, False, 'PLSI (spline)',    '#e67e22', '--', 0.15),
        ('NeuralPLSI', True,  True,  'NPLSI (ws+init)',  '#2980b9', '-',  0.15),
    ]

    fig, axes = plt.subplots(3, 3, figsize=(14, 12))
    fig.suptitle(f"g-Function Recovery — PLSI vs NeuralPLSI (n={n}, {n_reps} reps)",
                 fontsize=14, fontweight='bold')

    print(f"  Queueing simulations for g-Function Recovery (PLSI vs ws+init)...")

    for i, gfn in enumerate(g_fns):
        g_truth_raw = _get_true_g(G_GRID, gfn)
        # Center the true g too (for consistent comparison)
        g_truth = g_truth_raw - np.mean(g_truth_raw)
        for j, oc in enumerate(outcomes_list):
            ax = axes[i, j]
            # True g (centered)
            ax.plot(G_GRID, g_truth, 'k-', lw=2.5, label='True g', zorder=10)

            for m_name, ws, init, label, color, ls, fa in methods_g:
                results = joblib.Parallel(n_jobs=-1)(
                    joblib.delayed(_worker_sim)(
                        oc, gfn, rep*100, m_name, ws, init, n, return_g=True
                    ) for rep in range(n_reps)
                )

                g_curves = [r[2] for r in results if r is not None and r[2] is not None]
                if not g_curves:
                    continue

                # Center each g-curve (subtract its mean) for identifiability
                gs = np.array(g_curves)
                for k in range(gs.shape[0]):
                    gs[k] -= np.nanmean(gs[k])

                g_mean = np.nanmean(gs, axis=0)
                g_sd   = np.nanstd(gs, axis=0)

                ax.fill_between(G_GRID, g_mean - 2*g_sd, g_mean + 2*g_sd,
                                color=color, alpha=fa)
                ax.plot(G_GRID, g_mean, color=color, lw=2.0, ls=ls, label=label)

            ax.set_title(f"g = {gfn},  y = {oc}", fontsize=10)
            ax.set_xlim(-3, 3)
            ax.set_ylim(-4.5, 4.5)
            ax.grid(axis='both', linestyle=':', alpha=0.3)
            if i == 0 and j == 0:
                ax.legend(fontsize=8, loc='best')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    path = out_dir / "ablation_g_recovery.png"
    fig.savefig(path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f"  Saved: {path}")


# ============================================================================
# Run all
# ============================================================================
if __name__ == "__main__":
    print("\n1. MAB(β) and MAB(γ) bar plots (single panel each)...")
    plot_bias_panel()

    print("\n2. L2(g) bar plot (single panel)...")
    plot_l2g_panel()

    print("\n3. MSE + Timing boxplots...")
    plot_mse_and_timing_boxplots()

    print("\n4. Average summary bar chart...")
    plot_avg_summary()

    print("\n5. Per-element bias grid (3×3)...")
    plot_per_element_bias_grid()

    print("\n6. g-Function recovery panel (3×3 — re-fits models)...")
    plot_g_recovery_panel()

    print("\n✓ All ablation visualizations complete!")
