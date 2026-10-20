
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
import matplotlib.pyplot as plt
import time

# Add root directory to path to allow imports
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.append(root_dir)

from models import NeuralPLSI, SplinePLSI

# Set style for high-quality plots
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Helvetica', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12
plt.rcParams['legend.fontsize'] = 12
plt.rcParams['figure.titlesize'] = 16

def main():
    os.makedirs('output/nhanes', exist_ok=True)

    # --- 1. Load and Preprocess Data ---
    print("Loading NHANES data...")
    csv_path = r"NHANES/12940_2020_644_MOESM2_ESM.csv"
    nhanes = pd.read_csv(csv_path)

    sex_map = {1: "Male", 2: "Female"}
    race_map = {
        1: "Non-Hispanic White",
        2: "Non-Hispanic Black",
        3: "Mexican American",
        4: "Other Race - Including Multi-Racial",
        5: "Other Hispanic",
    }

    nhanes["SEX"] = nhanes["sex"].map(sex_map).astype("category")
    nhanes["RACE"] = nhanes["race"].map(race_map).astype("category")

    exposure_cols = nhanes.columns[2:24].tolist()
    no_log_name = "a5.Retinol"

    df = nhanes.copy()

    def zscore(s: pd.Series) -> pd.Series:
        scaler = StandardScaler()
        arr = s.to_numpy().reshape(-1, 1)
        mask = np.isfinite(arr[:, 0])
        out = np.full_like(arr, np.nan, dtype=float)
        if mask.sum() > 0:
            out[mask, 0] = scaler.fit_transform(arr[mask]).flatten()
        return pd.Series(out[:, 0], index=s.index)

    for col in exposure_cols:
        new_col = f"normed_{col}"
        if col == no_log_name:
            df[new_col] = zscore(df[col].astype(float))
        else:
            vals = pd.to_numeric(df[col], errors="coerce")
            vals = vals.where(vals > 0, np.nan)
            df[new_col] = zscore(np.log(vals))

    normed_exposure_cols = [f"normed_{c}" for c in exposure_cols]

    y_raw = pd.to_numeric(df["triglyceride"], errors="coerce")
    y_log = y_raw.where(y_raw > 0, np.nan)
    df["normed_triglyceride"] = zscore(np.log(y_log))

    # One-hot encode race (drop first category 'Non-Hispanic White' implicitly by starting k from 2)
    for k in range(2, 6):
        df[f"race{k-1}"] = (df["race"] == k).astype(int)

    covariates = ["age", "sex", "race1", "race2", "race3", "race4"]
    y_name = "normed_triglyceride"
    y = df[y_name].values

    # Pick specific exposures as in original script
    exposures_pick = [
        "normed_a7.a.Tocopherol",
        "normed_a6.g.tocopherol",
        "normed_a4.Retinyl.stearate",
        "normed_a5.Retinol",
        "normed_a20.3.3.4.4.5.pncb",
        "normed_a17.PCB194",
        "normed_a22.2.3.4.6.7.8.hxcdf",
        "normed_a1.trans.b.carotene"
    ]

    valid = (np.isfinite(df[exposures_pick+covariates+[y_name]].to_numpy(dtype=float)).all(axis=1)
             & df['race'].isin(range(1, 6)).to_numpy() & df['sex'].isin([1, 2]).to_numpy())
    print(f"Using {valid.sum()} complete observations; excluded {(~valid).sum()} rows")
    df = df.loc[valid].copy()
    y = df[y_name].to_numpy(dtype=float)
    x = df[exposures_pick].copy().values
    z = df[covariates].copy().values
    z[:, 1] = z[:, 1] - 1  # 1/2 -> 0/1 for sex

    # Clean names for plotting
    clean_exposure_names = [n.replace('normed_', '').replace('.', ' ') for n in exposures_pick]
    clean_covariate_names = ["Age", "Sex (Female)", "Non-Hispanic Black", "Mexican American", "Other Race", "Other Hispanic"]
    coef_names = np.concatenate([clean_exposure_names, clean_covariate_names, ['Intercept']])


    # --- 2. Fit NeuralPLSI + Bootstrap ---
    print("\n--- Fitting NeuralPLSI (Bootstrap Inference) ---")
    m_neural = NeuralPLSI(family='continuous', add_intercept=True)
    t0 = time.time()
    m_neural.fit(x, z, y)
    fit_time_neural = time.time() - t0
    print(f"NeuralPLSI Fit Time: {fit_time_neural:.2f}s")

    # Bootstrap Inference
    g_grid = np.linspace(-2, 4, 200)
    n_boot_neural = 200
    t0 = time.time()
    boot_res_n = m_neural.inference_bootstrap(x, z, y, n_samples=n_boot_neural, g_grid=g_grid, n_jobs='auto')
    boot_time_neural = time.time() - t0
    print(f"NeuralPLSI Bootstrap Time: {boot_time_neural:.2f}s")

    # Extract Neural Results
    beta_n = m_neural.beta
    gamma_n = m_neural.gamma
    icept_n = m_neural.intercept_val

    beta_se_n = boot_res_n['beta_se']
    gamma_se_n = boot_res_n['gamma_se']

    beta_ci_n = np.column_stack([boot_res_n['beta_lb'], boot_res_n['beta_ub']])
    gamma_ci_n = np.column_stack([boot_res_n['gamma_lb'], boot_res_n['gamma_ub']])

    # Bootstrap the sum jointly to include intercept uncertainty and covariance.
    curve_samples = boot_res_n['g_samples'] + boot_res_n['intercept_samples']
    g_mean_n = curve_samples.mean(axis=0)
    g_lb_n, g_ub_n = np.percentile(curve_samples, [2.5, 97.5], axis=0)


    # --- 3. Fit SplinePLSI + Bootstrap ---
    print("\n--- Fitting SplinePLSI (Bootstrap Inference) ---")
    m_spline = SplinePLSI(family='continuous', num_knots=5, spline_degree=3) # Default params
    t0 = time.time()
    m_spline.fit(x, z, y, random_state=42)
    fit_time_spline = time.time() - t0
    print(f"SplinePLSI Fit Time: {fit_time_spline:.2f}s")

    # Bootstrap Inference
    n_boot = 500
    print(f"Running Bootstrap (n={n_boot})...")
    t0 = time.time()
    m_spline.inference_bootstrap(x, z, y, n_samples=n_boot, n_jobs=-1, g_grid=g_grid)
    boot_time = time.time() - t0
    print(f"SplinePLSI Bootstrap Time: {boot_time:.2f}s")

    # Extract spline estimates in their canonical orientation.
    beta_s = m_spline.beta
    beta_ci_s = np.column_stack([m_spline.beta_lb, m_spline.beta_ub])

    gamma_s = m_spline.gamma
    beta_se_s = m_spline.beta_se

    gamma_se_s = m_spline.gamma_se
    gamma_ci_s = np.column_stack([m_spline.gamma_lb, m_spline.gamma_ub])

    g_mean_s = m_spline.g_grid_mean
    g_lb_s = m_spline.g_grid_lb
    g_ub_s = m_spline.g_grid_ub

    # Both models use the same canonical orientation, beta[0] >= 0.

    # --- 4. Visualizations ---
    # Plot 1: g(x)

    # Calculate NeuralPLSI partial residuals
    eta_n = x @ beta_n
    res_n = y - z @ gamma_n

    # Calculate PLSI partial residuals
    eta_s = x @ beta_s
    res_s = y - z @ gamma_s

    # NeuralPLSI Figure
    fig1a = plt.figure(figsize=(6, 6))
    ax1a = fig1a.add_subplot(111)
    ax1a.scatter(eta_n, res_n, facecolors='none', edgecolors='gray', alpha=0.5, label='Partial Residuals')
    ax1a.plot(g_grid, g_mean_n, color='#d62728', label='NeuralPLSI', linewidth=2)
    ax1a.fill_between(g_grid, g_lb_n, g_ub_n, color='#d62728', alpha=0.2)
    ax1a.set_xlim(-2, 4)
    ax1a.set_xlabel('Index', fontsize=14)
    ax1a.set_ylabel('g(Index)', fontsize=14)
    ax1a.set_title('NeuralPLSI', fontsize=16)
    ax1a.legend(loc='upper left', frameon=True, fontsize=12)
    ax1a.grid(True, linestyle=':', alpha=0.6)
    ax1a.set_box_aspect(1)
    fig1a.tight_layout()
    fig1a.savefig('output/nhanes/nhanes_g_function_neural.png', dpi=300, bbox_inches='tight')

    # PLSI Figure
    fig1b = plt.figure(figsize=(6, 6))
    ax1b = fig1b.add_subplot(111)
    ax1b.scatter(eta_s, res_s, facecolors='none', edgecolors='gray', alpha=0.5, label='Partial Residuals')
    ax1b.plot(g_grid, g_mean_s, color='#1f77b4', label='PLSI', linewidth=2, linestyle='--')
    ax1b.fill_between(g_grid, g_lb_s, g_ub_s, color='#1f77b4', alpha=0.2)
    ax1b.set_xlim(-2, 4)
    ax1b.set_xlabel('Index', fontsize=14)
    ax1b.set_ylabel('g(Index)', fontsize=14)
    ax1b.set_title('PLSI', fontsize=16)
    ax1b.legend(loc='upper left', frameon=True, fontsize=12)
    ax1b.grid(True, linestyle=':', alpha=0.6)
    ax1b.set_box_aspect(1)
    fig1b.tight_layout()
    fig1b.savefig('output/nhanes/nhanes_g_function_plsi.png', dpi=300, bbox_inches='tight')

    # Plot 2: Forest Plot (Coefficients)
    fig2 = plt.figure(figsize=(10, 14))
    ax2 = fig2.add_subplot(111)

    # Draw the actual interval even when it does not contain the point estimate.
    est_n = np.concatenate([beta_n, gamma_n])
    est_s = np.concatenate([beta_s, gamma_s])
    ci_n = np.vstack([beta_ci_n, gamma_ci_n])
    ci_s = np.vstack([beta_ci_s, gamma_ci_s])
    plot_names = np.concatenate([clean_exposure_names, clean_covariate_names])
    sort_idx = np.argsort(np.abs(est_n))[::-1]
    est_n, est_s = est_n[sort_idx], est_s[sort_idx]
    ci_n, ci_s = ci_n[sort_idx], ci_s[sort_idx]
    plot_names = plot_names[sort_idx]
    y_pos = np.arange(len(plot_names))
    height = .35
    for estimate, interval, offset, color, label, marker in [
            (est_n, ci_n, -height/2, '#d62728', 'NeuralPLSI', 'o'),
            (est_s, ci_s, height/2, '#1f77b4', 'SplinePLSI', 's')]:
        ax2.hlines(y_pos+offset, interval[:, 0], interval[:, 1], color=color)
        ax2.scatter(estimate, y_pos+offset, color=color, label=label, marker=marker)

    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(plot_names)
    ax2.axvline(0, color='black', linewidth=1, linestyle='-')
    ax2.invert_yaxis()  # Labels read top-to-bottom
    ax2.set_xlabel('Coefficient Estimate (95% CI)', fontsize=14)
    ax2.set_title('Parameter Estimates: Exposures ($X$) and Covariates ($Z$)', fontsize=16)
    ax2.legend(loc='lower right', fontsize=12)
    ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

    fig2.tight_layout()
    out_file2 = 'output/nhanes/nhanes_forest_plot.png'
    fig2.savefig(out_file2, dpi=300, bbox_inches='tight')
    print(f"Forest plot saved to {out_file2}")
    print("g-function plots saved to output/nhanes/nhanes_g_function_neural.png and output/nhanes/nhanes_g_function_plsi.png")

    # Save CSV results
    df_res = pd.DataFrame({
        'Neural_Est': est_n, 'Neural_LB': ci_n[:, 0], 'Neural_UB': ci_n[:, 1],
        'Neural_SE': np.concatenate([beta_se_n, gamma_se_n])[sort_idx],
        'PLSI_SE': np.concatenate([beta_se_s, gamma_se_s])[sort_idx],
        'PLSI_Est': est_s, 'PLSI_LB': ci_s[:, 0], 'PLSI_UB': ci_s[:, 1],
    }, index=plot_names)
    csv_file = 'output/nhanes/nhanes_comparison.csv'
    df_res.to_csv(csv_file)
    print(f"Results saved to {csv_file}")


if __name__ == "__main__":
    main()
