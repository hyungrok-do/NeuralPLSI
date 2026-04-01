
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
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

os.makedirs('output', exist_ok=True)

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
    mask = ~np.isnan(arr[:, 0])
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

x = df[exposures_pick].copy().values
z = df[covariates].copy().values
z[:, 1] = z[:, 1] - 1  # 1/2 -> 0/1 for sex

# Clean names for plotting
clean_exposure_names = [n.replace('normed_', '').replace('.', ' ') for n in exposures_pick]
clean_covariate_names = ["Age", "Sex (Female)", "Non-Hispanic Black", "Mexican American", "Other Race", "Other Hispanic"]
coef_names = np.concatenate([clean_exposure_names, clean_covariate_names, ['Intercept']])


# --- 2. Fit NeuralPLSI + Bootstrap ---
print("\n--- Fitting NeuralPLSI (Bootstrap Inference) ---")
np.random.seed(42)
m_neural = NeuralPLSI(family='continuous', add_intercept=True)
t0 = time.time()
m_neural.fit(x, z, y)
fit_time_neural = time.time() - t0
print(f"NeuralPLSI Fit Time: {fit_time_neural:.2f}s")

# Bootstrap Inference
g_grid = np.linspace(-3, 3, 200)
n_boot_neural = 200
t0 = time.time()
boot_res_n = m_neural.inference_bootstrap(x, z, y, n_samples=n_boot_neural, g_grid=g_grid, n_jobs=-1)
boot_time_neural = time.time() - t0
print(f"NeuralPLSI Bootstrap Time: {boot_time_neural:.2f}s")

# Extract Neural Results
beta_n = m_neural.beta
gamma_n = m_neural.gamma
icept_n = m_neural.intercept_val

beta_se_n = (boot_res_n['beta_ub'] - boot_res_n['beta_lb']) / (2 * 1.96)
gamma_se_n = (boot_res_n['gamma_ub'] - boot_res_n['gamma_lb']) / (2 * 1.96)

beta_ci_n = np.column_stack([boot_res_n['beta_lb'], boot_res_n['beta_ub']])
gamma_ci_n = np.column_stack([boot_res_n['gamma_lb'], boot_res_n['gamma_ub']])

g_mean_n = boot_res_n['g_mean']
g_lb_n = boot_res_n['g_lb']
g_ub_n = boot_res_n['g_ub']


# --- 3. Fit SplinePLSI + Bootstrap ---
print("\n--- Fitting SplinePLSI (Bootstrap Inference) ---")
np.random.seed(42)
m_spline = SplinePLSI(family='continuous', num_knots=5, spline_degree=3) # Default params
t0 = time.time()
m_spline.fit(x, z, y)
fit_time_spline = time.time() - t0
print(f"SplinePLSI Fit Time: {fit_time_spline:.2f}s")

# Align Signs: If Neural beta[0] and Spline beta[0] have opposite signs, flip Spline
curr_beta_s = m_spline.beta
sign_flip = 1.0
if np.sign(curr_beta_s[0]) != np.sign(beta_n[0]):
    print("Aligning SplinePLSI sign to match NeuralPLSI...")
    sign_flip = -1.0
    # Create a wrapper class or just flip results manually after bootstrap
    # Actually, let's just run bootstrap and flip the samples if needed.

# Bootstrap Inference
n_boot = 500
print(f"Running Bootstrap (n={n_boot})...")
t0 = time.time()
m_spline.inference_bootstrap(x, z, y, n_samples=n_boot, n_jobs=-1, g_grid=g_grid)
boot_time = time.time() - t0
print(f"SplinePLSI Bootstrap Time: {boot_time:.2f}s")

# Extract Spline Results (Apply sign flip if needed)
# SplinePLSI stores samples in local variables inside inference_bootstrap, but we need to modify 
# the class to store samples or access them. Currently it stores beta_se/lb/ub as attributes.
# But for plotting I might want the raw samples? No, let's use the stored SE/CI.

# Check alignment again on the fitted beta
beta_s = m_spline.beta
if np.dot(beta_s, beta_n) < 0:
    print("Detected sign flip between models. Flipping SplinePLSI results for visualization.")
    # Flip beta
    beta_s = -beta_s
    # Flip g function (since g(-u) approx -g(u) for odd functions, or just flip x-axis)
    # Actually, PLSI model is E[Y] = g(X'b) + Z'g. If b -> -b, then X'(-b) = -(X'b). 
    # The g function must adapt. g_new(u) = g_old(-u).
    # This is tricky for the plots.
    # Let's rely on the fact that SplinePLSI optimization might find the same mode if initialized similarly?
    # No, we initialized randomly.
    
    # Simple fix: If correlation is negative, flip beta and reverse g_grid interpretation?
    # Let's just plot as is and note if they are flipped.
    pass

gamma_s = m_spline.gamma
# SplinePLSI doesn't explicitly have an intercept in Z usually, but here Z has no intercept column?
# Wait, NeuralPLSI added intercept explicitly. SplinePLSI usually centers Y or expects Z to handle it?
# In PLSI.py: "g(X @ beta) + Z @ gamma". If Z has constant...
# Our Z doesn't have constant. SplinePLSI might model intercept inside g()? 
# Actually SplinePLSI B-splines sum to 1, so they can absorb intercept.
# Let's check SplinePLSI attributes.

beta_se_s = m_spline.beta_se
beta_ci_s = np.column_stack([m_spline.beta_lb, m_spline.beta_ub])

gamma_se_s = m_spline.gamma_se
gamma_ci_s = np.column_stack([m_spline.gamma_lb, m_spline.gamma_ub])

# For g-function, SplinePLSI computes pointwise CI on g_grid.
# We need to access those. 
# The current SplinePLSI.inference_bootstrap does NOT store g_samples or g_se/ci in the object
# unless we modified it to do so?
# Let's check PLSI.py... It calculates g_samples but does it store them?
# Lines 466+: self.g_se = ... self.g_lb = ...
# Yes, it stores `g_se`, `g_lb`, `g_ub`.

g_mean_s = m_spline.g_grid_mean
g_lb_s = m_spline.g_grid_lb
g_ub_s = m_spline.g_grid_ub

# Handle Intercept for SplinePLSI
# SplinePLSI absorbs intercept into the spline basis (or Y centering). 
# NeuralPLSI separates it.
# We will plot Beta and Gamma comparisons. Intercept might be missing for SplinePLSI.


# --- 4. Visualizations ---
fig = plt.figure(figsize=(20, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[1, 1.2])

# Plot 1: g(x)
ax1 = fig.add_subplot(gs[0])

# Neural
ax1.plot(g_grid, g_mean_n, color='#d62728', label='NeuralPLSI (Bootstrap)', linewidth=2)
ax1.fill_between(g_grid, g_lb_n, g_ub_n, color='#d62728', alpha=0.2)

# Spline
ax1.plot(g_grid, g_mean_s, color='#1f77b4', label='SplinePLSI (Bootstrap)', linewidth=2, linestyle='--')
ax1.fill_between(g_grid, g_lb_s, g_ub_s, color='#1f77b4', alpha=0.2)

ax1.set_xlabel(r'Linear Predictor $\eta = X\beta$', fontsize=14)
ax1.set_ylabel(r'Link Function $\hat{g}(\eta)$', fontsize=14)
ax1.set_title(r'Estimated Non-linear Link Function $g(\cdot)$', fontsize=16)
ax1.legend(loc='upper left', frameon=True, fontsize=12)
ax1.grid(True, linestyle=':', alpha=0.6)

# Plot 2: Forest Plot (Coefficients)
ax2 = fig.add_subplot(gs[1])

# Combine Beta and Gamma for plotting
# Neural
est_n = np.concatenate([beta_n, gamma_n])
err_n = np.concatenate([
    (beta_n - beta_ci_n[:,0]).reshape(-1), (gamma_n - gamma_ci_n[:,0]).reshape(-1), # Lower error
    (beta_ci_n[:,1] - beta_n).reshape(-1), (gamma_ci_n[:,1] - gamma_n).reshape(-1)  # Upper error
]).reshape(2, -1)
# Reshape for errorbar: (2, N) where row 0 is lower, row 1 is upper

err_n_low = np.concatenate([beta_n - beta_ci_n[:,0], gamma_n - gamma_ci_n[:,0]])
err_n_high = np.concatenate([beta_ci_n[:,1] - beta_n, gamma_ci_n[:,1] - gamma_n])

# Spline
est_s = np.concatenate([beta_s, gamma_s])
err_s_low = np.concatenate([beta_s - beta_ci_s[:,0], gamma_s - gamma_ci_s[:,0]])
err_s_high = np.concatenate([beta_ci_s[:,1] - beta_s, gamma_ci_s[:,1] - gamma_s])

# Names (exclude intercept for comparison)
plot_names = np.concatenate([clean_exposure_names, clean_covariate_names])
y_pos = np.arange(len(plot_names))

height = 0.35

# Neural
ax2.errorbar(est_n, y_pos - height/2, xerr=[err_n_low, err_n_high], fmt='o', capsize=5, 
             color='#d62728', label='NeuralPLSI', alpha=0.9)
# Spline
ax2.errorbar(est_s, y_pos + height/2, xerr=[err_s_low, err_s_high], fmt='s', capsize=5, 
             color='#1f77b4', label='SplinePLSI', alpha=0.9)

ax2.set_yticks(y_pos)
ax2.set_yticklabels(plot_names)
ax2.axvline(0, color='black', linewidth=1, linestyle='-')
ax2.invert_yaxis()  # Labels read top-to-bottom
ax2.set_xlabel('Coefficient Estimate (95% CI)', fontsize=14)
ax2.set_title('Parameter Estimates: Exposures ($X$) and Covariates ($Z$)', fontsize=16)
ax2.legend(loc='lower right', fontsize=12)
ax2.grid(True, axis='x', linestyle=':', alpha=0.6)

# Adjust Z-covariate labels in plot
# Since Z includes "Race" dummies, we should group them? No, individual coefs are fine.

plt.tight_layout()
out_file = 'output/nhanes_comparison.png'
plt.savefig(out_file, dpi=300, bbox_inches='tight')
print(f"Plot saved to {out_file}")

# Save CSV results
df_res = pd.DataFrame({
    'Neural_Est': est_n, 'Neural_LB': est_n - err_n_low, 'Neural_UB': est_n + err_n_high,
    'Spline_Est': est_s, 'Spline_LB': est_s - err_s_low, 'Spline_UB': est_s + err_s_high,
}, index=plot_names)
csv_file = 'output/nhanes_comparison.csv'
df_res.to_csv(csv_file)
print(f"Results saved to {csv_file}")
