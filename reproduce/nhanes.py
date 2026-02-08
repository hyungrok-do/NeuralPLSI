import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os
import matplotlib.pyplot as plt
import time

root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from models import NeuralPLSI

os.makedirs('output', exist_ok=True)

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

exposures = normed_exposure_cols

for k in range(2, 6):
    df[f"race{k-1}"] = (df["race"] == k).astype(int)

covariates = ["age", "sex", "race1", "race2", "race3", "race4"]

y_name = "normed_triglyceride"
y = df[y_name].values

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
z[:, 1] = z[:, 1] - 1

model = NeuralPLSI(family='continuous', add_intercept=True)
model.fit(x, z, y)

gxb = model.predict_gxb(x)
g_grid = np.linspace(gxb.min(), gxb.max(), 1000)
g_fn_pred = model.g_function(g_grid)

# Hessian Inference
print("Running Hessian Inference...")
t_start_hess = time.time()
inf_res = model.inference_hessian(x, z, y)
g_bands = model.inference_hessian_g(x, z, y, mode="g_of_t", g_grid=g_grid, include_beta=True)
t_hess = time.time() - t_start_hess
print(f"Hessian Inference Time: {t_hess:.2f}s")

beta_est = inf_res['beta_hat']
gamma_est = inf_res['gamma_hat']
beta_se = inf_res['beta_se']
gamma_se = inf_res['gamma_se']
beta_lb, beta_ub = inf_res['beta_lb'], inf_res['beta_ub']
gamma_lb, gamma_ub = inf_res['gamma_lb'], inf_res['gamma_ub']

intercept_est = inf_res['intercept_hat']
intercept_se = inf_res['intercept_se']
intercept_lb = inf_res['intercept_lb']
intercept_ub = inf_res['intercept_ub']

g_mean_hess = g_bands['g_mean']
g_lb_hess = g_bands['g_lb']
g_ub_hess = g_bands['g_ub']

# Bootstrap Inference
n_bootstrap = 100
print(f"Running Bootstrap Inference (n={n_bootstrap})...")
t_start_boot = time.time()
boot_res = model.inference_bootstrap(x, z, y, n_samples=n_bootstrap, g_grid=g_grid, n_jobs=1)
t_boot = time.time() - t_start_boot
print(f"Bootstrap Inference Time: {t_boot:.2f}s")

beta_est_boot = boot_res['beta_hat']
beta_lb_boot, beta_ub_boot = boot_res['beta_lb'], boot_res['beta_ub']
beta_err_boot = np.column_stack([beta_est_boot - beta_lb_boot, beta_ub_boot - beta_est_boot]).T

gamma_est_boot = boot_res['gamma_hat']
gamma_lb_boot, gamma_ub_boot = boot_res['gamma_lb'], boot_res['gamma_ub']
gamma_err_boot = np.column_stack([gamma_est_boot - gamma_lb_boot, gamma_ub_boot - gamma_est_boot]).T

intercept_est_boot = boot_res.get('intercept_hat', np.array([0.0]))
intercept_lb_boot = boot_res.get('intercept_lb', np.array([0.0]))
intercept_ub_boot = boot_res.get('intercept_ub', np.array([0.0]))
intercept_err_boot = np.column_stack([intercept_est_boot - intercept_lb_boot, intercept_ub_boot - intercept_est_boot]).T

g_mean_boot = boot_res['g_mean']
g_lb_boot = boot_res['g_lb']
g_ub_boot = boot_res['g_ub']

# Combine Coefficients for Plotting
covariate_names = ["age", "sex", "Non-Hispanic Black", "Mexican American", "Other Race", "Other Hispanic"]
coef_names = np.concatenate([exposures_pick, covariate_names, ['Intercept']])
hessian_est = np.concatenate([beta_est, gamma_est, intercept_est])
hessian_err = np.column_stack([
    np.concatenate([beta_est - beta_lb, gamma_est - gamma_lb, intercept_est - intercept_lb]),
    np.concatenate([beta_ub - beta_est, gamma_ub - gamma_est, intercept_ub - intercept_est])
]).T

boot_est = np.concatenate([beta_est_boot, gamma_est_boot, intercept_est_boot])
boot_err = np.column_stack([
    np.concatenate([beta_est_boot - beta_lb_boot, gamma_est_boot - gamma_lb_boot, intercept_est_boot - intercept_lb_boot]),
    np.concatenate([beta_ub_boot - beta_est_boot, gamma_ub_boot - gamma_est_boot, intercept_ub_boot - intercept_est_boot])
]).T

# Save Results
res = {
    'coefs': hessian_est,
    'se': np.concatenate([beta_se, gamma_se, intercept_se]),
    'ub': np.concatenate([beta_ub, gamma_ub, intercept_ub]),
    'lb': np.concatenate([beta_lb, gamma_lb, intercept_lb])
}

pd.DataFrame(res, index=np.concatenate([exposures_pick, covariate_names, ['Intercept']])).to_csv('output/nhanes_nplsi_results.csv')

plt.figure(figsize=(18, 8))

# Plot 1: g(x)
plt.subplot(1, 2, 1)
plt.plot(g_grid, g_fn_pred, color='black', label='Main Fit', linewidth=1.5)
plt.fill_between(g_grid, g_lb_hess, g_ub_hess, color='blue', alpha=0.3, label='Hessian 95% CI')
plt.fill_between(g_grid, g_lb_boot, g_ub_boot, color='orange', alpha=0.3, label='Bootstrap 95% CI')
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
plt.xlabel('Linear Predictor Index')
plt.ylabel('g(index)')
plt.title(f'Non-linear function g(x) Estimate\nHessian ({t_hess:.1f}s) vs Bootstrap ({t_boot:.1f}s)')
plt.legend()

# Plot 2: Coefficients
plt.subplot(1, 2, 2)
x_pos = np.arange(len(coef_names))
width = 0.35

plt.bar(x_pos - width/2, hessian_est, width, label='Hessian', yerr=hessian_err, capsize=5, color='skyblue', alpha=0.8)
plt.bar(x_pos + width/2, boot_est, width, label='Bootstrap', yerr=boot_err, capsize=5, color='salmon', alpha=0.8)

plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.xticks(x_pos, coef_names, rotation=90)
plt.ylabel('Coefficient Value')
plt.title('Coefficient Estimates (Beta, Gamma, Intercept)')
plt.legend()

plt.tight_layout()
plt.savefig('output/nhanes.png', dpi=300)
plt.close()
