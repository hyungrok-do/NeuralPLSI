import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import sys
import os

try:
    from models import NeuralPLSI
except ImportError:
    # Add root directory to path to allow imports
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    sys.path.append(root_dir)
    from models import NeuralPLSI
import matplotlib.pyplot as plt
from tqdm import tqdm

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

weights = None

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
gxb_fn = model.g_function(gxb)
deconfounded = y - model.gamma @ z.T

g_grid = np.linspace(gxb.min(), gxb.max(), 1000)
g_fn_pred = model.g_function(g_grid)

beta_est = model.beta.copy()
gamma_est = model.gamma.copy()
intercept_est = model.intercept_val.copy()

beta_stack = []
gamma_stack = []
intercept_stack = []
g_stack = []

n_bootstrap = 500
for i in tqdm(range(n_bootstrap)):
    np.random.seed(i)
    sample_indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
    z_sample = z[sample_indices]
    x_sample = x[sample_indices]
    y_sample = y[sample_indices]
    model = NeuralPLSI(family='continuous', add_intercept=True)
    model.fit(x_sample, z_sample, y_sample)

    beta_stack.append(model.beta)
    gamma_stack.append(model.gamma)
    intercept_stack.append(model.intercept_val)
    g_stack.append(model.g_function(g_grid))

beta_stack = np.array(beta_stack)
gamma_stack = np.array(gamma_stack)
intercept_stack = np.array(intercept_stack)
g_stack = np.array(g_stack)
g_mean = np.mean(g_stack, axis=0)
g_ub = np.percentile(g_stack, 97.5, axis=0)
g_lb = np.percentile(g_stack, 2.5, axis=0)

beta_ub = np.percentile(beta_stack, 97.5, axis=0)
beta_lb = np.percentile(beta_stack, 2.5, axis=0)
gamma_ub = np.percentile(gamma_stack, 97.5, axis=0)
gamma_lb = np.percentile(gamma_stack, 2.5, axis=0)
intercept_ub = np.percentile(intercept_stack, 97.5, axis=0)
intercept_lb = np.percentile(intercept_stack, 2.5, axis=0)

beta_se = np.std(beta_stack, axis=0, ddof=1)
gamma_se = np.std(gamma_stack, axis=0, ddof=1)
intercept_se = np.std(intercept_stack, axis=0, ddof=1)

res = {
    'coefs': np.concatenate([beta_est, gamma_est, intercept_est]),
    'se': np.concatenate([beta_se, gamma_se, intercept_se]),
    'ub': np.concatenate([beta_ub, gamma_ub, intercept_ub]),
    'lb': np.concatenate([beta_lb, gamma_lb, intercept_lb])
}

covariate_names = ["age", "sex", "Non-Hispanic Black", "Mexican American", "Other Race", "Other Hispanic"]

pd.DataFrame(res, index=np.concatenate([exposures_pick, covariate_names, ['Intercept']])).to_csv('output/nhanes_nplsi_results.csv')

plt.figure(figsize=(10, 6))
plt.plot(g_grid, g_fn_pred, color='dimgrey')
plt.fill_between(g_grid, g_lb, g_ub, color='lightblue', alpha=0.4, label='95% CI')
plt.axhline(0, color='grey', linestyle='--', linewidth=0.8)
plt.axvline(0, color='grey', linestyle='--', linewidth=0.8)
plt.xlabel('index')
plt.ylabel('g(index)')
plt.legend()
plt.tight_layout()
plt.savefig('output/nhanes.png', dpi=300)
plt.close()
