# --- Part 0: Imports ---
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# --- Part 0.1: Data import ---
# NOTE: Update the path if needed.
csv_path = r"NHANES/12940_2020_644_MOESM2_ESM.csv"
nhanes = pd.read_csv(csv_path)

# --- Part 0.2: Data descriptions (factor-like columns) ---
# Create labeled categorical versions (to mirror R's factor)
sex_map = {1: "Male", 2: "Female"}
race_map = {
    1: "Non-Hispanic White",
    2: "Non-Hispanic Black",
    3: "Mexican American",
    4: "Other Race - Including Multi-Racial",
    5: "Other Hispanic",
}

# These keep the original numeric codes in `sex` and `race` but add labeled columns
nhanes["SEX"] = nhanes["sex"].map(sex_map).astype("category")
nhanes["RACE"] = nhanes["race"].map(race_map).astype("category")

# --- Part 0.3: Exposure preprocessing (log-transform + scale) ---

# Columns 3:24 in R are 1-indexed; in pandas they're 0-indexed.
# R used `colnames(nhanes)[3:24]`, i.e., columns with indices 2..23 in pandas.
exposure_cols = nhanes.columns[2:24].tolist()

# In the R script, "a5.Retinol" (called e5.Retinol in the comment) is NOT log-transformed.
# Everything else among the 22 exposures is log-transformed then standardized.
no_log_name = "a5.Retinol"

# Safety: working copy so we don’t clobber original columns
df = nhanes.copy()

# Helper: z-score a series (ignoring NaNs)
def zscore(s: pd.Series) -> pd.Series:
    scaler = StandardScaler()
    arr = s.to_numpy().reshape(-1, 1)
    # Fit on non-NaN
    mask = ~np.isnan(arr[:, 0])
    out = np.full_like(arr, np.nan, dtype=float)
    if mask.sum() > 0:
        out[mask, 0] = scaler.fit_transform(arr[mask]).flatten()
    return pd.Series(out[:, 0], index=s.index)

# 1) Log-transform + scale for all exposure columns except a5.Retinol
for col in exposure_cols:
    new_col = f"normed_{col}"
    if col == no_log_name:
        # Standardize as-is
        df[new_col] = zscore(df[col].astype(float))
    else:
        # Log-transform first, but avoid non-positive values
        vals = pd.to_numeric(df[col], errors="coerce")
        # Set non-positive to NaN to mirror R's log behavior (would be -Inf)
        vals = vals.where(vals > 0, np.nan)
        df[new_col] = zscore(np.log(vals))

# Rename the exposure columns to their normed_* counterparts (like the R script)
# R literally replaced colnames(nhanes)[3:24] with prefixed names.
# We'll keep originals and ALSO track the normalized versions explicitly.
normed_exposure_cols = [f"normed_{c}" for c in exposure_cols]

# --- Outcome transform (triglyceride) ---
# Create `normed_triglyceride = scale(log(triglyceride))`
# (with the same non-positive guard)
y_raw = pd.to_numeric(df["triglyceride"], errors="coerce")
y_log = y_raw.where(y_raw > 0, np.nan)
df["normed_triglyceride"] = zscore(np.log(y_log))

# For later correlation/labels, R stripped prefixes when plotting; we just keep names.

# --- Part 0.4: Basic engineered variables used later ---

# Selected exposures are determined by modeling (stepwise) in the R script;
# since you asked for preprocessing only, we DO NOT perform selection here.
# We expose the full normalized exposure list for downstream modeling.
exposures = normed_exposure_cols  # 22 normalized exposure columns

# Create race dummy variables (race1..race4) matching R:
# race1 = 1{race==2}, ..., race4 = 1{race==5}
for k in range(2, 6):
    df[f"race{k-1}"] = (df["race"] == k).astype(int)

# Covariates used later (age, sex, race dummies). In R, sex is used as-is (numeric in the model).
covariates = ["age", "sex", "race1", "race2", "race3", "race4"]

# Outcomes
y_name = "normed_triglyceride"
y = df[y_name]

# Binary outcome for high triglycerides: trigl_bin = 1{triglyceride >= 150}
df["trigl_bin"] = (df["triglyceride"] >= 150).astype(int)
y_bin = df["trigl_bin"]

# Placeholder for weights (R sets weights <- NULL initially)
weights = None

# --- Export the prepared DataFrame and key lists ---
# df: contains original + normalized exposures + engineered columns
# exposures: list of normalized exposure column names
# covariates: list of covariates used later
# y_name, y, y_bin, weights are ready for downstream modeling

# If you want to keep only the columns needed downstream, uncomment:
# keep_cols = (["triglyceride", y_name, "trigl_bin", "age", "sex", "race", "SEX", "RACE"] 
#              + exposures + covariates + ["race1", "race2", "race3", "race4"])
# df = df.loc[:, sorted(set(keep_cols), key=keep_cols.index)]

# Example peek:
print(f"N = {len(df)}")
print("First few normalized exposure columns:", exposures[:5])
print("Covariates:", covariates)

from models.nPLSI import neuralPLSI

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

print(df.columns)
x = df[exposures_pick].copy().values

# Covariates: demographic/confounders
z = df[covariates].copy().values
z[:, 0] = (z[:, 0] - z[:, 0].mean()) / z[:, 0].std()
z[:, 1] = z[:, 1] - 1

y = y_bin.copy().values
print(np.unique(y, return_counts=True))

print(f"Prepared X shape: {x.shape}, Z shape: {z.shape}, y shape: {y.shape}")

#model.fit(covariates.values, select_z.values, TELO)
import matplotlib.pyplot as plt
from tqdm import tqdm

model = neuralPLSI(family='binary')
model.fit(x, z, y)

gxb = model.predict_gxb(x)
gxb_fn = model.g_function(gxb)
deconfounded = y - model.gamma @ z.T

g_grid = np.linspace(gxb.min(), gxb.max(), 1000)
g_fn_pred = model.g_function(g_grid)

beta_est = model.beta.copy()
gamma_est = model.gamma.copy()

# bootstrap

beta_stack = []
gamma_stack = []
g_stack = []

n_bootstrap = 1000
for i in tqdm(range(n_bootstrap)):
    print(f'Bootstrap iteration {i+1}')
    np.random.seed(i)
    sample_indices = np.random.choice(np.arange(len(x)), size=len(x), replace=True)
    z_sample = z[sample_indices]
    x_sample = x[sample_indices]
    y_sample = y[sample_indices]
    model = neuralPLSI(family='binary')
    model.fit(x_sample, z_sample, y_sample)

    beta_stack.append(model.beta)
    gamma_stack.append(model.gamma)
    g_stack.append(model.g_function(g_grid))

beta_stack = np.array(beta_stack)
gamma_stack = np.array(gamma_stack)
g_stack = np.array(g_stack)
g_mean = np.mean(g_stack, axis=0)
g_ub = np.percentile(g_stack, 97.5, axis=0)
g_lb = np.percentile(g_stack, 2.5, axis=0)

beta_ub = np.percentile(beta_stack, 97.5, axis=0)
beta_lb = np.percentile(beta_stack, 2.5, axis=0)
gamma_ub = np.percentile(gamma_stack, 97.5, axis=0)
gamma_lb = np.percentile(gamma_stack, 2.5, axis=0)
beta_se = np.std(beta_stack, axis=0, ddof=1)
gamma_se = np.std(gamma_stack, axis=0, ddof=1)

res = {
    'coefs': np.concatenate([beta_est, gamma_est]),
    'se': np.concatenate([beta_se, gamma_se]),
    'ub': np.concatenate([beta_ub, gamma_ub]),
    'lb': np.concatenate([beta_lb, gamma_lb])
}

pd.DataFrame(res, index=np.concatenate([exposures_pick, covariates])).to_csv('output/nhanes_new_nplsi_results.csv')

plt.figure(figsize=(8, 8))
plt.plot(g_grid, g_fn_pred, color='black')
plt.fill_between(g_grid, g_lb, g_ub, color='blue', alpha=0.3, label='95% CI')
plt.xlabel('index')
plt.ylabel('g(index)')
plt.legend()
plt.tight_layout()
plt.savefig('output/nhanes_nplsi_g_function.png', dpi=300)
plt.close()

import seaborn as sns  # for rugplot
plt.figure(figsize=(8, 8))
plt.plot(g_grid, g_fn_pred, color='black')
plt.fill_between(g_grid, g_lb, g_ub, color='blue', alpha=0.3, label='95% CI')
sns.rugplot(x=gxb, height=0.05, color='black')  # height controls tick size
plt.xlabel('index')
plt.ylabel('g(index)')
plt.tight_layout()
plt.savefig('output/nhanes_nplsi_g_function2.png', dpi=300)
plt.close()