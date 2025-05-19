import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('output/standard_error_results.json', 'r') as f:
    res = json.load(f)

res = pd.DataFrame(res)

palette = sns.color_palette("Paired", n_colors=len(res['model'].unique()))

g_dict = {
        'linear': lambda x, a=1: a*x,
        'sigmoid': lambda x, a=2: (1/(1+np.exp(-a*x))-0.5)*5,
        'sfun': lambda x: (2/(1+np.exp(-x))-0.2*x-1)*10,
        'logsquare': lambda x: np.log(1 + x**2)
    }

g_grid = np.linspace(-3, 3, 1000)
ns = res['n'].unique()

summary = []
for n in ns:
    for i, g_fn in enumerate(res['g_fn'].unique()):
        subset = res[(res['g_fn'] == g_fn) & (res['n'] == n)]
        print(g_fn)

        sd = np.concatenate([
            np.stack(res['beta'].apply(np.array).values).std(0),
            np.stack(res['gamma'].apply(np.array).values).std(0)
        ])

        se_sandwich = np.concatenate([
            np.stack(res['beta_se_sandwich'].apply(np.array).values).mean(0),
            np.stack(res['gamma_se_sandwich'].apply(np.array).values).mean(0)
        ])

        se_bootstrap = np.concatenate([
            np.stack(res['beta_se_bootstrap'].apply(np.array).values).mean(0),
            np.stack(res['gamma_se_bootstrap'].apply(np.array).values).mean(0)
        ])

        cv_sandwich  = np.concatenate([
            np.stack(res['beta_coverage_sandwich'].apply(np.array).values).mean(0),
            np.stack(res['gamma_coverage_sandwich'].apply(np.array).values).mean(0)
        ])

        cv_bootstrap = np.concatenate([
            np.stack(res['beta_coverage_bootstrap'].apply(np.array).values).mean(0),
            np.stack(res['gamma_coverage_bootstrap'].apply(np.array).values).mean(0)
        ])

        summary.append(pd.DataFrame([sd, se_sandwich, se_bootstrap, cv_sandwich, cv_bootstrap], index=['sd', 'se_sandwich', 'se_bootstrap', 'coverage_sandwich', 'coverage_bootstrap']).T)
        print(res['estimate_time'].mean(), res['sandwich_time'].mean(), res['bootstrap_time'].mean())

for i in range(len(summary)):
    summary[i].to_csv(f'output/summary_{i}.csv', index=False)
    print(summary[i])