
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('output/simulator_results.json', 'r') as f:
    res = json.load(f)

res = pd.DataFrame(res)

ns = res['n'].unique()
for n in [500, 2000]:
    plt.figure(figsize=(25, 20))
    for i, g_fn in enumerate():
sns.lineplot(x=g_grid, y=g_dict[g_fn](g_grid), label='True', color='black', linestyle='--')
for j, model in enumerate(res['model'].unique()):
    g_estimates = np.stack(res[(res['g_fn'] == g_fn) & (res['n'] == n) & (res['model'] == model)]['g_pred']).astype(float)
    #g_estimates = g_estimates[~np.isnan(g_estimates).any(axis=1)]  # remove NaN rows
    plt.plot(g_grid, np.mean(g_estimates, 0), label=model, color=palette[j])
    plt.fill_between(g_grid, np.percentile(g_estimates, 2.5, axis=0), np.percentile(g_estimates, 97.5, axis=0), alpha=0.2, color=palette[j])
plt.legend()
plt.plot([0], [0], color='black', marker='s')