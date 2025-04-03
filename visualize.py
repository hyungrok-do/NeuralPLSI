import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open('output/simulator_results.json', 'r') as f:
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

for n in ns:
    plt.figure(figsize=(25, 20))
    for i, g_fn in enumerate(res['g_fn'].unique()):
        # prediction MSE
        plt.subplot(4, 5, i*5 + 1)
        sns.boxplot(data=res[(res['g_fn'] == g_fn) & (res['n'] == n)], x='model', y='pred_mse', hue='model', palette=palette)

        # visualize g function
        plt.subplot(4, 5, i*5 + 2)
        sns.lineplot(x=g_grid, y=g_dict[g_fn](g_grid), label='True', color='black', linestyle='--')
        for j, model in enumerate(res['model'].unique()):
            g_estimates = np.stack(res[(res['g_fn'] == g_fn) & (res['n'] == n) & (res['model'] == model)]['g_pred']).astype(float)
            #g_estimates = g_estimates[~np.isnan(g_estimates).any(axis=1)]  # remove NaN rows
            plt.plot(g_grid, np.mean(g_estimates, 0), label=model, color=palette[j])
            plt.fill_between(g_grid, np.percentile(g_estimates, 2.5, axis=0), np.percentile(g_estimates, 97.5, axis=0), alpha=0.2, color=palette[j])
        plt.legend()
        plt.plot([0], [0], color='black', marker='s')

        # pivot beta estimates with models and seeds
        plt.subplot(4, 5, i*5 + 3)
        beta_est = res[(res['g_fn'] == g_fn) & (res['n'] == n)][['model', 'seed', 'beta_bias']].explode('beta_bias').reset_index(drop=True)
        beta_est['coefficient'] = beta_est.groupby(['model', 'seed']).cumcount() + 1
        beta_est['coefficient'] = 'beta_' + beta_est['coefficient'].astype(str)
        beta_est.rename(columns={'beta_bias': 'bias'}, inplace=True)
        sns.boxplot(data=beta_est, x='coefficient', y='bias', hue='model', palette=palette)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.ylim(-0.9, 0.9)

        # pivot gamma estimates with models appnd seeds
        plt.subplot(4, 5, i*5 + 4)
        gamma_est = res[(res['g_fn'] == g_fn) & (res['n'] == n)][['model', 'seed', 'gamma_bias']].explode('gamma_bias').reset_index(drop=True)
        gamma_est['coefficient'] = gamma_est.groupby(['model', 'seed']).cumcount() + 1
        gamma_est['coefficient'] = 'gamma_' + gamma_est['coefficient'].astype(str)
        gamma_est.rename(columns={'gamma_bias': 'bias'}, inplace=True)
        sns.boxplot(data=gamma_est, x='coefficient', y='bias', hue='model', palette=palette)
        plt.axhline(0, color='black', linestyle='--', linewidth=1)
        plt.ylim(-0.9, 0.9)

        plt.subplot(4, 5, i*5 + 5)
        sns.boxplot(data=res[(res['g_fn'] == g_fn) & (res['n'] == n)], x='model', y='time', hue='model', palette=palette)

    plt.tight_layout()
    plt.savefig(f'output/simulator_boxplots_{n:05d}.png')
    plt.close()