import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = os.listdir('output')
keys = ['n', 'g_fn', 'outcome', 'model', 'seed', 'time', 'performance', 'g_pred', 'beta_boot', 'gamma_boot']
stack = {}
for k in keys:
    stack[k] = []

beta = np.array([1, 0.7, -0.5, 0.5, 0.3, -0.1, 0, 0])
beta = beta / np.linalg.norm(beta)
gamma = np.array([1, -0.5, 0.5])
alpha = 0.05

stack = []
for g_fn in ['linear', 'logsquare', 'sfun', 'sigmoid']:
    for outcome in ['continuous', 'binary']:
        for n in [500, 2000]:
            filename = f'output/bootstrap_PLSI_res_{n}_{g_fn}_{outcome}.json'
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                continue

            with open(filename, 'r') as file:
                res = json.load(file)
                for key in res:
                    res[key] = np.array(res[key])

            for model in ['PLSI', 'NeuralPLSI']:
                mask = (res['g_fn'] == g_fn) & (res['model'] == model) & (res['n'] == n)
                if not np.any(mask):
                    continue

                row = [n, g_fn, outcome, model]

                # --- Standard deviation over replicates ---
                row += res['beta'][mask].std(axis=0).tolist()
                row += res['beta_boot'][mask].std(axis=1, ddof=1).mean(axis=0).tolist()
                row += res['gamma'][mask].std(axis=0).tolist()
                row += res['gamma_boot'][mask].std(axis=1, ddof=1).mean(axis=0).tolist()

                # --- Coverage probability computation ---
                beta_boot = res['beta_boot'][mask]  # shape: (n_reps, n_boot, p)
                gamma_boot = res['gamma_boot'][mask]  # shape: (n_reps, n_boot, q)

                # Percentile CI bounds
                beta_lower = np.percentile(beta_boot, 100 * (alpha / 2), axis=1)
                beta_upper = np.percentile(beta_boot, 100 * (1 - alpha / 2), axis=1)
                gamma_lower = np.percentile(gamma_boot, 100 * (alpha / 2), axis=1)
                gamma_upper = np.percentile(gamma_boot, 100 * (1 - alpha / 2), axis=1)

                # Per-coefficient coverage
                beta_covered = (beta >= beta_lower) & (beta <= beta_upper)
                gamma_covered = (gamma >= gamma_lower) & (gamma <= gamma_upper)

                beta_coverage = beta_covered.mean(axis=0).tolist()
                gamma_coverage = gamma_covered.mean(axis=0).tolist()

                # Append coverage results
                row += beta_coverage
                row += gamma_coverage

                stack.append(row)

print(pd.DataFrame(stack).shape)
pd.DataFrame(stack).to_csv('output/PLSI_boot_results_.csv', index=False)


        

import sys; sys.exit()

stack = pd.DataFrame(stack)
       
palette = sns.color_palette("pastel", n_colors=len(stack['model'].unique()))

g_dict = {
        'linear': lambda x, a=1: a*x,
        'sigmoid': lambda x, a=2: (1/(1+np.exp(-a*x))-0.5)*5,
        'sfun': lambda x: (2/(1+np.exp(-x))-0.2*x-1)*10,
        'logsquare': lambda x: np.log(1 + x**2)
    }

g_grid = np.linspace(-3, 3, 1000)
pnames = {
    'continuous': 'MSE',
    'binary': 'AUROC',
    'cox': 'C-index'
}
for outcome in stack['outcome'].unique():
    res = stack[stack['outcome'] == outcome]
    ns = res['n'].unique()
    for n in ns:
        plt.figure(figsize=(20, 15))
        try:
            for i, g_fn in enumerate(res['g_fn'].unique()):
                res[pnames[outcome]] = res['performance']
                plt.subplot(3, 5, i*5 + 1)
                sns.boxplot(data=res[(res['g_fn'] == g_fn) & (res['n'] == n)], x='model', y=pnames[outcome], hue='model', palette=palette)

                # visualize g function
                plt.subplot(3, 5, i*5 + 2)
                sns.lineplot(x=g_grid, y=g_dict[g_fn](g_grid), label='True', color='black', linestyle='--')
                for j, model in enumerate(res['model'].unique()):
                    g_estimates = np.stack(res[(res['g_fn'] == g_fn) & (res['n'] == n) & (res['model'] == model)]['g_pred']).astype(float)
                    
                    #g_estimates = g_estimates[~np.isnan(g_estimates).any(axis=1)]  # remove NaN rows
                    plt.plot(g_grid, np.mean(g_estimates, 0), label=model, color=palette[j])
                    plt.fill_between(g_grid, np.percentile(g_estimates, 2.5, axis=0), np.percentile(g_estimates, 97.5, axis=0), alpha=0.2, color=palette[j])
                plt.legend()
                plt.plot([0], [0], color='black', marker='s')

                # pivot beta estimates with models and seeds
                plt.subplot(3, 5, i*5 + 3)
                beta_est = res[(res['g_fn'] == g_fn) & (res['n'] == n)][['model', 'seed', 'beta_bias']].explode('beta_bias').reset_index(drop=True)
                beta_est['coefficient'] = beta_est.groupby(['model', 'seed']).cumcount() + 1
                beta_est['coefficient'] = 'beta_' + beta_est['coefficient'].astype(str)
                beta_est.rename(columns={'beta_bias': 'bias'}, inplace=True)
                sns.boxplot(data=beta_est, x='coefficient', y='bias', hue='model', palette=palette)
                plt.axhline(0, color='black', linestyle='--', linewidth=1)
                plt.ylim(-0.9, 0.9)

                # pivot gamma estimates with models appnd seeds
                plt.subplot(3, 5, i*5 + 4)
                gamma_est = res[(res['g_fn'] == g_fn) & (res['n'] == n)][['model', 'seed', 'gamma_bias']].explode('gamma_bias').reset_index(drop=True)
                gamma_est['coefficient'] = gamma_est.groupby(['model', 'seed']).cumcount() + 1
                gamma_est['coefficient'] = 'gamma_' + gamma_est['coefficient'].astype(str)
                gamma_est.rename(columns={'gamma_bias': 'bias'}, inplace=True)
                sns.boxplot(data=gamma_est, x='coefficient', y='bias', hue='model', palette=palette)
                plt.axhline(0, color='black', linestyle='--', linewidth=1)
                plt.ylim(-0.9, 0.9)

                plt.subplot(3, 5, i*5 + 5)
                sns.boxplot(data=res[(res['g_fn'] == g_fn) & (res['n'] == n)], x='model', y='time', hue='model', palette=palette)

            plt.tight_layout()
            plt.savefig(f'output/PLSI_sim_{outcome}_{n:05d}.png')
            plt.close()
        except:
            print(f"Error processing outcome {outcome} with n={n} and g_fn={g_fn}.")