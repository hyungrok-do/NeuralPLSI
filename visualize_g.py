import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import json
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

files = os.listdir('output')
keys = ['n', 'g_fn', 'outcome', 'model', 'seed', 'time', 'performance', 'g_pred', 'beta_bias', 'gamma_bias']
stack = {}
for k in keys:
    stack[k] = []

g_dict = {
        'linear': lambda x, a=1: a * x,
        'sigmoid': lambda x, a=2: (1 / (1 + np.exp(-a * x)) - 0.5) * 5,
        'sfun': lambda x: (2 / (1 + np.exp(-x)) - 0.2 * x - 1) * 10,
        'logsquare': lambda x: np.log(1 + x**2)
    }

g_grid = np.linspace(-3, 3, 1000)
model = 'NeuralPLSI'
plt.figure(figsize=(12, 4))

for g_fn, caption in zip(['linear', 'sfun', 'sigmoid'], ['Linear', 's-shaped', 'Sigmoid']):
    g_true = g_dict[g_fn](g_grid)
    outcome = 'continuous'
    n = 2000
    f = f'PLSI_res_{n}_{g_fn}_{outcome}.json'
    with open(f'output/{f}', 'r') as file:
        res = json.load(file)

    del res['seed']
    res = pd.DataFrame(res)
    print(res['model'].unique())

    plt.subplot(1, 3, ['linear', 'sfun', 'sigmoid'].index(g_fn) + 1)
    sns.lineplot(x=g_grid, y=g_true, label='True', color='black', linestyle='--')
    g_estimates = np.stack(res[res['model'] == model]['g_pred']).astype(float)
    plt.plot(g_grid, np.mean(g_estimates, 0), label=model, color=sns.color_palette("pastel")[['NeuralPLSI', 'PLSI'].index(model)])
    plt.fill_between(g_grid, np.percentile(g_estimates, 2.5, axis=0), np.percentile(g_estimates, 97.5, axis=0), alpha=0.2, color=sns.color_palette("pastel")[['NeuralPLSI', 'PLSI'].index(model)])

    plt.legend()
    plt.title(f'{caption}')
    plt.xlabel('index')
    plt.ylabel('g(index)')
    plt.xlim(-3, 3)
    plt.ylim(-4.5, 4.5)

plt.tight_layout()
plt.savefig('output/simulation_g_fn.pdf', dpi=400)
plt.close()
