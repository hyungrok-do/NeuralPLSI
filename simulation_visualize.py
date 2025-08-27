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
for g_fn in ['linear', 'sfun', 'sigmoid']:
    for outcome in ['continuous', 'binary', 'cox']:
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
                beta_mean = res['beta'][mask].mean(axis=0)    
                gamma_mean = res['gamma'][mask].mean(axis=0)  

                beta_bias = (beta_mean - beta).tolist()
                gamma_bias = (gamma_mean - gamma).tolist()

                # Coverage probability
                beta_boot = res['beta_boot'][mask]
                gamma_boot = res['gamma_boot'][mask]

                beta_lower = np.percentile(beta_boot, 100 * (alpha / 2), axis=1)
                beta_upper = np.percentile(beta_boot, 100 * (1 - alpha / 2), axis=1)
                gamma_lower = np.percentile(gamma_boot, 100 * (alpha / 2), axis=1)
                gamma_upper = np.percentile(gamma_boot, 100 * (1 - alpha / 2), axis=1)

                beta_covered = (beta >= beta_lower) & (beta <= beta_upper)
                gamma_covered = (gamma >= gamma_lower) & (gamma <= gamma_upper)

                beta_coverage = beta_covered.mean(axis=0).tolist()
                gamma_coverage = gamma_covered.mean(axis=0).tolist()
                
                # Append to row
                row += res['beta'][mask].std(axis=0).tolist()
                row += res['beta_boot'][mask].std(axis=1, ddof=1).mean(axis=0).tolist()
                row += res['gamma'][mask].std(axis=0).tolist()
                row += res['gamma_boot'][mask].std(axis=1, ddof=1).mean(axis=0).tolist()
                row += beta_bias
                row += gamma_bias
                row += beta_coverage
                row += gamma_coverage

                stack.append(row)

colnames = ['n', 'g_fn', 'outcome', 'model']
colnames += [f'beta_sd_{i+1}' for i in range(len(beta))]
colnames += [f'gamma_sd_{i+1}' for i in range(len(gamma))]
colnames += [f'beta_se_{i+1}' for i in range(len(beta))]
colnames += [f'gamma_se_{i+1}' for i in range(len(gamma))]
colnames += [f'beta_bias_{i+1}' for i in range(len(beta))]
colnames += [f'gamma_bias_{i+1}' for i in range(len(gamma))]
colnames += [f'beta_coverage_{i+1}' for i in range(len(beta))]
colnames += [f'gamma_coverage_{i+1}' for i in range(len(gamma))]

#print(pd.DataFrame(stack).shape)
res = pd.DataFrame(stack, columns=colnames)
print(len(colnames), res.shape)
res.to_csv('output/simulation_results.csv', index=False)


        