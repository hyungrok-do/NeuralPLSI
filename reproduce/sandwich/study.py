"""Sandwich SE vs bootstrap on the same fits: one (outcome, link) per run.

Usage: python reproduce/sandwich/study.py <outcome> <g> <first seed> <last seed> <B> <out.jsonl>

For each dataset (seeds as the main simulation, n = 2000): the L-BFGS
NeuralPLSI fit (NEURALPLSI_LIBRARY may point at a build with another stopping
rule), its full sandwich SEs,
and a cold-start bootstrap of the same fit (B draws, all saved). Rows are JSON
lines, written as they finish.
"""
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]
from neuralplsi import NeuralPLSI
from neuralplsi.simulation import simulate_data
from sklearn.model_selection import train_test_split

outcome, g, first, last, B, out = sys.argv[1:7]
first, last, B, jobs = int(first), int(last), int(B), int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
done = set()
if os.path.exists(out):
    done = {json.loads(line)['seed'] for line in open(out)}
with open(out, 'a') as f:
    for seed in range(first, last):
        if seed in done:
            continue
        X, Z, y, *_ = simulate_data(4000, outcome=outcome, g_type=g, seed=seed)
        X, _, Z, _, y, _ = train_test_split(X, Z, y, test_size=2000, random_state=seed)
        t = time.time(); m = NeuralPLSI(family=outcome); m.fit(X, Z, y, random_state=1337); tf = time.time()-t
        t = time.time(); s = m.inference_sandwich(X, Z, y, method='full', n_jobs=jobs); ts = time.time()-t
        se_b, se_g = s['beta_se'], s['gamma_se']
        info = dict(newton_beta=s['newton_beta'], newton_gamma=s['newton_gamma'], min_eig=s['min_eigenvalue'],
                    grad_norm=s['gradient_norm'])
        t = time.time(); boot = m.inference_bootstrap(X, Z, y, n_samples=B, random_state=seed, n_jobs=jobs); tb = time.time()-t
        row = dict(outcome=outcome, g=g, seed=seed, beta=m.beta.tolist(), gamma=m.gamma.tolist(),
                   sandwich_beta=se_b.tolist(), sandwich_gamma=se_g.tolist(),
                   boot_se_beta=boot['beta_se'].tolist(), boot_se_gamma=boot['gamma_se'].tolist(),
                   boot_lb_beta=boot['beta_lb'].tolist(), boot_ub_beta=boot['beta_ub'].tolist(),
                   boot_lb_gamma=boot['gamma_lb'].tolist(), boot_ub_gamma=boot['gamma_ub'].tolist(),
                   draws_beta=np.round(boot['beta_samples'], 7).tolist(), draws_gamma=np.round(boot['gamma_samples'], 7).tolist(),
                   iterations=int(m.fit_diagnostics['iterations']), time_fit=tf, time_sandwich=ts, time_bootstrap=tb, **info)
        f.write(json.dumps(row)+'\n'); f.flush()
        print(seed, f'fit {tf:.1f}s sandwich {ts:.1f}s bootstrap {tb:.1f}s', flush=True)
