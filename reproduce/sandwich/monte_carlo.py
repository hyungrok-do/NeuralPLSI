"""Sandwich SEs over simulated datasets: one JSON line per (setting, seed).

Usage: python reproduce/sandwich/monte_carlo.py <out.jsonl> <datasets per setting> [lbfgs|adam|adam_exact_gamma]

Continuous and binary outcomes, linear, sigmoid and threshold g, n = 2000 as in
the main simulation (seeds 0, 1, ...; fit seed 1337). Each row has the estimates,
the full-parameter sandwich SEs (sandwich.py) and the fit's gradient norm and
smallest Hessian eigenvalue. For the converged arm, set NEURALPLSI_LIBRARY to a
library from build_converged.sh. Runs 10 processes with one thread each.
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]
# The AdamW settings of reproduce/main_simulation.py (ADAM).
ADAM = dict(max_iter=200, batch_size=64, learning_rate=1e-3, weight_decay=1e-4, patience=10)
OPTIONS = {'lbfgs': {}, 'adam': dict(optimizer='adam', **ADAM), 'adam_exact_gamma': dict(optimizer='adam_exact_gamma', **ADAM)}


def task(args):
    import torch
    torch.set_num_threads(1)
    from neuralplsi import NeuralPLSI
    from neuralplsi.simulation import simulate_data
    from sklearn.model_selection import train_test_split
    from sandwich import Objective
    outcome, g, seed, options = args
    X, Z, y, *_ = simulate_data(4000, outcome=outcome, g_type=g, seed=seed)
    X, _, Z, _, y, _ = train_test_split(X, Z, y, test_size=2000, random_state=seed)
    t = time.time(); m = NeuralPLSI(family=outcome, **options); m.fit(X, Z, y, random_state=1337); tf = time.time()-t
    t = time.time(); se_b, se_g, info = Objective(m.net, X, Z, y).sandwich(); ts = time.time()-t
    return dict(outcome=outcome, g=g, seed=seed, beta=m.beta.tolist(), gamma=m.gamma.tolist(), se_beta=se_b.tolist(),
                se_gamma=se_g.tolist(), iterations=int(m.fit_diagnostics['iterations']), time_fit=tf, time_sandwich=ts, **info)


if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2])
    options = OPTIONS[sys.argv[3] if len(sys.argv) > 3 else 'lbfgs']
    jobs = [(o, g, s, options) for o in ('continuous', 'binary') for g in ('linear', 'sigmoid', 'threshold') for s in range(reps)]
    with Pool(10) as pool, open(out, 'w') as f:
        for k, row in enumerate(pool.imap_unordered(task, jobs, chunksize=2)):
            f.write(json.dumps(row)+'\n'); f.flush()
            if k % 100 == 0:
                print(k, len(jobs), flush=True)
