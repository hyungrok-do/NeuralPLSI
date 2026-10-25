"""Sandwich SEs of SplinePLSI over simulated datasets: one JSON line per (setting, seed).

Usage: python reproduce/sandwich/spline_monte_carlo.py <out.jsonl> <datasets per setting>

The data and fits of the main simulation (n = 2000, seeds 0, 1, ..., fit seed =
dataset seed, as reproduce/main_simulation.py fits SplinePLSI). Each row has the
fit's estimates with their sandwich SEs (knots held at their fitted values), and
the same after Newton polishing on the fixed-knot objective. Runs 10 processes.
"""
import json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]


def task(args):
    import numpy as np, torch
    torch.set_num_threads(1)
    from neuralplsi import SplinePLSI
    from neuralplsi.simulation import simulate_data
    from sklearn.model_selection import train_test_split
    from spline_sandwich import SplineObjective
    outcome, g, seed = args
    X, Z, y, *_ = simulate_data(4000, outcome=outcome, g_type=g, seed=seed)
    X, _, Z, _, y, _ = train_test_split(X, Z, y, test_size=2000, random_state=seed)
    t = time.time(); m = SplinePLSI(family=outcome); m.fit(X, Z, y, random_state=seed); tf = time.time()-t
    t = time.time(); o = SplineObjective(m, X, Z, y)
    _, _, se_b, se_g, info = o.sandwich(); ts = time.time()-t
    beta2, gamma2, se_b2, se_g2, info2 = o.polish().sandwich()
    return dict(outcome=outcome, g=g, seed=seed, beta=m.beta.tolist(), gamma=m.gamma.tolist(), se_beta=se_b.tolist(),
                se_gamma=se_g.tolist(), grad_norm=info['grad_norm'], min_eig=info['min_eig'],
                polished_beta=beta2.tolist(), polished_gamma=gamma2.tolist(), polished_se_beta=se_b2.tolist(),
                polished_se_gamma=se_g2.tolist(), polished_grad_norm=info2['grad_norm'], time_fit=tf, time_sandwich=ts,
                iterations=int(m.n_iter_), converged=bool(m.converged_))


if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2])
    jobs = [(o, g, s) for o in ('continuous', 'binary') for g in ('linear', 'sigmoid', 'threshold') for s in range(reps)]
    with Pool(10) as pool, open(out, 'w') as f:
        for k, row in enumerate(pool.imap_unordered(task, jobs, chunksize=4)):
            f.write(json.dumps(row)+'\n'); f.flush()
            if k % 300 == 0:
                print(k, len(jobs), flush=True)
