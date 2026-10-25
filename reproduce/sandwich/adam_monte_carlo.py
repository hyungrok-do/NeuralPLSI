"""Sandwich SEs at full-batch Adam fits run to first-order optimality: one JSON line per (setting, seed).

Usage: python reproduce/sandwich/adam_monte_carlo.py <out.jsonl> <datasets per setting> [full|minibatch]

The data of monte_carlo.py (n = 2000, seeds 0, 1, ...). Each fit starts from the
engine's initialization (fit seed 1337), runs full-batch Adam on the penalized
objective (adam_first_order.fit: learning rate 3e-3 / (1 + step/5000), at most
40,000 steps) until the gradient norm is below 1e-4, and is then written back
to the native model (in single precision) for the sandwich. With minibatch,
Adam takes minibatches of 64 rows of all the data instead (learning rate
1e-3 / (1 + step/20000)), with the same first-order check every epoch and a
budget of 160,000 steps: about the row-gradients of the full-batch fits. beta is reported
with a positive first coordinate, as the package does. Runs 10 processes.
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]


def task(args):
    import time
    import numpy as np, torch
    torch.set_num_threads(1)
    from neuralplsi import NeuralPLSI
    from neuralplsi.simulation import simulate_data
    from sklearn.model_selection import train_test_split
    from sandwich import Objective
    from adam_first_order import fit
    outcome, g, seed, mode = args
    X, Z, y, *_ = simulate_data(4000, outcome=outcome, g_type=g, seed=seed)
    X, _, Z, _, y, _ = train_test_split(X, Z, y, test_size=2000, random_state=seed)
    init = NeuralPLSI(family=outcome, max_iter=0); init.fit(X, Z, y, random_state=1337)
    net = init.net
    if mode == 'minibatch':
        w, info = fit(net, X, Z, y, batch=64, lr=1e-3, decay=20000., tol=1e-4, max_steps=160000, seed=seed)
    else:
        w, info = fit(net, X, Z, y, lr=3e-3, decay=5000., tol=1e-4, max_steps=40000)
    net.set_weights(w.numpy().astype(np.float32))
    t = time.time(); se_b, se_g, s_info = Objective(net, X, Z, y).sandwich(); ts = time.time()-t
    p, q = X.shape[1], Z.shape[1]
    beta = float(net.hyper[1])*w[:p].numpy()/np.linalg.norm(w[:p].numpy())
    if beta[0] < 0:
        beta = -beta
    return dict(outcome=outcome, g=g, seed=seed, beta=beta.tolist(), gamma=w[p:p+q].numpy().tolist(),
                se_beta=se_b.tolist(), se_gamma=se_g.tolist(), steps=info['steps'], adam_grad_norm=info['grad_norm'],
                objective=info['objective'], time_fit=info['seconds'], time_sandwich=ts, **s_info)


if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2])
    mode = sys.argv[3] if len(sys.argv) > 3 else 'full'
    jobs = [(o, g, s, mode) for o in ('continuous', 'binary') for g in ('linear', 'sigmoid', 'threshold') for s in range(reps)]
    with Pool(10) as pool, open(out, 'w') as f:
        for k, row in enumerate(pool.imap_unordered(task, jobs, chunksize=2)):
            f.write(json.dumps(row)+'\n'); f.flush()
            if k % 100 == 0:
                print(k, len(jobs), flush=True)
