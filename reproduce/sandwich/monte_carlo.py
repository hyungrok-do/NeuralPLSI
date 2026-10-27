"""Sandwich inference for NeuralPLSI's fitting methods, against the truth.

Usage: python reproduce/sandwich/monte_carlo.py <out.jsonl> <datasets per setting> [outcomes, e.g. continuous,binary,cox]

The main simulation's design (n = 2000 of 4000 rows, seeds 0, 1, ...; continuous,
binary and Cox outcomes, linear, sigmoid and threshold g). Each dataset gets
converged L-BFGS with the full sandwich, AdamW + frozen correction with the
frozen sandwich, and, as a comparator, AdamW + exact gamma with the full
sandwich. Runs 10 processes, one thread each.
"""
import json, os, sys
os.environ.setdefault('OMP_NUM_THREADS', '1')
from multiprocessing import Pool
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, '..', '..', 'python'), HERE]
ADAM = dict(max_iter=200, batch_size=64, learning_rate=1e-3, weight_decay=1e-4, patience=10)


def result(model, inference):
    keep = ('beta_se', 'gamma_se', 'newton_beta', 'newton_gamma', 'min_eigenvalue')
    return dict(beta=model.beta.tolist(), gamma=model.gamma.tolist(),
                **{k: (v.tolist() if hasattr(v, 'tolist') else v) for k, v in inference.items() if k in keep})


def task(args):
    from neuralplsi import NeuralPLSI
    from neuralplsi.simulation import simulate_data
    from sklearn.model_selection import train_test_split
    outcome, g, seed = args
    X, Z, y, *_ = simulate_data(4000, outcome=outcome, g_type=g, seed=seed)
    X, _, Z, _, y, _ = train_test_split(X, Z, y, test_size=2000, random_state=seed)
    row = dict(outcome=outcome, g=g, seed=seed)
    m = NeuralPLSI(family=outcome).fit(X, Z, y, random_state=seed)
    row['lbfgs'] = result(m, m.inference_sandwich(X, Z, y, method='full', n_jobs=1))
    m = NeuralPLSI(family=outcome, optimizer='adam_exact_gamma', **ADAM).fit(X, Z, y, random_state=1337)
    row['adam_exact_gamma'] = result(m, m.inference_sandwich(X, Z, y, method='full', n_jobs=1))
    m = NeuralPLSI(family=outcome, optimizer='adam_frozen', **ADAM).fit(X, Z, y, random_state=1337)
    row['adam_frozen'] = result(m, m.inference_sandwich(X, Z, y, method='frozen'))
    row['adam_frozen']['correction_iterations'] = m.fit_diagnostics['correction_iterations']
    return row


if __name__ == '__main__':
    out, reps = sys.argv[1], int(sys.argv[2])
    outcomes = sys.argv[3].split(',') if len(sys.argv) > 3 else ('continuous', 'binary', 'cox')
    jobs = [(o, g, s) for o in outcomes for g in ('linear', 'sigmoid', 'threshold') for s in range(reps)]
    with Pool(10) as pool, open(out, 'w') as f:
        for k, row in enumerate(pool.imap_unordered(task, jobs, chunksize=2)):
            f.write(json.dumps(row)+'\n'); f.flush()
            if k % 100 == 0:
                print(k, len(jobs), flush=True)
