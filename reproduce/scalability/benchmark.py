"""Fit time and accuracy as n grows: L-BFGS NeuralPLSI, minibatch AdamW NeuralPLSI and SplinePLSI.

Usage: python reproduce/scalability/benchmark.py [--out output/scalability/fits.jsonl] [--sandwich_max_n 20000]

Sigmoid g; continuous, binary and Cox outcomes; n from 500 to 100,000 training
rows; 3 datasets per n (seeds 0-2, fit seed 1337). Every fit runs on one thread
(the engine's fits are single-threaded; only its bootstrap uses threads), one at
a time, so times compare across methods. Each row has the fit time, the angle
between the fitted and true beta, and for NeuralPLSI (continuous and binary,
up to --sandwich_max_n) the time of the full-parameter sandwich on one thread.
A bootstrap of B refits costs about B times the fit time divided by the threads.
Rows already in --out are skipped, so the run can be resumed.
"""
import argparse, json, os, sys, time
os.environ.setdefault('OMP_NUM_THREADS', '1')
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'python'), str(ROOT/'reproduce'/'sandwich')]
from neuralplsi import NeuralPLSI, SplinePLSI
from neuralplsi.simulation import simulate_data, beta as TRUE_BETA

# The AdamW settings of reproduce/main_simulation.py (ADAM).
ADAM = dict(optimizer='adam', max_iter=200, batch_size=64, learning_rate=1e-3, weight_decay=1e-4, patience=10)
SIZES = [500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]


def model(method, outcome):
    if method == 'NeuralPLSI':
        return NeuralPLSI(family=outcome)
    if method == 'NeuralPLSI-Adam':
        return NeuralPLSI(family=outcome, **ADAM)
    return SplinePLSI(family=outcome)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--out', default='output/scalability/fits.jsonl')
    parser.add_argument('--sandwich_max_n', type=int, default=20000)
    parser.add_argument('--seeds', type=int, default=3)
    args = parser.parse_args(argv)
    import torch
    torch.set_num_threads(1)
    from sandwich import Objective
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    done = {(r['method'], r['outcome'], r['n'], r['seed']) for r in map(json.loads, out.open())} if out.exists() else set()
    with out.open('a') as f:
        for n in SIZES:
            for outcome in ('continuous', 'binary', 'cox'):
                for seed in range(args.seeds):
                    X, Z, y, *_ = simulate_data(n, outcome=outcome, g_type='sigmoid', seed=seed)
                    for method in ('NeuralPLSI', 'NeuralPLSI-Adam', 'SplinePLSI'):
                        if (method, outcome, n, seed) in done:
                            continue
                        m = model(method, outcome)
                        start = time.perf_counter(); m.fit(X, Z, y, random_state=1337); seconds = time.perf_counter()-start
                        cosine = abs(float(np.dot(m.beta, TRUE_BETA)))
                        row = dict(method=method, outcome=outcome, n=n, seed=seed, time_fit=seconds,
                                   angle=float(np.degrees(np.arccos(min(1., cosine)))))
                        if method == 'NeuralPLSI':
                            row['iterations'] = int(m.fit_diagnostics['iterations'])
                            if outcome != 'cox' and n <= args.sandwich_max_n:
                                start = time.perf_counter(); Objective(m.net, X, Z, y).sandwich()
                                row['time_sandwich'] = time.perf_counter()-start
                        elif method == 'NeuralPLSI-Adam':
                            row['epochs'] = int(m.fit_diagnostics['iterations'])
                        else:
                            row['iterations'] = int(m.n_iter_)
                        f.write(json.dumps(row)+'\n'); f.flush()
                        print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
