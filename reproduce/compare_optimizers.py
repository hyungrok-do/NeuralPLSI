#!/usr/bin/env python3
"""Compare optimizer error and coefficient RMSE without changing network size.

Point fits only. Bootstrap coverage requires the main simulation runner.
ThreadPoolExecutor parallelizes independent diagnostics, not bootstrap refits.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import NeuralPLSI, SplinePLSI
from models.native import library
from simulation import beta, gamma, simulate_data
from reproduce.main_simulation import atomic_json


def run(task):
    family, link, data_seed, fit_seed, method, kind, n, epochs, link_epochs = task
    x, z, y, *_, g = simulate_data(2*n, outcome=family, g_type=link, seed=data_seed)
    x, _, z, _, y, _ = train_test_split(x, z, y, test_size=n, random_state=data_seed)
    if method == 'spline_full':
        model = SplinePLSI(family=family)
    else:
        model = NeuralPLSI(family=family, activation='ELU', hidden_units=32, n_hidden_layers=2,
            max_epoch=epochs, link_epochs=link_epochs,
            optimizer='alternating' if method == 'alternating_full' else 'joint',
            validation_fraction=.2 if method == 'joint_split' else 0.)
    start = perf_counter()
    model.fit(x, z, y, random_state=fit_seed)
    seconds = perf_counter()-start
    grid = np.linspace(-3., 3., 301)
    return dict(family=family, link=link, data_seed=data_seed, fit_seed=fit_seed, method=method, kind=kind,
        beta=model.beta.tolist(), gamma=model.gamma.tolist(), beta_mse=float(np.mean((model.beta-beta)**2)),
        gamma_mse=float(np.mean((model.gamma-gamma)**2)), g_mse=float(np.mean((model.g_function(grid)-g(grid))**2)),
        seconds=seconds, diagnostics=model.fit_diagnostics if isinstance(model, NeuralPLSI) else None)


def summarize(records):
    summaries = []
    groups = sorted({(r['kind'], r['family'], r['link'], r['method']) for r in records})
    for kind, family, link, method in groups:
        rows = [r for r in records if (r['kind'], r['family'], r['link'], r['method']) == (kind, family, link, method)]
        row = dict(kind=kind, family=family, link=link, method=method, fits=len(rows))
        for block in ('beta', 'gamma', 'g'):
            row[block+'_rmse'] = float(np.sqrt(np.mean([r[block+'_mse'] for r in rows])))
        row['median_seconds'] = float(np.median([r['seconds'] for r in rows]))
        if kind == 'optimizer_seeds' and len(rows) > 1:
            row['beta_optimizer_sd'] = float(np.sqrt(np.mean(np.var([r['beta'] for r in rows], axis=0, ddof=1))))
        summaries.append(row)
    return summaries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--replicates', type=int, default=20)
    parser.add_argument('--optimizer_seeds', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--link_epochs', type=int, default=1)
    parser.add_argument('--jobs', type=int, default=1)
    parser.add_argument('--out', default='output/optimizer_comparison.json')
    args = parser.parse_args()
    if min(args.n, args.replicates, args.epochs, args.link_epochs, args.jobs) < 1 or args.optimizer_seeds < 0:
        parser.error('Require positive n, replicates, epochs, link_epochs, jobs and nonnegative optimizer_seeds')
    library()  # Build outside timing and worker threads.
    tasks = []
    for family in ('continuous', 'binary', 'cox'):
        for link in ('linear', 'sfun', 'sigmoid'):
            for seed in range(args.replicates):
                for method in ('joint_split', 'joint_full', 'alternating_full', 'spline_full'):
                    tasks.append((family, link, seed, 0, method, 'datasets', args.n, args.epochs, args.link_epochs))
            # No random holdout here: only initialization and minibatch order vary.
            for seed in range(args.optimizer_seeds):
                for method in ('joint_full', 'alternating_full'):
                    tasks.append((family, link, 0, seed, method, 'optimizer_seeds', args.n, args.epochs, args.link_epochs))
    digest = hashlib.sha256()
    for path in sorted((ROOT/'models').rglob('*')):
        if path.suffix in ('.py', '.cpp', '.h'):
            digest.update(str(path.relative_to(ROOT)).encode())
            digest.update(path.read_bytes())
    output = dict(configuration=vars(args), source_sha256=digest.hexdigest(), records=[])
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for index, row in enumerate(pool.map(run, tasks), 1):
            output['records'].append(row)
            if index % 20 == 0 or index == len(tasks):
                atomic_json(args.out, output)
                print(f'{index}/{len(tasks)} fits completed', flush=True)
    output['summary'] = summarize(output['records'])
    atomic_json(args.out, output)
    print(json.dumps(output['summary'], indent=2))


if __name__ == '__main__':
    main()
