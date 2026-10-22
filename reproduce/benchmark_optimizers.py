#!/usr/bin/env python3
"""Matched full-data optimizer pilot, separate from bootstrap coverage studies.

Accuracy varies the simulated dataset with fit seed zero. Stability holds each
scenario's dataset fixed and varies initialization and minibatch seeds. Both
neural optimizers use the same 2x32 ELU link, all n rows, and 200 neural epochs.
Python only schedules these independent point fits. Production bootstrap stays
inside one C++ call. Timings with --jobs>1 include CPU contention.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
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
from models.native import build_library
from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA


def fit_case(task, args):
    kind, family, link, data_seed, fit_seed, method = task
    x, z, y, _, _, g = simulate_data(2*args.n, outcome=family, g_type=link, seed=data_seed)
    x, _, z, _, y, _ = train_test_split(x, z, y, test_size=args.n, random_state=data_seed)
    model = (SplinePLSI(family=family) if method == 'spline' else
             NeuralPLSI(family=family, activation='ELU', optimizer=method,
                        max_epoch=args.epochs, validation_fraction=0., link_epochs=args.link_epochs))
    start = perf_counter()
    model.fit(x, z, y, random_state=fit_seed)
    elapsed = perf_counter()-start
    grid = np.linspace(-3, 3, 301)
    result = dict(kind=kind, family=family, link=link, data_seed=data_seed,
                  fit_seed=fit_seed, method=method, beta=model.beta.tolist(), gamma=model.gamma.tolist(),
                  beta_mse=float(np.mean((model.beta-TRUE_BETA)**2)),
                  gamma_mse=float(np.mean((model.gamma-TRUE_GAMMA)**2)),
                  g_mse=float(np.mean((model.g_function(grid)-g(grid))**2)), seconds=elapsed)
    if method != 'spline':
        result['diagnostics'] = model.fit_diagnostics
    return result


def summarize(records):
    summary = []
    groups = sorted({(r['kind'], r['family'], r['link'], r['method']) for r in records})
    for kind, family, link, method in groups:
        rows = [r for r in records if (r['kind'], r['family'], r['link'], r['method']) ==
                (kind, family, link, method)]
        values = np.array([r['beta'] for r in rows])
        summary.append(dict(kind=kind, family=family, link=link, method=method, count=len(rows),
            beta_rmse=float(np.sqrt(np.mean([r['beta_mse'] for r in rows]))),
            gamma_rmse=float(np.sqrt(np.mean([r['gamma_mse'] for r in rows]))),
            g_rmse=float(np.sqrt(np.mean([r['g_mse'] for r in rows]))),
            beta_rms_sd=float(np.sqrt(np.mean(values.var(axis=0, ddof=1)))) if len(rows)>1 else None,
            median_seconds=float(np.median([r['seconds'] for r in rows]))))
    return summary


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--n', type=int, default=2000)
    p.add_argument('--datasets', type=int, default=10)
    p.add_argument('--starts', type=int, default=10)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--link_epochs', type=int, default=1)
    p.add_argument('--jobs', type=int, default=1)
    p.add_argument('--families', nargs='+', default=['continuous', 'binary', 'cox'])
    p.add_argument('--links', nargs='+', default=['linear', 'sigmoid', 'sfun'])
    p.add_argument('--out', type=Path, default=ROOT/'output/optimizer_pilot.json')
    args = p.parse_args()
    if min(args.n, args.datasets, args.starts, args.epochs, args.link_epochs, args.jobs) < 1:
        p.error('All counts must be positive')
    library = build_library().name  # compilation excluded from fit timing
    digest = hashlib.sha256()
    for name in ('models/native/core.cpp', 'models/native/core.h', 'models/native/__init__.py',
                 'models/NeuralPLSI.py', 'models/PLSI.py', 'models/native/spline.cpp',
                 'simulation/__init__.py', 'reproduce/benchmark_optimizers.py'):
        digest.update(name.encode())
        digest.update((ROOT/name).read_bytes())
    config = {**vars(args), 'out': str(args.out), 'native_build': library,
              'source_sha256': digest.hexdigest(), 'hidden_units': 32, 'n_hidden_layers': 2,
              'activation': 'ELU', 'validation_fraction': 0., 'numpy': np.__version__}
    tasks = []
    for family in args.families:
        for link in args.links:
            for seed in range(args.datasets):
                for method in ('joint', 'alternating', 'spline'):
                    tasks.append(('accuracy', family, link, seed, 0, method))
            for seed in range(args.starts):
                for method in ('joint', 'alternating'):
                    tasks.append(('stability', family, link, 0, seed, method))
    results = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    started = perf_counter()
    with ThreadPoolExecutor(max_workers=args.jobs) as pool:
        futures = [pool.submit(fit_case, task, args) for task in tasks]
        for future in as_completed(futures):
            results.append(future.result())  # no silent skips or retries
            if len(results) % 20 == 0 or len(results) == len(tasks):
                print(f'{len(results)}/{len(tasks)} fits, {perf_counter()-started:.1f}s', flush=True)
    results.sort(key=lambda r: (r['kind'], r['family'], r['link'], r['data_seed'], r['fit_seed'], r['method']))
    report = dict(configuration=config, summary=summarize(results), records=results)
    temporary = args.out.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    temporary.replace(args.out)
    for kind in ('accuracy', 'stability'):
        print(kind)
        for family in args.families:
            for link in args.links:
                rows = [r for r in report['summary'] if (r['kind'], r['family'], r['link']) == (kind, family, link)]
                metric = 'beta_rmse' if kind == 'accuracy' else 'beta_rms_sd'
                print(family, link, {r['method']: round(r[metric], 6) for r in rows})


if __name__ == '__main__':
    main()
