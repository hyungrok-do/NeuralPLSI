#!/usr/bin/env python3
"""Time the standalone native implementation. No coverage study is performed."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from neuralplsi import NeuralPLSI
from neuralplsi.native import library, RNG_PROTOCOL, BOOTSTRAP_RNG_PROTOCOL
from neuralplsi.simulation import simulate_data
from reproduce.main_simulation import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--outcomes', nargs='+', choices=['continuous', 'binary', 'cox'], default=['continuous', 'binary', 'cox'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--max_iter', type=int, default=5000)
    parser.add_argument('--bootstrap', type=int, default=3)
    parser.add_argument('--jobs', type=int, default=1, help='Native C++ bootstrap threads (-1: available CPUs)')
    parser.add_argument('--bootstrap_seed', type=int, default=None, help='Defaults to each dataset seed')
    parser.add_argument('--g_grid_n', type=int, default=0, help='Optional link-grid size on [-3, 3]')
    parser.add_argument('--out', default='output/benchmark_native.json')
    args = parser.parse_args(argv)
    if (args.repeats < 1 or args.bootstrap < 2 or args.n < 10 or args.max_iter < 1 or
            args.jobs == 0 or args.g_grid_n < 0 or (args.bootstrap_seed is not None and args.bootstrap_seed < 0)):
        parser.error('Require positive repeats/max_iter, n >= 10, bootstrap >= 2, nonzero jobs and nonnegative grid size/seed')
    start = perf_counter()
    library()
    load_seconds = perf_counter()-start
    records = []
    for family in args.outcomes:
        for seed in range(args.repeats):
            x, z, y, *_ = simulate_data(args.n, outcome=family, g_type='threshold', seed=seed)
            model = NeuralPLSI(family=family, max_iter=args.max_iter, activation='ELU')
            start = perf_counter()
            model.fit(x, z, y, random_state=seed)
            fit_seconds = perf_counter()-start
            start = perf_counter()
            prediction = model.predict(x, z)
            predict_seconds = perf_counter()-start
            bootstrap_seed = seed if args.bootstrap_seed is None else args.bootstrap_seed
            grid = np.linspace(-3, 3, args.g_grid_n) if args.g_grid_n else None
            start = perf_counter()
            boot = model.inference_bootstrap(x, z, y, n_samples=args.bootstrap, random_state=bootstrap_seed,
                                            n_jobs=args.jobs, g_grid=grid)
            row = dict(outcome=family, seed=seed, n=args.n, max_iter=args.max_iter,
                n_bootstrap=args.bootstrap, native_load_seconds=load_seconds, backend='cpp', native_rng=RNG_PROTOCOL,
                bootstrap_engine='cpp', bootstrap_rng=BOOTSTRAP_RNG_PROTOCOL, bootstrap_workers=boot['n_workers'],
                bootstrap_seed=bootstrap_seed, g_grid_n=args.g_grid_n,
                fit_seconds=fit_seconds, prediction_seconds=predict_seconds,
                bootstrap_seconds=perf_counter()-start, diagnostics=model.fit_diagnostics)
            if not np.isfinite(prediction).all() or not np.isfinite(boot['gamma_samples']).all():
                raise FloatingPointError('Nonfinite benchmark output')
            records.append(row)
            atomic_json(args.out, records)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
