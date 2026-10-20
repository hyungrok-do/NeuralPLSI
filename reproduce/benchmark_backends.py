#!/usr/bin/env python3
"""Small deterministic timing/parity check, not a coverage simulation."""
import argparse
import json
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from models import NeuralPLSI
from models.native import library
from simulation import simulate_data
from reproduce.main_simulation import atomic_json


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--outcomes', nargs='+', choices=['continuous', 'binary', 'cox'], default=['continuous', 'binary', 'cox'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--bootstrap', type=int, default=3)
    parser.add_argument('--out', default='output/benchmark_backends.json')
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.bootstrap < 2 or args.n < 10 or args.epochs < 1:
        parser.error('Require positive repeats/epochs, n >= 10, bootstrap >= 2')
    torch.set_num_threads(1)
    start = perf_counter()
    library()  # Exclude compilation/loading from steady-state timings.
    build_seconds = perf_counter()-start
    warm_data = simulate_data(40, seed=0)[:3]
    for backend in ('torch', 'cpp'):
        NeuralPLSI(max_epoch=1, backend=backend, device='cpu').fit(*warm_data)
    records = []
    for family in args.outcomes:
        for seed in range(args.repeats):
            x, z, y, *_ = simulate_data(args.n, outcome=family, g_type='sfun', seed=seed)
            fits, draws, row = {}, {}, dict(outcome=family, seed=seed, n=args.n,
                max_epoch=args.epochs, n_bootstrap=args.bootstrap, native_load_seconds=build_seconds)
            # Alternate order to reduce systematic warm-cache/timing effects.
            for backend in (('torch', 'cpp') if seed % 2 == 0 else ('cpp', 'torch')):
                model = NeuralPLSI(family=family, max_epoch=args.epochs, backend=backend,
                                   activation='ELU', device='cpu')
                start = perf_counter()
                model.fit(x, z, y, random_state=seed)
                fit_time = perf_counter()-start
                start = perf_counter()
                boot = model.inference_bootstrap(x, z, y, n_samples=args.bootstrap,
                                                random_state=seed, n_jobs=1)
                row[backend] = dict(fit_seconds=fit_time, bootstrap_seconds=perf_counter()-start,
                                    diagnostics=model.net.fit_diagnostics)
                fits[backend], draws[backend] = model, boot
            for block in ('beta', 'gamma'):
                row[block+'_max_difference'] = float(np.max(np.abs(getattr(fits['torch'], block)-getattr(fits['cpp'], block))))
                row[block+'_bootstrap_max_difference'] = float(np.max(np.abs(draws['torch'][block+'_samples']-draws['cpp'][block+'_samples'])))
            for phase in ('fit', 'bootstrap'):
                row[phase+'_speedup'] = row['torch'][phase+'_seconds']/row['cpp'][phase+'_seconds']
            records.append(row)
            atomic_json(args.out, records)
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
