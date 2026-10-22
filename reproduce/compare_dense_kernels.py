#!/usr/bin/env python3
"""Compare dense-kernel runtime and exact results with the pre-optimization revision.

Both versions use the same full-batch L-BFGS estimator and native bootstrap.
Compilation and data generation are excluded from the timing intervals.
"""
import argparse
import ctypes
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from neuralplsi import NeuralPLSI
from neuralplsi import native
from neuralplsi.simulation import simulate_data
from reproduce.main_simulation import atomic_json


def run(command):
    return subprocess.run(command, cwd=ROOT, check=True, text=True, capture_output=True).stdout


def load_reference(path, prototype):
    lib = ctypes.CDLL(str(path))
    for name in dir(prototype):
        if name.startswith('nplsi_'):
            source, target = getattr(prototype, name), getattr(lib, name)
            target.argtypes, target.restype = source.argtypes, source.restype
    return lib


def measure(lib, family, x, z, y, args, jobs):
    grid = np.linspace(-3, 3, args.grid) if args.grid else None
    with patch.object(native, 'library', return_value=lib):
        model = NeuralPLSI(family=family, max_iter=args.max_iter, activation=args.activation)
        start = perf_counter()
        model.fit(x, z, y, random_state=args.seed)
        fit_seconds = perf_counter()-start
        start = perf_counter()
        prediction = model.predict(x, z)
        prediction_seconds = perf_counter()-start
        start = perf_counter()
        boot = model.inference_bootstrap(x, z, y, n_samples=args.bootstrap,
            random_state=args.bootstrap_seed, n_jobs=jobs, g_grid=grid)
        bootstrap_seconds = perf_counter()-start
        result = dict(fit_seconds=fit_seconds, prediction_seconds=prediction_seconds,
                      bootstrap_seconds=bootstrap_seconds, workers=boot['n_workers'],
                      diagnostics=model.fit_diagnostics)
        snapshot = dict(state=model.net.to_bytes(), prediction=prediction,
                        **{k: v for k, v in boot.items() if isinstance(v, np.ndarray)})
    return result, snapshot


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline-ref', default='b74333674c63adee16b61a64f4aad9240a78a56b')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--bootstrap', type=int, default=4)
    parser.add_argument('--bootstrap-seed', type=int, default=11)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid', type=int, default=301)
    parser.add_argument('--max-iter', type=int, default=5000)
    parser.add_argument('--activation', choices=list(native.ACTIVATIONS), default='ELU')
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--jobs', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--outcomes', nargs='+', choices=['continuous', 'binary', 'cox'],
                        default=['continuous', 'binary', 'cox'])
    parser.add_argument('--out', default='output/dense_kernel_benchmark.json')
    args = parser.parse_args(argv)
    if (args.n < 10 or args.bootstrap < 2 or args.repeats < 1 or args.max_iter < 1 or
            args.grid < 0 or args.seed < 0 or args.bootstrap_seed < 0 or 0 in args.jobs):
        parser.error('Invalid dataset, bootstrap, iteration, worker or grid configuration')
    baseline = run(['git', 'rev-parse', '--verify', '--end-of-options',
                    args.baseline_ref+'^{commit}']).strip()
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    flags = ['-O3', '-std=c++17', '-ffp-contract=off', '-pthread', '-shared', '-fPIC']
    current = native.library()
    source = ROOT/'core'/'src'/'neuralplsi.cpp'
    results = dict(baseline_commit=baseline, current_commit=run(['git', 'rev-parse', 'HEAD']).strip(),
        working_tree_dirty=bool(run(['git', 'status', '--porcelain']).strip()),
        current_core_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        current_library=native.library_path().name,
        compiler=run(compiler+['--version']).splitlines()[0], compiler_flags=flags,
        platform=platform.platform(), settings=vars(args), timings=[])
    with tempfile.TemporaryDirectory(prefix='neuralplsi-dense-') as temporary:
        temp = Path(temporary)
        # Baseline revisions predate core/: the engine was models/native/core.cpp.
        for filename in ('core.cpp', 'core.h'):
            (temp/filename).write_text(run(['git', 'show', baseline+':models/native/'+filename]))
        results['baseline_core_sha256'] = hashlib.sha256((temp/'core.cpp').read_bytes()).hexdigest()
        reference_path = temp/'reference.so'
        run(compiler+flags+[str(temp/'core.cpp'), '-o', str(reference_path)])
        reference = load_reference(reference_path, current)
        for family in args.outcomes:
            x, z, y, *_ = simulate_data(args.n, outcome=family, g_type='sfun', seed=args.seed)
            for jobs in args.jobs:
                for repeat in range(args.repeats):
                    versions = [('before', reference), ('after', current)]
                    if repeat % 2:
                        versions.reverse()
                    snapshots, rows = [], []
                    for label, lib in versions:
                        row, snapshot = measure(lib, family, x, z, y, args, jobs)
                        row.update(version=label, family=family, jobs=jobs, repeat=repeat)
                        rows.append(row)
                        snapshots.append(snapshot)
                    if snapshots[0]['state'] != snapshots[1]['state']:
                        raise AssertionError('Fitted parameters, settings or stopping diagnostics differ')
                    for key in snapshots[0]:
                        if key != 'state':
                            np.testing.assert_array_equal(snapshots[0][key], snapshots[1][key], err_msg=key)
                    for row in rows:
                        row['reference_exact'] = True
                        results['timings'].append(row)
                        print(json.dumps(row), flush=True)
                    atomic_json(args.out, results)


if __name__ == '__main__':
    main()
