#!/usr/bin/env python3
"""Compare native runtime, exact results and isolated peak RSS with a git revision.

Compilation and data generation are excluded from fit/bootstrap timings. Memory
checks run in separate C++ processes and use zero learning rate for fixed stopping
times; they are allocation checks, not statistical or convergence experiments.
"""
import argparse
import ctypes
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from models import NeuralPLSI
from models import native
from simulation import simulate_data
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
        model = NeuralPLSI(family=family, max_epoch=args.epochs, activation=args.activation,
            optimizer=args.optimizer, validation_fraction=args.validation_fraction,
            link_epochs=args.link_epochs)
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
    parser.add_argument('--baseline-ref', default='114133d4a94f77a8b7ee83ac4aee1a1e296beea1')
    parser.add_argument('--n', type=int, default=2000)
    parser.add_argument('--bootstrap', type=int, default=24)
    parser.add_argument('--bootstrap-seed', type=int, default=11)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--grid', type=int, default=1000)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--activation', choices=list(native.ACTIVATIONS), default='ELU')
    parser.add_argument('--optimizer', choices=list(native.OPTIMIZERS), default='alternating')
    parser.add_argument('--validation-fraction', type=float, default=0.)
    parser.add_argument('--link-epochs', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--jobs', nargs='+', type=int, default=[1, 4])
    parser.add_argument('--outcomes', nargs='+', choices=['continuous', 'binary', 'cox'],
                        default=['continuous', 'binary', 'cox'])
    parser.add_argument('--memory-n', type=int, default=100000, help='0 skips isolated RSS checks')
    parser.add_argument('--memory-only', action='store_true', help='Run only the isolated RSS checks')
    parser.add_argument('--out', default='output/native_revision_comparison.json')
    args = parser.parse_args(argv)
    if (args.n < 10 or args.bootstrap < 2 or args.repeats < 1 or args.epochs < 1 or
            args.grid < 0 or args.seed < 0 or args.bootstrap_seed < 0 or
            0 in args.jobs or args.link_epochs < 1 or not 0 <= args.validation_fraction < 1 or
            args.memory_n < 0 or 0 < args.memory_n < 10 or
            (args.memory_only and not args.memory_n)):
        parser.error('Invalid dataset, bootstrap, epoch, worker or memory-check configuration')
    # Resolve to a commit before using the ref in further git commands.
    baseline = run(['git', 'rev-parse', '--verify', '--end-of-options',
                    args.baseline_ref+'^{commit}']).strip()
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    flags = ['-O3', '-std=c++17', '-ffp-contract=off', '-pthread']
    current = native.library()
    results = dict(baseline_commit=baseline, current_commit=run(['git', 'rev-parse', 'HEAD']).strip(),
        working_tree_dirty=bool(run(['git', 'status', '--porcelain']).strip()),
        current_library=native.build_library().name, compiler=run(compiler+['--version']).splitlines()[0],
        settings=vars(args), timings=[], memory=[])
    with tempfile.TemporaryDirectory(prefix='neuralplsi-compare-') as temporary:
        temp = Path(temporary)
        for filename in ('core.cpp', 'core.h'):
            (temp/filename).write_text(run(['git', 'show', baseline+':models/native/'+filename]))
        reference_path = temp/'reference.so'
        run(compiler+flags+['-shared', '-fPIC', str(temp/'core.cpp'), '-o', str(reference_path)])
        reference = load_reference(reference_path, current)
        for family in ([] if args.memory_only else args.outcomes):
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
        if args.memory_n:
            # Each executable starts a new process, avoiding Python's RSS and
            # allocator retention from previous measurements. Include input data.
            for label, source in [('before', temp), ('after', ROOT/'models'/'native')]:
                target = temp/('memory-'+label)
                run(compiler+flags+['-I', str(source), str(source/'core.cpp'),
                    str(ROOT/'reproduce'/'native_memory_check.cpp'), '-o', str(target)])
                for B, jobs in [(0, 1), (4, 4)]:
                    row = json.loads(run([str(target), str(args.memory_n), str(B), str(jobs), '200']))
                    row.update(version=label)
                    if row['joint_epochs'] != row['max_epoch'] or row['gamma_iterations'] < 0:
                        raise AssertionError('Memory check did not complete the neural epoch budget')
                    results['memory'].append(row)
                    print(json.dumps(row), flush=True)
            atomic_json(args.out, results)


if __name__ == '__main__':
    main()
