#!/usr/bin/env python3
"""Initialization ablation using the same checked runner as the main study."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import itertools
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT/'python')]  # this checkout's reproduce/ and neuralplsi
from reproduce.main_simulation import bootstrap_jobs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--n_replicates', type=int, default=50)
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--nn_backend', choices=['cpp'], default='cpp')
    parser.add_argument('--out', default='output/ablation_fixed')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    commands = []
    for n, outcome, link in itertools.product([500, 2000], ['continuous', 'binary', 'cox'], ['linear', 'nlrelu', 'sigmoid']):
        for model, initial in [('NeuralPLSI', False), ('NeuralPLSI', True), ('PLSI', False)]:
            command = [sys.executable, str(ROOT/'reproduce/main_simulation.py'),
                '--n_instances', str(n), '--outcome', outcome, '--g_fn', link, '--models', model,
                '--n_replicates', str(args.n_replicates), '--n_bootstrap', str(args.n_bootstrap),
                '--bootstrap_jobs', '1', '--device', 'cpu', '--nn_backend', args.nn_backend,
                '--out', str(Path(args.out).resolve()/f'{args.nn_backend}_init{int(initial)}')]
            if initial:
                command.append('--initial')
            commands.append(command)
    if args.dry_run:
        for command in commands:
            print(' '.join(command))
        return
    environment = dict(os.environ, OMP_NUM_THREADS='1', MKL_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1')
    def run(command):
        subprocess.run(command, cwd=ROOT, env=environment, check=True)
    with ThreadPoolExecutor(max_workers=bootstrap_jobs(args.workers, len(commands))) as executor:
        list(executor.map(run, commands))
    print(f'Completed {len(commands)} ablation settings')


if __name__ == '__main__':
    main()
