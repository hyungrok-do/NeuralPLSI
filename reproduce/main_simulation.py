#!/usr/bin/env python3
"""
Simulation study comparing SplinePLSI and NeuralPLSI with bootstrap inference.

For each replicate the script records:
  - point estimates  (beta, gamma, g-function)
  - standard errors  and 95 % confidence intervals
  - computation time  for fitting and bootstrap inference
  - predictive performance  (MSE / AUC / C-index)
"""

import argparse
import hashlib
import json
import os
import sys
import tempfile
from pathlib import Path
from contextlib import contextmanager
from time import perf_counter

import numpy as np
import sklearn
import scipy
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from models import SplinePLSI, NeuralPLSI
from models.base import bootstrap_jobs
from models.native import BOOTSTRAP_RNG_PROTOCOL
from simulation import simulate_data, beta as TRUE_BETA, gamma as TRUE_GAMMA


def to_json(obj):
    """Recursively convert numpy types to JSON-safe Python types."""
    if isinstance(obj, dict):
        return {k: to_json(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, (np.floating, np.integer)):
        return float(obj)
    if isinstance(obj, list):
        return [to_json(v) for v in obj]
    return obj


def evaluate(model, X_te, Z_te, y_te, outcome):
    """Compute predictive performance on the test set."""
    if outcome == 'continuous':
        preds = model.predict(X_te, Z_te)
        return float(np.mean((preds - y_te) ** 2))
    elif outcome == 'binary':
        preds = model.predict_proba(X_te, Z_te)
        return float(roc_auc_score(y_te, preds))
    elif outcome == 'cox':
        preds = model.predict(X_te, Z_te)  # Concordance is invariant to exp; avoid overflow.
        return float(concordance_index(y_te[:, 0], -preds, y_te[:, 1]))
    raise ValueError(f"Unknown outcome: {outcome}")


def parse_models(arg):
    """Parse model names from CLI into canonical keys."""
    alias = {'plsi': 'PLSI', 'splineplsi': 'PLSI',
             'neuralplsi': 'NeuralPLSI', 'nplsi': 'NeuralPLSI'}
    if len(arg) == 1 and arg[0].lower() in ('all', 'both'):
        return ['PLSI', 'NeuralPLSI']
    out = []
    for m in arg:
        key = alias.get(m.strip().lower())
        if key is None:
            raise ValueError(f"Unknown model: {m}. Use PLSI, NeuralPLSI, or all.")
        if key not in out:
            out.append(key)
    return out


def output_path(base, model, n, g_fn, outcome, x_dist, initial=False):
    """Resolve the JSON output path for a model run."""
    init_tag = 'init1' if initial else 'init0'
    name = f"simulation+{model}+{n}+{g_fn}+{outcome}+{x_dist}+{init_tag}.json"
    if base is None:
        os.makedirs("output/simulation", exist_ok=True)
        return os.path.join("output/simulation", name)
    os.makedirs(base, exist_ok=True)
    return os.path.join(base, name)


MODEL_CLS = {'PLSI': SplinePLSI, 'NeuralPLSI': NeuralPLSI}


def run_configuration(args, model):
    """Record the estimator, inference settings and exact simulation source."""
    paths = ['reproduce/main_simulation.py', 'simulation/__init__.py',
             'models/base.py',
             'models/NeuralPLSI.py' if model == 'NeuralPLSI' else 'models/PLSI.py']
    paths += sorted(str(p.relative_to(ROOT)) for p in Path(ROOT, 'models/native').glob('*')
                    if p.suffix in ('.py', '.cpp', '.h'))
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.encode())
        with open(os.path.join(ROOT, path), 'rb') as source:
            digest.update(source.read())
    native_build = None
    if model != 'NeuralPLSI' or args.nn_backend == 'cpp':
        from models.native import build_library
        native_build = build_library().name
    resolved_device = (str(NeuralPLSI(device=args.device, backend=args.nn_backend).device)
                       if model == 'NeuralPLSI' else 'cpu')
    return dict(protocol_version=7, model=model, n=args.n_instances,
                g_fn=args.g_fn, outcome=args.outcome, x_dist=args.exposure_dist,
                activation=args.activation if model == 'NeuralPLSI' else None,
                n_bootstrap=args.n_bootstrap, ci=0.95,
                quantile_method=(getattr(args, 'nn_bootstrap_quantile', 'linear')
                                 if model == 'NeuralPLSI' else 'linear'),
                fit_random_state=args.fit_seed if model == 'NeuralPLSI' else 'replicate_seed',
                initial=args.initial if model == 'NeuralPLSI' else None,
                training_policy=('full_budget_validation_checkpoint_exact_gamma_v1'
                                 if model == 'NeuralPLSI' else None),
                backend=args.nn_backend if model == 'NeuralPLSI' else 'cpp',
                device=resolved_device, native_build=native_build,
                normalization='l2', native_rng='splitmix64_v1',
                bootstrap_engine='cpp', bootstrap_rng=BOOTSTRAP_RNG_PROTOCOL,
                g_grid=[args.g_grid_min, args.g_grid_max, args.g_grid_n],
                source_sha256=digest.hexdigest(), numpy=np.__version__,
                sklearn=sklearn.__version__, scipy=scipy.__version__)


def load_compatible_results(path, configuration):
    """Never mix unverified legacy outputs with a new simulation run."""
    if not os.path.exists(path):
        return []
    with open(path) as stream:
        records = json.load(stream)
    if not isinstance(records, list):
        raise ValueError(f'Expected a list of replicates in {path}')
    seeds = set()
    for row in records:
        if row.get('run_configuration') != configuration:
            raise ValueError(
                f'Cannot resume {path}: its code/settings differ or its run '
                'configuration is missing. Use a new --out directory for this run.')
        if row['seed'] in seeds:
            raise ValueError(f'Duplicate simulation seed {row["seed"]} in {path}')
        seeds.add(row['seed'])
    return records


@contextmanager
def result_lock(path):
    import fcntl
    lock = open(str(path)+'.lock', 'a')
    try:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f'Another job is writing {path}') from exc
        yield
    finally:
        lock.close()


def atomic_json(path, records):
    """A terminated job must not destroy its last complete checkpoint."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, delete=False) as stream:
            temporary = stream.name
            json.dump(to_json(records), stream, allow_nan=False)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if temporary and os.path.exists(temporary):
            os.unlink(temporary)


def run_replicate(args, mname, seed, configuration):
    n = args.n_instances
    X, Z, y, *_ = simulate_data(2*n, outcome=args.outcome, g_type=args.g_fn,
                               seed=seed, x_dist=args.exposure_dist)
    X_tr, X_te, Z_tr, Z_te, y_tr, y_te = train_test_split(X, Z, y, test_size=n, random_state=seed)
    if mname == 'NeuralPLSI':
        model = NeuralPLSI(family=args.outcome, activation=args.activation,
            initial=args.initial, device=args.device, backend=args.nn_backend)
        fit_options = {'random_state': args.fit_seed}
    else:
        model = SplinePLSI(family=args.outcome)
        fit_options = {'random_state': seed}
    start = perf_counter()
    model.fit(X_tr, Z_tr, y_tr, **fit_options)
    time_fit = perf_counter()-start
    performance = evaluate(model, X_te, Z_te, y_te, args.outcome)
    if not np.isfinite(performance):
        raise FloatingPointError('Nonfinite test-set performance')
    grid = np.linspace(args.g_grid_min, args.g_grid_max, args.g_grid_n)
    g_estimate = model.g_function(grid)
    start = perf_counter()
    options = {'quantile_method': args.nn_bootstrap_quantile} if mname == 'NeuralPLSI' else {}
    jobs = args.bootstrap_jobs
    boot = model.inference_bootstrap(X_tr, Z_tr, y_tr, n_samples=args.n_bootstrap,
        random_state=seed, ci=.95, g_grid=grid, n_jobs=jobs, **options)
    time_boot = perf_counter()-start
    beta, gamma = model.beta, model.gamma
    row = dict(run_configuration=configuration, seed=seed, n=n, g_fn=args.g_fn,
        x_dist=args.exposure_dist, outcome=args.outcome, model=mname, performance=performance,
        time_fit=time_fit, time_bootstrap=time_boot, beta_est=beta, gamma_est=gamma,
        g_grid=grid, g_pred=g_estimate, g_se=boot['g_se'], g_lb=boot['g_lb'], g_ub=boot['g_ub'],
        n_bootstrap=len(boot['beta_samples']))
    for block, estimate, target in [('beta', beta, TRUE_BETA), ('gamma', gamma, TRUE_GAMMA)]:
        samples = boot[block+'_samples']
        if samples.shape != (args.n_bootstrap, len(target)) or not np.isfinite(samples).all():
            raise FloatingPointError(f'Incomplete or nonfinite {block} bootstrap draws')
        lower, upper = boot[block+'_lb'], boot[block+'_ub']
        bootstrap_mean = boot[block+'_mean'] if block+'_mean' in boot else samples.mean(axis=0)
        row.update({block+'_bias': estimate-target, block+'_se': boot[block+'_se'],
                    block+'_lb': lower, block+'_ub': upper,
                    block+'_cov': ((lower <= target) & (target <= upper)).astype(int),
                    block+'_bootstrap_mean': bootstrap_mean,
                    block+'_bootstrap_bias': bootstrap_mean-estimate})
    row['bootstrap_workers'] = boot['n_workers']
    if mname == 'NeuralPLSI':
        row['fit_diagnostics'] = model.fit_diagnostics
        row['device'] = str(model.device)
    else:
        row['fit_diagnostics'] = dict(converged=bool(model.converged_), iterations=model.n_iter_,
                                      profile_loss=model.loss_)
    if args.outcome == 'cox':
        row['event_fraction'] = float(y_tr[:, 1].mean())
    if args.save_bootstrap_samples:
        sample_dir = Path(args.out or 'output/simulation')/'bootstrap_samples'
        sample_dir.mkdir(parents=True, exist_ok=True)
        tag = hashlib.sha256(json.dumps(configuration, sort_keys=True).encode()).hexdigest()[:12]
        path = sample_dir/f'{mname}_{n}_{args.g_fn}_{args.outcome}_{args.exposure_dist}_{tag}_{seed}.npz'
        with tempfile.NamedTemporaryFile(dir=sample_dir, delete=False) as stream:
            temporary = stream.name
            np.savez_compressed(stream, beta_samples=boot['beta_samples'],
                                gamma_samples=boot['gamma_samples'])
        os.replace(temporary, path)
        row['bootstrap_samples_file'] = str(path)
    print(f'{mname} seed={seed} performance={performance:.4f} fit={time_fit:.2f}s bootstrap={time_boot:.2f}s', flush=True)
    return to_json(row)


def build_parser():
    parser = argparse.ArgumentParser(description='PLSI simulation with reproducible bootstrap inference.')
    parser.add_argument('--n_instances', type=int, default=500)
    parser.add_argument('--n_replicates', type=int, default=500)
    parser.add_argument('--n_bootstrap', type=int, default=100)
    parser.add_argument('--nn_bootstrap_quantile', choices=['linear','weibull'], default='linear')
    parser.add_argument('--bootstrap_jobs', default='auto',
                        type=lambda value: value if value == 'auto' else int(value))
    parser.add_argument('--g_fn', default='sigmoid', choices=['linear','sfun','sigmoid'])
    parser.add_argument('--outcome', default='continuous', choices=['continuous','binary','cox'])
    parser.add_argument('--exposure_dist', default='normal', choices=['normal','uniform','t'])
    parser.add_argument('--models', nargs='+', default=['all'])
    parser.add_argument('--seed0', type=int, default=0)
    parser.add_argument('--fit_seed', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=1)
    parser.add_argument('--save_bootstrap_samples', action='store_true')
    parser.add_argument('--out', default=None)
    parser.add_argument('--g_grid_min', type=float, default=-3.)
    parser.add_argument('--g_grid_max', type=float, default=3.)
    parser.add_argument('--g_grid_n', type=int, default=1000)
    parser.add_argument('--activation', default='ELU')
    parser.add_argument('--initial', action='store_true')
    parser.add_argument('--device', default='cpu', choices=['auto','cpu'])
    parser.add_argument('--nn_backend', default='cpp', choices=['cpp'])
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if (args.n_instances < 5 or args.n_replicates < 1 or args.n_bootstrap < 2 or
            args.save_every < 1 or args.g_grid_n < 2 or args.g_grid_max <= args.g_grid_min or
            args.seed0 < 0 or args.fit_seed < 0):
        parser.error('Invalid sample, replicate, bootstrap, checkpoint, grid, or seed setting')
    bootstrap_jobs(args.bootstrap_jobs, args.n_bootstrap)
    for mname in parse_models(args.models):
        path = output_path(args.out, mname, args.n_instances, args.g_fn,
                           args.outcome, args.exposure_dist, args.initial if mname == 'NeuralPLSI' else False)
        with result_lock(path):
            configuration = run_configuration(args, mname)
            records = load_compatible_results(path, configuration)
            completed = {row['seed'] for row in records}
            try:
                for seed in range(args.seed0, args.seed0+args.n_replicates):
                    if seed in completed:
                        continue
                    try:
                        records.append(run_replicate(args, mname, seed, configuration))
                    except Exception as exc:
                        # Fail the scheduler job instead of reporting coverage conditional
                        # on successful fits. Earlier completed replicates are preserved.
                        raise RuntimeError(f'{mname} simulation failed at seed {seed}') from exc
                    if len(records) % args.save_every == 0:
                        atomic_json(path, records)
            finally:
                atomic_json(path, records)
            print(f'Saved {len(records)} replicates to {path}', flush=True)


if __name__ == '__main__':
    main()
