"""Read simulation outputs without silently pooling incompatible runs."""
import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from simulation import beta as TRUE_BETA, gamma as TRUE_GAMMA, simulate_data

SCENARIO = ('model', 'n', 'g_fn', 'outcome', 'x_dist')


def load_results(data_dir='output/simulation'):
    files = sorted(Path(data_dir).glob('simulation+*.json'))
    if not files:
        raise FileNotFoundError(f'No simulation+*.json files in {data_dir}')
    records, seen = [], set()
    for path in files:
        raw = json.loads(path.read_text())
        if not raw:
            continue  # A failed job may have an empty checkpoint.
        if isinstance(raw, list):
            first = raw[0]
            config = first.get('run_configuration')
            identity = tuple(first.get(k, 'normal' if k == 'x_dist' else '?') for k in SCENARIO)
            seeds = set()
            for row in raw:
                if tuple(row.get(k, 'normal' if k == 'x_dist' else '?') for k in SCENARIO) != identity:
                    raise ValueError(f'Mixed scenarios in {path}')
                if row.get('run_configuration') != config:
                    raise ValueError(f'Mixed run configurations in {path}')
                if row['seed'] in seeds:
                    raise ValueError(f'Duplicate seed in {path}: {row["seed"]}')
                seeds.add(row['seed'])
            keys = set().union(*(row.keys() for row in raw))
            rec = {key: [row.get(key) for row in raw] for key in keys}
        elif isinstance(raw, dict):
            rec, config = raw.copy(), None
        else:
            raise ValueError(f'Unsupported result format in {path}')
        for key in SCENARIO:
            value = rec.get(key, ['normal' if key == 'x_dist' else '?'])
            rec['_'+key] = str(value[0] if isinstance(value, list) else value)
        identity = tuple(rec['_'+key] for key in SCENARIO)
        if identity in seen:
            raise ValueError(f'Multiple runs for {identity}. Put each backend/initialization/protocol in a separate results directory.')
        seen.add(identity)
        rec['_file'], rec['_configuration'] = str(path), config
        if config is None:
            warnings.warn(f'{path.name}: legacy results have no verifiable run configuration', stacklevel=2)
        records.append(rec)
    return records


def true_g(x, g_fn):
    *_, function = simulate_data(5, outcome='continuous', g_type=g_fn, seed=0)
    return np.asarray(function(np.asarray(x)))


def curves(rec):
    pairs = [(i, g) for i, g in enumerate(rec.get('g_pred', [])) if g is not None and len(g)]
    if not pairs:
        return None, None
    values = np.array([g for _, g in pairs], dtype=float)
    grids = rec.get('g_grid', [])
    if grids and all(grids[i] is not None for i, _ in pairs):
        grid = np.asarray(grids[pairs[0][0]], dtype=float)
        if any(not np.array_equal(grid, grids[i]) for i, _ in pairs):
            raise ValueError(f'Inconsistent g grids in {rec["_file"]}')
    elif values.shape[1] == 1000:
        grid = np.linspace(-3, 3, 1000)  # Historical runner's documented grid.
    else:
        raise ValueError(f'Missing g grid in {rec["_file"]}')
    if grid.ndim != 1 or len(grid) != values.shape[1] or np.any(np.diff(grid) <= 0):
        raise ValueError(f'Invalid g grid in {rec["_file"]}')
    if not np.isfinite(values).all() or not np.isfinite(grid).all():
        raise ValueError(f'Nonfinite g values in {rec["_file"]}')
    if rec['_outcome'] == 'cox':
        if not grid[0] <= 0 <= grid[-1]:
            raise ValueError('Cox curve comparison needs a grid containing zero')
        values = values-np.array([np.interp(0, grid, row) for row in values])[:, None]
    return grid, values


def _matrix(values, width):
    valid = [value for value in values if value is not None]
    if not valid:
        return np.empty((0, width))
    result = np.asarray(valid, dtype=float)
    if result.ndim != 2 or result.shape[1] != width or not np.isfinite(result).all():
        raise ValueError('Invalid coefficient or uncertainty array in simulation results')
    return result


def compute_summaries(records):
    rows = []
    for rec in records:
        common = {k: rec['_'+k] for k in SCENARIO}
        config = rec['_configuration'] or {}
        common.update(backend=config.get('backend', 'legacy'),
                      quantile_method=config.get('quantile_method', 'unknown'),
                      n_bootstrap=config.get('n_bootstrap', np.nan),
                      source_sha256=config.get('source_sha256', 'unverified'))
        direct = any(v is not None for v in rec.get('beta_se', []))
        methods = [('Bootstrap', None)] if direct else []
        if not direct:
            for label, key in [('Bootstrap', 'bootstrap_summary'), ('Hessian', 'hessian_summary')]:
                summaries = rec.get(key, [])
                if any(s for s in summaries):
                    methods.append((label, summaries))
            if not methods and any(s for s in rec.get('inference_summary', [])):
                # The old generic key does not identify the inference method.
                methods.append(('Legacy', rec['inference_summary']))
        if not methods:
            methods = [('Unavailable', [])]
        for method, summaries in methods:
            for block, truth in [('beta', TRUE_BETA), ('gamma', TRUE_GAMMA)]:
                estimates = _matrix(rec.get(block+'_est', rec.get(block+'_estimate', [])), len(truth))
                if not len(estimates):
                    continue
                def field(suffix):
                    return (rec.get(block+'_'+suffix, []) if summaries is None else
                            [(s or {}).get(block+'_'+suffix) for s in summaries])
                standard_errors = _matrix(field('se'), len(truth))
                bounds = [(a, b) for a, b in zip(field('lb'), field('ub')) if a is not None and b is not None]
                if bounds:
                    lower, upper = np.asarray([a for a, _ in bounds]), np.asarray([b for _, b in bounds])
                    if not np.isfinite(lower).all() or not np.isfinite(upper).all() or np.any(lower > upper):
                        raise ValueError('Invalid confidence interval endpoints')
                    coverage = (lower <= truth) & (truth <= upper)
                else:
                    coverage = _matrix(field('cov'), len(truth))
                sd = estimates.std(axis=0, ddof=1) if len(estimates)>1 else np.full(len(truth), np.nan)
                se = standard_errors.mean(axis=0) if len(standard_errors) else np.full(len(truth), np.nan)
                cov = coverage.mean(axis=0) if len(coverage) else np.full(len(truth), np.nan)
                boot_bias = _matrix(rec.get(block+'_bootstrap_bias', []), len(truth))
                for j in range(len(truth)):
                    rows.append(dict(common, param=f'{block}_{j}', true_val=truth[j],
                        bias=estimates[:, j].mean()-truth[j], emp_sd=sd[j], se=se[j], coverage=cov[j],
                        bias_mcse=sd[j]/np.sqrt(len(estimates)),
                        coverage_mcse=np.sqrt(cov[j]*(1-cov[j])/len(coverage)) if len(coverage) else np.nan,
                        bootstrap_bias=boot_bias[:, j].mean() if len(boot_bias) else np.nan,
                        n_reps=len(estimates), n_intervals=len(coverage), n_se=len(standard_errors), inference_type=method))
    return pd.DataFrame(rows)
