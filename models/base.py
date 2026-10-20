import numpy as np
import pandas as pd
import os
from numbers import Integral

try:
    from joblib import Parallel, delayed, cpu_count, parallel_config
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


class _SummaryMixin:
    def _clear_inference(self):
        """A new fit invalidates uncertainty estimates from a previous fit."""
        for name in list(vars(self)):
            if (name.endswith(('_se', '_lb', '_ub')) or
                    name.startswith('g_grid') or name == '_g_samples'):
                delattr(self, name)

    def _build_summary(self, blocks, prefix_order):
        names, coeffs, ses, lbs, ubs = [], [], [], [], []
        for name in prefix_order:
            blk = blocks.get(name)
            if not blk:
                continue
            c = np.asarray(blk.get('coeff', []), dtype=float)
            if c.size == 0:
                continue
            se = blk.get('se')
            lb = blk.get('lb')
            ub = blk.get('ub')
            if se is None: se = np.full_like(c, np.nan, dtype=float)
            if lb is None: lb = np.full_like(c, np.nan, dtype=float)
            if ub is None: ub = np.full_like(c, np.nan, dtype=float)

            if name == 'beta':
                names.extend([f'beta_{i:02d}' for i in range(c.size)])
            elif name == 'gamma':
                names.extend([f'gamma_{i:02d}' for i in range(c.size)])
            elif name == 'spline':
                names.extend([f'spline_{i:02d}' for i in range(c.size)])
            else:
                names.extend([f'{name}_{i:02d}' for i in range(c.size)])
            coeffs.append(c)
            ses.append(np.asarray(se))
            lbs.append(np.asarray(lb))
            ubs.append(np.asarray(ub))

        if not coeffs:
            return pd.DataFrame(columns=['Parameter', 'Coefficient', 'SE (bootstrap)', 'CI Lower', 'CI Upper'])
        coeffs = np.concatenate(coeffs)
        ses = np.concatenate(ses)
        lbs = np.concatenate(lbs)
        ubs = np.concatenate(ubs)

        return pd.DataFrame({
            'Parameter': names,
            'Coefficient': coeffs,
            'SE (bootstrap)': ses,
            'CI Lower': lbs,
            'CI Upper': ubs
        })

    def _percentile_ci(self, samples, ci=0.95, method='linear'):
        lo = (1 - ci) / 2 * 100
        hi = (1 + ci) / 2 * 100
        return np.percentile(samples, [lo, hi], axis=0, method=method)


def validate_data(X, Z, y, family, dtype=np.float64):
    if family not in ('continuous', 'binary', 'cox'):
        raise ValueError("family must be 'continuous', 'binary', or 'cox'")
    X, Z, y = [np.asarray(a, dtype=dtype) for a in (X, Z, y)]
    if X.ndim != 2 or Z.ndim != 2 or X.shape[1] < 1:
        raise ValueError('X and Z must be matrices, with at least one exposure')
    if len(X) < 2 or len(Z) != len(X) or y.ndim == 0 or len(y) != len(X):
        raise ValueError('X, Z, and y must have the same number of rows (at least two)')
    if not all(np.isfinite(a).all() for a in (X, Z, y)):
        raise ValueError('X, Z, and y must contain only finite values')
    if family == 'cox':
        if y.shape != (len(X), 2) or np.any(y[:, 0] < 0):
            raise ValueError('Cox y must contain nonnegative durations and event indicators')
        if not np.isin(y[:, 1], [0, 1]).all() or y[:, 1].sum() == 0:
            raise ValueError('Cox outcomes need binary event indicators and at least one event')
    else:
        if y.shape not in ((len(X),), (len(X), 1)):
            raise ValueError('y must be a vector or a single column')
        y = y.reshape(-1)
        if family == 'binary' and not np.isin(y, [0, 1]).all():
            raise ValueError('Binary outcomes must be coded 0 or 1')
    return X, Z, y


def validate_bootstrap(n_samples, ci, random_state):
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or n_samples < 2:
        raise ValueError('n_samples must be an integer of at least two')
    if not 0 < ci < 1:
        raise ValueError('ci must lie strictly between zero and one')
    if isinstance(random_state, bool) or not isinstance(random_state, Integral) or random_state < 0:
        raise ValueError('random_state must be a nonnegative integer')


def bootstrap_jobs(n_jobs, n_samples):
    """Respect both process affinity/cgroup limits and the SLURM allocation."""
    if n_jobs == 'auto':
        n_jobs = -1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or n_jobs == 0:
        raise ValueError('n_jobs must be a nonzero integer or auto')
    available = cpu_count() if _HAVE_JOBLIB else 1
    if os.environ.get('SLURM_CPUS_PER_TASK'):
        available = min(available, max(1, int(os.environ['SLURM_CPUS_PER_TASK'])))
    requested = n_jobs if n_jobs > 0 else max(1, available+1+n_jobs)
    return min(requested, available, n_samples)


def draw_bootstrap_indices(N, rng):
    return rng.integers(0, N, size=N)


def run_parallel_bootstrap(refit_func, X, Z, y, n_samples, random_state, n_jobs=-1):
    N = len(X)
    n_jobs = bootstrap_jobs(n_jobs, n_samples)

    def bootstrap_iteration(b, random_state_base):
        rng = np.random.default_rng(random_state_base + b)
        idx = draw_bootstrap_indices(N, rng)
        Xb, Zb, yb = X[idx], Z[idx], y[idx]
        return refit_func(Xb, Zb, yb, random_state_base + 1337 + b)

    if not _HAVE_JOBLIB or n_jobs == 1:
        results = []
        for b in range(n_samples):
            results.append(bootstrap_iteration(b, random_state))
        return results

    with parallel_config(backend='loky', inner_max_num_threads=1):
        results = Parallel(n_jobs=n_jobs, verbose=0)(
            delayed(bootstrap_iteration)(b, random_state) for b in range(n_samples)
        )
    return results
