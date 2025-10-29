# ==========================================================
# Base classes and utilities shared across PLSI models
# ==========================================================
import numpy as np
import pandas as pd

# Try to import joblib for parallel bootstrap, fall back to sequential
try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


class _SummaryMixin:
    """Mixin class providing summary table and CI calculation methods."""

    def _build_summary(self, blocks, prefix_order):
        """Build a summary DataFrame from coefficient blocks.

        Args:
            blocks: dict mapping block names to dicts with 'coeff', 'se', 'lb', 'ub'
            prefix_order: list of block names in display order

        Returns:
            pandas DataFrame with parameter estimates and confidence intervals
        """
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

    def _percentile_ci(self, samples, ci=0.95):
        """Calculate percentile-based confidence intervals.

        Args:
            samples: array of bootstrap samples (n_samples, n_params)
            ci: confidence level (default 0.95)

        Returns:
            tuple of (lower_bounds, upper_bounds)
        """
        lo = (1 - ci) / 2 * 100
        hi = (1 + ci) / 2 * 100
        return np.percentile(samples, [lo, hi], axis=0)


def draw_bootstrap_indices(N, rng, cluster_ids=None):
    """Draw bootstrap sample indices with optional cluster resampling.

    Args:
        N: sample size
        rng: numpy random number generator
        cluster_ids: optional array of cluster IDs for clustered bootstrap

    Returns:
        array of bootstrap sample indices
    """
    if cluster_ids is None:
        return rng.integers(0, N, size=N)
    cluster_ids = np.asarray(cluster_ids)
    unique_c = np.unique(cluster_ids)
    c_sample = rng.choice(unique_c, size=len(unique_c), replace=True)
    idx = np.concatenate([np.where(cluster_ids == c)[0] for c in c_sample])
    if idx.size == 0:
        idx = rng.integers(0, N, size=N)
    return idx


def run_parallel_bootstrap(refit_func, X, Z, y, n_samples, random_state, cluster_ids=None, n_jobs=-1):
    """Run bootstrap resampling in parallel using joblib.

    Args:
        refit_func: function that takes (Xb, Zb, yb, seed) and returns (beta, gamma, ...)
        X, Z, y: data arrays
        n_samples: number of bootstrap samples
        random_state: random seed
        cluster_ids: optional cluster IDs for clustered bootstrap
        n_jobs: number of parallel jobs (-1 uses all cores)

    Returns:
        list of results from refit_func calls

    Note:
        For CPU-based PyTorch models, each worker should manage its own thread pool.
        Workers use 'loky' backend which creates separate processes.
    """
    if not _HAVE_JOBLIB:
        # Fall back to sequential processing
        results = []
        N = len(X)
        rng = np.random.default_rng(random_state)
        for b in range(n_samples):
            idx = draw_bootstrap_indices(N, rng, cluster_ids)
            Xb, Zb, yb = X[idx], Z[idx], y[idx]
            results.append(refit_func(Xb, Zb, yb, random_state + 1337 + b))
        return results

    # Parallel processing with joblib
    N = len(X)

    def bootstrap_iteration(b, random_state_base):
        rng = np.random.default_rng(random_state_base + b)
        idx = draw_bootstrap_indices(N, rng, cluster_ids)
        Xb, Zb, yb = X[idx], Z[idx], y[idx]
        return refit_func(Xb, Zb, yb, random_state_base + 1337 + b)

    # Use 'loky' backend for better CPU isolation (each process gets own resources)
    # Use 'processes' for better performance with CPU-bound tasks
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(bootstrap_iteration)(b, random_state) for b in range(n_samples)
    )
    return results
