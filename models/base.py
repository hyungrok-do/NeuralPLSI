import numpy as np
import pandas as pd

try:
    from joblib import Parallel, delayed
    _HAVE_JOBLIB = True
except ImportError:
    _HAVE_JOBLIB = False


class _SummaryMixin:
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

    def _percentile_ci(self, samples, ci=0.95):
        lo = (1 - ci) / 2 * 100
        hi = (1 + ci) / 2 * 100
        return np.percentile(samples, [lo, hi], axis=0)


def draw_bootstrap_indices(N, rng):
    return rng.integers(0, N, size=N)


def run_parallel_bootstrap(refit_func, X, Z, y, n_samples, random_state, n_jobs=-1):
    if not _HAVE_JOBLIB:
        results = []
        N = len(X)
        rng = np.random.default_rng(random_state)
        for b in range(n_samples):
            idx = draw_bootstrap_indices(N, rng)
            Xb, Zb, yb = X[idx], Z[idx], y[idx]
            results.append(refit_func(Xb, Zb, yb, random_state + 1337 + b))
        return results

    N = len(X)

    def bootstrap_iteration(b, random_state_base):
        rng = np.random.default_rng(random_state_base + b)
        idx = draw_bootstrap_indices(N, rng)
        Xb, Zb, yb = X[idx], Z[idx], y[idx]
        return refit_func(Xb, Zb, yb, random_state_base + 1337 + b)

    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(bootstrap_iteration)(b, random_state) for b in range(n_samples)
    )
    return results
