import ctypes

import numpy as np
from scipy.special import expit

from .base import _SummaryMixin, covariates, validate_data
from .native import (BOOTSTRAP_RNG_PROTOCOL, FAMILIES, bootstrap_options, library, seed_value,
                     spline_checked)


def _contiguous(*arrays):
    return [np.ascontiguousarray(a, dtype=np.float64) for a in arrays]


class SplinePLSI(_SummaryMixin):
    """Partial linear single-index model g(X @ beta) + Z @ gamma with a B-spline g.

    Fitting and the bootstrap run in C++ in double precision. The index is
    profiled over the unit sphere: for each beta, num_knots knots sit at equally
    spaced order statistics of X @ beta (Wang et al., 2020), clamped at the
    boundary knots, and gamma and the spline coefficients are the exact
    ridge-penalized fit on [Z, B(X @ beta)]: RSS + alpha*|w|^2 for continuous
    outcomes, negative log-likelihood + alpha*|w|^2/2 for binary and Breslow Cox
    outcomes. beta is optimized by BFGS over the sphere, using central finite
    differences, from a seeded random direction, followed by a compass search
    on the profile loss; beta[0] >= 0 identifies its sign. max_iter bounds the
    BFGS iterations; BFGS stops when a step improves the profile loss by less
    than tol. The B-spline basis sums to one,
    so g includes the intercept; a Cox likelihood cannot identify that level,
    so g is shifted to g(0) = 0 for Cox outcomes, as in NeuralPLSI. Outside the
    boundary knots, g is constant.
    """
    def __init__(self, family='continuous', num_knots=5, spline_degree=3, alpha=1e-6, max_iter=200, tol=1e-6):
        self.family = family
        self.num_knots = num_knots
        self.spline_degree = spline_degree
        self.alpha = alpha
        self.max_iter = max_iter
        self.tol = tol

        self.beta = None
        self.gamma = None
        self.spline_coeffs = None
        self.knot_vector = None

    def _native_config(self, p, q):
        if self.max_iter < 1 or self.alpha < 0 or self.num_knots < 2 or self.spline_degree < 0:
            raise ValueError('Invalid iteration count, penalty, or spline dimensions')
        config = np.array([p, q, FAMILIES[self.family], self.num_knots, self.spline_degree,
                           self.max_iter], dtype=np.int32)
        return config, np.array([self.alpha, self.tol], dtype=np.float64)

    def _outputs(self, p, q):
        n_basis = self.num_knots+self.spline_degree-1
        return (np.empty(p), np.empty(q), np.empty(n_basis),
                np.empty(self.num_knots+2*self.spline_degree))

    def fit(self, X, Z, y, beta_init=None, random_state=None):
        """Fit the profiled index, spline and linear component.

        random_state seeds the random starting direction. None draws the seed
        from NumPy's global generator, for callers that seed it themselves.
        """
        X, Z, y = _contiguous(*validate_data(X, Z, y, self.family))
        self._clear_inference()
        config, hyper = self._native_config(X.shape[1], Z.shape[1])
        if random_state is None:
            random_state = int(np.random.randint(0, 2**31))
        start = None
        if beta_init is not None:
            start = np.ascontiguousarray(beta_init, dtype=np.float64)
            if start.shape != (X.shape[1],) or not np.isfinite(start).all() or not np.any(start):
                raise ValueError('beta_init must be a finite nonzero exposure direction')
        beta, gamma, spline, knots = self._outputs(X.shape[1], Z.shape[1])
        diagnostics = np.empty(3)
        spline_checked(library().splsi_fit(config, hyper, X, Z, y, len(X), seed_value(random_state),
            None if start is None else start.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            beta, gamma, spline, knots, diagnostics))
        self.beta, self.gamma, self.spline_coeffs, self.knot_vector = beta, gamma, spline, knots
        self.loss_, self.n_iter_, self.converged_ = float(diagnostics[0]), int(diagnostics[1]), bool(diagnostics[2])
        return self

    def profile(self, X, Z, y, beta):
        """Return (gamma, spline coefficients, knots, profile loss) at a fixed unit beta."""
        X, Z, y = _contiguous(*validate_data(X, Z, y, self.family))
        config, hyper = self._native_config(X.shape[1], Z.shape[1])
        beta = np.ascontiguousarray(beta, dtype=np.float64)
        if beta.shape != (X.shape[1],):
            raise ValueError('beta has the wrong length')
        _, gamma, spline, knots = self._outputs(X.shape[1], Z.shape[1])
        loss = np.empty(1)
        spline_checked(library().splsi_profile(config, hyper, X, Z, y, len(X), beta,
                                               gamma, spline, knots, loss))
        return gamma, spline, knots, float(loss[0])

    def _ensure_fitted(self):
        if self.beta is None or self.gamma is None or self.spline_coeffs is None:
            raise RuntimeError("Call fit() before predict().")

    def basis(self, x):
        """B-spline basis of the fitted knots at x, clipped to the boundary knots."""
        self._ensure_fitted()
        x = np.ascontiguousarray(np.asarray(x, dtype=np.float64).reshape(-1))
        result = np.empty((len(x), len(self.spline_coeffs)))
        spline_checked(library().splsi_basis(self.knot_vector, len(self.knot_vector),
                                             self.spline_degree, x, len(x), result))
        return result

    def predict(self, X, Z):
        """Predict raw output (linear predictor). Z=None means no covariates."""
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64)
        if X.ndim != 2 or X.shape[1] != len(self.beta):
            raise ValueError('X must match the fitted exposure dimensions')
        Z = covariates(Z, X)
        if Z.shape != (len(X), len(self.gamma)):
            raise ValueError(f'Z must have shape ({len(X)}, {len(self.gamma)}) to match the '
                             'fitted covariates (None when there are none)')
        return Z @ self.gamma + self.basis(X @ self.beta) @ self.spline_coeffs

    def predict_proba(self, X, Z):
        """Predict probabilities (for binary outcome)."""
        return expit(self.predict(X, Z))

    def predict_partial_hazard(self, X, Z):
        """Predict partial hazard exp(g(Xb) + Zg) (for Cox outcome)."""
        return np.exp(self.predict(X, Z))

    def g_function(self, x):
        """Estimate g(x) for given scalar inputs x."""
        return self.basis(x) @ self.spline_coeffs

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs=-1):
        """Pairs bootstrap in C++ with native threads.

        Draws use the same resampled rows as NeuralPLSI.inference_bootstrap for
        the same random_state, and every refit starts from its own seeded random
        direction. Percentile intervals use linear interpolation.
        """
        jobs, method = bootstrap_options(n_samples, random_state, ci, n_jobs, 'linear')
        X, Z, y = _contiguous(*validate_data(X, Z, y, self.family))
        if g_grid is not None:
            g_grid = np.ascontiguousarray(np.asarray(g_grid, dtype=np.float64).reshape(-1))
            if not g_grid.size or not np.isfinite(g_grid).all():
                raise ValueError('g_grid must be nonempty and finite')
        grid = g_grid if g_grid is not None else np.empty(0)
        if self.beta is None:
            self.fit(X, Z, y, random_state=random_state)
        if len(self.beta) != X.shape[1] or len(self.gamma) != Z.shape[1]:
            raise ValueError('Bootstrap dimensions differ from the fitted model')
        self._clear_inference()
        config, hyper = self._native_config(X.shape[1], Z.shape[1])
        p, q, s = len(self.beta), len(self.gamma), len(self.spline_coeffs)
        columns = p+q+s+len(grid)
        draws, summary = np.empty((n_samples, columns)), np.empty((4, columns))
        workers = np.empty(1, dtype=np.int32)
        spline_checked(library().splsi_bootstrap(config, hyper, X, Z, y, len(X), n_samples,
            seed_value(random_state), jobs, grid, len(grid), float(ci), method, draws, summary, workers))
        blocks = {'beta': slice(0, p), 'gamma': slice(p, p+q), 'spline': slice(p+q, p+q+s)}
        if len(grid):
            blocks['g'] = slice(p+q+s, columns)
        out = dict(beta_hat=self.beta, gamma_hat=self.gamma, spline_hat=self.spline_coeffs,
                   quantile_method='linear', ci=ci, n_bootstrap=n_samples, n_workers=int(workers[0]),
                   bootstrap_engine='cpp', bootstrap_rng=BOOTSTRAP_RNG_PROTOCOL)
        for block, span in blocks.items():
            out[block+'_samples'] = draws[:, span]
            out[block+'_mean'], out[block+'_se'] = summary[0, span], summary[1, span]
            out[block+'_lb'], out[block+'_ub'] = summary[2, span], summary[3, span]
        for block in ('beta', 'gamma', 'spline'):
            for key in ('se', 'lb', 'ub'):
                setattr(self, f'{block}_{key}', out[f'{block}_{key}'])
        if len(grid):
            out['g_grid'] = grid
            self.g_grid, self.g_grid_mean, self.g_grid_se = grid, out['g_mean'], out['g_se']
            self.g_grid_lb, self.g_grid_ub = out['g_lb'], out['g_ub']
            self._g_samples = out['g_samples']
        return out

    def summary(self, include_beta=True, include_gamma=True, include_spline=False):
        if self.beta is None:
            raise ValueError("Model has not been fitted yet.")
        blocks = {}
        if include_beta:
            blocks['beta'] = dict(coeff=self.beta, se=getattr(self, 'beta_se', None),
                                  lb=getattr(self, 'beta_lb', None), ub=getattr(self, 'beta_ub', None))
        if include_gamma:
            blocks['gamma'] = dict(coeff=self.gamma, se=getattr(self, 'gamma_se', None),
                                   lb=getattr(self, 'gamma_lb', None), ub=getattr(self, 'gamma_ub', None))
        if include_spline:
            blocks['spline'] = dict(coeff=self.spline_coeffs, se=getattr(self, 'spline_se', None),
                                    lb=getattr(self, 'spline_lb', None), ub=getattr(self, 'spline_ub', None))
        return self._build_summary(blocks, ['beta', 'gamma', 'spline'])
