# ==========================================================
# SplinePLSI (NumPy/Scikit/Lifelines) — unified API
# ==========================================================
import numpy as np
import pandas as pd
import scipy.optimize as opt
from sklearn.linear_model import Ridge, LogisticRegression
from lifelines import CoxPHFitter

from .base import _SummaryMixin, draw_bootstrap_indices, run_parallel_bootstrap

# Optional Numba for spline basis acceleration
try:
    from numba import njit
    _HAVE_NUMBA = True
except Exception:
    _HAVE_NUMBA = False


class SplinePLSI(_SummaryMixin):
    def __init__(self, family='continuous', num_knots=5, spline_degree=3, alpha=1.0, max_iter=50, tol=1e-6):
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

    # ---------- Static helpers (Numba-JIT'able) ----------
    @staticmethod
    def _make_knot_vector(eta_min, eta_max, num_knots, degree):
        knots = np.linspace(eta_min, eta_max, num_knots)
        t = np.empty(num_knots + 2 * degree, dtype=np.float64)
        for i in range(degree):
            t[i] = knots[0]
        t[degree:degree + num_knots] = knots
        for i in range(degree):
            t[degree + num_knots + i] = knots[-1]
        return t

    @staticmethod
    def _n_basis(t, degree):
        return t.size - degree - 1

    @staticmethod
    def _bspline_basis_matrix(eta, t, degree):
        eta = eta.astype(np.float64)
        t = t.astype(np.float64)
        n = eta.size
        k = degree
        m = t.size
        nb = m - k - 1
        B = np.zeros((n, nb), dtype=np.float64)
        # degree 0
        for r in range(n):
            x = eta[r]
            for i in range(nb):
                if (x >= t[i] and x < t[i + 1]) or (i == nb - 1 and x == t[i + 1]):
                    B[r, i] = 1.0
        # elevate degree
        for d in range(1, k + 1):
            tmp = np.zeros((n, nb), dtype=np.float64)
            for i in range(nb):
                denom1 = t[i + d] - t[i]
                denom2 = t[i + d + 1] - t[i + 1]
                for r in range(n):
                    x = eta[r]
                    left = 0.0
                    right = 0.0
                    if denom1 > 0.0:
                        left = (x - t[i]) / denom1 * B[r, i]
                    if (i + 1) < nb and denom2 > 0.0:
                        right = (t[i + d + 1] - x) / denom2 * B[r, i + 1]
                    tmp[r, i] = left + right
            B = tmp
        return B

    @staticmethod
    def _sigmoid(u):
        out = np.empty_like(u, dtype=np.float64)
        for i in range(u.size):
            x = u[i]
            if x >= 0:
                z = np.exp(-x)
                out[i] = 1.0 / (1.0 + z)
            else:
                z = np.exp(x)
                out[i] = z / (1.0 + z)
        return out

    @staticmethod
    def _linear_predict(X, Z, beta, gamma, spline_coeffs, t, degree):
        eta = X @ beta
        B = SplinePLSI._bspline_basis_matrix(eta, t, degree)
        return Z @ gamma + B @ spline_coeffs

    # ---------- Internals ----------
    def _initialize_params(self, X):
        beta_init = np.random.randn(X.shape[1])
        if beta_init[0] < 0: beta_init = -beta_init
        nrm = np.linalg.norm(beta_init) or 1.0
        return beta_init / nrm

    def _construct_spline_basis(self, eta):
        t = self._make_knot_vector(eta.min(), eta.max(), self.num_knots, self.spline_degree)
        B = self._bspline_basis_matrix(eta, t, self.spline_degree)
        return B, t

    def _objective_factory(self, X, Z, y):
        family = self.family; alpha = self.alpha
        spline_degree = self.spline_degree; num_knots = self.num_knots

        def objective(beta):
            beta = np.asarray(beta, dtype=np.float64)
            eta = X @ beta
            t = SplinePLSI._make_knot_vector(eta.min(), eta.max(), num_knots, spline_degree)
            B = SplinePLSI._bspline_basis_matrix(eta, t, spline_degree)
            Xd = np.hstack((Z, B))

            if family == 'continuous':
                model = Ridge(alpha=alpha, fit_intercept=False).fit(Xd, y)
                pred = model.predict(Xd)
                return float(np.sum((y - pred) ** 2))
            elif family == 'binary':
                model = LogisticRegression(penalty='l2', C=1.0/alpha, fit_intercept=False,
                                           solver='lbfgs', max_iter=200).fit(Xd, y)
                p = model.predict_proba(Xd)[:, 1]
                eps = 1e-9
                return float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
            elif family == 'cox':
                df = pd.DataFrame(Xd, columns=[f'z{i}' for i in range(Z.shape[1])] +
                                           [f'b{i}' for i in range(B.shape[1])])
                df['T'], df['E'] = y[:, 0], y[:, 1]
                # Increase step_size for better convergence, use higher penalizer for stability
                cph = CoxPHFitter(penalizer=alpha, l1_ratio=0.0)
                try:
                    cph.fit(df, duration_col='T', event_col='E', show_progress=False, step_size=0.5)
                except Exception:
                    # If fitting fails, try with stronger penalization
                    cph = CoxPHFitter(penalizer=alpha * 10, l1_ratio=0.0)
                    cph.fit(df, duration_col='T', event_col='E', show_progress=False, step_size=0.5)
                return float(-cph.log_likelihood_)
            else:
                raise ValueError("Unsupported family")
        return objective

    def _optimize_beta(self, X, Z, y, beta_init):
        objective = self._objective_factory(X, Z, y)
        constraints = [{'type': 'eq', 'fun': lambda b: np.linalg.norm(b) - 1}]
        result = opt.minimize(objective, beta_init, method='SLSQP', constraints=constraints,
                              options={'maxiter': self.max_iter})
        beta = result.x
        if beta[0] < 0: beta = -beta
        nrm = np.linalg.norm(beta) or 1.0
        return beta / nrm

    # ---------- Public API ----------
    def fit(self, X, Z, y):
        X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
        self.beta = self._initialize_params(X)
        prev_loss = np.inf

        for _ in range(self.max_iter):
            eta = X @ self.beta
            B, t = self._construct_spline_basis(eta)
            Xd = np.hstack((Z, B))

            if self.family == 'continuous':
                model = Ridge(alpha=self.alpha, fit_intercept=False).fit(Xd, y)
                resid = y - model.predict(Xd)
                loss = float(np.sum(resid ** 2))
                coef = np.ravel(model.coef_)
                self.gamma = coef[:Z.shape[1]]
                self.spline_coeffs = coef[Z.shape[1]:]

            elif self.family == 'binary':
                model = LogisticRegression(penalty='l2', C=1.0/self.alpha, fit_intercept=False,
                                           solver='lbfgs', max_iter=500).fit(Xd, y)
                p = model.predict_proba(Xd)[:, 1]; eps = 1e-9
                loss = float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
                coef = model.coef_[0]
                self.gamma = coef[:Z.shape[1]]
                self.spline_coeffs = coef[Z.shape[1]:]

            elif self.family == 'cox':
                df = pd.DataFrame(Xd, columns=[f'z{i}' for i in range(Z.shape[1])] +
                                           [f'b{i}' for i in range(B.shape[1])])
                df['T'], df['E'] = y[:, 0], y[:, 1]
                # Use step_size for better convergence, pure L2 penalty
                cph = CoxPHFitter(penalizer=self.alpha, l1_ratio=0.0)
                try:
                    cph.fit(df, duration_col='T', event_col='E', show_progress=False, step_size=0.5)
                except Exception as e:
                    # If fitting fails, try with stronger penalization for numerical stability
                    cph = CoxPHFitter(penalizer=self.alpha * 10, l1_ratio=0.0)
                    cph.fit(df, duration_col='T', event_col='E', show_progress=False, step_size=0.5)
                loss = float(-cph.log_likelihood_)
                params = cph.params_.values
                self.gamma = params[:Z.shape[1]]
                self.spline_coeffs = params[Z.shape[1]:]
            else:
                raise ValueError("Unsupported family")

            self.knot_vector = t
            if abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            self.beta = self._optimize_beta(X, Z, y, self.beta)

    def _ensure_fitted(self):
        if self.beta is None or self.gamma is None or self.spline_coeffs is None:
            raise RuntimeError("Call fit() before predict().")

    def predict(self, X, Z):
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
        t = self.knot_vector
        if t is None:
            eta = X @ self.beta
            t = self._make_knot_vector(eta.min(), eta.max(), self.num_knots, self.spline_degree)
        return self._linear_predict(X, Z, self.beta, self.gamma, self.spline_coeffs, t, self.spline_degree)

    def predict_proba(self, X, Z):
        return self._sigmoid(self.predict(X, Z))

    def predict_partial_hazard(self, X, Z):
        return np.exp(self.predict(X, Z))

    def g_function(self, x):
        self._ensure_fitted()
        x = np.asarray(x, dtype=np.float64)
        t = self.knot_vector
        if t is None:
            t = self._make_knot_vector(x.min(), x.max(), self.num_knots, self.spline_degree)
        B = self._bspline_basis_matrix(x, t, self.spline_degree)
        return B @ self.spline_coeffs

    # ---------- Bootstrap ----------
    def _refit_clone(self, Xb, Zb, yb, seed):
        m = SplinePLSI(self.family, self.num_knots, self.spline_degree, self.alpha, self.max_iter, self.tol)
        np.random.seed(seed)
        m.fit(Xb, Zb, yb)
        return m.beta.copy(), m.gamma.copy(), m.spline_coeffs.copy()

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, cluster_ids=None, g_grid=None, n_jobs=-1):
        """
        Perform bootstrap inference for parameter uncertainty estimation.

        Args:
            X, Z, y: data arrays
            n_samples: number of bootstrap samples (default 200)
            random_state: random seed
            ci: confidence level (default 0.95)
            cluster_ids: optional cluster IDs for clustered bootstrap
            g_grid: optional grid points for g function estimation
            n_jobs: number of parallel jobs (default -1 for all cores, 1 for sequential)

        Returns:
            dict with bootstrap results
        """
        X = np.asarray(X); Z = np.asarray(Z); y = np.asarray(y)
        if self.beta is None:
            self.fit(X, Z, y)
        N = len(X)

        p, q, s = self.beta.size, self.gamma.size, self.spline_coeffs.size
        beta_samples = np.empty((n_samples, p))
        gamma_samples = np.empty((n_samples, q))
        spline_samples = np.empty((n_samples, s))

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=float).reshape(-1)
            g_samples = np.empty((n_samples, g_grid.size))

        if n_jobs == 1:
            # Sequential bootstrap
            rng = np.random.default_rng(random_state)
            for b in range(n_samples):
                idx = draw_bootstrap_indices(N, rng, cluster_ids)
                Xb, Zb, yb = X[idx], Z[idx], y[idx]
                b_beta, b_gamma, b_spline = self._refit_clone(Xb, Zb, yb, random_state + 1337 + b)
                beta_samples[b] = b_beta; gamma_samples[b] = b_gamma; spline_samples[b] = b_spline

                if do_g:
                    t = self._make_knot_vector(g_grid.min(), g_grid.max(), self.num_knots, self.spline_degree)
                    B = self._bspline_basis_matrix(g_grid, t, self.spline_degree)
                    g_samples[b] = B @ b_spline
        else:
            # Parallel bootstrap
            def refit_wrapper(Xb, Zb, yb, seed):
                b_beta, b_gamma, b_spline = self._refit_clone(Xb, Zb, yb, seed)
                if do_g:
                    t = SplinePLSI._make_knot_vector(g_grid.min(), g_grid.max(), self.num_knots, self.spline_degree)
                    B = SplinePLSI._bspline_basis_matrix(g_grid, t, self.spline_degree)
                    g_vals = B @ b_spline
                    return b_beta, b_gamma, b_spline, g_vals
                return b_beta, b_gamma, b_spline, None

            results = run_parallel_bootstrap(refit_wrapper, X, Z, y, n_samples, random_state, cluster_ids, n_jobs)
            for b, result in enumerate(results):
                beta_samples[b], gamma_samples[b], spline_samples[b] = result[0], result[1], result[2]
                if do_g:
                    g_samples[b] = result[3]

        # point estimates
        beta_hat, gamma_hat, spline_hat = self.beta, self.gamma, self.spline_coeffs

        # se / CI
        self.beta_se = beta_samples.std(axis=0, ddof=1)
        self.gamma_se = gamma_samples.std(axis=0, ddof=1)
        self.spline_se = spline_samples.std(axis=0, ddof=1)
        self.beta_lb, self.beta_ub = self._percentile_ci(beta_samples, ci)
        self.gamma_lb, self.gamma_ub = self._percentile_ci(gamma_samples, ci)
        self.spline_lb, self.spline_ub = self._percentile_ci(spline_samples, ci)

        out = {
            "beta_hat": beta_hat, "beta_se": self.beta_se, "beta_lb": self.beta_lb, "beta_ub": self.beta_ub,
            "gamma_hat": gamma_hat, "gamma_se": self.gamma_se, "gamma_lb": self.gamma_lb, "gamma_ub": self.gamma_ub,
            "spline_hat": spline_hat, "spline_se": self.spline_se, "spline_lb": self.spline_lb, "spline_ub": self.spline_ub,
            "beta_samples": beta_samples, "gamma_samples": gamma_samples, "spline_samples": spline_samples
        }

        if do_g:
            g_mean = g_samples.mean(axis=0); g_se = g_samples.std(axis=0, ddof=1)
            g_lb, g_ub = self._percentile_ci(g_samples, ci)
            self.g_grid, self.g_grid_mean, self.g_grid_se = g_grid, g_mean, g_se
            self.g_grid_lb, self.g_grid_ub = g_lb, g_ub
            self._g_samples = g_samples
            out.update({"g_grid": g_grid, "g_mean": g_mean, "g_se": g_se, "g_lb": g_lb, "g_ub": g_ub})
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


if _HAVE_NUMBA:
    def _raw(func):
        return getattr(func, "__func__", func)  # works for staticmethod or plain function

    raw_mkv = _raw(SplinePLSI._make_knot_vector)
    raw_nb  = _raw(SplinePLSI._n_basis)
    raw_bsm = _raw(SplinePLSI._bspline_basis_matrix)
    raw_sig = _raw(SplinePLSI._sigmoid)

    # First JIT the leaf/helpers
    _jit_mkv = njit(cache=True)(raw_mkv)
    _jit_nb  = njit(cache=True)(raw_nb)
    _jit_bsm = njit(cache=True, parallel=True)(raw_bsm)
    _jit_sig = njit(cache=True)(raw_sig)

    # Now define a NEW jitted linear_predict that calls _jit_bsm directly
    @njit(cache=True, parallel=True)
    def _jit_lin(X, Z, beta, gamma, spline_coeffs, t, degree):
        eta = X @ beta
        B = _jit_bsm(eta, t, degree)          # <- call the jitted function, not via class
        return Z @ gamma + B @ spline_coeffs

    # Re-attach as staticmethods
    SplinePLSI._make_knot_vector       = staticmethod(_jit_mkv)
    SplinePLSI._n_basis                = staticmethod(_jit_nb)
    SplinePLSI._bspline_basis_matrix   = staticmethod(_jit_bsm)
    SplinePLSI._sigmoid                = staticmethod(_jit_sig)
    SplinePLSI._linear_predict         = staticmethod(_jit_lin)
    