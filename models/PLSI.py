import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import Ridge, LogisticRegression
from .base import _SummaryMixin, draw_bootstrap_indices, run_parallel_bootstrap

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False





def _fast_ridge_loss(Xd, y, alpha):
    """Solve ridge regression and return RSS (closed-form)."""
    d = Xd.shape[1]
    A = Xd.T @ Xd + alpha * np.eye(d)
    w = np.linalg.solve(A, Xd.T @ y)
    resid = y - Xd @ w
    return float(resid @ resid)


def _irls_binary_core(Xd, y, alpha, max_iter=25):
    """IRLS logistic regression returning neg log-lik + L2 penalty."""
    n, d = Xd.shape
    w = np.zeros(d)
    for _ in range(max_iter):
        eta = Xd @ w
        for i in range(n):
            if eta[i] > 30.0: eta[i] = 30.0
            elif eta[i] < -30.0: eta[i] = -30.0
        mu = np.empty(n)
        for i in range(n):
            mu[i] = 1.0 / (1.0 + np.exp(-eta[i]))
        s = mu * (1.0 - mu) + 1e-12
        z = eta + (y - mu) / s
        XtSX = np.zeros((d, d))
        for i in range(n):
            for j in range(d):
                for k in range(d):
                    XtSX[j, k] += s[i] * Xd[i, j] * Xd[i, k]
        for j in range(d):
            XtSX[j, j] += alpha
        rhs = np.zeros(d)
        for i in range(n):
            for j in range(d):
                rhs[j] += s[i] * z[i] * Xd[i, j]
        w_new = np.linalg.solve(XtSX, rhs)
        diff = 0.0
        for j in range(d):
            diff += (w_new[j] - w[j]) ** 2
        w = w_new
        if diff < 1e-12:
            break
    eta = Xd @ w
    nll = 0.0
    for i in range(n):
        e = eta[i]
        if e > 30.0: e = 30.0
        elif e < -30.0: e = -30.0
        p = 1.0 / (1.0 + np.exp(-e))
        nll -= y[i] * np.log(p + 1e-9) + (1.0 - y[i]) * np.log(1.0 - p + 1e-9)
    penalty = 0.0
    for j in range(d):
        penalty += w[j] * w[j]
    return nll + 0.5 * alpha * penalty


def _cox_nll(Xd_s, E_s, w, alpha, d):
    """Compute penalized negative partial log-likelihood (log-cumsum-exp stable)."""
    n = Xd_s.shape[0]
    risk = Xd_s @ w
    for i in range(n):
        if risk[i] > 30.0: risk[i] = 30.0
        elif risk[i] < -30.0: risk[i] = -30.0
    max_risk = risk[0]
    for i in range(1, n):
        if risk[i] > max_risk:
            max_risk = risk[i]
    shifted = np.empty(n)
    for i in range(n):
        shifted[i] = risk[i] - max_risk
    exp_shifted = np.exp(shifted)
    cum_exp = np.cumsum(exp_shifted)
    log_cum = np.empty(n)
    for i in range(n):
        log_cum[i] = np.log(cum_exp[i] + 1e-30) + max_risk
    nll = 0.0
    for i in range(n):
        nll -= E_s[i] * (risk[i] - log_cum[i])
    penalty = 0.0
    for j in range(d):
        penalty += w[j] * w[j]
    return nll + 0.5 * alpha * penalty


def _cox_core(Xd_s, E_s, alpha, d):
    """Newton-Raphson Cox solver with log-cumsum-exp and step damping."""
    n = Xd_s.shape[0]
    w = np.zeros(d)
    cur_loss = _cox_nll(Xd_s, E_s, w, alpha, d)
    for _ in range(50):
        risk = Xd_s @ w
        for i in range(n):
            if risk[i] > 30.0: risk[i] = 30.0
            elif risk[i] < -30.0: risk[i] = -30.0
        max_risk = risk[0]
        for i in range(1, n):
            if risk[i] > max_risk:
                max_risk = risk[i]
        shifted = np.empty(n)
        for i in range(n):
            shifted[i] = risk[i] - max_risk
        exp_shifted = np.exp(shifted)
        cum_exp = np.cumsum(exp_shifted)
        ratio = exp_shifted / cum_exp
        grad = Xd_s.T @ (E_s * (1.0 - ratio)) - alpha * w
        gnorm = 0.0
        for j in range(d):
            gnorm += grad[j] * grad[j]
        if gnorm < 1e-12:
            break
        diag_h = E_s * ratio * (1.0 - ratio)
        H = -(Xd_s.T * diag_h) @ Xd_s
        for j in range(d):
            H[j, j] -= alpha
        try:
            step = np.linalg.solve(H, grad)
        except Exception:
            step = -0.01 * grad
        lr = 1.0
        for _ in range(8):
            w_new = w - lr * step
            new_loss = _cox_nll(Xd_s, E_s, w_new, alpha, d)
            if new_loss < cur_loss + 1e-8:
                break
            lr *= 0.5
        w = w - lr * step
        new_loss = _cox_nll(Xd_s, E_s, w, alpha, d)
        if abs(cur_loss - new_loss) < 1e-8:
            cur_loss = new_loss
            break
        cur_loss = new_loss
    return w, _cox_nll(Xd_s, E_s, w, alpha, d)


if _HAVE_NUMBA:
    _irls_binary_core = njit(cache=True)(_irls_binary_core)
    _cox_nll = njit(cache=True)(_cox_nll)
    _cox_core = njit(cache=True)(_cox_core)


def _fast_binary_loss(Xd, y, alpha):
    return float(_irls_binary_core(Xd, y, alpha))


def _fast_cox_loss(Xd, T, E, alpha, sorted_data=False):
    if not sorted_data:
        order = np.argsort(-T)
        Xd, E = Xd[order], E[order]
    _, val = _cox_core(Xd, E, alpha, Xd.shape[1])
    return float(val)


def _fast_cox_fit(Xd, T, E, alpha, sorted_data=False):
    if not sorted_data:
        order = np.argsort(-T)
        Xd, E = Xd[order], E[order]
    w, val = _cox_core(Xd, E, alpha, Xd.shape[1])
    return w, float(val)


class SplinePLSI(_SummaryMixin):
    """
    Partial Linear Single Index Model (PLSI) using B-splines.
    
    Fits a model of the form:
    g(X @ beta) + Z @ gamma
    
    where g is estimated using B-splines.
    Supports continuous, binary, and time-to-event (Cox) outcomes.
    """
    def __init__(self, family='continuous', num_knots=5, spline_degree=3, alpha=1e-6, max_iter=50, tol=1e-6):
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
        _BIG = 1e12

        def objective(beta):
            beta = np.asarray(beta, dtype=np.float64)
            nrm = np.linalg.norm(beta)
            if nrm > 0:
                beta = beta / nrm
            eta = X @ beta
            if not np.all(np.isfinite(eta)):
                return _BIG
            t = SplinePLSI._make_knot_vector(eta.min(), eta.max(), num_knots, spline_degree)
            B = SplinePLSI._bspline_basis_matrix(eta, t, spline_degree)
            Xd = np.hstack((Z, B))

            try:
                if family == 'continuous':
                    val = _fast_ridge_loss(Xd, y, alpha)
                elif family == 'binary':
                    val = _fast_binary_loss(Xd, y, max(alpha, 1e-4))
                elif family == 'cox':
                    cox_pen = max(alpha, 0.1)
                    # Xd is already sorted because X, Z, y are sorted in fit()
                    val = _fast_cox_loss(Xd, None, y[:, 1], cox_pen, sorted_data=True)
                else:
                    raise ValueError("Unsupported family")
            except Exception:
                return _BIG
            return val if np.isfinite(val) else _BIG
        return objective

    def _optimize_beta(self, X, Z, y, beta_init):
        objective = self._objective_factory(X, Z, y)
        constraints = [{'type': 'eq', 'fun': lambda b: np.linalg.norm(b) - 1}]
        result = opt.minimize(objective, beta_init, method='SLSQP', constraints=constraints,
                              options={'maxiter': min(self.max_iter, 20), 'ftol': 1e-6})
        beta = result.x
        if not np.all(np.isfinite(beta)):
            beta = beta_init.copy()
        if beta[0] < 0: beta = -beta
        nrm = np.linalg.norm(beta) or 1.0
        return beta / nrm

    def fit(self, X, Z, y, beta_init=None):
        """
        Fit the SplinePLSI model.
        
        Args:
            X: Exposure matrix (n_samples, p)
            Z: Covariate matrix (n_samples, q)
            y: Outcome vector (n_samples, ) or (n_samples, 2) for Cox
            beta_init: Optional warm-start for beta (e.g. from a previous fit)
        """
        X = np.asarray(X, dtype=np.float64)
        Z = np.asarray(Z, dtype=np.float64)

        if self.family == 'cox':
            # Pre-sort by duration descending to speed up Cox objective and fit
            order = np.argsort(-y[:, 0])
            X, Z, y = X[order], Z[order], y[order]
        
        self.beta = beta_init.copy() if beta_init is not None else self._initialize_params(X)
        prev_loss = np.inf

        n_increases = 0
        for _ in range(self.max_iter):
            eta = X @ self.beta
            if not np.all(np.isfinite(eta)):
                break
            B, t = self._construct_spline_basis(eta)
            Xd = np.hstack((Z, B))

            if self.family == 'continuous':
                model = Ridge(alpha=self.alpha, fit_intercept=False).fit(Xd, y)
                resid = y - model.predict(Xd)
                loss = float(np.sum(resid ** 2))
                coef = np.ravel(model.coef_)
                gamma_new = coef[:Z.shape[1]]
                spline_new = coef[Z.shape[1]:]

            elif self.family == 'binary':
                C_val = min(1.0 / max(self.alpha, 1e-12), 100.0)
                model = LogisticRegression(penalty='l2', C=C_val, fit_intercept=False,
                                           solver='lbfgs', max_iter=500).fit(Xd, y)
                p = model.predict_proba(Xd)[:, 1]
                eps = 1e-9
                loss = float(-np.sum(y * np.log(p + eps) + (1 - y) * np.log(1 - p + eps)))
                coef = model.coef_[0]
                gamma_new = coef[:Z.shape[1]]
                spline_new = coef[Z.shape[1]:]

            elif self.family == 'cox':
                # Xd, y are already sorted
                cox_pen = max(self.alpha, 0.1)
                w, loss = _fast_cox_fit(Xd, None, y[:, 1], cox_pen, sorted_data=True)
                gamma_new = w[:Z.shape[1]]
                spline_new = w[Z.shape[1]:]
                
            else:
                raise ValueError("Unsupported family")

            if not np.isfinite(loss):
                break

            self.gamma = gamma_new
            self.spline_coeffs = spline_new
            self.knot_vector = t
            
            # Check convergence
            if abs(prev_loss - loss) < self.tol:
                break

            if loss > prev_loss:
                n_increases += 1
                if n_increases >= 3:
                    break
            else:
                n_increases = 0

            prev_loss = loss
            
            # Update beta
            prev_beta = self.beta.copy()
            self.beta = self._optimize_beta(X, Z, y, self.beta)
            if not np.all(np.isfinite(self.beta)):
                self.beta = prev_beta
                break

    def _ensure_fitted(self):
        if self.beta is None or self.gamma is None or self.spline_coeffs is None:
            raise RuntimeError("Call fit() before predict().")

    def predict(self, X, Z):
        """Predict raw output (linear predictor)."""
        self._ensure_fitted()
        X = np.asarray(X, dtype=np.float64); Z = np.asarray(Z, dtype=np.float64)
        t = self.knot_vector
        if t is None:
            eta = X @ self.beta
            t = self._make_knot_vector(eta.min(), eta.max(), self.num_knots, self.spline_degree)
        return self._linear_predict(X, Z, self.beta, self.gamma, self.spline_coeffs, t, self.spline_degree)

    def predict_proba(self, X, Z):
        """Predict probabilities (for binary outcome)."""
        return self._sigmoid(self.predict(X, Z))

    def predict_partial_hazard(self, X, Z):
        """Predict partial hazard exp(g(Xb) + Zg) (for Cox outcome)."""
        return np.exp(self.predict(X, Z))

    def g_function(self, x):
        """Estimate g(x) for given scalar inputs x."""
        self._ensure_fitted()
        x = np.asarray(x, dtype=np.float64)
        t = self.knot_vector
        if t is None:
            t = self._make_knot_vector(x.min(), x.max(), self.num_knots, self.spline_degree)
        B = self._bspline_basis_matrix(x, t, self.spline_degree)
        return B @ self.spline_coeffs

    def _refit_clone(self, Xb, Zb, yb, seed, beta_init=None):
        m = SplinePLSI(self.family, self.num_knots, self.spline_degree, self.alpha, self.max_iter, self.tol)
        np.random.seed(seed)
        m.fit(Xb, Zb, yb, beta_init=beta_init)
        return m.beta.copy(), m.gamma.copy(), m.spline_coeffs.copy(), m.knot_vector.copy()

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs=-1):
        """
        Perform bootstrap inference to estimate SE and CI for beta, gamma, and g(x).
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

        warm_beta = None

        if n_jobs == 1:
            rng = np.random.default_rng(random_state)
            for b in range(n_samples):
                idx = draw_bootstrap_indices(N, rng)
                Xb, Zb, yb = X[idx], Z[idx], y[idx]
                b_beta, b_gamma, b_spline, b_knots = self._refit_clone(
                    Xb, Zb, yb, random_state + 1337 + b, beta_init=warm_beta)
                beta_samples[b] = b_beta; gamma_samples[b] = b_gamma; spline_samples[b] = b_spline

                if do_g:
                    B = self._bspline_basis_matrix(g_grid, b_knots, self.spline_degree)
                    g_samples[b] = B @ b_spline
        else:
            def refit_wrapper(Xb, Zb, yb, seed):
                b_beta, b_gamma, b_spline, b_knots = self._refit_clone(
                    Xb, Zb, yb, seed, beta_init=warm_beta)
                if do_g:
                    B = SplinePLSI._bspline_basis_matrix(g_grid, b_knots, self.spline_degree)
                    g_vals = B @ b_spline
                    return b_beta, b_gamma, b_spline, g_vals
                return b_beta, b_gamma, b_spline, None

            results = run_parallel_bootstrap(refit_wrapper, X, Z, y, n_samples, random_state, n_jobs)
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
        return getattr(func, "__func__", func)

    raw_mkv = _raw(SplinePLSI._make_knot_vector)
    raw_nb  = _raw(SplinePLSI._n_basis)
    raw_bsm = _raw(SplinePLSI._bspline_basis_matrix)
    raw_sig = _raw(SplinePLSI._sigmoid)

    _jit_mkv = njit(cache=True)(raw_mkv)
    _jit_nb  = njit(cache=True)(raw_nb)
    _jit_bsm = njit(cache=True)(raw_bsm)
    _jit_sig = njit(cache=True)(raw_sig)

    @njit(cache=True)
    def _jit_lin(X, Z, beta, gamma, spline_coeffs, t, degree):
        eta = X @ beta
        B = _jit_bsm(eta, t, degree)
        return Z @ gamma + B @ spline_coeffs

    SplinePLSI._make_knot_vector       = staticmethod(_jit_mkv)
    SplinePLSI._n_basis                = staticmethod(_jit_nb)
    SplinePLSI._bspline_basis_matrix   = staticmethod(_jit_bsm)
    SplinePLSI._sigmoid                = staticmethod(_jit_sig)
    SplinePLSI._linear_predict         = staticmethod(_jit_lin)