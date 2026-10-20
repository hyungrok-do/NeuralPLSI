import numpy as np
import scipy.optimize as opt
from .base import (_SummaryMixin, run_parallel_bootstrap, validate_data,
                   validate_bootstrap)

try:
    from numba import njit
    _HAVE_NUMBA = True
except ImportError:
    _HAVE_NUMBA = False





def _fast_ridge_loss(Xd, y, alpha):
    """Solve ridge regression and return the penalized RSS."""
    d = Xd.shape[1]
    A = Xd.T @ Xd + alpha * np.eye(d)
    w = np.linalg.solve(A, Xd.T @ y)
    resid = y - Xd @ w
    return float(resid @ resid + alpha*(w @ w))


def _binary_fit(Xd, y, alpha):
    """One penalized likelihood for both profiling and the reported fit."""
    from scipy.special import expit

    def objective(w):
        eta = Xd @ w
        loss = np.logaddexp(0., eta).sum() - y @ eta + .5*alpha*(w @ w)
        gradient = Xd.T @ (expit(eta)-y) + alpha*w
        return float(loss), gradient

    result = opt.minimize(objective, np.zeros(Xd.shape[1]), jac=True,
                          method='L-BFGS-B', options={'maxiter': 500, 'ftol': 1e-12, 'gtol': 1e-7})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise FloatingPointError('Nonfinite spline logistic fit')
    return result.x, float(result.fun)


def _fast_binary_loss(Xd, y, alpha):
    return _binary_fit(Xd, y, alpha)[1]


def _cox_loss_and_grad(Xd_s, E_s, w, alpha, d=None, group_ends=None):
    """Breslow partial likelihood with stable complete tied risk sets.

    Xd_s and E_s are sorted by decreasing duration. group_ends gives the last
    row of each tied-duration group. Omitting it represents distinct times.
    """
    risk = Xd_s @ w
    n = len(risk)
    if group_ends is None:
        group_ends = np.arange(n)
    starts = np.r_[0, group_ends[:-1]+1]
    counts = group_ends-starts+1
    events = np.add.reduceat(E_s, starts)
    log_risk = np.logaddexp.accumulate(risk)[group_ends]
    denominator = np.repeat(log_risk, counts)
    loss = np.sum(E_s*(denominator-risk)) + .5*alpha*(w @ w)
    log_event = np.full(len(events), -np.inf)
    active = events > 0
    log_event[active] = np.log(events[active])-log_risk[active]
    tail = np.logaddexp.accumulate(log_event[::-1])[::-1]
    derivative = np.exp(risk+np.repeat(tail, counts))-E_s
    return float(loss), Xd_s.T @ derivative + alpha*w


def _cox_nll(Xd_s, E_s, w, alpha, d=None, group_ends=None):
    return _cox_loss_and_grad(Xd_s, E_s, w, alpha, d, group_ends)[0]


def _cox_core(Xd_s, E_s, alpha, d, group_ends=None):
    def objective(w):
        return _cox_loss_and_grad(Xd_s, E_s, w, alpha, d, group_ends)
    result = opt.minimize(objective, np.zeros(d), method='L-BFGS-B', jac=True,
                          options={'maxiter': 500, 'ftol': 1e-10, 'gtol': 1e-7})
    if not np.isfinite(result.fun) or not np.isfinite(result.x).all():
        raise FloatingPointError('Nonfinite spline Cox fit')
    return result.x, float(result.fun)


def _fast_cox_fit(Xd, T, E, alpha, sorted_data=False):
    T = np.asarray(T)
    if not sorted_data:
        order = np.argsort(-T, kind='stable')
        Xd, T, E = Xd[order], T[order], E[order]
    ends = np.r_[np.flatnonzero(T[:-1] != T[1:]), len(T)-1]
    return _cox_core(Xd, E, alpha, Xd.shape[1], ends)


def _fast_cox_loss(Xd, T, E, alpha, sorted_data=False):
    return _fast_cox_fit(Xd, T, E, alpha, sorted_data)[1]


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
    def _make_knot_vector(eta, num_knots, degree):
        eta_sorted = np.sort(eta.astype(np.float64))
        if eta_sorted[-1] <= eta_sorted[0]:
            raise ValueError('The exposure index must vary to fit a spline')
        
        # Following Wang et al. (2020), knots are placed at equally spaced sample quantiles
        # of the estimated single index to ensure even data distribution across splines.
        idx = np.linspace(0, eta_sorted.size - 1, num_knots).astype(np.int64)
        knots = eta_sorted[idx]
        
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
        eta = np.clip(eta.astype(np.float64), t[0], t[-1])
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

    def _initialize_params(self, X, random_state=None):
        rng = np.random if random_state is None else np.random.RandomState(random_state)
        beta_init = rng.standard_normal(X.shape[1])
        if beta_init[0] < 0: beta_init = -beta_init
        nrm = np.linalg.norm(beta_init) or 1.0
        return beta_init / nrm

    def _construct_spline_basis(self, eta):
        t = self._make_knot_vector(eta, self.num_knots, self.spline_degree)
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
            t = SplinePLSI._make_knot_vector(eta, num_knots, spline_degree)
            B = SplinePLSI._bspline_basis_matrix(eta, t, spline_degree)
            Xd = np.hstack((Z, B))

            try:
                if family == 'continuous':
                    val = _fast_ridge_loss(Xd, y, alpha)
                elif family == 'binary':
                    val = _fast_binary_loss(Xd, y, alpha)
                elif family == 'cox':
                    cox_pen = alpha
                    # Xd is already sorted because X, Z, y are sorted in fit()
                    val = _fast_cox_loss(Xd, y[:, 0], y[:, 1], cox_pen, sorted_data=True)
                else:
                    raise ValueError("Unsupported family")
            except (np.linalg.LinAlgError, FloatingPointError, ValueError):
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

    def _fit_linear_part(self, X, Z, y, beta):
        B, knots = self._construct_spline_basis(X @ beta)
        design = np.column_stack([Z, B])
        if self.family == 'continuous':
            coefficient = np.linalg.solve(design.T @ design + self.alpha*np.eye(design.shape[1]),
                                          design.T @ y)
            residual = y-design @ coefficient
            loss = float(residual @ residual + self.alpha*(coefficient @ coefficient))
        elif self.family == 'binary':
            coefficient, loss = _binary_fit(design, y, self.alpha)
        else:
            coefficient, loss = _fast_cox_fit(design, y[:, 0], y[:, 1],
                                              self.alpha, sorted_data=True)
        return coefficient[:Z.shape[1]], coefficient[Z.shape[1]:], knots, loss

    def fit(self, X, Z, y, beta_init=None, random_state=None):
        """Fit a coherent profiled index, spline, and linear component.

        random_state provides local reproducible initialization. None retains
        compatibility with callers that seed NumPy's global generator.
        """
        X, Z, y = validate_data(X, Z, y, self.family)
        if self.max_iter < 1 or self.alpha < 0 or self.num_knots < 2 or self.spline_degree < 0:
            raise ValueError('Invalid iteration count, penalty, or spline dimensions')
        self._clear_inference()
        if self.family == 'cox':
            order = np.argsort(-y[:, 0], kind='stable')
            X, Z, y = X[order], Z[order], y[order]
        beta = (np.asarray(beta_init, dtype=float).copy() if beta_init is not None
                else self._initialize_params(X, random_state))
        if beta.shape != (X.shape[1],) or not np.isfinite(beta).all() or np.linalg.norm(beta) == 0:
            raise ValueError('beta_init must be a finite nonzero exposure direction')
        beta /= np.linalg.norm(beta)
        if beta[0] < 0:
            beta *= -1
        gamma, spline, knots, loss = self._fit_linear_part(X, Z, y, beta)
        self.converged_ = False
        self.n_iter_ = 0
        for iteration in range(self.max_iter):
            candidate = self._optimize_beta(X, Z, y, beta)
            new_gamma, new_spline, new_knots, new_loss = self._fit_linear_part(X, Z, y, candidate)
            self.n_iter_ = iteration+1
            if not np.isfinite(new_loss):
                raise FloatingPointError('Nonfinite spline profile objective')
            if new_loss > loss:
                self.converged_ = abs(new_loss-loss) < self.tol
                break
            improvement = loss-new_loss
            beta, gamma, spline, knots, loss = candidate, new_gamma, new_spline, new_knots, new_loss
            if improvement < self.tol:
                self.converged_ = True
                break
        # Every reported component belongs to the same final index.
        self.beta, self.gamma = beta, gamma
        self.spline_coeffs, self.knot_vector, self.loss_ = spline, knots, loss
        return self

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
            t = self._make_knot_vector(eta, self.num_knots, self.spline_degree)
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
        x = np.asarray(x, dtype=np.float64).reshape(-1)
        t = self.knot_vector
        if t is None:
            t = self._make_knot_vector(x, self.num_knots, self.spline_degree)
        B = self._bspline_basis_matrix(x, t, self.spline_degree)
        return B @ self.spline_coeffs

    def _refit_clone(self, Xb, Zb, yb, seed, beta_init=None):
        m = SplinePLSI(self.family, self.num_knots, self.spline_degree, self.alpha, self.max_iter, self.tol)
        m.fit(Xb, Zb, yb, beta_init=beta_init, random_state=seed)
        return m.beta.copy(), m.gamma.copy(), m.spline_coeffs.copy(), m.knot_vector.copy()

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs=-1):
        """
        Perform bootstrap inference to estimate SE and CI for beta, gamma, and g(x).
        """
        validate_bootstrap(n_samples, ci, random_state)
        X, Z, y = validate_data(X, Z, y, self.family)
        if self.beta is None:
            self.fit(X, Z, y, random_state=random_state)
        self._clear_inference()
        N = len(X)

        p, q, s = self.beta.size, self.gamma.size, self.spline_coeffs.size
        beta_samples = np.empty((n_samples, p))
        gamma_samples = np.empty((n_samples, q))
        spline_samples = np.empty((n_samples, s))

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=float).reshape(-1)
            if not g_grid.size or not np.isfinite(g_grid).all():
                raise ValueError('g_grid must be nonempty and finite')
            g_samples = np.empty((n_samples, g_grid.size))

        warm_beta = None

        def refit_wrapper(Xb, Zb, yb, seed):
            b_beta, b_gamma, b_spline, b_knots = self._refit_clone(
                Xb, Zb, yb, seed, beta_init=warm_beta)
            g_values = None
            if do_g:
                basis = self._bspline_basis_matrix(g_grid, b_knots, self.spline_degree)
                g_values = basis @ b_spline
            return b_beta, b_gamma, b_spline, g_values

        results = run_parallel_bootstrap(refit_wrapper, X, Z, y, n_samples, random_state, n_jobs)
        for b, result in enumerate(results):
            beta_samples[b], gamma_samples[b], spline_samples[b] = result[:3]
            if do_g:
                g_samples[b] = result[3]
        arrays = [beta_samples, gamma_samples, spline_samples]+([g_samples] if do_g else [])
        if not all(np.isfinite(a).all() for a in arrays):
            raise FloatingPointError('Nonfinite spline bootstrap estimates')

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
            "beta_samples": beta_samples, "gamma_samples": gamma_samples, "spline_samples": spline_samples,
            "quantile_method": "linear", "ci": ci, "n_bootstrap": n_samples
        }

        if do_g:
            g_mean = g_samples.mean(axis=0); g_se = g_samples.std(axis=0, ddof=1)
            g_lb, g_ub = self._percentile_ci(g_samples, ci)
            self.g_grid, self.g_grid_mean, self.g_grid_se = g_grid, g_mean, g_se
            self.g_grid_lb, self.g_grid_ub = g_lb, g_ub
            self._g_samples = g_samples
            out.update({"g_grid": g_grid, "g_mean": g_mean, "g_se": g_se, "g_lb": g_lb, "g_ub": g_ub, "g_samples": g_samples})
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

    _jit_mkv = njit(cache=False)(raw_mkv)
    _jit_nb  = njit(cache=False)(raw_nb)
    _jit_bsm = njit(cache=False)(raw_bsm)
    _jit_sig = njit(cache=False)(raw_sig)

    @njit(cache=False)
    def _jit_lin(X, Z, beta, gamma, spline_coeffs, t, degree):
        eta = X @ beta
        B = _jit_bsm(eta, t, degree)
        return Z @ gamma + B @ spline_coeffs

    SplinePLSI._make_knot_vector       = staticmethod(_jit_mkv)
    SplinePLSI._n_basis                = staticmethod(_jit_nb)
    SplinePLSI._bspline_basis_matrix   = staticmethod(_jit_bsm)
    SplinePLSI._sigmoid                = staticmethod(_jit_sig)
    SplinePLSI._linear_predict         = staticmethod(_jit_lin)
