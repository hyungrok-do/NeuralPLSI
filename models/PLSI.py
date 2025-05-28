import numpy as np
import scipy.optimize as opt
from sklearn.linear_model import Ridge, LogisticRegression
from scipy.interpolate import BSpline
from lifelines import CoxPHFitter
import pandas as pd

class SplinePLSI:
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
        self.spline_basis = None

    def _initialize_params(self, X):
        beta_init = np.random.randn(X.shape[1])
        if beta_init[0] < 0:
            beta_init = -beta_init
        return beta_init / np.linalg.norm(beta_init)

    def _construct_spline_basis(self, eta):
        knots = np.linspace(np.min(eta), np.max(eta), self.num_knots)
        t = np.concatenate(([knots[0]] * self.spline_degree, knots, [knots[-1]] * self.spline_degree))
        basis = np.array([
            BSpline(t, (np.arange(len(t) - self.spline_degree - 1) == i).astype(int), self.spline_degree)(eta)
            for i in range(len(t) - self.spline_degree - 1)
        ]).T
        return basis, t

    def _optimize_beta(self, X, Z, y, beta_init):
        def objective(beta):
            eta = X @ beta
            B, _ = self._construct_spline_basis(eta)
            X_design = np.hstack((Z, B))

            if self.family == 'continuous':
                ridge = Ridge(alpha=self.alpha, fit_intercept=False)
                ridge.fit(X_design, y)
                pred = ridge.predict(X_design)
                return np.sum((y - pred) ** 2)
            elif self.family == 'binary':
                model = LogisticRegression(penalty='l2', C=1.0/self.alpha, fit_intercept=False, solver='lbfgs')
                model.fit(X_design, y)
                pred_prob = model.predict_proba(X_design)[:, 1]
                eps = 1e-9
                return -np.sum(y * np.log(pred_prob + eps) + (1 - y) * np.log(1 - pred_prob + eps))
            elif self.family == 'cox':
                df = pd.DataFrame(np.hstack((Z, B)), columns=[f'z{i}' for i in range(Z.shape[1])] + [f'b{i}' for i in range(B.shape[1])])
                df['T'], df['E'] = y[:, 0], y[:, 1]
                cph = CoxPHFitter(penalizer=self.alpha)
                cph.fit(df, duration_col='T', event_col='E', show_progress=False)
                return -cph.log_likelihood_
            else:
                raise ValueError("Unsupported family")

        constraints = [{'type': 'eq', 'fun': lambda b: np.linalg.norm(b) - 1}]
        result = opt.minimize(objective, beta_init, method='SLSQP', constraints=constraints, options={'maxiter': self.max_iter})
        beta = result.x
        if self.beta[0] < 0:
            beta = -beta
        return beta / np.linalg.norm(beta)

    def fit(self, X, Z, y):
        self.beta = self._initialize_params(X)
        prev_loss = np.inf

        for _ in range(self.max_iter):
            eta = X @ self.beta
            B, spline_knots = self._construct_spline_basis(eta)
            X_design = np.hstack((Z, B))

            if self.family == 'continuous':
                model = Ridge(alpha=self.alpha, fit_intercept=False)
                model.fit(X_design, y)
                residuals = y - model.predict(X_design)
                loss = np.sum(residuals ** 2)
                self.gamma = model.coef_[:Z.shape[1]]
                self.spline_coeffs = model.coef_[Z.shape[1]:]
            elif self.family == 'binary':
                model = LogisticRegression(penalty='l2', C=1.0/self.alpha, fit_intercept=False, solver='lbfgs')
                model.fit(X_design, y)
                loss = -np.sum(y * np.log(model.predict_proba(X_design)[:, 1] + 1e-9))
                self.gamma = model.coef_[0][:Z.shape[1]]
                self.spline_coeffs = model.coef_[0][Z.shape[1]:]
            elif self.family == 'cox':
                df = pd.DataFrame(X_design, columns=[f'z{i}' for i in range(Z.shape[1])] + [f'b{i}' for i in range(B.shape[1])])
                df['T'], df['E'] = y[:, 0], y[:, 1]
                cph = CoxPHFitter(penalizer=self.alpha)
                cph.fit(df, duration_col='T', event_col='E', show_progress=False)
                loss = -cph.log_likelihood_
                self.gamma = cph.params_.values[:Z.shape[1]]
                self.spline_coeffs = cph.params_.values[Z.shape[1]:]
            else:
                raise ValueError("Unsupported family")

            self.spline_basis = spline_knots
            if np.abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss
            self.beta = self._optimize_beta(X, Z, y, self.beta)

    def predict(self, X, Z):
        eta = X @ self.beta
        B, _ = self._construct_spline_basis(eta)
        return Z @ self.gamma + B @ self.spline_coeffs
    
    def predict_proba(self, X, Z):
        eta = X @ self.beta
        B, _ = self._construct_spline_basis(eta)
        linear_pred = Z @ self.gamma + B @ self.spline_coeffs
        return 1 / (1 + np.exp(-linear_pred))
    
    def predict_partial_hazard(self, X, Z):
        eta = X @ self.beta
        B, _ = self._construct_spline_basis(eta)
        linear_pred = Z @ self.gamma + B @ self.spline_coeffs
        return np.exp(linear_pred)

    def g_function(self, x):
        B, _ = self._construct_spline_basis(x)
        return B @ self.spline_coeffs
