import numpy as np
import scipy.optimize as opt
from scipy.interpolate import BSpline
from sklearn.linear_model import Ridge
from sklearn.neighbors import KernelDensity


class SplinePLSI:
    def __init__(self, num_knots=5, spline_degree=3, alpha_values=None, max_iter=50, tol=1e-6):
        """
        Initializes the I-PLSI model with Generalized Cross-Validation (GCV).
        
        Parameters:
        - num_knots: Number of interior knots for the spline.
        - spline_degree: Degree of the B-spline.
        - alpha_values: List of alpha values to test for GCV. If None, defaults to a logarithmic range.
        - max_iter: Maximum number of iterations for optimization.
        - tol: Tolerance for convergence.
        """
        self.num_knots = num_knots
        self.spline_degree = spline_degree
        self.alpha_values = alpha_values if alpha_values is not None else np.logspace(-3, 3, 10)
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None  # Single-index coefficients
        self.gamma = None  # Linear coefficients
        self.spline_coeffs = None  # Coefficients for the spline function
        self.spline_basis = None  # Spline basis used in training
        self.best_alpha = None  # Optimal alpha selected by GCV
    
    def _initialize_params(self, X):
        """Initialize the index parameter beta."""
        _, p = X.shape
        beta_init = np.random.randn(p)
        beta_init /= np.linalg.norm(beta_init)  # Normalize beta
        return beta_init
    
    def _construct_spline_basis(self, eta):
        """Constructs B-spline basis for the estimated single-index variable eta."""
        knots = np.linspace(np.min(eta), np.max(eta), self.num_knots)
        t = np.concatenate(([knots[0]] * self.spline_degree, knots, [knots[-1]] * self.spline_degree))
        basis = np.array([BSpline(t, (np.arange(len(t) - self.spline_degree - 1) == i).astype(int), self.spline_degree)(eta)
                          for i in range(len(t) - self.spline_degree - 1)]).T
        return basis, t
    
    def _optimize_beta(self, X, Y, Z, beta_init):
        """Optimizes the single-index parameter beta."""
        def objective(beta):
            eta = X @ beta
            B, _ = self._construct_spline_basis(eta)
            ridge = Ridge(alpha=self.best_alpha, fit_intercept=False)
            ridge.fit(np.hstack((Z, B)), Y)
            residuals = Y - ridge.predict(np.hstack((Z, B)))
            return np.sum(residuals ** 2)

        norm_constraint = {'type': 'eq', 'fun': lambda beta: np.linalg.norm(beta) - 1}
        nonnegative_constraint = {'type': 'ineq', 'fun': lambda beta: beta[0]}
        result = opt.minimize(
            objective, 
            beta_init, 
            method='SLSQP',  # Supports constraints
            constraints=[norm_constraint],
            options={'maxiter': self.max_iter}
        )
        beta = result.x
        if beta[0] < 0:
            beta = -beta
        return beta
    
    def _gcv_score(self, X, Z, Y, alpha):
        """Computes the Generalized Cross-Validation (GCV) score for a given alpha."""
        eta = X @ self.beta
        B, _ = self._construct_spline_basis(eta)
        X_design = np.hstack((Z, B))
        
        # Compute hat matrix H
        n, d = X_design.shape
        I = np.eye(d)
        H = X_design @ np.linalg.inv(X_design.T @ X_design + alpha * I) @ X_design.T
        trace_H = np.trace(H)
        
        # Compute GCV score
        residuals = Y - H @ Y
        gcv_score = np.sum(residuals ** 2) / (1 - trace_H / n) ** 2
        return gcv_score
    
    def select_optimal_alpha(self, X, Z, Y):
        """Selects the best regularization parameter alpha using Generalized Cross-Validation (GCV)."""
        self.best_alpha = min(self.alpha_values, key=lambda alpha: self._gcv_score(X, Z, Y, alpha))
    
    def fit(self, X, Z, Y):
        """
        Fits the I-PLSI model with optimal alpha selection.
        
        Parameters:
        - X: High-dimensional covariates (n x p)
        - Z: Low-dimensional linear covariates (n x q)
        - Y: Response variable (n x 1)
        """
        self.beta = self._initialize_params(X)
        self.select_optimal_alpha(X, Z, Y)  # Select optimal alpha via GCV
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            eta = X @ self.beta
            B, spline_knots = self._construct_spline_basis(eta)
            ridge = Ridge(alpha=self.best_alpha, fit_intercept=False)
            ridge.fit(np.hstack((Z, B)), Y)
            self.gamma = ridge.coef_[:Z.shape[1]]
            self.spline_coeffs = ridge.coef_[Z.shape[1]:]
            self.spline_basis = spline_knots  # Save knots for later use
            
            loss = np.sum((Y - ridge.predict(np.hstack((Z, B)))) ** 2)
            if np.abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            self.beta = self._optimize_beta(X, Y, Z, self.beta)
    
    def predict(self, X, Z):
        """
        Predicts Y using the fitted I-PLSI model.
        
        Parameters:
        - X: High-dimensional covariates
        - Z: Low-dimensional linear covariates
        
        Returns:
        - Predicted Y values.
        """
        eta = X @ self.beta
        B, _ = self._construct_spline_basis(eta)
        return Z @ self.gamma + B @ self.spline_coeffs

    def g_function(self, x):
        """
        Estimates the nonlinear function g(x) = f(x).
        
        Parameters:
        - x: scaler value
        
        Returns:
        - Estimated g(x) values.
        """
        B, _ = self._construct_spline_basis(x)
        return B @ self.spline_coeffs




class KernelPLSI:
    def __init__(self, bandwidth=1.0, alpha_values=None, max_iter=50, tol=1e-6):
        """
        Initializes the Kernel PLSI model with Generalized Cross-Validation (GCV).
        
        Parameters:
        - bandwidth: Bandwidth for Gaussian kernel smoothing.
        - alpha_values: List of alpha values to test for GCV. If None, defaults to a logarithmic range.
        - max_iter: Maximum number of iterations for optimization.
        - tol: Tolerance for convergence.
        """
        self.bandwidth = bandwidth
        self.alpha_values = alpha_values if alpha_values is not None else np.logspace(-3, 3, 10)
        self.max_iter = max_iter
        self.tol = tol
        self.beta = None  # Single-index coefficients
        self.gamma = None  # Linear coefficients
        self.best_alpha = None  # Optimal alpha selected by GCV
        self.eta_train = None  # Store eta values for kernel smoothing
        self.Y_train = None  # Store Y values for kernel smoothing
        self.fitted = False  # Track whether model is fitted

    def _initialize_params(self, X):
        """Initialize the index parameter beta."""
        _, p = X.shape
        beta_init = np.random.randn(p)
        beta_init /= np.linalg.norm(beta_init)  # Normalize beta
        return beta_init
    
    def _kernel_smooth(self, eta):
        """
        Estimates g(eta) using Nadaraya-Watson kernel smoothing.
        Uses stored (eta_train, Y_train) to compute the smoothed function.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before applying kernel smoothing.")

        smoothed_g = np.zeros_like(eta)

        for i in range(len(eta)):
            # Compute Gaussian kernel weights
            weights = np.exp(-0.5 * ((eta[i] - self.eta_train) / self.bandwidth) ** 2)
            weights /= np.sum(weights)  # Normalize weights

            # Weighted sum to estimate g(eta)
            smoothed_g[i] = np.sum(weights * self.Y_train)

        return smoothed_g

    def _optimize_beta(self, X, Y, Z, beta_init):
        """Optimizes the single-index parameter beta."""
        def objective(beta):
            eta = X @ beta
            ridge = Ridge(alpha=self.best_alpha, fit_intercept=False)
            ridge.fit(Z, Y - self._kernel_smooth(eta))  # Subtract g(eta)
            residuals = Y - (ridge.predict(Z) + self._kernel_smooth(eta))
            return np.sum(residuals ** 2)

        norm_constraint = {'type': 'eq', 'fun': lambda beta: np.linalg.norm(beta) - 1}
        nonnegative_constraint = {'type': 'ineq', 'fun': lambda beta: beta}
        result = opt.minimize(
            objective, 
            beta_init, 
            method='SLSQP',  # Supports constraints
            constraints=[norm_constraint, nonnegative_constraint],
            options={'maxiter': self.max_iter}
        )
        return result.x
    
    def _gcv_score(self, X, Z, Y, alpha):
        """Computes the Generalized Cross-Validation (GCV) score for a given alpha."""
        eta = X @ self.beta
        smoothed_g = self._kernel_smooth(eta)
        X_design = np.hstack((Z, smoothed_g[:, None]))
        
        # Compute hat matrix H
        n, d = X_design.shape
        I = np.eye(d)
        H = X_design @ np.linalg.inv(X_design.T @ X_design + alpha * I) @ X_design.T
        trace_H = np.trace(H)
        
        # Compute GCV score
        residuals = Y - H @ Y
        gcv_score = np.sum(residuals ** 2) / (1 - trace_H / n) ** 2
        return gcv_score
    
    def select_optimal_alpha(self, X, Z, Y):
        """Selects the best regularization parameter alpha using Generalized Cross-Validation (GCV)."""
        self.best_alpha = min(self.alpha_values, key=lambda alpha: self._gcv_score(X, Z, Y, alpha))
    
    def fit(self, X, Z, Y):
        """
        Fits the Kernel PLSI model with optimal alpha selection.
        
        Parameters:
        - X: High-dimensional covariates (n x p)
        - Z: Low-dimensional linear covariates (n x q)
        - Y: Response variable (n x 1)
        """
        self.beta = self._initialize_params(X)
        self.select_optimal_alpha(X, Z, Y)  # Select optimal alpha via GCV
        prev_loss = np.inf

        for iteration in range(self.max_iter):
            eta = X @ self.beta
            self.eta_train = eta  # Store eta for kernel smoothing
            self.Y_train = Y  # Store Y values for kernel smoothing
            
            ridge = Ridge(alpha=self.best_alpha, fit_intercept=False)
            ridge.fit(Z, Y - self._kernel_smooth(eta))  # Estimate gamma without g(eta)
            self.gamma = ridge.coef_
            
            loss = np.sum((Y - (ridge.predict(Z) + self._kernel_smooth(eta))) ** 2)
            if np.abs(prev_loss - loss) < self.tol:
                break
            prev_loss = loss

            self.beta = self._optimize_beta(X, Y, Z, self.beta)
        
        self.fitted = True  # Mark model as fitted
    
    def predict(self, X, Z):
        """
        Predicts Y using the fitted Kernel PLSI model.
        
        Parameters:
        - X: High-dimensional covariates
        - Z: Low-dimensional linear covariates
        
        Returns:
        - Predicted Y values.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before making predictions.")

        eta = X @ self.beta
        return Z @ self.gamma + self._kernel_smooth(eta)

    def g_function(self, x):
        """
        Estimates the nonlinear function g(x) using kernel smoothing.
        
        Parameters:
        - x: Scalar value or array
        
        Returns:
        - Estimated g(x) values.
        """
        if not self.fitted:
            raise ValueError("Model must be fitted before estimating g(x).")

        x = np.array(x).reshape(-1)
        return self._kernel_smooth(x)