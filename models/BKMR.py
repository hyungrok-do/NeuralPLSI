import numpy as np
import scipy.linalg as la
from scipy.spatial.distance import cdist
from scipy.stats import invgamma

class BKMR:
    def __init__(self, num_samples=1000, burn_in=500, kernel_type="RBF", alpha=1.0, length_scale=1.0, reg_lambda=1e-6, num_inducing=50):
        """
        Bayesian Kernel Machine Regression (BKMR) with computational efficiency improvements.
        
        Parameters:
        - num_samples: Number of MCMC samples.
        - burn_in: Number of burn-in samples.
        - kernel_type: "RBF" for Radial Basis Function.
        - alpha: Kernel variance parameter.
        - length_scale: Kernel length-scale.
        - reg_lambda: Regularization for stability.
        - num_inducing: Number of inducing points for sparse kernel approximation.
        """
        self.num_samples = num_samples
        self.burn_in = burn_in
        self.kernel_type = kernel_type
        self.alpha = alpha
        self.length_scale = length_scale
        self.reg_lambda = reg_lambda
        self.num_inducing = num_inducing  # Number of inducing points for Nyström approximation

        # Internal storage for training data
        self.X_train = None
        self.Z_train = None
        self.h_mean = None
        self.gamma_mean = None
        self.sigma2_mean = None

    def _rbf_kernel(self, X1, X2):
        """Computes the RBF kernel with Nyström approximation for efficiency."""
        dists = cdist(X1, X2, metric="sqeuclidean")
        return self.alpha * np.exp(-dists / (2 * self.length_scale ** 2))

    def _initialize_params(self, n, q):
        """Initialize model parameters efficiently."""
        return np.random.normal(0, 1, size=n), np.random.normal(0, 1, size=q), 1.0

    def _compute_sparse_kernel(self, X):
        """Compute Nyström approximation for the kernel to improve efficiency."""
        inducing_idx = np.random.choice(X.shape[0], self.num_inducing, replace=False)
        X_inducing = X[inducing_idx, :]
        K_mm = self._rbf_kernel(X_inducing, X_inducing) + self.reg_lambda * np.eye(self.num_inducing)
        K_nm = self._rbf_kernel(X, X_inducing)

        # Efficiently approximate full kernel
        K_approx = K_nm @ la.inv(K_mm) @ K_nm.T
        return K_approx

    def _sample_h(self, X, Y, Z, gamma, sigma2, K_approx):
        """Efficient MCMC sampling for h(X) using Nyström kernel and Cholesky."""
        n = len(Y)
        noise_matrix = sigma2 * np.eye(n) + self.reg_lambda * np.eye(n)  # Add jitter for stability
        residuals = Y - Z @ gamma

        try:
            # Compute Cholesky decomposition of posterior covariance
            L = la.cholesky(K_approx + noise_matrix, lower=True)
            post_cov = la.cho_solve((L, True), np.eye(n))  # Faster than direct inversion
            post_mean = post_cov @ residuals

            # Generate sample using Cholesky factorization
            z = np.random.normal(0, 1, n)  # Standard normal noise
            h_sample = post_mean + L @ z  # Sampling via Cholesky
            return h_sample

        except la.LinAlgError:
            print("Cholesky failed, adding more jitter and retrying...")
            jitter = 1e-5 * np.eye(n)
            try:
                L = la.cholesky(K_approx + noise_matrix + jitter, lower=True)
                post_cov = la.cho_solve((L, True), np.eye(n))
                post_mean = post_cov @ residuals

                z = np.random.normal(0, 1, n)
                h_sample = post_mean + L @ z
                return h_sample

            except la.LinAlgError:
                print("Cholesky retry failed, using pseudoinverse.")
                post_cov = np.linalg.pinv(K_approx + noise_matrix + jitter)
                post_mean = post_cov @ residuals
                return np.random.multivariate_normal(post_mean, post_cov + jitter)

    def _sample_gamma(self, Z, Y, h, sigma2):
        """Efficient MCMC sampling for γ using Cholesky-based linear regression."""
        q = Z.shape[1]
        ZTZ = Z.T @ Z + self.reg_lambda * np.eye(q)
        ZTY = Z.T @ (Y - h)

        try:
            L = la.cholesky(ZTZ / sigma2, lower=True)
            posterior_variance = la.cho_solve((L, True), np.eye(q))
        except la.LinAlgError:
            posterior_variance = np.linalg.pinv(ZTZ / sigma2)

        posterior_mean = posterior_variance @ (ZTY / sigma2)
        return np.random.multivariate_normal(posterior_mean, posterior_variance)

    def _sample_sigma2(self, Y, Z, gamma, h):
        """Sample sigma^2 using inverse gamma distribution with regularization."""
        alpha_post = 0.5 * len(Y)
        beta_post = 0.5 * np.sum((Y - Z @ gamma - h) ** 2) + self.reg_lambda
        return invgamma.rvs(alpha_post, scale=beta_post)

    def fit(self, X, Z, Y):
        """
        Fits the BKMR model using efficient MCMC sampling.
        
        Parameters:
        - X: (n x p) exposure matrix.
        - Z: (n x q) covariate matrix.
        - Y: (n x 1) response variable.
        """
        n, p = X.shape
        q = Z.shape[1]  # Number of covariates

        # Store training data
        self.X_train = X
        self.Z_train = Z

        # Compute sparse kernel approximation
        K_approx = self._compute_sparse_kernel(X)

        # Initialize parameters
        h, gamma, sigma2 = self._initialize_params(n, q)

        # Storage for samples
        h_samples = np.zeros((self.num_samples, n))
        gamma_samples = np.zeros((self.num_samples, q))
        sigma2_samples = np.zeros(self.num_samples)

        # MCMC Sampling
        for i in range(self.num_samples):
            h = self._sample_h(X, Y, Z, gamma, sigma2, K_approx)
            gamma = self._sample_gamma(Z, Y, h, sigma2)
            sigma2 = self._sample_sigma2(Y, Z, gamma, h)

            h_samples[i, :] = h
            gamma_samples[i, :] = gamma
            sigma2_samples[i] = sigma2

        # Remove burn-in samples
        self.h_samples = h_samples[self.burn_in:, :]
        self.gamma_samples = gamma_samples[self.burn_in:, :]
        self.sigma2_samples = sigma2_samples[self.burn_in:]

        # Compute posterior means
        self.h_mean = np.mean(self.h_samples, axis=0)
        self.gamma_mean = np.mean(self.gamma_samples, axis=0)
        self.sigma2_mean = np.mean(self.sigma2_samples)

    def predict(self, X, Z):
        """
        Predicts Y using Nyström approximation.
        
        Parameters:
        - X: (m x p) new exposures.
        - Z: (m x q) new covariates.
        
        Returns:
        - Predicted Y.
        """
        if self.X_train is None or self.h_mean is None:
            raise ValueError("Model is not fitted. Call `fit()` first.")

        # Compute kernel function with training data
        K_new = self._rbf_kernel(X, self.X_train)
        h_pred = K_new @ self.h_mean  # Predict h(X)
        return Z @ self.gamma_mean + h_pred

    def get_posterior_samples(self):
        """Returns posterior samples for γ, h, and σ^2."""
        return self.gamma_samples, self.h_samples, self.sigma2_samples

    def get_posterior_means(self):
        """Returns posterior mean estimates for γ, h, and σ^2."""
        return self.gamma_mean, self.h_mean, self.sigma2_mean

    @property
    def gamma(self):
        """Returns the posterior mean of γ."""
        return self.gamma_mean