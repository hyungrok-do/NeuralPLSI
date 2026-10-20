"""Single-index neural model implemented entirely in the native C++ library."""
import functools
from numbers import Integral
from pathlib import Path

import numpy as np

from .base import (_SummaryMixin, run_parallel_bootstrap, validate_data,
                   validate_bootstrap, bootstrap_jobs)
from .native import NativeModel, FAMILIES, ACTIVATIONS, seed_value


def _refit_nplsi(Xb, Zb, yb, seed, settings, state, do_g, g_grid):
    model = NeuralPLSI(**settings)
    if state is None:
        model.fit(Xb, Zb, yb, random_state=seed)
    else:
        model.net = NativeModel.from_bytes(state)
        model.net.fit(Xb, Zb, yb, random_state=seed, reset=False)
    return model.beta, model.gamma, model.intercept_val, model.g_function(g_grid) if do_g else None


class NeuralPLSI(_SummaryMixin):
    """Fit g(X @ beta) + Z @ gamma using native C++ minibatch AdamW.

    C++ owns initialization, the network, gradients, optimizer, data splitting,
    shuffling, prediction and model serialization. Python handles array input,
    bootstrap orchestration and summary tables. CPU only, with float32 weights.
    """
    def __init__(self, family='continuous', max_epoch=200, batch_size=64,
                 learning_rate=1e-3, weight_decay=1e-4, weight_decay_beta=1e-5,
                 hidden_units=32, n_hidden_layers=2, grad_clip=1.0,
                 num_workers=0, add_intercept=False, activation='LeakyReLU',
                 initial=False, warmstart=True, device='cpu', backend='cpp'):
        if family not in FAMILIES:
            raise ValueError('family must be continuous, binary, or cox')
        if family == 'cox' and add_intercept:
            raise ValueError('A Cox partial likelihood cannot identify an intercept')
        if backend != 'cpp' or device not in ('cpu', 'auto'):
            raise ValueError('NeuralPLSI is C++ only: use backend="cpp" and device="cpu"')
        if activation not in ACTIVATIONS:
            raise ValueError('Supported activations: ELU, LeakyReLU, ReLU and Tanh')
        for name, value, minimum in [('max_epoch', max_epoch, 0), ('batch_size', batch_size, 1),
                                      ('hidden_units', hidden_units, 1), ('n_hidden_layers', n_hidden_layers, 1)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        for name, value in [('learning_rate', learning_rate), ('weight_decay', weight_decay),
                            ('weight_decay_beta', weight_decay_beta), ('grad_clip', grad_clip)]:
            if not np.isfinite(value) or value < 0:
                raise ValueError(f'{name} must be finite and nonnegative')
        if num_workers != 0:
            raise ValueError('num_workers is obsolete; use n_jobs for bootstrap parallelism')
        self.family, self.max_epoch, self.batch_size = family, int(max_epoch), int(batch_size)
        self.learning_rate, self.weight_decay, self.weight_decay_beta = learning_rate, weight_decay, weight_decay_beta
        self.hidden_units, self.n_hidden_layers, self.grad_clip = int(hidden_units), int(n_hidden_layers), grad_clip
        self.num_workers, self.add_intercept, self.activation = 0, bool(add_intercept), activation
        self.initial, self.warmstart, self.device, self.backend = bool(initial), bool(warmstart), 'cpu', 'cpp'
        self.net = None

    def get_params(self, deep=True):
        names = ('family', 'max_epoch', 'batch_size', 'learning_rate', 'weight_decay', 'weight_decay_beta',
                 'hidden_units', 'n_hidden_layers', 'grad_clip', 'num_workers', 'add_intercept',
                 'activation', 'initial', 'warmstart', 'device', 'backend')
        return {name: getattr(self, name) for name in names}

    def fit(self, X, Z, y, random_state=0):
        X, Z, y = validate_data(X, Z, y, self.family, dtype=np.float32)
        seed_value(random_state)
        options = {key: value for key, value in self.get_params().items()
                   if key not in ('device', 'backend', 'num_workers')}
        candidate = NativeModel(X.shape[1], Z.shape[1], random_state=random_state, **options)
        candidate.fit(X, Z, y, random_state=random_state)
        self._clear_inference()
        self.net = candidate
        return self

    def _ensure_fitted(self):
        if self.net is None:
            raise ValueError('Call fit() before prediction')
        return self.net

    def _ensure_dimensions(self, X, Z):
        net = self._ensure_fitted()
        if X.shape[1] != net.config[0] or Z.shape[1] != net.config[1]:
            raise ValueError('Input dimensions differ from the fitted model')

    @property
    def beta(self):
        return self._ensure_fitted().coefficients()[0]

    @property
    def gamma(self):
        return self._ensure_fitted().coefficients()[1]

    @property
    def intercept_val(self):
        return self._ensure_fitted().coefficients()[2] if self.add_intercept else None

    @property
    def fit_diagnostics(self):
        return self._ensure_fitted().fit_diagnostics.copy()

    def summary(self, include_beta=True, include_gamma=True, include_spline=False):
        self._ensure_fitted()
        blocks = {}
        for name, include in [('beta', include_beta), ('gamma', include_gamma)]:
            if include:
                blocks[name] = dict(coeff=getattr(self, name), se=getattr(self, name+'_se', None),
                                    lb=getattr(self, name+'_lb', None), ub=getattr(self, name+'_ub', None))
        return self._build_summary(blocks, ['beta', 'gamma'])

    def predict(self, X, Z, batch_size=128):
        return self._ensure_fitted().predict(X, Z, batch_size, mode=0)

    def predict_proba(self, X, Z, batch_size=128):
        return self._ensure_fitted().predict(X, Z, batch_size, mode=1)

    def predict_partial_hazard(self, X, Z, batch_size=128):
        return self._ensure_fitted().predict(X, Z, batch_size, mode=2)

    def predict_gxb(self, X, batch_size=128):
        return self._ensure_fitted().predict(X, batch_size=batch_size, mode=3)

    def g_function(self, x):
        return self._ensure_fitted().g_function(x)

    def save(self, path):
        """Save a portable native model, including training settings and orientation.

        Bootstrap samples/intervals are separate inference results and are not
        part of this prediction-model file.
        """
        self._ensure_fitted().save(Path(path))

    @classmethod
    def load(cls, path):
        net = NativeModel.load(Path(path))
        c, h = net.config, net.hyper
        model = cls(family=next(k for k, v in FAMILIES.items() if v == c[5]),
            activation=next(k for k, v in ACTIVATIONS.items() if v == c[6]),
            hidden_units=int(c[2]), n_hidden_layers=int(c[3]), add_intercept=bool(c[4]),
            batch_size=int(c[8]), max_epoch=int(c[9]), initial=bool(c[10]), warmstart=bool(c[11]),
            learning_rate=float(h[0]), weight_decay=float(h[1]), weight_decay_beta=float(h[2]), grad_clip=float(h[3]))
        model.net = net
        return model

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs='auto', quantile_method='linear'):
        """
        Perform bootstrap inference to estimate SE and CI for beta, gamma, and g(x).

        quantile_method selects NumPy's percentile interpolation: 'linear'
        preserves historical results, while 'weibull' uses order-statistic
        plotting positions k/(B+1). At small B this changes tail endpoints
        materially. It does not correct estimator bias or guarantee coverage.
        """
        if quantile_method not in ('linear', 'weibull'):
            raise ValueError("quantile_method must be 'linear' or 'weibull'")
        validate_bootstrap(n_samples, ci, random_state)
        X, Z, y = validate_data(X, Z, y, self.family, dtype=np.float32)
        if self.net is None:
            self.fit(X, Z, y, random_state=random_state)
        self._ensure_dimensions(X, Z)
        self._clear_inference()
        N, p, q = len(X), X.shape[1], Z.shape[1]
        n_jobs = bootstrap_jobs(n_jobs, n_samples)

        beta_samples = np.empty((n_samples, p))
        gamma_samples = np.empty((n_samples, q))
        intercept_samples = np.empty((n_samples, 1)) if self.add_intercept else None

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=np.float32).reshape(-1, 1)
            if not g_grid.size or not np.isfinite(g_grid).all():
                raise ValueError('g_grid must be nonempty and finite')
            g_samples = np.empty((n_samples, g_grid.shape[0]))
        else:
            g_grid = None

        state = self.net.to_bytes() if self.warmstart else None
        refit_func = functools.partial(_refit_nplsi, settings=self.get_params(), state=state,
                                       do_g=do_g, g_grid=g_grid)

        results = run_parallel_bootstrap(refit_func, X, Z, y, n_samples, random_state, n_jobs)
        for b, result in enumerate(results):
            beta_res, gamma_res, intercept_res, g_res = result
            beta_samples[b] = beta_res
            gamma_samples[b] = gamma_res
            if self.add_intercept and intercept_res is not None:
                intercept_samples[b] = intercept_res
            if do_g and g_res is not None:
                g_samples[b] = g_res

        arrays = [beta_samples, gamma_samples]
        if self.add_intercept:
            arrays.append(intercept_samples)
        if do_g:
            arrays.append(g_samples)
        if not all(np.isfinite(a).all() for a in arrays):
            raise FloatingPointError('Nonfinite bootstrap coefficient estimates')

        beta_hat, gamma_hat = self.beta, self.gamma
        intercept_hat = self.intercept_val

        self.beta_se = beta_samples.std(axis=0, ddof=1)
        self.gamma_se = gamma_samples.std(axis=0, ddof=1)
        self.beta_lb, self.beta_ub = self._percentile_ci(beta_samples, ci, method=quantile_method)
        self.gamma_lb, self.gamma_ub = self._percentile_ci(gamma_samples, ci, method=quantile_method)
        
        out = {
            "beta_hat": beta_hat, "beta_se": self.beta_se, "beta_lb": self.beta_lb, "beta_ub": self.beta_ub,
            "gamma_hat": gamma_hat, "gamma_se": self.gamma_se, "gamma_lb": self.gamma_lb, "gamma_ub": self.gamma_ub,
            "beta_samples": beta_samples, "gamma_samples": gamma_samples,
            "quantile_method": quantile_method, "ci": ci, "n_bootstrap": n_samples
        }
        
        if self.add_intercept:
            self.intercept_se = intercept_samples.std(axis=0, ddof=1)
            self.intercept_lb, self.intercept_ub = self._percentile_ci(intercept_samples, ci, method=quantile_method)
            out.update({
                "intercept_hat": intercept_hat, "intercept_se": self.intercept_se,
                "intercept_lb": self.intercept_lb, "intercept_ub": self.intercept_ub,
                "intercept_samples": intercept_samples
            })

        if do_g:
            g_mean = g_samples.mean(axis=0); g_se = g_samples.std(axis=0, ddof=1)
            g_lb, g_ub = self._percentile_ci(g_samples, ci, method=quantile_method)
            self.g_grid = g_grid.ravel(); self.g_grid_mean = g_mean; self.g_grid_se = g_se
            self.g_grid_lb, self.g_grid_ub = g_lb, g_ub; self._g_samples = g_samples
            out.update({"g_grid": self.g_grid, "g_mean": g_mean, "g_se": g_se, "g_lb": g_lb, "g_ub": g_ub,
                        "g_samples": g_samples})
        return out

