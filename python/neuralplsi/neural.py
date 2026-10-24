"""Single-index neural model implemented entirely in the native C++ library."""
from numbers import Integral
from pathlib import Path

import numpy as np

from .base import _SummaryMixin, validate_data
from .native import (NativeModel, FAMILIES, ACTIVATIONS, OPTIMIZERS, seed_value,
                     bootstrap_options, link_grid, BOOTSTRAP_RNG_PROTOCOL)



class NeuralPLSI(_SummaryMixin):
    """Fit g(X @ beta) + Z @ gamma using native C++ full-batch L-BFGS.

    C++ owns initialization, the network, gradients, optimizer, prediction,
    serialization and the entire bootstrap workflow. Python handles array input
    and summary tables. CPU only, with float32 weights.

    Each fit uses all rows, with no validation split. L-BFGS minimizes the
    outcome loss plus the g(0)^2 anchor and weight_decay/2 times the squared
    link-network weights, renormalizing beta to unit length after every step,
    until the objective stops decreasing or max_iter iterations. With beta and
    the link fixed, gamma and the optional intercept are then fitted exactly by
    deterministic Newton iterations, with g(X @ beta) as an offset.

    optimizer='adam' is a baseline for typical neural-network training, not a
    recommended estimator: AdamW (learning_rate, decoupled weight_decay on the
    link network) on minibatches of batch_size rows from a random 80% of the
    data, for at most max_iter epochs, stopping after patience epochs without
    improvement in the loss on the other 20% and restoring the best
    checkpoint. Gamma is not refitted. optimizer='adam_exact_gamma' is the same
    AdamW fit followed by the exact gamma fit of L-BFGS on all rows, an
    ablation that separates the two parts of typical training.
    """
    def __init__(self, family='continuous', max_iter=5000, weight_decay=1e-3,
                 hidden_units=32, n_hidden_layers=2,
                 num_workers=0, add_intercept=False, activation='ELU',
                 initial=False, optimizer='lbfgs', batch_size=64, learning_rate=1e-3,
                 patience=10, device='cpu', backend='cpp'):
        if family not in FAMILIES:
            raise ValueError('family must be continuous, binary, or cox')
        if family == 'cox' and add_intercept:
            raise ValueError('A Cox partial likelihood cannot identify an intercept')
        if backend != 'cpp' or device not in ('cpu', 'auto'):
            raise ValueError('NeuralPLSI is C++ only: use backend="cpp" and device="cpu"')
        if activation not in ACTIVATIONS:
            raise ValueError('Supported activations: ELU, LeakyReLU, ReLU and Tanh')
        if optimizer not in OPTIMIZERS:
            raise ValueError("optimizer must be 'lbfgs', 'adam' or 'adam_exact_gamma'")
        for name, value, minimum in [('max_iter', max_iter, 0), ('hidden_units', hidden_units, 1),
                                      ('n_hidden_layers', n_hidden_layers, 1),
                                      ('batch_size', batch_size, 1), ('patience', patience, 1)]:
            if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
                raise ValueError(f'{name} must be an integer >= {minimum}')
        if not np.isfinite(weight_decay) or weight_decay < 0:
            raise ValueError('weight_decay must be finite and nonnegative')
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError('learning_rate must be finite and positive')
        if num_workers != 0:
            raise ValueError('num_workers is obsolete; use n_jobs for bootstrap parallelism')
        self.family, self.max_iter, self.weight_decay = family, int(max_iter), weight_decay
        self.hidden_units, self.n_hidden_layers = int(hidden_units), int(n_hidden_layers)
        self.num_workers, self.add_intercept, self.activation = 0, bool(add_intercept), activation
        self.initial, self.device, self.backend = bool(initial), 'cpu', 'cpp'
        self.optimizer, self.batch_size, self.patience = optimizer, int(batch_size), int(patience)
        self.learning_rate = learning_rate
        self.net = None

    def get_params(self, deep=True):
        names = ('family', 'max_iter', 'weight_decay', 'hidden_units', 'n_hidden_layers',
                 'num_workers', 'add_intercept', 'activation', 'initial', 'optimizer',
                 'batch_size', 'learning_rate', 'patience', 'device', 'backend')
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
            max_iter=int(c[8]), initial=bool(c[9]), weight_decay=float(h[0]),
            optimizer=next(k for k, v in OPTIMIZERS.items() if v == c[10]),
            batch_size=int(c[11]), patience=int(c[12]), learning_rate=float(h[2]))
        model.net = net
        return model

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95,
                            g_grid=None, n_jobs='auto', quantile_method='linear'):
        """Pairs bootstrap performed entirely in C++ with native worker threads.

        Resampling, independent model copies, refits, link evaluation, sample
        SDs and percentile intervals use one native call. n_jobs limits C++
        threads, capped by CPU affinity, cgroup quota, SLURM allocation and B.
        Replicate seeds are independent of worker scheduling. A failed draw
        raises an error and invalidates the whole bootstrap result.
        Each refit starts from a fresh seeded initialization (cold start) and
        uses the same training and stopping rule (and, for L-BFGS, the same
        exact gamma fit) as the original fit. Warm starts from the original fit were removed: refits
        inherited its optimizer error, so the bootstrap understated the
        variability of beta.

        'linear' and 'weibull' choose percentile interpolation. They do not
        change the draws or correct estimator bias. The native bootstrap RNG
        differs from the former NumPy resampling protocol.
        """
        bootstrap_options(n_samples, random_state, ci, n_jobs, quantile_method)
        grid = link_grid(g_grid)
        X, Z, y = validate_data(X, Z, y, self.family, dtype=np.float32)
        if self.net is None:
            self.fit(X, Z, y, random_state=random_state)
        self._ensure_dimensions(X, Z)
        self._clear_inference()
        draws, summary, workers = self.net.bootstrap(X, Z, y, n_samples=n_samples,
            random_state=random_state, ci=ci, g_grid=grid if g_grid is not None else None,
            n_jobs=n_jobs, quantile_method=quantile_method)
        out = dict(beta_hat=self.beta, gamma_hat=self.gamma,
                   quantile_method=quantile_method, ci=ci, n_bootstrap=n_samples,
                   bootstrap_engine='cpp', bootstrap_rng=BOOTSTRAP_RNG_PROTOCOL,
                   n_workers=workers)
        blocks = [('beta', X.shape[1]), ('gamma', Z.shape[1])]
        if self.add_intercept:
            blocks.append(('intercept', 1))
            out['intercept_hat'] = self.intercept_val
        if g_grid is not None:
            blocks.append(('g', len(grid)))
        position = 0
        for block, count in blocks:
            columns = slice(position, position+count)
            out[block+'_samples'] = draws[:, columns]
            for row, name in enumerate(('mean', 'se', 'lb', 'ub')):
                values = summary[row, columns]
                out[block+'_'+name] = values
                if block == 'g':
                    setattr(self, 'g_grid_'+name, values)
                elif name != 'mean':
                    setattr(self, block+'_'+name, values)
            position += count
        if g_grid is not None:
            self.g_grid = grid
            self._g_samples = out['g_samples']
            out['g_grid'] = grid
        return out
