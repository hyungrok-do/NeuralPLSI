"""NumPy/ctypes bindings to the standalone C++ NeuralPLSI and SplinePLSI implementations."""
import ctypes
import functools
import hashlib
import os
import platform
from pathlib import Path
import shlex
import subprocess
import tempfile
from numbers import Integral

import numpy as np

ACTIVATIONS = {'ELU': 0, 'LeakyReLU': 1, 'ReLU': 2, 'Tanh': 3}
FAMILIES = {'continuous': 0, 'binary': 1, 'cox': 2}
RNG_PROTOCOL = 'splitmix64_v1'
BOOTSTRAP_RNG_PROTOCOL = 'splitmix64_bootstrap_v1'


def build_library():
    """Build once per source/compiler configuration, safely across workers."""
    import fcntl
    sources = [Path(__file__).with_name(name) for name in ('core.cpp', 'spline.cpp')]
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off', '-pthread']
    try:
        version = subprocess.check_output(compiler+['--version'], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('NeuralPLSI needs a C++17 compiler. Install one or set CXX.') from exc
    code = b''.join(path.read_bytes() for path in sources+[Path(__file__).with_name('core.h')])
    digest = hashlib.sha256(code+version+repr((compiler, flags)).encode()+platform.machine().encode()+platform.system().encode()).hexdigest()[:20]
    cache = Path(os.environ.get('NEURALPLSI_CACHE_DIR', Path.home()/'.cache'/'neuralplsi'))
    cache.mkdir(parents=True, exist_ok=True)
    target = cache/f'core-{digest}.so'
    with (cache/f'core-{digest}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            temporary = None
            try:
                with tempfile.NamedTemporaryFile(dir=cache, suffix='.so', delete=False) as stream:
                    temporary = stream.name
                result = subprocess.run(compiler+flags+[str(path) for path in sources]+['-o', temporary],
                                        capture_output=True, text=True)
                if result.returncode:
                    raise RuntimeError('Native compilation failed:\n'+result.stderr)
                os.replace(temporary, target)
            finally:
                if temporary and os.path.exists(temporary):
                    os.unlink(temporary)
    return target


@functools.lru_cache(maxsize=1)
def library():
    lib = ctypes.CDLL(str(build_library()))
    fp = np.ctypeslib.ndpointer(dtype=np.float32, flags='C_CONTIGUOUS')
    ip = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
    dp = np.ctypeslib.ndpointer(dtype=np.float64, flags='C_CONTIGUOUS')
    pointer, integer, seed = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint64
    optional_dp = ctypes.POINTER(ctypes.c_double)
    size, chars = ctypes.c_size_t, ctypes.c_char_p
    signatures = {
        'nplsi_error': ([], chars),
        'nplsi_create': ([ip, fp, seed], pointer),
        'nplsi_destroy': ([pointer], None),
        'nplsi_split': ([integer, seed, ip], integer),
        'nplsi_fit': ([pointer, fp, fp, fp, integer, seed], integer),
        'nplsi_predict': ([pointer, fp, fp, integer, integer, integer, fp], integer),
        'nplsi_link': ([pointer, fp, integer, fp], integer),
        'nplsi_coefficients': ([pointer, fp, fp, fp], integer),
        'nplsi_get_config': ([pointer, ip, fp, dp], integer),
        'nplsi_get_weights': ([pointer, fp], integer),
        'nplsi_set_weights': ([pointer, fp, ctypes.c_float], integer),
        'nplsi_canonicalize': ([pointer], integer),
        'nplsi_bootstrap': ([pointer, fp, fp, fp, integer, integer, seed, integer,
                             fp, integer, ctypes.c_double, integer, dp, dp, ip], integer),
        'nplsi_bootstrap_workers': ([integer, integer], integer),
        'nplsi_bootstrap_indices': ([integer, seed, integer, ip], integer),
        'nplsi_bootstrap_summary': ([dp, integer, integer, ctypes.c_double, integer, dp], integer),
        'nplsi_state_size': ([pointer], size),
        'nplsi_export_state': ([pointer, pointer, size], integer),
        'nplsi_import_state': ([chars, size], pointer),
        'nplsi_save': ([pointer, chars], integer),
        'nplsi_load': ([chars], pointer),
        'nplsi_loss_grad': ([ip, fp, fp, fp, fp, fp, integer, integer, dp, fp], integer),
        'nplsi_adam_step': ([ip, fp, fp, fp, integer], integer),
        'nplsi_adam_updates': ([ip, fp, fp, fp, integer, integer], integer),
        'nplsi_outcome': ([fp, fp, integer, integer, dp, fp], integer),
        'nplsi_train': ([ip, fp, fp, fp, fp, fp, integer, fp, fp, fp, integer, ip, dp], integer),
        'splsi_error': ([], chars),
        'splsi_fit': ([ip, dp, dp, dp, dp, integer, seed, optional_dp, dp, dp, dp, dp, dp], integer),
        'splsi_profile': ([ip, dp, dp, dp, dp, integer, dp, dp, dp, dp, dp], integer),
        'splsi_basis': ([dp, integer, integer, dp, integer, dp], integer),
        'splsi_bootstrap': ([ip, dp, dp, dp, dp, integer, integer, seed, integer, dp, integer,
                             ctypes.c_double, integer, dp, dp, ip], integer),
    }
    for name, (arguments, result) in signatures.items():
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, result
    return lib


def checked(status):
    if status:
        raise RuntimeError(library().nplsi_error().decode())


def spline_checked(status):
    if status:
        raise RuntimeError(library().splsi_error().decode())


def seed_value(value):
    if isinstance(value, bool) or not isinstance(value, Integral) or not 0 <= value < 2**64:
        raise ValueError('random_state must be an integer in [0, 2**64)')
    return int(value)


def parameter_count(p, q, width, depth, intercept):
    return p+q+1+2*width+(depth-1)*(width*width+width)+width+1+int(intercept)


class NativeModel:
    """Own a native model handle. No Python network or automatic differentiation."""
    def __init__(self, p, q, hidden_units=32, n_hidden_layers=2, add_intercept=False,
                 family='continuous', activation='ELU', batch_size=64, max_epoch=200,
                 learning_rate=1e-3, weight_decay=1e-4, weight_decay_beta=1e-5,
                 grad_clip=1., initial=False, random_state=0):
        self._library = library()
        self._handle = None
        self.config = np.array([p, q, hidden_units, n_hidden_layers, int(add_intercept),
            FAMILIES[family], ACTIVATIONS[activation],
            parameter_count(p, q, hidden_units, n_hidden_layers, add_intercept),
            batch_size, max_epoch, int(initial)], dtype=np.int32)
        self.hyper = np.array([learning_rate, weight_decay, weight_decay_beta, grad_clip, 1.], dtype=np.float32)
        self._handle = self._library.nplsi_create(self.config, self.hyper, seed_value(random_state))
        self._check_handle()
        self._refresh()

    def _check_handle(self):
        if not self._handle:
            raise RuntimeError(self._library.nplsi_error().decode())

    def close(self):
        if getattr(self, '_handle', None):
            self._library.nplsi_destroy(self._handle)
            self._handle = None

    def __del__(self):
        self.close()

    def _refresh(self):
        self.config, self.hyper = np.empty(11, dtype=np.int32), np.empty(5, dtype=np.float32)
        diagnostics = np.empty(3, dtype=np.float64)
        checked(self._library.nplsi_get_config(self._handle, self.config, self.hyper, diagnostics))
        self.fit_diagnostics = dict(joint_epochs=int(diagnostics[0]), gamma_iterations=int(diagnostics[1]),
                                   validation_loss=float(diagnostics[2]))

    @classmethod
    def _from_handle(cls, handle):
        result = cls.__new__(cls)
        result._library, result._handle = library(), handle
        result._check_handle()
        result._refresh()
        return result

    def to_bytes(self):
        size = self._library.nplsi_state_size(self._handle)
        if not size:
            raise RuntimeError(self._library.nplsi_error().decode())
        buffer = ctypes.create_string_buffer(size)
        checked(self._library.nplsi_export_state(self._handle, buffer, size))
        return buffer.raw

    @classmethod
    def from_bytes(cls, data):
        if not isinstance(data, bytes):
            raise TypeError('Native model state must be bytes')
        return cls._from_handle(library().nplsi_import_state(data, len(data)))

    def __reduce__(self):
        return (type(self).from_bytes, (self.to_bytes(),))

    def save(self, path):
        checked(self._library.nplsi_save(self._handle, os.fsencode(path)))

    @classmethod
    def load(cls, path):
        return cls._from_handle(library().nplsi_load(os.fsencode(path)))

    @property
    def weights(self):
        weights = np.empty(int(self.config[7]), dtype=np.float32)
        checked(self._library.nplsi_get_weights(self._handle, weights))
        return weights

    def set_weights(self, weights, sign=None):
        weights = np.ascontiguousarray(weights, dtype=np.float32)
        if weights.shape != (int(self.config[7]),):
            raise ValueError('Wrong native weight dimensions')
        checked(self._library.nplsi_set_weights(self._handle, weights, float(self.hyper[4] if sign is None else sign)))
        self._refresh()

    @property
    def flip_sign(self):
        return self.hyper[4] < 0

    def resolve_sign_ambiguity(self):
        checked(self._library.nplsi_canonicalize(self._handle))
        self._refresh()

    def coefficients(self):
        beta, gamma, intercept = (np.empty(int(self.config[0]), dtype=np.float32),
                                  np.empty(int(self.config[1]), dtype=np.float32), np.empty(1, dtype=np.float32))
        checked(self._library.nplsi_coefficients(self._handle, beta, gamma, intercept))
        return beta, gamma, intercept if self.config[4] else None

    def fit(self, X, Z, y, random_state=0):
        from ..base import validate_data
        family = next(k for k, v in FAMILIES.items() if v == self.config[5])
        X, Z, y = validate_data(X, Z, y, family, dtype=np.float32)
        X, Z, y = [np.ascontiguousarray(a) for a in (X, Z, y)]
        if X.shape[1] != self.config[0] or Z.shape[1] != self.config[1]:
            raise ValueError('Fit dimensions differ from the native model')
        checked(self._library.nplsi_fit(self._handle, X, Z, y, len(X), seed_value(random_state)))
        self._refresh()
        return self

    def predict(self, X, Z=None, batch_size=128, mode=0):
        X = np.ascontiguousarray(X, dtype=np.float32)
        if (X.ndim != 2 or X.shape[1] != self.config[0] or not np.isfinite(X).all()):
            raise ValueError('X must be finite and match the fitted exposure dimensions')
        if mode == 3:
            Z = np.empty(0, dtype=np.float32)
        else:
            Z = np.ascontiguousarray(Z, dtype=np.float32)
            if Z.shape != (len(X), self.config[1]) or not np.isfinite(Z).all():
                raise ValueError('Z must be finite and match the fitted covariate dimensions')
        if isinstance(batch_size, bool) or not isinstance(batch_size, Integral) or batch_size < 1:
            raise ValueError('batch_size must be a positive integer')
        result = np.empty(len(X), dtype=np.float32)
        checked(self._library.nplsi_predict(self._handle, X, Z, len(X), int(batch_size), mode, result))
        return result

    def bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=.95,
                  g_grid=None, n_jobs='auto', quantile_method='linear'):
        """One native call for resampling, fitting, threads, SEs and intervals."""
        from ..base import validate_data
        family = next(k for k, v in FAMILIES.items() if v == self.config[5])
        options = bootstrap_options(n_samples, random_state, ci, n_jobs, quantile_method)
        X, Z, y = [np.ascontiguousarray(a) for a in validate_data(X, Z, y, family, dtype=np.float32)]
        if X.shape[1] != self.config[0] or Z.shape[1] != self.config[1]:
            raise ValueError('Bootstrap dimensions differ from the native model')
        grid = link_grid(g_grid)
        columns = int(self.config[0])+int(self.config[1])+int(self.config[4])+len(grid)
        if columns >= 2**31:
            raise ValueError('Bootstrap output has too many columns')
        draws = np.empty((n_samples, columns), dtype=np.float64)
        summary = np.empty((4, columns), dtype=np.float64)
        workers = np.empty(1, dtype=np.int32)
        jobs, method = options
        checked(self._library.nplsi_bootstrap(self._handle, X, Z, y, len(X), n_samples,
            seed_value(random_state), jobs, grid, len(grid), float(ci),
            method, draws, summary, workers))
        return draws, summary, int(workers[0])

    def g_function(self, grid):
        grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float32).reshape(-1))
        if not np.isfinite(grid).all():
            raise ValueError('Link grid must be finite')
        result = np.empty(len(grid), dtype=np.float32)
        checked(self._library.nplsi_link(self._handle, grid, len(grid), result))
        return result


def loss_and_gradient(net, X, Z, y, family=None, anchor=True):
    """Diagnostic entry point for independent numerical gradient checks."""
    from ..base import validate_data
    own_family = next(k for k, v in FAMILIES.items() if v == net.config[5])
    if family is not None and family != own_family:
        raise ValueError('Loss family differs from the model')
    arrays = [np.ascontiguousarray(a) for a in validate_data(X, Z, y, own_family, dtype=np.float32)]
    if arrays[0].shape[1] != net.config[0] or arrays[1].shape[1] != net.config[1]:
        raise ValueError('Wrong model dimensions')
    weights = net.weights
    loss, gradient = np.zeros(1, dtype=np.float64), np.zeros_like(weights)
    checked(library().nplsi_loss_grad(net.config, net.hyper, weights, *arrays, len(X), int(anchor), loss, gradient))
    return float(loss[0]), gradient


def outcome_loss_gradient(risk, y, family):
    risk = np.ascontiguousarray(np.asarray(risk, dtype=np.float32).reshape(-1))
    y = np.ascontiguousarray(y, dtype=np.float32)
    shape = (len(risk), 2) if family == 'cox' else (len(risk),)
    if y.shape != shape or not len(risk) or not np.isfinite(risk).all() or not np.isfinite(y).all():
        raise ValueError('Invalid loss inputs')
    loss, gradient = np.empty(1, dtype=np.float64), np.empty_like(risk)
    checked(library().nplsi_outcome(risk, y, len(risk), FAMILIES[family], loss, gradient))
    return float(loss[0]), gradient


def training_validation_indices(n, random_state=0):
    """Inspect the native 80/20 split without fitting or changing any RNG state."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 2:
        raise ValueError('Splitting requires at least two rows')
    order = np.empty(n, dtype=np.int32)
    checked(library().nplsi_split(n, seed_value(random_state), order))
    validation_size = (n+4)//5
    return order[validation_size:], order[:validation_size]


def bootstrap_options(n_samples, random_state, ci, n_jobs, quantile_method):
    """Check scalar inputs before allocation, fitting or conversion to C integers."""
    if quantile_method not in ('linear', 'weibull'):
        raise ValueError("quantile_method must be 'linear' or 'weibull'")
    if isinstance(n_samples, bool) or not isinstance(n_samples, Integral) or not 2 <= n_samples < 2**31:
        raise ValueError('n_samples must be an integer in [2, 2**31)')
    if not 0 < ci < 1:
        raise ValueError('ci must lie strictly between zero and one')
    seed = seed_value(random_state)
    if seed + 1337 + int(n_samples)-1 >= 2**64:
        raise ValueError('Bootstrap fit seeds would overflow uint64')
    if n_jobs == 'auto':
        n_jobs = -1
    if isinstance(n_jobs, bool) or not isinstance(n_jobs, Integral) or not -2**31 <= n_jobs < 2**31 or n_jobs == 0:
        raise ValueError('n_jobs must be a nonzero 32-bit integer or auto')
    return int(n_jobs), int(quantile_method == 'weibull')


def link_grid(grid):
    if grid is None:
        return np.empty(0, dtype=np.float32)
    grid = np.ascontiguousarray(np.asarray(grid, dtype=np.float32).reshape(-1))
    if not grid.size or not np.isfinite(grid).all():
        raise ValueError('g_grid must be nonempty and finite')
    return grid


def bootstrap_indices(n, random_state=0, replicate=0):
    """Inspect a native resample for diagnostics; production bootstrap stays in C++."""
    if isinstance(n, bool) or not isinstance(n, Integral) or not 2 <= n < 2**31:
        raise ValueError('n must be an integer in [2, 2**31)')
    if isinstance(replicate, bool) or not isinstance(replicate, Integral) or not 0 <= replicate < 2**31:
        raise ValueError('replicate must be a nonnegative 32-bit integer')
    seed = seed_value(random_state)
    if seed+int(replicate) >= 2**64:
        raise ValueError('Bootstrap resampling seed would overflow uint64')
    indices = np.empty(n, dtype=np.int32)
    checked(library().nplsi_bootstrap_indices(n, seed, replicate, indices))
    return indices


def bootstrap_summary(draws, ci=.95, quantile_method='linear'):
    """Diagnostic entry point to independently check native summaries."""
    draws = np.ascontiguousarray(draws, dtype=np.float64)
    if draws.ndim != 2 or draws.shape[1] < 1 or draws.shape[1] >= 2**31:
        raise ValueError('draws must be a matrix with at least one column')
    _, method = bootstrap_options(len(draws), 0, ci, 1, quantile_method)
    summary = np.empty((4, draws.shape[1]), dtype=np.float64)
    checked(library().nplsi_bootstrap_summary(draws, *draws.shape, float(ci), method, summary))
    return summary
