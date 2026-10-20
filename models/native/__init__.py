"""Optional C++ CPU training, with no third-party C++ dependencies.

The public model still uses PyTorch for initialization, serialization and
prediction. All training loops and gradients run inside the native library.
"""
import ctypes
import functools
import hashlib
import os
import platform
from pathlib import Path
import shlex
import subprocess
import tempfile

import numpy as np
import torch
from sklearn.model_selection import train_test_split

ACTIVATIONS = {'ELU': 0, 'LeakyReLU': 1, 'ReLU': 2, 'Tanh': 3}
FAMILIES = {'continuous': 0, 'binary': 1, 'cox': 2}


def build_library():
    """Build once per source/compiler configuration, safely across workers."""
    import fcntl
    source = Path(__file__).with_name('core.cpp')
    compiler = shlex.split(os.environ.get('CXX', 'g++'))
    flags = ['-O3', '-std=c++17', '-shared', '-fPIC', '-ffp-contract=off']
    try:
        version = subprocess.check_output(compiler+['--version'], stderr=subprocess.STDOUT)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RuntimeError('The cpp backend needs a C++17 compiler. Set CXX or use backend="torch".') from exc
    digest = hashlib.sha256(source.read_bytes()+version+repr(flags).encode()+platform.machine().encode()+platform.system().encode()).hexdigest()[:20]
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
                result = subprocess.run(compiler+flags+[str(source), '-o', temporary],
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
    integer = ctypes.c_int
    lib.nplsi_error.restype = ctypes.c_char_p
    lib.nplsi_loss_grad.argtypes = [ip, fp, fp, fp, fp, fp, integer, integer, dp, fp]
    lib.nplsi_adam_step.argtypes = [ip, fp, fp, fp, integer]
    lib.nplsi_train.argtypes = [ip, fp, fp, fp, fp, fp, integer, fp, fp, fp, integer, ip, dp]
    for name in ('nplsi_loss_grad', 'nplsi_adam_step', 'nplsi_train'):
        getattr(lib, name).restype = integer
    return lib


def checked(status):
    if status:
        raise RuntimeError(library().nplsi_error().decode())


def parameters(net):
    """Explicit layout, independent of nn.Module's root-parameter ordering."""
    params = [net.x_input.weight, net.z_input.weight, net.g_skip.weight]
    for layer in net.g_network:
        if isinstance(layer, torch.nn.Linear):
            params.extend([layer.weight, layer.bias])
    if net.add_intercept:
        params.append(net.intercept)
    return params


def pack(net):
    return np.concatenate([p.detach().cpu().numpy().ravel() for p in parameters(net)]).astype(np.float32)


def unpack(net, weights):
    offset = 0
    with torch.no_grad():
        for param in parameters(net):
            param.copy_(torch.from_numpy(weights[offset:offset+param.numel()].reshape(param.shape)))
            offset += param.numel()
    if offset != len(weights):
        raise ValueError('Wrong native parameter count')


def configuration(net, family, batch_size=64, max_epoch=200):
    if net.n_classes != 1 or any(p.dtype != torch.float32 or p.device.type != 'cpu' for p in parameters(net)):
        raise ValueError('The cpp backend supports scalar float32 models on CPU')
    activations = [layer for layer in net.g_network if not isinstance(layer, torch.nn.Linear)]
    name = type(activations[0]).__name__
    if name not in ACTIVATIONS or any(type(a).__name__ != name for a in activations):
        raise ValueError('The cpp backend supports ELU, LeakyReLU, ReLU and Tanh')
    return np.array([net.x_input.in_features, net.z_input.in_features,
        net.g_network[0].out_features, len(activations), net.add_intercept,
        FAMILIES[family], ACTIVATIONS[name], sum(p.numel() for p in parameters(net)),
        batch_size, max_epoch], dtype=np.int32)


def hyperparameters(net, learning_rate=1e-3, weight_decay=1e-4, weight_decay_beta=1e-5, grad_clip=1.):
    return np.array([learning_rate, weight_decay, weight_decay_beta, grad_clip,
                     float(net._index_sign)], dtype=np.float32)


def loss_and_gradient(net, X, Z, y, family, anchor=True):
    """Diagnostic API for numerical checks of the independent gradient code."""
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in (X, Z, y)]
    weights = pack(net)
    loss, gradient = np.zeros(1, dtype=np.float64), np.zeros_like(weights)
    checked(library().nplsi_loss_grad(configuration(net, family), hyperparameters(net), weights,
                                     *arrays, len(X), int(anchor), loss, gradient))
    return float(loss[0]), gradient


def train(net, X, Z, y, family, batch_size=64, max_epoch=200,
          learning_rate=1e-3, weight_decay=1e-4, weight_decay_beta=1e-5,
          grad_clip=1., random_state=0):
    config = configuration(net, family, batch_size, max_epoch)
    hyper = hyperparameters(net, learning_rate, weight_decay, weight_decay_beta, grad_clip)
    tx, vx, tz, vz, ty, vy = train_test_split(X, Z, y, test_size=.2, random_state=random_state)
    if family == 'cox':
        if ty[:, 1].sum() == 0 or vy[:, 1].sum() == 0:
            raise ValueError('Both Cox training and validation partitions need at least one event')
        order = torch.argsort(torch.from_numpy(vy[:, 0].copy()), descending=True).numpy()
        vx, vz, vy = vx[order], vz[order], vy[order]
    arrays = [np.ascontiguousarray(a, dtype=np.float32) for a in (tx, tz, ty, vx, vz, vy)]
    rng = np.random.RandomState(random_state)
    permutations = np.array([rng.permutation(len(tx)) for _ in range(max_epoch+min(max_epoch, 50))],
                            dtype=np.int32).reshape(-1, len(tx))
    weights, diagnostics = pack(net), np.empty(3, dtype=np.float64)
    checked(library().nplsi_train(config, hyper, weights, *arrays[:3], len(tx),
                                  *arrays[3:], len(vx), permutations, diagnostics))
    unpack(net, weights)
    net.resolve_sign_ambiguity()
    net.fit_diagnostics = dict(joint_epochs=int(diagnostics[0]), gamma_epochs=int(diagnostics[1]),
                               validation_loss=float(diagnostics[2]))
    return net
