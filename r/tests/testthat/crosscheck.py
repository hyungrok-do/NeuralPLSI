"""Fits, bootstraps and model files of the Python package, for test-python.R.

python crosscheck.py <folder> fit   fits each <family>.{x,z,y} and writes the results
python crosscheck.py <folder> load  predicts with each <family>.r.nplsi saved by R
Arrays are little-endian float64 files, row-major.
"""
import sys
from pathlib import Path

import numpy as np

from neuralplsi import NeuralPLSI, SplinePLSI

folder, action = Path(sys.argv[1]), sys.argv[2]


def read(name, shape):
    return np.fromfile(folder/name, dtype='<f8').reshape(shape)


def write(name, values):
    np.ascontiguousarray(values, dtype='<f8').tofile(folder/name)


for family in ('continuous', 'binary', 'cox'):
    n, p, q = (int(v) for v in (folder/f'{family}.dims').read_text().split())
    x, z = read(f'{family}.x', (n, p)), read(f'{family}.z', (n, q))
    y = read(f'{family}.y', (n, 2) if family == 'cox' else (n,))
    if action == 'load':
        write(f'{family}.r_link', NeuralPLSI.load(folder/f'{family}.r.nplsi').predict(x, z))
        continue
    if family == 'continuous':
        model = NeuralPLSI().fit(x, z, y, random_state=7)
        write('default.coef', np.concatenate([model.beta, model.gamma]))
    model = NeuralPLSI(family=family, hidden_units=8, max_iter=300).fit(x, z, y, random_state=7)
    write(f'{family}.coef', np.concatenate([model.beta, model.gamma]))
    write(f'{family}.link', model.predict(x, z))
    model.save(folder/f'{family}.py.nplsi')
    boot = model.inference_bootstrap(x, z, y, n_samples=4, random_state=3, n_jobs=2)
    write(f'{family}.draws', np.hstack([boot['beta_samples'], boot['gamma_samples']]))
    spline = SplinePLSI(family=family).fit(x, z, y, random_state=7)
    write(f'{family}.spline', np.concatenate([spline.beta, spline.gamma, spline.spline_coeffs]))
    write(f'{family}.spline_link', spline.predict(x, z))
    write(f'{family}.spline_g', spline.g_function(np.arange(-30, 31)/10))
    boot = spline.inference_bootstrap(x, z, y, n_samples=4, random_state=3, n_jobs=2)
    write(f'{family}.spline_draws', np.hstack([boot['beta_samples'], boot['gamma_samples']]))
