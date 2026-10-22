"""The public workflow must work even when a Torch import is impossible."""
import os
from pathlib import Path
import subprocess
import sys
import unittest


class NoTorchTest(unittest.TestCase):
    def test_native_workflow_with_torch_imports_blocked(self):
        script = r'''
import importlib.abc
import sys

class NoTorch(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'torch' or fullname.startswith('torch.'):
            raise AssertionError('Unexpected Torch import: ' + fullname)

sys.meta_path.insert(0, NoTorch())
import numpy as np
from models import NeuralPLSI
from simulation import simulate_data
from reproduce import main_simulation, benchmark_backends, nhanes
from reproduce import visualize_simulation, table_simulation_comparison

for family in ('continuous', 'binary', 'cox'):
    x, z, y, *_ = simulate_data(80, outcome=family, seed=7)
    model = NeuralPLSI(family=family, max_iter=2, hidden_units=4).fit(x, z, y)
    result = model.inference_bootstrap(x, z, y, n_samples=2, n_jobs=1, g_grid=[-1., 0., 1.])
    assert np.isfinite(model.predict(x, z)).all()
    assert np.isfinite(result['gamma_samples']).all()
    assert len(model.summary()) == x.shape[1] + z.shape[1]
assert not any(key == 'torch' or key.startswith('torch.') for key in sys.modules)
print('Native fits, bootstrap, prediction, summaries and report imports passed without Torch.')
'''
        env = dict(os.environ, OPENBLAS_NUM_THREADS='1', OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
        result = subprocess.run([sys.executable, '-c', script],
            cwd=Path(__file__).resolve().parents[1], env=env,
            text=True, capture_output=True, timeout=90)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == '__main__':
    unittest.main()
