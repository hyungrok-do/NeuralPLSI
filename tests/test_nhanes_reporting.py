import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from reproduce import nhanes, nhanes_comparison


class ReportModel:
    """Deliberately returns intervals that exclude the point estimate."""
    observed_sizes = []
    def __init__(self, *args, **kwargs):
        pass

    def fit(self, x, z, y, **kwargs):
        assert all(np.isfinite(a).all() for a in (x, z, y))
        self.observed_sizes.append(len(y))
        self.beta = np.ones(x.shape[1])/np.sqrt(x.shape[1])
        self.gamma = np.full(z.shape[1], .1)
        self.intercept_val = np.array([1.])

    def predict_gxb(self, x):
        return 3*(x@self.beta)

    def g_function(self, grid):
        return 3*np.asarray(grid)

    def inference_bootstrap(self, x, z, y, g_grid, **kwargs):
        out = {}
        for block, estimate in [('beta', self.beta), ('gamma', self.gamma), ('intercept', self.intercept_val)]:
            out[block+'_hat'] = estimate
            out[block+'_se'] = np.full_like(estimate, .123)
            out[block+'_lb'], out[block+'_ub'] = estimate+.2, estimate+.4
            out[block+'_samples'] = np.array([estimate+.2, estimate+.4])
            setattr(self, block+'_se', out[block+'_se'])
            setattr(self, block+'_lb', out[block+'_lb'])
            setattr(self, block+'_ub', out[block+'_ub'])
        out['g_samples'] = np.array([3*np.asarray(g_grid), 3*np.asarray(g_grid)+.2])
        out['g_mean'] = self.g_grid_mean = out['g_samples'].mean(axis=0)
        out['g_lb'], out['g_ub'] = np.percentile(out['g_samples'], [2.5, 97.5], axis=0)
        self.g_grid_lb, self.g_grid_ub = out['g_lb'], out['g_ub']
        return out


class NhanesReportingTest(unittest.TestCase):
    def test_complete_cases_se_and_exact_interval_endpoints(self):
        names = ['a7.a.Tocopherol', 'a6.g.tocopherol', 'a4.Retinyl.stearate', 'a5.Retinol',
                 'a20.3.3.4.4.5.pncb', 'a17.PCB194', 'a22.2.3.4.6.7.8.hxcdf', 'a1.trans.b.carotene']
        names += [f'exposure_{j}' for j in range(14)]
        rng = np.random.default_rng(3)
        df = pd.DataFrame({'age': rng.uniform(20, 70, 30), 'sex': np.tile([1, 2], 15)})
        for name in names:
            df[name] = rng.uniform(.5, 2, 30)
        df['race'], df['triglyceride'] = np.tile(np.arange(1, 6), 6), rng.uniform(1, 3, 30)
        df.loc[0, names[0]], df.loc[1, 'race'], df.loc[2, 'sex'] = np.nan, np.nan, 9
        df.loc[3, 'triglyceride'], df.loc[4, 'age'] = -1., np.nan
        cwd = os.getcwd()
        ReportModel.observed_sizes = []
        try:
            with tempfile.TemporaryDirectory() as temp:
                os.chdir(temp)
                with patch.object(pd, 'read_csv', return_value=df.copy()), patch.object(nhanes, 'NeuralPLSI', ReportModel):
                    nhanes.main()
                result = pd.read_csv('output/nhanes_nplsi_results.csv')
                np.testing.assert_allclose(result['se'], .123)
                np.testing.assert_allclose(result['lb'], result['coefs']+.2)
                with patch.object(pd, 'read_csv', return_value=df.copy()), patch.object(nhanes_comparison, 'NeuralPLSI', ReportModel), patch.object(nhanes_comparison, 'SplinePLSI', ReportModel):
                    nhanes_comparison.main()
                comparison = pd.read_csv('output/nhanes/nhanes_comparison.csv')
                np.testing.assert_allclose(comparison.Neural_LB, comparison.Neural_Est+.2)
                np.testing.assert_allclose(comparison.PLSI_UB, comparison.PLSI_Est+.4)
                np.testing.assert_allclose(comparison.Neural_SE, .123)
                self.assertEqual(ReportModel.observed_sizes, [25, 25, 25])
        finally:
            os.chdir(cwd)
            plt.close('all')


if __name__ == '__main__':
    unittest.main()
