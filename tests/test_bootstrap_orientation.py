import unittest
import numpy as np
import torch
from models import NeuralPLSI
from models.NeuralPLSI import _nPLSInet


class BootstrapOrientationTest(unittest.TestCase):
    def test_warm_start_preserves_link_for_both_orientations(self):
        """With zero training steps, a bootstrap copy must preserve g exactly."""
        torch.set_num_threads(1)
        rng=np.random.default_rng(123)
        X,Z,y=rng.normal(size=(50,8)),rng.normal(size=(50,3)),rng.normal(size=50)
        grid=np.linspace(-2,2,21)
        for flipped in [False,True]:
            with self.subTest(flipped=flipped):
                torch.manual_seed(19)
                m=NeuralPLSI(max_epoch=0,activation='ELU')
                m.device=torch.device('cpu')
                m.net=_nPLSInet(8,3,activation='ELU')
                with torch.no_grad():
                    m.net.x_input.weight[0,0]= -1 if flipped else 1
                m.net.resolve_sign_ambiguity()
                self.assertEqual(m.net.flip_sign,flipped)
                expected=m.g_function(grid)
                b=m.inference_bootstrap(X,Z,y,n_samples=2,g_grid=grid,n_jobs=1)
                np.testing.assert_allclose(b['g_mean'],expected,rtol=0,atol=1e-7)
                np.testing.assert_allclose(b['beta_samples'],np.tile(m.beta,(2,1)),rtol=0,atol=1e-7)


if __name__=='__main__':
    unittest.main()
