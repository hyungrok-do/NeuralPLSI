import unittest
import numpy as np
from models import NeuralPLSI
from models.native import NativeModel


class BootstrapOrientationTest(unittest.TestCase):
    def test_cold_refits_ignore_source_weights_and_orientation(self):
        """Every refit starts from a fresh seeded initialization, so the draws
        depend only on the data, the seeds and the configuration."""
        rng=np.random.default_rng(123)
        X,Z,y=rng.normal(size=(50,8)),rng.normal(size=(50,3)),rng.normal(size=50)
        grid=np.linspace(-2,2,21)
        draws=[]
        for flipped in [False,True]:
            with self.subTest(flipped=flipped):
                m=NeuralPLSI(max_iter=3,activation='ELU')
                m.net=NativeModel(8,3,activation='ELU',max_iter=3,random_state=19 if flipped else 7)
                weights=m.net.weights
                weights[0]= -1 if flipped else 1
                m.net.set_weights(weights,sign=1)
                m.net.resolve_sign_ambiguity()
                self.assertEqual(m.net.flip_sign,flipped)
                b=m.inference_bootstrap(X,Z,y,n_samples=3,g_grid=grid,n_jobs=1)
                self.assertTrue(np.all(b['beta_samples'][:,0]>=0))
                draws.append(b)
        for key in ('beta_samples','gamma_samples','g_samples'):
            np.testing.assert_array_equal(draws[0][key],draws[1][key])


if __name__=='__main__':
    unittest.main()
