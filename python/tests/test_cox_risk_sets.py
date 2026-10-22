import unittest
import numpy as np
from scipy.special import logsumexp
from neuralplsi.native import outcome_loss_gradient


class CoxRiskSetTest(unittest.TestCase):
    def test_loss_and_score_match_complete_breslow_risk_sets(self):
        for times in ([4., 3., 2., 1.], [4., 3., 3., 1.], [1., 1., 1., 1.]):
            r=np.array([.4, -.3, .8, .1], dtype=np.float32)
            t=np.array(times); e=np.array([1., 1., 1., 0.])
            reference, score=0., np.zeros(4)
            for i in range(4):
                if not e[i]:
                    continue
                mask=t>=t[i]
                logden=logsumexp(r[mask].astype(float))
                reference+=logden-r[i]
                score[mask]+=np.exp(r[mask].astype(float)-logden)
                score[i]-=1
            actual, gradient=outcome_loss_gradient(r,np.column_stack([t,e]),'cox')
            np.testing.assert_allclose(actual,reference/e.sum(),rtol=1e-7)
            np.testing.assert_allclose(gradient,score/e.sum(),atol=2e-8)
            order=np.array([2,0,3,1])
            permuted, pg=outcome_loss_gradient(r[order],np.column_stack([t,e])[order],'cox')
            self.assertEqual(actual,permuted)
            np.testing.assert_allclose(pg,gradient[order],atol=1e-8)
