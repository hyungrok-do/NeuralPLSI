import unittest
import torch
from models.NeuralPLSI import CoxCCLoss


class CoxRiskSetTest(unittest.TestCase):
    def test_loss_matches_explicit_breslow_risk_sets(self):
        for durations in [[4.,3.,2.,1.],[4.,3.,3.,1.]]:
            with self.subTest(durations=durations):
                r=torch.tensor([.4,-.3,.8,.1],dtype=torch.float64,requires_grad=True)
                d=torch.tensor(durations,dtype=torch.float64)
                e=torch.tensor([1.,1.,1.,0.],dtype=torch.float64)
                y=torch.column_stack([d,e])
                reference=-sum(e[i]*(r[i]-torch.logsumexp(r[d>=d[i]],dim=0))
                               for i in range(len(d)))/e.sum()
                actual=CoxCCLoss()(r,y)
                torch.testing.assert_close(actual,reference)
                ref_grad=torch.autograd.grad(reference,r,retain_graph=True)[0]
                actual_grad=torch.autograd.grad(actual,r)[0]
                torch.testing.assert_close(actual_grad,ref_grad)
                order=torch.tensor([2,0,3,1])
                torch.testing.assert_close(CoxCCLoss()(r[order],y[order]),reference)
                torch.testing.assert_close(CoxCCLoss(presorted=True)(r,y),reference)


if __name__=='__main__':
    unittest.main()
