import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from models import NeuralPLSI
from models.native import NativeModel, library, checked, loss_and_gradient
from native_reference import forward, loss


class NativeBackendTest(unittest.TestCase):
    def data(self, family, n=32):
        rng=np.random.default_rng(17)
        x,z=rng.normal(size=(n,3)).astype('float32'),rng.normal(size=(n,2)).astype('float32')
        y=x[:,0]+z[:,0]+rng.normal(size=n)
        if family=='binary': y=(y>0).astype(float)
        elif family=='cox': y=np.column_stack([rng.integers(1,8,n),rng.binomial(1,.7,n)])
        return x,z,y.astype('float32')

    def test_forward_loss_and_analytic_gradients_against_independent_reference(self):
        for family in ('continuous','binary','cox'):
            for activation in ('ELU','LeakyReLU','ReLU','Tanh'):
                with self.subTest(family=family,activation=activation):
                    x,z,y=self.data(family)
                    net=NativeModel(3,2,hidden_units=3,n_hidden_layers=2,
                        family=family,activation=activation,add_intercept=family!='cox',random_state=9)
                    weights=net.weights
                    net.set_weights(weights,sign=-1.)
                    expected,_=forward(net.config,weights,x,z,-1.)
                    np.testing.assert_allclose(net.predict(x,z),expected,rtol=2e-5,atol=2e-7)
                    value,gradient=loss_and_gradient(net,x,z,y)
                    np.testing.assert_allclose(value,loss(net.config,weights,x,z,y,-1.),rtol=2e-6,atol=2e-7)
                    # Differentiate an independent float64 scalar implementation.
                    numeric=np.empty(len(weights)); eps=1e-5
                    for j in range(len(weights)):
                        plus,minus=weights.astype(float),weights.astype(float)
                        plus[j]+=eps; minus[j]-=eps
                        numeric[j]=(loss(net.config,plus,x,z,y,-1.)-loss(net.config,minus,x,z,y,-1.))/(2*eps)
                    np.testing.assert_allclose(gradient,numeric,rtol=3e-4,atol=3e-6)

    def test_adamw_multiple_steps_match_explicit_update(self):
        net=NativeModel(3,2,hidden_units=4,n_hidden_layers=1,add_intercept=True,random_state=5)
        rng=np.random.default_rng(123)
        gradients=rng.normal(size=(5,len(net.weights))).astype('float32')
        for gamma_only in (False,True):
            weights=net.weights.copy(); expected=weights.astype(float)
            m,v=np.zeros_like(expected),np.zeros_like(expected)
            gamma=np.zeros(len(weights),dtype=bool); gamma[3:5]=True; gamma[-1]=True
            for t,gradient in enumerate(gradients.astype(float),1):
                active=gamma if gamma_only else np.ones(len(weights),dtype=bool)
                gradient=gradient.copy()
                gradient*=min(1.,float(net.hyper[3])/(np.linalg.norm(gradient[active])+1e-6))
                rate=np.where(gamma,.05,.001) if not gamma_only else np.full(len(weights),.01)
                decay=np.where(gamma,0.,1e-4); decay[:3]=1e-5
                m=.9*m+.1*gradient; v=.999*v+.001*gradient*gradient
                expected[active]*=1-rate[active]*decay[active]
                expected[active]-=rate[active]*(m[active]/(1-.9**t))/(np.sqrt(v[active]/(1-.999**t))+1e-8)
            checked(library().nplsi_adam_updates(net.config,net.hyper,weights,gradients,len(gradients),int(gamma_only)))
            np.testing.assert_allclose(weights,expected,rtol=2e-6,atol=2e-7)

    def test_all_families_fit_bootstrap_and_round_trip(self):
        for family in ('continuous','binary','cox'):
            with self.subTest(family=family):
                x,z,y=self.data(family,n=80)
                model=NeuralPLSI(family=family,max_epoch=8,hidden_units=6,batch_size=24,
                                 activation='ELU',add_intercept=family!='cox').fit(x,z,y,random_state=4)
                prediction=model.predict(x,z)
                np.testing.assert_allclose(prediction,model.predict_gxb(x)+z@model.gamma+
                                           (model.intercept_val if model.add_intercept else 0.),atol=2e-6)
                np.testing.assert_allclose(model.predict_gxb(x),model.g_function(x@model.beta),atol=2e-6)
                with tempfile.TemporaryDirectory() as temp:
                    path=Path(temp)/'model.nplsi'
                    model.save(path); restored=NeuralPLSI.load(path)
                    np.testing.assert_array_equal(prediction,restored.predict(x,z))
                    np.testing.assert_array_equal(prediction,pickle.loads(pickle.dumps(model)).predict(x,z))
                    self.assertEqual(model.fit_diagnostics,restored.fit_diagnostics)
                    draws=[m.inference_bootstrap(x,z,y,n_samples=2,random_state=4,n_jobs=1,g_grid=[-1.,0.,1.])
                           for m in (model,restored)]
                    for key in ('beta_samples','gamma_samples','g_samples'):
                        np.testing.assert_array_equal(draws[0][key],draws[1][key])
                    for broken in (b'',b'bad format',path.read_bytes()[:100]):
                        path.write_bytes(broken)
                        with self.assertRaises(RuntimeError): NeuralPLSI.load(path)

    def test_initialization_and_fit_are_reproducible_without_global_rng_changes(self):
        x,z,y=self.data('continuous',n=80)
        np.random.seed(17); before=np.random.get_state()
        a=NeuralPLSI(max_epoch=5).fit(x,z,y,random_state=7)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        b=NeuralPLSI(max_epoch=5).fit(x,z,y,random_state=7)
        np.testing.assert_array_equal(a.net.weights,b.net.weights)
        c=NeuralPLSI(max_epoch=5).fit(x,z,y,random_state=8)
        self.assertFalse(np.array_equal(a.net.weights,c.net.weights))

    def test_zero_covariates_and_empty_predictions(self):
        x,_,y=self.data('continuous',n=80)
        z=np.empty((len(x),0))
        model=NeuralPLSI(max_epoch=3,add_intercept=True).fit(x,z,y)
        self.assertEqual(model.gamma.size,0)
        self.assertEqual(model.predict(x[:0],z[:0]).shape,(0,))
        self.assertEqual(model.g_function([]).shape,(0,))
        self.assertTrue(np.isfinite(model.predict_gxb(x)).all())


if __name__=='__main__': unittest.main()
