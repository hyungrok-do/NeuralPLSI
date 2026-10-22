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

    def test_backprop_supports_different_depths_and_widths(self):
        for width, depth in ((1, 1), (5, 3), (7, 4)):
            for family in ('continuous', 'binary', 'cox'):
                with self.subTest(width=width, depth=depth, family=family):
                    x, z, y = self.data(family, n=16)
                    net = NativeModel(3, 2, hidden_units=width, n_hidden_layers=depth,
                                      family=family, activation='Tanh', random_state=9)
                    value, gradient = loss_and_gradient(net, x, z, y)
                    weights = net.weights.astype(float)
                    np.testing.assert_allclose(value, loss(net.config, weights, x, z, y), atol=2e-6)
                    numeric = np.empty(len(weights))
                    for j in range(len(weights)):
                        plus, minus = weights.copy(), weights.copy()
                        plus[j] += 1e-5
                        minus[j] -= 1e-5
                        numeric[j] = (loss(net.config, plus, x, z, y)-loss(net.config, minus, x, z, y))/2e-5
                    np.testing.assert_allclose(gradient, numeric, rtol=3e-4, atol=3e-6)

    def test_packed_neuron_tiles_preserve_predictions_and_gradients(self):
        # Complete tiles, both sides of a boundary, and multiple tiles.
        for width, family in ((31, 'continuous'), (32, 'binary'), (33, 'cox'), (65, 'continuous')):
            for activation in ('ELU', 'LeakyReLU', 'ReLU', 'Tanh'):
                with self.subTest(width=width, family=family, activation=activation):
                    x, z, y = self.data(family, n=259)
                    net = NativeModel(3, 2, hidden_units=width, n_hidden_layers=3,
                        family=family, activation=activation, add_intercept=family!='cox', random_state=9)
                    net.set_weights(net.weights, sign=-1.)
                    weights = net.weights.astype(float)
                    expected, _ = forward(net.config, weights, x, z, -1.)
                    actual = net.predict(x, z, batch_size=256)
                    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=3e-7)
                    for batch in (1, 31, 1024):
                        np.testing.assert_array_equal(actual, net.predict(x, z, batch_size=batch))
                    # Independent scalar finite differences of beta and weights
                    # spanning output-tile boundaries check the full gradient.
                    x, z, y = x[:17], z[:17], y[:17]
                    _, gradient = loss_and_gradient(net, x, z, y)
                    start = 3+2+1+2*width
                    positions = (0, 1, 2, start, start+min(31, width-1)*width,
                                 start+min(32, width-1)*width+width-1, start+width*width-1)
                    for j in positions:
                        plus, minus = weights.copy(), weights.copy()
                        plus[j] += 1e-5
                        minus[j] -= 1e-5
                        numeric = (loss(net.config, plus, x, z, y, -1.)-
                                   loss(net.config, minus, x, z, y, -1.))/2e-5
                        np.testing.assert_allclose(gradient[j], numeric, rtol=5e-4, atol=5e-6)

    def test_all_families_fit_bootstrap_and_round_trip(self):
        for family in ('continuous','binary','cox'):
            with self.subTest(family=family):
                x,z,y=self.data(family,n=80)
                model=NeuralPLSI(family=family,max_iter=8,hidden_units=6,
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
        a=NeuralPLSI(max_iter=5).fit(x,z,y,random_state=7)
        after=np.random.get_state()
        np.testing.assert_array_equal(before[1],after[1])
        b=NeuralPLSI(max_iter=5).fit(x,z,y,random_state=7)
        np.testing.assert_array_equal(a.net.weights,b.net.weights)
        c=NeuralPLSI(max_iter=5).fit(x,z,y,random_state=8)
        self.assertFalse(np.array_equal(a.net.weights,c.net.weights))

    def test_zero_covariates_and_empty_predictions(self):
        x,_,y=self.data('continuous',n=80)
        z=np.empty((len(x),0))
        model=NeuralPLSI(max_iter=3,add_intercept=True).fit(x,z,y)
        self.assertEqual(model.gamma.size,0)
        self.assertEqual(model.predict(x[:0],z[:0]).shape,(0,))
        self.assertEqual(model.g_function([]).shape,(0,))
        self.assertTrue(np.isfinite(model.predict_gxb(x)).all())


if __name__=='__main__': unittest.main()
