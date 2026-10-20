import unittest
import numpy as np
from models import NeuralPLSI
from models.native import NativeModel, checked, library, training_validation_indices


class TrainingPhasesTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(54)
        self.X=rng.normal(size=(80,3)).astype('float32')
        self.Z=rng.normal(size=(80,2)).astype('float32')
        self.y=(self.X[:,0]+self.Z@[2.,-1.]+rng.normal(size=80)).astype('float32')

    def test_fixed_parameter_gradients_cannot_affect_gamma_step(self):
        net=NativeModel(3,2,hidden_units=4,n_hidden_layers=1,add_intercept=True)
        rng=np.random.default_rng(19)
        gradient=rng.normal(size=len(net.weights)).astype('float32')
        distorted=gradient.copy(); active=np.zeros(len(gradient),dtype=bool)
        active[3:5]=True; active[-1]=True
        distorted[~active]*=1e8
        a,b=net.weights.copy(),net.weights.copy()
        for weights,g in ((a,gradient),(b,distorted)):
            checked(library().nplsi_adam_step(net.config,net.hyper,weights,g,1))
        np.testing.assert_array_equal(a,b)
        np.testing.assert_array_equal(a[~active],net.weights[~active])
        self.assertTrue(np.any(a[active]!=net.weights[active]))

    def test_returned_gamma_matches_best_recorded_validation_loss(self):
        model=NeuralPLSI(max_epoch=40,hidden_units=4,n_hidden_layers=1,batch_size=16).fit(self.X,self.Z,self.y,random_state=19)
        _,valid=training_validation_indices(len(self.X),19)
        loss=np.mean((model.predict(self.X[valid],self.Z[valid]).astype(float)-self.y[valid])**2)
        self.assertAlmostEqual(loss,model.fit_diagnostics['validation_loss'],places=6)

    def test_constant_validation_stops_after_patience(self):
        model=NeuralPLSI(max_epoch=40,hidden_units=4,n_hidden_layers=1,batch_size=1000,learning_rate=0)
        model.fit(self.X,self.Z,self.y,random_state=19)
        self.assertEqual(model.fit_diagnostics['joint_epochs'],6)
        self.assertEqual(model.fit_diagnostics['gamma_epochs'],21)
