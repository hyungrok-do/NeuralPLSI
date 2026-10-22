import unittest
import numpy as np
from models import NeuralPLSI
from models.native import (NativeModel, checked, library, outcome_loss_gradient,
                           training_validation_indices)


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

    def test_returned_model_matches_recorded_validation_loss(self):
        model=NeuralPLSI(max_epoch=40,hidden_units=4,n_hidden_layers=1,batch_size=16,optimizer='joint').fit(self.X,self.Z,self.y,random_state=19)
        _,valid=training_validation_indices(len(self.X),19)
        loss=np.mean((model.predict(self.X[valid],self.Z[valid]).astype(float)-self.y[valid])**2)
        self.assertAlmostEqual(loss,model.fit_diagnostics['validation_loss'],places=6)

    def test_constant_validation_runs_full_budgets(self):
        for epochs in (0, 1, 40, 80):
            with self.subTest(epochs=epochs):
                model=NeuralPLSI(max_epoch=epochs,hidden_units=4,n_hidden_layers=1,
                                batch_size=1000,learning_rate=0,optimizer='joint')
                model.fit(self.X,self.Z,self.y,random_state=19)
                self.assertEqual(model.fit_diagnostics['joint_epochs'],epochs)
                # max_epoch=0 performs initialization only, without the gamma fit.
                if epochs:
                    self.assertGreater(model.fit_diagnostics['gamma_iterations'],0)
                else:
                    self.assertEqual(model.fit_diagnostics['gamma_iterations'],0)

    def test_worsening_validation_restores_checkpoint_after_full_training(self):
        # Training and validation targets have opposite signs. Learning the
        # training slope makes validation worse after its first checkpoint.
        target=np.tile(np.array([-1.,1.],dtype='float32'),4)
        x,z=target[:,None],np.empty((8,0),dtype='float32')
        fits=[]
        for epochs in (1,40):
            net=NativeModel(1,0,hidden_units=1,n_hidden_layers=1,
                max_epoch=epochs,batch_size=8,weight_decay=0,weight_decay_beta=0)
            weights=np.zeros_like(net.weights); weights[0]=1.
            orders=np.tile(np.arange(8,dtype='int32'),(epochs,1))
            diagnostics=np.empty(3)
            checked(library().nplsi_train(net.config,net.hyper,weights,
                x,z,target,8,x,z,-target,8,orders,diagnostics))
            net.set_weights(weights)
            self.assertEqual(diagnostics[0],epochs)
            self.assertEqual(diagnostics[1],0)  # no gamma or intercept to fit
            fits.append(net)
        np.testing.assert_array_equal(fits[0].weights,fits[1].weights)
        self.assertGreater(np.dot(fits[1].predict(x,z),target),0.)

    def test_gamma_is_exact_full_data_fit_given_index_and_link(self):
        # Gamma and the intercept minimize the outcome loss on all rows, training
        # and validation, with the fitted g(x'beta) as an offset.
        rng=np.random.default_rng(7)
        n=300
        X=rng.normal(size=(n,3)).astype('float32')
        Z=rng.normal(size=(n,2)).astype('float32')
        eta=np.tanh(X[:,0]-X[:,1])+Z@[1.,-.5]
        outcomes={'continuous':eta+rng.normal(size=n),
                  'binary':(rng.uniform(size=n)<1/(1+np.exp(-eta))).astype(float),
                  'cox':np.column_stack([np.round(rng.exponential(np.exp(-eta))*20)+1,
                                         rng.binomial(1,.8,n)])}
        for family,y in outcomes.items():
            with self.subTest(family=family):
                y=y.astype('float32')
                model=NeuralPLSI(family=family,max_epoch=20,hidden_units=4,n_hidden_layers=1,
                                 batch_size=32,add_intercept=family!='cox').fit(X,Z,y,random_state=3)
                self.assertGreater(model.fit_diagnostics['gamma_iterations'],0)
                g=model.predict_gxb(X).astype(float)
                coefficients=np.append(model.gamma,model.intercept_val if family!='cox' else [])
                if family=='continuous':
                    design=np.column_stack([Z,np.ones(n)])
                    expected=np.linalg.lstsq(design,y-g,rcond=None)[0]
                    np.testing.assert_allclose(coefficients,expected,atol=1e-5)
                else:
                    _,delta=outcome_loss_gradient(model.predict(X,Z),y,family)
                    design=Z if family=='cox' else np.column_stack([Z,np.ones(n)])
                    np.testing.assert_allclose(design.T@delta.astype(float),0.,atol=1e-5)
