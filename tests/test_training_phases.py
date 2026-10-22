import unittest
import numpy as np
from models import NeuralPLSI
from models.native import NativeModel, loss_and_gradient, outcome_loss_gradient


def penalized_objective(net,X,Z,y):
    """Outcome loss, g(0)^2 anchor and weight_decay/2 times the squared link
    network (including the skip weight), summed in the native order."""
    value,_=loss_and_gradient(net,X,Z,y)
    p,q,intercept=int(net.config[0]),int(net.config[1]),int(net.config[4])
    decay=float(net.hyper[0])
    for w in net.weights[p+q:len(net.weights)-intercept].astype(float):
        value+=.5*decay*w*w
    return value


class TrainingPhasesTest(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(54)
        self.X=rng.normal(size=(80,3)).astype('float32')
        self.Z=rng.normal(size=(80,2)).astype('float32')
        self.y=(self.X[:,0]+self.Z@[2.,-1.]+rng.normal(size=80)).astype('float32')

    def test_recorded_objective_is_penalized_loss_of_returned_model(self):
        for intercept in (False,True):
            with self.subTest(intercept=intercept):
                model=NeuralPLSI(max_iter=40,hidden_units=4,n_hidden_layers=1,
                                 add_intercept=intercept).fit(self.X,self.Z,self.y,random_state=19)
                self.assertAlmostEqual(penalized_objective(model.net,self.X,self.Z,self.y),
                                       model.fit_diagnostics['objective'],places=12)

    def test_iteration_cap_and_initialization_only(self):
        for cap in (0,1,5):
            with self.subTest(cap=cap):
                model=NeuralPLSI(max_iter=cap,hidden_units=4,n_hidden_layers=1)
                model.fit(self.X,self.Z,self.y,random_state=19)
                self.assertEqual(model.fit_diagnostics['iterations'],cap)
                # max_iter=0 performs initialization only, without the gamma fit.
                if cap:
                    self.assertGreater(model.fit_diagnostics['gamma_iterations'],0)
                else:
                    self.assertEqual(model.fit_diagnostics['gamma_iterations'],0)
                    initial=NativeModel(3,2,hidden_units=4,n_hidden_layers=1,max_iter=0,random_state=19)
                    initial.resolve_sign_ambiguity()
                    np.testing.assert_array_equal(model.net.weights,initial.weights)

    def test_training_converges_before_the_cap_and_penalty_shrinks_the_link(self):
        norms=[]
        for decay in (1e-4,1e-1):
            with self.subTest(weight_decay=decay):
                model=NeuralPLSI(weight_decay=decay,hidden_units=4,n_hidden_layers=1)
                model.fit(self.X,self.Z,self.y,random_state=19)
                start=NeuralPLSI(weight_decay=decay,max_iter=0,hidden_units=4,n_hidden_layers=1)
                start.fit(self.X,self.Z,self.y,random_state=19)
                self.assertLess(model.fit_diagnostics['iterations'],model.max_iter)
                self.assertLess(model.fit_diagnostics['objective'],start.fit_diagnostics['objective'])
                norms.append(np.linalg.norm(model.net.weights[5:]))
        self.assertGreater(norms[0],norms[1])

    def test_gamma_is_exact_full_data_fit_given_index_and_link(self):
        # Gamma and the intercept minimize the outcome loss on all rows, with
        # the fitted g(x'beta) as an offset.
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
                model=NeuralPLSI(family=family,max_iter=20,hidden_units=4,n_hidden_layers=1,
                                 add_intercept=family!='cox').fit(X,Z,y,random_state=3)
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
