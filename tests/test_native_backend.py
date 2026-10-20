import copy
import unittest

import numpy as np
import torch

from models.NeuralPLSI import NeuralPLSI, _nPLSInet, CoxCCLoss
from models.native import (configuration, hyperparameters, library, checked,
                           loss_and_gradient, pack, parameters)


class NativeBackendTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        library()

    def data(self, family):
        rng = np.random.default_rng(17)
        x, z = rng.normal(size=(80, 4)).astype('float32'), rng.normal(size=(80, 2)).astype('float32')
        y = x[:, 0]+z[:, 0]+rng.normal(size=80)
        if family == 'binary':
            y = (y > 0).astype(float)
        elif family == 'cox':
            y = np.column_stack([rng.integers(1, 8, 80), rng.binomial(1, .7, 80)])
        return x, z, y.astype('float32')

    def loss(self, net, x, z, y, family):
        output = net(torch.from_numpy(x), torch.from_numpy(z)).view(-1)
        target = torch.from_numpy(y)
        if family == 'continuous':
            value = torch.nn.functional.mse_loss(output, target)
        elif family == 'binary':
            value = torch.nn.functional.binary_cross_entropy_with_logits(output, target)
        else:
            value = CoxCCLoss()(output, target)
        return value+net.g_network(torch.zeros(1, 1)).square().mean()

    def test_loss_gradients_and_first_adam_update(self):
        for family in ('continuous', 'binary', 'cox'):
            for activation in ('ELU', 'LeakyReLU', 'ReLU', 'Tanh'):
                with self.subTest(family=family, activation=activation):
                    torch.manual_seed(9)
                    net = _nPLSInet(4, 2, hidden_units=5, n_hidden_layers=2,
                                    add_intercept=(family != "cox"), activation=activation)
                    net.flip_sign = True
                    x, z, y = self.data(family)
                    value = self.loss(net, x, z, y, family)
                    value.backward()
                    grad = np.concatenate([p.grad.numpy().ravel() for p in parameters(net)])
                    native_loss, native_grad = loss_and_gradient(net, x, z, y, family)
                    np.testing.assert_allclose(native_loss, value.item(), rtol=3e-6, atol=2e-7)
                    np.testing.assert_allclose(native_grad, grad, rtol=2e-4, atol=2e-6)
                    weights = pack(net)
                    checked(library().nplsi_adam_step(configuration(net, family),
                        hyperparameters(net), weights, native_grad, 0))
                    optimizer = torch.optim.AdamW([
                        {'params': list(net.x_input.parameters()), 'weight_decay': 1e-5},
                        {'params': list(net.g_network.parameters())+list(net.g_skip.parameters()), 'weight_decay': 1e-4},
                        {'params': list(net.z_input.parameters())+([net.intercept] if net.add_intercept else []), 'weight_decay': 0, 'lr': .05},
                    ], lr=.001)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), 1.)
                    optimizer.step()
                    np.testing.assert_allclose(weights, pack(net), rtol=2e-5, atol=2e-6)

    def test_fit_and_bootstrap_agree(self):
        for family in ('continuous', 'binary', 'cox'):
            with self.subTest(family=family):
                x, z, y = self.data(family)
                models = []
                for backend in ('torch', 'cpp'):
                    model = NeuralPLSI(family=family, max_epoch=8, batch_size=24,
                        hidden_units=6, activation='ELU', add_intercept=(family != 'cox'), backend=backend, device='cpu')
                    model.fit(x, z, y, random_state=4)
                    models.append(model)
                for block in ('beta', 'gamma'):
                    np.testing.assert_allclose(getattr(models[0], block), getattr(models[1], block),
                                               rtol=3e-4, atol=3e-5)
                np.testing.assert_allclose(models[0].predict(x, z), models[1].predict(x, z),
                                           rtol=3e-4, atol=5e-5)
                draws = [m.inference_bootstrap(x, z, y, n_samples=2, random_state=4, n_jobs=1,
                            g_grid=np.linspace(-2, 2, 7)) for m in models]
                for block in ('beta_samples', 'gamma_samples', 'g_samples')+ (('intercept_samples',) if family != 'cox' else ()):
                    np.testing.assert_allclose(draws[0][block], draws[1][block], rtol=1e-3, atol=1e-4)

    def test_optional_linear_initialization_in_both_backends(self):
        for family in ('continuous', 'binary', 'cox'):
            x, z, y = self.data(family)
            models = [NeuralPLSI(family=family, max_epoch=3, initial=True, hidden_units=5, batch_size=24,
                                  backend=b, device='cpu').fit(x, z, y, random_state=9)
                      for b in ('torch', 'cpp')]
            np.testing.assert_allclose(models[0].predict(x, z), models[1].predict(x, z),
                                       rtol=1e-4, atol=2e-5)

    def test_zero_epoch_and_orientation(self):
        x, z, y = self.data('continuous')
        models = [NeuralPLSI(max_epoch=0, backend=b, device='cpu').fit(x, z, y) for b in ('torch', 'cpp')]
        np.testing.assert_array_equal(models[0].predict(x, z), models[1].predict(x, z))
        state = copy.deepcopy(models[1].net.state_dict())
        clone = copy.deepcopy(models[1].net)
        clone.flip_sign = not clone.flip_sign
        clone.load_state_dict(state)
        torch.testing.assert_close(clone(torch.from_numpy(x), torch.from_numpy(z)),
                                   models[1].net(torch.from_numpy(x), torch.from_numpy(z)))


if __name__ == '__main__':
    unittest.main()
