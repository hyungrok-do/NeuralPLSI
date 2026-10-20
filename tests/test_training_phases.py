import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from models.NeuralPLSI import NeuralPLSI, _nPLSInet


class TrainingPhasesTest(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        rng = np.random.default_rng(54)
        self.X = rng.normal(size=(80, 3)).astype('float32')
        self.Z = rng.normal(size=(80, 2)).astype('float32')
        self.y = (self.X[:, 0] + self.Z @ [2., -1.] +
                  rng.normal(size=80)).astype('float32')
        torch.manual_seed(17)
        self.net = _nPLSInet(3, 2, hidden_units=4, n_hidden_layers=1)

    def train(self, net, **kw):
        torch.manual_seed(19)
        return NeuralPLSI._train(net, self.X, self.Z, self.y, 'continuous',
                                 torch.device('cpu'), random_state=19, **kw)

    def test_fixed_link_backward_cannot_change_gamma_updates(self):
        """A derivative of an unoptimized parameter must not clip gamma."""
        regular = copy.deepcopy(self.net)
        perturbed = copy.deepcopy(self.net)
        gamma_start = []

        def distort_only_fixed_link(grad):
            return grad if perturbed.x_input.weight.requires_grad else grad*1e8

        perturbed.g_skip.weight.register_hook(distort_only_fixed_link)

        def capture_ft_start(module, args):
            if not module.x_input.weight.requires_grad and not gamma_start:
                gamma_start.append(module.z_input.weight.detach().clone())

        regular.register_forward_pre_hook(capture_ft_start)
        self.train(regular, max_epoch=4, batch_size=16)
        self.train(perturbed, max_epoch=4, batch_size=16)
        torch.testing.assert_close(regular.z_input.weight, perturbed.z_input.weight,
                                   rtol=0, atol=0)
        self.assertGreater(float((regular.z_input.weight.detach()-gamma_start[0]).norm()), 1e-5)
        self.assertTrue(regular.g_skip.weight.requires_grad)

    def test_returned_gamma_matches_best_recorded_validation_loss(self):
        from sklearn.model_selection import train_test_split
        net = copy.deepcopy(self.net)
        self.train(net, max_epoch=40, batch_size=16)
        _, vx, _, vz, _, vy = train_test_split(self.X, self.Z, self.y, test_size=.2, random_state=19)
        with torch.no_grad():
            loss = torch.nn.functional.mse_loss(net(torch.from_numpy(vx), torch.from_numpy(vz)).view(-1),
                                                torch.from_numpy(vy))
        self.assertAlmostEqual(loss.item(), net.fit_diagnostics['validation_loss'], places=6)

    def test_constant_validation_stops_joint_training_after_patience(self):
        """At LR zero, validation is constant and patience five means six epochs."""
        counts = {'joint': 0, 'gamma': 0}
        net = copy.deepcopy(self.net)
        original_step = torch.optim.AdamW.step

        def step(optimizer, *args, **kwargs):
            if any(p is net.x_input.weight for group in optimizer.param_groups
                   for p in group['params']):
                counts['joint'] += 1
            elif not net.x_input.weight.requires_grad:
                counts['gamma'] += 1
            return original_step(optimizer, *args, **kwargs)

        with patch.object(torch.optim.AdamW, 'step', step):
            self.train(net, max_epoch=40, batch_size=1000, learning_rate=0)
        self.assertEqual(counts['joint'], 6)
        self.assertEqual(counts['gamma'], 21)


if __name__ == '__main__':
    unittest.main()
