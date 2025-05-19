
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from torch.optim.swa_utils import AveragedModel, SWALR


class SchedulerCallback:
    def __init__(self, optimizer, patience=5, higher_is_better=False, factor=0.1, max_reductions=0):
        """
        A learning rate scheduler with early stopping after a specified number of patience periods.

        Args:
            optimizer (torch.optim.Optimizer): Optimizer whose learning rate needs adjustment.
            patience (int): Number of epochs to wait before reducing LR.
            higher_is_better (bool): Whether a higher metric is better (e.g., accuracy) or lower is better (e.g., loss).
            factor (float): Multiplicative factor to reduce the learning rate.
            max_reductions (int): Maximum number of times to reduce the learning rate before stopping training.
        """
        self.optimizer = optimizer
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.factor = factor
        self.max_reductions = max_reductions  # Stop after reducing LR this many times

        self.best_metric = -float('inf') if higher_is_better else float('inf')
        self.wait = 0  # Counter for consecutive epochs without improvement
        self.reduction_count = 0  # Number of times LR has been reduced

    def __call__(self, current_metric):
        """
        Check if the metric has improved and update learning rate if needed.

        Args:
            current_metric (float): The current metric value to compare.

        Returns:
            bool: True if training should stop, False otherwise.
        """
        if self.higher_is_better:
            improvement = current_metric > self.best_metric
        else:
            improvement = current_metric < self.best_metric

        if improvement:
            self.best_metric = current_metric
            self.wait = 0  # Reset patience counter
        else:
            self.wait += 1  # Increase patience counter

            if self.wait >= self.patience:
                if self.reduction_count < self.max_reductions:
                    self._reduce_lr()
                    self.wait = 0  # Reset patience counter after LR reduction
                else:
                    return True  # Stop training after max reductions

        return False  # Continue training

    def _reduce_lr(self):
        """
        Reduces the learning rate of the optimizer.
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] *= self.factor

        self.reduction_count += 1
    
    def reset(self):
        self.best_metric = -float('inf') if self.higher_is_better else float('inf')
        self.wait = 0
        self.patience = 0
        self.reduction_count = 0
    

class nPLSInet(nn.Module):
    def __init__(self, p, q):
        super(nPLSInet, self).__init__()
        
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, 1, bias=False)
        self.g_network = nn.Sequential(
            nn.Linear(1, 128),
            nn.SELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.SELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, x, z):
        xb = self.x_input(x)
        return self.g_network(xb) + self.z_input(z)
    
    def normalize_beta(self, optimizer=None):
        """
        Normalize and sign-fix the beta vector and adjust optimizer state if needed.
        """
        weight = self.x_input.weight.data[0]

        if weight[0] < 0:
            # Flip the weight
            with torch.no_grad():
                self.x_input.weight.data[0] = -weight
                if self.x_input.bias is not None:
                    self.x_input.bias.data = -self.x_input.bias.data

            
            # Flip optimizer momentum (Adam's exp_avg)
            if optimizer is not None:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if (p is self.x_input.weight) or (p is self.x_input.bias):
                            state = optimizer.state[p]
                            if ('exp_avg' in state) and (state['exp_avg'] is not None):
                                state['exp_avg'].mul_(-1.)

                            if ('momentum_buffer' in state) and (state['momentum_buffer'] is not None):
                                state['momentum_buffer'].mul_(-1.)

        self.x_input.weight.data[0] = self.x_input.weight.data[0] / self.x_input.weight.data[0].square().sum().sqrt()
        

class neuralPLSI:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = 100
        self.net = None
        
    def fit(self, X, Z, y):
        self.net = nPLSInet(X.shape[1], Z.shape[1]).to(self.device)
        self.net = self.train(self.net, X, Z, y, self.device, max_epoch=self.max_epoch)

    @staticmethod
    def train(net, X, Z, y, device, max_epoch=100, random_state=0):
        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=random_state)
        batch_size = 64
        
        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(tr_x).float(),
                        torch.from_numpy(tr_z).float(),
                        torch.from_numpy(tr_y).float()
                        ), batch_size=batch_size, sampler=torch.utils.data.RandomSampler(tr_x))
        
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_x).float(),
                        torch.from_numpy(val_z).float(),
                        torch.from_numpy(val_y).float()
                        ), batch_size=batch_size, shuffle=False)
        
        opt_g = torch.optim.Adam([
            {'params': net.g_network.parameters(), 'weight_decay': 1e-6},
            ], lr=1e-3
        )
        opt_z = torch.optim.SGD([
            {'params': net.x_input.parameters()},
            {'params': net.z_input.parameters()}
            ], lr=1e-2, weight_decay=0.
        )

        # SWA: Create averaged model and scheduler for opt_g
        swa_model = AveragedModel(net)
        swa_start = int(max_epoch * 0.75)  # Start SWA after 75% of training
        swa_scheduler = SWALR(opt_g, swa_lr=5e-4)  # Lower learning rate during SWA

        mse = nn.MSELoss()
        loss_fn = nn.MSELoss()
        
        net.normalize_beta(opt_z)
        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)
        for epoch in range(max_epoch):
            net.train()
            for batch_x, batch_z, batch_y in tr_loader:
                batch_x, batch_z, batch_y = batch_x.to(device), batch_z.to(device), batch_y.to(device)
                
                opt_g.zero_grad()
                opt_z.zero_grad()
                
                output = net(batch_x, batch_z).view(-1)

                batch_zero = torch.zeros_like(batch_y).view(-1, 1).to(device)
                loss = loss_fn(output, batch_y)
                loss += mse(net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                loss.backward()
                
                opt_g.step()
                opt_z.step()

                net.normalize_beta(opt_z)

            # If past swa_start, update the SWA model and adjust LR
            if epoch > 10:
                swa_model.update_parameters(net)
                swa_scheduler.step()

            else:
                # Normal scheduler step
                net.eval()
                val_loss = 0
                with torch.no_grad():
                    for batch_x, batch_z, batch_y in val_loader:
                        batch_x, batch_z, batch_y = batch_x.to(device), batch_z.to(device), batch_y.to(device)
                        output = net(batch_x, batch_z).view(-1)
                        loss = loss_fn(output, batch_y)
                        loss += mse(net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                        val_loss += loss.item()

                if sch_g(val_loss) and sch_z(val_loss):
                    break

        # SWA: update batch norm statistics for the averaged model
        torch.optim.swa_utils.update_bn(tr_loader, swa_model, device=device)
        net = swa_model.module
        return net

    @property
    def beta(self):
        return self.net.x_input.weight.data.cpu().flatten().flatten().numpy()

    @property
    def gamma(self):
        return self.net.z_input.weight.data.cpu().flatten().flatten().numpy()
            
    def summary(self):
        if self.net is None:
            raise ValueError("Model has not been fitted yet.")
        
        beta = self.beta
        gamma = self.gamma

        beta_se = self.beta_se
        gamma_se = self.gamma_se
        beta_lb = self.beta_lb
        beta_ub = self.beta_ub
        gamma_lb = self.gamma_lb
        gamma_ub = self.gamma_ub

        summary_df = pd.DataFrame({
            'Parameter': [f'beta_{i:02d}' for i in range(len(beta))] + [f'gamma_{i:02d}' for i in range(len(gamma))],
            'Coefficient': list(beta) + list(gamma),
            'Standard Error': list(beta_se) + list(gamma_se),
            '95% CI Lower Bound': list(beta_lb) + list(gamma_lb),
            '95% CI Upper Bound': list(beta_ub) + list(gamma_ub)
        })

        return summary_df
    
    def g_function(self, x):
        x = x.reshape(-1, 1)
        self.net.eval()
        x = torch.from_numpy(x).float().to(self.device)
        with torch.no_grad():
            return self.net.g_network(x).view(-1).cpu().numpy()

    def predict(self, X, Z):
        self.net.eval()
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(),
                          torch.from_numpy(Z).float()
                          ), batch_size=128, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, batch_z in test_loader:
                batch_x, batch_z = batch_x.to(self.device), batch_z.to(self.device)
                output = self.net(batch_x, batch_z).view(-1)
                preds.append(output.cpu())

        return torch.cat(preds, axis=0).numpy()

    def inference_bootstrap(self, X, Z, y, n_samples=100):
        beta_samples, gamma_samples = [], []
        for i in range(n_samples):
            bootstrap_indices = np.random.choice(len(X), len(X), replace=True)
            X_bootstrap = X[bootstrap_indices]
            Z_bootstrap = Z[bootstrap_indices]
            y_bootstrap = y[bootstrap_indices]

            net = nPLSInet(X.shape[1], Z.shape[1]).to(self.device)
            net = self.train(net, X_bootstrap, Z_bootstrap, y_bootstrap, self.device, max_epoch=self.max_epoch, random_state=i)
            beta_samples.append(net.x_input.weight.data.cpu().flatten().numpy())
            gamma_samples.append(net.z_input.weight.data.cpu().flatten().numpy())

        self.beta_lb = np.percentile(beta_samples, 2.5, axis=0)
        self.beta_ub = np.percentile(beta_samples, 97.5, axis=0)
        self.gamma_lb = np.percentile(gamma_samples, 2.5, axis=0)
        self.gamma_ub = np.percentile(gamma_samples, 97.5, axis=0)
        self.beta_se = np.std(beta_samples, axis=0)
        self.gamma_se = np.std(gamma_samples, axis=0)
        
    def inference_sandwich(self, X, Z, y):
        if self.net is None:
            raise ValueError("Model has not been fitted yet.")

        self.net.eval()

        for param in self.net.g_network.parameters():
            param.requires_grad = False

        X = torch.from_numpy(X).float().to(self.device)
        Z = torch.from_numpy(Z).float().to(self.device)
        y = torch.from_numpy(y).float().to(self.device)

        n = X.shape[0]

        beta_flat = self.net.x_input.weight.view(-1)
        gamma_flat = self.net.z_input.weight.view(-1)

        ### --- Functional Forward --- ###
        def forward_with_custom_beta_gamma(X, Z, beta_flat=None, gamma_flat=None):
            out = X
            if beta_flat is not None:
                out = F.linear(out, beta_flat.view_as(self.net.x_input.weight), bias=None)
            else:
                out = self.net.x_input(X)

            if gamma_flat is not None:
                out += F.linear(Z, gamma_flat.view_as(self.net.z_input.weight), bias=None)
            else:
                out += self.net.z_input(Z)

            out = self.net.g_network(out)
            return out

        ### --- Gamma Inference First --- ###
        def loss_fn_gamma(gamma_params):
            outputs = forward_with_custom_beta_gamma(X, Z, gamma_flat=gamma_params)
            loss = (outputs.view(-1) - y).pow(2).mean()
            return loss

        # Gamma Hessian
        hessian_gamma = torch.autograd.functional.hessian(loss_fn_gamma, gamma_flat)
        hessian_gamma_inv = torch.linalg.pinv(hessian_gamma)

        # Gamma gradients per sample
        grads_gamma = []
        for i in range(n):
            xi = X[i:i+1]
            zi = Z[i:i+1]
            yi = y[i:i+1]

            def loss_i_gamma(gamma_params):
                output = forward_with_custom_beta_gamma(xi, zi, gamma_flat=gamma_params)
                loss = (output.view(-1) - yi).pow(2)
                return loss

            grad_gamma_i = torch.autograd.grad(loss_i_gamma(gamma_flat), gamma_flat, retain_graph=True)[0]
            grads_gamma.append(grad_gamma_i.unsqueeze(0))

        grads_gamma = torch.cat(grads_gamma, dim=0)  

        S_gamma = (grads_gamma.T @ grads_gamma) / n
        cov_gamma = (hessian_gamma_inv @ S_gamma @ hessian_gamma_inv) / n

        se_gamma = torch.sqrt(torch.diag(cov_gamma)).cpu().numpy()

        self.gamma_se = se_gamma

        ### --- Now do Beta Inference separately --- ###
        def loss_fn_beta(beta_params):
            outputs = forward_with_custom_beta_gamma(X, Z, beta_flat=beta_params)
            loss = (outputs.view(-1) - y).pow(2).mean()
            return loss

        # Beta Hessian
        hessian_beta = torch.autograd.functional.hessian(loss_fn_beta, beta_flat)
        hessian_beta_inv = torch.linalg.pinv(hessian_beta)

        # Beta gradients per sample
        grads_beta = []
        for i in range(n):
            xi = X[i:i+1]
            zi = Z[i:i+1]
            yi = y[i:i+1]

            def loss_i_beta(beta_params):
                output = forward_with_custom_beta_gamma(xi, zi, beta_flat=beta_params)
                loss = (output.view(-1) - yi).pow(2)
                return loss

            grad_beta_i = torch.autograd.grad(loss_i_beta(beta_flat), beta_flat, retain_graph=True)[0]
            grads_beta.append(grad_beta_i.unsqueeze(0))

        grads_beta = torch.cat(grads_beta, dim=0)  # (n, p_beta)

        S_beta = (grads_beta.T @ grads_beta) / n
        cov_beta = (hessian_beta_inv @ S_beta @ hessian_beta_inv) / n

        # Project beta covariance to tangent space
        beta_norm = beta_flat.detach()
        beta_norm = beta_norm / beta_norm.norm()

        P = torch.eye(len(beta_norm), device=self.device) - beta_norm.unsqueeze(1) @ beta_norm.unsqueeze(0)  # (p, p)
        cov_beta_adjusted = P @ cov_beta @ P

        se_beta = torch.sqrt(torch.diag(cov_beta_adjusted)).cpu().numpy()

        self.beta_se = se_beta

        self.beta_lb = self.beta - 1.96 * self.beta_se
        self.beta_ub = self.beta + 1.96 * self.beta_se
        self.gamma_lb = self.gamma - 1.96 * self.gamma_se
        self.gamma_ub = self.gamma + 1.96 * self.gamma_se
