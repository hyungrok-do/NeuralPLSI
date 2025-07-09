import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader, TensorDataset, Dataset
from sklearn.model_selection import train_test_split

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
    

class CoxPHNLLLoss(nn.Module):
    def forward(self, risk_scores, targets):
        """
        Compute the negative partial log-likelihood for Cox Proportional Hazards model.
        
        Args:
            risk_scores (torch.Tensor): Predicted risk scores for each individual.
            durations (torch.Tensor): Event or censoring times.
            events (torch.Tensor): Event indicators (1 if event occurred, 0 if censored).
        
        Returns:
            torch.Tensor: Negative partial log-likelihood loss.
        """

        durations, events = targets[:, 0], targets[:, 1]
        if risk_scores.dim() > 1:
            risk_scores = risk_scores.squeeze(1)

        # Sort by durations in descending order
        sorted_indices = torch.argsort(durations, descending=True)
        risk_scores = risk_scores[sorted_indices]
        durations = durations[sorted_indices]
        events = events[sorted_indices]

        eps = 1e-8

        gamma = risk_scores.max()
        log_cumsum_h = risk_scores.sub(gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return -risk_scores.sub(log_cumsum_h).mul(events).sum().div(events.sum())


class nPLSInet(nn.Module):
    def __init__(self, p, q):
        super(nPLSInet, self).__init__()
        
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, 1, bias=False)
        self.g_network = nn.Sequential(
            nn.Linear(1, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 64),
            nn.SELU(),
            nn.Linear(64, 1)
        )

        self.flip_sign = False

    def forward(self, x, z):
        xb = self.x_input(x)
        if self.flip_sign:
            xb = -xb

        return self.g_network(xb) + self.z_input(z)

    def g_function(self, x):
        """
        Compute the g function for the input x.
        """
        if self.flip_sign:
            x = -x
        return self.g_network(x)
    
    def normalize_beta(self, optimizer=None):
        """
        Normalize and sign-fix the beta vector and adjust optimizer state if needed.
        """
        weight = self.x_input.weight.data[0]

        if weight[0] < 0:
            # Flip the weight
            self.flip_sign = not self.flip_sign
            
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
    def __init__(self, family='continuous'):
        self.family = family
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = 500
        self.net = None
        
    def fit(self, X, Z, y):
        torch.manual_seed(0)
        self.net = nPLSInet(X.shape[1], Z.shape[1]).to(self.device)
        self.net = self.train(self.net, X, Z, y, self.family, self.device, max_epoch=self.max_epoch)

    @staticmethod
    def train(net, X, Z, y, family, device, batch_size=None, max_epoch=500, random_state=0):
        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=random_state)

        batch_size = 32

        tr_loader = DataLoader(
            TensorDataset(torch.from_numpy(tr_x).float(),
                          torch.from_numpy(tr_z).float(),
                          torch.from_numpy(tr_y).float()
                          ), batch_size=batch_size, sampler=torch.utils.data.RandomSampler(tr_x))
        
        val_loader = DataLoader(
            TensorDataset(torch.from_numpy(val_x).float(),
                          torch.from_numpy(val_z).float(),
                          torch.from_numpy(val_y).float()
                         ), batch_size=batch_size if batch_size > len(val_x) else len(val_x), shuffle=False)
                        
        opt_g = torch.optim.Adam([
            {'params': net.x_input.parameters(), 'weight_decay': 0.},
            {'params': net.g_network.parameters(), 'weight_decay': 1e-4},
            ], lr=1e-3,
        )

        opt_z = torch.optim.SGD([
            {'params': net.z_input.parameters()}
            ], lr=1e-3, momentum=0.9, weight_decay=0.
        )

        mse = nn.MSELoss()
        if family == 'continuous':
            loss_fn = nn.MSELoss()
        elif family == 'binary':
            loss_fn = nn.BCEWithLogitsLoss()
        elif family == 'cox':
            loss_fn = CoxPHNLLLoss()
        else:
            raise ValueError("Unsupported family type. Use 'continuous', 'binary', or 'cox'.")
                
        net.normalize_beta(opt_g)
        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)
        for epoch in range(max_epoch):
            net.train()
            for batch_x, batch_z, batch_y in tr_loader:
                batch_x, batch_z, batch_y = batch_x.to(device), batch_z.to(device), batch_y.to(device)
                
                opt_g.zero_grad()
                opt_z.zero_grad()
                
                output = net(batch_x, batch_z).view(-1)

                batch_zero = torch.zeros((1, 1)).to(device)
                loss = loss_fn(output, batch_y)
                loss += mse(net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                loss.backward()
                
                opt_g.step()
                opt_z.step()

                net.normalize_beta(opt_g)

            net.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_z, batch_y in val_loader:
                    batch_x, batch_z, batch_y = batch_x.to(device), batch_z.to(device), batch_y.to(device)
                    output = net(batch_x, batch_z).view(-1)
                    loss = loss_fn(output, batch_y)
                    batch_zero = torch.zeros((1, 1)).to(device)
                    loss += mse(net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                    val_loss += loss.item()

            if sch_g(val_loss) and sch_z(val_loss):
                break

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
            return self.net.g_function(x).view(-1).cpu().numpy()

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
    
    def predict_proba(self, X, Z):
        self.net.eval()
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(),
                          torch.from_numpy(Z).float()
                          ), batch_size=128, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, batch_z in test_loader:
                batch_x, batch_z = batch_x.to(self.device), batch_z.to(self.device)
                output = self.net(batch_x, batch_z).view(-1).sigmoid()
                preds.append(output.cpu())

        return torch.cat(preds, axis=0).numpy()
    
    def predict_partial_hazard(self, X, Z):
        self.net.eval()
        test_loader = DataLoader(
            TensorDataset(torch.from_numpy(X).float(),
                            torch.from_numpy(Z).float()
                            ), batch_size=128, shuffle=False)
        preds = []
        with torch.no_grad():
            for batch_x, batch_z in test_loader:
                batch_x, batch_z = batch_x.to(self.device), batch_z.to(self.device)
                output = self.net(batch_x, batch_z).view(-1).exp()
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
        