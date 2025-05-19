
import torch
import torch.nn as nn
import torch.utils
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, TensorDataset
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
    
    
class nPLSInet(nn.Module):
    def __init__(self, p, q):
        super(nPLSInet, self).__init__()
        
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, 1, bias=False)
        self.g_network = nn.Sequential(
            nn.Linear(1, 128),
            nn.ELU(),
            nn.Dropout(0.25),
            nn.Linear(128, 128),
            nn.ELU(),
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
            with torch.no_grad():
                self.x_input.weight.data[0] = -weight
                if self.x_input.bias is not None:
                    self.x_input.bias.data = -self.x_input.bias.data

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
        self.batch_size = 128
        self.max_epoch = 100
        self.net = None
        self._standard_errors = dict()
        
    def fit(self, X, Z, y):
        
        self.net = nPLSInet(X.shape[1], Z.shape[1]).to(self.device)
        
        opt_g = torch.optim.Adam([
            {'params': self.net.g_network.parameters(), 'weight_decay': 1e-6},
            {'params': self.net.x_input.parameters()},
            ], lr=1e-3
        )
        opt_z = torch.optim.SGD([
            {'params': self.net.z_input.parameters()}
            ], lr=1e-2, weight_decay=0.
        )

        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=42)
        batch_size = len(tr_x) // 10
        
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
        
        criterion = nn.MSELoss()
        sch_z = SchedulerCallback(opt_z)
        self.net.train()
        for epoch in range(self.max_epoch):
            for _, batch_z, batch_y in tr_loader:
                batch_z, batch_y = batch_z.to(self.device), batch_y.to(self.device)
                opt_z.zero_grad()
                
                output = self.net.z_input(batch_z).view(-1)
                loss = criterion(output, batch_y) 
                loss.backward()
                opt_z.step()

            with torch.no_grad():
                val_loss = 0.
                for _, batch_z, batch_y in val_loader:
                    batch_z, batch_y = batch_z.to(self.device), batch_y.to(self.device)
                    output = self.net.z_input(batch_z).view(-1)
                    val_loss += criterion(output, batch_y).item()

                if sch_z(val_loss):
                    break

        self.net.normalize_beta(opt_g)
        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)
        for epoch in range(self.max_epoch):
            self.net.train()
            for batch_x, batch_z, batch_y in tr_loader:
                batch_x, batch_z, batch_y = batch_x.to(self.device), batch_z.to(self.device), batch_y.to(self.device)
                
                opt_g.zero_grad()
                opt_z.zero_grad()
                
                batch_xb = self.net.x_input(batch_x)
                output = (self.net.g_network(batch_xb) + self.net.z_input(batch_z)).view(-1)
                #output = self.net(batch_x, batch_z).view(-1)

                batch_zero = torch.zeros_like(batch_y).view(-1, 1).to(self.device)
                loss = criterion(output, batch_y)
                loss += criterion(self.net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                #loss += nn.Softplus()(-self.net.x_input.weight.data[0][0])

                loss.backward()
                
                opt_g.step()
                opt_z.step()

                self.net.normalize_beta(opt_g)

            self.net.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_x, batch_z, batch_y in val_loader:
                    batch_x, batch_z, batch_y = batch_x.to(self.device), batch_z.to(self.device), batch_y.to(self.device)
                    output = self.net(batch_x, batch_z).view(-1)
                    loss = criterion(output, batch_y)
                    loss += criterion(self.net.g_network(batch_zero).view(-1), batch_zero.view(-1))
                    #loss += nn.Softplus()(-self.net.x_input.weight.data[0][0]).mean()
                    val_loss += loss.item()

            if sch_g(val_loss) and sch_z(val_loss):
                break

        self._get_standard_errors(X, Z, y)

    @property
    def beta(self):
        return self.net.x_input.weight.data.cpu().flatten().flatten().numpy()

    @property
    def gamma(self):
        return self.net.z_input.weight.data.cpu().flatten().flatten().numpy()
            
    
    @property
    def beta_se(self):
        return self._standard_errors['beta'] if hasattr(self, '_standard_errors') else None
    
    @property
    def gamma_se(self):
        return self._standard_errors['gamma'] if hasattr(self, '_standard_errors') else None
    
    def summary(self):
        if self.net is None:
            raise ValueError("Model has not been fitted yet.")
        
        beta = self.beta
        gamma = self.gamma
        beta_se = self.beta_se
        gamma_se = self.gamma_se

        summary_df = pd.DataFrame({
            'Parameter': [f'beta_{i:02d}' for i in range(len(beta))] + [f'gamma_{i:02d}' for i in range(len(gamma))],
            'Coefficient': list(beta) + list(gamma),
            'Standard Error': list(beta_se) + list(gamma_se),
            '95% CI Lower Bound': list(beta - 1.96 * beta_se) + list(gamma - 1.96 * gamma_se),
            '95% CI Upper Bound': list(beta + 1.96 * beta_se) + list(gamma + 1.96 * gamma_se)
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
                          ), batch_size=self.batch_size, shuffle=False)
        
        preds = []
        with torch.no_grad():
            for batch_x, batch_z in test_loader:
                batch_x, batch_z = batch_x.to(self.device), batch_z.to(self.device)
                output = self.net(batch_x, batch_z).view(-1)
                preds.append(output.cpu())

        return torch.cat(preds, axis=0).numpy()
