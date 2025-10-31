import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .base import _SummaryMixin, draw_bootstrap_indices, run_parallel_bootstrap

def _init_pytorch_threads():
    import os
    num_threads = os.cpu_count()
    if num_threads:
        try:
            torch.set_num_threads(num_threads)
        except:
            pass
        try:
            torch.set_num_interop_threads(num_threads)
        except:
            pass

try:
    _init_pytorch_threads()
except:
    pass

def _torch_version_geq(v: str) -> bool:
    try:
        from packaging.version import Version
        return Version(torch.__version__) >= Version(v)
    except Exception:
        return False

def _enable_fast_matmul():
    try:
        if hasattr(torch, "set_float32_matmul_precision"):
            torch.set_float32_matmul_precision("high")
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul"):
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
    except Exception:
        pass

def _optimize_cpu_training():
    import os
    num_threads = os.cpu_count()
    if num_threads:
        try:
            torch.set_num_threads(num_threads)
        except RuntimeError:
            pass
        try:
            torch.set_num_interop_threads(num_threads)
        except RuntimeError:
            pass
    try:
        if hasattr(torch.backends, "mkl") and torch.backends.mkl.is_available():
            torch.backends.mkl.enabled = True
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "mkldnn"):
            torch.backends.mkldnn.enabled = True
    except Exception:
        pass

class CoxPHNLLLoss(nn.Module):
    def forward(self, risk_scores, targets):
        durations, events = targets[:, 0], targets[:, 1]
        if risk_scores.dim() > 1:
            risk_scores = risk_scores.squeeze(1)
        idx = torch.argsort(durations, descending=True)
        r = risk_scores[idx]; e = events[idx]
        eps = 1e-8; gamma = r.max()
        log_cumsum_h = (r - gamma).exp().cumsum(0).add(eps).log().add(gamma)
        return -(r - log_cumsum_h).mul(e).sum() / (e.sum() + eps)

class SchedulerCallback:
    def __init__(self, optimizer, patience=5, higher_is_better=False, factor=0.1, max_reductions=0):
        self.optimizer = optimizer
        self.patience = patience
        self.higher_is_better = higher_is_better
        self.factor = factor
        self.max_reductions = max_reductions
        self.best_metric = -float('inf') if higher_is_better else float('inf')
        self.wait = 0
        self.reduction_count = 0

    def __call__(self, metric):
        improve = (metric > self.best_metric) if self.higher_is_better else (metric < self.best_metric)
        if improve:
            self.best_metric = metric; self.wait = 0
        else:
            self.wait += 1
            if self.wait >= self.patience:
                if self.reduction_count < self.max_reductions:
                    for g in self.optimizer.param_groups:
                        g['lr'] *= self.factor
                    self.reduction_count += 1
                    self.wait = 0
                else:
                    return True
        return False

class _nPLSInet(nn.Module):
    def __init__(self, p, q, hidden_units=32, n_hidden_layers=3, n_classes=1):
        super().__init__()
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, n_classes, bias=True)
        
        layers = []
        layers.append(nn.Linear(1, hidden_units))
        layers.append(nn.SELU())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(nn.SELU())
        layers.append(nn.Linear(hidden_units, n_classes))
        self.g_network = nn.Sequential(*layers)
        
        self.flip_sign = False
        self.n_classes = n_classes

    def forward(self, x, z):
        xb = self.x_input(x)
        if self.flip_sign: xb = -xb
        return self.g_network(xb) + self.z_input(z)

    def g_function(self, x):
        if self.flip_sign: x = -x
        return self.g_network(x)

    def gxb(self, x):
        xb = self.x_input(x)
        if self.flip_sign: xb = -xb
        return self.g_network(xb)

    def normalize_beta(self, optimizer=None):
        w = self.x_input.weight.data[0]
        if w[0] < 0:
            self.flip_sign = not self.flip_sign
            with torch.no_grad():
                self.x_input.weight.data[0] = -w
                if self.x_input.bias is not None:
                    self.x_input.bias.data = -self.x_input.bias.data
            if optimizer is not None:
                for group in optimizer.param_groups:
                    for p in group['params']:
                        if (p is self.x_input.weight) or (p is self.x_input.bias):
                            st = optimizer.state.get(p, {})
                            if 'exp_avg' in st and st['exp_avg'] is not None: st['exp_avg'].mul_(-1.)
                            if 'momentum_buffer' in st and st['momentum_buffer'] is not None: st['momentum_buffer'].mul_(-1.)
        denom = self.x_input.weight.data[0].square().sum().sqrt() + 1e-12
        self.x_input.weight.data[0] = self.x_input.weight.data[0] / denom

class neuralPLSI(_SummaryMixin):
    def __init__(self, family='continuous', max_epoch=200, batch_size=64,
                 learning_rate=1e-3, weight_decay=1e-4,
                 hidden_units=32, n_hidden_layers=3, grad_clip=1.0,
                 precompile=True, compile_backend=None, compile_mode=None, num_workers=0):
        self.family = family
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.grad_clip = grad_clip
        self.precompile = precompile
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self.num_workers = num_workers
        self.net = None
        self._net_infer = None
        self._compiled = False
        _enable_fast_matmul()
        if self.device.type == 'cpu':
            _optimize_cpu_training()

    def fit(self, X, Z, y, random_state=0):
        torch.manual_seed(0)
        p, q = X.shape[1], Z.shape[1]
        n_classes = 2 if self.family == 'binary' else 1
        self.net = _nPLSInet(p, q, hidden_units=self.hidden_units, 
                            n_hidden_layers=self.n_hidden_layers,
                            n_classes=n_classes).to(self.device)

        self.net = self._train(self.net, X, Z, y, self.family, self.device,
                               batch_size=self.batch_size, max_epoch=self.max_epoch,
                               learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                               grad_clip=self.grad_clip, random_state=random_state)

        if self.precompile:
            self._maybe_build_inference_engine(self.net, X, Z)
        else:
            self._net_infer = self.net.eval()

    @staticmethod
    def _train(net, X, Z, y, family, device, batch_size=64, max_epoch=200,
               learning_rate=1e-3, weight_decay=1e-4, grad_clip=1.0, random_state=0):
        X = np.asarray(X, dtype=np.float32)
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=random_state)
        tr_ds = TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_z), torch.from_numpy(tr_y))
        val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_z), torch.from_numpy(val_y))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size, len(val_x)), shuffle=False, num_workers=0)

        opt_g = torch.optim.AdamW(
            [{'params': net.x_input.parameters(), 'weight_decay': 0.},
             {'params': net.g_network.parameters(), 'weight_decay': weight_decay}], lr=learning_rate
        )
        opt_z = torch.optim.AdamW([{'params': net.z_input.parameters()}], lr=learning_rate, weight_decay=weight_decay)

        if family == 'continuous':
            loss_fn = nn.MSELoss()
        elif family == 'binary':
            loss_fn = nn.CrossEntropyLoss()
        elif family == 'cox':
            loss_fn = CoxPHNLLLoss()
        else:
            raise ValueError("family must be 'continuous', 'binary', or 'cox'.")

        mse0 = nn.MSELoss()
        net.normalize_beta(opt_g)
        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)

        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler(device.type, enabled=(device.type == 'cuda'))
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        for _ in range(max_epoch):
            net.train()
            for bx, bz, by in tr_loader:
                bx = bx.to(device)
                bz = bz.to(device)
                by = by.to(device)

                opt_g.zero_grad(set_to_none=True); opt_z.zero_grad(set_to_none=True)
                autocast_ctx = torch.amp.autocast(device.type) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=(device.type == 'cuda'))
                with autocast_ctx:
                    out = net(bx, bz)
                    if family == 'binary':
                        target = by.long()
                        loss = loss_fn(out, target)
                    elif family == 'cox':
                        target = by
                        loss = loss_fn(out.view(-1), target)
                    else:
                        target = by.view_as(out.view(-1))
                        loss = loss_fn(out.view(-1), target)
                    zero = torch.zeros((1, net.n_classes), device=device)
                    loss = loss + mse0(net.g_network(torch.zeros((1, 1), device=device)), zero)
                scaler.scale(loss).backward()

                if grad_clip > 0:
                    scaler.unscale_(opt_g)
                    scaler.unscale_(opt_z)
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)

                scaler.step(opt_g); scaler.step(opt_z); scaler.update()
                net.normalize_beta(opt_g)

            net.eval(); val_loss = 0.0
            with torch.no_grad():
                for bx, bz, by in val_loader:
                    bx = bx.to(device)
                    bz = bz.to(device)
                    by = by.to(device)
                    autocast_ctx = torch.amp.autocast(device.type) if hasattr(torch, 'amp') and hasattr(torch.amp, 'autocast') else torch.cuda.amp.autocast(enabled=(device.type == 'cuda'))
                    with autocast_ctx:
                        out = net(bx, bz)
                        if family == 'binary':
                            target = by.long()
                            l = loss_fn(out, target)
                        elif family == 'cox':
                            target = by
                            l = loss_fn(out.view(-1), target)
                        else:
                            target = by.view_as(out.view(-1))
                            l = loss_fn(out.view(-1), target)
                        zero = torch.zeros((1, net.n_classes), device=device)
                        l = l + mse0(net.g_network(torch.zeros((1, 1), device=device)), zero)
                        val_loss += float(l.item())
                        
            if sch_g(val_loss) and sch_z(val_loss):
                break
        return net

    def _maybe_build_inference_engine(self, trained_net, X, Z):
        self._net_infer = trained_net.eval()
        try:
            if _torch_version_geq("2.0"):
                compiled = torch.compile(
                    trained_net.eval(),
                    backend=self.compile_backend,
                    mode=self.compile_mode,
                    fullgraph=False, dynamic=False
                )
                xb = torch.from_numpy(X[:min(64, len(X))].astype(np.float32)).to(self.device)
                zb = torch.from_numpy(Z[:min(64, len(Z))].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    _ = compiled(xb, zb)
                self._net_infer = compiled
                self._compiled = True
                return
        except Exception:
            self._compiled = False

    def _infer_net(self):
        if self._net_infer is None:
            self._net_infer = self.net.eval()
        return self._net_infer.eval()

    @property
    def beta(self):
        return self.net.x_input.weight.detach().cpu().flatten().numpy()

    @property
    def gamma(self):
        if self.family == 'binary':
            return self.net.z_input.weight.detach().cpu().numpy()[1, :]
        return self.net.z_input.weight.detach().cpu().flatten().numpy()

    def summary(self, include_beta=True, include_gamma=True, include_spline=False):
        if self.net is None:
            raise ValueError("Model has not been fitted yet.")
        blocks = {}
        if include_beta:
            blocks['beta'] = dict(coeff=self.beta,
                                  se=getattr(self, 'beta_se', None),
                                  lb=getattr(self, 'beta_lb', None),
                                  ub=getattr(self, 'beta_ub', None))
        if include_gamma:
            blocks['gamma'] = dict(coeff=self.gamma,
                                   se=getattr(self, 'gamma_se', None),
                                   lb=getattr(self, 'gamma_lb', None),
                                   ub=getattr(self, 'gamma_ub', None))
        return self._build_summary(blocks, ['beta', 'gamma'])

    def g_function(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        net = self._infer_net()
        with torch.no_grad():
            tx = torch.from_numpy(x).to(self.device)
            result = net.g_function(tx)
            if self.family == 'binary':
                return result[:, 1].cpu().numpy() - result[:, 0].cpu().numpy()
            return result.view(-1).cpu().numpy()

    def predict_gxb(self, X, batch_size=128):
        net = self._infer_net()
        net.eval()
        X = np.asarray(X, dtype=np.float32)
        ds = TensorDataset(torch.from_numpy(X))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        outs = []
        with torch.no_grad():
            for (bx,) in dl:
                bx = bx.to(self.device)
                result = net.gxb(bx)
                if self.family == 'binary':
                    outs.append((result[:, 1] - result[:, 0]).cpu())
                else:
                    outs.append(result.view(-1).cpu())
        return torch.cat(outs, 0).numpy()

    def predict(self, X, Z, batch_size=128):
        net = self._infer_net()
        net.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Z.astype(np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        outs = []
        with torch.no_grad():
            for bx, bz in dl:
                bx = bx.to(self.device)
                bz = bz.to(self.device)
                result = net(bx, bz)
                if self.family == 'binary':
                    outs.append((result[:, 1] - result[:, 0]).cpu())
                else:
                    outs.append(result.view(-1).cpu())
        return torch.cat(outs, 0).numpy()

    def predict_proba(self, X, Z, batch_size=128):
        net = self._infer_net()
        net.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Z.astype(np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        outs = []
        with torch.no_grad():
            for bx, bz in dl:
                bx = bx.to(self.device)
                bz = bz.to(self.device)
                result = net(bx, bz)
                outs.append(result.softmax(dim=1)[:, 1].cpu())
        return torch.cat(outs, 0).numpy()

    def predict_partial_hazard(self, X, Z, batch_size=128):
        net = self._infer_net()
        net.eval()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Z.astype(np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        outs = []
        with torch.no_grad():
            for bx, bz in dl:
                bx = bx.to(self.device)
                bz = bz.to(self.device)
                outs.append(net(bx, bz).view(-1).exp().cpu())
        return torch.cat(outs, 0).numpy()

    def _refit_one(self, Xb, Zb, yb, seed):
        torch.manual_seed(seed)
        p, q = Xb.shape[1], Zb.shape[1]
        n_classes = 2 if self.family == 'binary' else 1
        net = _nPLSInet(p, q, hidden_units=self.hidden_units, n_hidden_layers=self.n_hidden_layers, n_classes=n_classes).to(self.device)
        net = self._train(net, Xb, Zb, yb, self.family, self.device,
                          batch_size=self.batch_size, max_epoch=self.max_epoch,
                          learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                          momentum=self.momentum, grad_clip=self.grad_clip,
                          label_smoothing=self.label_smoothing, random_state=seed)
        beta = net.x_input.weight.detach().cpu().flatten().numpy()
        if self.family == 'binary':
            gamma = net.z_input.weight.detach().cpu().numpy()[1, :]
        else:
            gamma = net.z_input.weight.detach().cpu().flatten().numpy()
        return beta, gamma, net

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs='auto'):
        X = np.asarray(X, dtype=np.float32); Z = np.asarray(Z, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        if self.net is None:
            self.fit(X, Z, y)
        N, p, q = len(X), X.shape[1], Z.shape[1]

        if n_jobs == 'auto':
            if self.device.type == 'cpu':
                n_jobs = -1
                print(f"Auto-detected CPU device: using parallel bootstrap with all cores (n_jobs=-1)")
            else:
                n_jobs = 1
                print(f"Auto-detected GPU device: using sequential bootstrap (n_jobs=1)")

        beta_samples = np.empty((n_samples, p))
        gamma_samples = np.empty((n_samples, q))

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=np.float32).reshape(-1, 1)
            g_samples = np.empty((n_samples, g_grid.shape[0]))

        if n_jobs == 1:
            rng = np.random.default_rng(random_state)
            for b in range(n_samples):
                idx = draw_bootstrap_indices(N, rng)
                Xb, Zb, yb = X[idx], Z[idx], y[idx]
                beta_b, gamma_b, net_b = self._refit_one(Xb, Zb, yb, random_state + 1337 + b)
                beta_samples[b] = beta_b; gamma_samples[b] = gamma_b

                if do_g:
                    net_b.eval()
                    with torch.no_grad():
                        gx = net_b.g_function(torch.from_numpy(g_grid).to(self.device)).view(-1).cpu().numpy()
                    g_samples[b] = gx
        else:
            print(f"Running parallel bootstrap with n_jobs={n_jobs} (CPU-optimized)")

            family = self.family
            device_type = self.device.type
            batch_size = self.batch_size
            max_epoch = self.max_epoch
            learning_rate = self.learning_rate
            weight_decay = self.weight_decay
            momentum = self.momentum
            grad_clip = self.grad_clip
            label_smoothing = self.label_smoothing
            hidden_units = self.hidden_units
            n_hidden_layers = self.n_hidden_layers

            def refit_wrapper(Xb, Zb, yb, seed):
                device = torch.device(device_type)
                p, q = Xb.shape[1], Zb.shape[1]
                n_classes = 2 if family == 'binary' else 1

                torch.manual_seed(seed)
                net_b = _nPLSInet(p, q, hidden_units=hidden_units, n_hidden_layers=n_hidden_layers, n_classes=n_classes).to(device)
                net_b = neuralPLSI._train(
                    net_b, Xb, Zb, yb, family, device,
                    batch_size=batch_size, max_epoch=max_epoch,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    momentum=momentum, grad_clip=grad_clip,
                    label_smoothing=label_smoothing, random_state=seed
                )

                beta_b = net_b.x_input.weight.detach().cpu().flatten().numpy()
                if family == 'binary':
                    gamma_b = net_b.z_input.weight.detach().cpu().numpy()[1, :]
                else:
                    gamma_b = net_b.z_input.weight.detach().cpu().flatten().numpy()

                if do_g:
                    net_b.eval()
                    with torch.no_grad():
                        gx = net_b.g_function(torch.from_numpy(g_grid).to(device)).view(-1).cpu().numpy()
                    return beta_b, gamma_b, gx
                return beta_b, gamma_b, None

            results = run_parallel_bootstrap(refit_wrapper, X, Z, y, n_samples, random_state, n_jobs)
            for b, result in enumerate(results):
                if do_g:
                    beta_samples[b], gamma_samples[b], g_samples[b] = result
                else:
                    beta_samples[b], gamma_samples[b] = result[0], result[1]

        beta_hat, gamma_hat = self.beta, self.gamma

        self.beta_se = beta_samples.std(axis=0, ddof=1)
        self.gamma_se = gamma_samples.std(axis=0, ddof=1)
        self.beta_lb, self.beta_ub = self._percentile_ci(beta_samples, ci)
        self.gamma_lb, self.gamma_ub = self._percentile_ci(gamma_samples, ci)

        out = {
            "beta_hat": beta_hat, "beta_se": self.beta_se, "beta_lb": self.beta_lb, "beta_ub": self.beta_ub,
            "gamma_hat": gamma_hat, "gamma_se": self.gamma_se, "gamma_lb": self.gamma_lb, "gamma_ub": self.gamma_ub,
            "beta_samples": beta_samples, "gamma_samples": gamma_samples
        }

        if do_g:
            g_mean = g_samples.mean(axis=0); g_se = g_samples.std(axis=0, ddof=1)
            g_lb, g_ub = self._percentile_ci(g_samples, ci)
            self.g_grid = g_grid.ravel(); self.g_grid_mean = g_mean; self.g_grid_se = g_se
            self.g_grid_lb, self.g_grid_ub = g_lb, g_ub; self._g_samples = g_samples
            out.update({"g_grid": self.g_grid, "g_mean": g_mean, "g_se": g_se, "g_lb": g_lb, "g_ub": g_ub})
        return out

    def summary(self):
        blocks = {}
        if self.beta is not None:
            blocks['beta'] = {'coeff': self.beta, 'se': getattr(self, 'beta_se', None),
                             'lb': getattr(self, 'beta_lb', None), 'ub': getattr(self, 'beta_ub', None)}
        if self.gamma is not None:
            blocks['gamma'] = {'coeff': self.gamma, 'se': getattr(self, 'gamma_se', None),
                              'lb': getattr(self, 'gamma_lb', None), 'ub': getattr(self, 'gamma_ub', None)}
        return self._build_summary(blocks, ['beta', 'gamma'])

