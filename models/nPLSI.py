# ==========================================================
# neuralPLSI (PyTorch) — eager training + compiled inference
# ==========================================================
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


# --------- Helpers for speed ---------
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


class _SummaryMixin:
    def _build_summary(self, blocks, prefix_order):
        names, coeffs, ses, lbs, ubs = [], [], [], [], []
        for name in prefix_order:
            blk = blocks.get(name)
            if not blk:
                continue
            c = np.asarray(blk.get('coeff', []), dtype=float)
            if c.size == 0:
                continue
            se = blk.get('se')
            lb = blk.get('lb')
            ub = blk.get('ub')
            if se is None: se = np.full_like(c, np.nan, dtype=float)
            if lb is None: lb = np.full_like(c, np.nan, dtype=float)
            if ub is None: ub = np.full_like(c, np.nan, dtype=float)
            if name == 'beta':
                names.extend([f'beta_{i:02d}' for i in range(c.size)])
            elif name == 'gamma':
                names.extend([f'gamma_{i:02d}' for i in range(c.size)])
            else:
                names.extend([f'{name}_{i:02d}' for i in range(c.size)])
            coeffs.append(c); ses.append(np.asarray(se)); lbs.append(np.asarray(lb)); ubs.append(np.asarray(ub))

        if not coeffs:
            return pd.DataFrame(columns=['Parameter', 'Coefficient', 'SE (bootstrap)', 'CI Lower', 'CI Upper'])
        coeffs = np.concatenate(coeffs); ses = np.concatenate(ses); lbs = np.concatenate(lbs); ubs = np.concatenate(ubs)
        return pd.DataFrame({
            'Parameter': names,
            'Coefficient': coeffs,
            'SE (bootstrap)': ses,
            'CI Lower': lbs,
            'CI Upper': ubs
        })

    def _percentile_ci(self, samples, ci=0.95):
        lo = (1 - ci) / 2 * 100
        hi = (1 + ci) / 2 * 100
        return np.percentile(samples, [lo, hi], axis=0)


# --------- Loss (Cox) ---------
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


# --------- Small scheduler/ES ---------
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


# --------- Network ---------
class _nPLSInet(nn.Module):
    def __init__(self, p, q):
        super().__init__()
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, 1, bias=True)
        self.g_network = nn.Sequential(
            nn.Linear(1, 32),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 32),
            nn.SELU(),
            nn.Dropout(0.1),
            nn.Linear(32, 1)
        )
        self.flip_sign = False

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


# --------- Public Model ---------
class neuralPLSI(_SummaryMixin):
    def __init__(self, family='continuous', max_epoch=200, batch_size=64,
                 precompile=True, compile_backend=None, compile_mode=None, num_workers=0):
        self.family = family
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.precompile = precompile
        self.compile_backend = compile_backend
        self.compile_mode = compile_mode
        self.num_workers = num_workers
        self.net = None
        self._net_infer = None
        self._compiled = False
        _enable_fast_matmul()

    # ---------- Fit / Train (eager) ----------
    def fit(self, X, Z, y, random_state=0):
        torch.manual_seed(0)
        p, q = X.shape[1], Z.shape[1]
        self.net = _nPLSInet(p, q).to(self.device)

        # Train in eager mode (normalize_beta is a Python helper)
        self.net = self._train(self.net, X, Z, y, self.family, self.device,
                               batch_size=self.batch_size, max_epoch=self.max_epoch, random_state=random_state)

        # Build a separate inference engine (optional)
        if self.precompile:
            self._maybe_build_inference_engine(self.net, X, Z)
        else:
            self._net_infer = self.net.eval()

    @staticmethod
    def _train(net, X, Z, y, family, device, batch_size=32, max_epoch=100, random_state=0):
        X = np.asarray(X, dtype=np.float32)
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=random_state)
        tr_ds = TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_z), torch.from_numpy(tr_y))
        val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_z), torch.from_numpy(val_y))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=min(batch_size if batch_size else len(val_x), len(val_x)),
                                shuffle=False, num_workers=0)

        opt_g = torch.optim.Adam(
            [{'params': net.x_input.parameters(), 'weight_decay': 0.},
             {'params': net.g_network.parameters(), 'weight_decay': 1e-4}], lr=1e-3
        )
        opt_z = torch.optim.SGD([{'params': net.z_input.parameters()}], lr=1e-3, momentum=0.9, weight_decay=0.)

        if family == 'continuous':
            loss_fn = nn.MSELoss()
        elif family == 'binary':
            loss_fn = nn.BCEWithLogitsLoss()
        elif family == 'cox':
            loss_fn = CoxPHNLLLoss()
        else:
            raise ValueError("family must be 'continuous', 'binary', or 'cox'.")

        mse0 = nn.MSELoss()
        net.normalize_beta(opt_g)
        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)
        scaler = torch.cuda.amp.GradScaler(enabled=(device.type == 'cuda'))

        for _ in range(max_epoch):
            net.train()
            for bx, bz, by in tr_loader:
                bx = bx.to(device)
                bz = bz.to(device)
                by = by.to(device)

                opt_g.zero_grad(set_to_none=True); opt_z.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                    out = net(bx, bz).view(-1)
                    if family == 'binary':
                        target = by.float().view_as(out)
                    elif family == 'cox':
                        target = by
                    else:
                        target = by.view_as(out)
                    loss = loss_fn(out, target)
                    zero = torch.zeros((1, 1), device=device)
                    loss = loss + mse0(net.g_network(zero).view(-1), zero.view(-1))
                scaler.scale(loss).backward()
                scaler.step(opt_g); scaler.step(opt_z); scaler.update()
                net.normalize_beta(opt_g)

            # validation to drive LR scheduler / early stop
            net.eval(); val_loss = 0.0
            with torch.no_grad():
                for bx, bz, by in val_loader:
                    bx = bx.to(device)
                    bz = bz.to(device)
                    by = by.to(device)
                    with torch.cuda.amp.autocast(enabled=(device.type == 'cuda')):
                        out = net(bx, bz).view(-1)
                        if family == 'binary':
                            target = by.float().view_as(out)
                        elif family == 'cox':
                            target = by
                        else:
                            target = by.view_as(out)
                        l = loss_fn(out, target)
                        zero = torch.zeros((1, 1), device=device)
                        l = l + mse0(net.g_network(zero).view(-1), zero.view(-1))
                        val_loss += float(l.item())
                        
            if sch_g(val_loss) and sch_z(val_loss):
                break
        return net

    # ---------- Inference engine (compile → eager) ----------
    def _maybe_build_inference_engine(self, trained_net, X, Z):
        self._net_infer = trained_net.eval()  # default: eager inference
        try:
            if _torch_version_geq("2.0"):
                compiled = torch.compile(
                    trained_net.eval(),
                    backend=self.compile_backend,   # e.g., "inductor"
                    mode=self.compile_mode,         # e.g., "max-autotune"
                    fullgraph=False, dynamic=False
                )
                xb = torch.from_numpy(X[:min(64, len(X))].astype(np.float32)).to(self.device)
                zb = torch.from_numpy(Z[:min(64, len(Z))].astype(np.float32)).to(self.device)
                with torch.no_grad():
                    _ = compiled(xb, zb)  # warmup
                self._net_infer = compiled
                self._compiled = True
                return
        except Exception:
            # If compile fails, we simply fall back to eager inference.
            self._compiled = False

    def _infer_net(self):
        # Always returns an eval()'d module (compiled or eager)
        if self._net_infer is None:
            self._net_infer = self.net.eval()
        return self._net_infer.eval()

    # ---------- Accessors / Summary ----------
    @property
    def beta(self):
        return self.net.x_input.weight.detach().cpu().flatten().numpy()

    @property
    def gamma(self):
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
        # include_spline is ignored for neural model
        return self._build_summary(blocks, ['beta', 'gamma'])

    # ---------- g() and predictions ----------
    def g_function(self, x):
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        net = self._infer_net()
        with torch.no_grad():
            tx = torch.from_numpy(x).to(self.device)
            return net.g_function(tx).view(-1).cpu().numpy()

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
                outs.append(net(bx, bz).view(-1).cpu())
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
                outs.append(net(bx, bz).view(-1).sigmoid().cpu())
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

    # ---------- Bootstrap ----------
    def _draw_bootstrap_indices(self, N, rng, cluster_ids=None):
        if cluster_ids is None:
            return rng.integers(0, N, size=N)
        cluster_ids = np.asarray(cluster_ids)
        unique_c = np.unique(cluster_ids)
        c_sample = rng.choice(unique_c, size=len(unique_c), replace=True)
        idx = np.concatenate([np.where(cluster_ids == c)[0] for c in c_sample])
        if idx.size == 0: idx = rng.integers(0, N, size=N)
        return idx

    def _refit_one(self, Xb, Zb, yb, seed):
        torch.manual_seed(seed)
        p, q = Xb.shape[1], Zb.shape[1]
        net = _nPLSInet(p, q).to(self.device)
        net = self._train(net, Xb, Zb, yb, self.family, self.device,
                          batch_size=self.batch_size, max_epoch=self.max_epoch, random_state=seed)
        beta = net.x_input.weight.detach().cpu().flatten().numpy()
        gamma = net.z_input.weight.detach().cpu().flatten().numpy()
        return beta, gamma, net  # net is eager; safe to call .g_function

    def inference_bootstrap(self, X, Z, y, n_samples=100, random_state=0, ci=0.95, cluster_ids=None, g_grid=None):
        X = np.asarray(X, dtype=np.float32); Z = np.asarray(Z, dtype=np.float32); y = np.asarray(y, dtype=np.float32)
        if self.net is None:
            self.fit(X, Z, y)
        N, p, q = len(X), X.shape[1], Z.shape[1]
        rng = np.random.default_rng(random_state)

        beta_samples = np.empty((n_samples, p))
        gamma_samples = np.empty((n_samples, q))

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=np.float32).reshape(-1, 1)
            g_samples = np.empty((n_samples, g_grid.shape[0]))

        for b in range(n_samples):
            idx = self._draw_bootstrap_indices(N, rng, cluster_ids)
            Xb, Zb, yb = X[idx], Z[idx], y[idx]
            beta_b, gamma_b, net_b = self._refit_one(Xb, Zb, yb, random_state + 1337 + b)
            beta_samples[b] = beta_b; gamma_samples[b] = gamma_b

            if do_g:
                net_b.eval()
                with torch.no_grad():
                    gx = net_b.g_function(torch.from_numpy(g_grid).to(self.device)).view(-1).cpu().numpy()
                g_samples[b] = gx

        # point estimates
        beta_hat, gamma_hat = self.beta, self.gamma

        # se / CI
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

    # ---------- Utility (inherit summary helpers) ----------
    def _build_summary(self, blocks, prefix_order):
        return super()._build_summary(blocks, prefix_order)

    def _percentile_ci(self, samples, ci=0.95):
        return super()._percentile_ci(samples, ci)
