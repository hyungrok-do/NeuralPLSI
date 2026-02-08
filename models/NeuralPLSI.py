import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from .base import _SummaryMixin, draw_bootstrap_indices, run_parallel_bootstrap
from .inference import hessian_inference_beta_gamma, hessian_g_bands
from .utils import SchedulerCallback
import functools

class CoxCCLoss(nn.Module):
    """Negative Cox partial log-likelihood (Breslow approximation).

    Sorts by descending duration and computes log-cumsum-exp for
    numerical stability.  Pass ``presorted=True`` to skip the argsort
    when data is already ordered (e.g. a fixed validation set).
    """
    def __init__(self, presorted=False):
        super().__init__()
        self.presorted = presorted

    def forward(self, risk_scores, targets):
        durations, events = targets[:, 0], targets[:, 1]
        if risk_scores.dim() > 1:
            risk_scores = risk_scores.squeeze(1)

        if not self.presorted:
            idx = torch.argsort(durations, descending=True)
            r = risk_scores[idx]
            e = events[idx]
        else:
            r = risk_scores
            e = events

        n_events = e.sum()
        if n_events == 0:
            return risk_scores.sum() * 0.0   # grad-safe zero

        gamma = r.max()
        log_cumsum_h = (r - gamma).exp().cumsum(0).log().add(gamma)
        return -(r - log_cumsum_h).mul(e).sum() / n_events

class _nPLSInet(nn.Module):
    def __init__(self, p, q, hidden_units=32, n_hidden_layers=2, n_classes=1, add_intercept=False, activation='Tanh'):
        super().__init__()
        self.x_input = nn.Linear(p, 1, bias=False)
        self.z_input = nn.Linear(q, n_classes, bias=False)
        
        act_cls = getattr(nn, activation)

        layers = []
        layers.append(nn.Linear(1, hidden_units))
        layers.append(act_cls())
        for _ in range(n_hidden_layers - 1):
            layers.append(nn.Linear(hidden_units, hidden_units))
            layers.append(act_cls())
        layers.append(nn.Linear(hidden_units, n_classes))
        self.g_network = nn.Sequential(*layers)
        
        self.flip_sign = False
        self.n_classes = n_classes
        self.add_intercept = add_intercept
        if self.add_intercept:
            self.intercept = nn.Parameter(torch.zeros(n_classes))

    def forward(self, x, z):
        w = self.x_input.weight
        w_norm = w / (w.norm() + 1e-8)
        xb = F.linear(x, w_norm, self.x_input.bias)
        if self.flip_sign: xb = -xb
        out = self.g_network(xb) + self.z_input(z)
        if self.add_intercept:
            out = out + self.intercept
        return out

    def g_function(self, x):
        if self.flip_sign: x = -x
        return self.g_network(x)

    def gxb(self, x):
        w = self.x_input.weight
        w_norm = w / (w.norm() + 1e-8)
        xb = F.linear(x, w_norm, self.x_input.bias)
        if self.flip_sign: xb = -xb
        return self.g_network(xb)

    def resolve_sign_ambiguity(self):
        with torch.no_grad():
            w = self.x_input.weight.data[0]
            if w[0] < 0:
                self.flip_sign = not self.flip_sign
                self.x_input.weight.data[0].mul_(-1.)
                if self.x_input.bias is not None:
                    self.x_input.bias.data.mul_(-1.)


def _refit_nplsi(Xb, Zb, yb, seed, family, device_type, batch_size, max_epoch, learning_rate, weight_decay, grad_clip, hidden_units, n_hidden_layers, do_g, g_grid, add_intercept=False, activation='Tanh'):
    device = torch.device(device_type)
    p, q = Xb.shape[1], Zb.shape[1]

    torch.manual_seed(seed)
    net_b = _nPLSInet(p, q, hidden_units=hidden_units, n_hidden_layers=n_hidden_layers, add_intercept=add_intercept, activation=activation).to(device)
    net_b = NeuralPLSI._train(
        net_b, Xb, Zb, yb, family, device,
        batch_size=batch_size, max_epoch=max_epoch,
        learning_rate=learning_rate, weight_decay=weight_decay,
        grad_clip=grad_clip, random_state=seed
    )

    w = net_b.x_input.weight.detach().cpu().flatten()
    beta_b = (w / (w.norm() + 1e-8)).numpy()
    gamma_b = net_b.z_input.weight.detach().cpu().flatten().numpy()

    intercept_b = None
    if add_intercept:
        intercept_b = net_b.intercept.detach().cpu().flatten().numpy()

    if do_g:
        net_b.eval()
        with torch.no_grad():
            gx_raw = net_b.g_function(torch.from_numpy(g_grid).to(device))
            gx = gx_raw.view(-1).cpu().numpy()
        return beta_b, gamma_b, intercept_b, gx
    return beta_b, gamma_b, intercept_b, None

class NeuralPLSI(_SummaryMixin):
    """
    Neural Partial Linear Single Index Model (NeuralPLSI).
    
    Fits a model of the form:
    g(X @ beta) + Z @ gamma
    
    where g is a neural network, beta and gamma are learned parameters.
    Supports continuous, binary, and time-to-event (Cox) outcomes.
    """
    def __init__(self, family='continuous', max_epoch=200, batch_size=64,
                 learning_rate=1e-3, weight_decay=1e-4,
                 hidden_units=32, n_hidden_layers=2, grad_clip=1.0,
                 num_workers=0, add_intercept=False, activation='Tanh'):
        self.family = family
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.hidden_units = hidden_units
        self.n_hidden_layers = n_hidden_layers
        self.grad_clip = grad_clip
        self.num_workers = num_workers
        self.add_intercept = add_intercept
        self.activation = activation
        self.net = None
        self._net_infer = None

    def fit(self, X, Z, y, random_state=0):
        """
        Fit the NeuralPLSI model.
        
        Args:
            X: Exposure matrix (n_samples, p)
            Z: Covariate matrix (n_samples, q)
            y: Outcome vector (n_samples, ) or (n_samples, 2) for Cox
            random_state: Seed for reproducibility
        """
        torch.manual_seed(random_state)
        p, q = X.shape[1], Z.shape[1]

        self.net = _nPLSInet(p, q, hidden_units=self.hidden_units, 
                            n_hidden_layers=self.n_hidden_layers,
                            add_intercept=self.add_intercept, 
                            activation=self.activation).to(self.device)

        self.net = self._train(self.net, X, Z, y, self.family, self.device,
                               batch_size=self.batch_size, max_epoch=self.max_epoch,
                               learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                               grad_clip=self.grad_clip, random_state=random_state)
        
        self._net_infer = self.net.eval()

    @staticmethod
    def _train(net, X, Z, y, family, device, batch_size=64, max_epoch=200,
               learning_rate=1e-3, weight_decay=1e-4, grad_clip=1.0, random_state=0):
        X = np.asarray(X, dtype=np.float32)
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)

        tr_x, val_x, tr_z, val_z, tr_y, val_y = train_test_split(X, Z, y, test_size=0.2, random_state=random_state)

        opt_g = torch.optim.AdamW(
            [{'params': net.x_input.parameters(), 'weight_decay': 0.},
             {'params': net.g_network.parameters(), 'weight_decay': weight_decay}], lr=learning_rate
        )
        z_params = [{'params': net.z_input.parameters(), 'weight_decay': 0.}]
        if net.add_intercept:
            z_params.append({'params': [net.intercept], 'weight_decay': 0.})
        opt_z = torch.optim.AdamW(z_params, lr=learning_rate * 10)

        if family == 'continuous':
            loss_fn = nn.MSELoss()
        elif family == 'binary':
            loss_fn = nn.BCEWithLogitsLoss()
        elif family == 'cox':
            loss_fn = CoxCCLoss()
        else:
            raise ValueError("family must be 'continuous', 'binary', or 'cox'.")

        # Validation loss: presorted for cox to skip argsort every epoch
        if family == 'cox':
            sort_idx = torch.argsort(torch.from_numpy(val_y[:, 0]), descending=True)
            val_x = val_x[sort_idx.numpy()]
            val_z = val_z[sort_idx.numpy()]
            val_y = val_y[sort_idx.numpy()]
            val_loss_fn = CoxCCLoss(presorted=True)
        else:
            val_loss_fn = loss_fn

        tr_ds = TensorDataset(torch.from_numpy(tr_x), torch.from_numpy(tr_z), torch.from_numpy(tr_y))
        val_ds = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_z), torch.from_numpy(val_y))
        tr_loader = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=len(val_x), shuffle=False, num_workers=0)

        mse0 = nn.MSELoss()

        sch_z = SchedulerCallback(opt_z)
        sch_g = SchedulerCallback(opt_g)

        zero_input = torch.zeros((1, 1), device=device)
        zero_target = torch.zeros((1, net.n_classes), device=device)

        for _ in range(max_epoch):
            net.train()
            for bx, bz, by in tr_loader:
                bx = bx.to(device)
                bz = bz.to(device)
                by = by.to(device)

                opt_g.zero_grad(set_to_none=True); opt_z.zero_grad(set_to_none=True)

                out = net(bx, bz)
                if family == 'binary':
                    loss = loss_fn(out, by.float().view_as(out))
                elif family == 'cox':
                    loss = loss_fn(out.view(-1), by)
                else:
                    loss = loss_fn(out.view(-1), by.view_as(out.view(-1)))
                loss = loss + mse0(net.g_network(zero_input), zero_target)
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), grad_clip)
                opt_g.step(); opt_z.step()

            net.eval(); val_loss = 0.0
            with torch.inference_mode():
                for bx, bz, by in val_loader:
                    bx = bx.to(device)
                    bz = bz.to(device)
                    by = by.to(device)
                    out = net(bx, bz)
                    if family == 'binary':
                        l = val_loss_fn(out, by.float().view_as(out))
                    elif family == 'cox':
                        l = val_loss_fn(out.view(-1), by)
                    else:
                        l = val_loss_fn(out.view(-1), by.view_as(out.view(-1)))
                    l = l + mse0(net.g_network(zero_input), zero_target)
                    val_loss += float(l.item())

            if sch_g(val_loss) and sch_z(val_loss):
                break
        
        net.resolve_sign_ambiguity()
        return net

    def _infer_net(self):
        if self._net_infer is None:
            self._net_infer = self.net.eval()
        return self._net_infer.eval()

    @property
    def beta(self):
        w = self.net.x_input.weight.detach().cpu().flatten()
        return (w / (w.norm() + 1e-8)).numpy()

    @property
    def gamma(self):
        return self.net.z_input.weight.detach().cpu().flatten().numpy()

    @property
    def intercept_val(self):
        if self.net is not None and self.add_intercept:
            return self.net.intercept.detach().cpu().flatten().numpy()
        return None

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
        """Estimate g(x) for given scalar inputs x."""
        x = np.asarray(x, dtype=np.float32).reshape(-1, 1)
        net = self._infer_net()
        with torch.no_grad():
            tx = torch.from_numpy(x).to(self.device)
            result = net.g_function(tx)
            return result.view(-1).cpu().numpy()

    def predict_gxb(self, X, batch_size=128):
        """Predict g(X @ beta)."""
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
                outs.append(result.view(-1).cpu())
        return torch.cat(outs, 0).numpy()

    def _batched_forward(self, X, Z, transform=None, batch_size=128):
        """Shared batched inference. transform is applied to each batch output."""
        net = self._infer_net()
        ds = TensorDataset(torch.from_numpy(X.astype(np.float32)), torch.from_numpy(Z.astype(np.float32)))
        dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        outs = []
        with torch.no_grad():
            for bx, bz in dl:
                out = net(bx.to(self.device), bz.to(self.device)).view(-1)
                if transform is not None:
                    out = transform(out)
                outs.append(out.cpu())
        return torch.cat(outs, 0).numpy()

    def predict(self, X, Z, batch_size=128):
        """Predict raw output (linear predictor)."""
        return self._batched_forward(X, Z, batch_size=batch_size)

    def predict_proba(self, X, Z, batch_size=128):
        """Predict probabilities (for binary outcome)."""
        return self._batched_forward(X, Z, transform=torch.sigmoid, batch_size=batch_size)

    def predict_partial_hazard(self, X, Z, batch_size=128):
        """Predict partial hazard exp(g(Xb) + Zg) (for Cox outcome)."""
        return self._batched_forward(X, Z, transform=torch.exp, batch_size=batch_size)

    def _refit_one(self, Xb, Zb, yb, seed):
        torch.manual_seed(seed)
        p, q = Xb.shape[1], Zb.shape[1]

        net = _nPLSInet(p, q, hidden_units=self.hidden_units, n_hidden_layers=self.n_hidden_layers, add_intercept=self.add_intercept, activation=self.activation).to(self.device)
        net = self._train(net, Xb, Zb, yb, self.family, self.device,
                          batch_size=self.batch_size, max_epoch=self.max_epoch,
                          learning_rate=self.learning_rate, weight_decay=self.weight_decay,
                          grad_clip=self.grad_clip, random_state=seed)
        beta = net.x_input.weight.detach().cpu().flatten().numpy()
        gamma = net.z_input.weight.detach().cpu().flatten().numpy()
        return beta, gamma, net

    def inference_bootstrap(self, X, Z, y, n_samples=200, random_state=0, ci=0.95, g_grid=None, n_jobs='auto'):
        """
        Perform bootstrap inference to estimate SE and CI for beta, gamma, and g(x).
        """
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
        intercept_samples = np.empty((n_samples, 1)) if self.add_intercept else None

        do_g = g_grid is not None
        if do_g:
            g_grid = np.asarray(g_grid, dtype=np.float32).reshape(-1, 1)
            g_samples = np.empty((n_samples, g_grid.shape[0]))
        else:
            g_grid = None

        print(f"Running bootstrap with n_jobs={n_jobs}")

        refit_func = functools.partial(_refit_nplsi,
                                       family=self.family,
                                       device_type=self.device.type,
                                       batch_size=self.batch_size,
                                       max_epoch=self.max_epoch,
                                       learning_rate=self.learning_rate,
                                       weight_decay=self.weight_decay,
                                       grad_clip=self.grad_clip,
                                       hidden_units=self.hidden_units,
                                       n_hidden_layers=self.n_hidden_layers,
                                       do_g=do_g,
                                       g_grid=g_grid,
                                       add_intercept=self.add_intercept,
                                       activation=self.activation)

        results = run_parallel_bootstrap(refit_func, X, Z, y, n_samples, random_state, n_jobs)
        for b, result in enumerate(results):
            beta_res, gamma_res, intercept_res, g_res = result
            beta_samples[b] = beta_res
            gamma_samples[b] = gamma_res
            if self.add_intercept and intercept_res is not None:
                intercept_samples[b] = intercept_res
            if do_g and g_res is not None:
                g_samples[b] = g_res

        beta_hat, gamma_hat = self.beta, self.gamma
        intercept_hat = self.intercept_val

        self.beta_se = beta_samples.std(axis=0, ddof=1)
        self.gamma_se = gamma_samples.std(axis=0, ddof=1)
        self.beta_lb, self.beta_ub = self._percentile_ci(beta_samples, ci)
        self.gamma_lb, self.gamma_ub = self._percentile_ci(gamma_samples, ci)
        
        out = {
            "beta_hat": beta_hat, "beta_se": self.beta_se, "beta_lb": self.beta_lb, "beta_ub": self.beta_ub,
            "gamma_hat": gamma_hat, "gamma_se": self.gamma_se, "gamma_lb": self.gamma_lb, "gamma_ub": self.gamma_ub,
            "beta_samples": beta_samples, "gamma_samples": gamma_samples
        }
        
        if self.add_intercept:
            self.intercept_se = intercept_samples.std(axis=0, ddof=1)
            self.intercept_lb, self.intercept_ub = self._percentile_ci(intercept_samples, ci)
            out.update({
                "intercept_hat": intercept_hat, "intercept_se": self.intercept_se,
                "intercept_lb": self.intercept_lb, "intercept_ub": self.intercept_ub,
                "intercept_samples": intercept_samples
            })

        if do_g:
            g_mean = g_samples.mean(axis=0); g_se = g_samples.std(axis=0, ddof=1)
            g_lb, g_ub = self._percentile_ci(g_samples, ci)
            self.g_grid = g_grid.ravel(); self.g_grid_mean = g_mean; self.g_grid_se = g_se
            self.g_grid_lb, self.g_grid_ub = g_lb, g_ub; self._g_samples = g_samples
            out.update({"g_grid": self.g_grid, "g_mean": g_mean, "g_se": g_se, "g_lb": g_lb, "g_ub": g_ub})
        return out

    def inference_hessian(self, X, Z, y, batch_size=None, damping=1e-3, max_cg_it=200, tol=1e-5, z_alpha=1.959963984540054):
        """
        Perform Hessian-based inference for beta and gamma.
        Delegates to hessian_inference_beta_gamma in models.inference.
        """
        return hessian_inference_beta_gamma(self, X, Z, y, batch_size=batch_size, 
                                            damping=damping, max_cg_it=max_cg_it, tol=tol, z_alpha=z_alpha)

    def inference_hessian_g(self, X, Z, y, mode="g_of_t", g_grid=None, X_eval=None, 
                            batch_size=None, include_beta=False, 
                            damping=1e-3, max_cg_it=200, tol=1e-5, ci=0.95, 
                            simultaneous=False, n_draws=1000, random_state=0):
        """
        Calculate confidence bands for g(t) or g(x.beta) using Hessian sandwich.
        Delegates to hessian_g_bands in models.inference.
        """
        return hessian_g_bands(self, X, Z, y, mode=mode, g_grid=g_grid, X_eval=X_eval,
                               batch_size=batch_size, include_beta=include_beta,
                               damping=damping, max_cg_it=max_cg_it, tol=tol, ci=ci,
                               simultaneous=simultaneous, n_draws=n_draws, random_state=random_state)
