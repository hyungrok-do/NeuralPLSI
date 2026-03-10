
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

class HessianEngine:
    """
    Helper class to compute Sandwich Covariance Matrix (H^-1 G H^-1) / n
    for a specified set of parameters.
    """
    def __init__(self, model, X, Z, y, batch_size=None):
        self.model = model
        self.device = model.device
        self.params = []
        
        X = np.asarray(X, dtype=np.float32)
        Z = np.asarray(Z, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32)
        
        self.n = len(X)
        if batch_size is None:
            batch_size = model.batch_size
        
        ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(Z), torch.from_numpy(y))
        self.dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)
        
    def _per_sample_losses(self, bx, bz, by, net):
        out = net(bx, bz).view(-1)
        if self.model.family == 'continuous':
            return (out - by.view(-1))**2
        elif self.model.family == 'binary':
            return torch.nn.functional.binary_cross_entropy_with_logits(
                out, by.float().view(-1), reduction='none'
            )
        elif self.model.family == 'cox':
            durations, events = by[:, 0], by[:, 1]
            idx = torch.argsort(durations, descending=True)
            r = out[idx]; e = events[idx]
            gamma = r.max()
            log_cumsum = torch.log(torch.cumsum(torch.exp(r - gamma), 0)) + gamma
            contrib_sorted = -(r - log_cumsum) * e
            contrib = torch.zeros_like(contrib_sorted); contrib.scatter_(0, idx, contrib_sorted)
            return contrib
        else:
            raise ValueError("family must be 'continuous', 'binary', or 'cox'.")

    def compute_covariance(self, param_list, damping=1e-3, max_cg_it=200, tol=1e-5):
        """
        Compute Sigma = (H^-1 G H^-1) / n for parameters in param_list.
        """
        net = self.model._infer_net()
        net.eval()
        
        # Helper to flatten/unflatten
        shapes = [p.shape for p in param_list]
        sizes = [int(p.numel()) for p in param_list]
        idxs = np.cumsum([0] + sizes)
        d = idxs[-1]
        
        def _flatten(tup):
            return torch.cat([p.reshape(-1) for p in tup])

        def _split_like(v):
            parts = []
            for i in range(len(param_list)):
                parts.append(v[idxs[i]:idxs[i+1]].view(shapes[i]))
            return tuple(parts)

        # 1. Compute G = (1/n) * sum_i [grad_i @ grad_i^T]
        #    = (1/n) * J^T @ J  where J is the (n, d) Jacobian
        G = torch.zeros((d, d), device=self.device)
        m = 0

        for bx, bz, by in self.dl:
            bx = bx.to(self.device); bz = bz.to(self.device); by = by.to(self.device)
            losses = self._per_sample_losses(bx, bz, by, net)
            B = losses.numel()

            grads_list = []
            for li in losses:
                net.zero_grad(set_to_none=True)
                g = torch.autograd.grad(li, param_list, retain_graph=True, create_graph=False)
                grads_list.append(_flatten(g))
            J = torch.stack(grads_list)  # (B, d)
            G += J.T @ J
            m += B
        G = (G / m).detach()

        # 2. HVP function
        def avg_loss():
            L = 0.0
            for bx, bz, by in self.dl:
                bx = bx.to(self.device); bz = bz.to(self.device); by = by.to(self.device)
                Lb = self._per_sample_losses(bx, bz, by, net).mean()
                L = L + Lb * (bx.shape[0] / self.n)
            return L

        def hvp(v_flat):
            v_parts = _split_like(v_flat)
            net.zero_grad(set_to_none=True)
            L = avg_loss()
            # create_graph=True for 2nd derivative
            grads = torch.autograd.grad(L, param_list, create_graph=True)
            Hv_parts = torch.autograd.grad(grads, param_list, grad_outputs=v_parts, retain_graph=False)
            Hv = _flatten(Hv_parts)
            return Hv + damping * v_flat

        # 3. Invert H
        build_explicit = (d <= 2000)
        if build_explicit:
            H = torch.zeros((d, d), device=self.device)
            I = torch.eye(d, device=self.device)
            for j in range(d):
                H[:, j] = hvp(I[:, j])
            
            # Robust inverse with Cholesky
            try:
                Lc = torch.linalg.cholesky(H)
                H_inv = torch.cholesky_inverse(Lc)
            except RuntimeError:
                # Fallback to pseudo-inverse or diagonal loading if cholesky fails
                H_inv = torch.linalg.pinv(H)
            
            Sigma = (H_inv @ G @ H_inv) / self.n
        else:
            # CG Solve
            def cg_solve(b):
                x = torch.zeros_like(b)
                r = b.clone(); pvec = r.clone()
                rs_old = (r @ r)
                for _ in range(max_cg_it):
                    Ap = hvp(pvec)
                    div = pvec @ Ap
                    if div.abs() < 1e-12: break
                    alpha = rs_old / div
                    x = x + alpha * pvec
                    r = r - alpha * Ap
                    rs_new = (r @ r)
                    if rs_new.sqrt() < tol: break
                    pvec = r + (rs_new/rs_old) * pvec
                    rs_old = rs_new
                return x

            Z = torch.zeros((d, d), device=self.device)
            for j in range(d):
                Z[:, j] = cg_solve(G[:, j])
            Sigma = torch.zeros((d, d), device=self.device)
            for j in range(d):
                Sigma[:, j] = cg_solve(Z[:, j])
            Sigma = Sigma / self.n
            
        return Sigma.detach().cpu().numpy()


def hessian_inference_beta_gamma(self, X, Z, y, batch_size=None, damping=1e-3, max_cg_it=200, tol=1e-5, z_alpha=1.959963984540054):
    if self.net is None:
        self.fit(X, Z, y)
    
    net = self._infer_net()
    # Parameters: w (raw), gamma
    w_param = net.x_input.weight
    g_param = net.z_input.weight
    params = [w_param, g_param]
    p = w_param.numel()
    q_tot = g_param.numel()
    
    # Handle intercept if present
    intercept_param = getattr(net, 'intercept', None)
    has_intercept = intercept_param is not None
    if has_intercept:
        params.append(intercept_param)
        n_int = intercept_param.numel()
    else:
        n_int = 0
    
    engine = HessianEngine(self, X, Z, y, batch_size=batch_size)
    Sigma_wg = engine.compute_covariance(params, damping=damping, max_cg_it=max_cg_it, tol=tol)
    
    # Delta method: w -> beta = w / ||w||
    with torch.no_grad():
        w_hat = w_param.detach().view(-1)
        rnorm = torch.linalg.norm(w_hat).clamp_min(1e-12)
        beta_hat_t = (w_hat / rnorm)
        beta_hat = beta_hat_t.cpu().numpy()
        
        # orient sign: if first non-zero element is negative, flip
        nz = np.flatnonzero(beta_hat)
        if len(nz) > 0 and beta_hat[nz[0]] < 0:
            beta_hat = -beta_hat
            # Note: Covariance of (-beta) is same as Covariance of (beta)
        
        # Jacobian J = d(beta)/dw
        # J = (I - beta * beta.T) / ||w||
        Iw = torch.eye(p, device=self.device)
        Jw = (Iw - torch.outer(beta_hat_t, beta_hat_t)) / rnorm
        
        # Full Jacobian block diagonal
        # J_full = [ Jw  0   0 ]
        #          [ 0   I_g 0 ]
        #          [ 0   0   I_int ]
        d_full = p + q_tot + n_int
        J_full = torch.zeros((d_full, d_full), device=self.device)
        J_full[:p, :p] = Jw
        J_full[p:p+q_tot, p:p+q_tot] = torch.eye(q_tot, device=self.device)
        if has_intercept:
            J_full[p+q_tot:, p+q_tot:] = torch.eye(n_int, device=self.device)
    
    Sigma_wg_t = torch.from_numpy(Sigma_wg).to(self.device)
    Sigma_bg = (J_full @ Sigma_wg_t @ J_full.T).cpu().numpy()
    
    gamma_hat = g_param.detach().cpu().flatten().numpy()
    
    diag = np.diag(Sigma_bg)
    se_beta = np.sqrt(np.clip(diag[:p], 0.0, np.inf))
    se_gamma = np.sqrt(np.clip(diag[p:p+q_tot], 0.0, np.inf))
    
    beta_lb = beta_hat - z_alpha * se_beta
    beta_ub = beta_hat + z_alpha * se_beta
    gamma_lb = gamma_hat - z_alpha * se_gamma
    gamma_ub = gamma_hat + z_alpha * se_gamma
    
    out = {
        "beta_hat": beta_hat,
        "gamma_hat": gamma_hat,
        "beta_se": se_beta,
        "gamma_se": se_gamma,
        "beta_lb": beta_lb,
        "beta_ub": beta_ub,
        "gamma_lb": gamma_lb,
        "gamma_ub": gamma_ub,
        "cov_beta_gamma": Sigma_bg,
        "cov_w_gamma": Sigma_wg,
        "damping": damping
    }
    
    if has_intercept:
        intercept_hat = intercept_param.detach().cpu().flatten().numpy()
        se_int = np.sqrt(np.clip(diag[p+q_tot:], 0.0, np.inf))
        int_lb = intercept_hat - z_alpha * se_int
        int_ub = intercept_hat + z_alpha * se_int
        out.update({
             "intercept_hat": intercept_hat,
             "intercept_se": se_int,
             "intercept_lb": int_lb,
             "intercept_ub": int_ub
        })
    return out

def hessian_g_bands(self, X, Z, y, mode="g_of_t", g_grid=None, X_eval=None, batch_size=None, include_beta=False, damping=1e-3, max_cg_it=200, tol=1e-5, ci=0.95, simultaneous=True, n_draws=1000, random_state=0):
    torch.manual_seed(random_state); np.random.seed(random_state)
    if self.net is None:
        self.fit(X, Z, y)
        
    net = self._infer_net()
    params_g = list(net.g_network.parameters())
    theta_list = params_g[:]
    if mode == "g_of_xbeta" and include_beta:
        theta_list.insert(0, net.x_input.weight)
        
    engine = HessianEngine(self, X, Z, y, batch_size=batch_size)
    Sigma_theta = engine.compute_covariance(theta_list, damping=damping, max_cg_it=max_cg_it, tol=tol)
    Sigma_theta_t = torch.from_numpy(Sigma_theta).to(self.device)
    
    # Jacobian of target values w.r.t parameters
    d = Sigma_theta.shape[0]
    
    def _flatten(tup):
        return torch.cat([p.reshape(-1) for p in tup])

    if mode == "g_of_t":
        if g_grid is None:
            raise ValueError("g_grid must be provided when mode is 'g_of_t'")
        t = np.asarray(g_grid, dtype=np.float32).reshape(-1, 1)
        t_t = torch.from_numpy(t).to(self.device)
        
        with torch.no_grad():
            g_vals = net.g_function(t_t).view(-1).cpu().numpy()
            
        J = torch.zeros((len(t), d), device=self.device)
        for k in range(len(t)):
            net.zero_grad(set_to_none=True)
            gk = net.g_function(t_t[k:k+1]).view([])
            grads = torch.autograd.grad(gk, theta_list, retain_graph=True)
            J[k, :] = _flatten(grads)
            
        label_x = t.reshape(-1)
        
    elif mode == "g_of_xbeta":
        if X_eval is None:
            raise ValueError("X_eval must be provided when mode is 'g_of_xbeta'")
        Xe = np.asarray(X_eval, dtype=np.float32)
        Xe_t = torch.from_numpy(Xe).to(self.device)
        
        with torch.no_grad():
            g_vals = net.gxb(Xe_t).view(-1).cpu().numpy()
            w = net.x_input.weight
            beta = w / (w.norm() + 1e-8)
            label_x = (Xe_t @ beta.view(-1, 1)).view(-1).cpu().numpy()
            
        J = torch.zeros((len(Xe), d), device=self.device)
        for i in range(len(Xe)):
            net.zero_grad(set_to_none=True)
            gi = net.gxb(Xe_t[i:i+1]).view([])
            grads = torch.autograd.grad(gi, theta_list, retain_graph=True)
            J[i, :] = _flatten(grads)
            
    else:
        raise ValueError(f"Unknown mode {mode}")
        
    # Sort by x coordinate
    order = np.argsort(label_x)
    label_x = label_x[order]
    g_mean = g_vals[order]
    J = J[order]
    
    # Sigma_f = J * Sigma * J.T
    Sigma_f = (J @ Sigma_theta_t @ J.T).detach().cpu().numpy()
    sd = np.sqrt(np.clip(np.diag(Sigma_f), 0.0, np.inf))
    
    # For CI, we need critical value
    # For simultaneous, we simulation
    from scipy.stats import norm
    z = norm.ppf(1 - (1 - ci)/2)
    
    lb = g_mean - z * sd
    ub = g_mean + z * sd
    
    out = {"t": label_x, "g_mean": g_mean, "g_se": sd, "g_lb": lb, "g_ub": ub}
    
    if simultaneous:
        rng = np.random.RandomState(random_state)
        # Avoid full Cholesky of Sigma_f if sparse or large? No, we need samples.
        # Add epsilon for stability
        Sigma_f_stab = Sigma_f + 1e-10 * np.eye(len(Sigma_f))
        try:
            draws = rng.multivariate_normal(np.zeros_like(g_mean), Sigma_f_stab, size=n_draws)
        except np.linalg.LinAlgError:
            # SVD fallback
            u, s, vh = np.linalg.svd(Sigma_f_stab)
            draws = (u * np.sqrt(s)) @ rng.randn(len(s), n_draws)
            draws = draws.T
            
        denom = np.where(sd > 1e-8, sd, 1.0)
        # max |draw / sd|
        max_stat = np.max(np.abs(draws / denom), axis=1)
        c = np.quantile(max_stat, ci)
        
        out["g_lb_simul"] = g_mean - c * sd
        out["g_ub_simul"] = g_mean + c * sd
        
    return out
