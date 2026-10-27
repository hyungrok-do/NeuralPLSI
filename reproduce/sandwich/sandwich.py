"""Sandwich standard errors for a fitted NeuralPLSI (continuous, binary and Cox outcomes).

The fit minimizes Q(w) = mean_i l_i(w) + g(0)^2 + (lambda/2) |network|^2 (the
engine's objective, core/src/neuralplsi.cpp). At a converged fit, the M-estimator
sandwich is Var(w_hat) ~ H^-1 Cov(s_i) H^-1 / n, with H the Hessian of Q and s_i
the gradient of l_i. beta is only identified up to scale, so it is parameterized
in the tangent plane of the unit sphere at beta_hat: beta = b + T d, with T an
orthonormal basis of the complement of b; Var(beta) = T Var(d) T' / |b|^2.

For Cox outcomes the loss is the Breslow negative log partial likelihood over
the number of events E, and the rows' scores are Lin and Wei's score residuals,
computed here directly from their definition, one event time at a time; then
Var = H^-1 (sum U_i U_i') H^-1 / E^2.
"""
import numpy as np
import torch
from torch.func import grad, jacrev, vmap

torch.set_default_dtype(torch.float64)
ACTIVATIONS = {0: torch.nn.functional.elu, 1: lambda v: torch.nn.functional.leaky_relu(v, 0.01),
               2: torch.relu, 3: torch.tanh}


class Objective:
    def __init__(self, net, X, Z, y):
        config, hyper = net.config, net.hyper
        self.p, self.q, self.h, self.depth = int(config[0]), int(config[1]), int(config[2]), int(config[3])
        self.intercept, self.family, self.activation = int(config[4]), int(config[5]), int(config[6])
        self.lam, self.sign = float(hyper[0]), float(hyper[1])
        self.w = torch.tensor(net.weights.astype(np.float64))
        self.X, self.Z = torch.tensor(np.asarray(X, np.float32).astype(np.float64)), torch.tensor(np.asarray(Z, np.float32).astype(np.float64))
        y = np.asarray(y, np.float32).astype(np.float64)
        if self.family == 2:
            self.time, self.event = torch.tensor(y[:, 0]), torch.tensor(y[:, 1])
            self.y = self.event
        else:
            self.y = torch.tensor(y).reshape(-1)
        p, q = self.p, self.q
        # Layers: weights (out x in, row-major), then biases, after beta, gamma and the skip weight.
        self.layers, pos, fan_in = [], p+q+1, 1
        for k in range(self.depth+1):
            out = 1 if k == self.depth else self.h
            self.layers.append((pos, pos+fan_in*out, fan_in, out))
            pos += fan_in*out+out
            fan_in = out
        self.size = pos+self.intercept
        assert self.size == len(self.w)
        # The penalty covers the skip weight and the layers, not beta, gamma or the intercept.
        self.penalized = torch.zeros(self.size)
        self.penalized[p+q:pos] = 1.
        b = self.w[:p]/self.w[:p].norm()
        basis = torch.linalg.svd(torch.eye(p)-torch.outer(b, b))[0][:, :p-1]
        self.b, self.T = self.w[:p].clone(), basis

    def link(self, w, u):
        a = u.reshape(-1, 1)
        for k, (wpos, bpos, fan_in, out) in enumerate(self.layers):
            a = a @ w[wpos:bpos].reshape(out, fan_in).T+w[bpos:bpos+out]
            if k < self.depth:
                a = ACTIVATIONS[self.activation](a)
        return a.reshape(-1)+w[self.p+self.q]*u

    def full(self, phi):
        """The engine's weights from phi = [tangent coordinates of beta, every other weight]."""
        return torch.cat([self.b+self.T @ phi[:self.p-1], phi[self.p-1:]])

    def phi_hat(self):
        return torch.cat([torch.zeros(self.p-1), self.w[self.p:]])

    def predictor(self, w, X, Z):
        beta = w[:self.p]
        u = self.sign*(X @ (beta/(beta.norm()+1e-6)))
        r = self.link(w, u)+Z @ w[self.p:self.p+self.q]
        return r+w[-1] if self.intercept else r

    def cox_loss(self, r):
        """Breslow: sum over events of log sum_{t_j >= t_i} exp(r_j) - r_i, over the events."""
        at_risk = (self.time[None, :] >= self.time[:, None]).to(r.dtype)   # row i: its risk set
        log_s0 = torch.logsumexp(r[None, :]+torch.log(at_risk), dim=1)
        return (self.event*(log_s0-r)).sum()/self.event.sum()

    def per_row(self, w, X, Z, y):
        r = self.predictor(w, X, Z)
        if self.family == 0:
            return (r-y)**2
        return torch.nn.functional.softplus(r)-y*r

    def extra(self, w):
        """The g(0)^2 anchor and the penalty."""
        g0 = self.link(w, torch.zeros(1))[0]-w[self.p+self.q]*0.
        return g0**2+0.5*self.lam*(self.penalized*w*w).sum()

    def value(self, w):
        if self.family == 2:
            return self.cox_loss(self.predictor(w, self.X, self.Z))+self.extra(w)
        return self.per_row(w, self.X, self.Z, self.y).mean()+self.extra(w)

    def cox_scores(self, f):
        """Lin and Wei's score residuals in phi."""
        r = self.predictor(self.full(f), self.X, self.Z).detach()
        J = jacrev(lambda g: self.predictor(self.full(g), self.X, self.Z), chunk_size=64)(f)
        return self.lin_wei(r, J)

    def lin_wei(self, r, J):
        """Score residuals for linear predictors r with gradients J (n x k), one event time at a time."""
        e = torch.exp(r-r.max())
        U = torch.zeros_like(J)
        for t in torch.unique(self.time[self.event == 1]):
            risk = self.time >= t
            mean = (e[risk, None]*J[risk]).sum(0)/e[risk].sum()
            events = ((self.time == t) & (self.event == 1)).to(r.dtype)
            U += events[:, None]*(J-mean)
            U -= (events.sum()*risk.to(r.dtype)*e/e[risk].sum())[:, None]*(J-mean)
        return U, self.event.sum()

    def sandwich(self):
        phi = self.phi_hat()
        Q = lambda f: self.value(self.full(f))
        H = jacrev(jacrev(Q), chunk_size=128)(phi)
        Hinv = torch.linalg.inv(H)
        if self.family == 2:
            U, events = self.cox_scores(phi)
            V = Hinv @ (U.T @ U) @ Hinv/events**2
        else:
            row = lambda f, x, z, y: self.per_row(self.full(f), x[None], z[None], y[None])[0]
            S = vmap(grad(row), in_dims=(None, 0, 0, 0), chunk_size=500)(phi, self.X, self.Z, self.y)
            n = len(self.y)
            C = torch.cov(S.T, correction=0)
            V = Hinv @ C @ Hinv/n
        k = self.p-1
        # phi moves the raw weights w[:p] = b + T d; the unit-length beta moves by
        # T d / |b|. L-BFGS keeps |b| = 1, AdamW does not (about 0.5).
        vbeta = self.T @ V[:k, :k] @ self.T.T/(self.b @ self.b)
        vgamma = V[k:k+self.q, k:k+self.q]
        gnorm = jacrev(Q)(phi).norm().item()
        return (np.sqrt(np.diag(vbeta.numpy())), np.sqrt(np.diag(vgamma.numpy())),
                dict(grad_norm=gnorm, min_eig=torch.linalg.eigvalsh(H).min().item(), params=len(phi)))
