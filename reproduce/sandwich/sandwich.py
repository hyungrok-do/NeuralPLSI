"""Sandwich standard errors for a fitted NeuralPLSI (continuous and binary outcomes).

The fit minimizes Q(w) = mean_i l_i(w) + g(0)^2 + (lambda/2) |network|^2 (the
engine's objective, core/src/neuralplsi.cpp). At a converged fit, the M-estimator
sandwich is Var(w_hat) ~ H^-1 Cov(s_i) H^-1 / n, with H the Hessian of Q and s_i
the gradient of l_i. beta is only identified up to scale, so it is parameterized
in the tangent plane of the unit sphere at beta_hat: beta = b + T d, with T an
orthonormal basis of the complement of b; Var(beta) = T Var(d) T'.
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
        if self.family == 2:
            raise ValueError('Cox outcomes are not supported here')
        self.lam, self.sign = float(hyper[0]), float(hyper[1])
        self.w = torch.tensor(net.weights.astype(np.float64))
        self.X, self.Z = torch.tensor(np.asarray(X, np.float32).astype(np.float64)), torch.tensor(np.asarray(Z, np.float32).astype(np.float64))
        self.y = torch.tensor(np.asarray(y, np.float32).astype(np.float64)).reshape(-1)
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

    def per_row(self, w, X, Z, y):
        beta = w[:self.p]
        u = self.sign*(X @ (beta/(beta.norm()+1e-6)))
        r = self.link(w, u)+Z @ w[self.p:self.p+self.q]
        if self.intercept:
            r = r+w[-1]
        if self.family == 0:
            return (r-y)**2
        return torch.nn.functional.softplus(r)-y*r

    def extra(self, w):
        """The g(0)^2 anchor and the penalty."""
        g0 = self.link(w, torch.zeros(1))[0]-w[self.p+self.q]*0.
        return g0**2+0.5*self.lam*(self.penalized*w*w).sum()

    def value(self, w):
        return self.per_row(w, self.X, self.Z, self.y).mean()+self.extra(w)

    def sandwich(self):
        phi = self.phi_hat()
        Q = lambda f: self.value(self.full(f))
        H = jacrev(jacrev(Q), chunk_size=128)(phi)
        row = lambda f, x, z, y: self.per_row(self.full(f), x[None], z[None], y[None])[0]
        S = vmap(grad(row), in_dims=(None, 0, 0, 0), chunk_size=500)(phi, self.X, self.Z, self.y)
        n = len(self.y)
        C = torch.cov(S.T, correction=0)
        Hinv = torch.linalg.inv(H)
        V = Hinv @ C @ Hinv/n
        k = self.p-1
        vbeta = self.T @ V[:k, :k] @ self.T.T
        vgamma = V[k:k+self.q, k:k+self.q]
        gnorm = jacrev(Q)(phi).norm().item()
        return (np.sqrt(np.diag(vbeta.numpy())), np.sqrt(np.diag(vgamma.numpy())),
                dict(grad_norm=gnorm, min_eig=torch.linalg.eigvalsh(H).min().item(), params=len(phi)))
