"""Sandwich standard errors for a fitted SplinePLSI (continuous and binary outcomes).

The fit's knots sit at quantiles of X @ beta and move with beta; here they are
held at their fitted values, as for a regression spline, so the model is
g(u) = B(u) c with the fitted knot vector. The objective is the engine's
(core/src/spline.cpp): RSS + alpha |w|^2 for continuous outcomes, negative
log-likelihood + alpha |w|^2 / 2 for binary ones, w = (gamma, c). beta is
parameterized in the tangent plane of the unit sphere at beta_hat, as in
sandwich.py, and Var = H^-1 C H^-1 with H the Hessian of the objective and C
the sum of the centered outer products of the per-row score.

With fixed knots, the fit's beta need not be a stationary point (its profile
optimum moved the knots, and BFGS stops at a tolerance), so polish() takes
Newton steps on the fixed-knot objective before the sandwich.
"""
import numpy as np
import torch
from torch.func import grad, hessian, vmap

torch.set_default_dtype(torch.float64)


class SplineObjective:
    def __init__(self, model, X, Z, y):
        if model.family not in ('continuous', 'binary'):
            raise ValueError('Cox outcomes are not supported here')
        self.family, self.alpha, self.degree = model.family, float(model.alpha), int(model.spline_degree)
        self.t = torch.tensor(np.asarray(model.knot_vector, float))
        self.X, self.Z = torch.tensor(np.asarray(X, float)), torch.tensor(np.asarray(Z, float))
        self.y = torch.tensor(np.asarray(y, float)).reshape(-1)
        self.p, self.q, self.nb = self.X.shape[1], self.Z.shape[1], len(model.spline_coeffs)
        self.reset(torch.tensor(np.asarray(model.beta, float)), torch.tensor(np.asarray(model.gamma, float)),
                   torch.tensor(np.asarray(model.spline_coeffs, float)))

    def reset(self, beta, gamma, coeffs):
        self.b = beta/beta.norm()
        self.T = torch.linalg.svd(torch.eye(self.p)-torch.outer(self.b, self.b))[0][:, :self.p-1]
        self.phi = torch.cat([torch.zeros(self.p-1), gamma, coeffs])

    def basis(self, u):
        """The engine's Cox-de Boor recursion (basis_row), vectorized."""
        t, nb = self.t, self.nb
        # Clipped as the engine clips, with the derivative passed through: the
        # boundary knots are the smallest and largest index, so two rows sit on
        # them, where clipping has a kink; they take the boundary polynomial's
        # derivative from the inside, as a smooth local model at the fit.
        u = (u+(torch.clamp(u, t[0].item(), t[-1].item())-u).detach())[:, None]
        i = torch.arange(nb)
        b = (((u >= t[i]) & (u < t[i+1])) | ((i == nb-1) & (u == t[i+1]))).to(u.dtype)
        for d in range(1, self.degree+1):
            d1, d2 = t[i+d]-t[i], t[torch.clamp(i+d+1, max=len(t)-1)]-t[torch.clamp(i+1, max=len(t)-1)]
            left = torch.where(d1 > 0, (u-t[i])/torch.where(d1 > 0, d1, 1.)*b, 0.)
            nxt = torch.cat([b[:, 1:], torch.zeros_like(b[:, :1])], dim=1)
            ok = (i+1 < nb) & (d2 > 0)
            right = torch.where(ok, (t[torch.clamp(i+d+1, max=len(t)-1)]-u)/torch.where(ok, d2, 1.)*nxt, 0.)
            b = left+right
        return b

    def parts(self, phi):
        k = self.p-1
        beta = self.b+self.T @ phi[:k]
        return beta/beta.norm(), phi[k:k+self.q], phi[k+self.q:]

    def per_row(self, phi, X, Z, y):
        beta, gamma, c = self.parts(phi)
        eta = self.basis(X @ beta) @ c+Z @ gamma
        return (eta-y)**2 if self.family == 'continuous' else torch.nn.functional.softplus(eta)-y*eta

    def penalty(self, phi):
        w = phi[self.p-1:]
        return self.alpha*(w @ w)*(1. if self.family == 'continuous' else .5)

    def value(self, phi):
        return self.per_row(phi, self.X, self.Z, self.y).sum()+self.penalty(phi)

    def gradient_norm(self):
        return torch.func.grad(self.value)(self.phi).norm().item()

    def polish(self, steps=20, tolerance=1e-10):
        """Newton steps on the fixed-knot objective, re-centring the tangent plane after each."""
        for _ in range(steps):
            g = torch.func.grad(self.value)(self.phi)
            if g.norm() < tolerance:
                break
            step = torch.linalg.solve(hessian(self.value)(self.phi), g)
            beta, gamma, c = self.parts(self.phi-step)
            self.reset(beta, gamma, c)
        return self

    def sandwich(self):
        H = hessian(self.value)(self.phi)
        row = lambda f, x, z, y: self.per_row(f, x[None], z[None], y[None])[0]
        S = vmap(grad(row), in_dims=(None, 0, 0, 0))(self.phi, self.X, self.Z, self.y)
        S = S-S.mean(dim=0)
        V = torch.linalg.solve(H, torch.linalg.solve(H, S.T @ S).T)
        k = self.p-1
        vbeta = self.T @ V[:k, :k] @ self.T.T
        beta, gamma, _ = self.parts(self.phi)
        return (beta.detach().numpy(), gamma.detach().numpy(), np.sqrt(np.diag(vbeta.numpy())),
                np.sqrt(np.diag(V[k:k+self.q, k:k+self.q].numpy())),
                dict(grad_norm=self.gradient_norm(), min_eig=torch.linalg.eigvalsh(H).min().item()))
