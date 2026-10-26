"""The frozen-network correction of a NeuralPLSI fit: one Newton step on beta and gamma alone.

theta = (beta in the tangent plane of the unit sphere, gamma) is 10 of the fit's
~1,160 coordinates (sandwich.Objective); the network and the intercept are held
at their fitted values. With Q the penalized objective restricted to theta,

    theta_1 = theta_0 - H^-1 grad Q,   Var(theta_1) = H^-1 C H^-1 / n,

H the 10 x 10 Hessian of Q and C the covariance of the rows' scores, all in
theta. Nothing of the network's size is formed: H takes 10 Hessian-vector
products and the scores 10 forward-mode passes, so the cost is about 30 passes
over the data at any network size. Holding the network fixed ignores how it
could absorb a change of beta, so the SEs are somewhat smaller than the full
sandwich's (1-7% at converged L-BFGS fits in the simulation design).
"""
import time
import torch
from torch.func import grad, jvp


def frozen(obj):
    """The corrected beta and gamma of the fit in obj, and their SEs."""
    k, q, n = obj.p-1, obj.q, len(obj.y)
    m = k+q
    phi = obj.phi_hat()
    eta = phi[m:]
    w_of = lambda th: obj.full(torch.cat([th, eta]))
    Q = lambda th: obj.value(w_of(th))
    L = lambda th: obj.per_row(w_of(th), obj.X, obj.Z, obj.y)
    th0 = phi[:m].clone()
    start = time.time()
    gQ = grad(Q)
    g = gQ(th0)
    basis = torch.eye(m, dtype=torch.float64)
    H = torch.stack([jvp(gQ, (th0,), (e,))[1] for e in basis], 1)
    S = torch.stack([jvp(L, (th0,), (e,))[1] for e in basis], 1)
    S = S-S.mean(0)
    Hi = torch.linalg.inv((H+H.T)/2)
    step = Hi @ g
    V = Hi @ (S.T @ S) @ Hi/n**2
    # The unit-length beta moves by T d / |b| (sandwich.py).
    se_beta = torch.sqrt(torch.diag(obj.T @ V[:k, :k] @ obj.T.T))/obj.b.norm()
    se_gamma = torch.sqrt(torch.diag(V[k:, k:]))
    w = w_of(th0-step)
    beta = obj.sign*w[:obj.p]/w[:obj.p].norm()
    beta = beta if beta[0] >= 0 else -beta
    return dict(beta=beta.numpy(), gamma=w[obj.p:obj.p+q].numpy(), se_beta=se_beta.numpy(), se_gamma=se_gamma.numpy(),
                seconds=time.time()-start)
