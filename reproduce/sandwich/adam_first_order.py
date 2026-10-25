"""Adam on the penalized NeuralPLSI objective, on all rows, without early stopping,
until first-order optimality: the gradient norm of the objective below `tol`.

The objective is the engine's (sandwich.Objective): mean loss + g(0)^2 +
(lambda/2)|network|^2, over all rows. The penalty is part of the objective, not
AdamW's decoupled weight decay, whose fixed points do not solve
grad(loss) + lambda w = 0. Training starts from the engine's initialization
(the same seeds as its fits), keeps beta on the unit sphere after every step, as
the engine does, and decays the learning rate as lr0 / (1 + step/decay). The
gradient on all rows is checked every `check` steps (every epoch for minibatches).
"""
import time
import numpy as np
import torch
from sandwich import Objective


def fit(net, X, Z, y, batch=None, lr=1e-3, decay=2000., tol=1e-4, max_steps=200000, check=None, seed=0):
    obj = Objective(net, X, Z, y)
    w = obj.w.clone().requires_grad_(True)
    opt = torch.optim.Adam([w], lr=lr)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda step: 1./(1.+step/decay))
    n, p = len(obj.y), obj.p
    rng = torch.Generator().manual_seed(seed)
    full = lambda v: obj.value(v)
    check = check or (1 if batch is None else max(1, n//batch))
    order, at, start, gnorm = torch.randperm(n, generator=rng), 0, time.time(), float('inf')
    for step in range(1, max_steps+1):
        opt.zero_grad()
        if batch is None:
            loss = full(w)
        else:
            if at+batch > n:
                order, at = torch.randperm(n, generator=rng), 0
            rows = order[at:at+batch]; at += batch
            loss = obj.per_row(w, obj.X[rows], obj.Z[rows], obj.y[rows]).mean()+obj.extra(w)
        loss.backward()
        opt.step(); sched.step()
        with torch.no_grad():
            w[:p] /= w[:p].norm()
        if step % check == 0:
            g = torch.func.grad(full)(w.detach())
            # Only the tangent part of beta's gradient counts: the objective ignores beta's scale.
            b = w[:p].detach()
            g[:p] -= (g[:p] @ b)*b
            gnorm = g.norm().item()
            if gnorm < tol:
                break
    return w.detach(), dict(steps=step, grad_norm=gnorm, seconds=time.time()-start, objective=full(w.detach()).item())
