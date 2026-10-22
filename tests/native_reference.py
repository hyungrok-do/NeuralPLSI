"""Small float64 NumPy reference used only for independent numerical tests."""
import numpy as np
from scipy.special import logsumexp, expit


def forward(config, weights, X, Z, sign=1.):
    p,q,h,depth,intercept,family,activation=map(int,config[:7])
    w=np.asarray(weights,dtype=float)
    epsilon=0. if config[11] else 1e-6
    index=np.asarray(X,dtype=float)@(w[:p]/(np.linalg.norm(w[:p])+epsilon))*sign
    a=np.r_[index,0.].reshape(-1,1)
    position=p+q+1
    for k in range(depth+1):
        width=1 if k==depth else h
        size=a.shape[1]*width
        matrix=w[position:position+size].reshape(width,a.shape[1]); position+=size
        bias=w[position:position+width]; position+=width
        a=a@matrix.T+bias
        if k!=depth:
            if activation==0: a=np.where(a>0,a,np.expm1(a))
            elif activation==1: a=np.where(a>0,a,.01*a)
            elif activation==2: a=np.maximum(a,0)
            else: a=np.tanh(a)
    raw=a[:-1,0]+w[p+q]*index+np.asarray(Z,dtype=float)@w[p:p+q]
    if intercept: raw+=w[-1]
    return raw,a[-1,0]


def loss(config, weights, X, Z, y, sign=1., anchor=True):
    raw,g0=forward(config,weights,X,Z,sign)
    y=np.asarray(y,dtype=float)
    if config[5]==0: result=np.mean((raw-y)**2)
    elif config[5]==1: result=np.mean(np.logaddexp(0,raw)-raw*y)
    else:
        events=y[:,1]
        result=sum(logsumexp(raw[y[:,0]>=y[i,0]])-raw[i]
                   for i in range(len(raw)) if events[i])/events.sum()
    return result+(g0*g0 if anchor else 0.)


def cox_loss_and_grad(Xd_s, E_s, w, alpha, group_ends=None):
    """Breslow partial likelihood + alpha*|w|^2/2 with complete tied risk sets.

    Xd_s and E_s are sorted by decreasing duration. group_ends gives the last
    row of each tied-duration group. Omitting it represents distinct times.
    """
    risk = Xd_s @ w
    n = len(risk)
    if group_ends is None:
        group_ends = np.arange(n)
    starts = np.r_[0, group_ends[:-1]+1]
    counts = group_ends-starts+1
    events = np.add.reduceat(E_s, starts)
    log_risk = np.logaddexp.accumulate(risk)[group_ends]
    denominator = np.repeat(log_risk, counts)
    value = np.sum(E_s*(denominator-risk)) + .5*alpha*(w @ w)
    log_event = np.full(len(events), -np.inf)
    active = events > 0
    log_event[active] = np.log(events[active])-log_risk[active]
    tail = np.logaddexp.accumulate(log_event[::-1])[::-1]
    derivative = np.exp(risk+np.repeat(tail, counts))-E_s
    return float(value), Xd_s.T @ derivative + alpha*w
