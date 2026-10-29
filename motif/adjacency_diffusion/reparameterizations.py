"""Experimental decoder coordinate transforms. Core model defaults are unchanged."""
import torch


def decode(raw, noisy, t, pairs, schedule, mode):
    a = schedule.coefficient(noisy, t)
    if bool((t == 0).any()):
        raise ValueError("Experimental denoising transforms require t>0")
    if mode == "epsilon":
        epsilon = raw
        clean = schedule.reconstruct(noisy, epsilon, t, pairs)
    elif mode == "preconditioned_epsilon":
        # epsilon = A_t/sqrt(1-a) - sqrt(a/(1-a))*f_theta.
        # Algebraic cancellation yields A_hat=f_theta. Compute that stable form;
        # this is equivalent to direct A prediction, not an independent method.
        clean = raw * pairs
        epsilon = (noisy-a.sqrt()*clean)/(1-a).sqrt() * pairs
    elif mode == "velocity":
        # v=sqrt(a)*epsilon-sqrt(1-a)*A_0 is an orthogonal coordinate transform.
        clean = (a.sqrt()*noisy-(1-a).sqrt()*raw) * pairs
        epsilon = ((1-a).sqrt()*noisy+a.sqrt()*raw) * pairs
    elif mode == "bounded_A":
        clean = raw.sigmoid() * pairs
        epsilon = (noisy-a.sqrt()*clean)/(1-a).sqrt() * pairs
    else:
        raise ValueError(mode)
    return clean, epsilon


def reconstruction_objective(clean, target, pairs, positive_rate, balanced=False, degree_weight=0.):
    error = (clean-target).square() * pairs
    if balanced:
        p = min(max(float(positive_rate), 1e-6), 1-1e-6)
        weights = torch.where(target > .5, (1-p)/p, 1.) / (2*(1-p))
        error = error * weights
    base = error.sum((1, 2)).mean()
    if degree_weight:
        # Target degree is a LOSS label only, never an encoder-facing condition.
        degree = ((clean.sum(-1)-target.sum(-1)).square()).sum(-1).mean()
        return base + degree_weight*degree
    return base


@torch.no_grad()
def degree_prior_baseline(condition, fallback_rate, iterations=20):
    """A condition-only baseline using total-degree minus H count from clean X.

    No target adjacency or BRICS membership is read. This is not an exact graph
    decoder: symmetric probability scaling only approximately matches degrees.
    """
    x=condition.x
    degree_class=x[:,:,101:108].argmax(-1)
    hydrogen_class=x[:,:,119:125].argmax(-1)
    known=(degree_class<6)&(hydrogen_class<5)
    n=condition.nodes.sum(-1,keepdim=True).to(x.dtype)
    fallback=fallback_rate*(n-1).clamp_min(0)
    degree=torch.where(known,(degree_class-hydrogen_class).to(x.dtype),fallback)
    degree=degree.clamp_min(0).minimum((n-1).clamp_min(0))*condition.nodes
    prob=degree[:,:,None]*degree[:,None,:]/degree.sum(-1).clamp_min(1)[:,None,None]
    prob=prob*condition.pairs
    for _ in range(iterations):
        scale=(degree/prob.sum(-1).clamp_min(1e-8)).sqrt()
        prob=(prob*scale[:,:,None]*scale[:,None,:]).clamp(0,1)*condition.pairs
    return prob
