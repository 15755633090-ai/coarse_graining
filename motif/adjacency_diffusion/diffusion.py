"""Symmetric Gaussian adjacency corruption and one-step clean reconstruction."""
import torch
from torch import nn


class Schedule(nn.Module):
    def __init__(self, steps=1000, beta_min=1e-4, beta_max=0.02):
        super().__init__()
        if steps < 1 or not 0 < beta_min <= beta_max < 1:
            raise ValueError("Invalid schedule")
        self.steps, self.beta_min, self.beta_max = steps, beta_min, beta_max
        # Paper Eq. 2 uses t/T, t=1..T (not linspace(beta_min,beta_max,T)).
        t = torch.arange(1, steps + 1, dtype=torch.float64)
        beta = beta_min + t / steps * (beta_max - beta_min)
        alpha = torch.cat((torch.ones(1, dtype=torch.float64), (1-beta).cumprod(0)))
        if not bool((alpha > 0).all()):
            raise ValueError("Schedule underflows; choose a less destructive horizon")
        self.register_buffer("alpha_bar", alpha)

    def description(self):
        return {"steps": self.steps, "beta_min": self.beta_min, "beta_max": self.beta_max,
                "discretization": "beta_min + (t/T)*(beta_max-beta_min), t=1..T",
                "alpha_bar_T": float(self.alpha_bar[-1]),
                "beta_endpoint_values": "explicit implementation choices, not verified paper values"}

    def coefficient(self, values, t):
        if t.dtype != torch.long or t.shape != (values.shape[0],) or bool(((t < 0) | (t > self.steps)).any()):
            raise ValueError("One integer timestep in [0,T] per graph is required")
        return self.alpha_bar[t].to(values.dtype)[:, None, None]

    def sample(self, target, t, pairs, generator=None):
        raw = torch.randn(target.shape, dtype=target.dtype, device=target.device, generator=generator)
        noise = torch.triu(raw, diagonal=1)
        noise = (noise + noise.transpose(1, 2)) * pairs
        a = self.coefficient(target, t)
        return (a.sqrt() * target + (1-a).sqrt() * noise) * pairs, noise

    def reconstruct(self, noisy, epsilon, t, pairs):
        a = self.coefficient(noisy, t)
        return ((noisy - (1-a).sqrt() * epsilon) / a.sqrt()) * pairs
