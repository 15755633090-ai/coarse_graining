"""Consistent epsilon-prediction Gaussian schedule; t=0 is exactly clean."""
import torch
from torch import nn


class GaussianSchedule(nn.Module):
    def __init__(self, steps=1000):
        super().__init__()
        if steps < 1:
            raise ValueError("steps must be positive")
        self.steps = steps
        beta = torch.cat((torch.zeros(1), torch.linspace(1e-4, 0.02, steps)))
        self.register_buffer("alpha_bar", torch.cumprod(1 - beta, 0))

    def node_steps(self, graph_steps, batch):
        if graph_steps.dtype != torch.long or graph_steps.ndim != 1:
            raise ValueError("Expected a vector of integer graph timesteps")
        if len(graph_steps) != batch.num_graphs or bool(((graph_steps < 0) | (graph_steps > self.steps)).any()):
            raise ValueError("One timestep in [0,T] is required per molecule")
        return graph_steps[batch.batch]

    def corrupt(self, x0, node_steps, noise):
        if noise.shape != x0.shape or node_steps.shape != (len(x0),):
            raise ValueError("Noise and node timesteps must align with atom rows")
        a = self.coefficient(x0, node_steps)
        return a.sqrt() * x0 + (1 - a).sqrt() * noise

    def reconstruct(self, xt, predicted_noise, node_steps):
        if xt.shape != predicted_noise.shape:
            raise ValueError("Predicted noise must align with noisy atom features")
        a = self.coefficient(xt, node_steps)
        return (xt - (1 - a).sqrt() * predicted_noise) / a.sqrt()

    def coefficient(self, values, node_steps):
        if (node_steps.dtype != torch.long or node_steps.shape != (len(values),)
                or bool(((node_steps < 0) | (node_steps > self.steps)).any())):
            raise ValueError("Expected one integer timestep in [0,T] per atom")
        return self.alpha_bar[node_steps].to(values.dtype)[:, None]
