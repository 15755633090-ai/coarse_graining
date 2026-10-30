"""All-pair noisy propagation; no clean sparse edge support in the encoder."""
from dataclasses import dataclass
import torch
from torch import nn

from .data import ATOM_DIM, Condition, condition_from_batch, adjacency_target
from .diffusion import Schedule
from .path_gates import controls,scaled


@dataclass
class Config:
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    noise_steps: int = 1000
    beta_min: float = 1e-4
    beta_max: float = .02
    motif_prior: bool = False
    conditional_gate: bool = False
    dual_state: bool = True
    time_mode: str = "terminal"
    message_normalization: str = "candidate"
    observation_weighting: str = "softmax"

    def __post_init__(self):
        if self.hidden_dim < 4 or self.layers < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid encoder configuration")
        if self.conditional_gate and not self.motif_prior:
            raise ValueError("Boundary gating requires the explicitly declared BRICS prior")
        if self.time_mode not in ("terminal", "random"):
            raise ValueError("time_mode must be terminal or random")
        if self.message_normalization not in ("candidate", "observation_mass"):
            raise ValueError("Unknown message normalization")
        if self.observation_weighting not in ("softmax", "gaussian_posterior"):
            raise ValueError("Unknown observation weighting")


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def pair_nodes(x):
    n = x.shape[1]
    return x[:, :, None, :].expand(-1, -1, n, -1), x[:, None, :, :].expand(-1, n, -1, -1)


def reduce_messages(messages, mask, noisy, normalization="candidate"):
    # The default retains signed noisy weights and candidate-count normalization.
    # The optional typed mode normalizes received nonnegative weights, never gates
    # or hidden clean degree. True target edges do not prune candidate support.
    weighted = messages * noisy[:, :, :, None] * mask[:, :, :, None]
    if normalization == "observation_mass":
        # Only received weights define the denominator; no target/clean support.
        # This mode is intended for nonnegative typed-edge probability weights.
        denominator=(noisy*mask).sum(2).clamp_min(1e-8)
    else:
        denominator=mask.sum(2).clamp_min(1)
    return weighted.sum(2) / denominator[:, :, None]


class Layer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.dual = config.dual_state
        self.message_normalization = config.message_normalization
        if self.dual:
            self.s_message = mlp(3*d, 2*d, d)
            self.s_update = mlp(2*d, 2*d, d)
            self.s_norm = nn.LayerNorm(d)
        self.h_message = mlp(3*d, 2*d, d)
        self.h_update = mlp(3*d, 2*d, d)
        self.h_norm = nn.LayerNorm(d)
        self.gate = mlp(5*d, d, 1) if config.conditional_gate else None
        self.drop = nn.Dropout(config.dropout)

    def forward(self, s, h, edge, noisy, condition):
        paths=controls(self)
        s=scaled(s,paths.internal_branch)
        a, b = pair_nodes(s)
        hi, hj = pair_nodes(h)
        inside = condition.same_fragment if condition.same_fragment is not None else condition.pairs
        if self.dual:
            incoming = reduce_messages(self.s_message(torch.cat((a, b, edge), -1)), inside, noisy, self.message_normalization)
            next_s = self.s_norm(s + self.drop(self.s_update(torch.cat((s, incoming), -1))))
        else:
            next_s = h
        if self.dual:
            next_s=scaled(next_s,paths.internal_branch)
        gate = noisy.new_ones(*noisy.shape, 1)
        if self.gate is not None and paths.boundary_mode=='learned':
            learned = torch.sigmoid(self.gate(torch.cat((a, b, hi, hj, edge), -1)))
            cross = condition.pairs & ~inside
            gate = torch.where(cross[:, :, :, None], learned, gate)
        message = self.h_message(torch.cat((hi, hj, edge), -1)) * gate
        if paths.context_within!=1 or paths.context_cross!=1:
            route=torch.where(inside,noisy.new_tensor(paths.context_within),noisy.new_tensor(paths.context_cross))
            message=message*route[...,None]
        incoming = reduce_messages(message, condition.pairs, noisy, self.message_normalization)
        next_h = self.h_norm(h + self.drop(self.h_update(torch.cat((h, next_s, incoming), -1))))
        node_mask = condition.nodes[:, :, None]
        return next_s * node_mask, next_h * node_mask, gate.squeeze(-1) * condition.pairs


class Encoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.h_input = nn.Linear(ATOM_DIM, d)
        self.s_input = nn.Linear(ATOM_DIM, d) if config.dual_state else None
        self.time = mlp(3, d, d)
        self.edge = mlp(1, d, d)
        self.layers = nn.ModuleList([Layer(config) for _ in range(config.layers)])
        self.fuse = mlp((2 if config.dual_state else 1)*d, 2*d, d)

    def forward(self, condition, noisy, t):
        if not isinstance(condition, Condition):
            raise TypeError("Encoder accepts Condition only, not a clean graph/target")
        if noisy.shape != condition.pairs.shape or t.shape != (len(noisy),):
            raise ValueError("Noisy adjacency or graph timesteps do not align")
        if (condition.same_fragment is not None) != self.config.motif_prior:
            raise ValueError("Condition/prior mismatch")
        noisy = noisy * condition.pairs
        fraction = t.to(noisy.dtype) / self.config.noise_steps
        embedding = self.time(torch.stack((fraction, torch.sin(fraction*torch.pi), torch.cos(fraction*torch.pi)), -1))[:, None]
        h = (self.h_input(condition.x) + embedding) * condition.nodes[:, :, None]
        s = (self.s_input(condition.x) + embedding) * condition.nodes[:, :, None] if self.s_input is not None else h
        edge = self.edge(noisy[:, :, :, None])
        gates = []
        for layer in self.layers:
            s, h, gate = layer(s, h, edge, noisy, condition)
            gates.append(gate)
        z = self.fuse(torch.cat((s, h), -1) if self.config.dual_state else h) * condition.nodes[:, :, None]
        return {"atoms": z, "internal": s, "context": h, "gates": gates}


class AdjacencyModel(nn.Module):
    def __init__(self, config=None, num_tasks=1):
        super().__init__()
        self.config = config or Config()
        self.num_tasks = num_tasks
        c = self.config
        self.schedule = Schedule(c.noise_steps, c.beta_min, c.beta_max)
        self.encoder = Encoder(c)
        self.noise_head = mlp(3*c.hidden_dim+1, c.hidden_dim, 1)
        self.property_head = mlp(c.hidden_dim, c.hidden_dim, num_tasks)

    def set_stage(self, stage):
        if stage not in ("pretrain", "finetune"):
            raise ValueError(stage)
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        for p in self.noise_head.parameters():
            p.requires_grad_(stage == "pretrain")
        for p in self.property_head.parameters():
            p.requires_grad_(stage == "finetune")

    def predict_noise(self, condition, noisy, t, return_states=False):
        noisy = noisy.masked_fill(~condition.pairs, 0)
        states = self.encoder(condition, noisy, t)
        a, b = pair_nodes(states["atoms"])
        raw = self.noise_head(torch.cat((a+b, (a-b).abs(), a*b, noisy[:, :, :, None]), -1)).squeeze(-1)
        # Explicit symmetrization also covers independently dropped node states.
        epsilon = (raw + raw.transpose(1, 2)) * .5 * condition.pairs
        return (epsilon, states) if return_states else epsilon

    def reconstruction_loss(self, batch, generator=None, curves=False):
        from .metrics import recovery_statistics
        condition = condition_from_batch(batch, self.config.motif_prior)
        target = adjacency_target(batch, condition)
        if self.config.time_mode == "terminal":
            t = torch.full((len(target),), self.config.noise_steps, device=target.device, dtype=torch.long)
        else:
            t = torch.randint(1, self.config.noise_steps+1, (len(target),), device=target.device, generator=generator)
        noisy, noise = self.schedule.sample(target, t, condition.pairs, generator)
        epsilon = self.predict_noise(condition, noisy, t)
        restored = self.schedule.reconstruct(noisy, epsilon, t, condition.pairs)
        # Paper Eq.5: full off-diagonal Frobenius sum per graph, mean over graphs.
        per_graph = ((restored-target).square() * condition.pairs).sum((1, 2))
        loss = per_graph.mean()
        stats = recovery_statistics(target, restored, noise, epsilon, condition, t, self.config.noise_steps)
        if curves:
            upper = torch.triu(condition.pairs, diagonal=1)
            return loss, stats, (target[upper].detach().cpu(), restored[upper].detach().cpu())
        return loss, stats

    def forward(self, batch, return_states=False):
        condition = condition_from_batch(batch, self.config.motif_prior)
        clean = adjacency_target(batch, condition)
        t = torch.zeros(len(clean), dtype=torch.long, device=clean.device)
        states = self.encoder(condition, clean, t)
        graph = states["atoms"].sum(1) / condition.nodes.sum(1).clamp_min(1)[:, None]
        output = self.property_head(graph)
        return (output, states) if return_states else output
