"""Atom states never pool to fragment nodes; only the final molecular readout pools."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch_geometric.nn import global_mean_pool

from .data import ATOM_DIM, BOND_DIM
from .diffusion import GaussianSchedule


@dataclass
class Config:
    hidden_dim: int = 128
    layers: int = 3
    dropout: float = 0.1
    noise_steps: int = 1000
    dual_state: bool = True
    motif_prior: bool = True
    conditional_gate: bool = True

    def __post_init__(self):
        if self.hidden_dim < 4 or self.layers < 1 or self.noise_steps < 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid model dimensions, schedule or dropout")


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


def aggregate(messages, destination, nodes):
    out = messages.new_zeros(nodes, messages.shape[-1])
    out.index_add_(0, destination, messages)
    degree = torch.bincount(destination, minlength=nodes).clamp_min(1).to(messages.dtype)
    return out / degree[:, None]


class TimeEmbedding(nn.Module):
    def __init__(self, dim, steps):
        super().__init__()
        self.steps = steps
        self.register_buffer("frequencies", torch.exp(torch.linspace(0, math.log(1000), dim // 2)))
        self.project = mlp(2 * (dim // 2) + 1, dim, dim)

    def forward(self, steps):
        t = steps.float() / self.steps
        angle = t[:, None] * self.frequencies[None]
        return self.project(torch.cat((t[:, None], angle.sin(), angle.cos()), -1))


class AtomLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        d = config.hidden_dim
        self.dual = config.dual_state
        self.conditional_gate = config.conditional_gate
        if self.dual:
            self.internal_message = mlp(3 * d, 2 * d, d)
            self.internal_update = mlp(2 * d, 2 * d, d)
            self.internal_norm = nn.LayerNorm(d)
        self.local_message = mlp(3 * d, 2 * d, d)
        self.cross_message = mlp(3 * d, 2 * d, d)
        if config.conditional_gate:
            self.gate = mlp(5 * d, d, 1)
        self.context_update = mlp(4 * d, 2 * d, d)
        self.context_norm = nn.LayerNorm(d)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, s, h, edge_index, bond, cross):
        src, dst = edge_index
        inside = ~cross
        old_s = s if self.dual else h
        if self.dual:
            messages = self.internal_message(torch.cat((s[src[inside]], s[dst[inside]], bond[inside]), -1))
            received = aggregate(messages, dst[inside], len(s))
            next_s = self.internal_norm(s + self.dropout(self.internal_update(torch.cat((s, received), -1))))
        else:
            next_s = h
        local = self.local_message(torch.cat((h[src[inside]], h[dst[inside]], bond[inside]), -1))
        cross_messages = self.cross_message(torch.cat((h[src[cross]], h[dst[cross]], bond[cross]), -1))
        if self.conditional_gate:
            gate = torch.sigmoid(self.gate(torch.cat((old_s[src[cross]], old_s[dst[cross]],
                                                       h[src[cross]], h[dst[cross]], bond[cross]), -1)))
        else:
            gate = h.new_ones(int(cross.sum()), 1)
        # Normalize by edge count, not by sum(gate), so attenuation is preserved.
        local_sum = aggregate(local, dst[inside], len(h))
        cross_sum = aggregate(gate * cross_messages, dst[cross], len(h))
        update = self.context_update(torch.cat((h, next_s, local_sum, cross_sum), -1))
        next_h = self.context_norm(h + self.dropout(update))
        return next_s, next_h, gate


class AtomEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        d = config.hidden_dim
        self.context_input = nn.Linear(ATOM_DIM, d)
        if config.dual_state:
            self.internal_input = nn.Linear(ATOM_DIM, d)
        self.boundary_input = nn.Linear(1, d, bias=False)
        self.time = TimeEmbedding(d, config.noise_steps)
        self.bond = nn.Linear(BOND_DIM, d)
        self.layers = nn.ModuleList([AtomLayer(config) for _ in range(config.layers)])
        self.fuse = mlp((2 if config.dual_state else 1) * d, 2 * d, d)

    def forward(self, data, x_input, node_steps):
        if x_input.shape != (data.num_nodes, ATOM_DIM) or node_steps.shape != (data.num_nodes,):
            raise ValueError("Atom input and time conditions must align with node rows")
        # No data.x values are read here: there is no clean attribute bypass.
        cross = data.is_cross if self.config.motif_prior else torch.zeros_like(data.is_cross)
        boundary = data.boundary if self.config.motif_prior else torch.zeros_like(data.boundary)
        condition = self.time(node_steps) + self.boundary_input(boundary.to(x_input.dtype)[:, None])
        h = self.context_input(x_input) + condition
        s = self.internal_input(x_input) + condition if self.config.dual_state else h
        bond = self.bond(data.edge_attr)
        gates = []
        for layer in self.layers:
            s, h, gate = layer(s, h, data.edge_index, bond, cross)
            gates.append(gate)
        z = self.fuse(torch.cat((s, h), -1) if self.config.dual_state else h)
        return {"internal": s, "context": h, "atoms": z,
                "cross_edge_index": data.edge_index[:, cross], "gates": gates}


class AtomDiffusionModel(nn.Module):
    def __init__(self, config=None, num_tasks=1):
        super().__init__()
        self.config = config or Config()
        self.num_tasks = num_tasks
        self.schedule = GaussianSchedule(self.config.noise_steps)
        self.encoder = AtomEncoder(self.config)
        self.noise_head = mlp(self.config.hidden_dim, self.config.hidden_dim, ATOM_DIM)
        self.property_head = mlp(self.config.hidden_dim, self.config.hidden_dim, num_tasks)

    def set_stage(self, stage):
        if stage not in ("pretrain", "finetune"):
            raise ValueError(stage)
        for p in self.encoder.parameters():
            p.requires_grad_(True)
        for p in self.noise_head.parameters():
            p.requires_grad_(stage == "pretrain")
        for p in self.property_head.parameters():
            p.requires_grad_(stage == "finetune")

    def predict_noise(self, data, xt, graph_steps, return_states=False):
        node_steps = self.schedule.node_steps(graph_steps, data)
        states = self.encoder(data, xt, node_steps)
        prediction = self.noise_head(states["atoms"])
        return (prediction, states) if return_states else prediction

    def denoise_loss(self, data, generator=None):
        t = torch.randint(1, self.config.noise_steps + 1, (data.num_graphs,), device=data.x.device, generator=generator)
        steps = self.schedule.node_steps(t, data)
        noise = torch.randn(data.x.shape, device=data.x.device, dtype=data.x.dtype, generator=generator)
        xt = self.schedule.corrupt(data.x, steps, noise)
        prediction = self.predict_noise(data, xt, t)
        errors = (prediction - noise).square().mean(-1)
        groups = {"all": torch.ones_like(data.boundary), "boundary": data.boundary, "internal": ~data.boundary}
        diagnostic = {}
        for name, mask in groups.items():
            diagnostic[f"{name}_error_sum"] = float(errors.detach()[mask].sum())
            diagnostic[f"{name}_atoms"] = int(mask.sum())
        return errors.mean(), diagnostic

    def forward(self, data, return_states=False):
        # A single clean forward at t=0; no diffusion sampling in prediction.
        steps = torch.zeros(data.num_nodes, dtype=torch.long, device=data.x.device)
        states = self.encoder(data, data.x, steps)
        prediction = self.property_head(global_mean_pool(states["atoms"], data.batch))
        return (prediction, states) if return_states else prediction
