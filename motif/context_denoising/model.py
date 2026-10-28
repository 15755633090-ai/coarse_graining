"""Only motif latent states are corrupted. Prediction always uses clean states."""
from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
from .diagnostics import denoise_statistics


VARIANTS = ("atom", "motif", "context", "fixed", "random", "no_communication", "mask", "shuffle")


@dataclass
class Config:
    hidden_dim: int = 128
    atom_layers: int = 3
    context_layers: int = 2
    dropout: float = 0.1
    noise_steps: int = 100
    fixed_step: int = 50
    corruption_rate: float = 0.5

    def __post_init__(self):
        if self.hidden_dim < 2 or self.atom_layers < 1 or self.context_layers < 1:
            raise ValueError("Invalid model dimensions/layer counts")
        if self.noise_steps < 1 or not 1 <= self.fixed_step <= self.noise_steps:
            raise ValueError("fixed_step must be in [1, noise_steps]")
        if not 0 < self.corruption_rate <= 1 or not 0 <= self.dropout < 1:
            raise ValueError("Invalid corruption rate or dropout")


def mlp(inputs, hidden, outputs):
    return nn.Sequential(nn.Linear(inputs, hidden), nn.SiLU(), nn.Linear(hidden, outputs))


class ContextLayer(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.message = mlp(dim * 3, dim * 2, dim)
        self.local = mlp(dim, dim, dim)
        self.gate = nn.Linear(dim * 2, dim)
        self.project = nn.Linear(dim, dim)

    def forward(self, x, edge_index, relation, communicate=True):
        aggregate = torch.zeros_like(x)
        if communicate and edge_index.shape[1]:
            src, dst = edge_index
            messages = self.message(torch.cat((x[src], x[dst], relation), -1))
            aggregate.index_add_(0, dst, messages)
            degree = torch.bincount(dst, minlength=len(x)).clamp_min(1).to(x.dtype)
            aggregate = aggregate / degree[:, None]
        context = self.local(x) + aggregate
        gate = torch.sigmoid(self.gate(torch.cat((x, context), -1)))
        return x + gate * self.project(context)


class MotifContextModel(nn.Module):
    def configure_training(self, variant, auxiliary_weight):
        """Freeze modules outside this variant's optimized paths; keep initialization identical."""
        if variant not in VARIANTS:
            raise ValueError(variant)
        enabled = {"atom_encoder", "bond_encoder", "atom_convs"}
        enabled.add("atom_head" if variant == "atom" else "property_head")
        if variant not in ("atom", "motif"):
            enabled.update(("static_relation", "clean_relation", "time_condition",
                            "damage_condition", "context_layers"))
        if variant in ("fixed", "random", "no_communication", "mask", "shuffle") and auxiliary_weight > 0:
            enabled.update(("noisy_relation", "denoise_head"))
        for name, parameter in self.named_parameters():
            parameter.requires_grad_(name.split(".")[0] in enabled)
        return {
            "instantiated": sum(p.numel() for p in self.parameters()),
            "optimized": sum(p.numel() for p in self.parameters() if p.requires_grad),
            "optimized_modules": sorted(enabled),
        }

    def __init__(self, num_tasks, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        d = c.hidden_dim
        self.atom_encoder = AtomEncoder(d)
        self.bond_encoder = BondEncoder(d)
        self.atom_convs = nn.ModuleList([
            GINEConv(mlp(d, 2 * d, d), train_eps=True) for _ in range(c.atom_layers)
        ])
        self.dropout = nn.Dropout(c.dropout)
        # Affine-free normalization prevents learning a trivial normalization scale.
        self.normalize = nn.LayerNorm(d, elementwise_affine=False)
        self.static_relation = BondEncoder(d)
        self.clean_relation = mlp(3 * d, 2 * d, d)
        self.noisy_relation = mlp(d, d, d)
        self.time_condition = mlp(3, d, d)
        self.damage_condition = nn.Embedding(2, d)
        self.context_layers = nn.ModuleList([ContextLayer(d) for _ in range(c.context_layers)])
        self.denoise_head = mlp(d, d, d)
        self.property_head = mlp(2 * d, d, num_tasks)
        self.atom_head = mlp(d, d, num_tasks)
        t = torch.arange(c.noise_steps + 1, dtype=torch.float32) / c.noise_steps
        signal = torch.cos((t + 0.008) / 1.008 * math.pi / 2).square()
        self.register_buffer("alpha_bar", (signal / signal[0]).clamp(1e-5, 1))

    def encode(self, data):
        h = self.atom_encoder(data.x)
        bond = self.bond_encoder(data.edge_attr)
        for conv in self.atom_convs:
            h = h + self.dropout(F.relu(conv(h, data.edge_index, bond)))
        q = self.normalize(global_mean_pool(h, data.atom_to_motif))
        graph_id = torch.repeat_interleave(
            torch.arange(data.num_graphs, device=h.device), data.motif_count
        )
        return h, q, graph_id

    def contextualize(self, q, edges, relation, steps, damaged, communicate=True):
        t = steps.to(q.dtype) / self.config.noise_steps
        condition = self.time_condition(torch.stack((t, torch.sin(math.pi * t), torch.cos(math.pi * t)), -1))
        x = q + condition + self.damage_condition(damaged.long())
        for layer in self.context_layers:
            x = layer(x, edges, relation, communicate)
        return x

    def clean_prediction(self, data, h, q, graph_id, variant):
        atom = global_add_pool(h, data.batch)
        if variant == "atom":
            return self.atom_head(atom)
        enhanced = q
        if variant != "motif":
            u, v = data.attachment_index
            structure = self.static_relation(data.motif_edge_attr)
            relation = self.clean_relation(torch.cat((h[u], h[v], structure), -1))
            enhanced = self.contextualize(
                q, data.motif_edge_index, relation,
                torch.zeros(len(q), device=q.device), torch.zeros(len(q), dtype=torch.bool, device=q.device),
            )
        return self.property_head(torch.cat((atom, global_mean_pool(enhanced, graph_id)), -1))

    def corrupt(self, q, graph_id, mode, generator=None):
        c = self.config
        damaged = torch.zeros(len(q), dtype=torch.bool, device=q.device)
        # Exact per-molecule sampling; at least one motif is corrupted, including K=1.
        for gid in graph_id.unique():
            indices = torch.where(graph_id == gid)[0]
            n = max(1, math.ceil(len(indices) * c.corruption_rate))
            chosen = torch.randperm(len(indices), device=q.device, generator=generator)[:n]
            damaged[indices[chosen]] = True
        if mode == "fixed":
            graph_steps = torch.full((int(graph_id.max()) + 1,), c.fixed_step, device=q.device, dtype=torch.long)
        else:
            graph_steps = torch.randint(1, c.noise_steps + 1, (int(graph_id.max()) + 1,), device=q.device, generator=generator)
        steps = graph_steps[graph_id]
        if mode == "mask":
            steps = torch.full_like(steps, c.noise_steps)
            noisy = torch.zeros_like(q)
        else:
            a = self.alpha_bar[steps, None]
            noise = torch.randn(q.shape, device=q.device, dtype=q.dtype, generator=generator)
            noisy = a.sqrt() * q + (1 - a).sqrt() * noise
        z = torch.where(damaged[:, None], noisy, q)
        return z, torch.where(damaged, steps, 0), damaged

    def forward(self, data, variant="random", auxiliary=False, generator=None):
        if variant not in VARIANTS:
            raise ValueError(variant)
        h, q, graph_id = self.encode(data)
        prediction = self.clean_prediction(data, h, q, graph_id, variant)
        if not auxiliary:
            return prediction
        loss = q.sum() * 0
        selected = 0
        group_statistics = {}
        if variant in ("fixed", "random", "no_communication", "mask", "shuffle"):
            z, steps, damaged = self.corrupt(q, graph_id, variant, generator)
            # Only static bond categories enter this branch: no clean attachment states.
            relation = self.noisy_relation(self.static_relation(data.motif_edge_attr))
            edges = data.motif_edge_index
            if variant == "shuffle":
                # Rewire only inside each molecule, retaining edge count and features.
                edges = edges.clone()
                for gid in graph_id.unique():
                    members = torch.where(graph_id == gid)[0]
                    permutation = members[torch.randperm(len(members), device=q.device, generator=generator)]
                    mapping = torch.arange(len(q), device=q.device)
                    mapping[members] = permutation
                    inside = graph_id[edges[0]] == gid
                    edges[:, inside] = mapping[edges[:, inside]]
            restored = self.denoise_head(self.contextualize(
                z, edges, relation, steps, damaged, communicate=variant != "no_communication"
            ))
            # Mean of squared L2 norms over selected motifs, as in the method definition.
            target_errors = (restored[damaged] - q.detach()[damaged]).square().sum(-1)
            loss = target_errors.mean()
            group_statistics = denoise_statistics(target_errors, graph_id[damaged], data.motif_count)
            selected = int(damaged.sum())
        diagnostics = {
            "q_norm": q.detach().norm(dim=-1).mean(),
            "q_dimension_variance": q.detach().var(dim=0, unbiased=False).mean(),
            "selected_motifs": selected,
            **group_statistics,
        }
        return prediction, loss, diagnostics
