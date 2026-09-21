"""Two-layer packed LocalBoundary-GINE residual predictor."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .layers import EdgeGIN
from .local_boundary_packed import PackedLocalBoundaryPlan


LOCAL_BOUNDARY_VARIANTS = ("self_only", "local_only", "local_boundary", "full")


def _segment(values: Tensor, lengths: Tensor, reduction: str) -> Tensor:
    """Packed reduction kept local so this predictor has no Adaptive-model dependency."""
    return torch.segment_reduce(values.float(), reduction, lengths=lengths).to(values.dtype)


@dataclass(frozen=True)
class LocalBoundaryNetworkConfig:
    input_dim: int
    base_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    dropout: float = 0.1

    def __post_init__(self):
        for name in ("input_dim", "base_dim", "hidden_dim", "output_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class LocalBoundaryOutput:
    prediction: Tensor
    correction: Tensor
    graph_embedding: Tensor
    l1_embedding: Tensor


class LocalBoundaryPredictor(nn.Module):
    """Same two GINE layers for every variant; only their edge tables differ."""

    def __init__(self, config: LocalBoundaryNetworkConfig, variant: str):
        super().__init__()
        if variant not in LOCAL_BOUNDARY_VARIANTS:
            raise ValueError(f"Unknown LocalBoundary variant: {variant}")
        self.config, self.variant = config, variant
        hidden = config.hidden_dim
        self.atom_adapter = nn.Sequential(nn.Linear(config.input_dim, hidden), nn.LayerNorm(hidden))
        self.l1_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, 2 * hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
        )
        self.gines = nn.ModuleList([
            EdgeGIN(hidden, 6, 1, config.dropout),
            EdgeGIN(hidden, 6, 1, config.dropout),
        ])
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, 2 * hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
        )
        self.base_projection = nn.Sequential(nn.Linear(config.base_dim, hidden), nn.LayerNorm(hidden))
        self.correction_head = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(hidden, config.output_dim),
        )
        nn.init.normal_(self.correction_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.correction_head[-1].bias)

    def _schedule(self, plan):
        empty_edges = plan.full_edges[:, :0]
        empty_attributes = plan.full_edge_attributes[:0]
        empty = (empty_edges, empty_attributes)
        if self.variant == "self_only":
            return empty, empty
        if self.variant == "local_only":
            return (plan.local_edges, plan.local_edge_attributes), empty
        if self.variant == "local_boundary":
            return (
                (plan.local_edges, plan.local_edge_attributes),
                (plan.boundary_edges, plan.boundary_edge_attributes),
            )
        full = (plan.full_edges, plan.full_edge_attributes)
        return full, full

    def forward(self, node_states_h2, base_embedding, base_prediction, plan):
        if node_states_h2.ndim != 3 or node_states_h2.size(1) != plan.padded_nodes:
            raise ValueError("node_states_h2 shape does not match the packed plan")
        batch = node_states_h2.size(0)
        if node_states_h2.size(2) != self.config.input_dim:
            raise ValueError("node state feature dimension does not match config")
        if base_embedding.shape != (batch, self.config.base_dim):
            raise ValueError("base_embedding has the wrong shape")
        if base_prediction.shape != (batch, self.config.output_dim):
            raise ValueError("base_prediction has the wrong shape")
        atoms = self.atom_adapter(node_states_h2.flatten(0, 1)[plan.atom_gather])
        mean = _segment(atoms, plan.l1_atom_lengths, "mean")
        maximum = _segment(atoms, plan.l1_atom_lengths, "max")
        size = plan.l1_atom_counts.to(atoms.dtype).log1p().unsqueeze(1)
        l1 = self.l1_mlp(torch.cat((mean, maximum, size), dim=1))
        for gine, (edges, attributes) in zip(self.gines, self._schedule(plan)):
            l1 = gine(l1, edges, attributes.to(l1.dtype))
        weights = plan.l1_atom_counts.to(l1.dtype).unsqueeze(1)
        graph_sum = _segment(l1 * weights, plan.graph_l1_lengths, "sum")
        graph_weight = _segment(weights, plan.graph_l1_lengths, "sum").clamp_min(1)
        graph_mean = graph_sum / graph_weight
        graph_max = _segment(l1, plan.graph_l1_lengths, "max")
        graph_size = plan.graph_atom_counts.to(l1.dtype).log1p().unsqueeze(1)
        graph = self.graph_mlp(torch.cat((graph_mean, graph_max, graph_size), dim=1))
        correction = self.correction_head(torch.cat((self.base_projection(base_embedding), graph), dim=1))
        return LocalBoundaryOutput(
            prediction=base_prediction + correction,
            correction=correction,
            graph_embedding=graph,
            l1_embedding=l1,
        )
