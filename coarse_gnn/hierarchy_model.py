"""Fully packed near-fine/far-coarse neural predictor."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .hierarchy_packed import PackedHierarchyLevelPlan, PackedHierarchyPlan
from .layers import EdgeGIN


@dataclass(frozen=True)
class HierarchyNetworkConfig:
    input_dim: int
    base_dim: int
    hidden_dim: int = 128
    output_dim: int = 1
    dropout: float = 0.1
    include_base_prediction: bool = False

    def __post_init__(self):
        for name in ("input_dim", "base_dim", "hidden_dim", "output_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")


@dataclass
class HierarchyOutput:
    prediction: Tensor
    correction: Tensor
    graph_embedding: Tensor
    level_embeddings: tuple[Tensor, Tensor, Tensor]


def _segment(values: Tensor, lengths: Tensor, reduction: str) -> Tensor:
    return torch.segment_reduce(values.float(), reduction, lengths=lengths).to(values.dtype)


class HierarchicalPredictor(nn.Module):
    """One vectorized forward pass over every graph, token and context pair."""

    def __init__(self, config: HierarchyNetworkConfig):
        super().__init__()
        self.config = config
        hidden = config.hidden_dim
        self.atom_adapter = nn.Sequential(nn.Linear(config.input_dim, hidden), nn.LayerNorm(hidden))
        self.l1_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, 2 * hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
        )
        self.parent_mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(2 * hidden + 6, 2 * hidden), nn.SiLU(), nn.Dropout(config.dropout),
                nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
            ) for _ in range(2)
        ])
        self.gines = nn.ModuleList([
            EdgeGIN(hidden, 6, 1, config.dropout), EdgeGIN(hidden, 6, 1, config.dropout),
        ])
        self.level_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden, hidden), nn.LayerNorm(hidden)) for _ in range(3)
        ])
        self.query_projection = nn.Linear(hidden, hidden)
        self.key_projection = nn.Linear(hidden, hidden)
        self.value_projection = nn.Linear(hidden, hidden)
        self.distance_mlp = nn.Sequential(nn.Linear(6, hidden), nn.SiLU(), nn.Linear(hidden, 1))
        self.level_bias = nn.Embedding(3, 1)
        self.beta_leaf = nn.Parameter(torch.tensor(1.0))
        self.beta_atom = nn.Parameter(torch.tensor(0.0))
        # No bias: a query with an empty context (single-L1 component) receives
        # an exact zero update even when batched with larger molecules.
        self.context_projection = nn.Linear(hidden, hidden, bias=False)
        self.gate = nn.Linear(2 * hidden, hidden)
        nn.init.constant_(self.gate.bias, -2.0)
        self.graph_mlp = nn.Sequential(
            nn.Linear(2 * hidden + 1, 2 * hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(2 * hidden, hidden), nn.LayerNorm(hidden),
        )
        self.base_projection = nn.Sequential(nn.Linear(config.base_dim, hidden), nn.LayerNorm(hidden))
        correction_input = 2 * hidden + (config.output_dim if config.include_base_prediction else 0)
        self.correction_head = nn.Sequential(
            nn.Linear(correction_input, hidden), nn.SiLU(), nn.Dropout(config.dropout),
            nn.Linear(hidden, config.output_dim),
        )
        nn.init.normal_(self.correction_head[-1].weight, std=1e-3)
        nn.init.zeros_(self.correction_head[-1].bias)

    def _make_parent_level(
        self, children: Tensor, plan: PackedHierarchyLevelPlan, level_index: int,
    ) -> Tensor:
        if plan.parent_lengths.numel() == 0:
            return children.new_empty((0, children.size(1)))
        local_children = self.gines[level_index](
            children, plan.edges, plan.edge_attributes.to(children.dtype),
        )
        gathered = local_children[plan.child_gather]
        weights = plan.child_atom_counts.to(gathered.dtype).unsqueeze(1)
        total = _segment(gathered * weights, plan.parent_lengths, "sum")
        denominator = _segment(weights, plan.parent_lengths, "sum").clamp_min(1)
        mean = total / denominator
        maximum = _segment(gathered, plan.parent_lengths, "max")
        structure = plan.structure_features.to(gathered.dtype)
        return self.parent_mlps[level_index](torch.cat((mean, maximum, structure), dim=1))

    def _adaptive_attention(self, levels: tuple[Tensor, Tensor, Tensor], plan: PackedHierarchyPlan) -> Tensor:
        context = plan.contexts
        queries = levels[0][context.query_l1_indices]
        if context.context_indices.numel() == 0:
            return levels[0]
        level_offsets = torch.stack((
            context.level_token_counts.new_zeros(()),
            context.level_token_counts[0],
            context.level_token_counts[:2].sum(),
        ))
        all_tokens = torch.cat(levels, dim=0)
        unified_indices = context.context_indices + level_offsets[context.context_levels - 1]
        selected = all_tokens[unified_indices]
        pair_queries = context.context_query_indices

        score = (self.query_projection(queries)[pair_queries] * self.key_projection(selected)).sum(1)
        score = score / math.sqrt(self.config.hidden_dim)
        diameter = context.query_diameters[pair_queries].to(score.dtype).clamp_min(1)
        distance_features = torch.stack((
            context.d_min.to(score.dtype), context.d_max.to(score.dtype), context.d_mean.to(score.dtype),
            context.d_min.to(score.dtype) / diameter,
            context.d_max.to(score.dtype) / diameter,
            context.d_mean.to(score.dtype) / diameter,
        ), dim=1)
        score = score + self.distance_mlp(distance_features).squeeze(1)
        score = score + self.level_bias(context.context_levels - 1).squeeze(1)
        score = score + self.beta_leaf * context.leaf_counts.to(score.dtype).log()
        score = score + self.beta_atom * context.atom_counts.to(score.dtype).log1p()

        query_count = queries.size(0)
        maxima = score.new_full((query_count,), -torch.inf)
        maxima.scatter_reduce_(0, pair_queries, score, reduce="amax", include_self=True)
        weights = torch.exp(score - maxima[pair_queries])
        sums = score.new_zeros(query_count).index_add(0, pair_queries, weights)
        weights = weights / sums[pair_queries].clamp_min(torch.finfo(weights.dtype).tiny)
        values = self.value_projection(selected)
        summaries = values.new_zeros((query_count, values.size(1)))
        summaries.index_add_(
            0, pair_queries, weights.to(values.dtype).unsqueeze(1) * values,
        )
        gated = torch.sigmoid(self.gate(torch.cat((queries, summaries), dim=1)))
        updated_queries = queries + gated * self.context_projection(summaries)
        return levels[0].index_copy(0, context.query_l1_indices, updated_queries)

    def forward(
        self,
        node_states_h2: Tensor,
        base_embedding: Tensor,
        base_prediction: Tensor,
        plan: PackedHierarchyPlan,
    ) -> HierarchyOutput:
        if node_states_h2.ndim != 3:
            raise ValueError("node_states_h2 must have shape [B, padded_nodes, input_dim]")
        batch = node_states_h2.size(0)
        if node_states_h2.size(1) != plan.padded_nodes or node_states_h2.size(2) != self.config.input_dim:
            raise ValueError("node state shape does not match the packed plan/config")
        if base_embedding.shape != (batch, self.config.base_dim):
            raise ValueError("base_embedding has the wrong shape")
        if base_prediction.shape != (batch, self.config.output_dim):
            raise ValueError("base_prediction has the wrong shape")

        atoms = self.atom_adapter(node_states_h2.flatten(0, 1)[plan.atom_gather])
        l1_mean = _segment(atoms, plan.l1_atom_lengths, "mean")
        l1_max = _segment(atoms, plan.l1_atom_lengths, "max")
        size = plan.l1_atom_counts.to(atoms.dtype).log1p().unsqueeze(1)
        l1 = self.l1_mlp(torch.cat((l1_mean, l1_max, size), dim=1))
        l2 = self._make_parent_level(l1, plan.l2, 0)
        l3 = self._make_parent_level(l2, plan.l3, 1)
        projected = tuple(module(value) for module, value in zip(self.level_projections, (l1, l2, l3)))
        updated_l1 = self._adaptive_attention(projected, plan)

        l1_weights = plan.l1_atom_counts.to(updated_l1.dtype).unsqueeze(1)
        graph_sum = _segment(updated_l1 * l1_weights, plan.graph_l1_lengths, "sum")
        graph_weight = _segment(l1_weights, plan.graph_l1_lengths, "sum").clamp_min(1)
        graph_mean = graph_sum / graph_weight
        graph_max = _segment(updated_l1, plan.graph_l1_lengths, "max")
        graph_size = plan.graph_atom_counts.to(updated_l1.dtype).log1p().unsqueeze(1)
        graph = self.graph_mlp(torch.cat((graph_mean, graph_max, graph_size), dim=1))
        correction_inputs = [self.base_projection(base_embedding), graph]
        if self.config.include_base_prediction:
            correction_inputs.append(base_prediction)
        correction = self.correction_head(torch.cat(correction_inputs, dim=1))
        return HierarchyOutput(
            prediction=base_prediction + correction,
            correction=correction,
            graph_embedding=graph,
            level_embeddings=(updated_l1, projected[1], projected[2]),
        )
