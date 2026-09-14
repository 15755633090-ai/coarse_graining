"""Generic coarse graph predictor accepting embeddings from any encoder."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .config import CoarseningConfig, NetworkConfig
from .layers import EdgeGIN
from .topology import CoarseTopology, build_topology


@dataclass
class GraphOutput:
    prediction: Tensor  # [output_dim], regression values or classification logits.
    graph_embedding: Tensor
    region_embeddings: Tensor  # Before coarse propagation.
    coarse_embeddings: Tensor  # After coarse propagation.
    coarse_edge_attr: Tensor  # count followed by sum/mean of original attributes.
    topology: CoarseTopology


class CoarseGraphPredictor(nn.Module):
    def __init__(self, config: NetworkConfig | None = None, coarsening: CoarseningConfig | None = None):
        super().__init__()
        self.config = config or NetworkConfig()
        self.coarsening = coarsening or CoarseningConfig()
        cfg = self.config
        self.input_projection = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim), nn.SiLU())
        self.region_gnn = EdgeGIN(cfg.hidden_dim, cfg.edge_dim, cfg.region_layers, cfg.dropout)
        self.size_projection = nn.Linear(1, cfg.hidden_dim, bias=False) if cfg.use_size_feature else None
        self.coarse_gnn = EdgeGIN(cfg.hidden_dim, cfg.edge_dim + 1, cfg.coarse_layers, cfg.dropout)
        self.head = nn.Sequential(nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(), nn.Linear(cfg.hidden_dim, cfg.output_dim))

    def forward(
        self, node_embeddings: Tensor, edge_index: Tensor, edge_attr: Tensor | None = None,
        *, node_ids: Tensor | None = None,
    ) -> GraphOutput:
        cfg = self.config
        if node_embeddings.ndim != 2 or node_embeddings.size(1) != cfg.input_dim:
            raise ValueError(f"node_embeddings must have shape [N, {cfg.input_dim}]")
        if not node_embeddings.is_floating_point() or not torch.isfinite(node_embeddings).all():
            raise ValueError("node_embeddings must be finite floating point values")
        topology = build_topology(node_embeddings.size(0), edge_index, self.coarsening, node_ids=node_ids)
        device = node_embeddings.device
        if edge_attr is None:
            if cfg.edge_dim:
                raise ValueError("edge_attr is required when edge_dim > 0")
            edge_attr = node_embeddings.new_empty((edge_index.size(1), 0))
        if edge_attr.shape != (edge_index.size(1), cfg.edge_dim):
            raise ValueError(f"edge_attr must have shape [E, {cfg.edge_dim}]")
        edge_attr = edge_attr.to(device=device, dtype=node_embeddings.dtype)
        if not torch.isfinite(edge_attr).all():
            raise ValueError("edge_attr must be finite")
        attributes = edge_attr[topology.edge_positions.to(device)]
        # Opposite directions are storage duplicates, not additional physical edges.
        canonical_ids = {tuple(pair): i for i, pair in enumerate(topology.edges.T.tolist())}
        input_pairs = edge_index.detach().cpu().T.tolist()
        inverse = torch.tensor([canonical_ids[(min(u, v), max(u, v))] for u, v in input_pairs], device=device, dtype=torch.long)
        if not torch.allclose(edge_attr, attributes[inverse]):
            raise ValueError("Opposite directions of an undirected edge must have identical attributes")

        projected = self.input_projection(node_embeddings)
        regions = []
        for context, local_edges, edge_ids, core_positions in zip(
            topology.contexts, topology.context_edges, topology.context_edge_ids, topology.core_positions,
        ):
            # A shared encoder also handles singleton and residual graphs.
            local = self.region_gnn(
                projected[context.to(device)], local_edges.to(device), attributes[edge_ids.to(device)],
            )
            core = local[core_positions.to(device)]
            regions.append(core.mean(0) if cfg.region_pool == "mean" else core.sum(0))
        region_embeddings = torch.stack(regions)
        sizes = node_embeddings.new_tensor([core.numel() for core in topology.cores])
        coarse_input = region_embeddings
        if self.size_projection is not None:
            coarse_input = coarse_input + self.size_projection(sizes.log1p().unsqueeze(1))

        counts = topology.edge_counts.to(device=device, dtype=node_embeddings.dtype).unsqueeze(1)
        reduced = attributes.new_zeros((counts.size(0), cfg.edge_dim)).index_add(
            0, topology.boundary_groups.to(device), attributes[topology.boundary_edge_ids.to(device)],
        )
        if cfg.edge_reduce == "mean":
            reduced = reduced / counts.clamp_min(1)
        coarse_edge_attr = torch.cat((counts, reduced), dim=1)
        coarse_embeddings = self.coarse_gnn(coarse_input, topology.coarse_edges.to(device), coarse_edge_attr)
        if cfg.graph_pool == "sum":
            graph_embedding = coarse_embeddings.sum(0)
        elif cfg.graph_pool == "size_weighted_mean":
            graph_embedding = (coarse_embeddings * sizes.unsqueeze(1)).sum(0) / sizes.sum()
        else:
            graph_embedding = coarse_embeddings.mean(0)
        return GraphOutput(
            self.head(graph_embedding), graph_embedding, region_embeddings,
            coarse_embeddings, coarse_edge_attr, topology,
        )
