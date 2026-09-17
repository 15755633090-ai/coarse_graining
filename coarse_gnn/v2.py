"""V2 exclusive-region coarsening and relation-only coarse prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .cache import TopologyCache
from .layers import EdgeGIN
from .model import GraphOutput
from .topology import CoarseTopology, build_v2_topology, topology_fingerprint


@dataclass(frozen=True)
class V2CoarseningConfig:
    """Frozen V2 topology definition.

    ``target_region_size`` is the scale hyperparameter ``s`` in
    K0=max(number_of_components, ceil(N/s)).  It is deliberately separate
    from ``max_region_radius``, which is a hard safety constraint.
    """

    target_region_size: int = 8
    max_region_radius: int = 4
    recenter_rounds: int = 2
    canonicalize: bool = True
    algorithm: str = "v2_exclusive_regions"

    def __post_init__(self):
        if not isinstance(self.target_region_size, int) or self.target_region_size < 1:
            raise ValueError("target_region_size must be a positive integer")
        if not isinstance(self.max_region_radius, int) or self.max_region_radius < 0:
            raise ValueError("max_region_radius must be a nonnegative integer")
        if not isinstance(self.recenter_rounds, int) or self.recenter_rounds < 0:
            raise ValueError("recenter_rounds must be a nonnegative integer")
        if not isinstance(self.canonicalize, bool):
            raise ValueError("canonicalize must be boolean")
        if self.algorithm != "v2_exclusive_regions":
            raise ValueError("algorithm is fixed by the V2 method definition")


@dataclass(frozen=True)
class V2NetworkConfig:
    input_dim: int = 128
    hidden_dim: int = 128
    edge_dim: int = 4
    output_dim: int = 1
    region_layers: int = 2
    coarse_layers: int = 1
    dropout: float = 0.1
    alpha_init: float = 0.1

    def __post_init__(self):
        for name in ("input_dim", "hidden_dim", "output_dim"):
            if not isinstance(getattr(self, name), int) or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.edge_dim != 4:
            raise ValueError("V2 molecular GINE requires four one-hot bond types")
        if self.region_layers != 2:
            raise ValueError("V2 fixes the Region GINE depth at two layers")
        if self.coarse_layers not in (0, 1):
            raise ValueError("V2 supports zero-layer Region-only or one relation layer")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if not isinstance(self.alpha_init, (int, float)) or not torch.isfinite(torch.tensor(float(self.alpha_init))):
            raise ValueError("alpha_init must be finite")


class _ZeroAnchoredHead(nn.Module):
    """Bias-free head, additionally evaluated relative to its exact zero input."""

    def __init__(self, hidden_dim: int, output_dim: int):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim, bias=False),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim, bias=False),
        )

    def forward(self, value: Tensor) -> Tensor:
        zero = torch.zeros_like(value)
        return self.layers(value) - self.layers(zero)


class V2CoarseGraphPredictor(nn.Module):
    """V2 predictor with a Region main path and a relation-only coarse correction."""

    def __init__(
        self,
        config: V2NetworkConfig | None = None,
        coarsening: V2CoarseningConfig | None = None,
        *,
        topology_cache: TopologyCache | None = None,
    ):
        super().__init__()
        self.config = config or V2NetworkConfig()
        self.coarsening = coarsening or V2CoarseningConfig()
        self.topology_cache = topology_cache
        cfg = self.config
        self.input_projection = nn.Sequential(nn.Linear(cfg.input_dim, cfg.hidden_dim), nn.SiLU())
        self.region_gnn = EdgeGIN(cfg.hidden_dim, cfg.edge_dim, cfg.region_layers, cfg.dropout)
        # [receiver, sender, log boundary count, four bond proportions]
        self.coarse_message = nn.Sequential(
            nn.Linear(2 * cfg.hidden_dim + 5, 2 * cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
        )
        self.delta_projection = nn.Sequential(
            nn.Linear(cfg.hidden_dim, 2 * cfg.hidden_dim),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(2 * cfg.hidden_dim, cfg.hidden_dim),
        )
        self.region_head = nn.Sequential(
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim), nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.output_dim),
        )
        self.delta_head = _ZeroAnchoredHead(cfg.hidden_dim, cfg.output_dim)
        self.alpha = nn.Parameter(torch.tensor(float(cfg.alpha_init)))

    def prepare_topology(
        self, num_nodes, edge_index, *, node_ids=None, node_labels=None, edge_labels=None,
    ) -> CoarseTopology:
        if node_ids is not None:
            raise ValueError("V2 canonical coarsening does not accept persistent node IDs")
        if self.topology_cache is not None:
            return self.topology_cache.get_or_build(
                num_nodes, edge_index, self.coarsening,
                node_labels=node_labels, edge_labels=edge_labels,
            )
        return build_v2_topology(
            num_nodes, edge_index, self.coarsening,
            node_labels=node_labels, edge_labels=edge_labels,
        )

    def _coarse_delta(self, regions: Tensor, edges: Tensor, attributes: Tensor) -> Tensor:
        if self.config.coarse_layers == 0 or edges.size(1) == 0:
            return torch.zeros_like(regions)
        source = torch.cat((edges[0], edges[1]))
        target = torch.cat((edges[1], edges[0]))
        directed_attributes = torch.cat((attributes, attributes))
        messages = self.coarse_message(torch.cat(
            (regions[target], regions[source], directed_attributes), dim=1,
        ))
        totals = torch.zeros_like(regions).index_add(0, target, messages)
        degree = torch.zeros(regions.size(0), device=regions.device, dtype=regions.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=regions.dtype))
        mean = totals / degree.clamp_min(1).unsqueeze(1)
        delta = self.delta_projection(mean)
        return delta * degree.gt(0).to(regions.dtype).unsqueeze(1)

    def forward(
        self,
        node_embeddings: Tensor,
        edge_index: Tensor,
        edge_attr: Tensor | None = None,
        *,
        node_ids: Tensor | None = None,
        node_labels: Tensor | None = None,
        edge_labels: Tensor | None = None,
        topology: CoarseTopology | None = None,
    ) -> GraphOutput:
        cfg = self.config
        if node_embeddings.ndim != 2 or node_embeddings.size(1) != cfg.input_dim:
            raise ValueError(f"node_embeddings must have shape [N, {cfg.input_dim}]")
        if edge_attr is None or edge_attr.shape != (edge_index.size(1), cfg.edge_dim):
            raise ValueError(f"V2 edge_attr must have shape [E, {cfg.edge_dim}]")
        if node_labels is None or edge_labels is None:
            raise ValueError("V2 requires raw discrete node and edge labels")
        if topology is None:
            topology = self.prepare_topology(
                node_embeddings.size(0), edge_index, node_ids=node_ids,
                node_labels=node_labels, edge_labels=edge_labels,
            )
        elif topology.input_fingerprint != topology_fingerprint(
            node_embeddings.size(0), edge_index, self.coarsening,
            node_ids=node_ids, node_labels=node_labels, edge_labels=edge_labels,
        ):
            raise ValueError("Precomputed V2 topology does not match the input")

        device, dtype = node_embeddings.device, node_embeddings.dtype
        edge_attr = edge_attr.to(device=device, dtype=dtype)
        physical_attributes = edge_attr[topology.edge_positions.to(device)]
        if not torch.allclose(edge_attr, physical_attributes[topology.input_edge_ids.to(device)]):
            raise ValueError("Opposite directions of an undirected edge must have identical attributes")

        projected = self.input_projection(node_embeddings[topology.atom_order.to(device)])
        regions = []
        for context, local_edges, edge_ids in zip(
            topology.contexts, topology.context_edges, topology.context_edge_ids,
        ):
            local = self.region_gnn(
                projected[context.to(device)], local_edges.to(device),
                physical_attributes[edge_ids.to(device)],
            )
            regions.append(local.mean(0))
        region_embeddings = torch.stack(regions)
        sizes = node_embeddings.new_tensor([core.numel() for core in topology.cores])

        counts = topology.edge_counts.to(device=device, dtype=dtype).unsqueeze(1)
        bond_counts = physical_attributes.new_zeros((counts.size(0), cfg.edge_dim)).index_add(
            0, topology.boundary_groups.to(device),
            physical_attributes[topology.boundary_edge_ids.to(device)],
        )
        proportions = bond_counts / counts.clamp_min(1)
        coarse_edge_attr = torch.cat((counts.log1p(), proportions), dim=1)
        delta = self._coarse_delta(
            region_embeddings, topology.coarse_edges.to(device), coarse_edge_attr,
        )
        coarse_embeddings = region_embeddings + delta
        denominator = sizes.sum().clamp_min(1)
        region_graph = (region_embeddings * sizes.unsqueeze(1)).sum(0) / denominator
        delta_graph = (delta * sizes.unsqueeze(1)).sum(0) / denominator
        region_prediction = self.region_head(region_graph)
        correction = self.delta_head(delta_graph)
        prediction = region_prediction + self.alpha * correction
        return GraphOutput(
            prediction=prediction,
            graph_embedding=torch.cat((region_graph, delta_graph)),
            region_embeddings=region_embeddings,
            coarse_embeddings=coarse_embeddings,
            coarse_edge_attr=coarse_edge_attr,
            topology=topology,
        )
