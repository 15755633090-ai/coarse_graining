"""V2 exclusive-region coarsening and relation-only coarse prediction."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor, nn

from .cache import TopologyCache
from .layers import EdgeGIN
from .model import GraphOutput
from .packed import PackedPlan, packed_gnn, segment_pool
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
        )).to(regions.dtype)
        totals = torch.zeros_like(regions).index_add(0, target, messages)
        degree = torch.zeros(regions.size(0), device=regions.device, dtype=regions.dtype)
        degree.index_add_(0, target, torch.ones_like(target, dtype=regions.dtype))
        mean = totals / degree.clamp_min(1).unsqueeze(1)
        delta = self.delta_projection(mean).to(regions.dtype)
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


def packed_v2_predict(predictor: V2CoarseGraphPredictor, node_states: Tensor, plan: PackedPlan) -> Tensor:
    """Execute V2's variable-size regions as one disjoint-union graph batch.

    V2 regions are exclusive, so their local region graphs have no shared
    atoms.  Concatenating them preserves every within-region edge while
    replacing thousands of tiny GPU launches with one launch per GNN layer.
    Coarse edges are likewise a disjoint union: offsets in ``plan`` ensure
    messages never cross molecule boundaries.
    """
    cfg = predictor.config

    def masked_mlp(module, values, mask):
        hidden = module[1](module[0](values))
        if mask is None:
            hidden = module[2](hidden)
        else:
            hidden = torch.ops.aten.native_dropout_backward(
                hidden, mask, 1.0 / (1.0 - module[2].p),
            )
        return module[3](hidden)

    canonical_nodes = node_states.flatten(0, 1)[plan.atom_gather]
    projected = predictor.input_projection(canonical_nodes)
    region_parts = [[] for _ in predictor.region_gnn.layers]
    message_parts, delta_parts = [], []
    # Draw masks in exactly the former per-graph execution order:
    # every region layer, then the coarse correction for that graph.
    for contexts, coarse_count, region_count in zip(
        plan.context_sizes, plan.coarse_lengths.tolist(), plan.graph_lengths.tolist(),
    ):
        for size in contexts:
            for index, layer in enumerate(predictor.region_gnn.layers):
                if predictor.training and layer.mlp[2].p:
                    region_parts[index].append(layer.mlp[2](torch.ones(
                        (size, 2 * cfg.hidden_dim), device=node_states.device,
                        dtype=node_states.dtype,
                    )).ne(0))
        if coarse_count and predictor.training and predictor.coarse_message[2].p:
            message_parts.append(predictor.coarse_message[2](torch.ones(
                (coarse_count, 2 * cfg.hidden_dim), device=node_states.device,
                dtype=node_states.dtype,
            )).ne(0))
            delta_parts.append(predictor.delta_projection[2](torch.ones(
                (region_count, 2 * cfg.hidden_dim), device=node_states.device,
                dtype=node_states.dtype,
            )).ne(0))
    region_masks = [torch.cat(parts) if parts else None for parts in region_parts]
    local = packed_gnn(
        predictor.region_gnn,
        projected[plan.context_gather],
        plan.region_source,
        plan.region_target,
        plan.region_attributes.to(dtype=node_states.dtype),
        region_masks,
    )
    regions = segment_pool(local[plan.core_gather], plan.core_lengths, "mean")
    sizes = plan.core_lengths.to(device=regions.device, dtype=regions.dtype)
    counts = plan.coarse_counts.to(device=regions.device, dtype=regions.dtype)
    bond_sums = plan.coarse_bond_sums.to(device=regions.device, dtype=regions.dtype)
    attributes = torch.cat((counts.log1p(), bond_sums / counts.clamp_min(1)), dim=1)
    if cfg.coarse_layers == 0 or plan.coarse_source.numel() == 0:
        delta = torch.zeros_like(regions)
    else:
        cursor = 0
        active = []
        for region_count, coarse_count in zip(plan.graph_lengths.tolist(), plan.coarse_lengths.tolist()):
            if coarse_count:
                active.append(torch.arange(cursor, cursor + region_count, device=regions.device))
            cursor += region_count
        message_mask = torch.cat(message_parts) if message_parts else None
        delta_mask = torch.cat(delta_parts) if delta_parts else None
        messages = masked_mlp(predictor.coarse_message, torch.cat((
            regions[plan.coarse_target], regions[plan.coarse_source], attributes,
        ), dim=1), message_mask).to(regions.dtype)
        totals = torch.zeros_like(regions).index_add(0, plan.coarse_target, messages)
        degree = torch.zeros(regions.size(0), device=regions.device, dtype=regions.dtype)
        degree.index_add_(0, plan.coarse_target, torch.ones_like(plan.coarse_target, dtype=regions.dtype))
        active = torch.cat(active)
        correction = masked_mlp(
            predictor.delta_projection,
            (totals / degree.clamp_min(1).unsqueeze(1))[active],
            delta_mask,
        ).to(regions.dtype)
        delta = torch.zeros_like(regions).index_copy(0, active, correction)
        delta = delta * degree.gt(0).to(regions.dtype).unsqueeze(1)
    region_graph = torch.segment_reduce(
        regions * sizes.unsqueeze(1), "sum", lengths=plan.graph_lengths,
    ) / torch.segment_reduce(sizes, "sum", lengths=plan.graph_lengths).unsqueeze(1)
    delta_graph = torch.segment_reduce(
        delta * sizes.unsqueeze(1), "sum", lengths=plan.graph_lengths,
    ) / torch.segment_reduce(sizes, "sum", lengths=plan.graph_lengths).unsqueeze(1)
    return predictor.region_head(region_graph) + predictor.alpha * predictor.delta_head(delta_graph)
