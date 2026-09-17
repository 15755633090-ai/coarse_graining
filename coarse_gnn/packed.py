"""Disjoint-union execution of the unchanged molecular coarse GNN.

Plans are constructed from verified CPU topologies, before device transfer.
There are no new weights, no cross-molecule edges, and no changes to pooling.
Dropout masks are drawn in the original graph/region/layer call order.
"""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import nn

from .diffusion_adapter import molecular_graph_inputs


@dataclass
class PackedPlan:
    atom_gather: torch.Tensor
    context_gather: torch.Tensor
    region_source: torch.Tensor
    region_target: torch.Tensor
    region_attributes: torch.Tensor
    core_gather: torch.Tensor
    core_lengths: torch.Tensor
    coarse_source: torch.Tensor
    coarse_target: torch.Tensor
    coarse_counts: torch.Tensor
    coarse_bond_sums: torch.Tensor
    graph_lengths: torch.Tensor
    coarse_lengths: torch.Tensor
    context_sizes: tuple[tuple[int, ...], ...]

    def to(self, device):
        return PackedPlan(**{f.name: (getattr(self, f.name).to(device, non_blocking=True)
                                     if isinstance(getattr(self, f.name), torch.Tensor) else getattr(self, f.name))
                             for f in fields(self)})


def make_plan(topologies, padded_nodes):
    atom, contexts, src, dst, attrs, core, lengths = [], [], [], [], [], [], []
    coarse_src, coarse_dst, counts, sums, graph_lengths, coarse_lengths, sizes = [], [], [], [], [], [], []
    node_offset = context_offset = region_offset = 0
    for graph_id, topology in enumerate(topologies):
        atom.append(topology.atom_order + graph_id * padded_nodes)
        bonds = nn.functional.one_hot(topology.edge_labels[:, 0] - 1, num_classes=4).float()
        graph_sizes = []
        for context, edges, ids, positions, owned in zip(
            topology.contexts, topology.context_edges, topology.context_edge_ids,
            topology.core_positions, topology.cores,
        ):
            contexts.append(context + node_offset)
            src.append(torch.cat((edges[0], edges[1])) + context_offset)
            dst.append(torch.cat((edges[1], edges[0])) + context_offset)
            attrs.append(bonds[ids].repeat(2, 1))
            core.append(positions + context_offset)
            lengths.append(len(owned))
            graph_sizes.append(len(context))
            context_offset += len(context)
        edges = topology.coarse_edges
        coarse_src.append(torch.cat((edges[0], edges[1])) + region_offset)
        coarse_dst.append(torch.cat((edges[1], edges[0])) + region_offset)
        coarse_lengths.append(2 * edges.size(1))
        edge_counts = topology.edge_counts.float().unsqueeze(1)
        boundary = torch.zeros(len(edge_counts), 4).index_add_(
            0, topology.boundary_groups, bonds[topology.boundary_edge_ids])
        counts.append(edge_counts.repeat(2, 1))
        sums.append(boundary.repeat(2, 1))
        graph_lengths.append(len(topology.cores))
        sizes.append(tuple(graph_sizes))
        node_offset += len(topology.owner)
        region_offset += len(topology.cores)
    return PackedPlan(torch.cat(atom), torch.cat(contexts), torch.cat(src), torch.cat(dst),
                      torch.cat(attrs), torch.cat(core), torch.tensor(lengths),
                      torch.cat(coarse_src), torch.cat(coarse_dst), torch.cat(counts), torch.cat(sums),
                      torch.tensor(graph_lengths), torch.tensor(coarse_lengths), tuple(sizes))


class PreparedGraphBatch:
    """Immutable graph input wrapper, produced by the formal CPU collator.

    Callers must not mutate graph tensors after preparation. General unprepared
    model inputs retain the original validation path.
    """
    def __init__(self, graph, topologies, plan=None):
        self.graph = graph
        self.topologies = tuple(topologies)
        self.plan = plan if plan is not None else make_plan(self.topologies, graph.node_features.size(1))

    def __getattr__(self, name):
        return getattr(self.graph, name)

    def to(self, device):
        return PreparedGraphBatch(self.graph.to(device), self.topologies, self.plan.to(device))


def prepare_graph_batch(graph, predictor):
    if graph.node_features.device.type != "cpu":
        raise ValueError("Prepare molecular topology on CPU before batch.to(device)")
    topologies = [predictor.prepare_topology(len(valid), edges, node_labels=nodes, edge_labels=labels)
                  for valid, edges, nodes, labels in molecular_graph_inputs(graph)]
    return PreparedGraphBatch(graph, topologies)


def dropout_masks(predictor, plan, dtype, device):
    cfg = predictor.config
    if not predictor.training or cfg.dropout == 0:
        return [None] * cfg.region_layers, [None] * cfg.coarse_layers
    regions = [[] for _ in range(cfg.region_layers)]
    coarse = [[] for _ in range(cfg.coarse_layers)]
    def mask(size, layer):
        # Same dropout operator, shape, dtype and RNG call ordering as legacy.
        values = torch.ones((size, 2 * cfg.hidden_dim), device=device, dtype=dtype)
        return layer.mlp[2](values).ne(0)
    for graph in plan.context_sizes:
        for size in graph:
            for index, layer in enumerate(predictor.region_gnn.layers):
                regions[index].append(mask(size, layer))
        for index, layer in enumerate(predictor.coarse_gnn.layers):
            coarse[index].append(mask(len(graph), layer))
    return [torch.cat(parts) for parts in regions], [torch.cat(parts) for parts in coarse]


def packed_gnn(gnn, x, source, target, attributes, masks):
    for layer, mask in zip(gnn.layers, masks):
        messages = x[source]
        if layer.edge_projection is not None:
            messages = torch.relu(messages + layer.edge_projection(attributes))
        summed = torch.zeros_like(x).index_add(0, target, messages)
        hidden = layer.mlp[1](layer.mlp[0]((1 + layer.epsilon) * x + summed))
        if mask is None:
            hidden = layer.mlp[2](hidden)
        else:
            # Dropout's linear mask/scale map equals its backward operator.
            # Unlike multiplying a rounded BF16 mask value, this retains the
            # original scalar precision and has the correct autograd derivative.
            hidden = torch.ops.aten.native_dropout_backward(hidden, mask, 1.0 / (1.0 - layer.mlp[2].p))
        x = layer.norm(x + layer.mlp[3](hidden))
    return x


def segment_pool(values, lengths, reduction):
    # mean/sum on BF16 tensors accumulate in FP32 in the original torch pooling.
    return torch.segment_reduce(values.float(), reduction, lengths=lengths).to(values.dtype)


def packed_predict(predictor, node_states, batch, *, return_intermediates=False):
    cfg, plan = predictor.config, batch.plan
    canonical_nodes = node_states.flatten(0, 1)[plan.atom_gather]
    projected = predictor.input_projection(canonical_nodes)
    region_masks, coarse_masks = dropout_masks(predictor, plan, projected.dtype, projected.device)
    local = packed_gnn(predictor.region_gnn, projected[plan.context_gather],
                       plan.region_source, plan.region_target, plan.region_attributes.to(node_states.dtype), region_masks)
    regions = segment_pool(local[plan.core_gather], plan.core_lengths, cfg.region_pool)
    sizes = plan.core_lengths.to(node_states.dtype)
    coarse_input = regions
    if predictor.size_projection is not None:
        coarse_input = coarse_input + predictor.size_projection(sizes.log1p().unsqueeze(1))
    enabled = []
    if cfg.use_coarse_edge_count:
        enabled.append(plan.coarse_counts.to(node_states.dtype))
    if cfg.use_coarse_edge_features:
        attributes = plan.coarse_bond_sums.to(node_states.dtype)
        if cfg.edge_reduce == "mean":
            attributes = attributes / plan.coarse_counts.clamp_min(1)
        enabled.append(attributes)
    coarse_attributes = torch.cat(enabled, 1) if enabled else node_states.new_empty((len(plan.coarse_counts), 0))
    coarse = packed_gnn(predictor.coarse_gnn, coarse_input, plan.coarse_source,
                        plan.coarse_target, coarse_attributes, coarse_masks)
    if cfg.graph_pool == "size_weighted_mean":
        graph = torch.segment_reduce(coarse * sizes.unsqueeze(1), "sum", lengths=plan.graph_lengths)
        graph = graph / torch.segment_reduce(sizes, "sum", lengths=plan.graph_lengths).unsqueeze(1)
    else:
        graph = segment_pool(coarse, plan.graph_lengths, cfg.graph_pool)
    predictions = predictor.head(graph)
    if return_intermediates:
        return predictions, regions, coarse, graph
    return predictions
