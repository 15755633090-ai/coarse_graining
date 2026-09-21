"""Packed disjoint-union execution plan for LocalBoundary-GINE."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .hierarchy import HierarchicalTopology
from .local_boundary import split_local_boundary_component


@dataclass
class PackedLocalBoundaryPlan:
    atom_gather: Tensor
    l1_atom_lengths: Tensor
    l1_atom_counts: Tensor
    graph_l1_lengths: Tensor
    graph_atom_counts: Tensor
    full_edges: Tensor
    full_edge_attributes: Tensor
    local_edges: Tensor
    local_edge_attributes: Tensor
    boundary_edges: Tensor
    boundary_edge_attributes: Tensor
    padded_nodes: int

    def to(self, device):
        return type(self)(**{
            item.name: (
                getattr(self, item.name).to(device, non_blocking=True)
                if isinstance(getattr(self, item.name), Tensor)
                else getattr(self, item.name)
            )
            for item in fields(self)
        })

    def pin_memory(self):
        return type(self)(**{
            item.name: (
                getattr(self, item.name).pin_memory()
                if isinstance(getattr(self, item.name), Tensor)
                else getattr(self, item.name)
            )
            for item in fields(self)
        })


def l1_edge_attributes(topology: HierarchicalTopology, component_index: int) -> Tensor:
    """Create the single authoritative six-dimensional feature table for E1."""
    component = topology.components[component_index]
    level = component.levels[0]
    columns = []
    for key in topology.edge_type_keys:
        if len(key) != 1 or not 1 <= key[0] <= 4:
            raise ValueError("LocalBoundary molecular edges require scalar bond labels 1..4")
        columns.append(key[0] - 1)
    rows = []
    for edge_id, (left, right) in enumerate(level.edges.T.tolist()):
        count = int(level.edge_counts[edge_id])
        if count < 1:
            raise ValueError("L1 coarse edge count must be positive")
        proportions = [0.0] * 4
        for source_column, target_column in enumerate(columns):
            proportions[target_column] = float(level.edge_type_counts[edge_id, source_column]) / count
        density = count / max(1, min(int(level.atom_counts[left]), int(level.atom_counts[right])))
        rows.append([torch.log1p(torch.tensor(float(count))).item(), *proportions, density])
    return torch.tensor(rows, dtype=torch.float32).reshape(-1, 6)


def pack_local_boundary(
    topologies: list[HierarchicalTopology], padded_nodes: int,
) -> PackedLocalBoundaryPlan:
    """Pack atoms, L1 regions and aligned E/EA/EB tables for one batch."""
    if not topologies:
        raise ValueError("topologies must be nonempty")
    if not isinstance(padded_nodes, int) or padded_nodes < 1:
        raise ValueError("padded_nodes must be a positive integer")
    if any(topology.atom_order.numel() > padded_nodes for topology in topologies):
        raise ValueError("padded_nodes is smaller than a graph in the batch")

    atom_gather, lengths, atom_counts, graph_lengths, graph_atoms = [], [], [], [], []
    full_rows, full_attrs, local_rows, local_attrs, boundary_rows, boundary_attrs = [], [], [], [], [], []
    token_base = 0
    for graph_index, topology in enumerate(topologies):
        graph_tokens = 0
        graph_atoms.append(topology.atom_order.numel())
        for component_index, component in enumerate(topology.components):
            l1 = component.levels[0]
            plan = split_local_boundary_component(component)
            for members in l1.members:
                atom_gather.extend((topology.atom_order[members] + graph_index * padded_nodes).tolist())
                lengths.append(members.numel())
                atom_counts.append(members.numel())
            attributes = l1_edge_attributes(topology, component_index)
            if attributes.size(0) != l1.edges.size(1):
                raise AssertionError("Full E1 attributes are not aligned with E1")
            shifted = l1.edges + token_base
            full_rows.append(shifted)
            full_attrs.append(attributes)
            local_rows.append(shifted[:, plan.local_edge_ids])
            local_attrs.append(attributes[plan.local_edge_ids])
            boundary_rows.append(shifted[:, plan.boundary_edge_ids])
            boundary_attrs.append(attributes[plan.boundary_edge_ids])
            graph_tokens += l1.num_tokens
            token_base += l1.num_tokens
        graph_lengths.append(graph_tokens)

    def edges(rows):
        return torch.cat(rows, dim=1) if rows else torch.empty((2, 0), dtype=torch.long)

    def attributes(rows):
        return torch.cat(rows, dim=0) if rows else torch.empty((0, 6), dtype=torch.float32)

    full_e, full_a = edges(full_rows), attributes(full_attrs)
    local_e, local_a = edges(local_rows), attributes(local_attrs)
    boundary_e, boundary_a = edges(boundary_rows), attributes(boundary_attrs)
    if full_e.size(1) != local_e.size(1) + boundary_e.size(1):
        raise AssertionError("Packed EA and EB do not partition packed E1")
    for name, edge_table, attr_table in (
        ("E", full_e, full_a), ("EA", local_e, local_a), ("EB", boundary_e, boundary_a),
    ):
        if edge_table.size(1) != attr_table.size(0):
            raise AssertionError(f"Packed {name} edge/attribute alignment failed")
    return PackedLocalBoundaryPlan(
        atom_gather=torch.tensor(atom_gather, dtype=torch.long),
        l1_atom_lengths=torch.tensor(lengths, dtype=torch.long),
        l1_atom_counts=torch.tensor(atom_counts, dtype=torch.long),
        graph_l1_lengths=torch.tensor(graph_lengths, dtype=torch.long),
        graph_atom_counts=torch.tensor(graph_atoms, dtype=torch.long),
        full_edges=full_e, full_edge_attributes=full_a,
        local_edges=local_e, local_edge_attributes=local_a,
        boundary_edges=boundary_e, boundary_edge_attributes=boundary_a,
        padded_nodes=padded_nodes,
    )
