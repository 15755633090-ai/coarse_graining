"""CPU compilation of hierarchical topology into one GPU execution plan."""
from __future__ import annotations

from dataclasses import dataclass, fields

import torch
from torch import Tensor

from .hierarchy import HierarchicalTopology, PackedHierarchyContextPlan, pack_hierarchy_contexts


class _TensorPlan:
    def to(self, device):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, Tensor):
                values[item.name] = value.to(device, non_blocking=True)
            elif hasattr(value, "to"):
                values[item.name] = value.to(device)
            else:
                values[item.name] = value
        return type(self)(**values)

    def pin_memory(self):
        values = {}
        for item in fields(self):
            value = getattr(self, item.name)
            values[item.name] = value.pin_memory() if isinstance(value, Tensor) else (
                value.pin_memory() if hasattr(value, "pin_memory") else value
            )
        return type(self)(**values)


@dataclass
class PackedHierarchyLevelPlan(_TensorPlan):
    """One packed recursive level, with parents grouped contiguously."""

    child_gather: Tensor
    parent_lengths: Tensor
    child_atom_counts: Tensor
    atom_counts: Tensor
    leaf_counts: Tensor
    edges: Tensor
    edge_attributes: Tensor
    structure_features: Tensor

    @property
    def num_tokens(self) -> int:
        return int(self.parent_lengths.numel())


@dataclass
class PackedHierarchyPlan(_TensorPlan):
    """Complete Atom->L1->L2->L3 and adaptive-context batch plan."""

    atom_gather: Tensor
    l1_atom_lengths: Tensor
    l1_atom_counts: Tensor
    graph_l1_lengths: Tensor
    graph_atom_counts: Tensor
    l2: PackedHierarchyLevelPlan
    l3: PackedHierarchyLevelPlan
    contexts: PackedHierarchyContextPlan
    padded_nodes: int


def _empty_level() -> PackedHierarchyLevelPlan:
    return PackedHierarchyLevelPlan(
        child_gather=torch.empty(0, dtype=torch.long),
        parent_lengths=torch.empty(0, dtype=torch.long),
        child_atom_counts=torch.empty(0, dtype=torch.long),
        atom_counts=torch.empty(0, dtype=torch.long),
        leaf_counts=torch.empty(0, dtype=torch.long),
        edges=torch.empty((2, 0), dtype=torch.long),
        edge_attributes=torch.empty((0, 6), dtype=torch.float32),
        structure_features=torch.empty((0, 6), dtype=torch.float32),
    )


def _bond_columns(topology: HierarchicalTopology) -> list[int]:
    columns = []
    for key in topology.edge_type_keys:
        if len(key) != 1 or not 1 <= key[0] <= 4:
            raise ValueError("packed molecular hierarchy requires scalar bond labels 1..4")
        columns.append(key[0] - 1)
    return columns


def _pack_level(topologies, target_level: int, bases) -> PackedHierarchyLevelPlan:
    child_gather, parent_lengths, child_atom_counts = [], [], []
    atom_counts, leaf_counts, edge_rows, edge_features, structures = [], [], [], [], []
    for graph_index, topology in enumerate(topologies):
        for component_index, component in enumerate(topology.components):
            if len(component.levels) < target_level:
                continue
            bond_columns = _bond_columns(topology)
            level = component.levels[target_level - 1]
            previous = component.levels[target_level - 2]
            child_base = bases[(graph_index, component_index, target_level - 2)]
            boundary = torch.zeros(level.num_tokens, dtype=torch.long)
            if level.edges.numel():
                boundary.index_add_(0, level.edges[0], level.edge_counts)
                boundary.index_add_(0, level.edges[1], level.edge_counts)
            for parent, members in enumerate(level.members):
                parent_lengths.append(members.numel())
                child_gather.extend((members + child_base).tolist())
                child_atom_counts.extend(previous.atom_counts[members].tolist())
                atom_counts.append(int(level.atom_counts[parent]))
                leaf_counts.append(level.leaf_sets[parent].numel())
                structures.append([
                    torch.log1p(level.atom_counts[parent].float()).item(),
                    torch.log1p(torch.tensor(float(level.leaf_sets[parent].numel()))).item(),
                    float(level.child_radii[parent]),
                    float(level.child_diameters[parent]),
                    torch.log1p(level.internal_edge_counts[parent].float()).item(),
                    torch.log1p(boundary[parent].float()).item(),
                ])
            # GINE runs on child tokens inside each parent block. Cross-parent
            # edges are deliberately excluded from this local representation.
            for edge_id, (left, right) in enumerate(previous.edges.T.tolist()):
                if int(level.owner[left]) != int(level.owner[right]):
                    continue
                count = int(previous.edge_counts[edge_id])
                proportions = [0.0] * 4
                for source_column, target_column in enumerate(bond_columns):
                    proportions[target_column] = float(previous.edge_type_counts[edge_id, source_column]) / count
                density = count / max(1, min(int(previous.atom_counts[left]), int(previous.atom_counts[right])))
                edge_rows.append((child_base + left, child_base + right))
                edge_features.append([torch.log1p(torch.tensor(float(count))).item(), *proportions, density])
    if not parent_lengths:
        return _empty_level()
    edges = torch.tensor(edge_rows, dtype=torch.long).T.contiguous() if edge_rows else torch.empty((2, 0), dtype=torch.long)
    return PackedHierarchyLevelPlan(
        child_gather=torch.tensor(child_gather, dtype=torch.long),
        parent_lengths=torch.tensor(parent_lengths, dtype=torch.long),
        child_atom_counts=torch.tensor(child_atom_counts, dtype=torch.long),
        atom_counts=torch.tensor(atom_counts, dtype=torch.long),
        leaf_counts=torch.tensor(leaf_counts, dtype=torch.long),
        edges=edges,
        edge_attributes=torch.tensor(edge_features, dtype=torch.float32).reshape(-1, 6),
        structure_features=torch.tensor(structures, dtype=torch.float32).reshape(-1, 6),
    )


def pack_hierarchy(
    topologies: list[HierarchicalTopology], padded_nodes: int, *, full_l1: bool = False,
) -> PackedHierarchyPlan:
    """Compile a batch on CPU; the returned object can be pinned and copied once."""
    if not topologies:
        raise ValueError("topologies must be nonempty")
    if not isinstance(padded_nodes, int) or padded_nodes < 1:
        raise ValueError("padded_nodes must be a positive integer")
    if any(topology.atom_order.numel() > padded_nodes for topology in topologies):
        raise ValueError("padded_nodes is smaller than a graph in the batch")

    bases = {}
    totals = [0, 0, 0]
    for graph_index, topology in enumerate(topologies):
        for component_index, component in enumerate(topology.components):
            for level_index, level in enumerate(component.levels):
                bases[(graph_index, component_index, level_index)] = totals[level_index]
                totals[level_index] += level.num_tokens

    atom_gather, l1_lengths, l1_counts, graph_l1, graph_atoms = [], [], [], [], []
    for graph_index, topology in enumerate(topologies):
        graph_tokens = 0
        graph_atoms.append(topology.atom_order.numel())
        for component in topology.components:
            level = component.levels[0]
            graph_tokens += level.num_tokens
            for members in level.members:
                input_positions = topology.atom_order[members] + graph_index * padded_nodes
                atom_gather.extend(input_positions.tolist())
                l1_lengths.append(members.numel())
                l1_counts.append(members.numel())
        graph_l1.append(graph_tokens)

    contexts = pack_hierarchy_contexts(topologies, full_l1=full_l1)
    if totals != contexts.level_token_counts.tolist():
        raise RuntimeError("hierarchy and context packing orders disagree")
    return PackedHierarchyPlan(
        atom_gather=torch.tensor(atom_gather, dtype=torch.long),
        l1_atom_lengths=torch.tensor(l1_lengths, dtype=torch.long),
        l1_atom_counts=torch.tensor(l1_counts, dtype=torch.long),
        graph_l1_lengths=torch.tensor(graph_l1, dtype=torch.long),
        graph_atom_counts=torch.tensor(graph_atoms, dtype=torch.long),
        l2=_pack_level(topologies, 2, bases),
        l3=_pack_level(topologies, 3, bases),
        contexts=contexts,
        padded_nodes=padded_nodes,
    )
