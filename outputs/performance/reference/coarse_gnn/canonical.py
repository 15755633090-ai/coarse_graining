"""BLISS canonical labeling of discrete vertex- and edge-attributed graphs.

Each physical edge becomes an auxiliary vertex. Atom and edge colors have
separate namespaces. Learned embeddings never enter the canonicalization.
"""
from __future__ import annotations

import torch
from torch import Tensor


def discrete_rows(labels: Tensor | None, count: int, name: str) -> Tensor:
    if labels is None:
        return torch.empty((count, 0), dtype=torch.long)
    if labels.dtype != torch.long or labels.ndim not in (1, 2) or labels.size(0) != count:
        raise ValueError(f"{name} must be a long tensor of shape [{count}] or [{count}, F]")
    return labels.detach().cpu().reshape(count, 1 if labels.ndim == 1 else labels.size(1))


def canonical_atom_order(num_nodes: int, edges: Tensor, node_labels: Tensor, edge_labels: Tensor) -> Tensor:
    """Return canonical-position -> input-atom index, excluding auxiliary nodes.

    ``edges`` must contain each undirected physical edge exactly once. Unique
    orders for individual symmetric atoms are neither promised nor necessary:
    the resulting attributed graph (and downstream canonical coarse graph) is
    invariant, while maps back to interchangeable input atoms can differ.
    """
    try:
        import igraph
    except ImportError as exc:
        raise ImportError("Canonical coarsening requires igraph; install the root requirements.txt") from exc
    atom_keys = [tuple(row) for row in node_labels.tolist()]
    bond_keys = [tuple(row) for row in edge_labels.tolist()]
    atom_palette = {key: i for i, key in enumerate(sorted(set(atom_keys)))}
    bond_palette = {key: len(atom_palette) + i for i, key in enumerate(sorted(set(bond_keys)))}
    colors = [atom_palette[key] for key in atom_keys] + [bond_palette[key] for key in bond_keys]
    incidence_edges = []
    for i, (u, v) in enumerate(edges.T.tolist()):
        incidence_edges.extend(((u, num_nodes + i), (v, num_nodes + i)))
    graph = igraph.Graph(n=num_nodes + edges.size(1), edges=incidence_edges, directed=False)
    # BLISS returns old-vertex -> new-label; sort labels of atoms ONLY.
    new_labels = graph.canonical_permutation(color=colors, sh="fl")
    return torch.tensor(sorted(range(num_nodes), key=new_labels.__getitem__), dtype=torch.long)
