"""Topology-only local/boundary edge schedules over canonical L1 tokens."""
from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .hierarchy import ComponentHierarchy, HierarchicalTopology


@dataclass(frozen=True)
class LocalBoundaryComponentPlan:
    """A disjoint, exhaustive split of one component's undirected L1 edges."""

    window_owner: Tensor
    window_members: tuple[Tensor, ...]
    local_edge_ids: Tensor
    boundary_edge_ids: Tensor
    local_edges: Tensor
    boundary_edges: Tensor


def _validate_parent_membership(component: ComponentHierarchy) -> tuple[Tensor, tuple[Tensor, ...]]:
    l1 = component.levels[0]
    count = l1.num_tokens
    if len(component.levels) < 2:
        members = (torch.arange(count, dtype=torch.long),)
        return torch.zeros(count, dtype=torch.long), members

    parent = component.levels[1]
    owner = parent.owner.to(dtype=torch.long, device="cpu").contiguous()
    if owner.shape != (count,):
        raise ValueError("Cached L2 owner is not aligned one-to-one with canonical L1 tokens")
    if owner.numel() and (int(owner.min()) < 0 or int(owner.max()) >= parent.num_tokens):
        raise ValueError("Cached L2 owner contains an invalid parent index")

    members = tuple(value.to(dtype=torch.long, device="cpu").contiguous()
                    for value in parent.members)
    flattened = torch.cat(members) if members else torch.empty(0, dtype=torch.long)
    if not torch.equal(torch.sort(flattened).values, torch.arange(count)):
        raise ValueError("Cached L2 members do not cover canonical L1 tokens exactly once")
    for parent_index, children in enumerate(members):
        expected = torch.nonzero(owner == parent_index, as_tuple=False).flatten()
        if not torch.equal(torch.sort(children).values, expected):
            raise ValueError("Cached L2 members disagree with the L1-to-parent owner map")
        leaves = parent.leaf_sets[parent_index].to(dtype=torch.long, device="cpu")
        if not torch.equal(torch.sort(leaves).values, expected):
            raise ValueError("Cached L2 leaf set is not expressed in canonical L1 indices")
    return owner, members


def split_local_boundary_component(component: ComponentHierarchy) -> LocalBoundaryComponentPlan:
    """Use cached L2 membership as A windows and split every real L1 edge."""
    l1 = component.levels[0]
    owner, members = _validate_parent_membership(component)
    edges = l1.edges.to(dtype=torch.long, device="cpu").contiguous()
    if edges.ndim != 2 or edges.size(0) != 2:
        raise ValueError("Canonical L1 edges must have shape [2, E]")
    if edges.numel() and (int(edges.min()) < 0 or int(edges.max()) >= l1.num_tokens):
        raise ValueError("Canonical L1 edge endpoint is out of range")

    is_local = owner[edges[0]] == owner[edges[1]]
    local_ids = torch.nonzero(is_local, as_tuple=False).flatten()
    boundary_ids = torch.nonzero(~is_local, as_tuple=False).flatten()
    all_ids = torch.cat((local_ids, boundary_ids))
    expected = torch.arange(edges.size(1), dtype=torch.long)
    if local_ids.numel() and boundary_ids.numel():
        if bool(torch.isin(local_ids, boundary_ids).any()):
            raise AssertionError("EA and EB overlap")
    if not torch.equal(torch.sort(all_ids).values, expected):
        raise AssertionError("EA and EB do not exactly cover E1")

    return LocalBoundaryComponentPlan(
        window_owner=owner,
        window_members=members,
        local_edge_ids=local_ids,
        boundary_edge_ids=boundary_ids,
        local_edges=edges[:, local_ids],
        boundary_edges=edges[:, boundary_ids],
    )


def split_local_boundary_topology(
    topology: HierarchicalTopology,
) -> tuple[LocalBoundaryComponentPlan, ...]:
    """Create one validated local/boundary plan per canonical component."""
    return tuple(split_local_boundary_component(component) for component in topology.components)


def temporal_reachability(num_tokens: int, edge_schedule: tuple[Tensor, ...]) -> Tensor:
    """Return ordered source reachability after a time-ordered undirected schedule."""
    reachable = torch.eye(num_tokens, dtype=torch.bool)
    for edges in edge_schedule:
        adjacency = torch.eye(num_tokens, dtype=torch.bool)
        if edges.numel():
            adjacency[edges[0], edges[1]] = True
            adjacency[edges[1], edges[0]] = True
        reachable = (adjacency.to(torch.int32) @ reachable.to(torch.int32)) > 0
    return reachable


def local_boundary_reachability(
    component: ComponentHierarchy,
    plan: LocalBoundaryComponentPlan,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compare Local-only (EA, empty) with LocalBoundary (EA, EB)."""
    count = component.levels[0].num_tokens
    empty = torch.empty((2, 0), dtype=torch.long)
    local = temporal_reachability(count, (plan.local_edges, empty))
    local_boundary = temporal_reachability(
        count, (plan.local_edges, plan.boundary_edges),
    )
    added = local_boundary & ~local
    if bool(torch.diagonal(added).any()):
        raise AssertionError("Boundary propagation cannot add a self relation")
    return local, local_boundary, added
