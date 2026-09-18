"""Deterministic topology-only hierarchy for near-fine/far-coarse models.

This module deliberately contains no learned operations.  It turns a discrete
attributed molecular graph into independent component hierarchies, preserves
raw edge sufficient statistics at every level, and constructs the exact
query-dependent, non-overlapping multiresolution cover used by the model.
"""
from __future__ import annotations

import hashlib
import json
import math
from collections import deque
from dataclasses import asdict, dataclass, field, fields

import torch
from torch import Tensor

from .canonical import canonical_atom_order, discrete_rows
from .topology import _bliss_version, _canonical_edges, _pairs_tensor


HIERARCHY_VERSION = 3


@dataclass(frozen=True)
class HierarchyConfig:
    l1_cover_radius: int = 2
    l1_soft_max_atoms: int = 12
    target_children: int = 3
    stop_num_tokens: int = 6
    stop_diameter: int = 2
    max_levels: int = 3
    expansion_threshold: float = 0.75
    canonicalize: bool = True
    algorithm: str = "near_fine_far_coarse_v1"

    def __post_init__(self):
        for name in (
            "l1_cover_radius", "l1_soft_max_atoms", "target_children",
            "stop_num_tokens", "stop_diameter", "max_levels",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or value < (0 if name in {"l1_cover_radius", "stop_diameter"} else 1):
                raise ValueError(f"{name} has an invalid value")
        if self.target_children < 2:
            raise ValueError("target_children must be at least two")
        if self.max_levels > 3:
            raise ValueError("V1 supports at most L1/L2/L3")
        if not math.isfinite(self.expansion_threshold) or self.expansion_threshold <= 0:
            raise ValueError("expansion_threshold must be finite and positive")
        if not isinstance(self.canonicalize, bool) or not self.canonicalize:
            raise ValueError("the hierarchy requires canonicalization")
        if self.algorithm != "near_fine_far_coarse_v1":
            raise ValueError("algorithm is fixed by the method definition")


@dataclass
class HierarchyLevel:
    """One level within one connected component.

    ``members`` contains canonical atom indices at L1 and child-token indices
    at higher levels. ``leaf_sets`` always contains L1 token indices.
    """

    level: int
    members: list[Tensor]
    leaf_sets: list[Tensor]
    centers: Tensor
    owner: Tensor
    edges: Tensor
    edge_counts: Tensor
    edge_type_counts: Tensor
    internal_edge_counts: Tensor
    internal_edge_type_counts: Tensor
    atom_counts: Tensor
    child_radii: Tensor
    child_diameters: Tensor
    leaf_diameters: Tensor

    @property
    def num_tokens(self) -> int:
        return len(self.members)


@dataclass(frozen=True)
class ContextToken:
    level: int
    index: int
    d_min: int
    d_max: int
    d_mean: float
    rho: int


@dataclass
class PackedHierarchyContextPlan:
    """CPU index plan for one packed cross-scale attention execution.

    Context levels are one-based. ``context_indices`` index the concatenated
    token tensor of the corresponding level, not a per-graph tensor.  The plan
    contains no neural activations and can therefore be cached with topology.
    """

    query_l1_indices: Tensor
    query_offsets: Tensor
    context_levels: Tensor
    context_indices: Tensor
    context_query_indices: Tensor
    d_min: Tensor
    d_max: Tensor
    d_mean: Tensor
    rho: Tensor
    leaf_counts: Tensor
    atom_counts: Tensor
    query_diameters: Tensor
    graph_query_lengths: Tensor
    level_token_counts: Tensor

    def to(self, device) -> "PackedHierarchyContextPlan":
        """Move every packed tensor together for non-blocking GPU execution."""
        return PackedHierarchyContextPlan(**{
            item.name: getattr(self, item.name).to(device, non_blocking=True)
            for item in fields(self)
        })

    def pin_memory(self) -> "PackedHierarchyContextPlan":
        """Allow DataLoader pin-memory traversal to prepare one async copy."""
        return PackedHierarchyContextPlan(**{
            item.name: getattr(self, item.name).pin_memory()
            for item in fields(self)
        })


@dataclass
class ComponentHierarchy:
    atom_indices: Tensor
    levels: list[HierarchyLevel]
    l1_distances: Tensor
    _context_cache: dict[float, list[list[ContextToken] | None]] = field(
        default_factory=dict, repr=False, compare=False,
    )

    def adaptive_context(
        self, query: int, threshold: float = 0.75,
    ) -> list[ContextToken]:
        """Return an exact cover of all L1 leaves except ``query``.

        Near leaves (distance at most one) are always expanded to L1.  A coarse
        token is retained when its induced-L1 diameter divided by d_min + 1 is
        no greater than ``threshold``.
        """
        if not 0 <= query < self.levels[0].num_tokens:
            raise IndexError("query is not an L1 token")
        if not math.isfinite(threshold) or threshold <= 0:
            raise ValueError("threshold must be finite and positive")
        cached = self._context_cache.get(float(threshold))
        if cached is not None and query < len(cached) and cached[query] is not None:
            return list(cached[query])
        if self.levels[0].num_tokens == 1:
            self._remember_context(query, threshold, [])
            return []

        context: list[ContextToken] = []

        def visit(level_index: int, token_index: int) -> None:
            level = self.levels[level_index]
            leaves = level.leaf_sets[token_index]
            if level_index == 0:
                if token_index != query:
                    distance = int(self.l1_distances[query, token_index])
                    context.append(ContextToken(1, token_index, distance, distance, float(distance), 0))
                return
            distances = self.l1_distances[query, leaves]
            d_min = int(distances.min())
            d_max = int(distances.max())
            d_mean = float(distances.float().mean())
            rho = int(level.leaf_diameters[token_index])
            contains_query = bool((leaves == query).any())
            if not contains_query and d_min > 1 and rho / (d_min + 1) <= threshold:
                context.append(ContextToken(level_index + 1, token_index, d_min, d_max, d_mean, rho))
                return
            for child in level.members[token_index].tolist():
                visit(level_index - 1, child)

        for root in range(self.levels[-1].num_tokens):
            visit(len(self.levels) - 1, root)

        covered: list[int] = []
        for token in context:
            covered.extend(self.levels[token.level - 1].leaf_sets[token.index].tolist())
        expected = [i for i in range(self.levels[0].num_tokens) if i != query]
        if sorted(covered) != expected or len(covered) != len(set(covered)):
            raise RuntimeError("adaptive context is not an exact, non-overlapping L1 cover")
        result = sorted(context, key=lambda token: (
            int(self.levels[token.level - 1].leaf_sets[token.index][0]), token.level, token.index,
        ))
        self._remember_context(query, threshold, result)
        return result

    def _remember_context(self, query: int, threshold: float, context: list[ContextToken]) -> None:
        key = float(threshold)
        slots = self._context_cache.setdefault(
            key, [None] * self.levels[0].num_tokens,
        )
        slots[query] = list(context)


@dataclass
class HierarchicalTopology:
    components: list[ComponentHierarchy]
    atom_order: Tensor
    input_to_canonical: Tensor
    node_labels: Tensor
    edges: Tensor
    edge_labels: Tensor
    edge_type_keys: tuple[tuple[int, ...], ...]
    input_fingerprint: str
    canonical_fingerprint: str
    config: HierarchyConfig

    def canonical_signature(self) -> str:
        payload = {
            "version": HIERARCHY_VERSION,
            "node_labels": self.node_labels.tolist(),
            "edges": self.edges.T.tolist(),
            "edge_labels": self.edge_labels.tolist(),
            "components": [
                {
                    "atoms": component.atom_indices.tolist(),
                    "levels": [
                        {
                            "members": [value.tolist() for value in level.members],
                            "leaves": [value.tolist() for value in level.leaf_sets],
                            "edges": level.edges.T.tolist(),
                            "edge_counts": level.edge_counts.tolist(),
                            "edge_types": level.edge_type_counts.tolist(),
                            "internal": level.internal_edge_counts.tolist(),
                            "internal_types": level.internal_edge_type_counts.tolist(),
                            "atom_counts": level.atom_counts.tolist(),
                            "centers": level.centers.tolist(),
                            "owner": level.owner.tolist(),
                            "child_radii": level.child_radii.tolist(),
                            "child_diameters": level.child_diameters.tolist(),
                            "leaf_diameters": level.leaf_diameters.tolist(),
                        }
                        for level in component.levels
                    ],
                }
                for component in self.components
            ],
        }
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()

    def diagnostics(self) -> dict[str, object]:
        level_counts = [[level.num_tokens for level in component.levels] for component in self.components]
        contexts = [
            component.adaptive_context(query, self.config.expansion_threshold)
            for component in self.components
            for query in range(component.levels[0].num_tokens)
        ]
        return {
            "num_components": len(self.components),
            "level_counts": level_counts,
            "built_l2": any(len(component.levels) >= 2 for component in self.components),
            "built_l3": any(len(component.levels) >= 3 for component in self.components),
            "used_l2": any(any(token.level == 2 for token in context) for context in contexts),
            "used_l3": any(any(token.level == 3 for token in context) for context in contexts),
            "mean_context_ratio": _mean_context_ratio(self.components, contexts),
        }


def pack_hierarchy_contexts(
    topologies: list[HierarchicalTopology],
    *,
    full_l1: bool = False,
) -> PackedHierarchyContextPlan:
    """Precompile every graph/query tree traversal into flat CPU tensors.

    This function is intended for data preparation or collation. A model can
    concatenate L1/L2/L3 embeddings once, gather context rows with these
    indices, and use ``query_offsets`` for segmented softmax without a Python
    loop over molecules or queries.
    """
    level_bases: dict[tuple[int, int, int], int] = {}
    level_counts = [0, 0, 0]
    for graph_index, topology in enumerate(topologies):
        for component_index, component in enumerate(topology.components):
            for level_index, level in enumerate(component.levels):
                level_bases[(graph_index, component_index, level_index)] = level_counts[level_index]
                level_counts[level_index] += level.num_tokens

    query_indices, offsets = [], [0]
    context_levels, context_indices, context_query_indices = [], [], []
    d_min, d_max, d_mean, rho = [], [], [], []
    leaf_counts, atom_counts, query_diameters = [], [], []
    graph_query_lengths = []
    for graph_index, topology in enumerate(topologies):
        graph_queries = 0
        for component_index, component in enumerate(topology.components):
            l1_base = level_bases[(graph_index, component_index, 0)]
            component_diameter = int(component.l1_distances.max())
            for query in range(component.levels[0].num_tokens):
                packed_query = len(query_indices)
                graph_queries += 1
                query_indices.append(l1_base + query)
                query_diameters.append(max(1, component_diameter))
                if full_l1:
                    context = [
                        ContextToken(
                            1, index,
                            int(component.l1_distances[query, index]),
                            int(component.l1_distances[query, index]),
                            float(component.l1_distances[query, index]), 0,
                        )
                        for index in range(component.levels[0].num_tokens)
                        if index != query
                    ]
                else:
                    context = component.adaptive_context(
                        query, topology.config.expansion_threshold,
                    )
                for token in context:
                    context_query_indices.append(packed_query)
                    context_levels.append(token.level)
                    context_indices.append(
                        level_bases[(graph_index, component_index, token.level - 1)] + token.index
                    )
                    d_min.append(token.d_min)
                    d_max.append(token.d_max)
                    d_mean.append(token.d_mean)
                    rho.append(token.rho)
                    token_level = component.levels[token.level - 1]
                    leaf_counts.append(token_level.leaf_sets[token.index].numel())
                    atom_counts.append(int(token_level.atom_counts[token.index]))
                offsets.append(offsets[-1] + len(context))
        graph_query_lengths.append(graph_queries)

    return PackedHierarchyContextPlan(
        query_l1_indices=torch.tensor(query_indices, dtype=torch.long),
        query_offsets=torch.tensor(offsets, dtype=torch.long),
        context_levels=torch.tensor(context_levels, dtype=torch.long),
        context_indices=torch.tensor(context_indices, dtype=torch.long),
        context_query_indices=torch.tensor(context_query_indices, dtype=torch.long),
        d_min=torch.tensor(d_min, dtype=torch.long),
        d_max=torch.tensor(d_max, dtype=torch.long),
        d_mean=torch.tensor(d_mean, dtype=torch.float32),
        rho=torch.tensor(rho, dtype=torch.long),
        leaf_counts=torch.tensor(leaf_counts, dtype=torch.long),
        atom_counts=torch.tensor(atom_counts, dtype=torch.long),
        query_diameters=torch.tensor(query_diameters, dtype=torch.long),
        graph_query_lengths=torch.tensor(graph_query_lengths, dtype=torch.long),
        level_token_counts=torch.tensor(level_counts, dtype=torch.long),
    )


def _mean_context_ratio(components, contexts) -> float:
    ratios, cursor = [], 0
    for component in components:
        count = component.levels[0].num_tokens
        for _ in range(count):
            context = contexts[cursor]
            cursor += 1
            if count > 1:
                ratios.append(len(context) / (count - 1))
    return sum(ratios) / len(ratios) if ratios else 0.0


def hierarchy_input_fingerprint(
    num_nodes: int,
    edge_index: Tensor,
    config: HierarchyConfig,
    *,
    node_labels: Tensor | None = None,
    edge_labels: Tensor | None = None,
) -> str:
    """Hash exact caller layout for safe cache reuse before canonicalization."""
    header = {
        "version": HIERARCHY_VERSION,
        "igraph": _bliss_version(),
        "num_nodes": num_nodes,
        "config": asdict(config),
    }
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    for value in (edge_index, node_labels, edge_labels):
        if value is None:
            digest.update(b"null;")
            continue
        cpu = value.detach().cpu().contiguous()
        digest.update(json.dumps([str(cpu.dtype), list(cpu.shape)]).encode())
        digest.update(cpu.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _adjacency(num_nodes: int, edges: Tensor) -> list[list[int]]:
    result = [[] for _ in range(num_nodes)]
    for u, v in edges.T.tolist():
        result[u].append(v)
        result[v].append(u)
    for neighbors in result:
        neighbors.sort()
    return result


def _components(adjacency: list[list[int]]) -> list[list[int]]:
    unseen = set(range(len(adjacency)))
    result = []
    for start in range(len(adjacency)):
        if start not in unseen:
            continue
        unseen.remove(start)
        queue, members = deque([start]), []
        while queue:
            node = queue.popleft()
            members.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        result.append(sorted(members))
    return result


def _distances(adjacency: list[list[int]], source: int, allowed=None) -> list[float]:
    permitted = None if allowed is None else set(allowed)
    distance = [math.inf] * len(adjacency)
    distance[source] = 0
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if (permitted is None or neighbor in permitted) and distance[neighbor] == math.inf:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return distance


def _distance_matrix(adjacency: list[list[int]]) -> Tensor:
    rows = [_distances(adjacency, source) for source in range(len(adjacency))]
    if any(not math.isfinite(value) for row in rows for value in row):
        raise RuntimeError("component graph is disconnected")
    return torch.tensor(rows, dtype=torch.long)


def _region_metrics(adjacency, members):
    members = sorted(members)
    if not members:
        raise RuntimeError("empty regions are forbidden")
    best = None
    diameter = 0
    for candidate in members:
        distances = _distances(adjacency, candidate, members)
        values = [distances[node] for node in members]
        if any(not math.isfinite(value) for value in values):
            raise RuntimeError("hierarchy region is disconnected")
        eccentricity = int(max(values, default=0))
        diameter = max(diameter, eccentricity)
        score = (eccentricity, sum(values), candidate)
        if best is None or score < best[0]:
            best = (score, candidate)
    return best[1], best[0][0], diameter


def _multisource_partition(adjacency, centers):
    """Synchronous connected BFS with the frozen compactness tie-break."""
    owner = [-1] * len(adjacency)
    rows = [_distances(adjacency, center) for center in centers]
    nearest = [min(row[node] for row in rows) for node in range(len(adjacency))]
    for region, center in enumerate(centers):
        owner[center] = region
    for depth in range(1, int(max(nearest, default=0)) + 1):
        snapshot = _regions(owner, len(centers))
        base_radii = [
            _region_metrics(adjacency, members)[1] if members else 0
            for members in snapshot
        ]
        assignments = []
        for node in (value for value in range(len(adjacency)) if nearest[value] == depth):
            candidates = []
            for region, row in enumerate(rows):
                if row[node] != depth:
                    continue
                if not any(owner[neighbor] == region for neighbor in adjacency[node]):
                    continue
                _, new_radius, _ = _region_metrics(adjacency, snapshot[region] + [node])
                candidates.append((
                    depth,
                    new_radius - base_radii[region],
                    len(snapshot[region]),
                    centers[region],
                    region,
                ))
            if not candidates:
                raise RuntimeError("synchronous BFS could not preserve connected ownership")
            assignments.append((node, min(candidates)[-1]))
        for node, region in assignments:
            owner[node] = region
    if any(value < 0 for value in owner):
        raise RuntimeError("partition left nodes unassigned")
    return owner


def _regions(owner, count=None):
    count = max(owner) + 1 if count is None else count
    return [[node for node, region in enumerate(owner) if region == index] for index in range(count)]


def _graph_center(adjacency):
    return min(range(len(adjacency)), key=lambda node: (
        max(_distances(adjacency, node)), sum(_distances(adjacency, node)), node,
    ))


def _coverage_partition(adjacency, radius, soft_max_atoms):
    centers = [_graph_center(adjacency)]
    all_distances = [_distances(adjacency, centers[0])]
    while max(min(row[node] for row in all_distances) for node in range(len(adjacency))) > radius:
        nearest = [min(row[node] for row in all_distances) for node in range(len(adjacency))]
        center = max(range(len(adjacency)), key=lambda node: (nearest[node], len(adjacency[node]), -node))
        centers.append(center)
        all_distances.append(_distances(adjacency, center))
    owner = _multisource_partition(adjacency, centers)

    # The size guard is deliberately soft: compact stars/cliques are not torn
    # apart merely because their atom count exceeds the engineering threshold.
    while True:
        regions = _regions(owner)
        candidate = None
        for index, members in enumerate(regions):
            center, region_radius, diameter = _region_metrics(adjacency, members)
            if len(members) > soft_max_atoms and region_radius > 1:
                distances = _distances(adjacency, center, members)
                new_center = max(members, key=lambda node: (distances[node], -node))
                score = (len(members), diameter, -index)
                if candidate is None or score > candidate[0]:
                    candidate = (score, index, center, new_center, members)
        if candidate is None:
            break
        # Repair only the affected induced region; unrelated ownership remains
        # exactly unchanged.
        _, region, center, new_center, members = candidate
        to_local = {node: index for index, node in enumerate(members)}
        local_edges = [
            (to_local[node], to_local[neighbor])
            for node in members for neighbor in adjacency[node]
            if node < neighbor and neighbor in to_local
        ]
        local_adjacency = _adjacency(len(members), _pairs_tensor(local_edges))
        local_owner = _multisource_partition(
            local_adjacency, [to_local[center], to_local[new_center]],
        )
        new_region = max(owner) + 1
        for local_node, global_node in enumerate(members):
            owner[global_node] = region if local_owner[local_node] == 0 else new_region

    owner = _merge_singletons(adjacency, owner, max_radius=radius, soft_max_size=soft_max_atoms)
    return owner


def _merge_singletons(adjacency, owner, *, max_radius=None, soft_max_size=None):
    owner = list(owner)
    changed = True
    while changed:
        changed = False
        regions = _regions(owner)
        for region, members in enumerate(regions):
            if len(members) != 1:
                continue
            node = members[0]
            candidates = sorted({owner[neighbor] for neighbor in adjacency[node] if owner[neighbor] != region})
            best = None
            for target in candidates:
                merged = regions[target] + [node]
                _, radius, diameter = _region_metrics(adjacency, merged)
                if max_radius is not None and radius > max_radius:
                    continue
                if soft_max_size is not None and len(merged) > soft_max_size and diameter > 2:
                    continue
                old_diameter = _region_metrics(adjacency, regions[target])[2]
                score = (diameter - old_diameter, len(merged), target)
                if best is None or score < best[0]:
                    best = (score, target)
            if best is not None:
                owner[node] = best[1]
                changed = True
                break
    remap = {old: new for new, old in enumerate(sorted(set(owner)))}
    return [remap[value] for value in owner]


def _budget_partition(adjacency, target_children):
    count = len(adjacency)
    budget = max(1, math.ceil(count / target_children))
    centers = [_graph_center(adjacency)]
    distance_rows = [_distances(adjacency, centers[0])]
    while len(centers) < budget:
        nearest = [min(row[node] for row in distance_rows) for node in range(count)]
        center = max((node for node in range(count) if node not in centers), key=lambda node: (
            nearest[node], len(adjacency[node]), -node,
        ))
        centers.append(center)
        distance_rows.append(_distances(adjacency, center))
    owner = _multisource_partition(adjacency, centers)
    return _merge_singletons(adjacency, owner)


def _aggregate_edges(
    child_edges, child_counts, child_types, child_internal_counts,
    child_internal_types, owner, num_parents,
):
    type_dim = child_types.size(1)
    internal_counts = torch.zeros(num_parents, dtype=torch.long)
    internal_types = torch.zeros((num_parents, type_dim), dtype=torch.long)
    for child, parent in enumerate(owner):
        internal_counts[parent] += child_internal_counts[child]
        internal_types[parent] += child_internal_types[child]
    external: dict[tuple[int, int], tuple[int, Tensor]] = {}
    for edge_id, (u, v) in enumerate(child_edges.T.tolist()):
        left, right = owner[u], owner[v]
        if left == right:
            internal_counts[left] += child_counts[edge_id]
            internal_types[left] += child_types[edge_id]
            continue
        pair = (min(left, right), max(left, right))
        if pair not in external:
            external[pair] = (0, torch.zeros(type_dim, dtype=torch.long))
        count, types = external[pair]
        external[pair] = (count + int(child_counts[edge_id]), types + child_types[edge_id])
    pairs = sorted(external)
    counts = torch.tensor([external[pair][0] for pair in pairs], dtype=torch.long)
    types = torch.stack([external[pair][1] for pair in pairs]) if pairs else torch.empty((0, type_dim), dtype=torch.long)
    return _pairs_tensor(pairs), counts, types, internal_counts, internal_types


def _make_level(
    level_index, child_adjacency, child_edges, child_counts, child_types,
    child_internal_counts, child_internal_types, child_atom_counts,
    child_leaf_sets, owner, atom_members=None, l1_adjacency=None,
):
    members_local = _regions(owner)
    num_parents = len(members_local)
    centers, radii, diameters = [], [], []
    for members in members_local:
        center, radius, diameter = _region_metrics(child_adjacency, members)
        centers.append(center)
        radii.append(radius)
        diameters.append(diameter)
    edges, counts, types, internal_counts, internal_types = _aggregate_edges(
        child_edges, child_counts, child_types, child_internal_counts,
        child_internal_types, owner, num_parents,
    )
    atom_counts = torch.tensor([
        sum(int(child_atom_counts[child]) for child in members) for members in members_local
    ], dtype=torch.long)
    if level_index == 1:
        members = [torch.tensor(atom_members[index], dtype=torch.long) for index in range(num_parents)]
        # L1 nodes are the leaves of the hierarchy. Atom membership is stored
        # separately above and must never be confused with L1 leaf indices.
        leaf_sets = [torch.tensor([index], dtype=torch.long) for index in range(num_parents)]
        leaf_diameters = torch.zeros(num_parents, dtype=torch.long)
    else:
        leaf_sets = []
        for child_members in members_local:
            leaves = torch.cat([child_leaf_sets[child] for child in child_members]).sort().values
            leaf_sets.append(leaves)
        members = [torch.tensor(value, dtype=torch.long) for value in members_local]
        leaf_diameters = []
        for leaves in leaf_sets:
            _, _, diameter = _region_metrics(l1_adjacency, leaves.tolist())
            leaf_diameters.append(diameter)
        leaf_diameters = torch.tensor(leaf_diameters, dtype=torch.long)
    return HierarchyLevel(
        level_index, members, leaf_sets, torch.tensor(centers, dtype=torch.long),
        torch.tensor(owner, dtype=torch.long), edges, counts, types,
        internal_counts, internal_types, atom_counts,
        torch.tensor(radii, dtype=torch.long), torch.tensor(diameters, dtype=torch.long),
        leaf_diameters,
    )


def _build_component(global_atoms, global_edges, edge_type_ids, type_dim, config):
    atom_to_local = {atom: index for index, atom in enumerate(global_atoms)}
    selected_edge_ids = [
        edge_id for edge_id, (u, v) in enumerate(global_edges.T.tolist())
        if u in atom_to_local and v in atom_to_local
    ]
    local_pairs = [
        (atom_to_local[int(global_edges[0, edge_id])], atom_to_local[int(global_edges[1, edge_id])])
        for edge_id in selected_edge_ids
    ]
    atomic_edges = _pairs_tensor(local_pairs)
    atomic_adjacency = _adjacency(len(global_atoms), atomic_edges)
    atomic_types = torch.zeros((len(selected_edge_ids), type_dim), dtype=torch.long)
    if selected_edge_ids:
        atomic_types[torch.arange(len(selected_edge_ids)), edge_type_ids[selected_edge_ids]] = 1
    atomic_counts = torch.ones(len(selected_edge_ids), dtype=torch.long)
    atomic_internal_counts = torch.zeros(len(global_atoms), dtype=torch.long)
    atomic_internal_types = torch.zeros((len(global_atoms), type_dim), dtype=torch.long)
    atomic_atom_counts = torch.ones(len(global_atoms), dtype=torch.long)
    atomic_leaf_sets = [torch.tensor([index], dtype=torch.long) for index in range(len(global_atoms))]

    l1_owner = _coverage_partition(
        atomic_adjacency, config.l1_cover_radius, config.l1_soft_max_atoms,
    )
    l1_atom_members = [
        [global_atoms[node] for node in members] for members in _regions(l1_owner)
    ]
    level1 = _make_level(
        1, atomic_adjacency, atomic_edges, atomic_counts, atomic_types,
        atomic_internal_counts, atomic_internal_types, atomic_atom_counts,
        atomic_leaf_sets, l1_owner, atom_members=l1_atom_members,
    )
    levels = [level1]
    if bool((level1.child_radii > config.l1_cover_radius).any()):
        raise RuntimeError("L1 repair violated the configured radius bound")
    l1_adjacency = _adjacency(level1.num_tokens, level1.edges)
    l1_distances = _distance_matrix(l1_adjacency)

    while len(levels) < config.max_levels:
        previous = levels[-1]
        previous_adjacency = _adjacency(previous.num_tokens, previous.edges)
        if previous.num_tokens <= config.stop_num_tokens:
            break
        if int(_distance_matrix(previous_adjacency).max()) <= config.stop_diameter:
            break
        owner = _budget_partition(previous_adjacency, config.target_children)
        if max(owner) + 1 >= previous.num_tokens:
            break
        next_level = _make_level(
            len(levels) + 1, previous_adjacency, previous.edges,
            previous.edge_counts, previous.edge_type_counts,
            previous.internal_edge_counts, previous.internal_edge_type_counts,
            previous.atom_counts, previous.leaf_sets, owner,
            l1_adjacency=l1_adjacency,
        )
        levels.append(next_level)

    # Strong invariant: each recursive token is a connected induced L1 region.
    for level in levels[1:]:
        for leaves in level.leaf_sets:
            _region_metrics(l1_adjacency, leaves.tolist())
    return ComponentHierarchy(torch.tensor(global_atoms, dtype=torch.long), levels, l1_distances)


def build_hierarchical_topology(
    num_nodes: int,
    edge_index: Tensor,
    config: HierarchyConfig | None = None,
    *,
    node_labels: Tensor | None = None,
    edge_labels: Tensor | None = None,
) -> HierarchicalTopology:
    """Build independent connected-component L1/L2/L3 topology trees."""
    config = config or HierarchyConfig()
    if num_nodes < 1:
        raise ValueError("empty graphs are not supported")
    if node_labels is None or edge_labels is None:
        raise ValueError(
            "hierarchical molecular coarsening requires raw discrete node_labels and edge_labels"
        )
    input_fingerprint = hierarchy_input_fingerprint(
        num_nodes, edge_index, config,
        node_labels=node_labels, edge_labels=edge_labels,
    )
    edges, positions = _canonical_edges(num_nodes, edge_index)
    raw_nodes = discrete_rows(node_labels, num_nodes, "node_labels")
    raw_edge_rows = discrete_rows(edge_labels, edge_index.size(1), "edge_labels")
    unique_edge_rows = raw_edge_rows[positions]
    physical_ids = {tuple(pair): index for index, pair in enumerate(edges.T.tolist())}
    for position, (u, v) in enumerate(edge_index.detach().cpu().T.tolist()):
        expected = unique_edge_rows[physical_ids[(min(u, v), max(u, v))]]
        if not torch.equal(raw_edge_rows[position], expected):
            raise ValueError("opposite directions must have identical discrete edge_labels")

    order = canonical_atom_order(num_nodes, edges, raw_nodes, unique_edge_rows)
    inverse = torch.argsort(order)
    remapped = inverse[edges].sort(dim=0).values
    edge_order = torch.argsort(remapped[0] * num_nodes + remapped[1])
    edges = remapped[:, edge_order]
    unique_edge_rows = unique_edge_rows[edge_order]
    canonical_nodes = raw_nodes[order]

    type_keys = tuple(sorted(set(map(tuple, unique_edge_rows.tolist()))))
    if not type_keys:
        type_keys = ((),)
    type_map = {key: index for index, key in enumerate(type_keys)}
    edge_type_ids = torch.tensor([type_map[tuple(row)] for row in unique_edge_rows.tolist()], dtype=torch.long)
    adjacency = _adjacency(num_nodes, edges)
    component_atoms = _components(adjacency)
    components = [
        _build_component(atoms, edges, edge_type_ids, len(type_keys), config)
        for atoms in component_atoms
    ]

    fingerprint_payload = {
        "version": HIERARCHY_VERSION,
        "config": asdict(config),
        "nodes": canonical_nodes.tolist(),
        "edges": edges.T.tolist(),
        "edge_labels": unique_edge_rows.tolist(),
    }
    fingerprint = hashlib.sha256(json.dumps(fingerprint_payload, sort_keys=True).encode()).hexdigest()
    return HierarchicalTopology(
        components, order, inverse, canonical_nodes, edges, unique_edge_rows, type_keys,
        input_fingerprint, fingerprint, config,
    )
