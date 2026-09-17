"""Discrete CPU coarsening of undirected simple graphs (no learned gates)."""
from __future__ import annotations

import math
import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from functools import lru_cache
from importlib.metadata import version

import torch
from torch import Tensor

from .config import CoarseningConfig
from .canonical import canonical_atom_order, discrete_rows


# Bump when topology construction, serialization or index semantics change.
TOPOLOGY_VERSION = 1


@lru_cache(maxsize=1)
def _bliss_version() -> str:
    return version("igraph")


def topology_fingerprint(num_nodes, edge_index, config, *, node_ids=None,
                         node_labels=None, edge_labels=None) -> str:
    """Hash the exact input layout, NOT a canonical/isomorphism hash.

    O(N+E) verification avoids canonicalization on hits. Reordered inputs get
    their own entries because atom_order and edge_positions are input-specific.
    Learned embeddings and continuous message attributes are deliberately absent.
    """
    header = dict(version=TOPOLOGY_VERSION, config=asdict(config), num_nodes=num_nodes,
                  igraph=_bliss_version() if config.canonicalize else None)
    digest = hashlib.sha256(json.dumps(header, sort_keys=True).encode())
    for value in (edge_index, node_ids, node_labels, edge_labels):
        if value is None:
            digest.update(b"null;")
        else:
            value = value.detach().cpu().contiguous()
            digest.update(json.dumps([str(value.dtype), list(value.shape)]).encode())
            digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass
class CoarseTopology:
    # All structural tensors live on CPU. Node indices use canonical atom order
    # by default; use atom_order/owner_input to map back to the original input.
    centers: Tensor  # Only main centers; residual coarse nodes have no center.
    owner: Tensor  # [N], includes both main and residual regions.
    cores: list[Tensor]
    contexts: list[Tensor]
    is_residual: Tensor
    edges: Tensor  # [2, E], unique undirected input edges, u < v.
    edge_positions: Tensor  # Positions of these edges in caller's edge_index.
    context_edges: list[Tensor]  # Local indices within each context.
    context_edge_ids: list[Tensor]  # Indices into edges / canonical attributes.
    core_positions: list[Tensor]  # Core node positions within each context.
    coarse_edges: Tensor  # [2, Ec], each undirected coarse pair once.
    boundary_edge_ids: Tensor  # Input edges crossing different owners.
    boundary_groups: Tensor  # Coarse edge index for each boundary edge.
    edge_counts: Tensor  # Physical edges, not directed storage entries.
    stats: dict[str, float | int]
    atom_order: Tensor  # canonical -> input
    input_to_canonical: Tensor
    node_labels: Tensor  # Original discrete values in canonical order.
    edge_labels: Tensor  # Original discrete values aligned with edges.
    input_edge_ids: Tensor  # Caller edge positions -> canonical physical edge IDs.
    input_fingerprint: str  # Exact input layout + coarsening config + version.

    @property
    def owner_input(self) -> Tensor:
        return self.owner[self.input_to_canonical]

    @property
    def centers_input(self) -> Tensor:
        return self.atom_order[self.centers]

    def canonical_signature(self) -> str:
        """Fingerprint attributed input and its COMPLETE ordered coarsening.

        Includes assignments, contexts, coarse edges, counts and residual flags,
        so equal node/edge counts alone cannot pass the invariance audit.
        No original IDs, edge storage order, or floating embeddings are hashed.
        """
        payload = {
            "nodes": self.node_labels.tolist(), "edges": self.edges.T.tolist(),
            "edge_labels": self.edge_labels.tolist(), "centers": self.centers.tolist(),
            "owner": self.owner.tolist(), "contexts": [x.tolist() for x in self.contexts],
            "is_residual": self.is_residual.tolist(),
            "coarse_edges": self.coarse_edges.T.tolist(), "counts": self.edge_counts.tolist(),
        }
        return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode()).hexdigest()


def _pairs_tensor(pairs: list[tuple[int, int]]) -> Tensor:
    return torch.tensor(pairs, dtype=torch.long).reshape(-1, 2).T.contiguous()


def _canonical_edges(n: int, edge_index: Tensor) -> tuple[Tensor, Tensor]:
    if edge_index.dtype != torch.long or edge_index.ndim != 2 or edge_index.size(0) != 2:
        raise ValueError("edge_index must be a long tensor of shape [2, E]")
    representatives: dict[tuple[int, int], int] = {}
    directions: set[tuple[int, int]] = set()
    for position, (u, v) in enumerate(edge_index.detach().cpu().T.tolist()):
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError("edge_index contains an out-of-range node")
        if u == v:
            raise ValueError("Input self-loops are not supported; the GNN adds self updates")
        if (u, v) in directions:
            raise ValueError("Duplicate directed edges / parallel edges are not supported")
        directions.add((u, v))
        representatives.setdefault((min(u, v), max(u, v)), position)
    pairs = sorted(representatives)
    return _pairs_tensor(pairs), torch.tensor([representatives[p] for p in pairs], dtype=torch.long)


def _distances(adjacency: list[list[int]], source: int) -> list[float]:
    distance = [math.inf] * len(adjacency)
    distance[source] = 0
    queue = deque([source])
    while queue:
        u = queue.popleft()
        for v in adjacency[u]:
            if distance[v] == math.inf:
                distance[v] = distance[u] + 1
                queue.append(v)
    return distance


def _components(adjacency: list[list[int]], nodes: set[int], keys: list[int]) -> list[list[int]]:
    unseen = set(nodes)
    components = []
    for start in sorted(nodes, key=keys.__getitem__):
        if start not in unseen:
            continue
        unseen.remove(start)
        queue, component = deque([start]), []
        while queue:
            u = queue.popleft()
            component.append(u)
            for v in adjacency[u]:
                if v in unseen:
                    unseen.remove(v)
                    queue.append(v)
        components.append(sorted(component))
    return components


def build_topology(
    num_nodes: int,
    edge_index: Tensor,
    config: CoarseningConfig | None = None,
    *,
    node_ids: Tensor | None = None,
    node_labels: Tensor | None = None,
    edge_labels: Tensor | None = None,
) -> CoarseTopology:
    """Build all assignments before encoding.

    Farthest-point sampling maximizes distance to the nearest existing center.
    Default: canonicalize the discrete attributed graph first, then break ties
    using canonical indices. Auxiliary edge vertices never enter coarsening.
    With canonicalize=False, the legacy input-ID-dependent behavior is retained
    only for comparisons. Persistent node_ids are accepted only in that mode.
    """
    config = config or CoarseningConfig()
    if num_nodes < 1:
        raise ValueError("Empty graphs are not supported")
    if config.canonicalize and node_ids is not None:
        raise ValueError("node_ids cannot override canonical labeling; use canonicalize=False for legacy tests")
    if node_ids is None:
        keys = list(range(num_nodes))
    else:
        if node_ids.dtype != torch.long or node_ids.shape != (num_nodes,):
            raise ValueError("node_ids must be a long tensor of shape [N]")
        keys = node_ids.detach().cpu().tolist()
        if len(set(keys)) != num_nodes:
            raise ValueError("node_ids must be unique within a graph")
    edges, positions = _canonical_edges(num_nodes, edge_index)
    raw_nodes = discrete_rows(node_labels, num_nodes, "node_labels")
    raw_edges = discrete_rows(edge_labels, edge_index.size(1), "edge_labels")
    unique_labels = raw_edges[positions]
    physical_ids = {tuple(pair): i for i, pair in enumerate(edges.T.tolist())}
    for i, (u, v) in enumerate(edge_index.detach().cpu().T.tolist()):
        if not torch.equal(raw_edges[i], unique_labels[physical_ids[(min(u, v), max(u, v))]]):
            raise ValueError("Opposite directions must have identical discrete edge_labels")
    order = canonical_atom_order(num_nodes, edges, raw_nodes, unique_labels) if config.canonicalize else torch.arange(num_nodes)
    inverse = torch.argsort(order)
    if config.canonicalize:
        remapped = inverse[edges].sort(dim=0).values
        edge_order = torch.argsort(remapped[0] * num_nodes + remapped[1])
        edges = remapped[:, edge_order]
        positions = positions[edge_order]
        unique_labels = unique_labels[edge_order]
        keys = list(range(num_nodes))
    raw_nodes = raw_nodes[order]
    canonical_ids = {tuple(pair): i for i, pair in enumerate(edges.T.tolist())}
    input_edge_ids = torch.tensor([
        canonical_ids[(min(u, v), max(u, v))]
        for u, v in inverse[edge_index.detach().cpu()].T.tolist()
    ], dtype=torch.long)
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    pairs = edges.T.tolist()
    for u, v in pairs:
        adjacency[u].append(v)
        adjacency[v].append(u)

    centers: list[int] = []
    contexts: list[Tensor] = []
    nearest = [math.inf] * num_nodes
    owner = [-1] * num_nodes

    def add_center(candidates):
        center = max(candidates, key=lambda v: (nearest[v], len(adjacency[v]), -keys[v]))
        region_id = len(centers)
        centers.append(center)
        distance = _distances(adjacency, center)
        contexts.append(torch.tensor([v for v, d in enumerate(distance) if d <= config.radius], dtype=torch.long))
        for v, d in enumerate(distance):
            # Equal distances retain the earlier center, consistently along BFS paths.
            if d < nearest[v]:
                nearest[v] = d
                owner[v] = region_id

    budget = max(1, math.ceil(num_nodes * config.center_fraction))
    for _ in range(budget):
        candidates = [v for v in range(num_nodes) if nearest[v] > config.radius]
        if not candidates:
            break
        add_center(candidates)
    # Supplement large holes; each iteration covers at least the new center.
    while True:
        uncovered = {v for v in range(num_nodes) if nearest[v] > config.radius}
        residuals = _components(adjacency, uncovered, keys)
        large = [v for component in residuals if len(component) > config.max_residual_size for v in component]
        if not large:
            break
        add_center(large)

    # Nearest-center assignments outside the radius are replaced by residual owners.
    for component in residuals:
        region_id = len(contexts)
        contexts.append(torch.tensor(component, dtype=torch.long))
        for v in component:
            owner[v] = region_id
    owner_tensor = torch.tensor(owner, dtype=torch.long)
    cores = [torch.where(owner_tensor == k)[0] for k in range(len(contexts))]
    context_edges, context_edge_ids, core_positions = [], [], []
    for core, context in zip(cores, contexts):
        local = {v: i for i, v in enumerate(context.tolist())}
        core_positions.append(torch.tensor([local[v] for v in core.tolist()], dtype=torch.long))
        ids = [i for i, (u, v) in enumerate(pairs) if u in local and v in local]
        context_edge_ids.append(torch.tensor(ids, dtype=torch.long))
        context_edges.append(_pairs_tensor([(local[pairs[i][0]], local[pairs[i][1]]) for i in ids]))

    boundary_ids, boundary_pairs = [], []
    for edge_id, (u, v) in enumerate(pairs):
        a, b = owner[u], owner[v]
        if a != b:
            boundary_ids.append(edge_id)
            boundary_pairs.append((min(a, b), max(a, b)))
    coarse_pairs = sorted(set(boundary_pairs))
    pair_to_id = {pair: i for i, pair in enumerate(coarse_pairs)}
    groups = torch.tensor([pair_to_id[p] for p in boundary_pairs], dtype=torch.long)
    counts = torch.bincount(groups, minlength=len(coarse_pairs))
    num_main = len(centers)
    num_coarse = len(contexts)
    stats = {
        "num_nodes": num_nodes,
        "num_edges": edges.size(1),
        "num_main_regions": num_main,
        "num_added_centers": max(0, num_main - budget),
        "num_residual_regions": len(residuals),
        "num_residual_nodes": sum(map(len, residuals)),
        "num_coarse_nodes": num_coarse,
        "num_coarse_edges": len(coarse_pairs),
        "coarse_node_ratio": num_coarse / num_nodes,
        "context_occurrence_ratio": sum(c.numel() for c in contexts) / num_nodes,
        "max_core_size": max(c.numel() for c in cores),
        "canonicalized": config.canonicalize,
    }
    return CoarseTopology(
        torch.tensor(centers, dtype=torch.long), owner_tensor, cores, contexts,
        torch.arange(num_coarse) >= num_main, edges, positions,
        context_edges, context_edge_ids, core_positions, _pairs_tensor(coarse_pairs),
        torch.tensor(boundary_ids, dtype=torch.long), groups, counts, stats,
        order, inverse, raw_nodes, unique_labels, input_edge_ids,
        topology_fingerprint(num_nodes, edge_index, config, node_ids=node_ids,
                             node_labels=node_labels, edge_labels=edge_labels),
    )


def _v2_multisource_partition(adjacency, centers, keys):
    """Connected graph Voronoi cells with deterministic whole-layer expansion."""
    import heapq

    owner = [-1] * len(adjacency)
    distance = [math.inf] * len(adjacency)
    queue = []
    for region, center in enumerate(centers):
        owner[center] = region
        distance[center] = 0
        heapq.heappush(queue, (0, keys[center], region, center))
    while queue:
        depth, _, region, node = heapq.heappop(queue)
        if depth != distance[node] or owner[node] != region:
            continue
        for neighbor in sorted(adjacency[node], key=keys.__getitem__):
            candidate = depth + 1
            choice = (candidate, keys[centers[region]])
            current = (distance[neighbor], keys[centers[owner[neighbor]]]) if owner[neighbor] >= 0 else (math.inf, math.inf)
            if choice < current:
                distance[neighbor] = candidate
                owner[neighbor] = region
                heapq.heappush(queue, (candidate, keys[neighbor], region, neighbor))
    if any(value < 0 for value in owner):
        raise RuntimeError("Every connected component must receive a V2 center")
    return owner


def _v2_induced_distances(adjacency, source, members):
    allowed = set(members)
    distance = {source: 0}
    queue = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    if len(distance) != len(allowed):
        raise RuntimeError("V2 produced a disconnected main region")
    return distance


def _v2_partition_objective(adjacency, pairs, centers, owner, keys):
    regions = [[v for v, value in enumerate(owner) if value == k] for k in range(len(centers))]
    radii = []
    for center, members in zip(centers, regions):
        distances = _v2_induced_distances(adjacency, center, members)
        radii.append(max(distances.values(), default=0))
    cut = sum(owner[u] != owner[v] for u, v in pairs)
    sizes = [len(members) for members in regions]
    # For fixed K, sum of squared sizes is monotonic with CV and avoids floats.
    imbalance = sum(size * size for size in sizes)
    canonical = tuple(sorted(keys[center] for center in centers))
    return (max(radii, default=0), sum(radii), cut, imbalance, canonical), radii, regions


def _v2_recenter(adjacency, pairs, centers, owner, keys, rounds):
    accepted = 0
    objective, _, regions = _v2_partition_objective(adjacency, pairs, centers, owner, keys)
    for _ in range(rounds):
        candidates = []
        for current, members in zip(centers, regions):
            best = None
            for candidate in members:
                distances = _v2_induced_distances(adjacency, candidate, members)
                value = (max(distances.values(), default=0), sum(distances.values()), keys[candidate])
                if best is None or value < best[0]:
                    best = (value, candidate)
            candidates.append(current if best is None else best[1])
        candidate_owner = _v2_multisource_partition(adjacency, candidates, keys)
        candidate_objective, _, candidate_regions = _v2_partition_objective(
            adjacency, pairs, candidates, candidate_owner, keys,
        )
        if candidate_objective >= objective:
            break
        centers, owner = candidates, candidate_owner
        objective, regions = candidate_objective, candidate_regions
        accepted += 1
    return centers, owner, accepted


def _v2_component_budgets(components, total_budget):
    """Largest-remainder allocation after assigning one center per component."""
    remaining = total_budget - len(components)
    budgets = [1] * len(components)
    if remaining <= 0:
        return budgets
    total_nodes = sum(map(len, components))
    quotas = [remaining * len(component) / total_nodes for component in components]
    floors = [math.floor(value) for value in quotas]
    budgets = [base + extra for base, extra in zip(budgets, floors)]
    leftover = remaining - sum(floors)
    order = sorted(range(len(components)), key=lambda i: (-(quotas[i] - floors[i]), components[i][0]))
    for index in order[:leftover]:
        budgets[index] += 1
    return budgets


def build_v2_topology(
    num_nodes: int,
    edge_index: Tensor,
    config,
    *,
    node_ids: Tensor | None = None,
    node_labels: Tensor | None = None,
    edge_labels: Tensor | None = None,
) -> CoarseTopology:
    """Build the frozen V2 exclusive, connected and radius-bounded partition."""
    if num_nodes < 1:
        raise ValueError("Empty graphs are not supported")
    if node_ids is not None:
        raise ValueError("V2 does not accept persistent node IDs")
    if not config.canonicalize:
        raise ValueError("V2 requires canonicalization for deterministic tie breaking")

    edges, positions = _canonical_edges(num_nodes, edge_index)
    raw_nodes = discrete_rows(node_labels, num_nodes, "node_labels")
    raw_edges = discrete_rows(edge_labels, edge_index.size(1), "edge_labels")
    unique_labels = raw_edges[positions]
    physical_ids = {tuple(pair): i for i, pair in enumerate(edges.T.tolist())}
    for i, (u, v) in enumerate(edge_index.detach().cpu().T.tolist()):
        if not torch.equal(raw_edges[i], unique_labels[physical_ids[(min(u, v), max(u, v))]]):
            raise ValueError("Opposite directions must have identical discrete edge_labels")
    order = canonical_atom_order(num_nodes, edges, raw_nodes, unique_labels)
    inverse = torch.argsort(order)
    remapped = inverse[edges].sort(dim=0).values
    edge_order = torch.argsort(remapped[0] * num_nodes + remapped[1])
    edges = remapped[:, edge_order]
    positions = positions[edge_order]
    unique_labels = unique_labels[edge_order]
    raw_nodes = raw_nodes[order]
    keys = list(range(num_nodes))
    canonical_ids = {tuple(pair): i for i, pair in enumerate(edges.T.tolist())}
    input_edge_ids = torch.tensor([
        canonical_ids[(min(u, v), max(u, v))]
        for u, v in inverse[edge_index.detach().cpu()].T.tolist()
    ], dtype=torch.long)
    adjacency = [[] for _ in range(num_nodes)]
    pairs = edges.T.tolist()
    for u, v in pairs:
        adjacency[u].append(v)
        adjacency[v].append(u)

    components = _components(adjacency, set(range(num_nodes)), keys)
    initial_budget = max(len(components), math.ceil(num_nodes / config.target_region_size))
    budgets = _v2_component_budgets(components, initial_budget)
    centers = []
    for component, budget in zip(components, budgets):
        # First center: graph center (minimum eccentricity), then canonical tie break.
        first = min(component, key=lambda candidate: (
            max(_distances(adjacency, candidate)[node] for node in component), keys[candidate],
        ))
        selected = [first]
        while len(selected) < min(budget, len(component)):
            candidate = max(
                (node for node in component if node not in selected),
                key=lambda node: (min(_distances(adjacency, center)[node] for center in selected), -keys[node]),
            )
            selected.append(candidate)
        centers.extend(selected)

    owner = _v2_multisource_partition(adjacency, centers, keys)
    centers, owner, recenter_accepts = _v2_recenter(
        adjacency, pairs, centers, owner, keys, config.recenter_rounds,
    )
    additions = 0
    while True:
        _, radii, regions = _v2_partition_objective(adjacency, pairs, centers, owner, keys)
        worst = max(range(len(radii)), key=lambda i: (radii[i], -keys[centers[i]]))
        if radii[worst] <= config.max_region_radius:
            break
        distances = _v2_induced_distances(adjacency, centers[worst], regions[worst])
        new_center = max(regions[worst], key=lambda node: (distances[node], -keys[node]))
        if new_center in centers:
            raise RuntimeError("V2 radius repair failed to add a distinct center")
        centers.append(new_center)
        additions += 1
        owner = _v2_multisource_partition(adjacency, centers, keys)
        centers, owner, accepted = _v2_recenter(
            adjacency, pairs, centers, owner, keys, config.recenter_rounds,
        )
        recenter_accepts += accepted
        if len(centers) > num_nodes:
            raise RuntimeError("V2 center repair did not terminate")

    owner_tensor = torch.tensor(owner, dtype=torch.long)
    cores = [torch.where(owner_tensor == k)[0] for k in range(len(centers))]
    contexts = [core.clone() for core in cores]
    context_edges, context_edge_ids, core_positions = [], [], []
    for core in cores:
        local = {v: i for i, v in enumerate(core.tolist())}
        ids = [i for i, (u, v) in enumerate(pairs) if u in local and v in local]
        context_edge_ids.append(torch.tensor(ids, dtype=torch.long))
        context_edges.append(_pairs_tensor([(local[pairs[i][0]], local[pairs[i][1]]) for i in ids]))
        core_positions.append(torch.arange(len(core), dtype=torch.long))

    boundary_ids, boundary_pairs = [], []
    for edge_id, (u, v) in enumerate(pairs):
        a, b = owner[u], owner[v]
        if a != b:
            boundary_ids.append(edge_id)
            boundary_pairs.append((min(a, b), max(a, b)))
    coarse_pairs = sorted(set(boundary_pairs))
    pair_to_id = {pair: i for i, pair in enumerate(coarse_pairs)}
    groups = torch.tensor([pair_to_id[pair] for pair in boundary_pairs], dtype=torch.long)
    counts = torch.bincount(groups, minlength=len(coarse_pairs))
    _, final_radii, _ = _v2_partition_objective(adjacency, pairs, centers, owner, keys)
    stats = {
        "num_nodes": num_nodes,
        "num_edges": edges.size(1),
        "num_components": len(components),
        "initial_center_budget": initial_budget,
        "num_main_regions": len(centers),
        "num_added_centers": additions,
        "num_residual_regions": 0,
        "num_residual_nodes": 0,
        "num_coarse_nodes": len(centers),
        "num_coarse_edges": len(coarse_pairs),
        "coarse_node_ratio": len(centers) / num_nodes,
        "context_occurrence_ratio": 1.0,
        "max_core_size": max(map(len, cores)),
        "max_region_radius": max(final_radii, default=0),
        "cut_edge_count": len(boundary_ids),
        "cut_edge_ratio": len(boundary_ids) / max(1, len(pairs)),
        "recenter_accepts": recenter_accepts,
        "canonicalized": True,
        "algorithm": config.algorithm,
    }
    fingerprint = topology_fingerprint(
        num_nodes, edge_index, config, node_ids=node_ids,
        node_labels=node_labels, edge_labels=edge_labels,
    )
    return CoarseTopology(
        torch.tensor(centers, dtype=torch.long), owner_tensor, cores, contexts,
        torch.zeros(len(centers), dtype=torch.bool), edges, positions,
        context_edges, context_edge_ids, core_positions, _pairs_tensor(coarse_pairs),
        torch.tensor(boundary_ids, dtype=torch.long), groups, counts, stats,
        order, inverse, raw_nodes, unique_labels, input_edge_ids, fingerprint,
    )
