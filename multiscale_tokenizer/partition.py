"""Stochastic multiscale tokenization for unweighted molecular graphs."""
from __future__ import annotations

import hashlib
import random
import struct
from collections import deque
from dataclasses import dataclass
from typing import Mapping, Sequence

import igraph as ig
import torch
from torch import Tensor


_INF = 10**9


@dataclass(frozen=True)
class TokenizationConfig:
    """Fixed radii from the first-version method definition."""

    center_radius: int = 3
    scale_radii: tuple[int, ...] = (1, 2, 3)

    def __post_init__(self) -> None:
        if self.center_radius < 0:
            raise ValueError("center_radius must be nonnegative")
        if len(self.scale_radii) != 3:
            raise ValueError("scale_radii must contain exactly three outer scales")
        if any(not isinstance(radius, int) or radius < 0 for radius in self.scale_radii):
            raise ValueError("scale_radii must be nonnegative integers")
        if tuple(self.scale_radii) != (1, 2, 3):
            raise ValueError("the first-version method fixes scale_radii=(1, 2, 3)")


@dataclass
class TokenPartition:
    """One sampled partition P(G; omega)."""

    members: tuple[Tensor, ...]
    centers: Tensor
    levels: Tensor
    residual: Tensor
    rounds: Tensor
    owner: Tensor
    q_mask: Tensor
    stats: Mapping[str, float | int]

    def level_tokens(self, level: int) -> tuple[int, ...]:
        indices = torch.where(self.levels == level)[0].tolist()
        return tuple(indices)

    def validate(self, num_nodes: int) -> None:
        if self.owner.shape != (num_nodes,):
            raise ValueError("owner has the wrong shape")
        if self.q_mask.shape != (num_nodes,):
            raise ValueError("q_mask has the wrong shape")
        if self.q_mask.any() and self.owner[self.q_mask].ne(-1).any():
            raise ValueError("center-window nodes must not have a region owner")
        assigned = ~self.q_mask
        if num_nodes and assigned.any() and self.owner[assigned].lt(0).any():
            raise ValueError("every non-window node must have a region owner")
        seen = torch.zeros(num_nodes, dtype=torch.bool)
        for members in self.members:
            if members.ndim != 1 or members.numel() == 0:
                raise ValueError("each token must contain at least one node")
            if seen[members].any():
                raise ValueError("token memberships overlap")
            seen[members] = True
        if not torch.equal(seen, assigned):
            raise ValueError("tokens must cover exactly the non-window nodes")


def derive_partition_seed(base_seed: int, sample_id: int, epoch: int = 0) -> int:
    """Derive a stable seed from an independent partition RNG stream."""

    if base_seed < 0 or sample_id < 0 or epoch < 0:
        raise ValueError("partition seed inputs must be nonnegative")
    payload = struct.pack(
        "<QQQ",
        int(base_seed) & ((1 << 64) - 1),
        int(sample_id) & ((1 << 64) - 1),
        int(epoch) & ((1 << 64) - 1),
    )
    digest = hashlib.blake2b(payload, digest_size=8, person=b"mtoken01").digest()
    return int.from_bytes(digest, "little", signed=False)


def adjacency_from_bonds(bonds: Tensor, node_mask: Tensor | None = None) -> list[list[int]]:
    """Convert a dense OGB bond matrix to a validated adjacency list."""

    if bonds.ndim != 2 or bonds.size(0) != bonds.size(1):
        raise ValueError("bonds must be a square matrix")
    num_nodes = bonds.size(0)
    if node_mask is None:
        node_mask = torch.ones(num_nodes, dtype=torch.bool, device=bonds.device)
    if node_mask.shape != (num_nodes,) or node_mask.dtype != torch.bool:
        raise ValueError("node_mask must be boolean with shape [num_nodes]")
    if not torch.equal(bonds, bonds.transpose(0, 1)):
        raise ValueError("bonds must be symmetric")
    active = bonds.detach().cpu()
    mask = node_mask.detach().cpu()
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for left in range(num_nodes):
        if not bool(mask[left]):
            continue
        if int(active[left, left]) != 0:
            raise ValueError("molecular partition does not accept self-loops")
        for right in range(left + 1, num_nodes):
            if not bool(mask[right]):
                continue
            if 0 < int(active[left, right]) < 5:
                adjacency[left].append(right)
                adjacency[right].append(left)
    return adjacency


def _validate_adjacency(adjacency: Sequence[Sequence[int]], num_nodes: int) -> None:
    if len(adjacency) != num_nodes:
        raise ValueError("adjacency length does not match the node count")
    for node, neighbors in enumerate(adjacency):
        if len(set(neighbors)) != len(neighbors):
            raise ValueError("adjacency contains duplicate neighbors")
        for neighbor in neighbors:
            if not isinstance(neighbor, int):
                raise TypeError("adjacency indices must be integers")
            if not 0 <= neighbor < num_nodes:
                raise ValueError("adjacency index is out of range")
            if neighbor == node:
                raise ValueError("adjacency contains a self-loop")
            if node not in adjacency[neighbor]:
                raise ValueError("adjacency must be symmetric")


def _induced_edges(
    adjacency: Sequence[Sequence[int]], nodes: set[int],
) -> tuple[list[int], list[tuple[int, int]]]:
    order = sorted(nodes)
    local = {node: index for index, node in enumerate(order)}
    edges: list[tuple[int, int]] = []
    for left, node in enumerate(order):
        for neighbor in adjacency[node]:
            right = local.get(neighbor)
            if right is not None and left < right:
                edges.append((left, right))
    return order, edges


def _canonical_ranks(
    adjacency: Sequence[Sequence[int]], nodes: set[int],
) -> dict[int, int]:
    order, edges = _induced_edges(adjacency, nodes)
    if not order:
        return {}
    graph = ig.Graph(n=len(order), edges=edges, directed=False)
    permutation = graph.canonical_permutation()
    return {order[old]: rank for rank, old in enumerate(permutation)}


def _canonical_signature(
    adjacency: Sequence[Sequence[int]], nodes: set[int],
) -> str:
    ranks = _canonical_ranks(adjacency, nodes)
    size = len(ranks)
    if size == 0:
        return hashlib.sha256(b"").hexdigest()
    bits = bytearray(size * (size - 1) // 2)
    edge_count = 0
    for node in nodes:
        for neighbor in adjacency[node]:
            if neighbor not in nodes:
                continue
            left, right = ranks[node], ranks[neighbor]
            if left > right:
                continue
            edge_count += 1
            bit = left * (2 * size - left - 1) // 2 + (right - left - 1)
            bits[bit] = 1
    payload = struct.pack("<II", size, edge_count) + bytes(bits)
    return hashlib.sha256(payload).hexdigest()


def _derive_seed(seed: int, salt: str) -> int:
    payload = struct.pack("<Q", seed & ((1 << 64) - 1))
    digest = hashlib.blake2b(
        payload + salt.encode("utf-8"),
        digest_size=8,
        person=b"mtokense",
    ).digest()
    return int.from_bytes(digest, "little")


def _choose_canonically(
    candidates: Sequence[int],
    ranks: Mapping[int, int],
    seed: int,
    salt: str,
) -> int:
    if not candidates:
        raise ValueError("cannot choose from an empty candidate set")

    def priority(node: int) -> tuple[int, int]:
        score = _derive_seed(seed, f"{salt}:{ranks[node]}")
        return score, ranks[node]

    return min(candidates, key=priority)


def _restricted_components(
    adjacency: Sequence[Sequence[int]], allowed: set[int],
) -> list[list[int]]:
    unseen = set(allowed)
    components: list[list[int]] = []
    while unseen:
        start = min(unseen)
        unseen.remove(start)
        queue = deque([start])
        component: list[int] = []
        while queue:
            node = queue.popleft()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    queue.append(neighbor)
        components.append(sorted(component))
    return components


def _ordered_components(
    adjacency: Sequence[Sequence[int]], nodes: set[int],
) -> list[list[int]]:
    components = _restricted_components(adjacency, nodes)
    return sorted(
        components,
        key=lambda component: (
            _canonical_signature(adjacency, set(component)),
            len(component),
        ),
    )


def _distances_from(
    adjacency: Sequence[Sequence[int]], sources: set[int], num_nodes: int,
) -> list[int]:
    distances = [_INF] * num_nodes
    queue: deque[int] = deque()
    for source in sorted(sources):
        distances[source] = 0
        queue.append(source)
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            candidate = distances[node] + 1
            if candidate < distances[neighbor]:
                distances[neighbor] = candidate
                queue.append(neighbor)
    return distances


def _distances_from_node(
    adjacency: Sequence[Sequence[int]], source: int, num_nodes: int,
) -> list[int]:
    return _distances_from(adjacency, {source}, num_nodes)


def _induced_ball(
    adjacency: Sequence[Sequence[int]],
    allowed: set[int],
    center: int,
    radius: int,
) -> list[int]:
    distance = {center: 0}
    queue = deque([center])
    while queue:
        node = queue.popleft()
        if distance[node] == radius:
            continue
        for neighbor in adjacency[node]:
            if neighbor in allowed and neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    return sorted(distance)


def _diameter(adjacency: Sequence[Sequence[int]], nodes: set[int]) -> int:
    diameter = 0
    for node in nodes:
        distances = _distances_from_node(adjacency, node, len(adjacency))
        diameter = max(diameter, max(distances[other] for other in nodes))
    return diameter


class _Emitter:
    def __init__(self, num_nodes: int):
        self.owner = torch.full((num_nodes,), -1, dtype=torch.long)
        self.members: list[Tensor] = []
        self.centers: list[int] = []
        self.levels: list[int] = []
        self.residual: list[bool] = []
        self.rounds: list[int] = []

    def emit(
        self,
        members: Sequence[int],
        *,
        center: int,
        level: int,
        residual: bool,
        round_index: int,
    ) -> None:
        if not members:
            raise ValueError("cannot emit an empty token")
        region = len(self.members)
        member_tensor = torch.tensor(sorted(members), dtype=torch.long)
        if self.owner[member_tensor].ne(-1).any():
            raise RuntimeError("attempted to emit overlapping token members")
        self.owner[member_tensor] = region
        self.members.append(member_tensor)
        self.centers.append(center)
        self.levels.append(level)
        self.residual.append(residual)
        self.rounds.append(round_index)


def _cover_one_round(
    adjacency: Sequence[Sequence[int]],
    component: set[int],
    internal: set[int],
    *,
    level: int,
    radius: int,
    center_distance: int,
    round_index: int,
    component_seed: int,
    emitter: _Emitter,
) -> set[int]:
    """Cover the current outside boundary without changing the fixed internal set."""

    fixed_internal = set(internal)
    remaining = set(component) - fixed_internal
    if not remaining:
        return set()
    distances = _distances_from(adjacency, fixed_internal, len(adjacency))
    emitted: set[int] = set()

    while remaining:
        boundary = {
            node for node in remaining
            if any(neighbor in fixed_internal for neighbor in adjacency[node])
        }
        if not boundary:
            break

        components = _ordered_components(adjacency, remaining)
        adjacent = [
            members for members in components
            if any(neighbor in fixed_internal for node in members for neighbor in adjacency[node])
        ]
        shallow = [
            members for members in adjacent
            if max(distances[node] for node in members) < center_distance
        ]
        if shallow:
            members = min(
                shallow,
                key=lambda component: (
                    _canonical_signature(adjacency, set(component)),
                    len(component),
                ),
            )
            emitter.emit(
                members,
                center=-1,
                level=level,
                residual=True,
                round_index=round_index,
            )
            emitted.update(members)
            remaining.difference_update(members)
            continue

        candidates = [
            node for members in adjacent for node in members
            if distances[node] == center_distance
        ]
        if not candidates:
            raise RuntimeError("a deep component has no legal standard center")
        ranks = _canonical_ranks(adjacency, remaining)
        center = _choose_canonically(
            candidates,
            ranks,
            component_seed,
            f"level{level}:round{round_index}:standard",
        )
        region = _induced_ball(adjacency, remaining, center, radius)
        if center not in region:
            raise RuntimeError("induced ball omitted its center")
        emitter.emit(
            region,
            center=center,
            level=level,
            residual=False,
            round_index=round_index,
        )
        emitted.update(region)
        remaining.difference_update(region)

    return emitted


def _partition_component(
    adjacency: Sequence[Sequence[int]],
    component: list[int],
    component_seed: int,
    config: TokenizationConfig,
    emitter: _Emitter,
) -> tuple[set[int], int]:
    allowed = set(component)
    eccentricities = {}
    for node in component:
        distances = _distances_from_node(adjacency, node, len(adjacency))
        eccentricities[node] = max(distances[other] for other in component)
    minimum = min(eccentricities.values())
    graph_centers = [
        node for node, value in eccentricities.items() if value == minimum
    ]
    initial_center = _choose_canonically(
        graph_centers,
        _canonical_ranks(adjacency, allowed),
        component_seed,
        "initial-center",
    )
    initial_distances = _distances_from_node(
        adjacency, initial_center, len(adjacency),
    )
    window = {
        node for node in component
        if initial_distances[node] <= config.center_radius
    }

    internal = set(window)
    for level, radius in zip((2, 3), config.scale_radii[:2]):
        internal.update(_cover_one_round(
            adjacency,
            allowed,
            internal,
            level=level,
            radius=radius,
            center_distance=radius + 1,
            round_index=0,
            component_seed=component_seed,
            emitter=emitter,
        ))

    level4_round = 0
    radius = config.scale_radii[2]
    while internal != allowed:
        previous_size = len(internal)
        internal.update(_cover_one_round(
            adjacency,
            allowed,
            internal,
            level=4,
            radius=radius,
            center_distance=radius + 1,
            round_index=level4_round,
            component_seed=component_seed,
            emitter=emitter,
        ))
        if len(internal) <= previous_size:
            raise RuntimeError("level-4 iteration failed to cover additional nodes")
        level4_round += 1

    return window, level4_round


def partition_graph(
    adjacency: Sequence[Sequence[int]],
    *,
    seed: int | None = None,
    config: TokenizationConfig | None = None,
    include_stats: bool = True,
) -> TokenPartition:
    """Sample P(G; omega) and return scale indices and region memberships."""

    config = config or TokenizationConfig()
    num_nodes = len(adjacency)
    if num_nodes == 0:
        raise ValueError("empty graphs are not supported")
    _validate_adjacency(adjacency, num_nodes)
    rng = random.Random(seed) if seed is not None else random.Random()
    graph_seed = rng.getrandbits(64)

    all_nodes = set(range(num_nodes))
    components = _ordered_components(adjacency, all_nodes)
    emitter = _Emitter(num_nodes)
    q_mask = torch.zeros(num_nodes, dtype=torch.bool)
    total_level4_rounds = 0
    for component in components:
        component_signature = _canonical_signature(adjacency, set(component))
        component_seed = _derive_seed(graph_seed, component_signature)
        window, level4_rounds = _partition_component(
            adjacency, component, component_seed, config, emitter,
        )
        q_mask[torch.tensor(sorted(window), dtype=torch.long)] = True
        total_level4_rounds = max(total_level4_rounds, level4_rounds)

    if emitter.members:
        members = tuple(emitter.members)
        centers = torch.tensor(emitter.centers, dtype=torch.long)
        levels = torch.tensor(emitter.levels, dtype=torch.long)
        residual = torch.tensor(emitter.residual, dtype=torch.bool)
        rounds = torch.tensor(emitter.rounds, dtype=torch.long)
    else:
        members = ()
        centers = torch.empty(0, dtype=torch.long)
        levels = torch.empty(0, dtype=torch.long)
        residual = torch.empty(0, dtype=torch.bool)
        rounds = torch.empty(0, dtype=torch.long)

    partition = TokenPartition(
        members=members,
        centers=centers,
        levels=levels,
        residual=residual,
        rounds=rounds,
        owner=emitter.owner,
        q_mask=q_mask,
        stats={},
    )
    partition.validate(num_nodes)

    stats: dict[str, float | int] = {}
    if include_stats:
        level_counts = {
            level: int((levels == level).sum()) for level in (2, 3, 4)
        }
        level_nodes = {
            level: int(sum(members[index].numel() for index in torch.where(levels == level)[0]))
            for level in (2, 3, 4)
        }
        residual_counts = {
            level: int(((levels == level) & residual).sum()) for level in (2, 3, 4)
        }
        q_size = int(q_mask.sum())
        stats = {
            "num_nodes": num_nodes,
            "num_components": len(components),
            "diameter": max((_diameter(adjacency, set(c)) for c in components), default=0),
            "q_size": q_size,
            "q_fraction": q_size / num_nodes,
            "level4_rounds": total_level4_rounds,
        }
        for level in (2, 3, 4):
            stats[f"level{level}_tokens"] = level_counts[level]
            stats[f"level{level}_nodes"] = level_nodes[level]
            stats[f"level{level}_residual_tokens"] = residual_counts[level]
            stats[f"level{level}_residual_rate"] = (
                residual_counts[level] / level_counts[level]
                if level_counts[level] else 0.0
            )
    return TokenPartition(
        members=partition.members,
        centers=partition.centers,
        levels=partition.levels,
        residual=partition.residual,
        rounds=partition.rounds,
        owner=partition.owner,
        q_mask=partition.q_mask,
        stats=stats,
    )


def partition_molecule(
    bonds: Tensor,
    *,
    node_mask: Tensor | None = None,
    seed: int | None = None,
    config: TokenizationConfig | None = None,
    include_stats: bool = True,
) -> TokenPartition:
    adjacency = adjacency_from_bonds(bonds, node_mask)
    active_nodes = {
        index for index, valid in enumerate(node_mask.tolist()) if valid
    } if node_mask is not None else set(range(bonds.size(0)))
    if not active_nodes:
        raise ValueError("molecule contains no active nodes")
    kept = sorted(active_nodes)
    compact_index = {node: index for index, node in enumerate(kept)}
    compact = [
        [compact_index[neighbor] for neighbor in adjacency[node]]
        for node in kept
    ]
    local = partition_graph(
        compact,
        seed=seed,
        config=config,
        include_stats=include_stats,
    )
    lookup = torch.tensor(kept, dtype=torch.long)
    owner = torch.full((bonds.size(0),), -1, dtype=torch.long)
    q_mask = torch.zeros(bonds.size(0), dtype=torch.bool)
    owner[lookup] = local.owner
    q_mask[lookup] = local.q_mask
    mapped_centers = torch.full_like(local.centers, -1)
    center_valid = local.centers.ge(0)
    mapped_centers[center_valid] = lookup[local.centers[center_valid]]
    return TokenPartition(
        members=tuple(lookup[index] for index in local.members),
        centers=mapped_centers,
        levels=local.levels,
        residual=local.residual,
        rounds=local.rounds,
        owner=owner,
        q_mask=q_mask,
        stats=local.stats,
    )
