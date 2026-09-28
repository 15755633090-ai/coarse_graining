"""Deterministic topology-guided, feature-diverse center tokens."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn


@dataclass(frozen=True)
class CenterSelection:
    """Indices are local to the unpadded molecule; scales are Near/Mid/Far."""

    indices: np.ndarray
    scales: np.ndarray
    core_radii: np.ndarray
    pair_buckets: np.ndarray
    cores: np.ndarray
    coverage_centers: int = 0


def _components(distances: np.ndarray, tie_keys) -> list[np.ndarray]:
    unseen = set(range(len(distances)))
    result = []
    while unseen:
        start = min(unseen)
        component = np.flatnonzero(distances[start] >= 0)
        result.append(component)
        unseen.difference_update(map(int, component))
    return sorted(result, key=lambda item: (
        -len(item), tuple(sorted(tie_keys[i] for i in item)), int(item[0]),
    ))


def _graph_center(distances: np.ndarray, nodes: np.ndarray, tie_keys) -> int:
    local = distances[np.ix_(nodes, nodes)]
    eccentricity = local.max(axis=1)
    mean_distance = local.mean(axis=1)
    return min(map(int, nodes), key=lambda i: (
        eccentricity[np.where(nodes == i)[0][0]],
        mean_distance[np.where(nodes == i)[0][0]], tie_keys[i], i,
    ))


def _choose_cores(distances: np.ndarray, budget: int, tie_keys) -> list[int]:
    components = _components(distances, tie_keys)
    first = _graph_center(distances, components[0], tie_keys)
    cores = [first]
    if budget == 1:
        return cores
    if len(components) > 1:
        cores.append(_graph_center(distances, components[1], tie_keys))
        return cores
    nodes = components[0]
    local = distances[np.ix_(nodes, nodes)]
    diameter = int(local.max())
    if diameter < 16:
        return cores
    eccentricity = local.max(axis=1)
    candidates = nodes[(eccentricity <= eccentricity.min() + math.ceil(diameter / 4))
                       & (distances[first, nodes] >= 3)]
    if not len(candidates):
        return cores
    cores.append(int(min(map(int, candidates), key=lambda i: (
        -int(distances[first, i]), int(distances[i, nodes].max()), tie_keys[i], i,
    ))))
    return cores


def bucket_graph_distances(distances: np.ndarray) -> np.ndarray:
    """Buckets 0,1,2,3,4,5+ and disconnected (6)."""
    return np.where(distances < 0, 6, np.minimum(distances, 5)).astype(np.int64)


def _atom_tie_keys(distances, states, atom_features=None, bonds=None):
    """Atom-order-independent signatures for exact score ties.

    With molecular inputs, signatures use atom labels, bond types, and the
    multiset of distances to labeled atoms. In direct array-only calls, stable
    rounded embedding values provide a fallback. Remaining exact symmetries
    use atom order, which cannot distinguish equivalent vertices.
    """
    n = len(distances)
    if atom_features is None:
        return [tuple(np.round(np.concatenate([h[i] for h in states]), 6))
                for i in range(n)]
    features = np.asarray(atom_features)
    edges = np.asarray(bonds)
    if features.shape[0] != n or edges.shape != (n, n):
        raise ValueError("atom labels and bonds must match graph distances")
    labels = [tuple(map(int, row)) for row in features]
    keys = []
    for i in range(n):
        neighbors = tuple(sorted((int(edges[i, j]), labels[j]) for j in range(n)
                                 if 0 < edges[i, j] < 5))
        distances_to_labels = tuple(sorted((int(distances[i, j]) if distances[i, j] >= 0 else n + 1,
                                           labels[j]) for j in range(n) if j != i))
        keys.append((labels[i], neighbors, distances_to_labels))
    return keys


def select_center_tokens(
    distances: np.ndarray,
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    stride: int = 4,
    min_centers: int = 3,
    max_centers: int = 32,
    min_separation: int = 2,
    coverage_radius: int = 3,
    atom_features: np.ndarray | None = None,
    bonds: np.ndarray | None = None,
) -> CenterSelection:
    """Select K centers using topology for spacing and H2/H3/H4 for novelty.

    A graph-derived core set starts a farthest-point topology coverage pass.
    Until every atom is within ``coverage_radius`` hops of a center, distance
    dominates feature novelty. Remaining slots use Near:Mid:Far weights 3:2:1;
    two-hop separation applies when possible, then cosine novelty ranks atoms.
    """
    d = np.asarray(distances, dtype=np.int32)
    n = len(d)
    if (n < 1 or d.shape != (n, n) or not np.array_equal(d, d.T)
            or np.any(np.diag(d) != 0) or np.any(d < -1)):
        raise ValueError("distances must be a nonempty symmetric graph-distance matrix")
    if min(stride, min_centers, max_centers, min_separation, coverage_radius) < 1:
        raise ValueError("center configuration must be positive")
    if min_centers > max_centers:
        raise ValueError("min_centers cannot exceed max_centers")
    if len(states) != 3 or any(np.asarray(h).shape[0] != n for h in states):
        raise ValueError("H2/H3/H4 must each contain one row per atom")
    normalized = []
    nonzero = []
    for h in states:
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2 or not np.isfinite(h).all():
            raise ValueError("center representations must be finite matrices")
        norms = np.linalg.norm(h, axis=1, keepdims=True)
        normalized.append(h / np.maximum(norms, 1e-12))
        nonzero.append(norms[:, 0] >= 1e-6)
    tie_keys = _atom_tie_keys(d, normalized, atom_features, bonds)
    k = min(n, max_centers, max(min_centers, math.ceil(n / stride)))
    cores = _choose_cores(d, k, tie_keys)
    nearest_core = np.where(d[cores] >= 0, d[cores], np.iinfo(np.int32).max).min(axis=0)
    finite = nearest_core < np.iinfo(np.int32).max
    radius = np.ones(n, dtype=np.float32)
    radius[finite] = nearest_core[finite] / max(1, int(nearest_core[finite].max()))
    bands = np.where(radius <= 1 / 3, 0, np.where(radius <= 2 / 3, 1, 2))
    selected = list(cores)
    selected_scales = [0] * len(cores)
    counts = [len(cores), 0, 0]
    weights = (3, 2, 1)
    # Topology coverage takes precedence over feature novelty and band quotas.
    while len(selected) < k:
        candidates = np.flatnonzero(~np.isin(np.arange(n), selected))
        topology = np.where(d[np.ix_(candidates, selected)] < 0, n + 1,
                            d[np.ix_(candidates, selected)]).min(axis=1)
        farthest = int(topology.max())
        if farthest <= coverage_radius:
            break
        eligible = np.flatnonzero(topology == farthest)
        def coverage_key(j):
            atom = int(candidates[j])
            band = int(bands[atom])
            novelty = (1 - normalized[band][atom] @ normalized[band][selected].T).min()
            if not nonzero[band][atom]:
                novelty = -1.0
            return (-float(novelty), tie_keys[atom], atom)
        chosen = min(map(int, eligible), key=coverage_key)
        atom = int(candidates[chosen])
        selected.append(atom)
        band = int(bands[atom])
        selected_scales.append(band)
        counts[band] += 1
    coverage_centers = len(selected) - len(cores)
    # Remaining slots increase representation diversity within spatial bands.
    while len(selected) < k:
        remaining = [np.flatnonzero((bands == band) & ~np.isin(np.arange(n), selected))
                     for band in range(3)]
        available = [band for band in range(3) if len(remaining[band])]
        if not available:
            raise AssertionError("center selection exhausted candidates before K")
        uncovered = [band for band in available if counts[band] == 0]
        band = max(uncovered) if uncovered else min(
            available, key=lambda b: (counts[b] / weights[b], -weights[b]),
        )
        candidates = remaining[band]
        topology = np.where(d[np.ix_(candidates, selected)] < 0, n + 1,
                            d[np.ix_(candidates, selected)]).min(axis=1)
        for threshold in range(min_separation, 0, -1):
            eligible = np.flatnonzero(topology >= threshold)
            if len(eligible):
                break
        h = normalized[band]
        novelty = (1 - h[candidates] @ h[selected].T).min(axis=1)
        novelty[~nonzero[band][candidates]] = -1.0
        chosen = min(map(int, eligible), key=lambda j: (
            -float(novelty[j]), -int(topology[j]), tie_keys[int(candidates[j])],
            int(candidates[j]),
        ))
        selected.append(int(candidates[chosen]))
        selected_scales.append(band)
        counts[band] += 1
    indices = np.asarray(selected, dtype=np.int64)
    return CenterSelection(
        indices=indices,
        scales=np.asarray(selected_scales, dtype=np.int64),
        core_radii=radius[indices],
        pair_buckets=bucket_graph_distances(d[np.ix_(indices, indices)]),
        cores=np.asarray(cores, dtype=np.int64),
        coverage_centers=coverage_centers,
    )


class CenterTokenBranch(nn.Module):
    """No region pooling: selected H2/H3/H4 atoms become the tokens."""

    def __init__(self, hidden_dim: int, dropout: float, heads: int = 4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.projections = nn.ModuleList(nn.Linear(hidden_dim, hidden_dim) for _ in range(3))
        self.core_position = nn.Sequential(
            nn.Linear(1, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        self.distance_bias = nn.Embedding(7, heads)
        nn.init.zeros_(self.distance_bias.weight)
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )

    def forward(self, states, mask, selections, max_centers, *, use_h4_only=False):
        batch, _, hidden = states[-1].shape
        if len(selections) != batch:
            raise ValueError("one center selection is required per molecule")
        batch_centers = max(len(selection.indices) for selection in selections)
        if batch_centers < 1 or batch_centers > max_centers:
            raise ValueError("center count exceeds safety limit")
        tokens = states[-1].new_zeros((batch, batch_centers, hidden))
        valid = torch.zeros((batch, batch_centers), device=tokens.device, dtype=torch.bool)
        buckets = torch.zeros((batch, batch_centers, batch_centers),
                              device=tokens.device, dtype=torch.long)
        counts = torch.zeros((batch, 3), device=tokens.device, dtype=torch.long)
        for graph, selection in enumerate(selections):
            size = len(selection.indices)
            valid_nodes = torch.where(mask[graph])[0]
            indices = torch.as_tensor(selection.indices, device=tokens.device)
            if torch.any(indices < 0) or torch.any(indices >= len(valid_nodes)):
                raise ValueError("center index is outside the molecule")
            scales = torch.as_tensor(selection.scales, device=tokens.device)
            radius = torch.as_tensor(selection.core_radii, device=tokens.device,
                                     dtype=tokens.dtype).unsqueeze(-1)
            projected = torch.zeros((size, hidden), device=tokens.device, dtype=tokens.dtype)
            for scale in range(3):
                slots = torch.where(scales == scale)[0]
                if len(slots):
                    source_layer = 2 if use_h4_only else scale
                    source = states[source_layer][graph, valid_nodes[indices[slots]]]
                    projected[slots] = self.projections[scale](source).to(projected.dtype)
                    counts[graph, scale] = len(slots)
            tokens[graph, :size] = projected + self.core_position(radius)
            valid[graph, :size] = True
            buckets[graph, :size, :size] = torch.as_tensor(
                selection.pair_buckets, device=tokens.device,
            )
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, batch_centers, 3, self.heads, hidden // self.heads,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hidden // self.heads)
        bias = self.distance_bias(buckets).permute(0, 3, 1, 2)
        scores = (scores + bias).masked_fill(~valid[:, None, None, :], float("-inf"))
        attention = self.dropout(scores.softmax(-1)).to(v.dtype)
        context = (attention @ v).transpose(1, 2).reshape(batch, batch_centers, hidden)
        tokens = tokens + self.dropout(self.project(context))
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        summed = tokens.sum(1)
        mean = summed / valid.sum(1, keepdim=True)
        maximum = tokens.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
        return torch.cat((mean, maximum), dim=-1), counts
