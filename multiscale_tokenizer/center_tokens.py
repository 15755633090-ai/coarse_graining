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


def _components(distances: np.ndarray) -> list[np.ndarray]:
    unseen = set(range(len(distances)))
    result = []
    while unseen:
        start = min(unseen)
        component = np.flatnonzero(distances[start] >= 0)
        result.append(component)
        unseen.difference_update(map(int, component))
    return sorted(result, key=lambda item: (-len(item), int(item[0])))


def _graph_center(distances: np.ndarray, nodes: np.ndarray) -> int:
    local = distances[np.ix_(nodes, nodes)]
    eccentricity = local.max(axis=1)
    mean_distance = local.mean(axis=1)
    return min(map(int, nodes), key=lambda i: (
        eccentricity[np.where(nodes == i)[0][0]],
        mean_distance[np.where(nodes == i)[0][0]], i,
    ))


def _choose_cores(distances: np.ndarray, count: int) -> list[int]:
    components = _components(distances)
    first = _graph_center(distances, components[0])
    cores = [first]
    if count == 1:
        return cores
    if len(components) > 1:
        cores.append(_graph_center(distances, components[1]))
        return cores
    nodes = components[0]
    eccentricity = distances[np.ix_(nodes, nodes)].max(axis=1)
    candidates = nodes[eccentricity <= eccentricity.min() + 1]
    candidates = candidates[candidates != first]
    if not len(candidates):
        candidates = nodes[nodes != first]
    cores.append(int(min(map(int, candidates), key=lambda i: (
        -int(distances[first, i]), int(distances[i, nodes].max()), i,
    ))))
    return cores


def bucket_graph_distances(distances: np.ndarray) -> np.ndarray:
    """Buckets 0,1,2,3,4,5+ and disconnected (6)."""
    return np.where(distances < 0, 6, np.minimum(distances, 5)).astype(np.int64)


def select_center_tokens(
    distances: np.ndarray,
    states: tuple[np.ndarray, np.ndarray, np.ndarray],
    *,
    stride: int = 4,
    min_centers: int = 3,
    max_centers: int = 8,
    min_separation: int = 2,
) -> CenterSelection:
    """Select K centers using topology for spacing and H2/H3/H4 for novelty.

    Core selection uses only graph distances. A band has priority if it has no
    selected center; otherwise allocation follows Near:Mid:Far weights 3:2:1.
    Within a band, a two-hop separation is a hard constraint when possible,
    then cosine novelty ranks candidates. Ties use graph distance and atom ID.
    """
    d = np.asarray(distances, dtype=np.int32)
    n = len(d)
    if (n < 1 or d.shape != (n, n) or not np.array_equal(d, d.T)
            or np.any(np.diag(d) != 0) or np.any(d < -1)):
        raise ValueError("distances must be a nonempty symmetric graph-distance matrix")
    if min(stride, min_centers, max_centers, min_separation) < 1:
        raise ValueError("center configuration must be positive")
    if len(states) != 3 or any(np.asarray(h).shape[0] != n for h in states):
        raise ValueError("H2/H3/H4 must each contain one row per atom")
    normalized = []
    for h in states:
        h = np.asarray(h, dtype=np.float64)
        if h.ndim != 2 or not np.isfinite(h).all():
            raise ValueError("center representations must be finite matrices")
        normalized.append(h / np.maximum(np.linalg.norm(h, axis=1, keepdims=True), 1e-12))
    k = min(n, max_centers, max(min_centers, math.ceil(n / stride)))
    core_count = 1 if k < 6 else 2
    cores = _choose_cores(d, core_count)
    nearest_core = np.where(d[cores] >= 0, d[cores], np.iinfo(np.int32).max).min(axis=0)
    finite = nearest_core < np.iinfo(np.int32).max
    radius = np.ones(n, dtype=np.float32)
    radius[finite] = nearest_core[finite] / max(1, int(nearest_core[finite].max()))
    bands = np.where(radius <= 1 / 3, 0, np.where(radius <= 2 / 3, 1, 2))
    selected = list(cores)
    selected_scales = [0] * len(cores)
    counts = [len(cores), 0, 0]
    weights = (3, 2, 1)
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
        chosen = min(map(int, eligible), key=lambda j: (
            -float(novelty[j]), -int(topology[j]), int(candidates[j]),
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

    def forward(self, states, mask, selections, max_centers):
        batch, _, hidden = states[-1].shape
        if len(selections) != batch:
            raise ValueError("one center selection is required per molecule")
        tokens = states[-1].new_zeros((batch, max_centers, hidden))
        valid = torch.zeros((batch, max_centers), device=tokens.device, dtype=torch.bool)
        buckets = torch.zeros((batch, max_centers, max_centers),
                              device=tokens.device, dtype=torch.long)
        counts = torch.zeros((batch, 3), device=tokens.device, dtype=torch.long)
        for graph, selection in enumerate(selections):
            size = len(selection.indices)
            if size < 1 or size > max_centers:
                raise ValueError("center count exceeds token budget")
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
                    source = states[scale][graph, valid_nodes[indices[slots]]]
                    projected[slots] = self.projections[scale](source).to(projected.dtype)
                    counts[graph, scale] = len(slots)
            tokens[graph, :size] = projected + self.core_position(radius)
            valid[graph, :size] = True
            buckets[graph, :size, :size] = torch.as_tensor(
                selection.pair_buckets, device=tokens.device,
            )
        qkv = self.qkv(self.norm1(tokens)).reshape(
            batch, max_centers, 3, self.heads, hidden // self.heads,
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hidden // self.heads)
        bias = self.distance_bias(buckets).permute(0, 3, 1, 2)
        scores = (scores + bias).masked_fill(~valid[:, None, None, :], float("-inf"))
        attention = self.dropout(scores.softmax(-1)).to(v.dtype)
        context = (attention @ v).transpose(1, 2).reshape(batch, max_centers, hidden)
        tokens = tokens + self.dropout(self.project(context))
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        tokens = tokens.masked_fill(~valid.unsqueeze(-1), 0)
        summed = tokens.sum(1)
        mean = summed / valid.sum(1, keepdim=True)
        maximum = tokens.masked_fill(~valid.unsqueeze(-1), float("-inf")).max(1).values
        return torch.cat((mean, maximum), dim=-1), counts
