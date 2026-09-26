"""Overlapping random regions and one distance-aware attention block."""
from __future__ import annotations

import math
from collections import deque
from functools import lru_cache

import numpy as np
import torch
from torch import nn


@lru_cache(maxsize=8192)
def _distances(nodes: int, adjacency_bytes: bytes) -> np.ndarray:
    adjacency = np.frombuffer(adjacency_bytes, dtype=np.bool_).reshape(nodes, nodes)
    neighbors = [np.flatnonzero(row).tolist() for row in adjacency]
    distances = np.full((nodes, nodes), -1, dtype=np.int32)
    for start in range(nodes):
        distances[start, start] = 0
        queue = deque([start])
        while queue:
            node = queue.popleft()
            for other in neighbors[node]:
                if distances[start, other] < 0:
                    distances[start, other] = distances[start, node] + 1
                    queue.append(other)
    distances.setflags(write=False)
    return distances


def graph_distances(bonds: torch.Tensor) -> np.ndarray:
    adjacency = ((bonds > 0) & (bonds < 5)).cpu().numpy()
    return _distances(len(adjacency), adjacency.tobytes())


def sample_regions(distances, seed, radius=2, atoms_per_center=8, max_centers=8):
    nodes = len(distances)
    if nodes < 1:
        raise ValueError("random regions require at least one atom")
    count = min(nodes, max_centers, max(2, math.ceil(nodes / atoms_per_center)))
    centers = np.random.default_rng(seed).choice(nodes, count, replace=False)
    membership = (distances[centers] >= 0) & (distances[centers] <= radius)
    return centers, membership


class RandomRegionBranch(nn.Module):
    def __init__(self, hidden_dim, dropout, heads=4):
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.region_mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.qkv = nn.Linear(hidden_dim, 3 * hidden_dim)
        # Exact distances 0..8, then 9..12, 13+, and disconnected.
        self.distance_bias = nn.Embedding(12, heads)
        nn.init.zeros_(self.distance_bias.weight)
        self.project = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, 2 * hidden_dim), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(2 * hidden_dim, hidden_dim),
        )

    @staticmethod
    def bucket_distances(distances):
        return np.where(distances < 0, 11,
                        np.where(distances <= 8, distances,
                                 np.where(distances <= 12, 9, 10)))

    def forward(self, h4, bonds, mask, seeds, *, radius, atoms_per_center, max_centers):
        batch, nodes, hidden = h4.shape
        memberships = np.zeros((batch, max_centers, nodes), dtype=np.float32)
        buckets = np.zeros((batch, max_centers, max_centers), dtype=np.int64)
        valid = np.zeros((batch, max_centers), dtype=np.bool_)
        bonds_cpu, mask_cpu = bonds.detach().cpu(), mask.detach().cpu()
        for index, seed in enumerate(seeds):
            atom_indices = torch.where(mask_cpu[index])[0]
            local_bonds = bonds_cpu[index][atom_indices][:, atom_indices]
            distances = graph_distances(local_bonds)
            centers, members = sample_regions(
                distances, seed, radius, atoms_per_center, max_centers,
            )
            count = len(centers)
            memberships[index, :count, :][:, atom_indices.numpy()] = members
            buckets[index, :count, :count] = self.bucket_distances(distances[np.ix_(centers, centers)])
            valid[index, :count] = True
        membership = torch.as_tensor(memberships, device=h4.device, dtype=h4.dtype)
        valid = torch.as_tensor(valid, device=h4.device)
        sums = membership @ h4
        means = sums / membership.sum(-1, keepdim=True).clamp_min(1)
        tokens = self.region_mlp(torch.cat((sums, means), -1))
        qkv = self.qkv(self.norm1(tokens)).reshape(batch, max_centers, 3, self.heads, hidden // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-1, -2)) / math.sqrt(hidden // self.heads)
        bias = self.distance_bias(torch.as_tensor(buckets, device=h4.device)).permute(0, 3, 1, 2)
        scores = (scores + bias).masked_fill(~valid[:, None, None, :], float("-inf"))
        attention = self.dropout(scores.softmax(-1)).to(v.dtype)
        context = (attention @ v).transpose(1, 2).reshape(batch, max_centers, hidden)
        tokens = tokens + self.dropout(self.project(context))
        tokens = tokens + self.dropout(self.ffn(self.norm2(tokens)))
        tokens = tokens * valid.unsqueeze(-1)
        sums = tokens.sum(1)
        counts = valid.sum(1)
        return torch.cat((sums, sums / counts[:, None]), -1), counts
