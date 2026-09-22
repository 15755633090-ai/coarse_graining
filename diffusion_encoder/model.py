"""The original four-layer diffusion backbone, without decoder or property heads."""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


MASK_BOND = 5


@dataclass(frozen=True)
class ModelConfig:
    hidden_dim: int = 128
    num_layers: int = 4
    dropout: float = 0.1
    time_dim: int = 64
    atom_vocab_size: int = 120
    charge_vocab_size: int = 12
    aromatic_vocab_size: int = 3
    hybrid_vocab_size: int = 8
    degree_vocab_size: int = 8
    bond_vocab_size: int = 6
    graph_dim: int = 256


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.project = nn.Sequential(
            nn.Linear(time_dim, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, timesteps: Tensor) -> Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=timesteps.device)
            / max(1, half - 1)
        )
        angles = timesteps.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        encoding = torch.cat((angles.sin(), angles.cos()), dim=-1)
        if encoding.size(-1) < self.time_dim:
            encoding = nn.functional.pad(encoding, (0, 1))
        return self.project(encoding)


class ChemicalMessagePassing(nn.Module):
    """The unchanged dense pair-aware layer from diffusion pretraining."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.SiLU(), nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: Tensor, edge_h: Tensor, bonds: Tensor, time_h: Tensor, node_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size, nodes, dimension = h.shape
        hi = h.unsqueeze(2).expand(batch_size, nodes, nodes, dimension)
        hj = h.unsqueeze(1).expand(batch_size, nodes, nodes, dimension)
        pair_input = torch.cat((hi, hj, edge_h), dim=-1)
        pair_valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        eye = torch.eye(nodes, dtype=torch.bool, device=h.device).unsqueeze(0)
        pair_valid &= (~eye) & (bonds > 0) & (bonds < MASK_BOND)
        messages = self.message(pair_input) * self.gate(pair_input)
        neighbor_context = _masked_mean(messages, pair_valid, dim=2)
        delta = self.update(torch.cat((
            h, neighbor_context, time_h.unsqueeze(1).expand_as(h),
        ), dim=-1))
        h = self.norm(h + delta) * node_mask.unsqueeze(-1)
        return h, neighbor_context


class DiffusionEncoder(nn.Module):
    """Frozen four-layer encoder whose forward returns h1, h2, h3 and h4."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        if config.num_layers != 4:
            raise ValueError("This clean project is locked to the original four-layer encoder")
        self.config = config
        vocabulary_sizes = (
            config.atom_vocab_size, config.charge_vocab_size,
            config.aromatic_vocab_size, config.hybrid_vocab_size,
            config.degree_vocab_size,
        )
        self.node_embeddings = nn.ModuleList([
            nn.Embedding(size, config.hidden_dim) for size in vocabulary_sizes
        ])
        self.bond_embedding = nn.Embedding(config.bond_vocab_size, config.hidden_dim)
        self.time_embedding = TimeEmbedding(config.time_dim, config.hidden_dim)
        self.layers = nn.ModuleList([
            ChemicalMessagePassing(config.hidden_dim, config.dropout)
            for _ in range(config.num_layers)
        ])

    def forward(
        self,
        node_features: Tensor,
        bonds: Tensor,
        node_mask: Tensor,
        timesteps: Tensor | None = None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        if node_features.ndim != 3 or node_features.size(-1) != 5:
            raise ValueError("node_features must have shape [batch, nodes, 5]")
        batch_size, nodes = node_features.shape[:2]
        if bonds.shape != (batch_size, nodes, nodes):
            raise ValueError("bonds must have shape [batch, nodes, nodes]")
        if node_mask.shape != (batch_size, nodes) or node_mask.dtype != torch.bool:
            raise ValueError("node_mask must be boolean with shape [batch, nodes]")
        if timesteps is None:
            timesteps = torch.zeros(batch_size, dtype=torch.long, device=node_features.device)
        h = sum(
            embedding(node_features[:, :, field])
            for field, embedding in enumerate(self.node_embeddings)
        )
        h = h * node_mask.unsqueeze(-1)
        edge_h = self.bond_embedding(bonds)
        time_h = self.time_embedding(timesteps)
        states = []
        for layer in self.layers:
            h, _ = layer(h, edge_h, bonds, time_h, node_mask)
            states.append(h)
        return tuple(states)
