from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

import torch
from torch import Tensor, nn

from .config import ModelConfig
from .data import MASK_BOND


def masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.dim() < values.dim():
        weights = weights.unsqueeze(-1)
    return (values * weights).sum(dim=dim) / weights.sum(dim=dim).clamp_min(1.0)


class TimeEmbedding(nn.Module):
    def __init__(self, time_dim: int, hidden_dim: int):
        super().__init__()
        self.time_dim = time_dim
        self.project = nn.Sequential(
            nn.Linear(time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, t: Tensor) -> Tensor:
        half = self.time_dim // 2
        frequencies = torch.exp(
            -math.log(10_000.0)
            * torch.arange(half, device=t.device)
            / max(1, half - 1)
        )
        angles = t.float().unsqueeze(-1) * frequencies.unsqueeze(0)
        encoding = torch.cat([angles.sin(), angles.cos()], dim=-1)
        if encoding.size(-1) < self.time_dim:
            encoding = nn.functional.pad(encoding, (0, 1))
        return self.project(encoding)


class ChemicalMessagePassing(nn.Module):
    """Dense pair-aware layer whose node update explicitly aggregates neighbors."""

    def __init__(self, hidden_dim: int, dropout: float):
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gate = nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(
        self, h: Tensor, edge_h: Tensor, bonds: Tensor, time_h: Tensor, node_mask: Tensor
    ) -> tuple[Tensor, Tensor]:
        bsz, n, dim = h.shape
        hi = h.unsqueeze(2).expand(bsz, n, n, dim)
        hj = h.unsqueeze(1).expand(bsz, n, n, dim)
        pair_input = torch.cat([hi, hj, edge_h], dim=-1)
        pair_valid = node_mask.unsqueeze(1) & node_mask.unsqueeze(2)
        eye = torch.eye(n, dtype=torch.bool, device=h.device).unsqueeze(0)
        # Chemical environment means currently observed bonded neighbors. Unknown
        # (mask) and no-bond pairs must not turn the graph into a complete graph.
        pair_valid &= (~eye) & (bonds > 0) & (bonds < MASK_BOND)
        messages = self.message(pair_input) * self.gate(pair_input)
        neighbor_context = masked_mean(messages, pair_valid, dim=2)
        delta = self.update(
            torch.cat([h, neighbor_context, time_h.unsqueeze(1).expand_as(h)], dim=-1)
        )
        h = self.norm(h + delta)
        h = h * node_mask.unsqueeze(-1)
        return h, neighbor_context


@dataclass
class ModelOutput:
    node_logits: List[Tensor]
    bond_logits: Tensor
    node_embeddings: Tensor
    graph_embedding: Tensor
    neighbor_context: Tensor


class BondAwareDiffusionModel(nn.Module):
    """Joint node/bond denoiser and reusable bonding-aware graph encoder."""

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        vocab_sizes = (
            config.atom_vocab_size,
            config.charge_vocab_size,
            config.aromatic_vocab_size,
            config.hybrid_vocab_size,
            config.degree_vocab_size,
        )
        self.node_embeddings = nn.ModuleList(
            [nn.Embedding(size, config.hidden_dim) for size in vocab_sizes]
        )
        self.bond_embedding = nn.Embedding(config.bond_vocab_size, config.hidden_dim)
        self.time_embedding = TimeEmbedding(config.time_dim, config.hidden_dim)
        self.layers = nn.ModuleList(
            [
                ChemicalMessagePassing(config.hidden_dim, config.dropout)
                for _ in range(config.num_layers)
            ]
        )
        self.node_heads = nn.ModuleList(
            [nn.Linear(config.hidden_dim, size - 1) for size in vocab_sizes]
        )
        # h_ij = MLP(h_i, h_j, neighbor(i), neighbor(j), e_ij, t)
        self.edge_head = nn.Sequential(
            nn.Linear(config.hidden_dim * 6, config.hidden_dim * 2),
            nn.SiLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim * 2, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.bond_vocab_size - 1),
        )
        self.graph_projection = nn.Sequential(
            nn.Linear(config.hidden_dim * 2, config.graph_dim),
            nn.SiLU(),
            nn.LayerNorm(config.graph_dim),
        )

    def _encode_backbone(
        self, node_features: Tensor, bonds: Tensor, node_mask: Tensor, timesteps: Tensor
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        h = sum(
            embedding(node_features[:, :, field])
            for field, embedding in enumerate(self.node_embeddings)
        )
        h = h * node_mask.unsqueeze(-1)
        edge_h = self.bond_embedding(bonds)
        time_h = self.time_embedding(timesteps)
        neighbor_context = torch.zeros_like(h)
        for layer in self.layers:
            h, neighbor_context = layer(h, edge_h, bonds, time_h, node_mask)
        return h, neighbor_context, edge_h, time_h

    def encode_nodes(
        self,
        node_features: Tensor,
        bonds: Tensor,
        node_mask: Tensor,
        timesteps: Tensor | None = None,
    ) -> Tensor:
        """Return final node states H without running diffusion decoder heads."""

        if timesteps is None:
            timesteps = torch.zeros(
                node_features.size(0), dtype=torch.long, device=node_features.device
            )
        h, _, _, _ = self._encode_backbone(
            node_features, bonds, node_mask, timesteps
        )
        return h

    def forward(
        self, node_features: Tensor, bonds: Tensor, node_mask: Tensor, timesteps: Tensor
    ) -> ModelOutput:
        h, neighbor_context, edge_h, time_h = self._encode_backbone(
            node_features, bonds, node_mask, timesteps
        )

        node_logits = [head(h) for head in self.node_heads]
        bsz, n, dim = h.shape
        hi = h.unsqueeze(2).expand(bsz, n, n, dim)
        hj = h.unsqueeze(1).expand(bsz, n, n, dim)
        ni = neighbor_context.unsqueeze(2).expand(bsz, n, n, dim)
        nj = neighbor_context.unsqueeze(1).expand(bsz, n, n, dim)
        time_pair = time_h[:, None, None, :].expand(bsz, n, n, dim)
        pair_features = torch.cat([hi + hj, hi * hj, ni + nj, ni * nj, edge_h, time_pair], dim=-1)
        bond_logits = self.edge_head(pair_features)
        # Enforce exact i,j symmetry even with floating-point implementation changes.
        bond_logits = 0.5 * (bond_logits + bond_logits.transpose(1, 2))
        graph_embedding = self.graph_projection(
            torch.cat(
                [
                    masked_mean(h, node_mask, dim=1),
                    masked_mean(neighbor_context, node_mask, dim=1),
                ],
                dim=-1,
            )
        )
        return ModelOutput(node_logits, bond_logits, h, graph_embedding, neighbor_context)

    def encode(self, node_features: Tensor, bonds: Tensor, node_mask: Tensor) -> Tensor:
        """Return z=f_theta(G) at diffusion step zero.

        Legacy interface: graph_projection is not optimized by the supplied
        ChemicalDiffusionLoss. Prefer encode_nodes() plus task-specific pooling
        when using the existing diffusion-pretrained checkpoint.

        Gradient tracking is controlled by the caller so this method supports
        both frozen feature extraction and full encoder fine-tuning.
        """

        timesteps = torch.zeros(
            node_features.size(0), dtype=torch.long, device=node_features.device
        )
        h, neighbor_context, _, _ = self._encode_backbone(
            node_features, bonds, node_mask, timesteps
        )
        return self.graph_projection(
            torch.cat(
                [
                    masked_mean(h, node_mask, dim=1),
                    masked_mean(neighbor_context, node_mask, dim=1),
                ],
                dim=-1,
            )
        )
