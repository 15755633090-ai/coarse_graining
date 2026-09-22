"""Frozen diffusion encoder plus the four-scale token readout."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from diffusion_encoder.encoder import DEFAULT_CHECKPOINT, load_frozen_encoder
from diffusion_encoder.model import DiffusionEncoder

from .partition import (
    TokenizationConfig,
    derive_partition_seed,
    partition_molecule,
)


@dataclass(frozen=True)
class MultiscaleModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    output_dim: int = 1
    default_partition_seed: int = 0
    tokenization: TokenizationConfig = TokenizationConfig()

    def __post_init__(self) -> None:
        if self.hidden_dim < 1 or self.output_dim < 1:
            raise ValueError("hidden_dim and output_dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.default_partition_seed < 0:
            raise ValueError("default_partition_seed must be nonnegative")


@dataclass
class TokenModelOutput:
    prediction: Tensor
    graph_representation: Tensor
    level_norms: Tensor
    token_counts: Tensor
    residual_counts: Tensor


def _region_mlp(hidden_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(hidden_dim, hidden_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(hidden_dim, hidden_dim),
    )


class MultiscaleMolecularModel(nn.Module):
    """First-version main model; the diffusion encoder remains frozen."""

    def __init__(
        self,
        encoder: DiffusionEncoder | None = None,
        *,
        checkpoint: str | Path = DEFAULT_CHECKPOINT,
        device: str | torch.device = "cpu",
        config: MultiscaleModelConfig | None = None,
    ):
        super().__init__()
        self.encoder = encoder or load_frozen_encoder(checkpoint, device=device)
        self.encoder.requires_grad_(False).eval()
        hidden_dim = self.encoder.config.hidden_dim
        self.config = config or MultiscaleModelConfig(hidden_dim=hidden_dim)
        if self.config.hidden_dim != hidden_dim:
            raise ValueError("config.hidden_dim must match the frozen encoder")
        self.region_encoders = nn.ModuleList([
            _region_mlp(hidden_dim, self.config.dropout) for _ in range(3)
        ])
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(self.config.dropout),
            nn.Linear(hidden_dim, self.config.output_dim),
        )

    def train(self, mode: bool = True) -> "MultiscaleMolecularModel":
        super().train(mode)
        self.encoder.eval()
        return self

    def _partition_seeds(
        self,
        partition_seeds: int | Sequence[int] | Tensor | None,
        batch_size: int,
    ) -> list[int]:
        if partition_seeds is None:
            return [
                derive_partition_seed(self.config.default_partition_seed, index)
                for index in range(batch_size)
            ]
        if isinstance(partition_seeds, int):
            return [
                derive_partition_seed(partition_seeds, index)
                for index in range(batch_size)
            ]
        seeds = (
            partition_seeds.detach().cpu().tolist()
            if isinstance(partition_seeds, Tensor)
            else list(partition_seeds)
        )
        if len(seeds) != batch_size:
            raise ValueError("one partition seed is required per graph")
        return [int(seed) for seed in seeds]

    def forward(
        self,
        node_features: Tensor,
        bonds: Tensor,
        node_mask: Tensor,
        partition_seeds: int | Sequence[int] | Tensor | None = None,
    ) -> TokenModelOutput:
        with torch.no_grad():
            h1, h2, h3, h4 = self.encoder(node_features, bonds, node_mask)
        hidden_states = (h1, h2, h3, h4)
        batch_size = node_features.size(0)
        seeds = self._partition_seeds(partition_seeds, batch_size)

        graph_representations: list[Tensor] = []
        level_norms: list[Tensor] = []
        token_counts: list[Tensor] = []
        residual_counts: list[Tensor] = []
        zero = h1.new_zeros(self.config.hidden_dim)
        bonds_cpu = bonds.detach().cpu()
        mask_cpu = node_mask.detach().cpu()

        for graph_index in range(batch_size):
            valid = mask_cpu[graph_index]
            partition = partition_molecule(
                bonds_cpu[graph_index],
                node_mask=valid,
                seed=seeds[graph_index],
                config=self.config.tokenization,
                include_stats=False,
            )
            q_indices = torch.where(partition.q_mask)[0].to(h1.device)
            zq = h1[graph_index, q_indices].mean(dim=0)
            scale_vectors = [zq]
            graph_token_counts = [int(q_indices.numel())]
            graph_residual_counts = [0]

            for encoder_level, scale_index in ((2, 0), (3, 1), (4, 2)):
                hidden = hidden_states[encoder_level - 1][graph_index]
                token_indices = partition.level_tokens(encoder_level)
                if token_indices:
                    pooled = torch.stack([
                        hidden[partition.members[index].to(h1.device)].mean(dim=0)
                        for index in token_indices
                    ])
                    encoded = self.region_encoders[scale_index](pooled)
                    sizes = torch.tensor(
                        [partition.members[index].numel() for index in token_indices],
                        dtype=encoded.dtype,
                        device=encoded.device,
                    )
                    scale_vectors.append((encoded * (sizes / sizes.sum()).unsqueeze(-1)).sum(dim=0))
                    graph_token_counts.append(len(token_indices))
                    graph_residual_counts.append(
                        int(partition.residual[list(token_indices)].sum())
                    )
                else:
                    scale_vectors.append(zero)
                    graph_token_counts.append(0)
                    graph_residual_counts.append(0)

            graph_representations.append(torch.stack(scale_vectors).sum(dim=0))
            level_norms.append(
                torch.stack([vector.norm() for vector in scale_vectors])
            )
            token_counts.append(torch.tensor(graph_token_counts))
            residual_counts.append(torch.tensor(graph_residual_counts))

        graph_representation = torch.stack(graph_representations)
        return TokenModelOutput(
            prediction=self.prediction_head(graph_representation),
            graph_representation=graph_representation,
            level_norms=torch.stack(level_norms),
            token_counts=torch.stack(token_counts).to(h1.device),
            residual_counts=torch.stack(residual_counts).to(h1.device),
        )
