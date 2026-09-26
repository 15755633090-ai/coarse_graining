"""Matched baseline/multiscale models with frozen or trainable encoders."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from torch import Tensor, nn

from diffusion_encoder.encoder import DEFAULT_CHECKPOINT, load_frozen_encoder
from diffusion_encoder.model import DiffusionEncoder

from .partition import (
    TokenPartition,
    TokenizationConfig,
    derive_partition_seed,
    partition_molecule,
)
from .random_regions import RandomRegionBranch


EXPERIMENT_MODES = (
    "random_region_stage2_frozen",
    "baseline_frozen",
    "multiscale_frozen",
    "baseline_finetune",
    "multiscale_finetune",
    "baseline_stage2_frozen",
    "multiscale_stage2_frozen",
    "multiscale_residual_frozen",
    "baseline_residual_frozen",
)


@dataclass(frozen=True)
class MultiscaleModelConfig:
    hidden_dim: int = 128
    dropout: float = 0.1
    output_dim: int = 1
    default_partition_seed: int = 0
    tokenization: TokenizationConfig = TokenizationConfig()
    experiment_mode: str = "multiscale_frozen"
    region_radius: int = 2
    atoms_per_center: int = 8
    max_centers: int = 8
    eval_views: int = 5

    def __post_init__(self) -> None:
        if self.region_radius < 0 or min(self.atoms_per_center, self.max_centers, self.eval_views) < 1:
            raise ValueError("invalid random region configuration")
        if self.hidden_dim < 1 or self.output_dim < 1:
            raise ValueError("hidden_dim and output_dim must be positive")
        if not 0 <= self.dropout < 1:
            raise ValueError("dropout must be in [0, 1)")
        if self.default_partition_seed < 0:
            raise ValueError("default_partition_seed must be nonnegative")
        if self.experiment_mode not in EXPERIMENT_MODES:
            raise ValueError(
                f"experiment_mode must be one of {EXPERIMENT_MODES}"
            )

    @property
    def uses_multiscale(self) -> bool:
        return self.experiment_mode.startswith("multiscale_")

    @property
    def finetunes_encoder(self) -> bool:
        return self.experiment_mode.endswith("_finetune")

    @property
    def is_residual(self) -> bool:
        return self.experiment_mode.endswith("_residual_frozen")


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
    """Unified four-arm property model.

    Multiscale modes preserve the existing near-fine/far-coarse architecture.
    Baseline modes preserve the historical diffusion Sum/Mean readout.
    """

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
        hidden_dim = self.encoder.config.hidden_dim
        self.config = config or MultiscaleModelConfig(hidden_dim=hidden_dim)
        if self.config.hidden_dim != hidden_dim:
            raise ValueError("config.hidden_dim must match the encoder")
        if self.config.experiment_mode.startswith("random_region_"):
            self.region_encoders = nn.ModuleList()
            self.random_regions = RandomRegionBranch(hidden_dim, self.config.dropout)
            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_dim * 4, hidden_dim), nn.SiLU(),
                nn.Dropout(self.config.dropout), nn.Linear(hidden_dim, self.config.output_dim),
            )
        elif self.config.uses_multiscale:
            # This is intentionally the legacy construction order so existing
            # frozen multiscale initialization and checkpoints remain unchanged.
            self.region_encoders = nn.ModuleList([
                _region_mlp(hidden_dim, self.config.dropout) for _ in range(3)
            ])
            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(hidden_dim, self.config.output_dim),
            )
        else:
            self.region_encoders = nn.ModuleList()
            self.prediction_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.SiLU(),
                nn.Dropout(self.config.dropout),
                nn.Linear(hidden_dim, self.config.output_dim),
            )
        if self.config.is_residual:
            self.base_head = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim), nn.SiLU(),
                nn.Dropout(self.config.dropout), nn.Linear(hidden_dim, self.config.output_dim),
            )
            self.base_head.requires_grad_(False).eval()
            nn.init.zeros_(self.prediction_head[-1].weight)
            nn.init.zeros_(self.prediction_head[-1].bias)
        self.set_encoder_frozen(not self.config.finetunes_encoder)

    def set_encoder_frozen(self, frozen: bool) -> None:
        """Apply the configured encoder policy without touching its weights."""

        self.encoder.requires_grad_(not frozen)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True) -> "MultiscaleMolecularModel":
        super().train(mode)
        if not self.config.finetunes_encoder:
            self.encoder.eval()
        if self.config.is_residual:
            self.base_head.eval()
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
        partitions: Sequence[TokenPartition] | None = None,
    ) -> TokenModelOutput:
        context = nullcontext() if self.config.finetunes_encoder else torch.no_grad()
        # Keep the complete frozen Base in FP32 even inside training autocast.
        precision = torch.autocast(device_type=node_features.device.type, enabled=False) if self.config.is_residual else nullcontext()
        with context, precision:
            h1, h2, h3, h4 = self.encoder(node_features, bonds, node_mask)
            base_prediction = None
            if self.config.is_residual:
                weights = node_mask.to(h4.dtype).unsqueeze(-1)
                summed = (h4 * weights).sum(dim=1)
                averaged = summed / weights.sum(dim=1).clamp_min(1.0)
                base_prediction = self.base_head(torch.cat((summed, averaged), dim=-1))
        hidden_states = (h1, h2, h3, h4)
        batch_size = node_features.size(0)
        if self.config.experiment_mode.startswith("random_region_"):
            if partitions is not None:
                raise ValueError("random regions do not use multiscale partitions")
            weights = node_mask.to(h4.dtype).unsqueeze(-1)
            summed = (h4 * weights).sum(1)
            base = torch.cat((summed, summed / weights.sum(1).clamp_min(1)), -1)
            seeds = self._partition_seeds(partition_seeds, batch_size)
            predictions, representations = [], []
            for view in range(1 if self.training else self.config.eval_views):
                view_seeds = [derive_partition_seed(seed, view) for seed in seeds]
                coarse, counts = self.random_regions(
                    h4, bonds, node_mask, view_seeds,
                    radius=self.config.region_radius,
                    atoms_per_center=self.config.atoms_per_center,
                    max_centers=self.config.max_centers,
                )
                representation = torch.cat((base, coarse), -1)
                predictions.append(self.prediction_head(representation))
                representations.append(representation)
            token_counts = torch.zeros((batch_size, 4), dtype=torch.long, device=h4.device)
            token_counts[:, 3] = counts
            return TokenModelOutput(
                prediction=torch.stack(predictions).mean(0),
                graph_representation=torch.stack(representations).mean(0),
                level_norms=torch.zeros((batch_size, 4), device=h4.device),
                token_counts=token_counts,
                residual_counts=torch.zeros_like(token_counts),
            )
        if not self.config.uses_multiscale:
            weights = node_mask.to(h4.dtype).unsqueeze(-1)
            summed = (h4 * weights).sum(dim=1)
            averaged = summed / weights.sum(dim=1).clamp_min(1.0)
            graph_representation = torch.cat((summed, averaged), dim=-1)
            level_norms = torch.zeros((batch_size, 4), device=h4.device)
            level_norms[:, 3] = averaged.norm(dim=-1)
            token_counts = torch.zeros(
                (batch_size, 4), dtype=torch.long, device=h4.device,
            )
            token_counts[:, 3] = node_mask.sum(dim=1)
            return TokenModelOutput(
                prediction=self.prediction_head(graph_representation) + (base_prediction if base_prediction is not None else 0),
                graph_representation=graph_representation,
                level_norms=level_norms,
                token_counts=token_counts,
                residual_counts=torch.zeros_like(token_counts),
            )
        if partitions is not None and len(partitions) != batch_size:
            raise ValueError("one partition is required per graph")
        seeds = (
            None if partitions is not None
            else self._partition_seeds(partition_seeds, batch_size)
        )

        bonds_cpu = bonds.detach().cpu() if partitions is None else None
        mask_cpu = node_mask.detach().cpu() if partitions is None else None

        if partitions is None:
            partitions = [
                partition_molecule(
                    bonds_cpu[graph_index],
                    node_mask=mask_cpu[graph_index],
                    seed=seeds[graph_index],
                    config=self.config.tokenization,
                    include_stats=False,
                )
                for graph_index in range(batch_size)
            ]

        node_count = node_features.size(1)
        device = h1.device
        graph_index_cpu: list[Tensor] = []
        q_index_cpu: list[Tensor] = []
        for graph_index, partition in enumerate(partitions):
            q_indices = torch.where(partition.q_mask)[0] + graph_index * node_count
            q_index_cpu.append(q_indices)
            graph_index_cpu.append(torch.full_like(q_indices, graph_index))
        q_index = torch.cat(q_index_cpu).to(device)
        q_graph = torch.cat(graph_index_cpu).to(device)
        q_count = torch.bincount(q_graph, minlength=batch_size)
        zq = h1.reshape(-1, self.config.hidden_dim).index_select(0, q_index)
        zq = torch.zeros((batch_size, self.config.hidden_dim), device=device).index_add_(
            0, q_graph, zq,
        )
        zq = zq / q_count.clamp_min(1).unsqueeze(-1)

        scale_vectors = [zq]
        token_count_columns = [q_count]
        residual_count_columns = [torch.zeros_like(q_count)]
        for encoder_level, scale_index in ((2, 0), (3, 1), (4, 2)):
            hidden = hidden_states[encoder_level - 1].reshape(
                -1, self.config.hidden_dim,
            )
            member_index_cpu: list[Tensor] = []
            member_token_cpu: list[Tensor] = []
            token_graph_cpu: list[Tensor] = []
            token_size_cpu: list[int] = []
            token_residual_cpu: list[bool] = []
            token_id = 0
            for graph_index, partition in enumerate(partitions):
                for local_index in partition.level_tokens(encoder_level):
                    members = (
                        partition.members[local_index] + graph_index * node_count
                    )
                    member_index_cpu.append(members)
                    member_token_cpu.append(
                        torch.full_like(members, token_id)
                    )
                    token_graph_cpu.append(
                        torch.full((1,), graph_index, dtype=torch.long)
                    )
                    token_size_cpu.append(int(members.numel()))
                    token_residual_cpu.append(
                        bool(partition.residual[local_index])
                    )
                    token_id += 1

            if token_id == 0:
                scale_vectors.append(
                    torch.zeros(
                        (batch_size, self.config.hidden_dim),
                        device=device,
                    )
                )
                token_count_columns.append(torch.zeros_like(q_count))
                residual_count_columns.append(torch.zeros_like(q_count))
                continue

            member_index = torch.cat(member_index_cpu).to(device)
            member_token = torch.cat(member_token_cpu).to(device)
            token_graph = torch.cat(token_graph_cpu).to(device)
            token_size = torch.tensor(
                token_size_cpu, dtype=torch.float32, device=device,
            )
            pooled = torch.zeros(
                (token_id, self.config.hidden_dim), device=device,
            ).index_add_(
                0, member_token, hidden.index_select(0, member_index),
            )
            pooled = pooled / token_size.unsqueeze(-1)
            encoded = self.region_encoders[scale_index](pooled)
            graph_size = torch.zeros(batch_size, device=device).index_add_(
                0, token_graph, token_size,
            )
            weights = token_size / graph_size[token_graph].clamp_min(1.0)
            scale = torch.zeros(
                (batch_size, self.config.hidden_dim), device=device,
            ).index_add_(
                0, token_graph, encoded * weights.unsqueeze(-1),
            )
            scale_vectors.append(scale)
            token_count_columns.append(
                torch.bincount(token_graph, minlength=batch_size)
            )
            residual_count_columns.append(torch.bincount(
                token_graph[torch.tensor(
                    token_residual_cpu, dtype=torch.bool, device=device,
                )],
                minlength=batch_size,
            ))

        scale_tensor = torch.stack(scale_vectors, dim=1)
        graph_representation = scale_tensor.sum(dim=1)
        level_norms = scale_tensor.norm(dim=-1)
        return TokenModelOutput(
            prediction=self.prediction_head(graph_representation) + (base_prediction if base_prediction is not None else 0),
            graph_representation=graph_representation,
            level_norms=level_norms,
            token_counts=torch.stack(token_count_columns, dim=1),
            residual_counts=torch.stack(residual_count_columns, dim=1),
        )
