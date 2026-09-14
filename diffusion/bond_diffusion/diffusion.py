from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor

from .config import DiffusionConfig, ModelConfig
from .data import MoleculeBatch


@dataclass
class NoisyGraph:
    node_features: Tensor
    bonds: Tensor
    timesteps: Tensor
    node_corruption_mask: Tensor
    bond_corruption_mask: Tensor


class DiscreteGraphDiffusion:
    """Mask-and-replace diffusion over both atom attributes and bond classes."""

    def __init__(self, config: DiffusionConfig, model_config: ModelConfig):
        self.config = config
        self.model_config = model_config
        self.node_vocab_sizes = (
            model_config.atom_vocab_size,
            model_config.charge_vocab_size,
            model_config.aromatic_vocab_size,
            model_config.hybrid_vocab_size,
            model_config.degree_vocab_size,
        )

    def noise_rate(self, timesteps: Tensor) -> Tensor:
        phase = timesteps.float() / max(1, self.config.timesteps - 1)
        # Cosine interpolation starts gently and destroys most information near T.
        smooth = 1.0 - torch.cos(phase * torch.pi / 2.0)
        return self.config.min_noise + (
            self.config.max_noise - self.config.min_noise
        ) * smooth

    def sample_timesteps(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.randint(0, self.config.timesteps, (batch_size,), device=device)

    def corrupt(self, batch: MoleculeBatch, timesteps: Tensor | None = None) -> NoisyGraph:
        x = batch.node_features.clone()
        e = batch.bonds.clone()
        bsz, n, num_fields = x.shape
        if timesteps is None:
            timesteps = self.sample_timesteps(bsz, x.device)
        rates = self.noise_rate(timesteps)

        node_valid = batch.node_mask.unsqueeze(-1).expand_as(x)
        node_corrupt = (torch.rand_like(x.float()) < rates[:, None, None]) & node_valid
        random_node = node_corrupt & (
            torch.rand_like(x.float()) < self.config.random_replace_prob
        )
        for field, vocab_size in enumerate(self.node_vocab_sizes):
            field_corrupt = node_corrupt[:, :, field]
            field_random = random_node[:, :, field]
            x[:, :, field] = torch.where(
                field_corrupt,
                torch.full_like(x[:, :, field], vocab_size - 1),
                x[:, :, field],
            )
            # Exclude the mask token; atomic number 0 remains reserved for padding.
            low = 1 if field == 0 else 0
            replacement = torch.randint(
                low, vocab_size - 1, (bsz, n), device=x.device
            )
            x[:, :, field] = torch.where(field_random, replacement, x[:, :, field])

        # Corrupt each undirected pair once, then mirror the result.
        pair_rates = rates[:, None, None]
        edge_corrupt_upper = (
            torch.rand_like(e.float()) < pair_rates
        ) & batch.pair_mask
        if self.config.substructure_mask_prob > 0:
            extra_sub = (
                torch.rand_like(e.float()) < self.config.substructure_mask_prob
            ) & batch.substructure_mask
            edge_corrupt_upper |= extra_sub
        random_edge = edge_corrupt_upper & (
            torch.rand_like(e.float()) < self.config.random_replace_prob
        )
        noisy_upper = torch.where(
            edge_corrupt_upper, torch.full_like(e, self.model_config.bond_vocab_size - 1), e
        )
        edge_replacement = torch.randint(
            0, self.model_config.bond_vocab_size - 1, e.shape, device=e.device
        )
        noisy_upper = torch.where(random_edge, edge_replacement, noisy_upper)
        upper = torch.triu(noisy_upper, diagonal=1)
        e = upper + upper.transpose(1, 2)
        edge_corrupt = edge_corrupt_upper | edge_corrupt_upper.transpose(1, 2)
        return NoisyGraph(x, e, timesteps, node_corrupt, edge_corrupt)
