from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .data import MoleculeBatch
from .diffusion import NoisyGraph
from .model import ModelOutput


BOND_ORDERS = (0.0, 1.0, 2.0, 3.0, 1.5)


@dataclass
class LossOutput:
    total: Tensor
    node: Tensor
    bond: Tensor
    valence: Tensor
    substructure: Tensor

    def scalars(self) -> Dict[str, float]:
        return {
            key: float(value.detach())
            for key, value in self.__dict__.items()
        }


class ChemicalDiffusionLoss(nn.Module):
    def __init__(
        self,
        lambda_bond: float = 2.0,
        lambda_valence: float = 0.2,
        lambda_substructure: float = 1.0,
        nonbond_weight: float = 0.15,
    ):
        super().__init__()
        self.lambda_bond = lambda_bond
        self.lambda_valence = lambda_valence
        self.lambda_substructure = lambda_substructure
        self.nonbond_weight = nonbond_weight
        self.register_buffer("bond_orders", torch.tensor(BOND_ORDERS))
        # Common maximum valence by atomic number; unknown elements use 8.
        max_valence = torch.full((119,), 8.0)
        for atomic_number, value in {
            1: 1, 5: 3, 6: 4, 7: 3, 8: 2, 9: 1,
            14: 4, 15: 5, 16: 6, 17: 1, 35: 1, 53: 1,
        }.items():
            max_valence[atomic_number] = value
        self.register_buffer("max_valence", max_valence)

    @staticmethod
    def _masked_ce(logits: Tensor, targets: Tensor, mask: Tensor) -> Tensor:
        if not mask.any():
            return logits.sum() * 0.0
        return F.cross_entropy(logits[mask], targets[mask])

    def forward(
        self, output: ModelOutput, clean: MoleculeBatch, noisy: NoisyGraph
    ) -> LossOutput:
        # Denoising is evaluated where corruption occurred. A zero-noise minibatch
        # falls back to all valid nodes/pairs to keep the objective defined.
        node_terms = []
        for field, logits in enumerate(output.node_logits):
            field_mask = noisy.node_corruption_mask[:, :, field]
            if not field_mask.any():
                field_mask = clean.node_mask
            node_terms.append(self._masked_ce(logits, clean.node_features[:, :, field], field_mask))
        node_loss = torch.stack(node_terms).mean()

        bond_mask = noisy.bond_corruption_mask & clean.pair_mask
        if not bond_mask.any():
            bond_mask = clean.pair_mask
        bond_weight = torch.ones(5, device=output.bond_logits.device)
        bond_weight[0] = self.nonbond_weight
        if bond_mask.any():
            bond_loss = F.cross_entropy(
                output.bond_logits[bond_mask], clean.bonds[bond_mask], weight=bond_weight
            )
        else:
            # A batch of single-atom graphs has no valid atom pairs. Cross
            # entropy on an empty target returns NaN; there is no bond task
            # for this batch, but retain a differentiable zero for the head.
            bond_loss = output.bond_logits.sum() * 0.0

        probabilities = output.bond_logits.softmax(dim=-1)
        expected_orders = (probabilities * self.bond_orders).sum(dim=-1)
        # Both directions are needed for per-atom valence; diagonal self-pairs are not.
        all_valid_pairs = clean.pair_mask | clean.pair_mask.transpose(1, 2)
        expected_valence = (expected_orders * all_valid_pairs).sum(dim=-1)
        atomic_numbers = clean.node_features[:, :, 0].clamp(0, 118)
        limits = self.max_valence[atomic_numbers].clone()
        formal_charge = clean.node_features[:, :, 1] - 5
        # Avoid penalizing common charged but valid centers.
        limits = torch.where(
            (atomic_numbers == 7) & (formal_charge > 0),
            torch.full_like(limits, 4.0),
            limits,
        )
        limits = torch.where(
            (atomic_numbers == 8) & (formal_charge > 0),
            torch.full_like(limits, 3.0),
            limits,
        )
        valence_excess = F.relu(expected_valence - limits)
        valence_loss = (
            (valence_excess.square() * clean.node_mask).sum()
            / clean.node_mask.sum().clamp_min(1)
        )

        sub_mask = clean.substructure_mask
        substructure_loss = self._masked_ce(
            output.bond_logits, clean.bonds, sub_mask
        )
        total = (
            node_loss
            + self.lambda_bond * bond_loss
            + self.lambda_valence * valence_loss
            + self.lambda_substructure * substructure_loss
        )
        return LossOutput(total, node_loss, bond_loss, valence_loss, substructure_loss)
