"""Integration with the existing dense molecular diffusion encoder."""
from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from diffusion.bond_diffusion.data import MASK_BOND, MoleculeBatch
from diffusion.bond_diffusion.trainer import load_encoder

from .config import CoarseningConfig, NetworkConfig
from .model import CoarseGraphPredictor, GraphOutput


@dataclass
class BatchOutput:
    predictions: Tensor  # [B, output_dim]
    graph_embeddings: Tensor  # [B, hidden_dim]
    graphs: list[GraphOutput]


class DiffusionCoarseModel(nn.Module):
    def __init__(self, encoder: nn.Module, predictor: CoarseGraphPredictor, freeze_encoder: bool = True):
        super().__init__()
        if encoder.config.hidden_dim != predictor.config.input_dim:
            raise ValueError("Predictor input_dim must match encoder hidden_dim")
        if predictor.config.edge_dim not in (0, MASK_BOND - 1):
            raise ValueError("The molecular adapter accepts edge_dim=0 or 4 (one-hot bond types)")
        self.encoder = encoder
        self.predictor = predictor
        self.set_encoder_frozen(freeze_encoder)

    @classmethod
    def from_checkpoint(
        cls, checkpoint: str | Path, *, network: NetworkConfig | None = None,
        coarsening: CoarseningConfig | None = None, freeze_encoder: bool = True,
        device: str | torch.device = "cpu",
    ) -> "DiffusionCoarseModel":
        encoder = load_encoder(checkpoint, device)
        network = network or NetworkConfig(input_dim=encoder.config.hidden_dim, edge_dim=4)
        return cls(encoder, CoarseGraphPredictor(network, coarsening), freeze_encoder).to(device)

    def set_encoder_frozen(self, frozen: bool) -> None:
        self.freeze_encoder = frozen
        self.encoder.requires_grad_(not frozen)
        # The downstream task uses only encode_nodes; decoder heads are inactive.
        for name in ("node_heads", "edge_head", "graph_projection"):
            module = getattr(self.encoder, name, None)
            if module is not None:
                module.requires_grad_(False)
        self.encoder.train(self.training and not frozen)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()  # Frozen features must not acquire dropout noise.
        return self

    def forward(self, batch: MoleculeBatch, *, node_ids: list[Tensor] | None = None) -> BatchOutput:
        if batch.node_features.ndim != 3 or batch.node_features.size(-1) != 5:
            raise ValueError("Expected padded molecular node_features [B, N, 5]")
        b, n = batch.node_features.shape[:2]
        if b < 1 or batch.node_mask.shape != (b, n) or batch.node_mask.dtype != torch.bool:
            raise ValueError("Expected nonempty batch with boolean node_mask [B, N]")
        if batch.bonds.dtype != torch.long or batch.bonds.shape != (b, n, n):
            raise ValueError("Expected long bonds [B, N, N]")
        if not batch.node_mask.any(dim=1).all():
            raise ValueError("Empty graphs are not supported")
        if node_ids is not None and len(node_ids) != b:
            raise ValueError("node_ids must contain one tensor per graph")
        # Coarsening uses clean observed topology, never masked diffusion pairs.
        for i in range(b):
            valid = torch.where(batch.node_mask[i])[0]
            bonds = batch.bonds[i][valid][:, valid]
            if (bonds < 0).any() or (bonds >= MASK_BOND).any():
                raise ValueError("Coarsening requires clean bond labels 0..4")
            if not torch.equal(bonds, bonds.T) or bonds.diagonal().any():
                raise ValueError("Bonds must be symmetric with zero diagonal")
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with context:
            nodes = self.encoder.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)
        outputs = []
        for i in range(b):
            valid = torch.where(batch.node_mask[i])[0]
            bonds = batch.bonds[i][valid][:, valid]
            edges = torch.triu(bonds > 0, diagonal=1).nonzero().T.contiguous()
            labels = bonds[edges[0], edges[1]]
            attributes = nn.functional.one_hot(labels - 1, num_classes=4).to(nodes.dtype) if self.predictor.config.edge_dim else None
            outputs.append(self.predictor(
                nodes[i, valid], edges, attributes,
                node_ids=None if node_ids is None else node_ids[i],
                node_labels=batch.node_features[i, valid], edge_labels=labels,
            ))
        return BatchOutput(
            torch.stack([out.prediction for out in outputs]),
            torch.stack([out.graph_embedding for out in outputs]), outputs,
        )
