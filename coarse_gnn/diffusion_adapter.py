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
from .cache import TopologyCache
from .topology import CoarseTopology


def encode_nodes_with_intermediates(
    encoder: nn.Module,
    node_features: Tensor,
    bonds: Tensor,
    node_mask: Tensor,
    *,
    layers=(2,),
    timesteps: Tensor | None = None,
) -> tuple[Tensor, dict[int, Tensor]]:
    """Capture selected 1-based backbone layers without modifying the encoder.

    Hooks preserve the original checkpoint class, its forward behavior and the
    autograd graph. The dense backbone still runs exactly once for the batch.
    """
    requested = tuple(sorted(set(layers)))
    backbone = getattr(encoder, "layers", None)
    if backbone is None:
        raise TypeError("encoder must expose its message-passing layers")
    if any(not isinstance(index, int) or isinstance(index, bool) for index in requested):
        raise TypeError("intermediate layer indices must be integers")
    if any(index < 1 or index > len(backbone) for index in requested):
        raise ValueError(f"intermediate layers must be in [1, {len(backbone)}]")
    captured: dict[int, Tensor] = {}
    handles = []

    def capture(index):
        def hook(_module, _inputs, output):
            captured[index] = output[0] if isinstance(output, tuple) else output
        return hook

    try:
        for index in requested:
            handles.append(backbone[index - 1].register_forward_hook(capture(index)))
        if timesteps is None:
            final = encoder.encode_nodes(node_features, bonds, node_mask)
        else:
            final = encoder.encode_nodes(node_features, bonds, node_mask, timesteps)
    finally:
        for handle in handles:
            handle.remove()
    if set(captured) != set(requested):
        raise RuntimeError("encoder did not execute every requested intermediate layer")
    return final, captured


def molecular_graph_inputs(batch: MoleculeBatch):
    """Validate clean molecular graphs and extract unpadded sparse inputs."""
    if batch.node_features.ndim != 3 or batch.node_features.size(-1) != 5:
        raise ValueError("Expected padded molecular node_features [B, N, 5]")
    b, n = batch.node_features.shape[:2]
    if b < 1 or batch.node_mask.shape != (b, n) or batch.node_mask.dtype != torch.bool:
        raise ValueError("Expected nonempty batch with boolean node_mask [B, N]")
    if batch.bonds.dtype != torch.long or batch.bonds.shape != (b, n, n):
        raise ValueError("Expected long bonds [B, N, N]")
    if not batch.node_mask.any(dim=1).all():
        raise ValueError("Empty graphs are not supported")
    inputs = []
    for i in range(b):
        valid = torch.where(batch.node_mask[i])[0]
        bonds = batch.bonds[i][valid][:, valid]
        if (bonds < 0).any() or (bonds >= MASK_BOND).any():
            raise ValueError("Coarsening requires clean bond labels 0..4")
        if not torch.equal(bonds, bonds.T) or bonds.diagonal().any():
            raise ValueError("Bonds must be symmetric with zero diagonal")
        edges = torch.triu(bonds > 0, diagonal=1).nonzero().T.contiguous()
        inputs.append((valid, edges, batch.node_features[i, valid], bonds[edges[0], edges[1]]))
    return inputs


def precompute_batch_topologies(batch: MoleculeBatch, cache: TopologyCache,
                               config: CoarseningConfig | None = None) -> list[CoarseTopology]:
    """Populate a cache without constructing/loading a neural encoder."""
    return [cache.get_or_build(len(valid), edges, config, node_labels=nodes, edge_labels=labels)
            for valid, edges, nodes, labels in molecular_graph_inputs(batch)]


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
        topology_cache: TopologyCache | None = None,
    ) -> "DiffusionCoarseModel":
        encoder = load_encoder(checkpoint, device)
        network = network or NetworkConfig(input_dim=encoder.config.hidden_dim, edge_dim=4)
        return cls(encoder, CoarseGraphPredictor(network, coarsening, topology_cache=topology_cache), freeze_encoder).to(device)

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

    def precompute_topologies(self, batch: MoleculeBatch, *, node_ids: list[Tensor] | None = None) -> list[CoarseTopology]:
        inputs = molecular_graph_inputs(batch)
        if node_ids is not None and len(node_ids) != len(inputs):
            raise ValueError("node_ids must contain one tensor per graph")
        return [self.predictor.prepare_topology(
            len(valid), edges, node_labels=nodes, edge_labels=labels,
            node_ids=None if node_ids is None else node_ids[i],
        ) for i, (valid, edges, nodes, labels) in enumerate(inputs)]

    def forward(self, batch: MoleculeBatch, *, node_ids: list[Tensor] | None = None,
                topologies: list[CoarseTopology] | None = None) -> BatchOutput:
        inputs = molecular_graph_inputs(batch)
        if node_ids is not None and len(node_ids) != len(inputs):
            raise ValueError("node_ids must contain one tensor per graph")
        if topologies is not None and len(topologies) != len(inputs):
            raise ValueError("topologies must contain one topology per graph in batch order")
        context = torch.no_grad() if self.freeze_encoder else nullcontext()
        with context:
            nodes = self.encoder.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)
        outputs = []
        for i, (valid, edges, raw_nodes, labels) in enumerate(inputs):
            attributes = nn.functional.one_hot(labels - 1, num_classes=4).to(nodes.dtype) if self.predictor.config.edge_dim else None
            outputs.append(self.predictor(
                nodes[i, valid], edges, attributes,
                node_ids=None if node_ids is None else node_ids[i],
                node_labels=raw_nodes, edge_labels=labels,
                topology=None if topologies is None else topologies[i],
            ))
        return BatchOutput(
            torch.stack([out.prediction for out in outputs]),
            torch.stack([out.graph_embedding for out in outputs]), outputs,
        )
