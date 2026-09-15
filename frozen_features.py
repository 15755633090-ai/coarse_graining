"""Immutable FP32 encoder features shared by the frozen mechanism experiment."""
from pathlib import Path

import torch

from coarse_gnn.packed import PreparedGraphBatch, packed_predict
import run_lipo_formal as formal


class FrozenGraphBatch(PreparedGraphBatch):
    def __init__(self, graph, topologies, node_states, plan=None):
        super().__init__(graph, topologies, plan)
        self.node_states = node_states

    def to(self, device):
        return FrozenGraphBatch(self.graph.to(device), self.topologies,
                                self.node_states.to(device), self.plan.to(device))


class FrozenCoarseModel(formal.FormalCoarsePredictor):
    def forward(self, batch):
        if not self.freeze_encoder or self.encoder.training:
            raise RuntimeError("Frozen experiment requires an eval-mode, frozen encoder")
        if isinstance(batch, FrozenGraphBatch):
            nodes = batch.node_states
        else:
            with torch.no_grad(), torch.autocast(batch.node_features.device.type, enabled=False):
                nodes = self.encoder.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)
        return packed_predict(self.predictor, nodes, batch)


def frozen_factory(original_factory, cache):
    construct = formal.make_factory(original_factory, cache)
    def create(variant, num_tasks, checkpoint, device, dropout):
        if variant not in ("region_only", "base_coarse"):
            raise ValueError("Stage one runs only Region-only and Base coarse")
        original = construct(variant, num_tasks, checkpoint, device, dropout)
        return FrozenCoarseModel(original.encoder, original.predictor, freeze_encoder=True).to(device)
    return create


class FrozenDataset:
    def __init__(self, prepared, features):
        self.prepared, self.features = prepared, features

    def __len__(self):
        return len(self.prepared)

    def __getattr__(self, key):
        return getattr(self.prepared, key)

    def __getitem__(self, index):
        return (*self.prepared[index], self.features[index])


def frozen_tools(legacy):
    collate_original, slice_original = legacy.collate_property_batch, legacy._slice_property_batch
    def collate(rows):
        batch = collate_original([row[:3] for row in rows])
        shape = (*batch.graph.node_mask.shape, rows[0][4].shape[-1])
        nodes = torch.zeros(shape, dtype=torch.float32)
        for index, row in enumerate(rows):
            nodes[index, :len(row[4])] = row[4]
        batch.graph = FrozenGraphBatch(batch.graph, [row[3] for row in rows], nodes)
        return batch

    def slice_batch(batch, start, end):
        sliced = slice_original(batch, start, end)
        nodes = batch.graph.node_states[start:end, :sliced.graph.node_features.size(1)]
        sliced.graph = FrozenGraphBatch(sliced.graph, batch.graph.topologies[start:end], nodes)
        return sliced
    return dict(collate_property_batch=collate, _slice_property_batch=slice_batch)


def feature_identity(legacy, args, prepared, split_sha):
    return dict(version=1, checkpoint=legacy.file_identity(args.checkpoint), split_sha256=split_sha,
                graph_fingerprints=[prepared[i][3].input_fingerprint for i in range(len(prepared))],
                encoder_mode="eval_no_grad", precision="float32_autocast_disabled",
                extraction_batch_size=args.micro_batch_size, device=str(args.device),
                torch_version=str(torch.__version__), structure=formal.STRUCTURE.__dict__)


def load_features(legacy, args, prepared, encoder, directory, split_sha):
    identity = feature_identity(legacy, args, prepared, split_sha)
    path = Path(directory) / (formal.digest(identity) + ".pt")
    hit = path.exists()
    if hit:
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Frozen feature cache identity mismatch")
        features = saved["features"]
    else:
        features = []
        encoder.eval()
        if any(p.requires_grad for p in encoder.parameters()):
            raise ValueError("Feature precomputation requires a fully frozen encoder")
        with torch.no_grad(), torch.autocast(torch.device(args.device).type, enabled=False):
            for start in range(0, len(prepared), args.micro_batch_size):
                rows = [prepared[i][:3] for i in range(start, min(start + args.micro_batch_size, len(prepared)))]
                batch = legacy.collate_property_batch(rows).to(args.device)
                graph = batch.graph
                nodes = encoder.encode_nodes(graph.node_features, graph.bonds, graph.node_mask)
                features.extend(nodes[i, graph.node_mask[i]].detach().float().cpu().clone() for i in range(len(rows)))
                if start % 512 == 0:
                    print(f"Frozen encoder precompute: {start + len(rows)}/{len(prepared)}", flush=True)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(dict(identity=identity, features=features), temporary)
        temporary.replace(path)
    if len(features) != len(prepared):
        raise ValueError("Frozen feature cache sample count mismatch")
    for index, nodes in enumerate(features):
        expected = (len(prepared[index][3].owner), encoder.config.hidden_dim)
        if tuple(nodes.shape) != expected or nodes.dtype != torch.float32 or not torch.isfinite(nodes).all() or nodes.requires_grad:
            raise ValueError(f"Invalid frozen feature tensor at sample {index}")
    return FrozenDataset(prepared, features), dict(identity=identity, path=str(path), hit=hit,
                                                  file_sha256=legacy.file_identity(path)["sha256"])
