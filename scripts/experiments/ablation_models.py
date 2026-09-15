"""First-round switches on the existing Frozen Base coarse mother model."""
from dataclasses import replace

import torch

import run_lipo_formal as formal
from frozen_features import FrozenCoarseModel, FrozenGraphBatch
from coarse_gnn.layers import EdgeGIN
from coarse_gnn.packed import segment_pool


# B-weighted is a check only; C/D are read from the existing experiment.
TRAIN_VARIANTS = ("atom", "direct_mean", "no_edge", "deeper_region", "core_only")
CHECK_VARIANTS = ("direct_weighted", "region_only", "base_coarse")


def apply_switch(predictor, variant):
    """Change only the requested modules, without drawing additional randomness."""
    if variant not in TRAIN_VARIANTS + CHECK_VARIANTS:
        raise ValueError(f"Unknown first-round variant: {variant}")
    cfg = predictor.config
    expected = replace(formal.network_config("base_coarse", cfg.input_dim, cfg.dropout),
                       output_dim=cfg.output_dim)
    if cfg != expected:
        raise ValueError("Ablations require an unmodified Base coarse predictor")
    if variant in ("atom", "direct_mean", "direct_weighted"):
        predictor.region_gnn = EdgeGIN(cfg.hidden_dim, 0, 0, cfg.dropout)
        predictor.coarse_gnn = EdgeGIN(cfg.hidden_dim, 0, 0, cfg.dropout)
        cfg = replace(cfg, region_layers=0, coarse_layers=0,
                      graph_pool="size_weighted_mean" if variant == "direct_weighted" else "mean")
    elif variant == "region_only":
        predictor.coarse_gnn = EdgeGIN(cfg.hidden_dim, 0, 0, cfg.dropout)
        cfg = replace(cfg, coarse_layers=0)
    elif variant == "deeper_region":
        # These are identical edge-free GIN layer types. Relocate the exact
        # initialized layers, preserving parameter count and the post-factory RNG.
        predictor.region_gnn.layers.extend(predictor.coarse_gnn.layers)
        predictor.coarse_gnn = EdgeGIN(cfg.hidden_dim, 0, 0, cfg.dropout)
        cfg = replace(cfg, region_layers=5, coarse_layers=0)
    predictor.config = cfg
    return predictor


class AblationModel(FrozenCoarseModel):
    def __init__(self, encoder, predictor, variant):
        super().__init__(encoder, predictor, freeze_encoder=True)
        self.variant = variant

    def forward(self, batch):
        if not isinstance(batch, FrozenGraphBatch):
            raise ValueError("First-round ablations require the shared frozen feature cache")
        if not self.freeze_encoder or self.encoder.training:
            raise RuntimeError("Encoder must remain frozen and in eval mode")
        if self.variant == "core_only" and any(
            not torch.equal(core, context)
            for topology in batch.topologies for core, context in zip(topology.cores, topology.contexts)
        ):
            raise ValueError("core_only requires the CoreOnlyDataset context view")
        if self.variant == "atom":
            # Same input projection and prediction head as B/C/D. Atom lengths
            # exclude padding; atom_gather applies the same canonical ordering.
            nodes = batch.node_states.flatten(0, 1)[batch.plan.atom_gather]
            projected = self.predictor.input_projection(nodes)
            lengths = batch.node_mask.sum(1)
            return self.predictor.head(segment_pool(projected, lengths, "mean"))
        if self.variant == "no_edge":
            plan = batch.plan
            plan = replace(plan, coarse_source=plan.coarse_source[:0],
                           coarse_target=plan.coarse_target[:0],
                           coarse_counts=plan.coarse_counts[:0],
                           coarse_bond_sums=plan.coarse_bond_sums[:0])
            batch = FrozenGraphBatch(batch.graph, batch.topologies, batch.node_states, plan)
        return super().forward(batch)


def make_factory(original_factory, cache):
    construct = formal.make_factory(original_factory, cache)

    def create(variant, num_tasks, checkpoint, device, dropout):
        if variant not in TRAIN_VARIANTS + CHECK_VARIANTS:
            raise ValueError(f"Unknown first-round variant: {variant}")
        mother = construct("base_coarse", num_tasks, checkpoint, device, dropout)
        predictor = apply_switch(mother.predictor, variant)
        return AblationModel(mother.encoder, predictor, variant).to(device)

    return create


def core_only_topology(topology):
    """Keep owners, cores, centers and coarse edges; restrict local context only."""
    contexts, local_edges, edge_ids, positions = [], [], [], []
    for core in topology.cores:
        inverse = torch.full_like(topology.owner, -1)
        inverse[core] = torch.arange(len(core))
        remapped = inverse[topology.edges]
        keep = (remapped >= 0).all(0)
        contexts.append(core)
        local_edges.append(remapped[:, keep])
        edge_ids.append(torch.where(keep)[0])
        positions.append(torch.arange(len(core)))
    return replace(topology, contexts=contexts, context_edges=local_edges,
                   context_edge_ids=edge_ids, core_positions=positions)


class CoreOnlyDataset:
    """A separate context view; the shared cached topology stays immutable."""
    def __init__(self, source):
        self.source = source
        self.rows = {}

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __len__(self):
        return len(self.source)

    def __getitem__(self, index):
        if index not in self.rows:
            row = self.source[index]
            self.rows[index] = (*row[:3], core_only_topology(row[3]), row[4])
        return self.rows[index]
