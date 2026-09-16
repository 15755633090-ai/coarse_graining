"""Matched C/D models whose only change is the graph readout weighting."""
from dataclasses import replace

import run_lipo_formal as formal
from coarse_gnn.layers import EdgeGIN
from frozen_features import FrozenCoarseModel


TRAIN_VARIANTS = ("region_size_weighted", "base_size_weighted")
REFERENCE_VARIANTS = ("region_only", "base_coarse")
ALL_VARIANTS = TRAIN_VARIANTS + REFERENCE_VARIANTS

REFERENCE_BY_VARIANT = {
    "region_size_weighted": "region_only",
    "base_size_weighted": "base_coarse",
}


def apply_readout_switch(predictor, variant):
    """Derive a reference or weighted model from one initialized Base model."""
    if variant not in ALL_VARIANTS:
        raise ValueError(f"Unknown readout variant: {variant}")

    cfg = predictor.config
    expected = replace(
        formal.network_config("base_coarse", cfg.input_dim, cfg.dropout),
        output_dim=cfg.output_dim,
    )
    if cfg != expected:
        raise ValueError("Readout ablations require an unmodified Base coarse predictor")

    if variant in ("region_only", "region_size_weighted"):
        predictor.coarse_gnn = EdgeGIN(cfg.hidden_dim, 0, 0, cfg.dropout)
        cfg = replace(cfg, coarse_layers=0)
    if variant in TRAIN_VARIANTS:
        cfg = replace(cfg, graph_pool="size_weighted_mean")

    predictor.config = cfg
    return predictor


def make_factory(original_factory, cache):
    """Build every variant from the identical initialized Base mother model."""
    construct = formal.make_factory(original_factory, cache)

    def create(variant, num_tasks, checkpoint, device, dropout):
        if variant not in ALL_VARIANTS:
            raise ValueError(f"Unknown readout variant: {variant}")
        mother = construct("base_coarse", num_tasks, checkpoint, device, dropout)
        predictor = apply_readout_switch(mother.predictor, variant)
        return FrozenCoarseModel(mother.encoder, predictor, freeze_encoder=True).to(device)

    return create
