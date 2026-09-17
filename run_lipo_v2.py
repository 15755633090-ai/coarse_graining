"""Train the frozen V2 Region-only and full relation-correction models on Lipo.

This is a new experiment identity.  It does not resume or overwrite any V1
mechanism run.  The default action is a topology-only sanity preparation.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import torch

import run_lipo_formal as formal
from coarse_gnn import (
    TopologyCache,
    V2CoarseGraphPredictor,
    V2CoarseningConfig,
    V2NetworkConfig,
)
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel, precompute_batch_topologies


ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
VARIANTS = ("v2_region_only", "v2_full")
STRUCTURE = V2CoarseningConfig()


class V2FormalModel(DiffusionCoarseModel):
    @property
    def head(self):
        # The legacy benchmark optimizer treats the complete downstream model as
        # its head parameter group.
        return self.predictor

    def forward(self, batch):
        return super().forward(batch).predictions


def make_factory(original_factory, cache):
    def create(variant, num_tasks, checkpoint, device, dropout):
        if variant not in VARIANTS:
            raise ValueError(f"Unknown V2 variant: {variant}")
        original = original_factory(
            "pretrained_finetune", num_tasks, checkpoint, device, dropout,
        )
        network = V2NetworkConfig(
            input_dim=original.encoder.config.hidden_dim,
            hidden_dim=original.encoder.config.hidden_dim,
            output_dim=num_tasks,
            dropout=dropout,
            coarse_layers=0 if variant == "v2_region_only" else 1,
        )
        predictor = V2CoarseGraphPredictor(
            network, STRUCTURE, topology_cache=cache,
        )
        return V2FormalModel(
            original.encoder, predictor, freeze_encoder=False,
        ).to(device)
    return create


def prepare(legacy, args, cache):
    dataset, splits, spec = legacy.build_property_data(args.data_root, "lipo")
    totals = dict(graphs=0, nodes=0, coarse_nodes=0, cut_edges=0, added_centers=0)
    for start in range(0, len(dataset), 32):
        rows = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
        batch = legacy.collate_property_batch(rows).graph
        for topology in precompute_batch_topologies(batch, cache, STRUCTURE):
            totals["graphs"] += 1
            totals["nodes"] += topology.stats["num_nodes"]
            totals["coarse_nodes"] += topology.stats["num_coarse_nodes"]
            totals["cut_edges"] += topology.stats["cut_edge_count"]
            totals["added_centers"] += topology.stats["num_added_centers"]
    report = {
        "stage": "v2_topology_sanity",
        "coarsening": asdict(STRUCTURE),
        "split_counts": {name: len(indices) for name, indices in splits.items()},
        "totals": totals,
        "cache": cache.stats,
        "notice": "Topology-only preparation; no model training or evaluation was run.",
    }
    formal.write_json(args.output_dir / "topology_sanity.json", report)
    return report


def smoke(legacy, args, cache, factory):
    """Run one real-data optimizer step per V2 variant; never enter formal tuning."""
    dataset, splits, spec = legacy.build_property_data(args.data_root, "lipo")
    scaler = legacy.TargetScaler.fit(
        dataset.targets[splits["train"]], dataset.target_mask[splits["train"]],
    )
    rows = [dataset[i] for i in splits["train"][:args.micro_batch_size]]
    batch = legacy.collate_property_batch(rows).to(args.device)
    results = []
    for variant in VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        model = factory(
            variant, spec.num_tasks, args.checkpoint, args.device,
            args.search_candidates[0]["dropout"],
        ).train()
        optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=args.search_candidates[0]["head_lr"],
            weight_decay=args.weight_decay,
        )
        optimizer.zero_grad(set_to_none=True)
        if torch.device(args.device).type == "cuda":
            torch.cuda.reset_peak_memory_stats(torch.device(args.device))
        with torch.autocast(
            device_type=torch.device(args.device).type,
            dtype=torch.bfloat16,
            enabled=args.amp and torch.device(args.device).type == "cuda",
        ):
            predictions = model(batch.graph)
            loss = legacy.property_loss(
                predictions, batch.targets, batch.target_mask, spec, scaler, None,
            )
        loss.backward()
        gradients = [p.grad for p in model.parameters() if p.grad is not None]
        if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
            raise RuntimeError(f"Nonfinite or missing smoke gradients for {variant}")
        torch.nn.utils.clip_grad_norm_(
            [p for p in model.parameters() if p.requires_grad], args.grad_clip,
        )
        optimizer.step()
        results.append({
            "variant": variant,
            "loss": float(loss.detach()),
            "prediction_shape": list(predictions.shape),
            "parameters_with_gradients": len(gradients),
            "peak_cuda_bytes": (
                int(torch.cuda.max_memory_allocated(torch.device(args.device)))
                if torch.device(args.device).type == "cuda" else None
            ),
            "alpha": float(model.predictor.alpha.detach()),
        })
        del optimizer, model
    report = {
        "stage": "v2_one_batch_smoke",
        "encoder_mode": "pretrained_finetune",
        "device": str(args.device),
        "batch_size": len(rows),
        "variants": results,
        "cache": cache.stats,
        "notice": "One optimizer step per variant; not a scientific training result.",
    }
    formal.write_json(args.output_dir / "smoke.json", report)
    return report


def train(legacy, args, old_protocol, cache, factory):
    original_protocol = legacy.diffusion_tuning_protocol
    original_manifest = legacy.benchmark_run_config
    sources = formal.source_identity(legacy)
    sources[Path(__file__).name] = legacy.file_identity(Path(__file__))

    def protocol(*values):
        result = original_protocol(*values)
        result["payload"].update(
            method="v2_exclusive_relation_correction",
            variant=values[-1],
            encoder_mode="pretrained_finetune",
            coarsening=asdict(STRUCTURE),
            network={
                "region_layers": 2,
                "region_edge_features": True,
                "coarse_layers": 0 if values[-1] == "v2_region_only" else 1,
                "coarse_relation_only": True,
                "alpha_init": 0.1,
            },
            new_source_files=sources,
        )
        result["sha256"] = formal.digest(result["payload"])
        return result

    def manifest(values):
        result = original_manifest(values)
        result.update(
            method="v2_exclusive_relation_correction",
            seeds=list(args.seeds),
            encoder_mode="pretrained_finetune",
            coarsening=asdict(STRUCTURE),
            new_source_files=sources,
        )
        return result

    with formal.overrides(
        legacy,
        parse_args=lambda: args,
        create_property_model=factory,
        diffusion_tuning_protocol=protocol,
        benchmark_run_config=manifest,
    ):
        legacy.main()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "smoke", "train"), default="prepare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds) or not set(options.seeds) <= set(range(5)):
        parser.error("Only distinct seeds 0..4 are allowed")

    project = options.project_root.resolve()
    legacy, args, old_protocol, _ = formal.load_protocol(project)
    args.output_dir = (
        options.output_dir
        or project / "model/results_formal/05_coarse_gnn/v2"
    ).resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.seeds = options.seeds
    args.modes = list(VARIANTS)
    cache = TopologyCache(args.output_dir / "_topology_cache", max_memory_entries=8192)
    factory = make_factory(legacy.create_property_model, cache)
    if options.action == "prepare":
        report = prepare(legacy, args, cache)
        print(report["notice"], flush=True)
    elif options.action == "smoke":
        report = smoke(legacy, args, cache, factory)
        print(report["notice"], flush=True)
    else:
        train(legacy, args, old_protocol, cache, factory)


if __name__ == "__main__":
    main()
