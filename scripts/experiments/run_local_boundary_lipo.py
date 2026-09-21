"""Frozen-base Lipo orchestration for four packed LocalBoundary edge schedules."""
from __future__ import annotations

import argparse
import copy
import csv
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from coarse_gnn import (
    LOCAL_BOUNDARY_VARIANTS,
    LocalBoundaryNetworkConfig,
    LocalBoundaryPredictor,
    pack_local_boundary,
)
from scripts.experiments import run_hierarchy_lipo as hierarchy_entry


VARIANTS = LOCAL_BOUNDARY_VARIANTS
ALL_GROUPS = ("base", *VARIANTS)
SEEDS = tuple(range(5))
DEFAULT_OUTPUT = formal.ROOT / "outputs/local_boundary_lipo_frozen"


class LocalBoundaryGraphBatch:
    def __init__(self, h2, hbase, topologies, plan):
        self.h2, self.hbase = h2, hbase
        self.topologies, self.plan = topologies, plan

    def to(self, device):
        return type(self)(
            self.h2.to(device, non_blocking=True),
            self.hbase.to(device, non_blocking=True),
            self.topologies, self.plan.to(device),
        )


def local_boundary_tools(legacy, data):
    base_collate = legacy.collate_property_batch

    def collate(rows):
        batch = base_collate([row[:3] for row in rows])
        padded = batch.graph.node_features.size(1)
        h2 = torch.zeros((len(rows), padded, rows[0][4].size(1)), dtype=torch.float32)
        for index, row in enumerate(rows):
            h2[index, :row[4].size(0)] = row[4]
        topologies = [row[3] for row in rows]
        batch.graph = LocalBoundaryGraphBatch(
            h2, torch.stack([row[5] for row in rows]), topologies,
            pack_local_boundary(topologies, padded),
        )
        return batch

    def slice_batch(batch, start, end):
        if start == 0 and end == batch.targets.size(0):
            return batch
        graph = batch.graph
        topologies = graph.topologies[start:end]
        padded = max(topology.atom_order.numel() for topology in topologies)
        return type(batch)(
            LocalBoundaryGraphBatch(
                graph.h2[start:end, :padded], graph.hbase[start:end], topologies,
                pack_local_boundary(topologies, padded),
            ),
            batch.targets[start:end], batch.target_mask[start:end],
        )

    def make_loader(dataset, indices, batch_size, shuffle, num_workers, device, seed):
        generator = torch.Generator().manual_seed(seed)
        subset = Subset(dataset, list(indices))
        counts = [dataset.atom_counts[index] for index in indices]
        sampler = hierarchy_entry.GeneratorBucketBatchSampler(
            counts, batch_size, generator, shuffle,
        )
        options = dict(
            dataset=subset, batch_sampler=sampler, num_workers=num_workers,
            collate_fn=collate, generator=generator, pin_memory=False,
        )
        if num_workers > 0:
            options.update(persistent_workers=True, prefetch_factor=2)
        return DataLoader(**options)

    return dict(
        collate_property_batch=collate,
        _slice_property_batch=slice_batch,
        make_loader=make_loader,
    )


class FrozenLocalBoundaryModel(nn.Module):
    def __init__(self, base_model, predictor):
        super().__init__()
        self.encoder = base_model.encoder
        self.base_head = base_model.head
        self.head = predictor
        self.encoder.requires_grad_(False).eval()
        self.base_head.requires_grad_(False).eval()

    def train(self, mode=True):
        super().train(mode)
        self.encoder.eval()
        self.base_head.eval()
        self.head.train(mode)
        return self

    def forward(self, batch):
        with torch.no_grad():
            base_prediction = self.base_head(batch.hbase)
        return self.head(batch.h2, batch.hbase, base_prediction, batch.plan).prediction


class LocalBoundaryFactory:
    def __init__(self, legacy, baseline, hidden_dim):
        self.create_base_model = legacy.create_property_model
        self.baseline, self.hidden_dim, self.seed = baseline, hidden_dim, None

    def __call__(self, variant, num_tasks, checkpoint, device, dropout):
        if variant not in VARIANTS or self.seed not in SEEDS:
            raise ValueError("LocalBoundary factory requires a variant and active seed")
        base = self.create_base_model(
            "pretrained_frozen", num_tasks, checkpoint, device, dropout,
        )
        saved = torch.load(
            self.baseline / f"lipo/pretrained_frozen/seed_{self.seed}/best.pt",
            map_location=device, weights_only=False,
        )
        base.load_state_dict(saved["model_state"])
        predictor = LocalBoundaryPredictor(LocalBoundaryNetworkConfig(
            input_dim=base.encoder.config.hidden_dim,
            base_dim=2 * base.encoder.config.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_tasks,
            dropout=dropout,
        ), variant)
        return FrozenLocalBoundaryModel(base, predictor).to(device)


def preflight(legacy, args, data, spec, factory):
    boundary_index = next((
        index for index in range(len(data))
        if pack_local_boundary(
            [data[index][3]], data[index][0].node_features.size(0),
        ).boundary_edges.numel()
    ), None)
    if boundary_index is None:
        raise AssertionError("No real Lipo sample contains an EB edge")
    indices = [boundary_index] + [
        index for index in range(len(data)) if index != boundary_index
    ][:max(0, min(args.batch_size, len(data)) - 1)]
    tools = local_boundary_tools(legacy, data)
    batch = tools["collate_property_batch"]([data[index] for index in indices]).to(args.device)
    states, reports = [], []
    for variant in VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        factory.seed = 0
        model = factory(variant, spec.num_tasks, args.checkpoint, args.device, args.dropout).train()
        output = model(batch.graph)
        output.square().mean().backward()
        if not torch.isfinite(output).all():
            raise AssertionError(f"Nonfinite LocalBoundary preflight: {variant}")
        if any(parameter.grad is not None for parameter in model.encoder.parameters()):
            raise AssertionError("Frozen base encoder received gradients")
        if any(parameter.grad is not None for parameter in model.base_head.parameters()):
            raise AssertionError("Frozen base prediction head received gradients")
        if not any(parameter.grad is not None for parameter in model.head.parameters()):
            raise AssertionError("LocalBoundary branch received no gradients")
        edge_gradients = []
        mlp_gradients = []
        for gine in model.head.gines:
            layer = gine.layers[0]
            edge_gradients.append(
                layer.edge_projection.weight.grad is not None
                and bool((layer.edge_projection.weight.grad.abs() > 0).any())
            )
            mlp_gradients.append(any(
                parameter.grad is not None and bool((parameter.grad.abs() > 0).any())
                for parameter in layer.mlp.parameters()
            ))
        expected_edge_gradients = {
            "self_only": [False, False],
            "local_only": [True, False],
            "local_boundary": [True, True],
            "full": [True, True],
        }[variant]
        if edge_gradients != expected_edge_gradients or mlp_gradients != [True, True]:
            raise AssertionError(f"Unexpected GINE gradient path for {variant}")
        state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
        if states:
            for key, value in states[0].items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        states.append(state)
        reports.append({
            "variant": variant,
            "prediction_shape": list(output.shape),
            "full_edges": batch.graph.plan.full_edges.size(1),
            "local_edges": batch.graph.plan.local_edges.size(1),
            "boundary_edges": batch.graph.plan.boundary_edges.size(1),
            "edge_projection_gradients": edge_gradients,
            "self_mlp_gradients": mlp_gradients,
            "trainable_parameters": sum(
                parameter.numel() for parameter in model.parameters() if parameter.requires_grad
            ),
        })
    return {
        "passed": True,
        "variants": reports,
        "base_frozen": True,
        "identical_initialization": True,
        "edge_attribute_dim": 6,
        "test_metrics_used": False,
    }


def summarize(output, baseline, seeds):
    rows = []
    for seed in seeds:
        base_path = baseline / f"lipo/pretrained_frozen/seed_{seed}/result.json"
        base = formal.read_json(base_path)
        rows.append({
            "mode": "base", "seed": seed,
            "validation_rmse": base["validation_metrics"]["rmse"],
            "best_epoch": base["best_epoch"], "source": str(base_path.resolve()),
        })
        for variant in VARIANTS:
            path = output / "lipo" / variant / f"seed_{seed}" / "result.json"
            if path.exists():
                result = formal.read_json(path)
                rows.append({
                    "mode": variant, "seed": seed,
                    "validation_rmse": result["validation_metrics"]["rmse"],
                    "best_epoch": result["best_epoch"], "source": str(path.resolve()),
                })
    summary = []
    for mode in ALL_GROUPS:
        values = [row["validation_rmse"] for row in rows if row["mode"] == mode]
        if values:
            summary.append({
                "mode": mode, "n": len(values), "mean_rmse": statistics.mean(values),
                "std_rmse": statistics.stdev(values) if len(values) > 1 else None,
            })
    lookup = {(row["mode"], row["seed"]): row["validation_rmse"] for row in rows}
    paired = []
    for before, after in (
        ("base", "self_only"), ("self_only", "local_only"),
        ("local_only", "local_boundary"), ("local_boundary", "full"),
    ):
        deltas = [
            {"seed": seed, "delta_rmse": lookup[after, seed] - lookup[before, seed]}
            for seed in seeds if (before, seed) in lookup and (after, seed) in lookup
        ]
        if deltas:
            values = [row["delta_rmse"] for row in deltas]
            paired.append({
                "before": before, "after": after, "per_seed": deltas,
                "mean_delta_rmse": statistics.mean(values),
                "std_delta_rmse": statistics.stdev(values) if len(values) > 1 else None,
            })
    report = {
        "metric_split": "validation", "test_metrics_used": False,
        "runs": rows, "summary": summary, "paired": paired,
        "complete": all(
            any(row["mode"] == mode and row["seed"] == seed for row in rows)
            for mode in ALL_GROUPS for seed in seeds
        ),
    }
    formal.write_json(output / "comparison_validation.json", report)
    if rows:
        with (output / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train", "summarize"), default="prepare")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--micro-batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int, default=128)
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds) or not set(options.seeds) <= set(SEEDS):
        parser.error("Use distinct seeds from 0..4")
    if len(set(options.variants)) != len(options.variants):
        parser.error("Variants must be distinct")
    if options.hidden_dim < 1:
        parser.error("--hidden-dim must be positive")
    if options.batch_size is not None and options.batch_size < 1:
        parser.error("--batch-size must be positive")
    if options.micro_batch_size is not None and options.micro_batch_size < 1:
        parser.error("--micro-batch-size must be positive")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    legacy, args, protocol, baseline, selection = frozen.load_protocol(formal.PROJECT)
    output = options.output_dir.resolve()
    args.output_dir, args.encoder_lr = output, 0.0
    if options.batch_size is not None:
        args.batch_size = options.batch_size
    args.micro_batch_size = options.micro_batch_size or args.batch_size
    args.seeds = options.seeds
    output.mkdir(parents=True, exist_ok=True)
    if options.action == "summarize":
        summarize(output, baseline, options.seeds)
        return

    source, splits, spec, inputs = formal.check_inputs(legacy, args, protocol, baseline)
    encoder_verification = hierarchy_entry.verify_all_seed_encoders(legacy, args, baseline)
    features, feature_report = hierarchy_entry.load_or_extract_features(
        legacy, args, source, protocol["split_sha256"], protocol,
        hierarchy_entry.DEFAULT_OUTPUT / "_feature_cache",
    )
    cache = hierarchy_entry.HierarchyCache(
        hierarchy_entry.DEFAULT_OUTPUT / "_hierarchy_cache", max_memory_entries=8192,
    )
    data = hierarchy_entry.HierarchyFeatureDataset(source, features, cache)
    for index in range(len(data)):
        data[index]
        if index % 512 == 0:
            print(f"LocalBoundary topology: {index + 1}/{len(data)}", flush=True)

    factory = LocalBoundaryFactory(legacy, baseline, options.hidden_dim)
    feature_dim = feature_report["identity"]["hidden_dim"]
    config = {
        "stage": "local_boundary_lipo_frozen_v1", "schema_version": 1,
        "groups": list(ALL_GROUPS), "train_variants": list(VARIANTS), "seeds": options.seeds,
        "edge_schedule": {
            "self_only": ["empty", "empty"],
            "local_only": ["EA", "empty"],
            "local_boundary": ["EA", "EB"],
            "full": ["E", "E"],
        },
        "network": asdict(LocalBoundaryNetworkConfig(
            input_dim=feature_dim, base_dim=2 * feature_dim,
            hidden_dim=options.hidden_dim, output_dim=spec.num_tasks, dropout=args.dropout,
        )),
        "edge_features": ["log1p_boundary_count", "bond_type_proportions_1_to_4", "density"],
        "encoder_verification": encoder_verification,
        "feature_cache": {
            key: feature_report[key] for key in ("identity", "path", "file_sha256")
        },
        "split_sha256": protocol["split_sha256"], "inputs": inputs,
        "frozen_selection_file": selection,
        "training": {
            "batch_size": args.batch_size, "micro_batch_size": args.micro_batch_size,
            "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "epochs": args.epochs, "patience": args.patience, "amp": args.amp,
            "num_workers": args.num_workers, "bucket_batching": True,
            "test_metrics_used": False,
        },
        "source_files": {
            **formal.source_identity(legacy),
            "run_lipo_frozen.py": legacy.file_identity(formal.ROOT / "run_lipo_frozen.py"),
            str(Path(__file__).relative_to(formal.ROOT)): legacy.file_identity(Path(__file__)),
            "coarse_gnn/local_boundary.py": legacy.file_identity(formal.ROOT / "coarse_gnn/local_boundary.py"),
            "coarse_gnn/local_boundary_packed.py": legacy.file_identity(formal.ROOT / "coarse_gnn/local_boundary_packed.py"),
            "coarse_gnn/local_boundary_model.py": legacy.file_identity(formal.ROOT / "coarse_gnn/local_boundary_model.py"),
        },
    }
    legacy.initialize_or_validate_run_config(output, config)
    report = preflight(legacy, args, data, spec, factory)
    formal.write_json(output / "preflight.json", report)
    print(f"prepare passed; feature_cache_hit={feature_report['hit']}; cache={cache.stats}", flush=True)

    if options.action == "train":
        tools = local_boundary_tools(legacy, data)
        for seed in options.seeds:
            factory.seed = seed
            for variant in VARIANTS:
                if variant not in options.variants:
                    continue
                run_args = copy.copy(args)
                payload = dict(config, mode=variant, seed=seed)
                run_args.tuning_protocol = {"payload": payload, "sha256": formal.digest(payload)}
                directory = output / "lipo" / variant / f"seed_{seed}"
                if (directory / "result.json").exists():
                    print(f"Completed already: {variant} seed={seed}", flush=True)
                    continue
                print(f"Start {variant} seed={seed}; frozen Base, validation-only", flush=True)
                with formal.overrides(legacy, create_property_model=factory, **tools):
                    legacy.run_single(
                        run_args, data, splits, spec, variant, seed,
                        torch.device(args.device), directory, evaluate_test=False,
                    )
                summarize(output, baseline, options.seeds)
    summarize(output, baseline, options.seeds)
    print(f"{options.action} finished: {output}", flush=True)


if __name__ == "__main__":
    main()
