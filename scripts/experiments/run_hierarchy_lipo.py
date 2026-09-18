"""Frozen-base Lipo orchestration for Base, Full-L1 and Adaptive hierarchy."""
from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
from dataclasses import asdict
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Sampler, Subset

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from coarse_gnn import (
    HierarchicalPredictor,
    HierarchyCache,
    HierarchyConfig,
    HierarchyNetworkConfig,
    pack_hierarchy,
)
from coarse_gnn.diffusion_adapter import (
    encode_nodes_with_intermediates,
)


VARIANTS = ("full_l1", "adaptive")
ALL_GROUPS = ("base", *VARIANTS)
SEEDS = tuple(range(5))
DEFAULT_OUTPUT = formal.ROOT / "outputs/hierarchy_lipo_frozen"
HIERARCHY = HierarchyConfig()


class HierarchyFeatureDataset:
    def __init__(self, source, features, cache):
        self.source, self.features, self.cache = source, features, cache
        self.rows = {}
        self.atom_counts = [int(value.size(0)) for value in features["h2"]]

    def __len__(self):
        return len(self.source)

    def __getattr__(self, name):
        return getattr(self.source, name)

    def __getitem__(self, index):
        if index not in self.rows:
            graph, target, mask = self.source[index]
            pairs = torch.triu(graph.bonds > 0, diagonal=1).nonzero().T.contiguous()
            labels = graph.bonds[pairs[0], pairs[1]]
            topology = self.cache.get_or_build(
                graph.node_features.size(0), pairs, HIERARCHY,
                node_labels=graph.node_features, edge_labels=labels,
            )
            self.rows[index] = (
                graph, target, mask, topology,
                self.features["h2"][index], self.features["hbase"][index],
            )
        return self.rows[index]


class HierarchyGraphBatch:
    def __init__(self, h2, hbase, topologies, plan, full_l1):
        self.h2, self.hbase = h2, hbase
        self.topologies, self.plan, self.full_l1 = topologies, plan, full_l1

    def to(self, device):
        return HierarchyGraphBatch(
            self.h2.to(device, non_blocking=True),
            self.hbase.to(device, non_blocking=True),
            self.topologies, self.plan.to(device), self.full_l1,
        )


class GeneratorBucketBatchSampler(Sampler[list[int]]):
    """Sortish buckets driven by the trainer's checkpointed RNG generator."""
    def __init__(self, counts, batch_size, generator, shuffle, multiplier=20):
        self.counts = torch.as_tensor(counts, dtype=torch.long)
        self.batch_size, self.generator = batch_size, generator
        self.shuffle, self.multiplier = shuffle, multiplier

    def __len__(self):
        return math.ceil(self.counts.numel() / self.batch_size)

    def __iter__(self):
        count = self.counts.numel()
        order = (
            torch.randperm(count, generator=self.generator)
            if self.shuffle else torch.arange(count)
        )
        batches = []
        width = self.batch_size * self.multiplier
        for start in range(0, count, width):
            pool = order[start:start + width]
            pool = pool[torch.argsort(self.counts[pool], stable=True)]
            batches.extend(
                pool[offset:offset + self.batch_size].tolist()
                for offset in range(0, pool.numel(), self.batch_size)
            )
        if self.shuffle and len(batches) > 1:
            permutation = torch.randperm(len(batches), generator=self.generator).tolist()
            batches = [batches[index] for index in permutation]
        yield from batches


def hierarchy_tools(legacy, data, variant):
    full_l1 = variant == "full_l1"

    def collate(rows):
        batch = legacy.collate_property_batch([row[:3] for row in rows])
        padded = batch.graph.node_features.size(1)
        h2 = torch.zeros((len(rows), padded, rows[0][4].size(1)), dtype=torch.float32)
        for index, row in enumerate(rows):
            h2[index, :row[4].size(0)] = row[4]
        topologies = [row[3] for row in rows]
        batch.graph = HierarchyGraphBatch(
            h2, torch.stack([row[5] for row in rows]), topologies,
            pack_hierarchy(topologies, padded, full_l1=full_l1), full_l1,
        )
        return batch

    def slice_batch(batch, start, end):
        if start == 0 and end == batch.targets.size(0):
            return batch
        graph = batch.graph
        topologies = graph.topologies[start:end]
        padded = max(topology.atom_order.numel() for topology in topologies)
        return type(batch)(
            HierarchyGraphBatch(
                graph.h2[start:end, :padded], graph.hbase[start:end], topologies,
                pack_hierarchy(topologies, padded, full_l1=full_l1), full_l1,
            ),
            batch.targets[start:end], batch.target_mask[start:end],
        )

    def make_loader(dataset, indices, batch_size, shuffle, num_workers, device, seed):
        generator = torch.Generator().manual_seed(seed)
        subset = Subset(dataset, list(indices))
        counts = [dataset.atom_counts[index] for index in indices]
        sampler = GeneratorBucketBatchSampler(counts, batch_size, generator, shuffle)
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


class FrozenHierarchyModel(nn.Module):
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


class HierarchyFactory:
    def __init__(self, legacy, baseline, hidden_dim):
        self.legacy, self.baseline, self.hidden_dim = legacy, baseline, hidden_dim
        self.seed = None

    def __call__(self, variant, num_tasks, checkpoint, device, dropout):
        if variant not in VARIANTS or self.seed not in SEEDS:
            raise ValueError("Hierarchy factory requires a variant and active seed")
        base = self.legacy.create_property_model(
            "pretrained_frozen", num_tasks, checkpoint, device, dropout,
        )
        saved = torch.load(
            self.baseline / f"lipo/pretrained_frozen/seed_{self.seed}/best.pt",
            map_location=device, weights_only=False,
        )
        base.load_state_dict(saved["model_state"])
        predictor = HierarchicalPredictor(HierarchyNetworkConfig(
            input_dim=base.encoder.config.hidden_dim,
            base_dim=2 * base.encoder.config.hidden_dim,
            hidden_dim=self.hidden_dim,
            output_dim=num_tasks,
            dropout=dropout,
        ))
        if variant == "full_l1":
            # These modules cannot affect an all-L1 context. Freeze them so
            # reported trainable capacity matches the actual baseline.
            predictor.parent_mlps.requires_grad_(False)
            predictor.gines.requires_grad_(False)
            predictor.level_projections[1:].requires_grad_(False)
        return FrozenHierarchyModel(base, predictor).to(device)


def feature_identity(legacy, args, split_sha):
    return dict(
        version=1, split_sha256=split_sha,
        checkpoint=legacy.file_identity(args.checkpoint),
        layer=2, precision="float32", encoder_mode="frozen_eval",
        torch_version=str(torch.__version__),
    )


def load_or_extract_features(legacy, args, source, split_sha, directory):
    identity = feature_identity(legacy, args, split_sha)
    path = Path(directory) / (formal.digest(identity) + ".pt")
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=True)
        if saved["identity"] != identity:
            raise ValueError("Hierarchy feature cache identity mismatch")
        return saved, dict(
            identity=identity, path=str(path), hit=True,
            file_sha256=legacy.file_identity(path)["sha256"],
        )

    model = legacy.create_property_model(
        "pretrained_frozen", 1, args.checkpoint, torch.device(args.device), args.dropout,
    ).eval()
    model.requires_grad_(False)
    h2_rows, hbase_rows = [], []
    with torch.no_grad(), torch.autocast(torch.device(args.device).type, enabled=False):
        for start in range(0, len(source), 64):
            rows = [source[index] for index in range(start, min(start + 64, len(source)))]
            graph = legacy.collate_property_batch(rows).graph.to(args.device)
            final, captured = encode_nodes_with_intermediates(
                model.encoder, graph.node_features, graph.bonds, graph.node_mask, layers=(2,),
            )
            hbase = model.readout(final, graph.node_mask)
            for index in range(len(rows)):
                count = int(graph.node_mask[index].sum())
                h2_rows.append(captured[2][index, :count].float().cpu().clone())
                hbase_rows.append(hbase[index].float().cpu().clone())
            if start % 512 == 0:
                print(f"Hierarchy frozen features: {min(start + len(rows), len(source))}/{len(source)}", flush=True)
    saved = dict(identity=identity, h2=h2_rows, hbase=hbase_rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    torch.save(saved, temporary)
    temporary.replace(path)
    return saved, dict(
        identity=identity, path=str(path), hit=False,
        file_sha256=legacy.file_identity(path)["sha256"],
    )


def summarize(output, baseline, seeds):
    rows = []
    for seed in seeds:
        base_path = baseline / f"lipo/pretrained_frozen/seed_{seed}/result.json"
        base = formal.read_json(base_path)
        rows.append(dict(
            mode="base", seed=seed,
            validation_rmse=base["validation_metrics"]["rmse"],
            best_epoch=base["best_epoch"], source=str(base_path.resolve()),
        ))
        for variant in VARIANTS:
            path = output / "lipo" / variant / f"seed_{seed}" / "result.json"
            if path.exists():
                result = formal.read_json(path)
                rows.append(dict(
                    mode=variant, seed=seed,
                    validation_rmse=result["validation_metrics"]["rmse"],
                    best_epoch=result["best_epoch"], source=str(path.resolve()),
                ))
    summary = []
    for mode in ALL_GROUPS:
        values = [row["validation_rmse"] for row in rows if row["mode"] == mode]
        if values:
            summary.append(dict(
                mode=mode, n=len(values), mean_rmse=statistics.mean(values),
                std_rmse=statistics.stdev(values) if len(values) > 1 else None,
            ))
    lookup = {(row["mode"], row["seed"]): row["validation_rmse"] for row in rows}
    paired = []
    for before, after in (("base", "full_l1"), ("base", "adaptive"), ("full_l1", "adaptive")):
        deltas = [
            dict(seed=seed, delta_rmse=lookup[after, seed] - lookup[before, seed])
            for seed in seeds if (before, seed) in lookup and (after, seed) in lookup
        ]
        if deltas:
            values = [row["delta_rmse"] for row in deltas]
            paired.append(dict(
                before=before, after=after, per_seed=deltas,
                mean_delta_rmse=statistics.mean(values),
                std_delta_rmse=statistics.stdev(values) if len(values) > 1 else None,
            ))
    report = dict(
        metric_split="validation", test_metrics_used=False, runs=rows,
        summary=summary, paired=paired,
        complete=all((mode, seed) in lookup for mode in ALL_GROUPS for seed in seeds),
    )
    formal.write_json(output / "comparison_validation.json", report)
    if rows:
        with (output / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    return report


def preflight(legacy, args, data, spec, factory):
    indices = list(range(min(args.batch_size, len(data))))
    states, reports = {}, []
    for variant in VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        factory.seed = 0
        model = factory(variant, spec.num_tasks, args.checkpoint, args.device, args.dropout).train()
        tools = hierarchy_tools(legacy, data, variant)
        batch = tools["collate_property_batch"]([data[index] for index in indices]).to(args.device)
        if not reports:
            raw = legacy.collate_property_batch([data[index][:3] for index in indices]).graph.to(args.device)
            with torch.no_grad():
                final = model.encoder.encode_nodes(raw.node_features, raw.bonds, raw.node_mask)
                weights = raw.node_mask.unsqueeze(-1).to(final.dtype)
                summed = (final * weights).sum(1)
                direct_hbase = torch.cat((summed, summed / weights.sum(1).clamp_min(1)), dim=1)
                direct_base = model.base_head(direct_hbase)
                cached_base = model.base_head(batch.graph.hbase)
            # Dense padded kernels can differ slightly with extraction batch
            # shape; this matches the established frozen-feature tolerance.
            torch.testing.assert_close(direct_hbase, batch.graph.hbase, atol=2e-5, rtol=2e-5)
            torch.testing.assert_close(direct_base, cached_base, atol=2e-5, rtol=2e-5)
        output = model(batch.graph)
        output.square().mean().backward()
        if not torch.isfinite(output).all():
            raise AssertionError(f"Nonfinite hierarchy preflight: {variant}")
        if any(parameter.grad is not None for parameter in model.encoder.parameters()):
            raise AssertionError("Frozen base encoder received gradients")
        if any(parameter.grad is not None for parameter in model.base_head.parameters()):
            raise AssertionError("Frozen base prediction head received gradients")
        if not any(parameter.grad is not None for parameter in model.head.parameters()):
            raise AssertionError("Hierarchy branch received no gradients")
        state = {key: value.detach().cpu() for key, value in model.head.state_dict().items()}
        if states:
            for key, value in next(iter(states.values())).items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
        states[variant] = state
        reports.append(dict(
            variant=variant, prediction_shape=list(output.shape),
            contexts=int(batch.graph.plan.contexts.context_indices.numel()),
            only_l1=bool((batch.graph.plan.contexts.context_levels == 1).all()),
            trainable_parameters=sum(
                parameter.numel() for parameter in model.parameters()
                if parameter.requires_grad
            ),
        ))
    if not reports[0]["only_l1"]:
        raise AssertionError("Full-L1 preflight used a higher-level context")
    return dict(passed=True, variants=reports, base_frozen=True,
                identical_hierarchy_initialization=True, test_metrics_used=False)


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
        parser.error("hidden-dim must be positive")
    if options.batch_size is not None and options.batch_size < 1:
        parser.error("batch-size must be positive")
    if options.micro_batch_size is not None and options.micro_batch_size < 1:
        parser.error("micro-batch-size must be positive")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    legacy, args, protocol, baseline, selection = frozen.load_protocol(formal.PROJECT)
    output = options.output_dir.resolve()
    args.output_dir = output
    args.encoder_lr = 0.0
    if options.batch_size is not None:
        args.batch_size = options.batch_size
    args.micro_batch_size = options.micro_batch_size or args.batch_size
    args.seeds = options.seeds
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if options.action == "summarize":
        summarize(output, baseline, options.seeds)
        return

    source, splits, spec, inputs = formal.check_inputs(legacy, args, protocol, baseline)
    features, feature_report = load_or_extract_features(
        legacy, args, source, protocol["split_sha256"], output / "_feature_cache",
    )
    cache = HierarchyCache(output / "_hierarchy_cache", max_memory_entries=8192)
    data = HierarchyFeatureDataset(source, features, cache)
    for index in range(len(data)):
        data[index]
        if index % 512 == 0:
            print(f"Hierarchy topology: {index + 1}/{len(data)}", flush=True)

    factory = HierarchyFactory(legacy, baseline, options.hidden_dim)
    base_files = {
        str(seed): legacy.file_identity(
            baseline / f"lipo/pretrained_frozen/seed_{seed}/best.pt"
        ) for seed in options.seeds
    }
    config = dict(
        stage="hierarchy_lipo_frozen_v1", schema_version=1,
        groups=list(ALL_GROUPS), train_variants=list(VARIANTS), seeds=options.seeds,
        hierarchy=asdict(HIERARCHY),
        network=asdict(HierarchyNetworkConfig(
            input_dim=128, base_dim=256, hidden_dim=options.hidden_dim,
            output_dim=spec.num_tasks, dropout=args.dropout,
        )),
        frozen_base_checkpoints=base_files,
        feature_cache={
            key: feature_report[key] for key in ("identity", "path", "file_sha256")
        },
        split_sha256=protocol["split_sha256"], inputs=inputs,
        training=dict(
            batch_size=args.batch_size, micro_batch_size=args.micro_batch_size,
            head_lr=args.head_lr, weight_decay=args.weight_decay,
            epochs=args.epochs, patience=args.patience, amp=args.amp,
            bucket_batching=True, test_metrics_used=False,
        ),
        source_files={
            **formal.source_identity(legacy),
            str(Path(__file__).relative_to(formal.ROOT)): legacy.file_identity(Path(__file__)),
        },
    )
    legacy.initialize_or_validate_run_config(output, config)
    report = preflight(legacy, args, data, spec, factory)
    formal.write_json(output / "preflight.json", report)
    print(f"prepare passed; feature_cache_hit={feature_report['hit']}; cache={cache.stats}", flush=True)

    if options.action == "train":
        for seed in options.seeds:
            factory.seed = seed
            for variant in VARIANTS:
                if variant not in options.variants:
                    continue
                tools = hierarchy_tools(legacy, data, variant)
                run_args = copy.copy(args)
                payload = dict(config, mode=variant, seed=seed)
                run_args.tuning_protocol = dict(payload=payload, sha256=formal.digest(payload))
                directory = output / "lipo" / variant / f"seed_{seed}"
                if (directory / "result.json").exists():
                    print(f"Completed already: {variant} seed={seed}", flush=True)
                    continue
                print(f"Start {variant} seed={seed}; frozen Base, validation-only", flush=True)
                with formal.overrides(
                    legacy, create_property_model=factory, **tools,
                ):
                    legacy.run_single(
                        run_args, data, splits, spec, variant, seed,
                        torch.device(args.device), directory, evaluate_test=False,
                    )
                summarize(output, baseline, options.seeds)
    summarize(output, baseline, options.seeds)
    print(f"{options.action} finished: {output}", flush=True)


if __name__ == "__main__":
    main()
