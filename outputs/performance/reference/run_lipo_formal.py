"""Add coarse models to the EXISTING Lipo protocol; never run old baselines.

The original benchmark.main/run_single, data utilities and metric code execute
unchanged. Scoped adapters replace only the model factory and extend provenance.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib
import json
import os
import sys
from contextlib import contextmanager
from dataclasses import asdict, replace
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig, TopologyCache
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel, precompute_batch_topologies
from coarse_gnn.layers import EdgeGIN

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
VARIANTS = ("region_only", "base_coarse", "enhanced_coarse")
STRUCTURE = CoarseningConfig(radius=4, center_fraction=0.1, max_residual_size=4, canonicalize=True)


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@contextmanager
def overrides(module, **values):
    previous = {key: getattr(module, key) for key in values}
    try:
        for key, value in values.items():
            setattr(module, key, value)
        yield
    finally:
        for key, value in previous.items():
            setattr(module, key, value)


class FormalCoarsePredictor(DiffusionCoarseModel):
    @property
    def head(self):
        # The legacy optimizer has exactly two groups: encoder and head.
        # All downstream layers must be in head, not just the final MLP.
        return self.predictor

    def forward(self, batch):
        return super().forward(batch).predictions


def network_config(variant, hidden_dim, dropout):
    if variant not in VARIANTS:
        raise ValueError(f"Only new coarse variants are allowed: {variant}")
    kwargs = dict(input_dim=hidden_dim, hidden_dim=hidden_dim, edge_dim=4,
                  region_layers=2, coarse_layers=3, dropout=dropout)
    if variant == "enhanced_coarse":
        return NetworkConfig(**kwargs)
    config = NetworkConfig.base(**kwargs)
    return replace(config, coarse_layers=0) if variant == "region_only" else config


def make_factory(original_factory, cache):
    def create(variant, num_tasks, pretrained_checkpoint, device, dropout):
        if variant not in VARIANTS:
            raise ValueError("This entry point cannot train existing baseline models")
        # Use the original encoder implementation AND its original checkpoint loader.
        original = original_factory("pretrained_finetune", num_tasks, pretrained_checkpoint, device, dropout)
        cfg = replace(network_config(variant, original.encoder.config.hidden_dim, dropout), output_dim=num_tasks)
        # Initialize shared region/head weights identically for a given seed.
        # Region-only then removes only the coarse message-passing layers.
        construction = replace(cfg, coarse_layers=3) if variant == "region_only" else cfg
        predictor = CoarseGraphPredictor(construction, STRUCTURE, topology_cache=cache)
        if variant == "region_only":
            predictor.coarse_gnn = EdgeGIN(cfg.hidden_dim, cfg.coarse_edge_dim, 0, cfg.dropout)
            predictor.config = cfg
        return FormalCoarsePredictor(original.encoder, predictor, freeze_encoder=False).to(device)
    return create


def load_protocol(project):
    baseline_root = project / "model/results_formal/04_diffusion"
    root_config = read_json(baseline_root / "run_config.json")
    selection = read_json(baseline_root / "lipo/pretrained_finetune/selected_config.json")
    protocol = selection["tuning_protocol"]
    legacy_root = Path(protocol["source_files"]["downstream_benchmark.py"]["path"]).parent
    sys.path.insert(0, str(legacy_root))
    legacy = importlib.import_module("downstream_benchmark")
    if Path(legacy.__file__).resolve().parent != legacy_root.resolve():
        raise RuntimeError("A different downstream_benchmark module was already imported")
    for item in protocol["source_files"].values():
        if legacy.file_identity(Path(item["path"]))["sha256"] != item["sha256"]:
            raise RuntimeError(f"Original formal source changed: {item['path']}")
    # parse_args supplies original defaults not recorded in old manifests.
    with overrides(sys, argv=[str(legacy_root / "downstream_benchmark.py")]):
        args = legacy.parse_args()
    for key in ("epochs", "patience", "tuning_seed", "overlap_policy"):
        setattr(args, key, protocol[key])
    args.weight_decay = protocol["optimizer"]["weight_decay"]
    args.grad_clip = protocol["optimizer"]["grad_clip"]
    for key in ("device", "deterministic", "amp", "micro_batch_size", "num_workers"):
        setattr(args, key, protocol["execution"][key])
    args.data_root = Path(root_config["data_root"])
    args.checkpoint = Path(protocol["checkpoint"]["path"])
    args.pretrain_smiles = Path(protocol["pretrain_smiles"]["path"])
    args.search_space = Path(root_config["search_space"]["path"])
    args.search_trials = root_config["search_trials"]
    args.search_candidates = protocol["search_candidates"]
    args.datasets = ["lipo"]
    args.modes = list(VARIANTS)
    args.audit_only = False
    if args.search_trials != 4 or len(args.search_candidates) != 4 or args.tuning_seed != 42:
        raise ValueError("Expected the existing four-trial, seed-42 formal protocol")
    if args.overlap_policy != "error":
        raise ValueError("Expected unchanged official splits with overlap_policy=error")
    for key, path in (("checkpoint", args.checkpoint), ("pretrain_smiles", args.pretrain_smiles)):
        if legacy.file_identity(path)["sha256"] != protocol[key]["sha256"]:
            raise ValueError(f"Original {key} file hash changed")
    if read_json(args.search_space)["diffusion_encoder"][:args.search_trials] != args.search_candidates:
        raise ValueError("Original search candidates changed")
    actual = legacy.dataset_file_manifest(args)["lipo"]
    for name, identity in root_config["dataset_files"]["lipo"].items():
        if actual[name]["sha256"] != identity["sha256"]:
            raise ValueError(f"Lipo data/split file changed: {name}")
    return legacy, args, protocol, baseline_root


def source_identity(legacy):
    paths = [Path(__file__), *sorted((ROOT / "coarse_gnn").glob("*.py"))]
    return {str(path.relative_to(ROOT)): legacy.file_identity(path) for path in paths}


def check_inputs(legacy, args, old_protocol, baseline_root):
    dataset, splits, spec = legacy.build_property_data(args.data_root, "lipo")
    signature = legacy.split_sha256(dataset, splits)
    if signature != old_protocol["split_sha256"]:
        raise ValueError("Lipo sample IDs / SMILES order differ from the old formal experiment")
    scaler = legacy.TargetScaler.fit(dataset.targets[splits["train"]], dataset.target_mask[splits["train"]])
    for seed in (0, 1):
        directory = baseline_root / f"lipo/pretrained_finetune/seed_{seed}"
        checkpoint = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
        for field in ("center", "scale"):
            torch.testing.assert_close(getattr(scaler, field), checkpoint["target_scaler"][field], rtol=0, atol=0)
        with (directory / "test_predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
            predictions = list(csv.DictReader(handle))
        if [int(row["sample_index"]) for row in predictions] != splits["test"]:
            raise ValueError("Old test prediction sample IDs do not match")
        if [row["smiles"] for row in predictions] != [dataset.smiles[i] for i in splits["test"]]:
            raise ValueError("Old test prediction SMILES order does not match")
    legacy.require_non_leaky_checkpoint(args.checkpoint)
    overlaps = legacy.pretraining_overlap_indices(args.pretrain_smiles, dataset, splits)
    if overlaps["valid"] or overlaps["test"]:
        raise ValueError("Pretraining overlaps held-out molecules; original protocol forbids this")
    return dataset, splits, spec, {
        "split_sha256": signature, "split_counts": {key: len(ids) for key, ids in splits.items()},
        "sample_ids_and_smiles_match": True, "old_test_prediction_order_match": True,
        "train_only_scaler_matches_old_seeds_0_1": True,
        "target_scaler": {key: getattr(scaler, key).tolist() for key in ("center", "scale")},
        "pretraining_overlap": legacy.summarize_pretraining_overlap(overlaps, splits),
        "checkpoint": legacy.file_identity(args.checkpoint),
    }


def prepare(legacy, args, old_protocol, baseline_root, cache, factory):
    dataset, splits, spec, report = check_inputs(legacy, args, old_protocol, baseline_root)
    print(f"Verified original Lipo split: {report['split_counts']}", flush=True)
    # Materialize ONLY structural data. Invalid molecules raise exactly as in the
    # original dataset; no sample is silently dropped or reordered.
    for start in range(0, len(dataset), 32):
        rows = [dataset[i] for i in range(start, min(start + 32, len(dataset)))]
        precompute_batch_topologies(legacy.collate_property_batch(rows).graph, cache, STRUCTURE)
        if start % 256 == 0:
            print(f"Topology precompute: {min(start + 32, len(dataset))}/{len(dataset)}", flush=True)
    device = torch.device(args.device)
    batch = legacy.collate_property_batch([dataset[i] for i in splits["train"][:args.micro_batch_size]]).to(device)
    initial_misses = cache.misses
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["model_state"]
    configs = {}
    shared_state = None
    for variant in VARIANTS:
        predictions = []
        for repeat in range(2):
            legacy.seed_everything(0, deterministic=True)
            torch.use_deterministic_algorithms(True)
            model = factory(variant, spec.num_tasks, args.checkpoint, device, args.search_candidates[0]["dropout"])
            for key, value in model.encoder.state_dict().items():
                torch.testing.assert_close(value.cpu(), weights[key], rtol=0, atol=0)
            if repeat == 0:
                state = {key: value.cpu().clone() for key, value in model.predictor.state_dict().items()
                         if not key.startswith("coarse_gnn.")}
                if variant == "region_only":
                    shared_state = state
                elif variant == "base_coarse":
                    for key in shared_state:
                        torch.testing.assert_close(state[key], shared_state[key], rtol=0, atol=0)
            configs[variant] = asdict(model.predictor.config)
            groups = [{"params": model.head.parameters(), "lr": args.search_candidates[0]["head_lr"]},
                      {"params": [p for p in model.encoder.parameters() if p.requires_grad],
                       "lr": args.search_candidates[0]["encoder_lr"]}]
            optimizer = torch.optim.AdamW(groups, weight_decay=args.weight_decay)
            covered = {id(p) for group in optimizer.param_groups for p in group["params"]}
            if not {id(p) for p in model.parameters() if p.requires_grad} <= covered:
                raise AssertionError("Some downstream parameters are absent from the optimizer")
            model.train()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=args.amp and device.type == "cuda"):
                output = model(batch.graph)
                scaler = legacy.TargetScaler.fit(dataset.targets[splits["train"]], dataset.target_mask[splits["train"]])
                loss = legacy.property_loss(output, batch.targets, batch.target_mask, spec, scaler, None)
            predictions.append(output.detach().float().cpu())
            loss.backward()
            if not torch.isfinite(loss) or not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise AssertionError("Nonfinite loss/gradients under original AMP/determinism settings")
            if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.encoder.layers.parameters()):
                raise AssertionError("Encoder finetuning gradient is missing")
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], args.grad_clip)
            optimizer.step()
            del model, optimizer
        torch.testing.assert_close(predictions[0], predictions[1], rtol=0, atol=0)
        print(f"Passed {variant}: fixed seed, AMP, gradients and optimizer coverage", flush=True)
    if cache.misses != initial_misses:
        raise AssertionError("Precomputed topology missed during GPU forward")
    report.update(passed=True, all_molecules_precomputed=len(dataset), cache=cache.stats,
                  networks=configs, coarsening=asdict(STRUCTURE), seed_reproducible=True,
                  shared_region_and_head_initialization=True, encoder_checkpoint_verified=True,
                  gradient_check="one diagnostic train batch per model; no formal training or test evaluation",
                  selection_rule="original run_single: validation RMSE selects checkpoint; evaluate_test=False for tuning",
                  checkpoint_every_batches=args.checkpoint_every_batches,
                  checkpoint_interval_source="original parse_args default; absent from old saved run_config",
                  original_protocol=old_protocol, new_source_files=source_identity(legacy))
    write_json(args.output_dir / "preflight.json", report)
    return report


def summarize_all(project, output):
    """Read old results only; write all comparisons exclusively in 05_coarse_gnn."""
    rows = []
    for folder in ("01_local", "02_dmpnn", "03_grover", "04_diffusion", "05_coarse_gnn"):
        path = (output if folder == "05_coarse_gnn" else project / "model/results_formal" / folder) / "downstream_results.json"
        if path.exists():
            rows.extend({**row, "source_group": folder} for row in read_json(path).get("runs", []) if row["dataset"] == "lipo")
    from bond_diffusion.property_prediction import summarize_runs
    selected = {}
    for variant in VARIANTS:
        path = output / "lipo" / variant / "selected_config.json"
        if path.exists():
            selected[variant] = read_json(path)["selected_hyperparameters"]
    paired = []
    for seed in (0, 1, 2, 3, 4):
        arms = {row["mode"]: row for row in rows if row["source_group"] == "05_coarse_gnn" and row["seed"] == seed}
        if "base_coarse" in arms and "region_only" in arms:
            paired.append(dict(seed=seed, rmse_base_minus_region=arms["base_coarse"]["test_metrics"]["rmse"] - arms["region_only"]["test_metrics"]["rmse"],
                               matched_hyperparameters=arms["base_coarse"]["selected_hyperparameters"] == arms["region_only"]["selected_hyperparameters"]))
    common = [row for row in rows if row["seed"] in (0, 1)]
    summary = summarize_runs(common)
    write_json(output / "comparison.json", dict(all_existing_and_new_lipo_runs=rows,
               seed_0_1_summary=summary, paired_coarse_propagation=paired,
               selected_hyperparameters=selected,
               notice="Old baselines are read-only. Negative paired RMSE favors coarse propagation. Two seeds are exploratory; differing selected hyperparameters confound strict causal attribution."))
    if summary:
        with (output / "comparison_seed_0_1.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=sorted({key for row in summary for key in row}))
            writer.writeheader()
            writer.writerows(summary)


def train(legacy, args, old_protocol, baseline_root, cache, factory):
    # Preflight runs against real data before ANY tuning/report training.
    prepare(legacy, args, old_protocol, baseline_root, cache, factory)
    original_protocol = legacy.diffusion_tuning_protocol
    original_manifest = legacy.benchmark_run_config
    files = source_identity(legacy)
    def protocol(*values):
        result = original_protocol(*values)
        result["payload"].update(coarsening=asdict(STRUCTURE), coarse_variant=values[-1],
                                 region_layers=2, coarse_layers=0 if values[-1] == "region_only" else 3,
                                 new_source_files=files, encoder_mode="pretrained_finetune",
                                 checkpoint_every_batches=args.checkpoint_every_batches)
        result["sha256"] = digest(result["payload"])
        return result
    def manifest(values):
        result = original_manifest(values)
        # A fixed eligible seed pool permits later --seeds 2 3 4 without changing
        # experiment identity. Only args.seeds actually execute in legacy.main.
        result.update(seeds=[0, 1, 2, 3, 4], seed_field_semantics="eligible formal seed pool; completed runs record actual seeds",
                      coarsening=asdict(STRUCTURE), new_source_files=files,
                      encoder_mode="pretrained_finetune", checkpoint_every_batches=args.checkpoint_every_batches)
        return result
    with overrides(legacy, parse_args=lambda: args, create_property_model=factory,
                   diffusion_tuning_protocol=protocol, benchmark_run_config=manifest):
        legacy.main()
    summarize_all(baseline_root.parents[2], args.output_dir)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train", "summarize"), default="prepare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    parser.add_argument("--project-root", type=Path, default=PROJECT)
    parser.add_argument("--output-dir", type=Path)
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds) or not set(options.seeds) <= set(range(5)):
        parser.error("Only distinct formal seeds 0..4 are allowed")
    project = options.project_root.resolve()
    output = (options.output_dir or project / "model/results_formal/05_coarse_gnn").resolve()
    formal_root = project / "model/results_formal"
    for folder in ("01_local", "02_dmpnn", "03_grover", "04_diffusion"):
        old = (formal_root / folder).resolve()
        if output == old or old in output.parents or output in old.parents:
            parser.error("Output must not overlap any existing baseline directory")
    legacy, args, old_protocol, baseline_root = load_protocol(project)
    args.output_dir = output
    args.seeds = options.seeds
    torch.set_num_threads(2)
    output.mkdir(parents=True, exist_ok=True)
    (output / "_logs").mkdir(exist_ok=True)
    for variant in VARIANTS:
        for seed in options.seeds:
            (output / "lipo" / variant / f"seed_{seed}").mkdir(parents=True, exist_ok=True)
    cache = TopologyCache(output / "_topology_cache", max_memory_entries=8192)
    factory = make_factory(legacy.create_property_model, cache)
    if options.action == "prepare":
        prepare(legacy, args, old_protocol, baseline_root, cache, factory)
        summarize_all(project, output)
        print("Preflight passed. No tuning or formal seed training started.", flush=True)
    elif options.action == "train":
        train(legacy, args, old_protocol, baseline_root, cache, factory)
    else:
        summarize_all(project, output)


if __name__ == "__main__":
    main()
