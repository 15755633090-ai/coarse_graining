"""Train C/D size-weighted readout controls without altering round-one runs."""
from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

import torch

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset
from frozen_features import feature_identity, frozen_tools, load_features
from scripts.experiments.frozen_reporting import preserve_training_manifest
from scripts.readout_ablation.models import (
    ALL_VARIANTS,
    REFERENCE_BY_VARIANT,
    REFERENCE_VARIANTS,
    TRAIN_VARIANTS,
    make_factory,
)


DEFAULT_OUTPUT = formal.ROOT / "outputs/frozen_size_weighted"
SEEDS = tuple(range(5))
EVALUATION_POLICY = {
    "development_metric": "validation_rmse",
    "evaluate_test_during_training": False,
    "automatic_final_test": False,
    "readout_only_ablation": True,
    "existing_cd_test_scores": "already exposed; excluded from development summaries",
}
COMPARISONS = (
    ("region_only", "region_size_weighted", "C_equal_vs_size_weighted"),
    ("base_coarse", "base_size_weighted", "D_equal_vs_size_weighted"),
    ("region_size_weighted", "base_size_weighted", "C_size_vs_D_size"),
)
COMPARISON_SCOPE = (
    "Only graph_pool changes from equal coarse-node mean to core-size-weighted mean. "
    "After Region/Coarse processing this is a conservation-oriented inductive bias, "
    "not an algebraic reconstruction of the atom mean."
)


def baseline_pending(root, seeds=SEEDS):
    missing = []
    for mode in REFERENCE_VARIANTS:
        for seed in seeds:
            path = root / "lipo" / mode / f"seed_{seed}" / "result.json"
            if not path.is_file():
                missing.append(f"{mode}/seed_{seed}")
                continue
            result = formal.read_json(path)
            score = (result.get("validation_metrics") or {}).get("rmse")
            if result.get("mode") != mode or result.get("seed") != seed or score is None or not math.isfinite(score):
                raise ValueError(f"Invalid completed baseline result: {path}")
    return missing


def verify_mother(legacy, args, protocol, selection, root):
    expected = frozen.experiment_config(legacy, args, protocol, selection)
    saved = formal.read_json(root / "run_config.json")
    compatible = preserve_training_manifest(legacy, expected, root, write_audit=False)
    if compatible != saved:
        raise ValueError("Current C/D source or fixed protocol differs from the saved mother experiment")
    return saved


def prepare_data(legacy, args, protocol, baseline, mother_root):
    source, splits, spec, inputs = formal.check_inputs(legacy, args, protocol, baseline)
    cache = TopologyCache(max_memory_entries=8192)
    prepared = PreparedPropertyDataset(source, legacy.collate_property_batch, cache, formal.STRUCTURE)
    identity = feature_identity(legacy, args, prepared, protocol["split_sha256"])
    cache_path = mother_root / "_encoder_cache" / (formal.digest(identity) + ".pt")
    if not cache_path.is_file():
        raise FileNotFoundError(f"Required mother feature cache is missing: {cache_path}")
    saved_cache = formal.read_json(mother_root / "preflight.json")["feature_cache"]
    if saved_cache["file_sha256"] != legacy.file_identity(cache_path)["sha256"]:
        raise ValueError("Shared encoder feature cache no longer matches the mother preflight")

    factory = make_factory(legacy.create_property_model, cache)
    encoder_model = factory("base_coarse", spec.num_tasks, args.checkpoint, "cpu", args.dropout)
    data, features = load_features(
        legacy,
        args,
        prepared,
        encoder_model.encoder,
        mother_root / "_encoder_cache",
        protocol["split_sha256"],
    )
    if not features["hit"]:
        raise RuntimeError("Readout ablations must reuse the existing frozen feature cache")
    return data, splits, spec, factory, features, inputs


def parameter_report(factory, args, tasks, legacy):
    report = {}
    for mode in ALL_VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        model = factory(mode, tasks, args.checkpoint, "cpu", args.dropout)
        report[mode] = {
            "network": asdict(model.predictor.config),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    for weighted, reference in REFERENCE_BY_VARIANT.items():
        if report[weighted]["trainable_parameters"] != report[reference]["trainable_parameters"]:
            raise AssertionError(f"Readout pair parameter mismatch: {reference} vs {weighted}")
    return report


def experiment_config(legacy, mother, mother_root, features, parameters):
    source_paths = sorted(Path(__file__).parent.glob("*.py"))
    return {
        "stage": "frozen_size_weighted_readout",
        "schema_version": 1,
        "evaluation_policy": dict(EVALUATION_POLICY),
        "mother_experiment": str(mother_root.resolve()),
        "mother_manifest_sha256": formal.digest(mother),
        "mother_protocol": mother,
        "variants": list(TRAIN_VARIANTS),
        "eligible_seeds": list(SEEDS),
        "new_search_trials": 0,
        "shared_features": {key: features[key] for key in ("identity", "path", "file_sha256")},
        "models": parameters,
        "change": "graph_pool only: mean -> size_weighted_mean",
        "interpretation_limit": COMPARISON_SCOPE,
        "initialization": "construct identical Base, then change pooling and optionally remove coarse layers; no new random draws",
        "new_source_files": {
            str(path.relative_to(formal.ROOT)): legacy.file_identity(path) for path in source_paths
        },
    }


def _predictor_state(model):
    return {
        key: value.detach().cpu().clone()
        for key, value in model.predictor.state_dict().items()
    }


def preflight(legacy, args, data, splits, spec, factory, tools, device):
    indices = splits["train"][: min(32, len(splits["train"]))]
    batch = tools["collate_property_batch"](
        [data[i] for i in indices[: args.micro_batch_size]]
    ).to(device)
    models = {}
    checks = []

    for mode in ALL_VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        model = factory(mode, spec.num_tasks, args.checkpoint, device, args.dropout)
        models[mode] = model
        model.train()
        output = model(batch.graph)
        if not torch.isfinite(output).all():
            raise AssertionError(f"Nonfinite preflight output: {mode}")
        output.square().mean().backward()
        if model.encoder.training or any(p.requires_grad or p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Readout ablation changed or trained the frozen encoder")
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()):
            raise AssertionError(f"Missing predictor gradients: {mode}")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.head.parameters()):
            raise AssertionError(f"Nonfinite predictor gradients: {mode}")
        checks.append(mode)

    for weighted, reference in REFERENCE_BY_VARIANT.items():
        weighted_state = _predictor_state(models[weighted])
        reference_state = _predictor_state(models[reference])
        if weighted_state.keys() != reference_state.keys():
            raise AssertionError(f"State structure differs: {reference} vs {weighted}")
        for key in weighted_state:
            torch.testing.assert_close(weighted_state[key], reference_state[key], rtol=0, atol=0)
        weighted_cfg = asdict(models[weighted].predictor.config)
        reference_cfg = asdict(models[reference].predictor.config)
        differences = {key for key in weighted_cfg if weighted_cfg[key] != reference_cfg[key]}
        if differences != {"graph_pool"}:
            raise AssertionError(f"Unexpected config differences for {weighted}: {differences}")

    return {
        "passed": True,
        "device": str(device),
        "gradient_checks": checks,
        "pair_parameters_identical": True,
        "pair_config_difference": ["graph_pool"],
        "train_samples": len(indices),
        "test_metrics_used": False,
    }


def _read_runs(root, modes):
    runs = []
    for mode in modes:
        for seed in SEEDS:
            path = root / "lipo" / mode / f"seed_{seed}" / "result.json"
            if not path.is_file():
                continue
            row = formal.read_json(path)
            rmse = (row.get("validation_metrics") or {}).get("rmse")
            if row.get("mode") != mode or row.get("seed") != seed or rmse is None or not math.isfinite(rmse):
                raise ValueError(f"Invalid result identity or validation score: {path}")
            runs.append({
                "mode": mode,
                "seed": seed,
                "validation_rmse": rmse,
                "best_epoch": row.get("best_epoch"),
                "source": str(path.resolve()),
            })
    return runs


def summarize(output, mother_root):
    runs = _read_runs(mother_root, REFERENCE_VARIANTS) + _read_runs(output, TRAIN_VARIANTS)
    summary = []
    for mode in ALL_VARIANTS:
        rows = [row for row in runs if row["mode"] == mode]
        if rows:
            scores = [row["validation_rmse"] for row in rows]
            summary.append({
                "mode": mode,
                "n": len(rows),
                "seeds": [row["seed"] for row in rows],
                "mean_rmse": statistics.mean(scores),
                "std_rmse": statistics.stdev(scores) if len(scores) > 1 else None,
            })

    lookup = {(row["mode"], row["seed"]): row["validation_rmse"] for row in runs}
    paired = []
    for before, after, question in COMPARISONS:
        deltas = [
            {"seed": seed, "delta_rmse": lookup[after, seed] - lookup[before, seed]}
            for seed in SEEDS
            if (before, seed) in lookup and (after, seed) in lookup
        ]
        if deltas:
            values = [row["delta_rmse"] for row in deltas]
            paired.append({
                "question": question,
                "before": before,
                "after": after,
                "n": len(values),
                "scope": COMPARISON_SCOPE,
                "per_seed": deltas,
                "mean_delta_rmse": statistics.mean(values),
                "std_delta_rmse": statistics.stdev(values) if len(values) > 1 else None,
            })

    interaction = []
    for seed in SEEDS:
        required = ("region_only", "base_coarse", "region_size_weighted", "base_size_weighted")
        if not all((mode, seed) in lookup for mode in required):
            continue
        equal_coarse = lookup["base_coarse", seed] - lookup["region_only", seed]
        weighted_coarse = lookup["base_size_weighted", seed] - lookup["region_size_weighted", seed]
        interaction.append({
            "seed": seed,
            "equal_readout_coarse_delta_rmse": equal_coarse,
            "size_weighted_coarse_delta_rmse": weighted_coarse,
            "interaction_delta_rmse": weighted_coarse - equal_coarse,
        })

    complete = all((mode, seed) in lookup for mode in ALL_VARIANTS for seed in SEEDS)
    report = {
        "metric_split": "validation",
        "runs": runs,
        "summary": summary,
        "paired": paired,
        "coarse_readout_interaction": {
            "scope": "Difference in the Coarse-GNN effect after switching to size-weighted readout.",
            "per_seed": interaction,
            "mean_interaction_delta_rmse": statistics.mean(row["interaction_delta_rmse"] for row in interaction) if interaction else None,
            "std_interaction_delta_rmse": statistics.stdev(row["interaction_delta_rmse"] for row in interaction) if len(interaction) > 1 else None,
        },
        "complete": complete,
        "interpretation": (
            "Negative weighted-minus-equal RMSE favors size weighting. "
            "No new test metrics are read or reported."
        ),
    }
    formal.write_json(output / "comparison_validation.json", report)
    if runs:
        with (output / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    return report


def train(legacy, args, data, splits, spec, factory, tools, config, modes, mother_root):
    if config.get("evaluation_policy") != EVALUATION_POLICY:
        raise ValueError("Size-weighted training requires the locked validation-only policy")
    original_save = legacy._save_downstream_resume
    times = {}

    def save(path, state):
        original_save(path, state)
        if state["next_batch"] == 0 and state["history"]:
            epoch = state["history"][-1]["epoch"]
            previous_epoch, previous_time = times.get(str(path), (None, None))
            if previous_epoch != epoch:
                now = time.monotonic()
                status = {
                    "mode": state["mode"],
                    "seed": state["seed"],
                    "completed_epoch": epoch,
                    "epoch_seconds": None if previous_time is None else now - previous_time,
                    "best_epoch": state["best_epoch"],
                    "best_validation_score": state["best_score"],
                    "updated_at": datetime.now(timezone.utc).isoformat(),
                }
                times[str(path)] = (epoch, now)
                formal.write_json(path.parent / "progress.json", status)
                print(status, flush=True)

    with formal.overrides(legacy, create_property_model=factory, _save_downstream_resume=save, **tools):
        for mode in modes:
            for seed in args.seeds:
                run_args = copy.copy(args)
                payload = dict(config, mode=mode)
                run_args.tuning_protocol = {"payload": payload, "sha256": formal.digest(payload)}
                directory = args.output_dir / "lipo" / mode / f"seed_{seed}"
                expected = legacy.single_run_config(
                    spec,
                    mode,
                    seed,
                    "report",
                    {key: getattr(args, key) for key in ("batch_size", "encoder_lr", "head_lr", "dropout")},
                    run_args.tuning_protocol,
                )
                legacy.initialize_single_run_config(directory, expected)
                if (directory / "result.json").is_file():
                    completed = formal.read_json(directory / "result.json")
                    if completed.get("test_metrics") is not None or (directory / "test_predictions.csv").exists():
                        raise ValueError(f"Refusing to reuse a test-evaluated run: {directory}")
                    print(f"Completed already: {mode} seed={seed}", flush=True)
                    continue
                print(f"Start {mode} seed={seed}; graph readout is the only model change", flush=True)
                legacy.run_single(
                    run_args,
                    data,
                    splits,
                    spec,
                    mode,
                    seed,
                    torch.device(args.device),
                    directory,
                    evaluate_test=False,
                )
                summarize(args.output_dir, mother_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train", "summarize"), default="prepare")
    parser.add_argument("--variants", nargs="+", choices=TRAIN_VARIANTS, default=list(TRAIN_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check-device",
        choices=("cpu", "cuda"),
        default="cpu",
        help="Prepare-only device; training still uses the fixed mother protocol device",
    )
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds) or not set(options.seeds) <= set(SEEDS):
        parser.error("Use distinct seeds from 0..4")
    if len(set(options.variants)) != len(options.variants):
        parser.error("Variants must be distinct")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    legacy, args, protocol, baseline, selection = frozen.load_protocol(formal.PROJECT)
    mother_root = args.output_dir
    mother = verify_mother(legacy, args, protocol, selection, mother_root)
    output = options.output_dir.resolve()
    protected = (formal.PROJECT / "model/results_formal").resolve()
    if output == protected or protected in output.parents:
        parser.error("Use a separate output directory outside existing formal experiment results")

    if options.action == "train":
        pending = baseline_pending(mother_root)
        if pending:
            parser.error("Finish existing C/D seeds 0..4 first: " + ", ".join(pending))
    args.seeds = options.seeds

    if options.action == "summarize":
        saved = formal.read_json(output / "run_config.json")
        if saved.get("mother_manifest_sha256") != formal.digest(mother):
            raise ValueError("Summary directory belongs to a different mother experiment")
        if saved.get("evaluation_policy") != EVALUATION_POLICY:
            raise ValueError("Summary directory does not use the validation-only readout policy")
        summarize(output, mother_root)
        return

    data, splits, spec, factory, features, inputs = prepare_data(
        legacy, args, protocol, baseline, mother_root
    )
    parameters = parameter_report(factory, args, spec.num_tasks, legacy)
    config = experiment_config(legacy, mother, mother_root, features, parameters)
    output.mkdir(parents=True, exist_ok=True)
    legacy.initialize_or_validate_run_config(output, config)
    tools = frozen_tools(legacy)
    device = args.device if options.action == "train" else options.check_device
    report = preflight(legacy, args, data, splits, spec, factory, tools, device)
    formal.write_json(output / "preflight.json", dict(report, inputs=inputs, feature_cache=features))
    args.output_dir = output

    if options.action == "train":
        modes = [mode for mode in TRAIN_VARIANTS if mode in options.variants]
        train(legacy, args, data, splits, spec, factory, tools, config, modes, mother_root)
    summarize(output, mother_root)
    print(f"{options.action} finished: {output}", flush=True)


if __name__ == "__main__":
    main()
