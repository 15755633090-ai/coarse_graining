"""Five first-round frozen ablations; explicit train action, no automatic expansion."""
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
from scripts.experiments.ablation_models import (
    CHECK_VARIANTS, TRAIN_VARIANTS, CoreOnlyDataset, make_factory,
)


DEFAULT_OUTPUT = formal.ROOT / "outputs/frozen_ablation_round1"
SEEDS = tuple(range(5))
COMPARISONS = (
    ("atom", "direct_mean", "A_to_B"),
    ("direct_mean", "region_only", "B_to_C"),
    ("region_only", "base_coarse", "C_to_D"),
    ("no_edge", "base_coarse", "inter_region_messages"),
    ("deeper_region", "base_coarse", "matched_depth"),
    ("core_only", "base_coarse", "overlap_context"),
)


def baseline_pending(root, seeds=SEEDS):
    missing = []
    for mode in frozen.VARIANTS:
        for seed in seeds:
            path = root / "lipo" / mode / f"seed_{seed}" / "result.json"
            if not path.exists():
                missing.append(f"{mode}/seed_{seed}")
                continue
            result = formal.read_json(path)
            score = (result.get("test_metrics") or {}).get("rmse")
            if result.get("mode") != mode or result.get("seed") != seed or score is None or not math.isfinite(score):
                raise ValueError(f"Invalid completed baseline result: {path}")
    return missing


def verify_mother(legacy, args, protocol, selection, root):
    expected = frozen.experiment_config(legacy, args, protocol, selection)
    saved = formal.read_json(root / "run_config.json")
    if expected != saved:
        raise ValueError("Current C/D source or fixed protocol differs from the saved mother experiment")
    return saved


def prepare_data(legacy, args, protocol, baseline, mother_root):
    source, splits, spec, inputs = formal.check_inputs(legacy, args, protocol, baseline)
    # Memory-only topology construction avoids writing the existing experiment's
    # cache. Its exact fingerprints must match the shared frozen feature cache.
    cache = TopologyCache(max_memory_entries=8192)
    prepared = PreparedPropertyDataset(source, legacy.collate_property_batch, cache, formal.STRUCTURE)
    identity = feature_identity(legacy, args, prepared, protocol["split_sha256"])
    cache_path = mother_root / "_encoder_cache" / (formal.digest(identity) + ".pt")
    if not cache_path.is_file():
        raise FileNotFoundError(f"Required mother feature cache is missing; no re-extraction is allowed: {cache_path}")
    saved_cache = formal.read_json(mother_root / "preflight.json")["feature_cache"]
    if saved_cache["file_sha256"] != legacy.file_identity(cache_path)["sha256"]:
        raise ValueError("Shared encoder feature cache no longer matches the mother preflight")
    factory = make_factory(legacy.create_property_model, cache)
    # Loading an already-existing CPU cache does not run the encoder. Keep the
    # protocol's extraction device in args so its identity stays identical.
    encoder_model = factory("base_coarse", spec.num_tasks, args.checkpoint, "cpu", args.dropout)
    data, features = load_features(legacy, args, prepared, encoder_model.encoder,
                                  mother_root / "_encoder_cache", protocol["split_sha256"])
    if not features["hit"]:
        raise RuntimeError("Ablations must reuse the existing cache")
    return data, splits, spec, factory, features, inputs


def parameter_report(factory, args, tasks, legacy):
    report = {}
    for mode in TRAIN_VARIANTS + CHECK_VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        model = factory(mode, tasks, args.checkpoint, "cpu", args.dropout)
        report[mode] = {
            "network": asdict(model.predictor.config),
            "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
        }
    counts = {mode: report[mode]["trainable_parameters"] for mode in ("base_coarse", "no_edge", "deeper_region", "core_only")}
    if len(set(counts.values())) != 1:
        raise AssertionError(f"D and its parameter-matched controls differ: {counts}")
    return report


def experiment_config(legacy, mother, mother_root, features, parameters):
    paths = [formal.ROOT / "scripts/__init__.py", *sorted(Path(__file__).parent.glob("*.py"))]
    return dict(
        stage="frozen_ablation_round1", mother_experiment=str(mother_root.resolve()),
        mother_manifest_sha256=formal.digest(mother), mother_protocol=mother,
        variants=list(TRAIN_VARIANTS), eligible_seeds=list(SEEDS), new_search_trials=0,
        shared_features={key: features[key] for key in ("identity", "path", "file_sha256")},
        models=parameters, b_weighted="forward equivalence check only; never trained",
        execution_order="variant priority, then seed; stop after core_only",
        topology="same centers/owners/cores/coarse edges; core_only restricts contexts only",
        initialization="construct identical Base, then remove or relocate existing layers; no new random draws",
        new_source_files={str(p.relative_to(formal.ROOT)): legacy.file_identity(p) for p in paths},
    )


def preflight(legacy, args, data, splits, spec, factory, tools, device):
    indices = splits["train"][:min(32, len(splits["train"]))]
    core_data = CoreOnlyDataset(data)
    shared = None
    checks = []
    models = {}
    for mode in TRAIN_VARIANTS + CHECK_VARIANTS:
        legacy.seed_everything(0, deterministic=True)
        model = factory(mode, spec.num_tasks, args.checkpoint, device, args.dropout)
        models[mode] = model
        common = {key: value.detach().cpu().clone() for key, value in model.predictor.state_dict().items()
                  if key.startswith(("input_projection.", "head."))}
        if shared is None:
            shared = common
        for key in shared:
            torch.testing.assert_close(shared[key], common[key], rtol=0, atol=0)
        selected = core_data if mode == "core_only" else data
        batch = tools["collate_property_batch"]([selected[i] for i in indices[:args.micro_batch_size]]).to(device)
        model.train()
        output = model(batch.graph)
        if not torch.isfinite(output).all():
            raise AssertionError(f"Nonfinite preflight output: {mode}")
        output.square().mean().backward()
        if model.encoder.training or any(p.requires_grad or p.grad is not None for p in model.encoder.parameters()):
            raise AssertionError("Ablation changed or trained the frozen encoder")
        if not any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()):
            raise AssertionError(f"Missing predictor gradients: {mode}")
        if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.head.parameters()):
            raise AssertionError(f"Nonfinite gradients: {mode}")
        model.zero_grad(set_to_none=True)
        model.eval()
        checks.append(mode)
    errors = {"fp32_prediction_max_abs": 0.0, "bf16_prediction_max_abs": 0.0}
    with torch.no_grad():
        for start in range(0, len(indices), args.micro_batch_size):
            batch = tools["collate_property_batch"]([data[i] for i in indices[start:start + args.micro_batch_size]]).to(device)
            a = models["atom"](batch.graph)
            weighted = models["direct_weighted"](batch.graph)
            torch.testing.assert_close(a, weighted, rtol=2e-5, atol=2e-6)
            errors["fp32_prediction_max_abs"] = max(errors["fp32_prediction_max_abs"], (a - weighted).abs().max().item())
            with torch.autocast(torch.device(device).type, dtype=torch.bfloat16, enabled=args.amp):
                a_amp = models["atom"](batch.graph)
                b_amp = models["direct_weighted"](batch.graph)
            if not torch.isfinite(a_amp).all() or not torch.isfinite(b_amp).all():
                raise AssertionError("Nonfinite mixed-precision pooling check")
            errors["bf16_prediction_max_abs"] = max(errors["bf16_prediction_max_abs"], (a_amp.float() - b_amp.float()).abs().max().item())
    return dict(passed=True, device=str(device), gradient_checks=checks, common_projection_and_head_identical=True,
                train_samples=len(indices), test_metrics_used=False, pooling_equivalence=errors,
                pooling_note="Exact arithmetic identity for a complete disjoint core partition. FP32 is tolerance-checked; BF16 two-stage rounding is reported, not assumed bitwise equal.")


def summarize(output, mother_root):
    runs = []
    for root, modes in ((mother_root, frozen.VARIANTS), (output, TRAIN_VARIANTS)):
        for mode in modes:
            for seed in SEEDS:
                path = root / "lipo" / mode / f"seed_{seed}" / "result.json"
                if path.exists():
                    row = formal.read_json(path)
                    if row.get("mode") != mode or row.get("seed") != seed:
                        raise ValueError(f"Result identity mismatch: {path}")
                    rmse = (row.get("test_metrics") or {}).get("rmse")
                    if rmse is None or not math.isfinite(rmse):
                        raise ValueError(f"Invalid final test score: {path}")
                    runs.append(dict(mode=mode, seed=seed, test_rmse=rmse,
                                     best_epoch=row.get("best_epoch"), source=str(path.resolve())))
    stats = []
    for mode in ("atom", "direct_mean", "region_only", "base_coarse", "no_edge", "deeper_region", "core_only"):
        rows = [r for r in runs if r["mode"] == mode]
        if rows:
            scores = [r["test_rmse"] for r in rows]
            stats.append(dict(mode=mode, n=len(rows), seeds=[r["seed"] for r in rows],
                              mean_rmse=statistics.mean(scores),
                              std_rmse=statistics.stdev(scores) if len(scores) > 1 else None))
    lookup = {(r["mode"], r["seed"]): r["test_rmse"] for r in runs}
    paired = []
    for before, after, question in COMPARISONS:
        deltas = [dict(seed=s, delta_rmse=lookup[after, s] - lookup[before, s]) for s in SEEDS
                  if (after, s) in lookup and (before, s) in lookup]
        if deltas:
            values = [r["delta_rmse"] for r in deltas]
            paired.append(dict(question=question, before=before, after=after, n=len(values),
                               per_seed=deltas, mean_delta_rmse=statistics.mean(values),
                               std_delta_rmse=statistics.stdev(values) if len(values) > 1 else None))
    complete = all((mode, seed) in lookup for mode in TRAIN_VARIANTS + frozen.VARIANTS for seed in SEEDS)
    report = dict(runs=runs, summary=stats, paired=paired, complete=complete,
                  interpretation="Fixed-protocol paired seed comparison; negative after-minus-before RMSE favors after. No significance claim from means alone.")
    formal.write_json(output / "comparison.json", report)
    if runs:
        with (output / "per_seed.csv").open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(runs[0]))
            writer.writeheader()
            writer.writerows(runs)
    return report


def train(legacy, args, data, splits, spec, factory, tools, config, modes, mother_root):
    original_save = legacy._save_downstream_resume
    times = {}

    def save(path, state):
        original_save(path, state)
        if state["next_batch"] == 0 and state["history"]:
            epoch = state["history"][-1]["epoch"]
            previous_epoch, previous_time = times.get(str(path), (None, None))
            if previous_epoch != epoch:
                now = time.monotonic()
                status = dict(mode=state["mode"], seed=state["seed"], completed_epoch=epoch,
                              epoch_seconds=None if previous_time is None else now - previous_time,
                              best_epoch=state["best_epoch"], best_validation_score=state["best_score"],
                              updated_at=datetime.now(timezone.utc).isoformat())
                times[str(path)] = (epoch, now)
                formal.write_json(path.parent / "progress.json", status)
                print(status, flush=True)

    core_data = CoreOnlyDataset(data)
    with formal.overrides(legacy, create_property_model=factory, _save_downstream_resume=save, **tools):
        for mode in modes:
            selected_data = core_data if mode == "core_only" else data
            for seed in args.seeds:
                run_args = copy.copy(args)
                payload = dict(config, mode=mode)
                run_args.tuning_protocol = dict(payload=payload, sha256=formal.digest(payload))
                directory = args.output_dir / "lipo" / mode / f"seed_{seed}"
                expected = legacy.single_run_config(spec, mode, seed, "report",
                           {key: getattr(args, key) for key in ("batch_size", "encoder_lr", "head_lr", "dropout")}, run_args.tuning_protocol)
                legacy.initialize_single_run_config(directory, expected)
                if (directory / "result.json").exists():
                    print(f"Completed already: {mode} seed={seed}", flush=True)
                    continue
                print(f"Start {mode} seed={seed}; fixed mother settings", flush=True)
                legacy.run_single(run_args, selected_data, splits, spec, mode, seed,
                                  torch.device(args.device), directory, evaluate_test=True)
                summarize(args.output_dir, mother_root)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train", "summarize"), default="prepare")
    parser.add_argument("--variants", nargs="+", choices=TRAIN_VARIANTS, default=list(TRAIN_VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check-device", choices=("cpu", "cuda"), default="cpu",
                        help="Prepare-only preflight device; training always checks the fixed training device")
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
            parser.error("Finish existing C/D seeds 0..4 before training ablations: " + ", ".join(pending))
    args.seeds = options.seeds
    if options.action == "summarize":
        saved = formal.read_json(output / "run_config.json")
        if saved.get("mother_manifest_sha256") != formal.digest(mother):
            raise ValueError("Summary directory belongs to a different mother experiment")
        summarize(output, mother_root)
        return
    data, splits, spec, factory, features, inputs = prepare_data(legacy, args, protocol, baseline, mother_root)
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
        # Keep user-selected subset in the predeclared priority order.
        modes = [mode for mode in TRAIN_VARIANTS if mode in options.variants]
        train(legacy, args, data, splits, spec, factory, tools, config, modes, mother_root)
    print(f"{options.action} finished: {output}", flush=True)


if __name__ == "__main__":
    main()
