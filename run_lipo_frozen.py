"""Stage one: fixed encoder, matched hyperparameters, two models and seeds 0/1."""
import argparse
import copy
import csv
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
import time

import torch

import run_lipo_formal as formal
from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset
from coarse_gnn.packed import PreparedGraphBatch
from frozen_features import frozen_factory, frozen_tools, load_features

VARIANTS = ("region_only", "base_coarse")


def load_protocol(project):
    legacy, args, finetune_protocol, baseline = formal.load_protocol(project)
    selection_path = baseline / "lipo/pretrained_frozen/selected_config.json"
    selection = formal.read_json(selection_path)
    protocol = selection["tuning_protocol"]
    # Reuse the same verified original trainer/data, and verify the frozen
    # baseline had exactly the same execution and data protocol.
    for key in ("split_sha256", "epochs", "patience", "optimizer", "execution", "checkpoint",
                "pretrain_smiles", "overlap_policy", "source_files", "search_candidates"):
        if protocol[key] != finetune_protocol[key]:
            raise ValueError(f"Frozen and finetune baseline protocols differ: {key}")
    for key, value in selection["selected_hyperparameters"].items():
        setattr(args, key, value)
    args.modes = list(VARIANTS)
    args.output_dir = project / "model/results_formal/05_coarse_gnn/frozen_mechanism"
    return legacy, args, protocol, baseline, legacy.file_identity(selection_path)


def sources(legacy):
    result = formal.source_identity(legacy)
    for name in ("frozen_features.py", "run_lipo_frozen.py"):
        result[name] = legacy.file_identity(formal.ROOT / name)
    return result


def experiment_config(legacy, args, old_protocol, selection):
    return dict(stage="frozen_mechanism", encoder_mode="pretrained_frozen",
                variants=list(VARIANTS), eligible_seeds=[0, 1, 2, 3, 4],
                original_protocol=old_protocol, parameter_selection=selection,
                new_search_trials=0, selection_rule="reuse prior frozen baseline validation selection for both models",
                fixed_hyperparameters={key: getattr(args, key) for key in ("head_lr", "dropout", "batch_size")},
                encoder_lr_effective=0, micro_batch_size=args.micro_batch_size,
                checkpoint_every_batches=args.checkpoint_every_batches,
                coarsening=asdict(formal.STRUCTURE),
                networks={v: asdict(formal.network_config(v, 128, args.dropout)) for v in VARIANTS},
                feature_precision="FP32 encoder in eval; same cached tensors during training and evaluation",
                new_source_files=sources(legacy))


def preflight(legacy, args, source, splits, spec, data, factory, tools, features, inputs):
    scaler = legacy.TargetScaler.fit(source.targets[splits["train"]], source.target_mask[splits["train"]])
    baseline = args.output_dir.parents[1] / "04_diffusion/lipo/pretrained_frozen"
    for seed in (0, 1):
        old = torch.load(baseline / f"seed_{seed}/best.pt", map_location="cpu", weights_only=False)
        for name in ("center", "scale"):
            torch.testing.assert_close(getattr(scaler, name), old["target_scaler"][name], rtol=0, atol=0)
    batch = tools["collate_property_batch"]([data[i] for i in splits["train"][:args.micro_batch_size]]).to(args.device)
    shared = None
    for variant in VARIANTS:
        repeated = []
        for _ in range(2):
            legacy.seed_everything(0, deterministic=True)
            model = factory(variant, spec.num_tasks, args.checkpoint, torch.device(args.device), args.dropout).train()
            assert not model.encoder.training and not any(p.requires_grad for p in model.encoder.parameters())
            state = {k: v.detach().clone() for k, v in model.encoder.state_dict().items()}
            common = {k: v.detach().cpu().clone() for k, v in model.predictor.state_dict().items() if not k.startswith("coarse_gnn.")}
            if shared is None:
                shared = common
            for key in shared:
                torch.testing.assert_close(shared[key], common[key], rtol=0, atol=0)
            model.eval()
            with torch.no_grad():
                cached = model(batch.graph)
                direct = model(PreparedGraphBatch(batch.graph.graph, batch.graph.topologies, batch.graph.plan))
            torch.testing.assert_close(cached, direct, atol=2e-5, rtol=2e-5)
            model.train()
            optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr, weight_decay=args.weight_decay)
            with torch.autocast(torch.device(args.device).type, dtype=torch.bfloat16, enabled=args.amp):
                output = model(batch.graph)
                loss = legacy.property_loss(output, batch.targets, batch.target_mask, spec, scaler, None)
            repeated.append(output.detach().float().cpu())
            loss.backward()
            assert torch.isfinite(loss) and all(p.grad is None for p in model.encoder.parameters())
            assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters())
            torch.nn.utils.clip_grad_norm_(model.head.parameters(), args.grad_clip)
            optimizer.step()
            for key, value in model.encoder.state_dict().items():
                torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            del model, optimizer
        torch.testing.assert_close(repeated[0], repeated[1], rtol=0, atol=0)
        print(f"Passed {variant}: frozen weights, shared initialization, cached/direct features, seeded gradients", flush=True)
    formal.write_json(args.output_dir / "preflight.json", dict(passed=True, inputs=inputs,
                      frozen_baseline_scaler_matches=True, encoder_unchanged_after_optimizer=True,
                      cache_matches_direct_encoder=True, identical_common_initialization=True,
                      feature_cache=features, test_metrics_not_used=True))


def summarize(args, baseline):
    from scripts.experiments.frozen_reporting import summarize_frozen
    return summarize_frozen(args, baseline)


def train(legacy, args, data, splits, spec, factory, tools, config, baseline):
    # The original training loop, validation selection, test isolation and
    # checkpoint/resume logic are reused without introducing a tuning loop.
    original_save = legacy._save_downstream_resume
    times = {}
    def save(path, state):
        original_save(path, state)
        if state["next_batch"] == 0 and state["history"]:
            epoch = state["history"][-1]["epoch"]
            previous_epoch, previous_time = times.get(str(path), (None, None))
            if epoch != previous_epoch:
                now = time.monotonic()
                status = dict(mode=state["mode"], seed=state["seed"], completed_epoch=epoch,
                              epoch_seconds=None if previous_time is None else now - previous_time,
                              best_epoch=state["best_epoch"], best_validation_score=state["best_score"],
                              updated_at=datetime.now(timezone.utc).isoformat())
                times[str(path)] = (epoch, now)
                formal.write_json(path.parent / "progress.json", status)
                print(status, flush=True)
    with formal.overrides(legacy, create_property_model=factory, _save_downstream_resume=save, **tools):
        for seed in args.seeds:
            for variant in VARIANTS:
                run_args = copy.copy(args)
                payload = dict(config, mode=variant)
                run_args.tuning_protocol = dict(payload=payload, sha256=formal.digest(payload))
                directory = args.output_dir / "lipo" / variant / f"seed_{seed}"
                expected = legacy.single_run_config(spec, variant, seed, "report",
                            {key: getattr(args, key) for key in ("batch_size", "encoder_lr", "head_lr", "dropout")}, run_args.tuning_protocol)
                directory.mkdir(parents=True, exist_ok=True)
                legacy.initialize_single_run_config(directory, expected)
                if (directory / "result.json").exists():
                    print(f"Completed already: {variant} seed={seed}", flush=True)
                    continue
                print(f"Start frozen {variant} seed={seed}; no hyperparameter search", flush=True)
                legacy.run_single(run_args, data, splits, spec, variant, seed, torch.device(args.device), directory, evaluate_test=True)
                summarize(args, baseline)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", choices=("prepare", "train", "summarize"), default="prepare")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    options = parser.parse_args()
    if len(set(options.seeds)) != len(options.seeds) or not set(options.seeds) <= set(range(5)):
        parser.error("Use distinct seeds from 0..4")
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    legacy, args, protocol, baseline, selection = load_protocol(formal.PROJECT)
    args.seeds = options.seeds
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if options.action == "summarize":
        summarize(args, baseline)
        return
    config = experiment_config(legacy, args, protocol, selection)
    from scripts.experiments.frozen_reporting import preserve_training_manifest
    config = preserve_training_manifest(legacy, config, args.output_dir)
    legacy.initialize_or_validate_run_config(args.output_dir, config)
    source, splits, spec, inputs = formal.check_inputs(legacy, args, protocol, baseline)
    print(f"Verified existing Lipo scaffold split: {inputs['split_counts']}", flush=True)
    cache = TopologyCache(args.output_dir.parent / "_topology_cache", max_memory_entries=8192)
    prepared = PreparedPropertyDataset(source, legacy.collate_property_batch, cache, formal.STRUCTURE)
    factory = frozen_factory(legacy.create_property_model, cache)
    encoder_model = factory("region_only", spec.num_tasks, args.checkpoint, args.device, args.dropout)
    data, features = load_features(legacy, args, prepared, encoder_model.encoder,
                                  args.output_dir / "_encoder_cache", protocol["split_sha256"])
    del encoder_model
    tools = frozen_tools(legacy)
    preflight(legacy, args, source, splits, spec, data, factory, tools, features, inputs)
    print(f"Frozen feature cache hit={features['hit']}; new runs=2 models x {len(args.seeds)} seeds", flush=True)
    if options.action == "train":
        train(legacy, args, data, splits, spec, factory, tools, config, baseline)
    summarize(args, baseline)


if __name__ == "__main__":
    main()
