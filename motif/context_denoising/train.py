"""Run from workspace root: python -m motif.context_denoising.train --help."""
import argparse
from copy import deepcopy
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

from .data import MotifDataset
from .model import Config, MotifContextModel, VARIANTS
from .diagnostics import gradient_diagnostics, summarize_steps
from .reporting import (
    collect_provenance, fingerprint, output_session, prepare_cohort,
    refresh_summaries, save_partition_audit, write_json,
)

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from benchmark import (  # noqa: E402
    TASK_SPECS, TargetScaler, better_score, build_property_data,
    classification_positive_weight, compute_property_metrics, make_loader,
    masked_loss, predict, primary_score, save_predictions, seed_everything,
)


def run(args, source, splits, spec, dataset, variant, seed, runtime, protocol_sha256):
    out = args.output_dir / args.dataset / variant / f"seed_{seed}"
    out.mkdir(parents=True, exist_ok=True)
    if (out / "config.json").exists():
        raise FileExistsError(f"Run exists; choose a new output directory: {out}")
    seed_everything(seed, deterministic=args.device == "cpu")
    config = Config(
        hidden_dim=args.hidden_dim, atom_layers=args.atom_layers,
        context_layers=args.context_layers, dropout=args.dropout,
        noise_steps=args.noise_steps, fixed_step=args.fixed_step,
        corruption_rate=args.corruption_rate,
    )
    model = MotifContextModel(spec.num_tasks, config).to(args.device)
    parameter_counts = model.configure_training(variant, args.denoise_weight)
    optimized_parameters = tuple(p for p in model.parameters() if p.requires_grad)
    provenance = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "model": asdict(config), "variant": variant, "seed": seed,
        "status": "smoke_only" if args.limit_per_split else "experiment",
        "partition": "RDKit BRICS cuts; connected components without dummy atoms",
        "splits": splits,
        "split_sha256": hashlib.sha256(json.dumps(splits, sort_keys=True).encode()).hexdigest(),
        "data_sha256": hashlib.sha256((args.data_root / spec.directory / "mapping" / "mol.csv.gz").read_bytes()).hexdigest(),
        **runtime,
        "protocol_sha256": protocol_sha256,
        "parameter_counts": parameter_counts,
        "torch": torch.__version__, "parameters": sum(p.numel() for p in model.parameters()),
        "cuda_deterministic": False if args.device.startswith("cuda") else None,
    }
    write_json(out / "config.json", provenance)
    train_y, train_mask = source.targets[splits["train"]], source.target_mask[splits["train"]]
    scaler = TargetScaler.fit(train_y, train_mask) if spec.task_type == "regression" else None
    weight = classification_positive_weight(train_y, train_mask) if spec.task_type == "classification" else None
    optimizer = torch.optim.AdamW(optimized_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    loaders = {s: make_loader(dataset, ids, args.batch_size, s == "train", seed) for s, ids in splits.items()}
    best_score, state, best_epoch, stale = None, None, 0, 0
    history = []
    start = time.perf_counter()
    # A separate generator keeps corruption RNG from changing clean-branch dropout draws.
    noise_generator = torch.Generator(device=args.device).manual_seed(seed + 7919)
    optimizer_steps = 0
    auxiliary_weight = args.denoise_weight if variant in ("fixed", "random", "no_communication", "mask", "shuffle") else 0.0
    for epoch in range(1, args.epochs + 1):
        model.train()
        logs = []
        for batch in loaders["train"]:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            logits, auxiliary, diagnostic = model(batch, variant, auxiliary=True, generator=noise_generator)
            supervised = masked_loss(logits, batch.y, batch.target_mask, spec, scaler, weight)
            if supervised is None:
                continue
            loss = supervised + args.denoise_weight * auxiliary
            if not torch.isfinite(loss):
                raise RuntimeError("Non-finite joint loss")
            gradients = {}
            if optimizer_steps % args.gradient_diagnostics_every == 0:
                gradients = gradient_diagnostics(supervised, auxiliary, optimized_parameters, auxiliary_weight)
            loss.backward()
            pre_clip_norm = float(torch.nn.utils.clip_grad_norm_(
                optimized_parameters, args.grad_clip, error_if_nonfinite=True
            ))
            clip_scale = min(1.0, args.grad_clip / (pre_clip_norm + 1e-6))
            gradient_parameters = sum(p.numel() for p in optimized_parameters if p.grad is not None)
            optimizer.step()
            optimizer_steps += 1
            logs.append({"loss": float(loss.detach()), "property_loss": float(supervised.detach()),
                         "denoise_loss": float(auxiliary.detach()),
                         "pre_clip_gradient_norm": pre_clip_norm,
                         "clip_triggered": float(clip_scale < 1.0), "clip_scale": clip_scale,
                         "parameters_with_grad": gradient_parameters,
                         **gradients,
                         **{k: float(v) for k, v in diagnostic.items()}})
        if not logs:
            raise RuntimeError("No observed training labels")
        p, y, mask, _ = predict(model_for_variant(model, variant), loaders["valid"], args.device, scaler, spec)
        metrics = compute_property_metrics(p, y, mask, spec)
        if spec.primary_metric not in metrics:
            raise RuntimeError(
                f"Validation cannot evaluate {spec.primary_metric}: check observed labels/classes "
                "and increase --limit-per-split if using smoke mode"
            )
        score = primary_score(metrics, spec)
        if not np.isfinite(score):
            raise RuntimeError("Validation primary metric is not finite; check split size and labels")
        history.append({"epoch": epoch, **summarize_steps(logs),
                        "validation": metrics})
        write_json(out / "history.json", history)
        print(f"{variant} seed={seed} epoch={epoch} valid_{spec.primary_metric}={score:.5f}", flush=True)
        if best_score is None or better_score(score, best_score, spec):
            best_score, state, best_epoch, stale = score, deepcopy(model.state_dict()), epoch, 0
        else:
            stale += 1
        if stale >= args.patience:
            break
    model.load_state_dict(state)
    torch.save({"state_dict": state, "config": asdict(config), "variant": variant,
                "num_tasks": spec.num_tasks, "epoch": best_epoch,
                "target_scaler": None if scaler is None else {"center": scaler.center, "scale": scaler.scale}}, out / "best.pt")
    write_json(out / "history.json", history)
    result = {"variant": variant, "seed": seed, "best_epoch": best_epoch,
              "validation": history[best_epoch - 1]["validation"],
              "elapsed_seconds": time.perf_counter() - start,
              "protocol_sha256": protocol_sha256,
              "parameter_counts": parameter_counts,
              "parameters_with_grad_max": max(r["parameters_with_grad_max"] for r in history),
              "smoke_only": bool(args.limit_per_split)}
    # Test is read once, after validation selection, and can be omitted during development.
    if args.evaluate_test:
        p, y, mask, ids = predict(model_for_variant(model, variant), loaders["test"], args.device, scaler, spec)
        result["test"] = compute_property_metrics(p, y, mask, spec)
        save_predictions(out / "test_predictions.csv", ids, p, y, mask, spec.label_columns, source.smiles)
    write_json(out / "metrics.json", result)
    return result


class model_for_variant(torch.nn.Module):
    def __init__(self, model, variant):
        super().__init__()
        self.model, self.variant = model, variant

    def forward(self, batch):
        return self.model(batch, self.variant)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(TASK_SPECS), default="esol")
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "runs")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--atom-layers", type=int, default=3)
    parser.add_argument("--context-layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.)
    parser.add_argument("--gradient-diagnostics-every", type=int, default=10,
                        help="Sample loss-component gradients every N optimizer steps; total clipping logged every step")
    parser.add_argument("--denoise-weight", type=float, default=0.01)
    parser.add_argument("--noise-steps", type=int, default=100)
    parser.add_argument("--fixed-step", type=int, default=50)
    parser.add_argument("--corruption-rate", type=float, default=0.5)
    parser.add_argument("--limit-per-split", type=int, default=0, help="Smoke only: first N official indices per split")
    parser.add_argument("--evaluate-test", action="store_true", help="Evaluate test once after validation selection")
    args = parser.parse_args()
    if (min(args.epochs, args.patience, args.batch_size, args.gradient_diagnostics_every) < 1
            or args.limit_per_split < 0 or args.denoise_weight < 0 or args.grad_clip <= 0):
        parser.error("Invalid training budget or loss weight")
    source, splits, spec = build_property_data(args.data_root, args.dataset)
    if args.limit_per_split:
        splits = {k: v[:args.limit_per_split] for k, v in splits.items()}
    runtime = collect_provenance(ROOT)
    settings = {k: v for k, v in vars(args).items()
                if k not in ("variants", "seeds", "output_dir", "data_root")}
    protocol = {
        "schema_version": 2, "settings": settings,
        "data_sha256": hashlib.sha256((args.data_root / spec.directory / "mapping" / "mol.csv.gz").read_bytes()).hexdigest(),
        "split_sha256": fingerprint(splits), "source_sha256": runtime["source_sha256"],
        "software": runtime["software"], "partition": "nonoverlapping_brics_components_v1",
    }
    with output_session(args.output_dir, args.dataset):
        protocol_sha256 = prepare_cohort(args, protocol)
        dataset = MotifDataset(source)
        # Materialize once and retain counts for each official split.
        selected = sorted({i for indices in splits.values() for i in indices})
        counts = [int(dataset[i].motif_count.item()) for i in selected]
        save_partition_audit(args.output_dir, args.dataset, {
            "molecules": len(counts), "single_motif": counts.count(1),
            "mean_motifs": float(np.mean(counts)), "max_motifs": max(counts),
            "smoke_only": bool(args.limit_per_split), "protocol_sha256": protocol_sha256,
            "by_split": {name: {"molecules": len(ids), "single_motif": sum(
                int(dataset[i].motif_count.item()) == 1 for i in ids
            )} for name, ids in splits.items()},
        })
        try:
            for seed in args.seeds:
                for variant in args.variants:
                    run(args, source, splits, spec, dataset, variant, seed, runtime, protocol_sha256)
                    refresh_summaries(args.output_dir, args.dataset, protocol_sha256)
        finally:
            refresh_summaries(args.output_dir, args.dataset, protocol_sha256)


if __name__ == "__main__":
    main()
