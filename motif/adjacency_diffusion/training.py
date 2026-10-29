"""Own checkpoint schema/objective, reusing audited generic IO and task evaluation."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from motif.brics_atom_diffusion import training as shared
from .data import FEATURE_SCHEMA
from .model import Config
from .diffusion import Schedule
from .metrics import summarize

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "paper_mechanism_binary_adjacency_reconstruction_v1"
atomic_write, digest = shared.atomic_write, shared.digest
run_directory, loader = shared.run_directory, shared.loader
TASK_SPECS, build_property_data = shared.TASK_SPECS, shared.build_property_data
TargetScaler, predict = shared.TargetScaler, shared.predict
compute_property_metrics, save_predictions = shared.compute_property_metrics, shared.save_predictions
classification_positive_weight = shared.classification_positive_weight


def arguments(parser):
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-5)
    parser.add_argument("--weight-decay", type=float, default=0.)
    parser.add_argument("--grad-clip", type=float, default=5.)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=.1)
    parser.add_argument("--noise-steps", type=int, default=1000)
    parser.add_argument("--beta-min", type=float, default=1e-4)
    parser.add_argument("--beta-max", type=float, default=.02)
    parser.add_argument("--time-mode", choices=("terminal", "random"), default="terminal")
    parser.add_argument("--motif-prior", action="store_true")
    parser.add_argument("--conditional-gate", action="store_true")
    parser.add_argument("--single-state", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true")


def initialize(args):
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.limit < 0:
        raise ValueError("Invalid budget")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("Invalid optimizer settings")
    torch.set_num_threads(args.threads)
    shared.seed_everything(args.seed, deterministic=args.device == "cpu")
    return Config(hidden_dim=args.hidden_dim, layers=args.layers, dropout=args.dropout,
                  noise_steps=args.noise_steps, beta_min=args.beta_min, beta_max=args.beta_max,
                  motif_prior=args.motif_prior, conditional_gate=args.conditional_gate,
                  dual_state=not args.single_state, time_mode=args.time_mode)


def metadata(args, config, inputs, stage):
    source = shared.provenance()
    source["source_sha256"].update({p.relative_to(ROOT).as_posix(): digest(p) for p in Path(__file__).parent.glob("*.py")})
    return {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA, "stage": stage,
            "args": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"},
            "model": asdict(config), "inputs": inputs,
            "noise_schedule": Schedule(config.noise_steps, config.beta_min, config.beta_max).description(),
            "target": "binary adjacency; full off-diagonal Frobenius sum per graph, batch mean",
            "encoder_conditions": "clean eight-attribute X; noisy complete candidate adjacency; no clean bond/endpoints",
            "extra_structural_prior": "clean BRICS membership on ALL candidate pairs" if config.motif_prior else None,
            "paper_alignment": "priming mechanism only; custom encoder, symmetric noise, masked reduction and beta endpoints disclosed",
            "random_t_is_extension": config.time_mode != "terminal", "smoke_only": bool(args.limit), **source}


def load_checkpoint(path):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    if saved.get("schema") != SCHEMA or saved.get("feature_schema") != FEATURE_SCHEMA:
        raise ValueError("Not an adjacency checkpoint; attribute/latent/MotiL checkpoints are incompatible")
    return saved


@torch.no_grad()
def evaluate(model, batches, args):
    model.eval()
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 100003)
    statistics, truth, scores = {}, [], []
    for batch in batches:
        _, row, curves = model.reconstruction_loss(batch.to(args.device), generator, curves=True)
        for key, value in row.items():
            statistics[key] = statistics.get(key, 0) + value
        truth.append(curves[0])
        scores.append(curves[1])
    result = summarize(statistics)
    y, score = torch.cat(truth).numpy(), torch.cat(scores).numpy()
    result["all"]["roc_auc"] = float(roc_auc_score(y, score)) if len(set(y.tolist())) == 2 else None
    result["all"]["average_precision"] = float(average_precision_score(y, score)) if len(set(y.tolist())) == 2 else None
    return result


def fit(model, train_loader, valid_loader, args, stage, spec=None, scaler=None, weight=None):
    model.set_stage(stage)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    # Adam (not AdamW), as in the paper priming description; other settings recorded.
    optimizer = torch.optim.Adam(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 7919)
    run_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    config_hash = digest(args.output_dir / "config.json")
    history, best, best_epoch, best_state, stale, first = [], None, 0, None, 0, 1
    if args.resume:
        saved = load_checkpoint(args.output_dir / "latest.pt")
        if saved["stage"] != stage or saved["config"] != asdict(model.config) or saved["run_config_sha256"] != config_hash:
            raise ValueError("Resume identity mismatch")
        best_state = shared.authoritative_best(saved)
        atomic_write(args.output_dir / "best.pt", best_state, binary=True)
        atomic_write(args.output_dir / "history.json", saved["history"])
        model.load_state_dict(saved["model"], strict=True)
        optimizer.load_state_dict(saved["optimizer"])
        history, best, best_epoch, stale = saved["history"], saved["best_score"], saved["best_epoch"], saved["stale"]
        first = saved["epoch"] + 1
        torch.set_rng_state(saved["torch_rng"])
        if args.device.startswith("cuda"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
        generator.set_state(saved["noise_rng"])
        train_loader.generator.set_state(saved["loader_rng"])
    started = time.perf_counter()
    for epoch in range(first, args.epochs+1):
        if stale >= args.patience:
            break
        model.train()
        total, steps, norm_sum, clipped, statistics = 0., 0, 0., 0, {}
        observed = 0
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            if stage == "pretrain":
                loss, row = model.reconstruction_loss(batch, generator)
                for key, value in row.items():
                    statistics[key] = statistics.get(key, 0) + value
            else:
                loss = shared.masked_loss(model(batch), batch.y, batch.target_mask, spec, scaler, weight)
                if loss is None:
                    continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite objective")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True))
            observed = max(observed, sum(p.numel() for p in parameters if p.grad is not None))
            optimizer.step()
            steps, total, norm_sum = steps+1, total+float(loss.detach()), norm_sum+norm
            clipped += norm+1e-6 > args.grad_clip
        if not steps:
            raise ValueError("No training batches/labels")
        if stage == "pretrain":
            validation = evaluate(model, valid_loader, args)
            score = validation["all"]["reconstruction_frobenius"]
            improved = best is None or score < best
        else:
            pred, target, mask, _ = predict(model, valid_loader, args.device, scaler, spec)
            validation = compute_property_metrics(pred, target, mask, spec)
            if spec.primary_metric not in validation:
                raise ValueError("Undefined validation metric; check classes/sample size")
            score = shared.primary_score(validation, spec)
            improved = best is None or shared.better_score(score, best, spec)
        if score is None or not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("Nonfinite validation")
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        row = {"epoch": epoch, "train_loss_batch_mean": total/steps, "pre_clip_norm": norm_sum/steps,
               "clip_fraction": clipped/steps, "optimizer_steps": steps,
               "observed_gradient_parameters": observed, "validation": validation}
        if stage == "pretrain":
            row["train_recovery"] = summarize(statistics)
        history.append(row)
        state = {"schema": SCHEMA, "feature_schema": FEATURE_SCHEMA, "checkpoint_protocol": 2,
                 "stage": stage, "config": asdict(model.config), "num_tasks": model.num_tasks,
                 "run_metadata": run_config, "run_config_sha256": config_hash,
                 "model": model.state_dict(), "encoder": model.encoder.state_dict(), "epoch": epoch,
                 "best_score": best, "best_epoch": best_epoch, "stale": stale,
                 "noise_schedule": model.schedule.description(),
                 "target_scaler": None if scaler is None else {"center": scaler.center, "scale": scaler.scale},
                 "parameter_counts": {"instantiated": sum(p.numel() for p in model.parameters()),
                                      "optimized": sum(p.numel() for p in parameters)}}
        if improved:
            best_state = shared.cpu_snapshot(state)
        latest = {**state, "best_checkpoint": best_state, "history": history, "optimizer": optimizer.state_dict(),
                  "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                  "noise_rng": generator.get_state(), "loader_rng": train_loader.generator.get_state()}
        atomic_write(args.output_dir / "latest.pt", latest, binary=True)
        if improved:
            atomic_write(args.output_dir / "best.pt", best_state, binary=True)
        atomic_write(args.output_dir / "history.json", history)
        atomic_write(args.output_dir / "status.json", {"state": "running", "epoch": epoch, "best_epoch": best_epoch,
                                                       "elapsed_seconds_this_process": time.perf_counter()-started})
        print(f"{stage} epoch={epoch} validation={score:.6f} best={best:.6f}", flush=True)
    if best_state is None:
        raise ValueError("No committed checkpoint")
    atomic_write(args.output_dir / "best.pt", best_state, binary=True)
    model.load_state_dict(best_state["model"], strict=True)
    return {"best_epoch": best_state["best_epoch"], "best_score": best_state["best_score"],
            "epochs_run": history[-1]["epoch"], "parameter_counts": best_state["parameter_counts"]}
