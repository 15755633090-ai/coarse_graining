"""Shared execution, checkpoint and evaluation utilities for the new atom model."""
import argparse
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import uuid

import torch
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader

from .data import SCHEMA
from .model import Config
from .diffusion import GaussianSchedule
from .metrics import summarize_recovery

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "model"))
from benchmark import (  # noqa: E402
    TASK_SPECS, TargetScaler, better_score, build_property_data, classification_positive_weight,
    compute_property_metrics, masked_loss, predict, primary_score, save_predictions, seed_everything,
)

CHECKPOINT_SCHEMA = "brics_atom_conditional_epsilon_v1"


def replace_with_retry(source, target):
    for attempt in range(12):
        try:
            source.replace(target)
            return
        except PermissionError as error:
            if getattr(error, "winerror", None) not in (5, 32, 33) or attempt == 11:
                raise
            time.sleep(min(0.05 * 2 ** attempt, 1.0))


def atomic_write(path, value, binary=False):
    temporary = path.with_name(path.name + f".{uuid.uuid4().hex}.tmp")
    try:
        if binary:
            torch.save(value, temporary)
        else:
            temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")
        replace_with_retry(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def provenance():
    paths = list(Path(__file__).parent.glob("*.py"))
    paths += [ROOT / "model" / name for name in ("benchmark.py", "gine.py", "pna.py", "morgan.py", "attentivefp.py")]
    paths += list((ROOT / "扩散GNN/01_扩散预训练/bond_diffusion").glob("*.py"))
    versions = {}
    for name in ("torch", "torch-geometric", "rdkit", "ogb", "numpy", "pandas", "scikit-learn"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {"source_sha256": {p.relative_to(ROOT).as_posix(): digest(p) for p in paths},
            "software": {"python": platform.python_version(), **versions, "cuda": torch.version.cuda}}


def common_arguments(parser):
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-5)
    parser.add_argument("--grad-clip", type=float, default=5.)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--noise-steps", type=int, default=1000)
    parser.add_argument("--single-state", action="store_true")
    parser.add_argument("--no-motif-prior", action="store_true")
    parser.add_argument("--uniform-boundary", action="store_true")
    parser.add_argument("--limit", type=int, default=0, help="Smoke only: limit samples per split")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true", help="Restore latest.pt including optimizer and RNG states")


def initialize(args):
    if min(args.epochs, args.patience, args.batch_size, args.threads) < 1 or args.limit < 0:
        raise ValueError("Invalid training budget")
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.grad_clip <= 0:
        raise ValueError("Invalid optimization settings")
    torch.set_num_threads(args.threads)
    seed_everything(args.seed, deterministic=args.device == "cpu")
    return Config(hidden_dim=args.hidden_dim, layers=args.layers, dropout=args.dropout,
                  noise_steps=args.noise_steps, dual_state=not args.single_state,
                  motif_prior=not args.no_motif_prior, conditional_gate=not args.uniform_boundary)


def loader(dataset, indices, args, shuffle):
    return DataLoader(Subset(dataset, indices), batch_size=args.batch_size, shuffle=shuffle,
                      num_workers=0, generator=torch.Generator().manual_seed(args.seed))


@contextmanager
def run_directory(args, metadata):
    out = args.output_dir
    if not args.resume:
        out.mkdir(parents=True, exist_ok=False)
        atomic_write(out / "config.json", metadata)
    else:
        if not (out / "latest.pt").exists():
            raise ValueError("Resume requires latest.pt")
        if (out / "completion.json").exists():
            raise ValueError("Run already complete; refuse repeated training/test evaluation")
        if json.loads((out / "config.json").read_text(encoding="utf-8")) != metadata:
            raise ValueError("Resume configuration/data/source differs")
    lock = out / ".lock"
    handle = lock.open("x", encoding="utf-8")
    try:
        with handle:
            handle.write(str(os.getpid()))
        yield
    except Exception as error:
        atomic_write(out / "status.json", {"state": "failed", "error": repr(error)})
        raise
    finally:
        lock.unlink()


def metadata(args, config, inputs, stage):
    settings = {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in vars(args).items() if k != "resume"}
    return {"schema": CHECKPOINT_SCHEMA, "feature_schema": SCHEMA, "stage": stage,
            "noise_schedule": GaussianSchedule(config.noise_steps).description(),
            "motif_ablation": "full motif relation mechanism removed; both paths use full graph; no boundary gating"
                if not config.motif_prior else "BRICS internal/cross relation mechanism enabled",
            "args": settings, "model": asdict(config), "inputs": inputs,
            "noise_target": "epsilon; MSE mean over all atoms and attribute channels",
            "structural_conditions": "clean topology, clean bond features, BRICS edge relations and boundary flags",
            "environment_claim": "internal state isolated from external input values, conditional on clean chemistry and boundary flags",
            "smoke_only": bool(args.limit), **provenance()}


def load_checkpoint(path):
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if checkpoint.get("schema") != CHECKPOINT_SCHEMA or checkpoint.get("feature_schema") != SCHEMA:
        raise ValueError("Incompatible checkpoint; old motif latent/MotiL weights are not accepted")
    return checkpoint


def cpu_snapshot(value):
    """The retained best state must not share mutable model/optimizer storage."""
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {k: cpu_snapshot(v) for k, v in value.items()}
    if isinstance(value, list):
        return [cpu_snapshot(v) for v in value]
    if isinstance(value, tuple):
        return tuple(cpu_snapshot(v) for v in value)
    return deepcopy(value)


def authoritative_best(latest):
    if latest.get("checkpoint_protocol") != 2 or "best_checkpoint" not in latest:
        raise ValueError("Legacy latest.pt has no authoritative best snapshot; unsafe automatic resume is refused")
    best = latest["best_checkpoint"]
    if any(best.get(key) != latest.get(key) for key in ("schema", "feature_schema", "stage", "config", "run_config_sha256")):
        raise ValueError("Embedded best checkpoint belongs to another run/model")
    if best["epoch"] != latest["best_epoch"] or best["best_epoch"] != latest["best_epoch"] or best["best_score"] != latest["best_score"]:
        raise ValueError("Embedded best checkpoint/version inconsistent with latest.pt")
    if best["epoch"] > latest["epoch"] or latest["history"][-1]["epoch"] != latest["epoch"]:
        raise ValueError("Invalid latest checkpoint epoch/history")
    return best


@torch.no_grad()
def evaluate_noise(model, batches, args):
    model.eval()
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 100003)
    sums = {}
    for batch in batches:
        _, stats = model.denoise_loss(batch.to(args.device), generator)
        for name, value in stats.items():
            sums[name] = sums.get(name, 0) + value
    return {**{name + "_mse": sums[name + "_error_sum"] / sums[name + "_atoms"]
               if sums.get(name + "_atoms", 0) else None for name in ("all", "boundary", "internal")},
            "recovery": summarize_recovery(sums)}


def fit(model, train_loader, valid_loader, args, stage, spec=None, scaler=None, positive_weight=None):
    model.set_stage(stage)
    parameters = tuple(p for p in model.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    generator = torch.Generator(device=args.device).manual_seed(args.seed + 7919)
    history, best, best_epoch, stale, first = [], None, 0, 0, 1
    best_checkpoint = None
    run_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
    run_config_sha256 = digest(args.output_dir / "config.json")
    if args.resume:
        saved = load_checkpoint(args.output_dir / "latest.pt")
        if saved["stage"] != stage or saved["config"] != asdict(model.config):
            raise ValueError("Resume stage/model mismatch")
        if saved.get("run_config_sha256") != run_config_sha256:
            raise ValueError("Checkpoint does not belong to this run configuration")
        best_checkpoint = authoritative_best(saved)
        # best.pt/history.json are caches. Roll them forward or back to the
        # committed latest.pt even if an export was interrupted or ahead.
        atomic_write(args.output_dir / "best.pt", best_checkpoint, binary=True)
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
    start = time.perf_counter()
    for epoch in range(first, args.epochs + 1):
        if stale >= args.patience:
            break
        model.train()
        loss_sum, steps, clipped, grad_sum, observed = 0., 0, 0, 0., 0
        groups = {}
        for batch in train_loader:
            batch = batch.to(args.device)
            optimizer.zero_grad(set_to_none=True)
            if stage == "pretrain":
                loss, stats = model.denoise_loss(batch, generator)
                for k, v in stats.items():
                    groups[k] = groups.get(k, 0) + v
            else:
                loss = masked_loss(model(batch), batch.y, batch.target_mask, spec, scaler, positive_weight)
                if loss is None:
                    continue
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite training loss")
            loss.backward()
            norm = float(torch.nn.utils.clip_grad_norm_(parameters, args.grad_clip, error_if_nonfinite=True))
            observed = max(observed, sum(p.numel() for p in parameters if p.grad is not None))
            optimizer.step()
            loss_sum += float(loss.detach())
            steps += 1
            clipped += norm + 1e-6 > args.grad_clip
            grad_sum += norm
        if not steps:
            raise ValueError("No training batches with valid targets")
        if stage == "pretrain":
            valid_metrics = evaluate_noise(model, valid_loader, args)
            score = valid_metrics["all_mse"]
            improved = best is None or score < best
        else:
            prediction, target, mask, _ = predict(model, valid_loader, args.device, scaler, spec)
            valid_metrics = compute_property_metrics(prediction, target, mask, spec)
            if spec.primary_metric not in valid_metrics:
                raise ValueError("Validation primary metric is undefined; check observed classes and sample limit")
            score = primary_score(valid_metrics, spec)
            improved = best is None or better_score(score, best, spec)
        if score is None or not torch.isfinite(torch.tensor(score)):
            raise FloatingPointError("Invalid validation score")
        if improved:
            best, best_epoch, stale = score, epoch, 0
        else:
            stale += 1
        record = {"epoch": epoch, "train_loss_batch_mean": loss_sum / steps,
                  "pre_clip_gradient_norm": grad_sum / steps, "clip_fraction": clipped / steps,
                  "optimizer_steps": steps, "observed_gradient_parameters": observed,
                  "validation": valid_metrics}
        if stage == "pretrain":
            record["train_noise_mse"] = {
                name: groups[name + "_error_sum"] / groups[name + "_atoms"] if groups.get(name + "_atoms", 0) else None
                for name in ("all", "boundary", "internal")}
            record["train_recovery"] = summarize_recovery(groups)
        history.append(record)
        state = {"schema": CHECKPOINT_SCHEMA, "feature_schema": SCHEMA, "stage": stage,
                 "run_config_sha256": run_config_sha256, "run_metadata": run_config,
                 "checkpoint_protocol": 2, "noise_schedule": model.schedule.description(),
                 "config": asdict(model.config), "num_tasks": model.num_tasks,
                 "model": model.state_dict(), "encoder": model.encoder.state_dict(),
                 "epoch": epoch, "best_score": best, "best_epoch": best_epoch, "stale": stale,
                 "target_scaler": None if scaler is None else {"center": scaler.center, "scale": scaler.scale},
                 "parameter_counts": {"instantiated": sum(p.numel() for p in model.parameters()),
                                      "optimized": sum(p.numel() for p in parameters)}}
        if improved:
            best_checkpoint = cpu_snapshot(state)
        latest = {**state, "optimizer": optimizer.state_dict(), "history": history,
                  "best_checkpoint": best_checkpoint,
                  "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all() if args.device.startswith("cuda") else [],
                  "noise_rng": generator.get_state(), "loader_rng": train_loader.generator.get_state()}
        atomic_write(args.output_dir / "latest.pt", latest, binary=True)
        # The latest replacement above is the only epoch commit point. A crash
        # before/after this export is repaired exclusively from latest on resume.
        if improved:
            atomic_write(args.output_dir / "best.pt", best_checkpoint, binary=True)
        atomic_write(args.output_dir / "history.json", history)
        atomic_write(args.output_dir / "status.json", {"state": "running", "epoch": epoch, "best_epoch": best_epoch,
                                                       "best_score": best, "elapsed_seconds_this_process": time.perf_counter() - start})
        print(f"{stage} epoch={epoch} validation={score:.6f} best={best:.6f}", flush=True)
    checkpoint = best_checkpoint
    if checkpoint is None:
        raise ValueError("No committed best checkpoint")
    atomic_write(args.output_dir / "best.pt", checkpoint, binary=True)
    model.load_state_dict(checkpoint["model"], strict=True)
    return {"best_epoch": checkpoint["best_epoch"], "best_score": checkpoint["best_score"],
            "epochs_run": history[-1]["epoch"], "parameter_counts": checkpoint["parameter_counts"]}
