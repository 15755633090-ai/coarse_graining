from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from torch.utils.data import DataLoader, Dataset, random_split

from bond_diffusion.config import DiffusionConfig, ModelConfig, TrainConfig
from bond_diffusion.data import (
    DOUBLE,
    SINGLE,
    JsonlGraphDataset,
    MoleculeGraph,
    SmilesDataset,
    collate_graphs,
    infer_substructure_mask,
)
from bond_diffusion.diffusion import DiscreteGraphDiffusion
from bond_diffusion.losses import ChemicalDiffusionLoss
from bond_diffusion.model import BondAwareDiffusionModel
from bond_diffusion.trainer import (
    ResumableBatchSampler,
    restore_rng_state,
    run_epoch,
    save_checkpoint,
    seed_everything,
)


def toy_graph(atom_types: List[int], edge_list: List[tuple[int, int, int]]) -> MoleculeGraph:
    n = len(atom_types)
    bonds = torch.zeros(n, n, dtype=torch.long)
    for i, j, bond_type in edge_list:
        bonds[i, j] = bonds[j, i] = bond_type
    degree = (bonds > 0).sum(dim=1).clamp_max(6)
    aromatic = torch.zeros(n, dtype=torch.long)
    hybrid = torch.tensor([2 if atom == 6 else 3 for atom in atom_types])
    features = torch.stack(
        [
            torch.tensor(atom_types),
            torch.full((n,), 5),  # neutral charge encoded as +5
            aromatic,
            hybrid,
            degree,
        ],
        dim=1,
    )
    return MoleculeGraph(features, bonds, infer_substructure_mask(features, bonds))


def demo_dataset() -> List[MoleculeGraph]:
    """Tiny tensor-only dataset used by CI and installation smoke tests."""

    motifs = [
        toy_graph([6, 6, 8], [(0, 1, SINGLE), (1, 2, SINGLE)]),  # ethanol skeleton
        toy_graph([6, 6, 8], [(0, 1, SINGLE), (1, 2, DOUBLE)]),  # acetaldehyde
        toy_graph([8, 6, 8], [(0, 1, DOUBLE), (1, 2, SINGLE)]),  # carboxyl motif
        toy_graph([6, 7, 6], [(0, 1, SINGLE), (1, 2, SINGLE)]),  # amine
        toy_graph([6, 6, 6, 8], [(0, 1, SINGLE), (1, 2, SINGLE), (2, 3, SINGLE)]),
        toy_graph([6, 8], [(0, 1, SINGLE)]),
    ]
    return motifs * 4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bond-aware molecular graph diffusion pretraining")
    parser.add_argument("--input", type=Path, help=".csv/.smi/.txt (SMILES) or preprocessed .jsonl")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--demo", action="store_true", help="run on a built-in tensor dataset")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--hidden-dim", type=int, default=192)
    parser.add_argument("--num-layers", type=int, default=6)
    parser.add_argument("--timesteps", type=int, default=200)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="use CUDA BF16 autocast (enabled by default)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument(
        "--resume",
        type=Path,
        help="resume from last.pt; --epochs is the final total epoch count",
    )
    parser.add_argument(
        "--save-every-batches",
        type=int,
        default=250,
        help="write resume.pt every N training batches (default: 250)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.demo and args.input is None:
        raise SystemExit("Provide --input PATH or use --demo")
    resume_checkpoint = None
    start_epoch = 1
    start_batch = 0
    partial_totals = {}
    partial_steps = 0
    if args.resume is not None:
        resume_checkpoint = torch.load(
            args.resume, map_location="cpu", weights_only=False
        )
        model_cfg = ModelConfig(**resume_checkpoint["model_config"])
        diffusion_cfg = DiffusionConfig(**resume_checkpoint["diffusion_config"])
        train_cfg = TrainConfig(**resume_checkpoint["train_config"])
        saved_batch_size = train_cfg.batch_size
        if args.batch_size is not None:
            train_cfg.batch_size = args.batch_size
        if args.num_workers is not None:
            train_cfg.num_workers = args.num_workers
        if args.learning_rate is not None:
            train_cfg.learning_rate = args.learning_rate
        if args.amp is not None:
            train_cfg.use_amp = args.amp
        train_cfg.epochs = args.epochs
        if "resume_epoch" in resume_checkpoint:
            start_epoch = int(resume_checkpoint["resume_epoch"])
            start_batch = int(resume_checkpoint["next_batch"])
            partial_totals = dict(resume_checkpoint.get("partial_totals", {}))
            partial_steps = int(resume_checkpoint.get("partial_steps", start_batch))
            if train_cfg.batch_size != saved_batch_size:
                raise ValueError(
                    "Cannot change --batch-size inside a partially completed epoch. "
                    "Resume from last.pt to restart that epoch with the new batch size."
                )
        else:
            start_epoch = int(resume_checkpoint["epoch"]) + 1
        if start_epoch > train_cfg.epochs:
            raise ValueError(
                f"checkpoint is already at epoch {start_epoch - 1}; "
                f"--epochs must be at least {start_epoch}"
            )
        print(
            f"resuming {args.resume} from epoch {start_epoch}, batch {start_batch}; "
            f"target total epochs={train_cfg.epochs}"
        )
    else:
        model_cfg = ModelConfig(hidden_dim=args.hidden_dim, num_layers=args.num_layers)
        diffusion_cfg = DiffusionConfig(timesteps=args.timesteps)
        train_cfg = TrainConfig(
            epochs=args.epochs,
            batch_size=args.batch_size or 32,
            learning_rate=args.learning_rate or 2e-4,
            num_workers=args.num_workers or 0,
            use_amp=True if args.amp is None else args.amp,
            seed=args.seed,
        )
    seed_everything(train_cfg.seed)

    if args.demo:
        dataset: Dataset[MoleculeGraph] = demo_dataset()
    elif args.input.suffix.lower() == ".jsonl":
        dataset = JsonlGraphDataset(args.input)
    else:
        dataset = SmilesDataset.from_file(args.input, args.smiles_column)
    if len(dataset) < 2:
        raise ValueError("At least two molecules are required")
    val_size = max(1, round(len(dataset) * train_cfg.val_fraction))
    train_size = len(dataset) - val_size
    if resume_checkpoint is not None:
        saved_dataset_size = resume_checkpoint.get("dataset_size")
        saved_train_size = resume_checkpoint.get("train_size")
        if saved_dataset_size is not None and int(saved_dataset_size) != len(dataset):
            raise ValueError("Dataset size differs from the interrupted run")
        if saved_train_size is not None and int(saved_train_size) != train_size:
            raise ValueError("Training split size differs from the interrupted run")
    generator = torch.Generator().manual_seed(train_cfg.seed)
    train_set, val_set = random_split(dataset, [train_size, val_size], generator=generator)
    loader_options = dict(
        num_workers=train_cfg.num_workers,
        collate_fn=collate_graphs,
        pin_memory=args.device.startswith("cuda"),
    )
    if train_cfg.num_workers > 0:
        loader_options["persistent_workers"] = True
        loader_options["prefetch_factor"] = 2
    val_loader = DataLoader(
        val_set, batch_size=train_cfg.batch_size, shuffle=False, **loader_options
    )

    device = torch.device(args.device)
    model = BondAwareDiffusionModel(model_cfg).to(device)
    diffusion = DiscreteGraphDiffusion(diffusion_cfg, model_cfg)
    criterion = ChemicalDiffusionLoss(
        train_cfg.lambda_bond,
        train_cfg.lambda_valence,
        train_cfg.lambda_substructure,
        train_cfg.nonbond_weight,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=train_cfg.learning_rate,
        weight_decay=train_cfg.weight_decay,
    )
    if resume_checkpoint is not None:
        model.load_state_dict(resume_checkpoint["model_state"])
        optimizer.load_state_dict(resume_checkpoint["optimizer_state"])
        if args.learning_rate is not None:
            for parameter_group in optimizer.param_groups:
                parameter_group["lr"] = args.learning_rate
        restore_rng_state(resume_checkpoint)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    best_path = args.output_dir / "best.pt"
    if resume_checkpoint is not None and best_path.exists():
        best_checkpoint = torch.load(best_path, map_location="cpu", weights_only=False)
        best_val = float(best_checkpoint["metrics"]["total"])
    elif resume_checkpoint is not None:
        best_val = float(resume_checkpoint["metrics"]["total"])
    else:
        best_val = float("inf")
    history_path = args.output_dir / "history.json"
    if resume_checkpoint is not None and history_path.exists():
        history = json.loads(history_path.read_text(encoding="utf-8"))
        completed_epoch = int(resume_checkpoint["epoch"])
        history = [row for row in history if int(row["epoch"]) <= completed_epoch]
    else:
        history = []
    for epoch in range(start_epoch, train_cfg.epochs + 1):
        epoch_start_batch = start_batch if epoch == start_epoch else 0
        epoch_partial_totals = partial_totals if epoch == start_epoch else {}
        epoch_partial_steps = partial_steps if epoch == start_epoch else 0
        batch_sampler = ResumableBatchSampler(
            len(train_set),
            train_cfg.batch_size,
            train_cfg.seed,
            epoch,
            epoch_start_batch,
        )
        train_loader = DataLoader(
            train_set,
            batch_sampler=batch_sampler,
            generator=torch.Generator().manual_seed(train_cfg.seed + 1_000_000 + epoch),
            **loader_options,
        )
        total_batches = batch_sampler.total_batches

        def save_batch_checkpoint(
            next_batch: int, running_totals: dict[str, float]
        ) -> None:
            if (
                args.save_every_batches <= 0
                or next_batch % args.save_every_batches != 0
                or next_batch >= total_batches
            ):
                return
            running_metrics = {
                key: value / next_batch for key, value in running_totals.items()
            }
            save_checkpoint(
                args.output_dir / "resume.pt",
                model,
                optimizer,
                epoch - 1,
                model_cfg,
                diffusion_cfg,
                train_cfg,
                running_metrics,
                extra_state={
                    "resume_epoch": epoch,
                    "next_batch": next_batch,
                    "partial_totals": dict(running_totals),
                    "partial_steps": next_batch,
                    "dataset_size": len(dataset),
                    "train_size": train_size,
                },
            )
            print(
                f"checkpoint epoch={epoch:03d} batch={next_batch}/{total_batches}",
                flush=True,
            )

        train_metrics = run_epoch(
            model,
            diffusion,
            criterion,
            train_loader,
            device,
            optimizer,
            train_cfg.grad_clip,
            initial_totals=epoch_partial_totals,
            initial_steps=epoch_partial_steps,
            batch_callback=save_batch_checkpoint,
            use_amp=train_cfg.use_amp,
        )
        val_metrics = run_epoch(
            model,
            diffusion,
            criterion,
            val_loader,
            device,
            use_amp=train_cfg.use_amp,
        )
        row = {"epoch": epoch, "train": train_metrics, "val": val_metrics}
        history.append(row)
        history_path.write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch:03d} train={train_metrics['total']:.4f} "
            f"val={val_metrics['total']:.4f} bond={val_metrics['bond']:.4f} "
            f"valence={val_metrics['valence']:.4f}"
        )
        save_checkpoint(
            args.output_dir / "last.pt",
            model,
            optimizer,
            epoch,
            model_cfg,
            diffusion_cfg,
            train_cfg,
            val_metrics,
            extra_state={"dataset_size": len(dataset), "train_size": train_size},
        )
        save_checkpoint(
            args.output_dir / "resume.pt",
            model,
            optimizer,
            epoch,
            model_cfg,
            diffusion_cfg,
            train_cfg,
            val_metrics,
            extra_state={"dataset_size": len(dataset), "train_size": train_size},
        )
        if val_metrics["total"] < best_val:
            best_val = val_metrics["total"]
            save_checkpoint(
                args.output_dir / "best.pt",
                model,
                optimizer,
                epoch,
                model_cfg,
                diffusion_cfg,
                train_cfg,
                val_metrics,
            )


if __name__ == "__main__":
    main()
