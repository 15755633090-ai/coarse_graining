"""CLI entry point for property-training and frozen Stage-2 comparisons."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_tokenizer.training import train_property_model
from multiscale_tokenizer.model import EXPERIMENT_MODES


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--mode", choices=EXPERIMENT_MODES, required=True)
    parser.add_argument("--base-init-checkpoint", help="Stage-2 Base best.pt for zero-initialized residual modes")
    parser.add_argument(
        "--encoder-init-checkpoint",
        help=(
            "baseline_finetune checkpoint supplying the frozen encoder for "
            "a stage2_frozen mode; its downstream head is never loaded"
        ),
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="defaults to the locked formal protocol (200)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="defaults to 64 for finetune modes, 32 for frozen modes",
    )
    parser.add_argument(
        "--micro-batch-size", type=int, default=None,
        help="gradient-accumulation chunk; formal protocol uses 8",
    )
    parser.add_argument(
        "--encoder-learning-rate", type=float, default=None,
        help="defaults to 1e-4 for finetune modes, 0 for frozen modes",
    )
    parser.add_argument(
        "--downstream-learning-rate", type=float, default=None,
        help="defaults to 5e-4 for finetune modes, 1e-3 for frozen modes",
    )
    parser.add_argument(
        "--weight-decay", type=float, default=None,
        help="defaults to the locked formal protocol (1e-5)",
    )
    parser.add_argument(
        "--grad-clip", type=float, default=None,
        help="defaults to the locked formal protocol (5.0)",
    )
    parser.add_argument(
        "--amp-dtype", choices=("none", "bf16"), default=None,
        help="defaults to the locked formal protocol (bf16 autocast)",
    )
    parser.add_argument(
        "--scheduler", choices=("none", "plateau"), default=None,
        help="formal benchmark protocol uses a constant LR (none)",
    )
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument(
        "--dropout", type=float, default=None,
        help="defaults to 0.3 for finetune modes, 0.1 for frozen modes",
    )
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--partition-seed", type=int, default=100_000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--region-radius", type=int, default=2)
    parser.add_argument("--atoms-per-center", type=int, default=8)
    parser.add_argument("--max-centers", type=int, default=8)
    parser.add_argument("--eval-views", type=int, default=5)
    parser.add_argument(
        "--num-workers", type=int, default=0,
        help="formal benchmark execution used 0; keep paired arms identical",
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--patience", type=int, default=None,
        help="defaults to the locked formal protocol (30)",
    )
    parser.add_argument(
        "--train-diagnostics", action="store_true",
        help=(
            "collect per-micro-batch level/token statistics during training; "
            "the old formal benchmark did not, so this is off by default"
        ),
    )
    args = parser.parse_args()
    result = train_property_model(
        args.dataset_root,
        task=args.task,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        micro_batch_size=args.micro_batch_size,
        encoder_learning_rate=args.encoder_learning_rate,
        downstream_learning_rate=args.downstream_learning_rate,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        grad_clip=args.grad_clip,
        amp_dtype=args.amp_dtype,
        scheduler=args.scheduler,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        model_seed=args.model_seed,
        partition_seed=args.partition_seed,
        data_seed=args.data_seed,
        num_workers=args.num_workers,
        resume=args.resume,
        patience=args.patience,
        train_diagnostics=args.train_diagnostics,
        experiment_mode=args.mode,
        encoder_init_checkpoint=args.encoder_init_checkpoint,
        base_init_checkpoint=args.base_init_checkpoint,
        region_radius=args.region_radius, atoms_per_center=args.atoms_per_center,
        max_centers=args.max_centers, eval_views=args.eval_views,
    )
    summary = {
        key: result[key]
        for key in (
            "task_spec", "experiment_mode", "model_seed", "partition_seed",
            "data_seed", "training_protocol", "best_epoch", "test_metrics",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
