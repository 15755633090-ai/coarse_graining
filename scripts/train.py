"""CLI entry point for the four-arm property-training protocol."""
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--encoder-learning-rate", type=float, default=1e-5)
    parser.add_argument("--downstream-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--scheduler-patience", type=int, default=10)
    parser.add_argument("--scheduler-factor", type=float, default=0.3)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--partition-seed", type=int, default=100_000)
    parser.add_argument("--data-seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--patience", type=int, default=25)
    args = parser.parse_args()
    result = train_property_model(
        args.dataset_root,
        task=args.task,
        output_dir=args.output_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        encoder_learning_rate=args.encoder_learning_rate,
        downstream_learning_rate=args.downstream_learning_rate,
        weight_decay=args.weight_decay,
        scheduler_patience=args.scheduler_patience,
        scheduler_factor=args.scheduler_factor,
        model_seed=args.model_seed,
        partition_seed=args.partition_seed,
        data_seed=args.data_seed,
        dropout=args.dropout,
        num_workers=args.num_workers,
        resume=args.resume,
        patience=args.patience,
        experiment_mode=args.mode,
    )
    summary = {
        key: result[key]
        for key in (
            "task_spec", "experiment_mode", "model_seed", "partition_seed",
            "data_seed", "best_epoch", "test_metrics",
        )
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
