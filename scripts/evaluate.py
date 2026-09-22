"""Evaluate a saved checkpoint on the fixed validation or test partition."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from multiscale_tokenizer.training import evaluate_property_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--split", choices=("valid", "test"), default="test")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    metrics = evaluate_property_model(
        args.checkpoint,
        dataset_root=args.dataset_root,
        task=args.task,
        split=args.split,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    rendered = json.dumps(metrics, indent=2)
    print(rendered)
    if args.output:
        args.output.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
