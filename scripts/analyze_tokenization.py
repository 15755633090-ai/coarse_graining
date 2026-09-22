"""Emit the agreed partition statistics before training."""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diffusion_encoder.data import OGBMoleculePropertyDataset
from multiscale_tokenizer.partition import (
    derive_partition_seed,
    partition_molecule,
)


def _mean(values: list[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--split", choices=("train", "valid", "test", "all"), default="train")
    parser.add_argument("--partition-seed", type=int, default=100_000)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    split = None if args.split == "all" else args.split
    dataset = OGBMoleculePropertyDataset(args.dataset_root, split=split)
    rows = []
    for position, record in enumerate(dataset):
        if args.limit and position >= args.limit:
            break
        seed = derive_partition_seed(args.partition_seed, record.sample_id, 0)
        partition = partition_molecule(
            record.graph.bonds,
            node_mask=None,
            seed=seed,
        )
        rows.append(dict(partition.stats))

    summary: dict[str, float | int] = {
        "molecules": len(rows),
        "mean_diameter": _mean([float(row["diameter"]) for row in rows]),
        "mean_q_fraction": _mean([float(row["q_fraction"]) for row in rows]),
    }
    for level in (2, 3, 4):
        summary[f"p_level{level}_present"] = _mean([
            float(int(row[f"level{level}_tokens"]) > 0) for row in rows
        ])
        summary[f"mean_level{level}_tokens"] = _mean([
            float(row[f"level{level}_tokens"]) for row in rows
        ])
        nodes = [float(row[f"level{level}_nodes"]) for row in rows]
        tokens = [float(row[f"level{level}_tokens"]) for row in rows]
        total_nodes = sum(nodes)
        total_tokens = sum(tokens)
        summary[f"level{level}_atoms_per_token"] = (
            total_nodes / total_tokens if total_tokens else 0.0
        )
        summary[f"mean_level{level}_atoms_per_token_when_present"] = _mean([
            node_count / token_count
            for node_count, token_count in zip(nodes, tokens)
            if token_count > 0
        ])
        summary[f"mean_level{level}_residual_rate"] = _mean([
            float(row[f"level{level}_residual_rate"]) for row in rows
        ])
    summary["mean_level4_rounds"] = _mean([float(row["level4_rounds"]) for row in rows])

    print(json.dumps(summary, indent=2))
    if args.output:
        args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
