"""Precompute molecular coarse topologies on CPU without loading model weights."""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import torch

from coarse_gnn import CoarseningConfig, TopologyCache
from coarse_gnn.diffusion_adapter import precompute_batch_topologies
from diffusion.bond_diffusion.data import JsonlGraphDataset, SmilesDataset, collate_graphs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="Existing graph JSONL, SMILES CSV, .smi or .txt dataset")
    source.add_argument("--smiles", nargs="+", help="Explicit molecular examples")
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/topology_cache"))
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--radius", type=int, default=4)
    parser.add_argument("--center-fraction", type=float, default=0.1)
    parser.add_argument("--max-residual-size", type=int, default=4)
    parser.add_argument("--legacy-coarsening", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    torch.set_num_threads(2)
    config = CoarseningConfig(args.radius, args.center_fraction, args.max_residual_size,
                              canonicalize=not args.legacy_coarsening)
    if args.smiles is not None:
        dataset = SmilesDataset(args.smiles)
    elif args.input.suffix.lower() == ".jsonl":
        dataset = JsonlGraphDataset(args.input)
    else:
        dataset = SmilesDataset.from_file(args.input, args.smiles_column)
    if not len(dataset):
        parser.error("Dataset contains no graphs")
    cache = TopologyCache(args.cache_dir)
    started = time.perf_counter()
    for start in range(0, len(dataset), args.batch_size):
        stop = min(start + args.batch_size, len(dataset))
        batch = collate_graphs([dataset[i] for i in range(start, stop)])
        precompute_batch_topologies(batch, cache, config)
        print(f"Precomputed {stop}/{len(dataset)} graphs; {cache.stats}", flush=True)
    report = dict(graphs=len(dataset), coarsening=asdict(config), cache=cache.stats,
                  seconds=time.perf_counter() - started,
                  input=str(args.input) if args.input else args.smiles)
    path = args.cache_dir / "precompute_report.json"
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Report: {path.resolve()}")


if __name__ == "__main__":
    main()
