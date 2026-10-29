"""Paper-mechanism-aligned binary adjacency priming, with disclosed adaptations."""
import argparse
from pathlib import Path
import numpy as np

from motif.brics_atom_diffusion.pretrain import read_smiles
from .data import GraphDataset
from .model import AdjacencyModel
from .training import arguments, initialize, metadata, run_directory, loader, fit, digest, atomic_write


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--smiles-csv", type=Path, required=True)
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--valid-fraction", type=float, default=.1)
    args = parser.parse_args()
    config = initialize(args)
    molecules = read_smiles(args.smiles_csv, args.smiles_column)
    if args.valid_csv:
        train, valid = molecules, read_smiles(args.valid_csv, args.smiles_column)
        if set(train) & set(valid):
            raise ValueError("Duplicate molecules across train/validation")
    else:
        if len(molecules) < 2 or not 0 < args.valid_fraction < 1:
            raise ValueError("Need >=2 molecules and a valid validation fraction")
        order = np.random.default_rng(args.seed).permutation(len(molecules))
        n = min(len(molecules)-1, max(1, round(len(molecules)*args.valid_fraction)))
        valid, train = [molecules[i] for i in order[:n]], [molecules[i] for i in order[n:]]
    if args.limit:
        train, valid = train[:args.limit], valid[:args.limit]
    if not train or not valid:
        raise ValueError("Empty corpus split")
    inputs = {"train_smiles": train, "valid_smiles": valid, "train_csv_sha256": digest(args.smiles_csv),
              "valid_csv_sha256": digest(args.valid_csv) if args.valid_csv else None}
    with run_directory(args, metadata(args, config, inputs, "pretrain")):
        data = GraphDataset(train+valid)
        train_batches = loader(data, list(range(len(train))), args, True)
        valid_batches = loader(data, list(range(len(train), len(train)+len(valid))), args, False)
        model = AdjacencyModel(config).to(args.device)
        result = fit(model, train_batches, valid_batches, args, "pretrain")
        result.update(stage="pretrain", smoke_only=bool(args.limit), test_evaluated=False,
                      train_molecules=len(train), validation_molecules=len(valid))
        atomic_write(args.output_dir / "completion.json", result)
        atomic_write(args.output_dir / "status.json", {"state": "complete", **result})


if __name__ == "__main__":
    main()
