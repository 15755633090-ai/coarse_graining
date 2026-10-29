"""One conditional atom-attribute diffusion pretraining task; no contrastive task."""
import argparse

import numpy as np
import pandas as pd
from rdkit import Chem

from .data import AtomDataset
from .model import AtomDiffusionModel
from .training import (atomic_write, common_arguments, digest, fit, initialize, loader, metadata, run_directory)


def read_smiles(path, column):
    frame = pd.read_csv(path)
    if column not in frame or frame[column].isna().any():
        raise ValueError("Missing SMILES column or empty values")
    unique = {}
    for value in frame[column].astype(str):
        mol = Chem.MolFromSmiles(value)
        if mol is None or mol.GetNumAtoms() == 0:
            raise ValueError(f"Invalid SMILES in corpus: {value}")
        canonical = Chem.MolToSmiles(mol, canonical=True)
        unique.setdefault(canonical, canonical)
    return list(unique)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    from pathlib import Path
    parser.add_argument("--smiles-csv", type=Path, required=True)
    parser.add_argument("--smiles-column", default="smiles")
    parser.add_argument("--valid-csv", type=Path)
    parser.add_argument("--valid-fraction", type=float, default=0.1)
    args = parser.parse_args()
    config = initialize(args)
    smiles = read_smiles(args.smiles_csv, args.smiles_column)
    if args.valid_csv:
        train_smiles, valid_smiles = smiles, read_smiles(args.valid_csv, args.smiles_column)
        if set(train_smiles) & set(valid_smiles):
            raise ValueError("Canonical duplicate molecules across pretraining train/validation corpora")
    else:
        if not 0 < args.valid_fraction < 1 or len(smiles) < 2:
            raise ValueError("Need at least two distinct molecules and 0 < valid_fraction < 1")
        indices = np.random.default_rng(args.seed).permutation(len(smiles))
        n = min(len(smiles) - 1, max(1, round(len(smiles) * args.valid_fraction)))
        valid_smiles = [smiles[i] for i in indices[:n]]
        train_smiles = [smiles[i] for i in indices[n:]]
    if args.limit:
        train_smiles, valid_smiles = train_smiles[:args.limit], valid_smiles[:args.limit]
    if not train_smiles or not valid_smiles:
        raise ValueError("Nonempty pretraining training and validation corpora are required")
    inputs = {"train_csv_sha256": digest(args.smiles_csv),
              "valid_csv_sha256": digest(args.valid_csv) if args.valid_csv else None,
              "train_smiles": train_smiles, "valid_smiles": valid_smiles}
    # This records the exact corpus and split; no downstream-label data are read.
    meta = metadata(args, config, inputs, "pretrain")
    with run_directory(args, meta):
        combined = train_smiles + valid_smiles
        dataset = AtomDataset(combined)
        train_loader = loader(dataset, list(range(len(train_smiles))), args, True)
        valid_loader = loader(dataset, list(range(len(train_smiles), len(combined))), args, False)
        model = AtomDiffusionModel(config).to(args.device)
        result = fit(model, train_loader, valid_loader, args, "pretrain")
        result.update(stage="pretrain", smoke_only=bool(args.limit), train_molecules=len(train_smiles),
                      validation_molecules=len(valid_smiles), test_evaluated=False)
        atomic_write(args.output_dir / "completion.json", result)
        atomic_write(args.output_dir / "status.json", {"state": "complete", **result})


if __name__ == "__main__":
    main()
