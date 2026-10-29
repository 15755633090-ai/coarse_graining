"""Clean adjacency prediction with strict adjacency-encoder checkpoint transfer."""
import argparse
from dataclasses import asdict
from pathlib import Path
from rdkit import Chem

from .data import GraphDataset
from .model import AdjacencyModel
from .training import (ROOT, TASK_SPECS, arguments, initialize, metadata, run_directory, load_checkpoint,
                       digest, build_property_data, loader, TargetScaler, classification_positive_weight,
                       fit, predict, compute_property_metrics, save_predictions, atomic_write)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    arguments(parser)
    parser.add_argument("--dataset", choices=sorted(TASK_SPECS), default="esol")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pretrained", type=Path)
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--allow-pretrain-overlap", action="store_true")
    args = parser.parse_args()
    config = initialize(args)
    source, full_splits, spec = build_property_data(args.data_root, args.dataset)
    splits = {k: v[:args.limit] if args.limit else v for k, v in full_splits.items()}
    checkpoint = load_checkpoint(args.pretrained) if args.pretrained else None
    inputs = {"mapping_sha256": digest(args.data_root/spec.directory/'mapping/mol.csv.gz'),
              "splits": splits, "pretrained_sha256": digest(args.pretrained) if args.pretrained else None}
    if checkpoint:
        if checkpoint["stage"] != "pretrain" or checkpoint["config"] != asdict(config):
            raise ValueError("Pretrain architecture/prior/objective configuration mismatch")
        corpus = checkpoint["run_metadata"]["inputs"]
        seen = set(corpus["train_smiles"]) | set(corpus["valid_smiles"])
        overlap = {k: [i for i in full_splits[k] if Chem.MolToSmiles(Chem.MolFromSmiles(source.smiles[i]), canonical=True) in seen]
                   for k in ("valid", "test")}
        inputs["pretrain_overlap"] = overlap
        if any(overlap.values()) and not args.allow_pretrain_overlap:
            raise ValueError("Pretraining overlaps downstream validation/test molecules")
    with run_directory(args, metadata(args, config, inputs, "finetune")):
        data = GraphDataset(source.smiles, source.targets, source.target_mask)
        batches = {k: loader(data, v, args, k == "train") for k, v in splits.items()}
        y, mask = source.targets[splits["train"]], source.target_mask[splits["train"]]
        if not mask.any(0).all():
            raise ValueError("Each task needs observed training labels")
        scaler = TargetScaler.fit(y, mask) if spec.task_type == "regression" else None
        weight = classification_positive_weight(y, mask) if spec.task_type == "classification" else None
        model = AdjacencyModel(config, spec.num_tasks).to(args.device)
        if checkpoint:
            model.encoder.load_state_dict(checkpoint["encoder"], strict=True)
        result = fit(model, batches["train"], batches["valid"], args, "finetune", spec, scaler, weight)
        pred, target, observed, _ = predict(model, batches["valid"], args.device, scaler, spec)
        result.update(validation=compute_property_metrics(pred, target, observed, spec), stage="finetune",
                      pretrained_encoder=bool(checkpoint), smoke_only=bool(args.limit), test_evaluated=False)
        if args.evaluate_test:
            pred, target, observed, ids = predict(model, batches["test"], args.device, scaler, spec)
            result.update(test=compute_property_metrics(pred, target, observed, spec), test_evaluated=True)
            save_predictions(args.output_dir/'test_predictions.csv', ids, pred, target, observed, spec.label_columns, source.smiles)
        atomic_write(args.output_dir/'completion.json', result)
        atomic_write(args.output_dir/'status.json', {"state": "complete", **result})


if __name__ == "__main__":
    main()
