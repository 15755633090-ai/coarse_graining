"""Clean t=0 property prediction, with optional strict pretrained encoder transfer."""
import argparse
from dataclasses import asdict
from pathlib import Path
from rdkit import Chem

from .data import AtomDataset
from .model import AtomDiffusionModel
from .training import (
    ROOT, TASK_SPECS, TargetScaler, atomic_write, build_property_data, classification_positive_weight,
    common_arguments, compute_property_metrics, digest, fit, initialize, load_checkpoint, loader,
    metadata, predict, run_directory, save_predictions,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    common_arguments(parser)
    parser.add_argument("--dataset", choices=sorted(TASK_SPECS), default="esol")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data")
    parser.add_argument("--pretrained", type=Path, help="New atom-diffusion pretrain checkpoint; omit for from-scratch control")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--allow-pretrain-overlap", action="store_true",
                        help="Explicitly permit/report pretraining exposure to downstream validation/test molecules")
    args = parser.parse_args()
    config = initialize(args)
    source, splits, spec = build_property_data(args.data_root, args.dataset)
    if args.limit:
        splits = {k: v[:args.limit] for k, v in splits.items()}
    inputs = {"mapping_sha256": digest(args.data_root / spec.directory / "mapping/mol.csv.gz"),
              "splits": splits, "pretrained_sha256": digest(args.pretrained) if args.pretrained else None}
    pretrained = load_checkpoint(args.pretrained) if args.pretrained else None
    if pretrained is not None and (pretrained["stage"] != "pretrain" or pretrained["config"] != asdict(config)):
        raise ValueError("Expected matching new pretrain architecture/config; old checkpoints are incompatible")
    if pretrained is not None:
        corpus = pretrained["run_metadata"]["inputs"]
        seen = set(corpus["train_smiles"]) | set(corpus["valid_smiles"])
        # Check full official held-out sets even if this invocation limits samples.
        _, full_splits, _ = build_property_data(args.data_root, args.dataset)
        overlap = {name: [i for i in full_splits[name] if Chem.MolToSmiles(
            Chem.MolFromSmiles(source.smiles[i]), canonical=True) in seen] for name in ("valid", "test")}
        inputs["pretrain_downstream_overlap"] = overlap
        if any(overlap.values()) and not args.allow_pretrain_overlap:
            raise ValueError("Pretraining corpus overlaps downstream validation/test; remove exposure or explicitly --allow-pretrain-overlap")
    meta = metadata(args, config, inputs, "finetune")
    with run_directory(args, meta):
        dataset = AtomDataset(source.smiles, source.targets, source.target_mask)
        batches = {name: loader(dataset, indices, args, name == "train") for name, indices in splits.items()}
        y, mask = source.targets[splits["train"]], source.target_mask[splits["train"]]
        if not mask.any(0).all():
            raise ValueError("Every task requires observed training labels")
        scaler = TargetScaler.fit(y, mask) if spec.task_type == "regression" else None
        weight = classification_positive_weight(y, mask) if spec.task_type == "classification" else None
        model = AtomDiffusionModel(config, spec.num_tasks).to(args.device)
        if pretrained is not None:
            model.encoder.load_state_dict(pretrained["encoder"], strict=True)
        result = fit(model, batches["train"], batches["valid"], args, "finetune", spec, scaler, weight)
        prediction, target, observed, _ = predict(model, batches["valid"], args.device, scaler, spec)
        result["validation"] = compute_property_metrics(prediction, target, observed, spec)
        result.update(stage="finetune", smoke_only=bool(args.limit), test_evaluated=False,
                      pretrained_encoder=bool(args.pretrained))
        if args.evaluate_test:
            prediction, target, observed, ids = predict(model, batches["test"], args.device, scaler, spec)
            result["test"] = compute_property_metrics(prediction, target, observed, spec)
            result["test_evaluated"] = True
            save_predictions(args.output_dir / "test_predictions.csv", ids, prediction, target, observed,
                             spec.label_columns, source.smiles)
        atomic_write(args.output_dir / "completion.json", result)
        atomic_write(args.output_dir / "status.json", {"state": "complete", **result})


if __name__ == "__main__":
    main()
