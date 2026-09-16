"""Collect the six independent portable jobs into one validation-only report."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import run_lipo_formal as formal
from scripts.readout_ablation.models import REFERENCE_BY_VARIANT, TRAIN_VARIANTS


def _load_job(manifest: dict, output_dir: Path, variant: str, seed: int) -> dict:
    job_root = output_dir / f"{variant}_seed_{seed}"
    config_path = job_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing batch job configuration: {config_path}")
    config = formal.read_json(config_path)
    if config.get("bundle_manifest_sha256") != formal.digest(manifest):
        raise ValueError(f"Job belongs to a different portable bundle: {job_root}")
    if config.get("protocol_scope") != manifest["protocol_scope"]:
        raise ValueError(f"Job has a different protocol scope: {job_root}")
    if config.get("evaluation_policy", {}).get("evaluate_test_during_training") is not False:
        raise ValueError(f"Job does not have the locked validation-only policy: {job_root}")
    result_path = job_root / "lipo" / variant / f"seed_{seed}" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing batch job result: {result_path}")
    result = formal.read_json(result_path)
    if result.get("test_metrics") is not None or (result_path.parent / "test_predictions.csv").exists():
        raise ValueError(f"Batch job used test data: {result_path}")
    score = (result.get("validation_metrics") or {}).get("rmse")
    if result.get("mode") != variant or result.get("seed") != seed or score is None:
        raise ValueError(f"Invalid batch result: {result_path}")
    return {"validation_rmse": score, "result_path": result_path}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    manifest = formal.read_json(options.bundle / "manifest.json")
    references = {(row["mode"], row["seed"]): row["validation_rmse"] for row in formal.read_json(options.bundle / manifest["reference_validation"])}
    rows = []
    for variant in TRAIN_VARIANTS:
        for seed in manifest["protocol_scope"]["seeds"]:
            job = _load_job(manifest, options.output_dir, variant, seed)
            score = job["validation_rmse"]
            reference = REFERENCE_BY_VARIANT[variant]
            rows.append({"mode": variant, "seed": seed, "validation_rmse": score,
                         "reference_mode": reference, "reference_validation_rmse": references[(reference, seed)],
                         "delta_rmse": score - references[(reference, seed)]})
    summary = []
    for variant in TRAIN_VARIANTS:
        subset = [row for row in rows if row["mode"] == variant]
        deltas = [row["delta_rmse"] for row in subset]
        summary.append({"mode": variant, "n": len(subset), "mean_validation_rmse": statistics.mean(row["validation_rmse"] for row in subset),
                        "mean_delta_rmse": statistics.mean(deltas), "std_delta_rmse": statistics.stdev(deltas) if len(deltas) > 1 else None})
    values = {(row["mode"], row["seed"]): row["validation_rmse"] for row in rows}
    weighted_coarse = [
        {"seed": seed, "delta_rmse": values[("base_size_weighted", seed)] - values[("region_size_weighted", seed)]}
        for seed in manifest["protocol_scope"]["seeds"]
    ]
    interaction = []
    for seed in manifest["protocol_scope"]["seeds"]:
        equal_coarse = references[("base_coarse", seed)] - references[("region_only", seed)]
        weighted_coarse = values[("base_size_weighted", seed)] - values[("region_size_weighted", seed)]
        interaction.append({
            "seed": seed,
            "equal_readout_coarse_delta_rmse": equal_coarse,
            "size_weighted_coarse_delta_rmse": weighted_coarse,
            "interaction_delta_rmse": weighted_coarse - equal_coarse,
        })
    report = {"protocol_scope": manifest["protocol_scope"], "metric_split": "validation", "test_metrics_used": False,
              "runs": rows, "summary": summary,
              "paired": [{
                  "question": "C_size_vs_D_size",
                  "before": "region_size_weighted",
                  "after": "base_size_weighted",
                  "n": len(weighted_coarse),
                  "per_seed": weighted_coarse,
                  "mean_delta_rmse": statistics.mean(row["delta_rmse"] for row in weighted_coarse),
                  "std_delta_rmse": statistics.stdev(row["delta_rmse"] for row in weighted_coarse) if len(weighted_coarse) > 1 else None,
              }],
              "coarse_readout_interaction": {
                  "scope": "Difference in the Coarse-GNN effect after switching to size-weighted readout.",
                  "per_seed": interaction,
                  "mean_interaction_delta_rmse": statistics.mean(row["interaction_delta_rmse"] for row in interaction),
                  "std_interaction_delta_rmse": statistics.stdev(row["interaction_delta_rmse"] for row in interaction) if len(interaction) > 1 else None,
              },
              "interpretation": "Negative delta_rmse favors size-weighted readout versus its equal-readout reference."}
    formal.write_json(options.output_dir / "comparison_validation.json", report)
    with (options.output_dir / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected six validation-only jobs: {options.output_dir}", flush=True)


if __name__ == "__main__":
    main()
