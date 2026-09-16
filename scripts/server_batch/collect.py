"""Collect the six independent portable jobs into one validation-only report."""
from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import run_lipo_formal as formal
from scripts.readout_ablation.models import REFERENCE_BY_VARIANT, TRAIN_VARIANTS


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
            result_path = options.output_dir / f"{variant}_seed_{seed}" / "lipo" / variant / f"seed_{seed}" / "result.json"
            if not result_path.is_file():
                raise FileNotFoundError(f"Missing batch job result: {result_path}")
            result = formal.read_json(result_path)
            if result.get("test_metrics") is not None or (result_path.parent / "test_predictions.csv").exists():
                raise ValueError(f"Batch job used test data: {result_path}")
            score = (result.get("validation_metrics") or {}).get("rmse")
            if result.get("mode") != variant or result.get("seed") != seed or score is None:
                raise ValueError(f"Invalid batch result: {result_path}")
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
    report = {"protocol_scope": manifest["protocol_scope"], "metric_split": "validation", "test_metrics_used": False,
              "runs": rows, "summary": summary,
              "interpretation": "Negative delta_rmse favors size-weighted readout versus its equal-readout reference."}
    formal.write_json(options.output_dir / "comparison_validation.json", report)
    with (options.output_dir / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected six validation-only jobs: {options.output_dir}", flush=True)


if __name__ == "__main__":
    main()
