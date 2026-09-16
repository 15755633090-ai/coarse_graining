"""Collect the six independent portable jobs into one validation-only report."""
from __future__ import annotations

import argparse
import csv
import hashlib
import math
import statistics
from pathlib import Path

import run_lipo_formal as formal
from scripts.readout_ablation.models import REFERENCE_BY_VARIANT, TRAIN_VARIANTS
from scripts.readout_ablation.run_size_weighted import (
    COARSE_COMPARISON_SCOPE,
    EVALUATION_POLICY,
    READOUT_COMPARISON_SCOPE,
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_references(bundle: Path, manifest: dict) -> dict:
    relative = manifest["reference_validation"]
    path = bundle / relative
    expected = manifest.get("files", {}).get(relative)
    if expected is None or not path.is_file() or _sha256(path) != expected.get("sha256"):
        raise ValueError("Reference validation file differs from the bundle manifest")
    references = {}
    for row in formal.read_json(path):
        key = (row.get("mode"), row.get("seed"))
        score = row.get("validation_rmse")
        if key in references or key[0] not in REFERENCE_BY_VARIANT.values():
            raise ValueError(f"Invalid or duplicate reference validation row: {row}")
        if key[1] not in manifest["protocol_scope"]["seeds"] or not isinstance(score, (int, float)) or not math.isfinite(score):
            raise ValueError(f"Invalid reference validation score: {row}")
        references[key] = score
    expected_keys = {
        (mode, seed)
        for mode in REFERENCE_BY_VARIANT.values()
        for seed in manifest["protocol_scope"]["seeds"]
    }
    if references.keys() != expected_keys:
        raise ValueError("Reference validation file does not contain the exact required C/D rows")
    return references


def _load_job(manifest: dict, output_dir: Path, variant: str, seed: int) -> dict:
    job_root = output_dir / f"{variant}_seed_{seed}"
    config_path = job_root / "run_config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Missing batch job configuration: {config_path}")
    config = formal.read_json(config_path)
    if config.get("stage") != "portable_frozen_size_weighted_batch":
        raise ValueError(f"Job has an unexpected stage: {job_root}")
    if config.get("variants") != [variant] or config.get("seed") != seed:
        raise ValueError(f"Job identity does not match its collection slot: {job_root}")
    if config.get("bundle_manifest_sha256") != formal.digest(manifest):
        raise ValueError(f"Job belongs to a different portable bundle: {job_root}")
    if config.get("protocol_scope") != manifest["protocol_scope"]:
        raise ValueError(f"Job has a different protocol scope: {job_root}")
    if config.get("evaluation_policy") != EVALUATION_POLICY:
        raise ValueError(f"Job does not have the locked validation-only policy: {job_root}")
    if config.get("execution") != manifest["server_execution"]:
        raise ValueError(f"Job execution differs from the locked bundle execution: {job_root}")
    result_path = job_root / "lipo" / variant / f"seed_{seed}" / "result.json"
    if not result_path.is_file():
        raise FileNotFoundError(f"Missing batch job result: {result_path}")
    result = formal.read_json(result_path)
    if result.get("test_metrics") is not None or (result_path.parent / "test_predictions.csv").exists():
        raise ValueError(f"Batch job used test data: {result_path}")
    score = (result.get("validation_metrics") or {}).get("rmse")
    if (
        result.get("mode") != variant
        or result.get("seed") != seed
        or not isinstance(score, (int, float))
        or not math.isfinite(score)
    ):
        raise ValueError(f"Invalid batch result: {result_path}")
    return {"validation_rmse": score, "result_path": result_path}


def collect(bundle: Path, output_dir: Path) -> dict:
    manifest = formal.read_json(bundle / "manifest.json")
    if manifest.get("schema_version") != 2 or not isinstance(manifest.get("server_execution"), dict):
        raise ValueError("Unsupported portable bundle manifest")
    references = _load_references(bundle, manifest)
    rows = []
    for variant in TRAIN_VARIANTS:
        for seed in manifest["protocol_scope"]["seeds"]:
            job = _load_job(manifest, output_dir, variant, seed)
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
    weighted_coarse_rows = [
        {"seed": seed, "delta_rmse": values[("base_size_weighted", seed)] - values[("region_size_weighted", seed)]}
        for seed in manifest["protocol_scope"]["seeds"]
    ]
    interaction = []
    for seed in manifest["protocol_scope"]["seeds"]:
        equal_coarse = references[("base_coarse", seed)] - references[("region_only", seed)]
        weighted_coarse_delta = values[("base_size_weighted", seed)] - values[("region_size_weighted", seed)]
        interaction.append({
            "seed": seed,
            "equal_readout_coarse_delta_rmse": equal_coarse,
            "size_weighted_coarse_delta_rmse": weighted_coarse_delta,
            "interaction_delta_rmse": weighted_coarse_delta - equal_coarse,
        })
    paired = []
    for variant, question in (
        ("region_size_weighted", "C_equal_vs_size_weighted"),
        ("base_size_weighted", "D_equal_vs_size_weighted"),
    ):
        subset = [row for row in rows if row["mode"] == variant]
        deltas = [{"seed": row["seed"], "delta_rmse": row["delta_rmse"]} for row in subset]
        paired.append({
            "question": question,
            "before": REFERENCE_BY_VARIANT[variant],
            "after": variant,
            "n": len(deltas),
            "scope": READOUT_COMPARISON_SCOPE,
            "per_seed": deltas,
            "mean_delta_rmse": statistics.mean(row["delta_rmse"] for row in deltas),
            "std_delta_rmse": statistics.stdev(row["delta_rmse"] for row in deltas) if len(deltas) > 1 else None,
        })
    paired.append({
        "question": "C_size_vs_D_size",
        "before": "region_size_weighted",
        "after": "base_size_weighted",
        "n": len(weighted_coarse_rows),
        "scope": COARSE_COMPARISON_SCOPE,
        "per_seed": weighted_coarse_rows,
        "mean_delta_rmse": statistics.mean(row["delta_rmse"] for row in weighted_coarse_rows),
        "std_delta_rmse": statistics.stdev(row["delta_rmse"] for row in weighted_coarse_rows) if len(weighted_coarse_rows) > 1 else None,
    })
    report = {"protocol_scope": manifest["protocol_scope"], "execution": manifest["server_execution"],
              "metric_split": "validation", "test_metrics_used": False,
              "runs": rows, "summary": summary,
              "paired": paired,
              "coarse_readout_interaction": {
                  "scope": "Difference in the Coarse-GNN effect after switching to size-weighted readout.",
                  "inference_limit": (
                      "The equal-readout C/D references and size-weighted runs may come from different "
                      "hardware/software environments. Treat this three-seed interaction as exploratory; "
                      "a formal mechanism claim requires all four cells in one controlled environment."
                  ),
                  "per_seed": interaction,
                  "mean_interaction_delta_rmse": statistics.mean(row["interaction_delta_rmse"] for row in interaction),
                  "std_interaction_delta_rmse": statistics.stdev(row["interaction_delta_rmse"] for row in interaction) if len(interaction) > 1 else None,
              },
              "interpretation": "Negative delta_rmse favors the model named in each comparison's after field."}
    formal.write_json(output_dir / "comparison_validation.json", report)
    with (output_dir / "per_seed_validation.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Collected six validation-only jobs: {output_dir}", flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    options = parser.parse_args()
    collect(options.bundle, options.output_dir)


if __name__ == "__main__":
    main()
