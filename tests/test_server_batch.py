"""Smoke tests for portable six-job collection and provenance checks."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import run_lipo_formal as formal
from scripts.server_batch.collect import collect


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ServerBatchCollectionTests(unittest.TestCase):
    def make_batch(self, root: Path):
        bundle = root / "bundle"
        output = root / "jobs"
        reference_path = bundle / "assets/reference_validation.json"
        references = []
        for seed in range(3):
            references.extend((
                {"mode": "region_only", "seed": seed, "validation_rmse": 0.90 + 0.01 * seed},
                {"mode": "base_coarse", "seed": seed, "validation_rmse": 0.83 + 0.01 * seed},
            ))
        formal.write_json(reference_path, references)
        manifest = {
            "schema_version": 1,
            "protocol_scope": {
                "seeds": [0, 1, 2],
                "status": "exploratory_three_seed_validation_only",
                "not_a_replacement_for": "five_seed_formal_protocol",
            },
            "reference_validation": "assets/reference_validation.json",
            "files": {
                "assets/reference_validation.json": {"sha256": sha256(reference_path)},
            },
        }
        formal.write_json(bundle / "manifest.json", manifest)
        bundle_digest = formal.digest(manifest)
        scores = {
            "region_size_weighted": 0.86,
            "base_size_weighted": 0.81,
        }
        for variant, base_score in scores.items():
            for seed in range(3):
                job_root = output / f"{variant}_seed_{seed}"
                formal.write_json(job_root / "run_config.json", {
                    "bundle_manifest_sha256": bundle_digest,
                    "protocol_scope": manifest["protocol_scope"],
                    "evaluation_policy": {"evaluate_test_during_training": False},
                })
                formal.write_json(
                    job_root / "lipo" / variant / f"seed_{seed}" / "result.json",
                    {
                        "mode": variant,
                        "seed": seed,
                        "validation_metrics": {"rmse": base_score + 0.01 * seed},
                        "test_metrics": None,
                    },
                )
        return bundle, output

    def test_collects_all_pairs_and_interaction(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, output = self.make_batch(Path(temporary))
            report = collect(bundle, output)
            paired = {row["question"]: row for row in report["paired"]}
            self.assertAlmostEqual(paired["C_equal_vs_size_weighted"]["mean_delta_rmse"], -0.04)
            self.assertAlmostEqual(paired["D_equal_vs_size_weighted"]["mean_delta_rmse"], -0.02)
            self.assertAlmostEqual(paired["C_size_vs_D_size"]["mean_delta_rmse"], -0.05)
            self.assertIn("Coarse GNN", paired["C_size_vs_D_size"]["scope"])
            self.assertAlmostEqual(
                report["coarse_readout_interaction"]["mean_interaction_delta_rmse"], 0.02
            )

    def test_rejects_tampered_reference_file(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, output = self.make_batch(Path(temporary))
            reference_path = bundle / "assets/reference_validation.json"
            reference_path.write_text("[]", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "differs from the bundle manifest"):
                collect(bundle, output)


if __name__ == "__main__":
    unittest.main()
