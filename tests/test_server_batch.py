"""Smoke tests for portable six-job collection and provenance checks."""
from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_lipo_formal as formal
from scripts.server_batch.collect import collect
from scripts.server_batch.run_job import _verify_code
from scripts.readout_ablation.run_size_weighted import EVALUATION_POLICY


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class ServerBatchCollectionTests(unittest.TestCase):
    def test_windows_launcher_exports_only_after_collection(self):
        launcher = (
            Path(__file__).resolve().parents[1]
            / "scripts/server_batch/run_experiment.cmd"
        ).read_text(encoding="utf-8")
        collect_position = launcher.index("scripts.server_batch.collect")
        export_position = launcher.index('robocopy "%OUTPUT_ROOT%"')
        self.assertLess(collect_position, export_position)
        self.assertIn(
            'set "NETWORK_RESULT_ROOT=N:\\coarse_graining_transfer\\results\\size_weighted"',
            launcher,
        )
        self.assertIn("if errorlevel 8", launcher)
        self.assertIn("Results remain safe at: %OUTPUT_ROOT%", launcher)

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
        execution = {
            "device": "cuda:0",
            "amp": True,
            "deterministic": True,
            "micro_batch_size": 8,
            "num_workers": 0,
            "cpu_threads": 8,
            "cublas_workspace_config": ":4096:8",
        }
        manifest = {
            "schema_version": 3,
            "server_execution": execution,
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
                    "stage": "portable_frozen_size_weighted_batch",
                    "bundle_manifest_sha256": bundle_digest,
                    "protocol_scope": manifest["protocol_scope"],
                    "evaluation_policy": dict(EVALUATION_POLICY),
                    "variants": [variant],
                    "seed": seed,
                    "execution": dict(execution),
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

    def test_rejects_wrong_job_provenance_and_execution(self):
        mutations = (
            ("bundle sha", lambda cfg: cfg.update(bundle_manifest_sha256="wrong"), "different portable bundle"),
            ("seed", lambda cfg: cfg.update(seed=99), "identity does not match"),
            ("stage", lambda cfg: cfg.update(stage="wrong"), "unexpected stage"),
            (
                "execution",
                lambda cfg: cfg["execution"].update(micro_batch_size=16),
                "execution differs",
            ),
        )
        for label, mutate, message in mutations:
            with self.subTest(label=label), tempfile.TemporaryDirectory() as temporary:
                bundle, output = self.make_batch(Path(temporary))
                config_path = output / "region_size_weighted_seed_0/run_config.json"
                config = formal.read_json(config_path)
                mutate(config)
                formal.write_json(config_path, config)
                with self.assertRaisesRegex(ValueError, message):
                    collect(bundle, output)

    def test_rejects_nonfinite_score_and_test_predictions(self):
        with tempfile.TemporaryDirectory() as temporary:
            bundle, output = self.make_batch(Path(temporary))
            result_path = output / "region_size_weighted_seed_0/lipo/region_size_weighted/seed_0/result.json"
            result = formal.read_json(result_path)
            result["validation_metrics"]["rmse"] = float("nan")
            formal.write_json(result_path, result)
            with self.assertRaisesRegex(ValueError, "Invalid batch result"):
                collect(bundle, output)

        with tempfile.TemporaryDirectory() as temporary:
            bundle, output = self.make_batch(Path(temporary))
            prediction_path = output / "region_size_weighted_seed_0/lipo/region_size_weighted/seed_0/test_predictions.csv"
            prediction_path.write_text("forbidden", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "used test data"):
                collect(bundle, output)

    def test_code_verifier_uses_packaged_source_hashes_without_git(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "module.py"
            source.write_text("VALUE = 1\n", encoding="utf-8")
            manifest = {
                "code": {
                    "mode": "self_contained_hash_locked",
                    "root": "code",
                    "files": {"module.py": {"sha256": sha256(source)}},
                }
            }
            with patch("scripts.server_batch.run_job.formal.ROOT", root):
                _verify_code(manifest)
                source.write_text("VALUE = 2\n", encoding="utf-8")
                with self.assertRaisesRegex(ValueError, "source differs"):
                    _verify_code(manifest)

            manifest["code"]["mode"] = "git_checkout"
            with patch("scripts.server_batch.run_job.formal.ROOT", root):
                with self.assertRaisesRegex(ValueError, "self-contained source lock"):
                    _verify_code(manifest)


if __name__ == "__main__":
    unittest.main()
