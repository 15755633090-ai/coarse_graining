"""Reporting and provenance regression checks using temporary fixture results."""
import copy
import hashlib
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import run_lipo_formal as formal
from scripts.experiments import frozen_reporting as reporting


def identity(path):
    path = Path(path)
    return dict(path=str(path.resolve()), size=path.stat().st_size,
                sha256=hashlib.sha256(path.read_bytes()).hexdigest())


class FrozenReportingTests(unittest.TestCase):
    def configs(self):
        current = dict(eligible_seeds=list(range(5)), head_lr=.001,
                       new_source_files={"run_lipo_frozen.py": identity(formal.ROOT / "run_lipo_frozen.py"),
                                         "model.py": {"sha256": "fixed-model"}})
        old = copy.deepcopy(current)
        old["new_source_files"]["run_lipo_frozen.py"]["sha256"] = reporting.ORIGINAL_SOURCE_SHA256
        old["new_source_files"]["run_lipo_frozen.py"]["size"] = 0
        return old, current

    def test_reporting_revision_keeps_every_original_training_ast_node(self):
        source = (formal.ROOT / "run_lipo_frozen.py").read_text(encoding="utf-8")
        self.assertEqual(reporting.training_ast_digest(source), reporting.TRAINING_AST_SHA256)
        old, current = self.configs()
        reporting.verify_reporting_revision(old, current)

    def test_preserves_old_manifest_and_records_actual_sources_separately(self):
        old, current = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            path = output / "run_config.json"
            path.write_text(json.dumps(old), encoding="utf-8")
            before = path.read_bytes()
            result = reporting.preserve_training_manifest(SimpleNamespace(file_identity=identity), current, output)
            self.assertEqual(result, old)
            self.assertEqual(path.read_bytes(), before)
            journal = json.loads((output / "reporting_revision.json").read_text())
            self.assertEqual(journal["actual_source_files"], current["new_source_files"])
            self.assertTrue(journal["original_manifest_preserved"])

    def test_read_only_mother_verification_does_not_write_audit(self):
        old, current = self.configs()
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "run_config.json").write_text(json.dumps(old))
            result = reporting.preserve_training_manifest(None, current, output, write_audit=False)
            self.assertEqual(result, old)
            self.assertEqual([p.name for p in output.iterdir()], ["run_config.json"])

    def test_rejects_unknown_source_hyperparameter_and_other_model_changes(self):
        old, current = self.configs()
        variants = []
        changed = copy.deepcopy(old)
        changed["new_source_files"]["run_lipo_frozen.py"]["sha256"] = "unknown"
        variants.append((changed, current))
        changed = copy.deepcopy(current)
        changed["head_lr"] = .1
        variants.append((old, changed))
        changed = copy.deepcopy(current)
        changed["new_source_files"]["model.py"]["sha256"] = "different"
        variants.append((old, changed))
        for before, after in variants:
            with self.assertRaises(ValueError):
                reporting.verify_reporting_revision(before, after)

    def test_rejects_training_change_even_with_matching_current_file_hash(self):
        old, current = self.configs()
        source = (formal.ROOT / "run_lipo_frozen.py").read_text(encoding="utf-8")
        changed = source.replace("args.seeds = options.seeds", "args.seeds = [99]")
        self.assertNotEqual(changed, source)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "run_lipo_frozen.py"
            path.write_text(changed, encoding="utf-8")
            current["new_source_files"]["run_lipo_frozen.py"]["sha256"] = identity(path)["sha256"]
            with self.assertRaisesRegex(ValueError, "training semantics"):
                reporting.verify_reporting_revision(old, current, Path(temporary))

    def test_five_seed_summary_handles_partial_runs_and_separates_history(self):
        with tempfile.TemporaryDirectory() as temporary:
            output, baseline = Path(temporary) / "frozen", Path(temporary) / "old"
            output.mkdir()
            historical_csv = output / "comparison_seed_0_1.csv"
            historical_csv.write_text("historical two-seed report")
            for mode, offset in (("region_only", .1), ("base_coarse", 0.)):
                for seed in range(5):
                    if mode == "base_coarse" and seed == 4:
                        continue
                    path = output / "lipo" / mode / f"seed_{seed}" / "result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(dict(mode=mode, seed=seed, validation_metrics={"rmse": .7 + .01 * seed + offset},
                                                    test_metrics={"rmse": .8 + .02 * seed + offset})))
            path = baseline / "lipo/pretrained_frozen/seed_0/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(mode="pretrained_frozen", seed=0, test_metrics={"rmse": 99.})))
            report = reporting.summarize_frozen(SimpleNamespace(output_dir=output), baseline)
            self.assertFalse(report["complete"])
            self.assertEqual(report["missing_runs"], ["base_coarse/seed_4"])
            self.assertEqual(report["paired_summary"]["validation"]["n"], 4)
            self.assertEqual(len(report["historical_baseline"]), 1)
            self.assertEqual(historical_csv.read_text(), "historical two-seed report")
            self.assertTrue((output / "comparison_seed_0_4.csv").exists())
            path = output / "lipo/base_coarse/seed_4/result.json"
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps(dict(mode="base_coarse", seed=4, validation_metrics={"rmse": .74}, test_metrics={"rmse": .88})))
            report = reporting.summarize_frozen(SimpleNamespace(output_dir=output), baseline)
            self.assertTrue(report["complete"])
            self.assertEqual([row["seed"] for row in report["paired"]], list(range(5)))
            self.assertEqual(report["paired_summary"]["test"]["n"], 5)
            self.assertAlmostEqual(report["paired_summary"]["validation"]["mean"], -.1)
            self.assertTrue(all(row["n"] == 5 for row in report["summary"]))


if __name__ == "__main__":
    unittest.main()
