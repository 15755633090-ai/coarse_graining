"""Provenance migration must never permit a changed experimental protocol."""
import copy
import tempfile
import unittest
from pathlib import Path

from coarse_gnn.execution_revision import ExecutionRevision, read, write


class ExecutionRevisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.old_files = {"run.py": {"sha256": "old"}}
        self.new_files = {"run.py": {"sha256": "new"}}
        write(self.root / "outputs/performance/packed_audit.json",
              {"passed": True, "source_hashes": {"run.py": "new"}})
        write(self.root / "outputs/performance/reference/manifest.json", {"run.py": "old"})
        self.output = self.root / "results"
        self.revision = ExecutionRevision(self.root, self.output, self.new_files)
        self.previous = dict(lr=0.001, radius=4, seeds=[0, 1],
                             new_source_files=self.old_files)
        self.current = dict(self.previous, new_source_files=self.new_files)

    def test_preserves_manifest_and_records_actual_execution(self):
        path = self.output / "run_config.json"
        write(path, self.previous)
        before = path.read_bytes()
        self.assertEqual(self.revision.manifest(self.current), self.previous)
        self.assertEqual(path.read_bytes(), before)
        journal = read(self.output / "execution_revision.json")
        self.assertEqual(journal["source_files"], self.new_files)
        self.assertEqual(journal["original_source_files"], self.old_files)

    def test_rejects_experimental_changes_and_unknown_source(self):
        for key, value in (("lr", 0.002), ("radius", 5), ("seeds", [2, 3])):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.revision.compatible(self.previous, dict(self.current, **{key: value}))
        unknown = copy.deepcopy(self.previous)
        unknown["new_source_files"]["run.py"]["sha256"] = "unaudited"
        with self.assertRaises(ValueError):
            self.revision.compatible(unknown, self.current)

    def test_rejects_stale_or_failed_audit(self):
        for audit in ({"passed": False, "source_hashes": {"run.py": "new"}},
                      {"passed": True, "source_hashes": {"run.py": "other"}}):
            write(self.root / "outputs/performance/packed_audit.json", audit)
            with self.assertRaises(ValueError):
                ExecutionRevision(self.root, self.output, self.new_files)

    def test_existing_trial_keeps_original_protocol_identity(self):
        write(self.output / "lipo/region_only/tuning/seed_42/trial_0/run_config.json",
              dict(tuning_protocol=self.previous, tuning_protocol_sha256="original"))
        result = self.revision.protocol(dict(payload=self.current, sha256="new"), "region_only")
        self.assertEqual(result, dict(payload=self.previous, sha256="original"))


if __name__ == "__main__":
    unittest.main()
