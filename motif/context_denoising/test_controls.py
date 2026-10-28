"""Regression checks for diagnostics, provenance and incremental result collection."""
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

import torch
from torch_geometric.data import Batch

from .data import graph_from_smiles
from .diagnostics import denoise_statistics, gradient_diagnostics, summarize_steps
from .model import Config, MotifContextModel
from .reporting import (collect_provenance, output_session, prepare_cohort, refresh_summaries,
                        save_partition_audit, write_json)


class DiagnosticsTests(unittest.TestCase):
    def test_diagnostics_preserve_rng_gradients_and_optimizer_update(self):
        torch.manual_seed(22)
        model = MotifContextModel(1, Config(hidden_dim=8, atom_layers=1, dropout=0))
        model.configure_training("random", 0.01)
        reference = deepcopy(model)
        batch = Batch.from_data_list([graph_from_smiles("CCOC(=O)c1ccccc1"), graph_from_smiles("C")])
        states = []
        for current, diagnostic in ((model, True), (reference, False)):
            parameters = tuple(p for p in current.parameters() if p.requires_grad)
            optimizer = torch.optim.AdamW(parameters, lr=1e-3)
            output, auxiliary, _ = current(batch, "random", True, torch.Generator().manual_seed(123))
            prop = output.square().mean()
            if diagnostic:
                rng = torch.get_rng_state().clone()
                report = gradient_diagnostics(prop, auxiliary, parameters, 0.01)
                self.assertTrue(torch.equal(rng, torch.get_rng_state()))
                self.assertTrue(all(p.grad is None for p in parameters))
                self.assertGreater(report["shared_gradient_parameters"], 0)
                self.assertGreater(report["auxiliary_gradient_norm_raw"], 0)
            (prop + 0.01 * auxiliary).backward()
            states.append([p.grad.detach().clone() if p.grad is not None else None for p in parameters])
            torch.nn.utils.clip_grad_norm_(parameters, 0.1)
            optimizer.step()
        for left, right in zip(*states):
            if left is None or right is None:
                self.assertIs(left, right)
            else:
                torch.testing.assert_close(left, right, atol=0, rtol=0)
        for left, right in zip(model.parameters(), reference.parameters()):
            torch.testing.assert_close(left, right, atol=0, rtol=0)

    def test_group_denominators_and_empty_groups(self):
        first = denoise_statistics(torch.tensor([1., 3., 9.]), torch.tensor([0, 1, 1]), torch.tensor([1, 2]))
        second = denoise_statistics(torch.tensor([5.]), torch.tensor([0]), torch.tensor([1]))
        logs = [{k: float(v) for k, v in row.items()} for row in (first, second)]
        result = summarize_steps(logs)
        self.assertEqual(result["denoise_single_mean"], 3)
        self.assertEqual(result["denoise_multi_mean"], 6)
        self.assertEqual(result["denoise_selected_mean"], 4.5)
        self.assertEqual(result["denoise_molecule_mean"], 4)
        self.assertEqual(result["denoise_molecule_count"], 3)
        empty = summarize_steps([{"loss": 0.0}])
        self.assertIsNone(empty["denoise_single_mean"])
        self.assertEqual(empty["gradient_diagnostic_samples"], 0)

    def test_only_variant_paths_are_optimized(self):
        model = MotifContextModel(1, Config(hidden_dim=8, atom_layers=1))
        atom = model.configure_training("atom", 0.01)
        self.assertFalse(model.property_head[0].weight.requires_grad)
        self.assertTrue(model.atom_head[0].weight.requires_grad)
        context = model.configure_training("context", 0.01)
        self.assertFalse(model.denoise_head[0].weight.requires_grad)
        random = model.configure_training("random", 0.01)
        self.assertTrue(model.denoise_head[0].weight.requires_grad)
        self.assertFalse(model.atom_head[0].weight.requires_grad)
        self.assertLess(atom["optimized"], context["optimized"])
        self.assertLess(context["optimized"], random["optimized"])
        self.assertEqual(random["instantiated"], atom["instantiated"])

    def test_dependency_provenance_has_actual_sources_and_versions(self):
        root = Path(__file__).resolve().parents[2]
        provenance = collect_provenance(root)
        self.assertIn("model/benchmark.py", provenance["source_sha256"])
        self.assertIn("扩散GNN/01_扩散预训练/bond_diffusion/property_prediction.py", provenance["source_sha256"])
        for package in ("torch-geometric", "ogb", "rdkit"):
            self.assertTrue(provenance["software"]["packages"][package])
        self.assertIn("repository_commit_sha", provenance["git"])


class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.args = SimpleNamespace(output_dir=self.root, dataset="esol", variants=["context"], seeds=[0])
        self.protocol = {"schema_version": 2, "data_sha256": "data", "settings": {"epochs": 2}}
        self.sha = prepare_cohort(self.args, self.protocol)

    def add_run(self, variant, seed, completed=True):
        folder = self.root / "esol" / variant / f"seed_{seed}"
        write_json(folder / "config.json", {"variant": variant, "seed": seed, "protocol_sha256": self.sha})
        if completed:
            write_json(folder / "metrics.json", {"variant": variant, "seed": seed, "best_epoch": 1,
                                                "validation": {"rmse": 1.0}})
        return folder

    def test_incremental_summary_retains_previous_runs_and_tracks_incomplete(self):
        first = self.add_run("context", 0)
        original = (first / "metrics.json").read_bytes()
        refresh_summaries(self.root, "esol", self.sha)
        self.args.variants = ["random"]
        self.assertEqual(prepare_cohort(self.args, self.protocol), self.sha)
        self.add_run("random", 0)
        self.add_run("mask", 0, completed=False)
        rows = refresh_summaries(self.root, "esol", self.sha)
        self.assertEqual({r["variant"] for r in rows}, {"context", "random"})
        self.assertEqual(original, (first / "metrics.json").read_bytes())
        status = json.loads((self.root / "esol_summary_status.json").read_text())
        self.assertEqual(status["incomplete_runs"], ["esol/mask/seed_0"])

    def test_existing_run_or_protocol_change_rejected_before_audit_changes(self):
        self.add_run("context", 0)
        save_partition_audit(self.root, "esol", {"molecules": 4})
        audit = (self.root / "esol_partition_audit.json").read_bytes()
        with self.assertRaises(FileExistsError):
            prepare_cohort(self.args, self.protocol)
        self.args.variants = ["random"]
        with self.assertRaises(ValueError):
            prepare_cohort(self.args, {**self.protocol, "data_sha256": "changed"})
        with self.assertRaises(ValueError):
            save_partition_audit(self.root, "esol", {"molecules": 5})
        self.assertEqual(audit, (self.root / "esol_partition_audit.json").read_bytes())

    def test_incompatible_result_does_not_replace_summary(self):
        self.add_run("context", 0)
        refresh_summaries(self.root, "esol", self.sha)
        original = (self.root / "esol_summary.json").read_bytes()
        bad = self.add_run("random", 0)
        write_json(bad / "config.json", {"protocol_sha256": "wrong"})
        with self.assertRaises(ValueError):
            refresh_summaries(self.root, "esol", self.sha)
        self.assertEqual(original, (self.root / "esol_summary.json").read_bytes())

    def test_exclusive_session_lock(self):
        with output_session(self.root, "esol"):
            with self.assertRaises(RuntimeError):
                with output_session(self.root, "esol"):
                    self.fail("Lock acquired twice")
        self.assertFalse((self.root / ".esol.lock").exists())


if __name__ == "__main__":
    unittest.main()
