"""Analytic recovery metrics and checkpoint authority invariants."""
from copy import deepcopy
import unittest

import torch
from torch_geometric.data import Batch

from .data import ATOM_DIM, ATOM_DIMS, graph_from_smiles, one_hot
from .diffusion import GaussianSchedule
from .metrics import recovery_statistics, summarize_recovery
from .model import AtomDiffusionModel, Config
from .training import authoritative_best, cpu_snapshot


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(18)
        self.schedule = GaussianSchedule(1000)
        self.x0 = one_hot(torch.tensor([[6, 5, 0], [8, 6, 1], [7, 5, 0],
                                       [6, 5, 1], [8, 4, 0], [7, 5, 1]]), ATOM_DIMS).double()
        self.steps = torch.tensor([10, 100, 101, 500, 501, 1000])
        self.boundary = torch.tensor([True, False, True, False, True, False])
        self.noise = torch.randn_like(self.x0)
        self.xt = self.schedule.corrupt(self.x0, self.steps, self.noise)

    def statistics(self, prediction):
        return recovery_statistics(self.x0, self.xt, self.noise, prediction, self.steps, self.boundary, self.schedule)

    def test_perfect_noise_recovers_each_block_and_time_group(self):
        result = summarize_recovery(self.statistics(self.noise))
        for time_name, groups in result.items():
            for group_name, blocks in groups.items():
                for block_name, values in blocks.items():
                    self.assertGreater(values["atoms"], 0)
                    self.assertEqual(values["noise_mse"], 0)
                    self.assertLess(values["x0_mse"], 1e-24)
                    self.assertEqual(values["joint_accuracy" if block_name == "all" else "accuracy"], 1)
        self.assertEqual(result["all"]["all"]["all"]["atoms"], 6)
        self.assertEqual(result["t_0_10pct"]["boundary"]["all"]["atoms"], 1)
        self.assertEqual(result["t_10_50pct"]["all"]["all"]["atoms"], 2)

    def test_attribute_dimensions_do_not_hide_block_errors(self):
        prediction = self.noise.clone()
        prediction[:, :119] += 1
        result = summarize_recovery(self.statistics(prediction))["all"]["all"]
        self.assertAlmostEqual(result["all"]["noise_mse"], 119 / ATOM_DIM)
        self.assertAlmostEqual(result["element"]["noise_mse"], 1)
        self.assertEqual(result["charge"]["noise_mse"], 0)
        self.assertEqual(result["aromatic"]["noise_mse"], 0)

    def test_reconstructed_block_accuracy_and_joint_accuracy(self):
        desired = self.x0.clone()
        for row, start, width in ((0, 0, 119), (1, 119, 12), (2, 131, 2)):
            old = int(desired[row, start:start+width].argmax())
            desired[row, start:start+width] = 0
            desired[row, start + (old+1) % width] = 1
        a = self.schedule.coefficient(self.xt, self.steps)
        prediction = (self.xt - a.sqrt() * desired) / (1-a).sqrt()
        result = summarize_recovery(self.statistics(prediction))["all"]["all"]
        for name in ("element", "charge", "aromatic"):
            self.assertAlmostEqual(result[name]["accuracy"], 5 / 6)
        self.assertEqual(result["all"]["joint_accuracy"], 0.5)

    def test_statistics_merge_by_counts_and_empty_groups_are_null(self):
        prediction = self.noise + 0.2
        whole = self.statistics(prediction)
        merged = {}
        for sl in (slice(0, 3), slice(3, 6)):
            part = recovery_statistics(self.x0[sl], self.xt[sl], self.noise[sl], prediction[sl],
                                       self.steps[sl], self.boundary[sl], self.schedule)
            for name, value in part.items():
                merged[name] = merged.get(name, 0) + value
        for name in whole:
            self.assertAlmostEqual(whole[name], merged[name], places=8)
        empty = summarize_recovery({})["t_0_10pct"]["boundary"]["charge"]
        self.assertEqual(empty["atoms"], 0)
        self.assertIsNone(empty["accuracy"])
        self.assertIsNone(empty["noise_mse"])

    def test_terminal_schedule_signal_is_recorded_and_not_budget_invariant(self):
        first, second = GaussianSchedule(1000).description(), GaussianSchedule(100).description()
        self.assertLess(first["alpha_bar_T"], 0.0001)
        self.assertGreater(second["alpha_bar_T"], 0.3)
        self.assertTrue(second["steps_change_terminal_snr"])

    def test_no_motif_is_whole_mechanism_ablation_and_inactive_paths_frozen(self):
        model = AtomDiffusionModel(Config(hidden_dim=8, layers=2, dropout=0, motif_prior=False)).eval()
        batch = Batch.from_data_list([graph_from_smiles("CCOC(=O)c1ccccc1")])
        self.assertTrue(batch.is_cross.any())
        before, states = model(batch, return_states=True)
        self.assertEqual(states["cross_edge_index"].shape[1], 0)
        model.set_stage("pretrain")
        for name, parameter in model.encoder.named_parameters():
            if name.startswith("boundary_input.") or ".cross_message." in name or ".gate." in name:
                self.assertFalse(parameter.requires_grad, name)
        torch.testing.assert_close(before, model(batch), rtol=0, atol=0)

    def test_best_snapshot_is_immutable_and_consistency_is_checked(self):
        best = {"schema": "test", "feature_schema": "features", "stage": "pretrain", "config": {},
                "run_config_sha256": "run", "epoch": 1, "best_epoch": 1, "best_score": 0.5,
                "model": {"weight": torch.tensor([1.])}}
        retained = cpu_snapshot(best)
        best["model"]["weight"].add_(10)
        self.assertEqual(float(retained["model"]["weight"]), 1)
        latest = {**retained, "epoch": 2, "checkpoint_protocol": 2, "history": [{"epoch": 1}, {"epoch": 2}],
                  "best_checkpoint": retained}
        self.assertIs(authoritative_best(latest), retained)
        corrupt = deepcopy(latest)
        corrupt["best_epoch"] = 2
        with self.assertRaises(ValueError):
            authoritative_best(corrupt)
        with self.assertRaises(ValueError):
            authoritative_best({"checkpoint_protocol": 1})


if __name__ == "__main__":
    unittest.main()
