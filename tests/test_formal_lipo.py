"""Integration checks against the original formal trainer, using synthetic data."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset, prepared_tools
from examples.run_coarse_demo import chain_graph
import run_lipo_formal as entry


class TinyDataset:
    def __init__(self):
        self.smiles = [f"diagnostic_{i}" for i in range(10)]
        self.targets = torch.arange(10).float().unsqueeze(1) / 10
        self.target_mask = torch.ones(10, 1, dtype=torch.bool)
        self.forbidden = {8, 9}
        self.calls = 0

    def __getitem__(self, index):
        self.calls += 1
        if index in self.forbidden:
            raise AssertionError("Tuning accessed held-out test sample")
        return chain_graph(index + 1), self.targets[index], self.target_mask[index]

    def __len__(self):
        return 10


class FormalLipoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (entry.PROJECT / "model/results_formal/04_diffusion/run_config.json").exists():
            raise unittest.SkipTest("Original sibling formal project is required for integration tests")
        torch.set_num_threads(2)
        cls.legacy, cls.args, cls.old_protocol, cls.baseline = entry.load_protocol(entry.PROJECT)

    def test_protocol_and_architecture_contract(self):
        self.assertEqual(self.args.epochs, 200)
        self.assertEqual(self.args.patience, 30)
        self.assertEqual(self.args.micro_batch_size, 8)
        self.assertEqual(len(self.args.search_candidates), 4)
        factory = entry.make_factory(self.legacy.create_property_model, TopologyCache())
        states = []
        for variant in entry.VARIANTS:
            self.legacy.seed_everything(0, deterministic=True)
            model = factory(variant, 1, self.args.checkpoint, torch.device("cpu"), 0.1)
            cfg = model.predictor.config
            self.assertFalse(model.freeze_encoder)
            self.assertEqual(cfg.coarse_layers, 0 if variant == "region_only" else 3)
            for name in ("use_region_edge_features", "use_coarse_edge_count", "use_coarse_edge_features", "use_size_feature"):
                self.assertEqual(getattr(cfg, name), variant == "enhanced_coarse")
            self.assertEqual(cfg.graph_pool, "size_weighted_mean" if variant == "enhanced_coarse" else "mean")
            head_ids = {id(p) for p in model.head.parameters()}
            encoder_ids = {id(p) for p in model.encoder.parameters() if p.requires_grad}
            self.assertFalse(head_ids & encoder_ids)
            self.assertEqual(head_ids | encoder_ids, {id(p) for p in model.parameters() if p.requires_grad})
            for name in ("node_heads", "edge_head", "graph_projection"):
                self.assertTrue(all(not p.requires_grad for p in getattr(model.encoder, name).parameters()))
            if variant != "enhanced_coarse":
                states.append({key: value.clone() for key, value in model.state_dict().items() if "coarse_gnn." not in key})
        self.assertEqual(states[0].keys(), states[1].keys())
        for key in states[0]:
            torch.testing.assert_close(states[0][key], states[1][key], rtol=0, atol=0)

    def test_original_training_loop_selects_validation_and_never_reads_test_during_tuning(self):
        dataset = TinyDataset()
        split = dict(train=list(range(6)), valid=[6, 7], test=[8, 9])
        cfg = copy.copy(self.args)
        # Diagnostic fixture only; no formal data, manifests or directories touched.
        cfg.epochs = 2
        cfg.patience = 2
        cfg.batch_size = 4
        cfg.micro_batch_size = 2
        cfg.checkpoint_every_batches = 1
        cfg.device = "cpu"
        cfg.amp = False
        cfg.dropout = 0
        cfg.tuning_protocol = dict(sha256="synthetic-integration-only", payload={"fixture": True})
        factory = entry.make_factory(self.legacy.create_property_model, TopologyCache())
        with tempfile.TemporaryDirectory() as temporary, entry.overrides(self.legacy, create_property_model=factory):
            directory = Path(temporary)
            result = self.legacy.run_single(cfg, dataset, split, self.legacy.TASK_SPECS["lipo"],
                                           "base_coarse", 42, torch.device("cpu"), directory,
                                           evaluate_test=False, role="tuning")
            self.assertIsNone(result["test_metrics"])
            self.assertFalse((directory / "test_predictions.csv").exists())
            history = entry.read_json(directory / "history.json")
            expected_epoch = min(history, key=lambda row: row["valid_metrics"]["rmse"])["epoch"]
            self.assertEqual(result["best_epoch"], expected_epoch)
            best = torch.load(directory / "best.pt", map_location="cpu", weights_only=False)
            scaler = self.legacy.TargetScaler.fit(dataset.targets[:6], dataset.target_mask[:6])
            torch.testing.assert_close(best["target_scaler"]["center"], scaler.center, rtol=0, atol=0)
            torch.testing.assert_close(best["target_scaler"]["scale"], scaler.scale, rtol=0, atol=0)
            self.assertFalse((directory / "resume.pt").exists())
            self.assertEqual(best["best_epoch"], expected_epoch)

    def test_original_serial_checkpoint_resumes_with_packed_data(self):
        source = TinyDataset()
        splits = dict(train=list(range(6)), valid=[6, 7], test=[8, 9])
        cfg = copy.copy(self.args)
        cfg.epochs, cfg.patience, cfg.batch_size, cfg.micro_batch_size = 2, 2, 4, 2
        cfg.device, cfg.amp, cfg.dropout = "cpu", False, 0.1
        cfg.checkpoint_every_batches = 1
        cfg.tuning_protocol = dict(sha256="synthetic-resume", payload={"fixture": True})
        cache = TopologyCache()
        factory = entry.make_factory(self.legacy.create_property_model, cache)
        original_save = self.legacy._save_downstream_resume
        def interrupt_after_save(path, state):
            original_save(path, state)
            if state["epoch"] == 1 and state["next_batch"] == 1:
                raise InterruptedError("Simulated interruption after durable checkpoint")
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with entry.overrides(self.legacy, create_property_model=factory, _save_downstream_resume=interrupt_after_save):
                with self.assertRaises(InterruptedError):
                    self.legacy.run_single(cfg, source, splits, self.legacy.TASK_SPECS["lipo"],
                                           "base_coarse", 42, torch.device("cpu"), directory, evaluate_test=False, role="tuning")
            self.assertTrue((directory / "resume.pt").exists())
            prepared = PreparedPropertyDataset(source, self.legacy.collate_property_batch, cache, entry.STRUCTURE)
            tools = prepared_tools(self.legacy, cache, entry.STRUCTURE)
            with entry.overrides(self.legacy, create_property_model=factory, **tools):
                result = self.legacy.run_single(cfg, prepared, splits, self.legacy.TASK_SPECS["lipo"],
                                               "base_coarse", 42, torch.device("cpu"), directory,
                                               evaluate_test=False, role="tuning")
            self.assertIsNone(result["test_metrics"])
            self.assertEqual(len(prepared.rows), 8)
            calls = source.calls
            for _ in range(3):
                for index in range(8):
                    prepared[index]
            self.assertEqual(source.calls, calls)
            self.assertFalse((directory / "resume.pt").exists())
            self.assertEqual(len(entry.read_json(directory / "history.json")), 2)


if __name__ == "__main__":
    unittest.main()
