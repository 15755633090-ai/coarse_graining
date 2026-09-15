"""Fixed-feature mechanism experiment: cache identity, gradients and legacy loop."""
import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from coarse_gnn import TopologyCache
from coarse_gnn.prepared_data import PreparedPropertyDataset
import run_lipo_formal as formal
import run_lipo_frozen as frozen
from frozen_features import frozen_factory, frozen_tools, load_features
from test_formal_lipo import TinyDataset


class FrozenLipoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not (formal.PROJECT / "model/results_formal/04_diffusion/run_config.json").exists():
            raise unittest.SkipTest("Original project is required")
        torch.set_num_threads(2)
        cls.legacy, cls.args, _, _, _ = frozen.load_protocol(formal.PROJECT)

    def fixture(self, directory):
        args = copy.copy(self.args)
        args.device, args.amp, args.micro_batch_size = "cpu", False, 2
        source = TinyDataset()
        source.forbidden = set()
        cache = TopologyCache()
        prepared = PreparedPropertyDataset(source, self.legacy.collate_property_batch, cache, formal.STRUCTURE)
        factory = frozen_factory(self.legacy.create_property_model, cache)
        model = factory("region_only", 1, args.checkpoint, "cpu", 0.1)
        data, report = load_features(self.legacy, args, prepared, model.encoder, directory, "fixture-split")
        return args, source, prepared, factory, model, data, report

    def test_cache_hit_and_split_invalidation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, prepared, _, model, data, first = self.fixture(temporary)
            self.assertFalse(first["hit"])
            with patch.object(model.encoder, "encode_nodes", side_effect=AssertionError("Cache hit recomputed encoder")):
                reused, second = load_features(self.legacy, args, prepared, model.encoder, temporary, "fixture-split")
            self.assertTrue(second["hit"])
            for a, b in zip(data.features, reused.features):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
            with patch.object(model.encoder, "encode_nodes", wraps=model.encoder.encode_nodes) as encode:
                _, changed = load_features(self.legacy, args, prepared, model.encoder, temporary, "different-split")
                self.assertGreater(encode.call_count, 0)
            self.assertNotEqual(first["path"], changed["path"])

    def test_only_predictor_updates_and_slicing_preserves_alignment(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, _, _, factory, _, data, _ = self.fixture(temporary)
            tools = frozen_tools(self.legacy)
            indices = [7, 0, 5, 2]
            batch = tools["collate_property_batch"]([data[i] for i in indices])
            sliced = tools["_slice_property_batch"](batch, 1, 3).to("cpu")
            for i, original in enumerate(indices[1:3]):
                torch.testing.assert_close(sliced.graph.node_states[i, :len(data.features[original])], data.features[original])
            model = factory("base_coarse", 1, args.checkpoint, "cpu", 0.1).train()
            before = {k: v.clone() for k, v in model.encoder.state_dict().items()}
            head_before = {k: v.clone() for k, v in model.head.state_dict().items()}
            with patch.object(model.encoder, "encode_nodes", side_effect=AssertionError("Training reran fixed encoder")):
                optimizer = torch.optim.AdamW(model.head.parameters(), lr=args.head_lr)
                model(sliced.graph).square().mean().backward()
                optimizer.step()
            self.assertFalse(model.encoder.training)
            self.assertTrue(all(p.grad is None and not p.requires_grad for p in model.encoder.parameters()))
            for key, value in before.items():
                torch.testing.assert_close(value, model.encoder.state_dict()[key], rtol=0, atol=0)
            self.assertTrue(any(not torch.equal(value, model.head.state_dict()[key]) for key, value in head_before.items()))

    def test_original_loop_uses_cached_features_and_selects_validation(self):
        with tempfile.TemporaryDirectory() as temporary:
            args, source, _, factory, _, data, _ = self.fixture(Path(temporary) / "cache")
            source.forbidden = {8, 9}
            args.epochs, args.patience, args.batch_size, args.checkpoint_every_batches = 2, 2, 4, 1
            args.tuning_protocol = dict(sha256="frozen-fixture", payload={"fixture": True})
            splits = dict(train=list(range(6)), valid=[6, 7], test=[8, 9])
            tools = frozen_tools(self.legacy)
            def guarded(*values):
                model = factory(*values)
                model.encoder.encode_nodes = lambda *a, **k: self.fail("Cached training invoked encoder")
                return model
            directory = Path(temporary) / "run"
            with formal.overrides(self.legacy, create_property_model=guarded, **tools):
                result = self.legacy.run_single(args, data, splits, self.legacy.TASK_SPECS["lipo"],
                                               "region_only", 0, torch.device("cpu"), directory, evaluate_test=False)
            history = formal.read_json(directory / "history.json")
            self.assertEqual(result["best_epoch"], min(history, key=lambda r: r["valid_metrics"]["rmse"])["epoch"])
            self.assertIsNone(result["test_metrics"])


if __name__ == "__main__":
    unittest.main()
