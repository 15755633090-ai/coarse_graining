"""Training-orchestration contracts without starting a formal experiment."""
import copy
from pathlib import Path
import tempfile
import unittest

import torch

import run_lipo_formal as formal
import run_lipo_frozen as frozen
from coarse_gnn import HierarchyCache
from scripts.experiments.run_hierarchy_lipo import GeneratorBucketBatchSampler
from scripts.experiments.run_hierarchy_lipo import (
    BaseCorrectionPredictor,
    HierarchyFactory,
    HierarchyFeatureDataset,
    hierarchy_tools,
    validate_features,
)
from tests.test_formal_lipo import TinyDataset


class HierarchyTrainingTests(unittest.TestCase):
    def test_bucket_sampler_is_complete_and_resume_reproducible(self):
        counts = [30, 2, 18, 7, 42, 10, 25, 4, 33, 12, 20]
        generator = torch.Generator().manual_seed(91)
        state = generator.get_state().clone()
        first = list(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        self.assertEqual(
            sorted(index for batch in first for index in batch), list(range(len(counts))),
        )
        generator.set_state(state)
        resumed = list(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        self.assertEqual(first, resumed)
        generator.set_state(state)
        iterator = iter(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        next(iterator)
        remaining = list(iterator)
        generator.set_state(state)
        replay = iter(GeneratorBucketBatchSampler(counts, 3, generator, True, multiplier=2))
        next(replay)
        self.assertEqual(remaining, list(replay))
        ordered = list(GeneratorBucketBatchSampler(
            counts, 3, torch.Generator().manual_seed(0), False, multiplier=20,
        ))
        flattened = [index for batch in ordered for index in batch]
        self.assertEqual(flattened, sorted(range(len(counts)), key=counts.__getitem__))

    def test_feature_cache_contract_and_base_correction_gradients(self):
        identity = dict(graph_fingerprints=["a", "b"], atom_counts=[2, 3], hidden_dim=4)
        saved = dict(
            h2=[torch.randn(2, 4), torch.randn(3, 4)],
            hbase=[torch.randn(8), torch.randn(8)],
        )
        validate_features(saved, identity)
        broken = dict(saved, h2=[saved["h2"][0], torch.full((3, 4), torch.nan)])
        with self.assertRaisesRegex(ValueError, "values"):
            validate_features(broken, identity)

        model = BaseCorrectionPredictor(8, 6, 1, 0)
        base = torch.randn(2, 8, requires_grad=True)
        prediction = torch.randn(2, 1)
        output = model(base, prediction)
        self.assertEqual(output.shape, (2, 1))
        output.square().mean().backward()
        self.assertGreater(base.grad.abs().sum().item(), 0)
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_hierarchy_orchestration_resumes_after_batch_checkpoint(self):
        if not (formal.PROJECT / "model/results_formal/04_diffusion/run_config.json").exists():
            self.skipTest("Original sibling formal project is required for integration test")
        torch.set_num_threads(2)
        legacy, args, _, baseline, _ = frozen.load_protocol(formal.PROJECT)
        source = TinyDataset()
        splits = dict(train=list(range(6)), valid=[6, 7], test=[8, 9])
        generator = torch.Generator().manual_seed(73)
        features = dict(
            h2=[torch.randn(index + 1, 128, generator=generator) for index in range(10)],
            hbase=[torch.randn(256, generator=generator) for _ in range(10)],
        )
        data = HierarchyFeatureDataset(source, features, HierarchyCache())
        cfg = copy.copy(args)
        cfg.epochs, cfg.patience, cfg.batch_size, cfg.micro_batch_size = 2, 2, 4, 2
        cfg.device, cfg.amp, cfg.dropout = "cpu", False, 0.1
        cfg.checkpoint_every_batches = 1
        cfg.tuning_protocol = dict(sha256="synthetic-hierarchy-resume", payload={"fixture": True})
        factory = HierarchyFactory(legacy, baseline, hidden_dim=16)
        factory.seed = 0
        tools = hierarchy_tools(legacy, data, "base_corr")
        original_save = legacy._save_downstream_resume

        def interrupt_after_save(path, state):
            original_save(path, state)
            if state["epoch"] == 1 and state["next_batch"] == 1:
                raise InterruptedError("Simulated interruption after durable hierarchy checkpoint")

        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            with formal.overrides(
                legacy,
                create_property_model=factory,
                _save_downstream_resume=interrupt_after_save,
                **tools,
            ):
                with self.assertRaises(InterruptedError):
                    legacy.run_single(
                        cfg, data, splits, legacy.TASK_SPECS["lipo"],
                        "base_corr", 0, torch.device("cpu"), directory,
                        evaluate_test=False, role="tuning",
                    )
            self.assertTrue((directory / "resume.pt").exists())
            with formal.overrides(legacy, create_property_model=factory, **tools):
                result = legacy.run_single(
                    cfg, data, splits, legacy.TASK_SPECS["lipo"],
                    "base_corr", 0, torch.device("cpu"), directory,
                    evaluate_test=False, role="tuning",
                )
            self.assertIsNone(result["test_metrics"])
            self.assertFalse((directory / "test_predictions.csv").exists())
            self.assertFalse((directory / "resume.pt").exists())
            self.assertEqual(len(formal.read_json(directory / "history.json")), 2)


if __name__ == "__main__":
    unittest.main()
