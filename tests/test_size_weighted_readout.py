"""Synthetic checks for the isolated C/D size-weighted experiment."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from torch import nn

from coarse_gnn import TopologyCache
from coarse_gnn.diffusion_adapter import precompute_batch_topologies
from diffusion.bond_diffusion.data import collate_graphs
from examples.run_coarse_demo import chain_graph
from frozen_features import FrozenGraphBatch
from coarse_gnn.packed import packed_predict
from scripts.readout_ablation.models import (
    ALL_VARIANTS,
    REFERENCE_BY_VARIANT,
    TRAIN_VARIANTS,
    make_factory,
)
from scripts.readout_ablation.run_size_weighted import EVALUATION_POLICY, summarize, train


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_dim=16)
        self.weight = nn.Parameter(torch.ones(16))

    def encode_nodes(self, *args):
        raise AssertionError("Readout ablations must use cached features")


def fake_original_factory(*args):
    return SimpleNamespace(encoder=FakeEncoder())


class SizeWeightedReadoutTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        cache = TopologyCache()
        self.factory = make_factory(fake_original_factory, cache)
        graph = collate_graphs([chain_graph(n) for n in (24, 31, 1)])
        topologies = precompute_batch_topologies(graph, cache)
        generator = torch.Generator().manual_seed(123)
        nodes = torch.randn(*graph.node_features.shape[:2], 16, generator=generator)
        self.batch = FrozenGraphBatch(graph, topologies, nodes)

    def model(self, mode, dropout=0):
        torch.manual_seed(7)
        return self.factory(mode, 1, Path("unused.pt"), "cpu", dropout)

    def test_each_pair_differs_only_in_graph_pool(self):
        for weighted, reference in REFERENCE_BY_VARIANT.items():
            weighted_model = self.model(weighted)
            reference_model = self.model(reference)
            self.assertEqual(weighted_model.predictor.config.graph_pool, "size_weighted_mean")
            self.assertEqual(reference_model.predictor.config.graph_pool, "mean")
            weighted_cfg = vars(weighted_model.predictor.config)
            reference_cfg = vars(reference_model.predictor.config)
            differences = {key for key in weighted_cfg if weighted_cfg[key] != reference_cfg[key]}
            self.assertEqual(differences, {"graph_pool"})
            for key, value in weighted_model.predictor.state_dict().items():
                torch.testing.assert_close(value, reference_model.predictor.state_dict()[key], rtol=0, atol=0)

    def test_architecture_and_parameter_counts_match_references(self):
        for weighted, reference in REFERENCE_BY_VARIANT.items():
            weighted_model = self.model(weighted)
            reference_model = self.model(reference)
            self.assertEqual(
                sum(p.numel() for p in weighted_model.parameters() if p.requires_grad),
                sum(p.numel() for p in reference_model.parameters() if p.requires_grad),
            )
            self.assertEqual(
                len(weighted_model.predictor.coarse_gnn.layers),
                len(reference_model.predictor.coarse_gnn.layers),
            )

    def test_weighted_graph_pool_uses_owned_core_sizes(self):
        for mode in TRAIN_VARIANTS:
            model = self.model(mode).eval()
            _, _, coarse, graph = packed_predict(
                model.predictor,
                self.batch.node_states,
                self.batch,
                return_intermediates=True,
            )
            sizes = self.batch.plan.core_lengths.to(coarse.dtype)
            expected = torch.segment_reduce(
                coarse * sizes.unsqueeze(1),
                "sum",
                lengths=self.batch.plan.graph_lengths,
            )
            expected = expected / torch.segment_reduce(
                sizes,
                "sum",
                lengths=self.batch.plan.graph_lengths,
            ).unsqueeze(1)
            torch.testing.assert_close(graph, expected, rtol=0, atol=0)

    def test_all_variants_use_frozen_cache_and_backpropagate(self):
        for mode in ALL_VARIANTS:
            model = self.model(mode, dropout=0.1).train()
            output = model(self.batch)
            self.assertTrue(torch.isfinite(output).all())
            output.square().mean().backward()
            self.assertFalse(model.encoder.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))

    def test_summary_pairs_equal_and_weighted_by_seed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mother, output = root / "mother", root / "weighted"
            offsets = {
                "region_only": 0.10,
                "region_size_weighted": 0.06,
                "base_coarse": 0.03,
                "base_size_weighted": 0.01,
            }
            for mode, offset in offsets.items():
                folder = output if mode in TRAIN_VARIANTS else mother
                for seed in range(5):
                    path = folder / "lipo" / mode / f"seed_{seed}" / "result.json"
                    path.parent.mkdir(parents=True)
                    path.write_text(
                        json.dumps({
                            "mode": mode,
                            "seed": seed,
                            "validation_metrics": {"rmse": 0.8 + 0.01 * seed + offset},
                            "test_metrics": {"rmse": 99.0},
                        }),
                        encoding="utf-8",
                    )
            report = summarize(output, mother)
            self.assertTrue(report["complete"])
            self.assertTrue(all("test_rmse" not in row for row in report["runs"]))
            paired = {row["question"]: row for row in report["paired"]}
            self.assertAlmostEqual(paired["C_equal_vs_size_weighted"]["mean_delta_rmse"], -0.04)
            self.assertAlmostEqual(paired["D_equal_vs_size_weighted"]["mean_delta_rmse"], -0.02)
            self.assertAlmostEqual(paired["C_size_vs_D_size"]["mean_delta_rmse"], -0.05)
            self.assertAlmostEqual(
                report["coarse_readout_interaction"]["mean_interaction_delta_rmse"], 0.02
            )

    def test_training_never_enables_test_evaluation(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            args = SimpleNamespace(
                seeds=[0],
                output_dir=output,
                device="cpu",
                batch_size=4,
                encoder_lr=0,
                head_lr=0.001,
                dropout=0.1,
            )
            legacy = SimpleNamespace(
                create_property_model=None,
                _save_downstream_resume=None,
                single_run_config=Mock(return_value={}),
                initialize_single_run_config=Mock(),
                run_single=Mock(),
            )
            train(
                legacy,
                args,
                [],
                {"train": [0], "valid": [1], "test": [2]},
                SimpleNamespace(),
                None,
                {},
                {"evaluation_policy": dict(EVALUATION_POLICY)},
                ["region_size_weighted"],
                output / "mother",
            )
            self.assertIs(legacy.run_single.call_args.kwargs["evaluate_test"], False)


if __name__ == "__main__":
    unittest.main()
