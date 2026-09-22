from __future__ import annotations

import unittest

import torch
from torch import Tensor

from diffusion_encoder.data import MoleculeBatch, MoleculeGraph, PropertyBatch

from multiscale_tokenizer.model import TokenModelOutput
from multiscale_tokenizer.training import (
    TaskSpec,
    TargetScaler,
    _is_better_validation,
    _loss,
    _metrics,
    _run_epoch,
)


class TrainingTests(unittest.TestCase):
    @staticmethod
    def _batch(targets: list[list[float]], valid: list[list[bool]]) -> PropertyBatch:
        graphs = [
            MoleculeGraph(
                torch.zeros((1, 5), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                "C",
            )
            for _ in targets
        ]
        return PropertyBatch(
            MoleculeBatch(
                torch.stack([graph.node_features for graph in graphs]),
                torch.stack([graph.bonds for graph in graphs]),
                torch.ones((len(graphs), 1), dtype=torch.bool),
                [graph.smiles for graph in graphs],
            ),
            torch.tensor(targets, dtype=torch.float32),
            torch.tensor(valid, dtype=torch.bool),
            torch.arange(len(targets), dtype=torch.long),
        )

    def test_validation_loss_is_weighted_by_valid_labels(self) -> None:
        class ZeroModel:
            def train(self, mode: bool):
                return self

            def __call__(
                self,
                node_features: Tensor,
                bonds: Tensor,
                node_mask: Tensor,
                partition_seeds,
            ) -> TokenModelOutput:
                batch_size = node_features.size(0)
                prediction = torch.zeros((batch_size, 1))
                representation = torch.zeros((batch_size, 8))
                return TokenModelOutput(
                    prediction=prediction,
                    graph_representation=representation,
                    level_norms=torch.zeros((batch_size, 4)),
                    token_counts=torch.zeros((batch_size, 4), dtype=torch.long),
                    residual_counts=torch.zeros((batch_size, 4), dtype=torch.long),
                )

        scaler = TargetScaler(
            mean=torch.tensor([0.0]),
            std=torch.tensor([1.0]),
        )
        batches = [
            self._batch([[1.0]], [[True]]),
            self._batch([[0.0], [2 ** 0.5]], [[False], [True]]),
        ]
        metrics = _run_epoch(
            ZeroModel(),
            batches,
            device=torch.device("cpu"),
            spec=TaskSpec("lipo", "regression", 1),
            scaler=scaler,
            base_partition_seed=0,
            epoch=0,
            training=False,
        )
        self.assertAlmostEqual(metrics["loss"], 1.5, places=6)

    def test_classification_checkpoint_uses_primary_metric(self) -> None:
        spec = TaskSpec("pcba", "classification", 2)
        better, value = _is_better_validation(
            {"average_precision": 0.7, "roc_auc": 0.8},
            best_value=0.6,
            spec=spec,
        )
        self.assertTrue(better)
        self.assertEqual(value, 0.7)
        better, value = _is_better_validation(
            {"average_precision": float("nan"), "roc_auc": 0.8},
            best_value=0.6,
            spec=spec,
        )
        self.assertFalse(better)
        self.assertEqual(value, 0.6)

    def test_masked_loss_ignores_nan_targets(self) -> None:
        prediction = torch.tensor([[0.2, -0.4]], requires_grad=True)
        targets = torch.tensor([[float("nan"), 0.0]])
        mask = torch.tensor([[False, True]])
        loss = _loss(
            prediction,
            targets,
            mask,
            TaskSpec("clintox", "classification", 2),
            scaler=None,
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertEqual(float(prediction.grad[0, 0]), 0.0)

    def test_regression_metrics_are_reported_in_original_scale(self) -> None:
        prediction = torch.tensor([[-4.5], [-3.5]])
        targets = torch.tensor([[1.0], [2.0]])
        mask = torch.ones_like(targets, dtype=torch.bool)
        scaler = TargetScaler(
            mean=torch.tensor([10.0]),
            std=torch.tensor([2.0]),
        )
        metrics = _metrics(
            prediction,
            targets,
            mask,
            TaskSpec("lipo", "regression", 1),
            scaler,
        )
        self.assertAlmostEqual(metrics["rmse"], 2 ** -0.5)
        self.assertAlmostEqual(metrics["mae"], 0.5)
        self.assertAlmostEqual(metrics["r2"], -1.0)

    def test_classification_metrics_ignore_missing_labels(self) -> None:
        prediction = torch.tensor([[0.0, 2.0], [3.0, float("nan")]])
        targets = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        mask = torch.tensor([[True, True], [True, False]])
        metrics = _metrics(
            prediction,
            targets,
            mask,
            TaskSpec("pcba", "classification", 2),
            scaler=None,
        )
        self.assertAlmostEqual(metrics["roc_auc"], 1.0)
        self.assertAlmostEqual(metrics["average_precision"], 1.0)


if __name__ == "__main__":
    unittest.main()
