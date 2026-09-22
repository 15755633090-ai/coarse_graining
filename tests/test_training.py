from __future__ import annotations

import unittest

import torch

from multiscale_tokenizer.training import (
    TaskSpec,
    TargetScaler,
    _loss,
    _metrics,
)


class TrainingTests(unittest.TestCase):
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
