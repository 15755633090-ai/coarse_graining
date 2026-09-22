from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch
from torch import Tensor
from torch.utils.data import Dataset
import multiscale_tokenizer.training as training_module

from diffusion_encoder.data import (
    MoleculeBatch,
    MoleculeGraph,
    PropertyBatch,
    PropertyRecord,
)

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import (
    MultiscaleModelConfig,
    MultiscaleMolecularModel,
    TokenModelOutput,
)
from multiscale_tokenizer.training import (
    _EpochAwareCollator,
    TaskSpec,
    TargetScaler,
    _is_better_validation,
    _loss,
    _metrics,
    _make_loader,
    _prepare_loader_iteration,
    _run_epoch,
    _set_loader_epoch,
    _validate_resume_protocol,
    build_optimizer,
    train_property_model,
)
from multiscale_tokenizer.partition import (
    TokenizationConfig,
    derive_partition_seed,
    partition_molecule,
)


class _PersistentLoaderDataset(Dataset):
    """Top-level fixture so Windows worker processes can import it."""

    def __init__(self):
        nodes = 14
        features = torch.zeros((nodes, 5), dtype=torch.long)
        features[:, 0] = 6
        features[:, 1] = 5
        bonds = torch.zeros((nodes, nodes), dtype=torch.long)
        indices = torch.arange(nodes - 1)
        bonds[indices, indices + 1] = 1
        bonds[indices + 1, indices] = 1
        self.record = PropertyRecord(
            MoleculeGraph(features, bonds, "C" * nodes),
            torch.tensor([0.0]),
            torch.tensor([True]),
            7,
            "train",
        )

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        return self.record


class _TinyPropertyDataset(Dataset):
    """Small property dataset that is picklable by persistent workers."""

    def __init__(self, _root: str | Path, split: str):
        sizes = {"train": 4, "valid": 2, "test": 2}
        offset = {"train": 0, "valid": 10, "test": 20}[split]
        self.records = []
        labels = []
        for index in range(sizes[split]):
            features = torch.zeros((3, 5), dtype=torch.long)
            features[:, 0] = torch.tensor([6, 7, 8])
            features[:, 1] = 5
            bonds = torch.tensor([
                [0, 1, 0], [1, 0, 1], [0, 1, 0],
            ])
            target = torch.tensor([float(index + offset) / 10.0])
            labels.append(target)
            self.records.append(PropertyRecord(
                MoleculeGraph(features, bonds, "CCO"),
                target,
                torch.tensor([True]),
                index + offset,
                split,
            ))
        self.labels = torch.stack(labels)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


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
                partitions,
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

    def test_epoch_aware_collator_updates_shared_partition_epoch(self) -> None:
        record = PropertyRecord(
            MoleculeGraph(
                torch.zeros((1, 5), dtype=torch.long),
                torch.zeros((1, 1), dtype=torch.long),
                "C",
            ),
            torch.tensor([0.0]),
            torch.tensor([True]),
            7,
            "train",
        )
        collator = _EpochAwareCollator(
            base_partition_seed=101,
            epoch=0,
            training=True,
            tokenization=TokenizationConfig(),
            use_partitions=True,
        )
        self.assertTrue(collator._epoch.is_shared())
        with patch(
            "multiscale_tokenizer.training.partition_molecule",
            return_value="partition",
        ) as mocked:
            collator.set_epoch(3)
            batch = collator([record])
        self.assertEqual(collator.epoch, 3)
        self.assertEqual(batch.partitions, ["partition"])
        self.assertEqual(
            mocked.call_args.kwargs["seed"],
            derive_partition_seed(101, 7, 3),
        )

    def test_persistent_worker_observes_new_partition_epoch(self) -> None:
        dataset = _PersistentLoaderDataset()
        generator = torch.Generator().manual_seed(43)
        loader = _make_loader(
            dataset,
            batch_size=1,
            shuffle=False,
            num_workers=1,
            base_partition_seed=101,
            epoch=0,
            training=True,
            generator=generator,
            use_partitions=True,
        )
        _set_loader_epoch(loader, 0)
        _prepare_loader_iteration(loader)
        first = next(iter(loader)).partitions[0]
        worker_pid = loader._iterator._workers[0].pid
        _set_loader_epoch(loader, 3)
        _prepare_loader_iteration(loader)
        second = next(iter(loader)).partitions[0]
        self.assertEqual(loader._iterator._workers[0].pid, worker_pid)
        bonds = dataset.record.graph.bonds
        expected_first = partition_molecule(
            bonds,
            seed=derive_partition_seed(101, 7, 0),
            include_stats=False,
        )
        expected_second = partition_molecule(
            bonds,
            seed=derive_partition_seed(101, 7, 3),
            include_stats=False,
        )
        self.assertTrue(first.owner.equal(expected_first.owner))
        self.assertTrue(second.owner.equal(expected_second.owner))
        persistent_state = generator.get_state()

        reference_generator = torch.Generator().manual_seed(43)
        for epoch in (0, 3):
            reference = _make_loader(
                dataset,
                batch_size=1,
                shuffle=False,
                num_workers=1,
                base_partition_seed=101,
                epoch=epoch,
                training=True,
                generator=reference_generator,
                use_partitions=True,
            )
            _prepare_loader_iteration(reference)
            next(iter(reference))
        self.assertTrue(persistent_state.equal(reference_generator.get_state()))

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

    def test_finetune_optimizer_has_named_differential_lr_groups(self) -> None:
        model = MultiscaleMolecularModel(
            DiffusionEncoder(ModelConfig(hidden_dim=8, dropout=0.0)),
            config=MultiscaleModelConfig(
                hidden_dim=8,
                experiment_mode="multiscale_finetune",
            ),
        )
        optimizer = build_optimizer(
            model,
            encoder_learning_rate=1e-5,
            downstream_learning_rate=1e-3,
            weight_decay=0.01,
        )
        self.assertEqual(
            [(group["name"], group["lr"]) for group in optimizer.param_groups],
            [("encoder", 1e-5), ("downstream", 1e-3)],
        )
        covered = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        }
        self.assertEqual(
            covered,
            {id(parameter) for parameter in model.parameters() if parameter.requires_grad},
        )

    def test_schema3_resume_rejects_protocol_mismatch(self) -> None:
        protocol = {"batch_size": 32, "dropout": 0.1}
        checkpoint = {
            "schema_version": 3,
            "task_spec": {"name": "lipo"},
            "experiment_mode": "multiscale_finetune",
            "model_seed": 0,
            "partition_seed": 100000,
            "data_seed": 0,
            "training_protocol": protocol,
            "provenance": {"encoder": "abc"},
        }
        with self.assertRaisesRegex(ValueError, "training_protocol"):
            _validate_resume_protocol(
                checkpoint,
                task_spec={"name": "lipo"},
                experiment_mode="multiscale_finetune",
                seeds={
                    "model_seed": 0,
                    "partition_seed": 100000,
                    "data_seed": 0,
                },
                training_protocol={"batch_size": 16, "dropout": 0.1},
                provenance={"encoder": "abc"},
            )

    def _assert_resume_trajectory(self, num_workers: int) -> None:
        with TemporaryDirectory() as temporary, patch(
            "multiscale_tokenizer.training.OGBMoleculePropertyDataset",
            _TinyPropertyDataset,
        ):
            root = Path(temporary)
            common = dict(
                dataset_root=root,
                task="lipo",
                device="cpu",
                batch_size=2,
                num_workers=num_workers,
                experiment_mode="baseline_finetune",
                model_seed=17,
                data_seed=23,
                partition_seed=29,
                scheduler_patience=0,
                patience=0,
            )
            uninterrupted = train_property_model(
                **common, output_dir=root / "full", epochs=2,
            )
            original_run_epoch = training_module._run_epoch
            calls = 0

            def interrupt_before_second_epoch(*args, **kwargs):
                nonlocal calls
                calls += 1
                if calls == 3:
                    raise RuntimeError("simulated interruption")
                return original_run_epoch(*args, **kwargs)

            with patch(
                "multiscale_tokenizer.training._run_epoch",
                side_effect=interrupt_before_second_epoch,
            ), self.assertRaisesRegex(RuntimeError, "simulated interruption"):
                train_property_model(
                    **common, output_dir=root / "resumed", epochs=2,
                )
            resumed = train_property_model(
                **common, output_dir=root / "resumed", epochs=2, resume=True,
            )
            required = {
                "optimizer_state", "scheduler_state", "current_epoch",
                "best_selection", "best_model_state",
                "epochs_without_improvement", "encoder_lr", "downstream_lr",
                "model_seed", "data_seed", "partition_seed", "experiment_mode",
                "training_protocol", "provenance",
            }
            self.assertTrue(required <= resumed.keys())
            self.assertEqual(resumed["current_epoch"], 1)
            self.assertEqual(
                len(resumed["provenance"]["encoder_checkpoint"]["sha256"]),
                64,
            )
            self.assertIn("commit_sha", resumed["provenance"]["git"])
            self.assertEqual(
                len(resumed["provenance"]["dataset"]["combined_sha256"]),
                64,
            )
            for name, expected in uninterrupted["model_state"].items():
                torch.testing.assert_close(
                    resumed["model_state"][name], expected, rtol=0, atol=0,
                )
            self.assertEqual(resumed["history"], uninterrupted["history"])
            completed = train_property_model(
                **common, output_dir=root / "resumed", epochs=2, resume=True,
            )
            self.assertEqual(completed["history"], resumed["history"])
            self.assertEqual(completed["test_metrics"], resumed["test_metrics"])

    def test_resume_restores_optimizer_scheduler_and_rng_trajectory(self) -> None:
        self._assert_resume_trajectory(num_workers=0)

    def test_persistent_workers_resume_matches_uninterrupted_training(self) -> None:
        self._assert_resume_trajectory(num_workers=1)


if __name__ == "__main__":
    unittest.main()
