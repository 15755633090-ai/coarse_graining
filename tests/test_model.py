from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel
from multiscale_tokenizer.partition import partition_molecule


def line_batch(batch_size: int, nodes: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    features = torch.zeros((batch_size, nodes, 5), dtype=torch.long)
    features[..., 0] = torch.randint(1, 20, (batch_size, nodes))
    features[..., 1] = 5
    features[..., 2] = torch.randint(0, 2, (batch_size, nodes))
    features[..., 3] = torch.randint(0, 5, (batch_size, nodes))
    features[..., 4] = torch.randint(0, 5, (batch_size, nodes))
    bonds = torch.zeros((batch_size, nodes, nodes), dtype=torch.long)
    for graph in range(batch_size):
        for node in range(nodes - 1):
            bonds[graph, node, node + 1] = 1
            bonds[graph, node + 1, node] = 1
    return features, bonds, torch.ones((batch_size, nodes), dtype=torch.bool)


def path_molecule(
    num_nodes: int,
    padded_nodes: int,
    *,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    features = torch.zeros((1, padded_nodes, 5), dtype=torch.long)
    features[..., 0] = torch.randint(1, 20, (1, padded_nodes), generator=generator)
    features[..., 1] = 5
    features[..., 2] = torch.randint(0, 2, (1, padded_nodes), generator=generator)
    features[..., 3] = torch.randint(0, 5, (1, padded_nodes), generator=generator)
    features[..., 4] = torch.randint(0, 5, (1, padded_nodes), generator=generator)
    bonds = torch.zeros((1, padded_nodes, padded_nodes), dtype=torch.long)
    for node in range(num_nodes - 1):
        bonds[0, node, node + 1] = 1
        bonds[0, node + 1, node] = 1
    mask = torch.zeros((1, padded_nodes), dtype=torch.bool)
    mask[0, :num_nodes] = True
    return features, bonds, mask


class ModelTests(unittest.TestCase):
    @staticmethod
    def _model(mode: str, *, seed: int = 0) -> MultiscaleMolecularModel:
        torch.manual_seed(seed)
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0.0))
        return MultiscaleMolecularModel(
            encoder,
            config=MultiscaleModelConfig(
                hidden_dim=16, dropout=0.0, experiment_mode=mode,
            ),
        )

    def test_forward_and_head_gradients_with_frozen_encoder(self) -> None:
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0.0))
        model = MultiscaleMolecularModel(
            encoder,
            config=MultiscaleModelConfig(hidden_dim=16, dropout=0.0),
        )
        features, bonds, mask = line_batch(2, 14)
        output = model(features, bonds, mask, partition_seeds=[1, 2])
        self.assertEqual(tuple(output.prediction.shape), (2, 1))
        self.assertEqual(tuple(output.graph_representation.shape), (2, 16))
        self.assertEqual(tuple(output.level_norms.shape), (2, 4))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.encoder.parameters()))
        loss = output.prediction.square().mean()
        loss.backward()
        self.assertIsNotNone(model.prediction_head[0].weight.grad)
        self.assertIsNotNone(model.region_encoders[0][0].weight.grad)
        self.assertIsNone(model.encoder.layers[0].message[0].weight.grad)

    def test_model_train_keeps_encoder_in_eval(self) -> None:
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=8, dropout=0.5))
        model = MultiscaleMolecularModel(
            encoder,
            config=MultiscaleModelConfig(hidden_dim=8, dropout=0.2),
        )
        model.train()
        self.assertTrue(model.training)
        self.assertFalse(model.encoder.training)

    def test_finetune_mode_trains_encoder_and_propagates_gradients(self) -> None:
        model = self._model("multiscale_finetune")
        model.train()
        self.assertTrue(model.encoder.training)
        features, bonds, mask = line_batch(2, 14)
        model(features, bonds, mask, partition_seeds=[1, 2]).prediction.sum().backward()
        gradient = model.encoder.layers[0].message[0].weight.grad
        self.assertIsNotNone(gradient)
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_baseline_uses_historical_final_layer_sum_mean(self) -> None:
        model = self._model("baseline_frozen").eval()
        features, bonds, mask = path_molecule(8, 12, seed=10)
        with torch.no_grad():
            final = model.encoder(features, bonds, mask)[-1][:, :8]
            expected = torch.cat((final.sum(dim=1), final.mean(dim=1)), dim=-1)
            actual = model(features, bonds, mask).graph_representation
        torch.testing.assert_close(actual, expected, rtol=0, atol=1e-6)
        self.assertEqual(model.prediction_head[0].in_features, 32)
        self.assertIsInstance(model.prediction_head[1], nn.SiLU)

    def test_seeded_modes_have_identical_encoder_initialization(self) -> None:
        baseline = self._model("baseline_finetune", seed=31)
        multiscale = self._model("multiscale_finetune", seed=31)
        for left, right in zip(
            baseline.encoder.parameters(), multiscale.encoder.parameters(),
        ):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        self.assertEqual(len(baseline.region_encoders), 0)

    def test_batch_padding_does_not_change_molecule_prediction(self) -> None:
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0.0))
        model = MultiscaleMolecularModel(
            encoder,
            config=MultiscaleModelConfig(hidden_dim=16, dropout=0.0),
        ).eval()
        single_features, single_bonds, single_mask = path_molecule(
            12, 12, seed=21,
        )
        other_features, other_bonds, other_mask = path_molecule(
            40, 40, seed=22,
        )
        features = torch.cat((
            torch.nn.functional.pad(single_features, (0, 0, 0, 28)),
            other_features,
        ), dim=0)
        bonds = torch.cat((
            torch.nn.functional.pad(single_bonds, (0, 28, 0, 28)),
            other_bonds,
        ), dim=0)
        mask = torch.cat((
            torch.nn.functional.pad(single_mask, (0, 28), value=False),
            other_mask,
        ), dim=0)

        with torch.no_grad():
            single = model(
                single_features, single_bonds, single_mask, partition_seeds=[7],
            )
            batched = model(features, bonds, mask, partition_seeds=[7, 8])

        torch.testing.assert_close(
            batched.graph_representation[0],
            single.graph_representation[0],
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            batched.prediction[0],
            single.prediction[0],
            rtol=0,
            atol=1e-6,
        )
        self.assertTrue(
            batched.token_counts[0].equal(single.token_counts[0]),
        )

    def test_precomputed_partitions_match_forward_partitioning(self) -> None:
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0.0))
        model = MultiscaleMolecularModel(
            encoder,
            config=MultiscaleModelConfig(hidden_dim=16, dropout=0.0),
        ).eval()
        features, bonds, mask = line_batch(2, 14)
        seeds = [5, 6]
        partitions = [
            partition_molecule(
                bonds[index],
                node_mask=mask[index],
                seed=seed,
                include_stats=False,
            )
            for index, seed in enumerate(seeds)
        ]
        with torch.no_grad():
            expected = model(features, bonds, mask, seeds)
            actual = model(
                features,
                bonds,
                mask,
                partition_seeds=None,
                partitions=partitions,
            )
        torch.testing.assert_close(
            actual.graph_representation,
            expected.graph_representation,
            rtol=0,
            atol=1e-6,
        )
        torch.testing.assert_close(
            actual.prediction,
            expected.prediction,
            rtol=0,
            atol=1e-6,
        )


if __name__ == "__main__":
    unittest.main()
