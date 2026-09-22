from __future__ import annotations

import unittest

import torch
from torch import nn

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel


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


class ModelTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
