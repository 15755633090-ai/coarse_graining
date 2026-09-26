import unittest
from unittest.mock import patch

import numpy as np
import torch

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel
from multiscale_tokenizer.random_regions import graph_distances, sample_regions, RandomRegionBranch


class RandomRegionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(7)
        self.features = torch.zeros(2, 20, 5, dtype=torch.long)
        self.features[..., 0] = torch.randint(1, 9, (2, 20))
        self.bonds = torch.zeros(2, 20, 20, dtype=torch.long)
        for i in range(19):
            self.bonds[:, i, i + 1] = self.bonds[:, i + 1, i] = 1
        self.mask = torch.ones(2, 20, dtype=torch.bool)
        self.model = MultiscaleMolecularModel(
            DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0)),
            config=MultiscaleModelConfig(hidden_dim=16, dropout=0,
                                        experiment_mode="random_region_stage2_frozen"),
        )

    def test_distances_regions_and_disconnected_bucket(self):
        distances = graph_distances(self.bonds[0])
        np.testing.assert_array_equal(distances[0], np.arange(20))
        centers, regions = sample_regions(distances, 7)
        self.assertEqual(len(centers), 3)
        self.assertEqual(len(set(centers)), 3)
        for center, region in zip(centers, regions):
            np.testing.assert_array_equal(region, np.abs(np.arange(20) - center) <= 2)
        np.testing.assert_array_equal(RandomRegionBranch.bucket_distances(np.array([0, 8, 9, 12, 13, -1])), [0, 8, 9, 9, 10, 11])

    def test_gradients_and_frozen_encoder(self):
        self.model.train()
        output = self.model(self.features, self.bonds, self.mask, [10, 11])
        self.assertEqual(output.graph_representation.shape, (2, 64))
        output.prediction.square().mean().backward()
        for parameter in self.model.encoder.parameters():
            self.assertIsNone(parameter.grad)
        gradient = self.model.random_regions.distance_bias.weight.grad
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(gradient.abs().sum(), 0)

    def test_eval_views_share_encoder_and_are_batch_independent(self):
        self.model.eval()
        with torch.no_grad(), patch.object(self.model.encoder, "forward", wraps=self.model.encoder.forward) as encoder:
            together = self.model(self.features, self.bonds, self.mask, [10, 11]).prediction
            self.assertEqual(encoder.call_count, 1)
            separate = torch.cat([self.model(self.features[i:i+1], self.bonds[i:i+1], self.mask[i:i+1], [10+i]).prediction for i in range(2)])
            repeated = self.model(self.features, self.bonds, self.mask, [10, 11]).prediction
        torch.testing.assert_close(together, separate)
        torch.testing.assert_close(together, repeated, atol=0, rtol=0)

    def test_single_atom_and_padding(self):
        self.model.eval()
        self.mask[0, 1:] = False
        with torch.no_grad():
            output = self.model(self.features, self.bonds, self.mask, [10, 11])
            single = self.model(self.features[:1, :1], self.bonds[:1, :1, :1], self.mask[:1, :1], [10])
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertEqual(output.token_counts[0, 3], 1)
        torch.testing.assert_close(output.prediction[:1], single.prediction)


if __name__ == "__main__":
    unittest.main()
