import unittest
from dataclasses import replace
from tempfile import TemporaryDirectory
from pathlib import Path
from dataclasses import asdict
from unittest.mock import patch

import numpy as np
import torch

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel
from multiscale_tokenizer.random_regions import (
    graph_distances, sample_regions, RandomRegionBranch, prepare_region_geometry,
)


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

    def test_five_views_prepare_geometry_once(self):
        self.model.eval()
        with patch("multiscale_tokenizer.model.prepare_region_geometry", wraps=prepare_region_geometry) as prepare, \
             patch("multiscale_tokenizer.random_regions.prepare_region_geometry", side_effect=AssertionError("unexpected per-view copy")), \
             patch("multiscale_tokenizer.random_regions.graph_distances", wraps=graph_distances) as distances:
            self.model(self.features, self.bonds, self.mask, [10, 11])
        self.assertEqual(prepare.call_count, 1)
        self.assertEqual(distances.call_count, 2)

    def test_h2_changes_only_base_with_identical_initialization(self):
        models = []
        for mode in ("random_region_stage2_frozen", "random_region_h2_stage2_frozen"):
            torch.manual_seed(123)
            model = MultiscaleMolecularModel(
                DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0)),
                config=replace(self.model.config, experiment_mode=mode),
            ).eval()
            models.append(model)
        for key, value in models[0].state_dict().items():
            torch.testing.assert_close(value, models[1].state_dict()[key], atol=0, rtol=0)
        self.mask[0, 12:] = False
        with torch.no_grad():
            states = models[0].encoder(self.features, self.bonds, self.mask)
            outputs = [model(self.features, self.bonds, self.mask, [10, 11]) for model in models]
        for output, layer in zip(outputs, (4, 2)):
            summed = (states[layer - 1] * self.mask.unsqueeze(-1)).sum(1)
            expected = torch.cat((summed, summed / self.mask.sum(1, keepdim=True)), -1)
            torch.testing.assert_close(output.graph_representation[:, :32], expected)
        torch.testing.assert_close(outputs[0].graph_representation[:, 32:],
                                   outputs[1].graph_representation[:, 32:], atol=0, rtol=0)
        self.assertFalse(torch.equal(outputs[0].graph_representation[:, :32],
                                     outputs[1].graph_representation[:, :32]))
        models[1].train()
        models[1](self.features, self.bonds, self.mask, [10, 11]).prediction.square().mean().backward()
        self.assertTrue(all(p.grad is None for p in models[1].encoder.parameters()))
        self.assertGreater(models[1].random_regions.distance_bias.weight.grad.abs().sum(), 0)

    def test_h2_checkpoint_load_and_v1_resume_rejection(self):
        from multiscale_tokenizer.training import load_property_model, _validate_resume_protocol, formal_protocol
        config = replace(self.model.config, experiment_mode="random_region_h2_stage2_frozen")
        model = MultiscaleMolecularModel(DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0)), config=config).eval()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "best.pt"
            torch.save(dict(schema_version=4, model_config=asdict(config),
                            model_state=model.state_dict(), task_spec=dict(name="lipo", task_type="regression", num_tasks=1)), path)
            with patch("multiscale_tokenizer.training.load_encoder", return_value=DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0))):
                loaded, _, _ = load_property_model(path)
        self.assertEqual(loaded.config.base_readout_layer, 2)
        with torch.no_grad():
            torch.testing.assert_close(model(self.features, self.bonds, self.mask, [10, 11]).prediction,
                                       loaded(self.features, self.bonds, self.mask, [10, 11]).prediction, atol=0, rtol=0)
        self.assertEqual(formal_protocol(config.experiment_mode), formal_protocol("random_region_stage2_frozen"))
        with self.assertRaisesRegex(ValueError, "experiment_mode"):
            _validate_resume_protocol(dict(schema_version=4, task_spec={}, experiment_mode="random_region_stage2_frozen",
                                           training_protocol={}, provenance={}), task_spec={},
                                      experiment_mode=config.experiment_mode, seeds={}, training_protocol={}, provenance={})

    def test_shared_geometry_preserves_predictions_gradients_and_rng(self):
        # Noncontiguous padding, a single atom, and disconnected components.
        self.mask[0] = False
        self.mask[0, 3] = True
        self.mask[1, 7] = False
        geometry = prepare_region_geometry(self.bonds, self.mask)
        branch = RandomRegionBranch(16, dropout=0.1)
        h4 = torch.randn(2, 20, 16, requires_grad=True)
        kwargs = dict(radius=2, atoms_per_center=8, max_centers=8)
        for training in (False, True):
            branch.train(training)
            results = []
            for shared in (False, True):
                branch.zero_grad(set_to_none=True)
                h4.grad = None
                torch.manual_seed(123)
                outputs = [branch(h4, self.bonds, self.mask, [10 + view, 20 + view],
                                  geometry=geometry if shared else None, **kwargs)[0]
                           for view in range(1 if training else 5)]
                prediction = torch.stack(outputs).mean(0)
                prediction.square().mean().backward()
                results.append((prediction.detach().clone(), h4.grad.clone(),
                                {name: p.grad.clone() for name, p in branch.named_parameters()},
                                torch.get_rng_state().clone()))
            for left, right in zip(results[0][:2], results[1][:2]):
                torch.testing.assert_close(left, right, atol=0, rtol=0)
            for name in results[0][2]:
                torch.testing.assert_close(results[0][2][name], results[1][2][name], atol=0, rtol=0)
            self.assertTrue(torch.equal(results[0][3], results[1][3]))


if __name__ == "__main__":
    unittest.main()
