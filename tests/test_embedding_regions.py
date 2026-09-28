import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.embedding_regions import spherical_regions
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel


class EmbeddingRegionTests(unittest.TestCase):
    def test_training_cache_and_resume_on_synthetic_dataset(self):
        from test_training import _TinyPropertyDataset
        from multiscale_tokenizer.training import train_property_model
        source = Path(__file__).resolve().parents[1] / "runs/lipo_formal_stage1_seed0_replay_v1/best.pt"
        if not source.is_file():
            self.skipTest("local Lipo source checkpoint unavailable")
        with TemporaryDirectory() as directory, patch(
            "multiscale_tokenizer.training.OGBMoleculePropertyDataset", _TinyPropertyDataset,
        ):
            common = dict(dataset_root=Path(directory) / "data", task="lipo", output_dir=Path(directory) / "run",
                          device="cpu", epochs=1, batch_size=2, micro_batch_size=2,
                          amp_dtype="none", experiment_mode="embedding_region_stage2_frozen",
                          encoder_init_checkpoint=source, eval_views=1)
            result = train_property_model(**common)
            self.assertTrue(result["embedding_region_cache"])
            self.assertFalse(result["training_protocol"]["embedding_regions"]["topology_in_partition"])
            self.assertTrue((Path(directory) / "run/embedding_partitions.pt").is_file())
            resumed = train_property_model(**common, resume=True)
            self.assertEqual(result["history"], resumed["history"])
            self.assertEqual(set(result["embedding_region_cache"]), set(resumed["embedding_region_cache"]))

    def test_cosine_groups_disconnected_indices_and_ignores_magnitude(self):
        x = np.array([[1, 0], [0, 1], [4, 0], [0, 3]], dtype=float)
        centers, members = spherical_regions(x, max_centers=2)
        self.assertEqual({tuple(np.flatnonzero(row)) for row in members}, {(0, 2), (1, 3)})
        np.testing.assert_array_equal(members.sum(0), 1)
        self.assertTrue(members[np.arange(2), centers].all())
        np.testing.assert_array_equal(members, spherical_regions(x * 7, max_centers=2)[1])

    def test_degenerate_vectors_exact_nonempty_count_and_determinism(self):
        for x in (np.zeros((17, 4)), np.ones((17, 4)), np.ones((1, 4)), np.eye(7)):
            a = spherical_regions(x, atoms_per_center=1, max_centers=8)
            b = spherical_regions(x, atoms_per_center=1, max_centers=8)
            np.testing.assert_array_equal(a[1], b[1])
            np.testing.assert_array_equal(a[1].sum(0), 1)
            self.assertTrue((a[1].sum(1) > 0).all())
            self.assertEqual(len(a[0]), min(len(x), 8))
        for x in (np.empty((0, 4)), np.array([[np.nan]])):
            with self.assertRaises(ValueError):
                spherical_regions(x)

    def test_model_cache_gradients_and_unchanged_architecture(self):
        torch.manual_seed(7)
        encoder = DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0))
        config = MultiscaleModelConfig(hidden_dim=16, dropout=0,
                                      experiment_mode="embedding_region_stage2_frozen")
        model = MultiscaleMolecularModel(encoder, config=config)
        random_model = MultiscaleMolecularModel(
            DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0)),
            config=replace(config, experiment_mode="random_region_stage2_frozen"))
        self.assertEqual({k: v.shape for k, v in model.state_dict().items()},
                         {k: v.shape for k, v in random_model.state_dict().items()})
        features = torch.zeros(2, 12, 5, dtype=torch.long)
        features[..., 0] = torch.arange(12) % 8 + 1
        bonds = torch.zeros(2, 12, 12, dtype=torch.long)
        for i in range(11):
            bonds[:, i, i+1] = bonds[:, i+1, i] = 1
        mask = torch.ones(2, 12, dtype=torch.bool)
        mask[1, 1:] = False
        model.prepare_embedding_regions(features, bonds, mask)
        self.assertEqual(len(model.embedding_region_cache), 2)
        with patch("multiscale_tokenizer.model.spherical_regions", side_effect=AssertionError("reclustered")):
            model.train()
            out = model(features, bonds, mask, [10, 11])
            out.prediction.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            self.assertGreater(model.random_regions.region_mlp[0].weight.grad.abs().sum(), 0)
            model.eval()
            first = model(features, bonds, mask, [100, 101]).prediction
            second = model(features, bonds, mask, [999, 998]).prediction
            torch.testing.assert_close(first, second, atol=0, rtol=0)
            single = model(features[1:2, :1], bonds[1:2, :1, :1], mask[1:2, :1]).prediction
            torch.testing.assert_close(first[1:], single)
            self.assertEqual(out.token_counts[:, 3].tolist(), [2, 1])
            self.assertTrue(torch.isfinite(out.prediction).all())
        # Cached membership is identical for the FP32 and BF16 downstream paths.
        with torch.autocast("cpu", dtype=torch.bfloat16):
            model(features, bonds, mask)
        self.assertEqual(len(model.embedding_region_cache), 2)


if __name__ == "__main__":
    unittest.main()
