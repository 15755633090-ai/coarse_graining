import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch

from diffusion_encoder.model import DiffusionEncoder, ModelConfig
from multiscale_tokenizer.center_tokens import (
    CenterTokenBranch, CenterSelection, bucket_graph_distances, select_center_tokens,
)
from multiscale_tokenizer.model import MultiscaleModelConfig, MultiscaleMolecularModel
from multiscale_tokenizer.random_regions import graph_distances


def chain(size):
    bonds = torch.zeros(size, size, dtype=torch.long)
    for i in range(size - 1):
        bonds[i, i + 1] = bonds[i + 1, i] = 1
    return bonds


class CenterTokenTests(unittest.TestCase):
    def test_budget_bands_determinism_and_small_molecules(self):
        for size, expected in ((1, 1), (2, 2), (11, 3), (27, 7), (45, 8)):
            states = tuple(np.random.default_rng(7 + layer).normal(size=(size, 8))
                           for layer in range(3))
            distances = graph_distances(chain(size))
            first = select_center_tokens(distances, states)
            second = select_center_tokens(distances, states)
            self.assertEqual(len(first.indices), expected)
            self.assertEqual(len(np.unique(first.indices)), expected)
            np.testing.assert_array_equal(first.indices, second.indices)
            np.testing.assert_array_equal(first.scales, second.scales)
            self.assertTrue(np.isfinite(first.core_radii).all())
            self.assertTrue(((0 <= first.core_radii) & (first.core_radii <= 1)).all())
            self.assertTrue(np.isin(first.scales, [0, 1, 2]).all())
            self.assertEqual(first.pair_buckets.shape, (expected, expected))
            if size == 11:
                self.assertEqual(set(first.scales.tolist()), {0, 1, 2})
            if size == 27:
                self.assertEqual(len(first.cores), 2)

    def test_disconnected_graph_and_distance_buckets(self):
        bonds = torch.block_diag(chain(4), chain(3))
        distances = graph_distances(bonds)
        states = tuple(np.random.default_rng(k).normal(size=(7, 8)) for k in range(3))
        selection = select_center_tokens(distances, states, min_centers=6)
        self.assertEqual(len(selection.indices), 6)
        self.assertEqual(len(selection.cores), 2)
        self.assertIn(6, selection.pair_buckets)
        np.testing.assert_array_equal(bucket_graph_distances(np.array([-1, 0, 4, 5, 9])),
                                      [6, 0, 4, 5, 5])

    def test_feature_diversity_changes_a_topologically_tied_far_center(self):
        distances = graph_distances(chain(7))
        h2 = np.tile([1.0, 0.0], (7, 1))
        h3 = h2.copy()
        far_left = h2.copy()
        far_left[0] = [-1, 0]
        far_right = h2.copy()
        far_right[6] = [-1, 0]
        left = select_center_tokens(distances, (h2, h3, far_left))
        right = select_center_tokens(distances, (h2, h3, far_right))
        self.assertEqual(left.cores.tolist(), [3])
        self.assertEqual(right.cores.tolist(), [3])
        self.assertEqual(left.indices[1], 0)
        self.assertEqual(right.indices[1], 6)
        self.assertEqual(left.scales[1], 2)

    def test_branch_uses_only_selected_atoms(self):
        branch = CenterTokenBranch(hidden_dim=8, dropout=0).eval()
        states = tuple(torch.randn(1, 5, 8) for _ in range(3))
        selection = CenterSelection(
            indices=np.array([0, 2, 4]), scales=np.array([0, 1, 2]),
            core_radii=np.array([0, 0.5, 1], dtype=np.float32),
            pair_buckets=np.array([[0, 2, 4], [2, 0, 2], [4, 2, 0]]),
            cores=np.array([0]),
        )
        mask = torch.ones(1, 5, dtype=torch.bool)
        original, counts = branch(states, mask, [selection], 4)
        changed = tuple(h.clone() for h in states)
        for h in changed:
            h[:, 1] += 100
            h[:, 3] -= 100
        after, _ = branch(changed, mask, [selection], 4)
        torch.testing.assert_close(original, after, atol=0, rtol=0)
        self.assertEqual(counts.tolist(), [[1, 1, 1]])

    def test_model_uses_selected_atom_states_and_fixed_cache(self):
        torch.manual_seed(4)
        model = MultiscaleMolecularModel(
            DiffusionEncoder(ModelConfig(hidden_dim=16, dropout=0)),
            config=MultiscaleModelConfig(hidden_dim=16, dropout=0,
                                         experiment_mode="center_token_stage2_frozen"),
        )
        features = torch.zeros(2, 20, 5, dtype=torch.long)
        features[..., 0] = torch.arange(20) % 8 + 1
        features[..., 1] = 5
        bonds = chain(20).repeat(2, 1, 1)
        mask = torch.ones(2, 20, dtype=torch.bool)
        mask[1, 1:] = False
        model.prepare_center_tokens(features, bonds, mask)
        self.assertEqual(len(model.center_token_cache), 2)
        with patch("multiscale_tokenizer.model.select_center_tokens",
                   side_effect=AssertionError("center selection repeated")):
            model.train()
            output = model(features, bonds, mask)
            self.assertEqual(output.graph_representation.shape, (2, 64))
            self.assertEqual(output.token_counts[:, 1:].sum(1).tolist(), [5, 1])
            output.prediction.square().mean().backward()
            self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for p in model.center_tokens.parameters()))
            model.eval()
            with torch.no_grad():
                whole = model(features, bonds, mask).prediction
                single = model(features[1:2, :1], bonds[1:2, :1, :1], mask[1:2, :1]).prediction
            torch.testing.assert_close(whole[1:], single)
            with torch.autocast("cpu", dtype=torch.bfloat16):
                model(features, bonds, mask)
        self.assertEqual(len(model.center_token_cache), 2)

    def test_training_cache_resume_and_checkpoint_reload(self):
        from test_training import _TinyPropertyDataset
        from multiscale_tokenizer.training import train_property_model, load_property_model

        source = Path(__file__).resolve().parents[1] / "runs/esol_seed0_stage1_base_finetune/best.pt"
        if not source.is_file():
            self.skipTest("local ESOL Stage-1 checkpoint unavailable")
        with TemporaryDirectory() as directory, patch(
            "multiscale_tokenizer.training.OGBMoleculePropertyDataset", _TinyPropertyDataset,
        ):
            output = Path(directory) / "run"
            common = dict(dataset_root=Path(directory) / "data", task="esol", output_dir=output,
                          device="cpu", epochs=1, batch_size=2, micro_batch_size=2,
                          amp_dtype="none", experiment_mode="center_token_stage2_frozen",
                          encoder_init_checkpoint=source)
            result = train_property_model(**common)
            self.assertTrue(result["center_token_cache"])
            self.assertTrue((output / "center_selections.pt").is_file())
            restored, _, _ = load_property_model(output / "best.pt", device="cpu")
            self.assertEqual(set(restored.center_token_cache), set(result["center_token_cache"]))
            completed = train_property_model(**common, resume=True)
            self.assertEqual(result["history"], completed["history"])


if __name__ == "__main__":
    unittest.main()
