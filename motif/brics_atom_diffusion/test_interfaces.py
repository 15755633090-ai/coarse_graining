"""Scientific interface invariants; no performance claims."""
from copy import deepcopy
import unittest
from unittest.mock import patch

import torch
from torch_geometric.data import Batch

from .data import ATOM_DIM, AtomDataset, graph_from_smiles
from .diffusion import GaussianSchedule
from .model import AtomDiffusionModel, Config


class InterfaceTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(41)
        self.graph = graph_from_smiles("CCOC(=O)c1ccccc1")
        self.batch = Batch.from_data_list([self.graph])
        self.model = AtomDiffusionModel(Config(hidden_dim=16, layers=2, dropout=0)).eval()
        self.t = torch.tensor([200])

    def test_noisy_input_changes_output_and_has_gradient(self):
        xt = torch.randn_like(self.batch.x, requires_grad=True)
        prediction = self.model.predict_noise(self.batch, xt, self.t)
        self.assertEqual(prediction.shape, (self.batch.num_nodes, ATOM_DIM))
        gradient = torch.autograd.grad(prediction.square().sum(), xt)[0]
        self.assertGreater(float(gradient.abs().sum()), 0)
        changed = self.model.predict_noise(self.batch, xt.detach() + torch.randn_like(xt), self.t)
        self.assertFalse(torch.allclose(prediction, changed))

    def test_timestep_reaches_node_output(self):
        xt = torch.randn_like(self.batch.x)
        first = self.model.predict_noise(self.batch, xt, torch.tensor([0]))
        second = self.model.predict_noise(self.batch, xt, torch.tensor([900]))
        self.assertFalse(torch.allclose(first, second))

    def test_no_clean_atom_feature_or_fragment_identity_bypass(self):
        xt = torch.randn_like(self.batch.x)
        expected = self.model.predict_noise(self.batch, xt, self.t)
        altered = self.batch.clone()
        altered.x = torch.randn_like(altered.x) * 20
        altered.fragment_id = torch.randint(0, 100, altered.fragment_id.shape)
        actual = self.model.predict_noise(altered, xt, self.t)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)

    def test_internal_state_isolated_but_context_responds(self):
        xt = torch.randn_like(self.batch.x)
        _, first = self.model.predict_noise(self.batch, xt, self.t, return_states=True)
        chosen = self.batch.fragment_id == 0
        self.assertTrue((~chosen).any())
        modified = xt.clone()
        modified[~chosen] += 5 * torch.randn_like(modified[~chosen])
        _, second = self.model.predict_noise(self.batch, modified, self.t, return_states=True)
        torch.testing.assert_close(first["internal"][chosen], second["internal"][chosen], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first["context"][chosen], second["context"][chosen]))

    def test_atom_and_edge_permutation_equivariance(self):
        permutation = torch.randperm(self.batch.num_nodes)
        inverse = torch.argsort(permutation)
        edge_order = torch.randperm(self.batch.edge_index.shape[1])
        permuted = self.batch.clone()
        for name in ("x", "boundary", "fragment_id", "batch"):
            permuted[name] = self.batch[name][permutation]
        permuted.edge_index = inverse[self.batch.edge_index[:, edge_order]]
        permuted.edge_attr = self.batch.edge_attr[edge_order]
        permuted.is_cross = self.batch.is_cross[edge_order]
        xt = torch.randn_like(self.batch.x)
        expected = self.model.predict_noise(self.batch, xt, self.t)
        actual = self.model.predict_noise(permuted, xt[permutation], self.t)
        torch.testing.assert_close(actual, expected[permutation], atol=1e-6, rtol=1e-5)
        torch.testing.assert_close(self.model(permuted), self.model(self.batch), atol=1e-6, rtol=1e-5)

    def test_graph_batching_and_endpoint_masks(self):
        other = graph_from_smiles("[Na+].[Cl-]")
        joined = Batch.from_data_list([self.graph, other])
        src, dst = joined.edge_index
        self.assertTrue(torch.equal(joined.is_cross, joined.fragment_id[src] != joined.fragment_id[dst]))
        self.assertTrue(torch.equal(joined.batch[src], joined.batch[dst]))
        self.assertTrue(joined.boundary[src[joined.is_cross]].all())
        combined = self.model(joined)
        separate = torch.cat([self.model(Batch.from_data_list([g])) for g in (self.graph, other)])
        torch.testing.assert_close(combined, separate, atol=1e-6, rtol=1e-5)

    def test_boundary_gate_alignment_and_range(self):
        _, states = self.model(self.batch, return_states=True)
        edges = self.batch.edge_index[:, self.batch.is_cross]
        self.assertTrue(torch.equal(states["cross_edge_index"], edges))
        for gate in states["gates"]:
            self.assertEqual(gate.shape, (edges.shape[1], 1))
            self.assertTrue(((gate > 0) & (gate < 1)).all())

    def test_uniform_boundary_ablation_keeps_edges_and_sets_gate_to_one(self):
        model = AtomDiffusionModel(Config(hidden_dim=16, layers=2, dropout=0, conditional_gate=False)).eval()
        _, states = model(self.batch, return_states=True)
        expected_edges = self.batch.edge_index[:, self.batch.is_cross]
        self.assertGreater(expected_edges.shape[1], 0)
        self.assertTrue(torch.equal(states["cross_edge_index"], expected_edges))
        for gate in states["gates"]:
            torch.testing.assert_close(gate, torch.ones_like(gate), atol=0, rtol=0)
        self.assertFalse(any(".gate." in name for name, _ in model.named_parameters()))

    def test_schedule_known_noise_reconstruction(self):
        schedule = GaussianSchedule(1000)
        x0 = torch.randn(4, ATOM_DIM, dtype=torch.float64)
        noise = torch.randn_like(x0)
        steps = torch.tensor([0, 1, 500, 1000])
        xt = schedule.corrupt(x0, steps, noise)
        restored = schedule.reconstruct(xt, noise, steps)
        torch.testing.assert_close(restored, x0, atol=1e-12, rtol=1e-12)
        torch.testing.assert_close(xt[0], x0[0], atol=0, rtol=0)
        self.assertTrue((schedule.alpha_bar[1:] < schedule.alpha_bar[:-1]).all())
        for invalid in (torch.tensor([-1, 1, 1, 1]), torch.tensor([1001, 1, 1, 1])):
            with self.assertRaises(ValueError):
                schedule.corrupt(x0, invalid, noise)

    def test_one_timestep_per_molecule(self):
        joined = Batch.from_data_list([self.graph, graph_from_smiles("CC")])
        steps = self.model.schedule.node_steps(torch.tensor([10, 20]), joined)
        self.assertTrue((steps[joined.batch == 0] == 10).all())
        self.assertTrue((steps[joined.batch == 1] == 20).all())
        with self.assertRaises(ValueError):
            self.model.predict_noise(joined, joined.x, torch.tensor([10]))

    def test_single_atom_and_ablation_switches(self):
        single = Batch.from_data_list([graph_from_smiles("C")])
        configs = (Config(hidden_dim=8, dropout=0), Config(hidden_dim=8, dropout=0, dual_state=False),
                   Config(hidden_dim=8, dropout=0, motif_prior=False),
                   Config(hidden_dim=8, dropout=0, conditional_gate=False))
        for config in configs:
            model = AtomDiffusionModel(config).eval()
            for data in (single, self.batch):
                loss, stats = model.denoise_loss(data, torch.Generator().manual_seed(7))
                self.assertTrue(torch.isfinite(loss))
                self.assertEqual(stats["all_atoms"], data.num_nodes)
                self.assertEqual(stats["boundary_atoms"] + stats["internal_atoms"], data.num_nodes)
                self.assertTrue(torch.isfinite(model(data)).all())

    def test_pretrain_and_finetune_gradients_reach_encoder(self):
        self.model.set_stage("pretrain")
        loss, _ = self.model.denoise_loss(self.batch, torch.Generator().manual_seed(99))
        loss.backward()
        self.assertGreater(float(self.model.encoder.layers[0].gate[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.property_head[0].weight.grad)
        self.model.zero_grad(set_to_none=True)
        self.model.set_stage("finetune")
        self.model(self.batch).square().sum().backward()
        self.assertGreater(float(self.model.encoder.layers[0].gate[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.noise_head[0].weight.grad)
        self.assertTrue(all(p.requires_grad for p in self.model.encoder.parameters()))

    def test_encoder_transfer_and_clean_prediction_no_sampling(self):
        transferred = AtomDiffusionModel(self.model.config)
        transferred.encoder.load_state_dict(self.model.encoder.state_dict(), strict=True)
        transferred.eval()
        with patch("torch.randn", side_effect=AssertionError("Unexpected sampling")), \
             patch("torch.randint", side_effect=AssertionError("Unexpected sampling")):
            _, source = self.model(self.batch, return_states=True)
            _, target = transferred(self.batch, return_states=True)
        torch.testing.assert_close(source["atoms"], target["atoms"], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
