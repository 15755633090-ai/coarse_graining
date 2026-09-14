import unittest

import torch

from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig, build_topology
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.config import ModelConfig
from diffusion.bond_diffusion.data import collate_graphs
from diffusion.bond_diffusion.model import BondAwareDiffusionModel
from run_coarse_demo import chain_graph


def chain_edges(n):
    return torch.stack((torch.arange(n - 1), torch.arange(1, n)))


class CoarseFrameworkTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(12)

    def check_partition(self, topology, n):
        self.assertEqual(sorted(torch.cat(topology.cores).tolist()), list(range(n)))
        self.assertTrue((topology.owner >= 0).all())
        for k, (core, context) in enumerate(zip(topology.cores, topology.contexts)):
            self.assertGreater(core.numel(), 0)
            self.assertTrue(set(core.tolist()) <= set(context.tolist()))
            self.assertTrue((topology.owner[core] == k).all())
        expected = {}
        for u, v in topology.edges.T.tolist():
            a, b = sorted((topology.owner[u].item(), topology.owner[v].item()))
            if a != b:
                expected[(a, b)] = expected.get((a, b), 0) + 1
        actual = dict(zip(map(tuple, topology.coarse_edges.T.tolist()), topology.edge_counts.tolist()))
        self.assertEqual(actual, expected)

    def test_chains_stars_cycles_disconnected_and_singleton(self):
        for n, edges in [
            (1, chain_edges(1)), (100, chain_edges(100)),
            (20, torch.stack((torch.zeros(19, dtype=torch.long), torch.arange(1, 20)))),
            (12, torch.cat((chain_edges(12), torch.tensor([[0], [11]])), 1)),
            (10, torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]])),
        ]:
            with self.subTest(n=n, edges=edges.size(1)):
                topology = build_topology(n, edges)
                self.check_partition(topology, n)
                self.assertGreaterEqual(topology.stats["context_occurrence_ratio"], 1)
                for i in torch.where(topology.is_residual)[0].tolist():
                    self.assertLessEqual(len(topology.cores[i]), 4)

    def test_large_holes_add_centers_and_small_holes_survive(self):
        topology = build_topology(60, chain_edges(60), CoarseningConfig(radius=1, center_fraction=0.01, max_residual_size=2, canonicalize=False))
        self.check_partition(topology, 60)
        self.assertGreater(topology.stats["num_added_centers"], 0)
        small = build_topology(10, chain_edges(10), CoarseningConfig(radius=1, center_fraction=0.01, max_residual_size=10, canonicalize=False))
        self.check_partition(small, 10)
        self.assertEqual(small.stats["num_residual_regions"], 1)
        self.assertEqual(small.stats["num_residual_nodes"], 7)

    def test_large_radius_does_not_connect_disconnected_components(self):
        topology = build_topology(4, torch.tensor([[0, 2], [1, 3]]), CoarseningConfig(radius=100))
        self.check_partition(topology, 4)
        self.assertEqual(len(topology.contexts), 2)
        self.assertEqual(topology.coarse_edges.size(1), 0)

    def test_bidirectional_storage_counts_edges_once_and_sums_attributes(self):
        n = 6
        edges = chain_edges(n)
        attributes = torch.arange(1, n, dtype=torch.float32).unsqueeze(1)
        model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, edge_dim=1, dropout=0), CoarseningConfig(radius=1, center_fraction=0.4, canonicalize=False))
        x = torch.randn(n, 3)
        once = model(x, edges, attributes)
        twice = model(x, torch.cat((edges, edges.flip(0)), 1), attributes.repeat(2, 1))
        torch.testing.assert_close(once.prediction, twice.prediction)
        torch.testing.assert_close(once.coarse_edge_attr, twice.coarse_edge_attr)
        self.assertEqual(int(once.coarse_edge_attr[:, 0].sum()), len(once.topology.boundary_edge_ids))
        for i in range(once.topology.coarse_edges.size(1)):
            ids = once.topology.boundary_edge_ids[once.topology.boundary_groups == i]
            torch.testing.assert_close(once.coarse_edge_attr[i, 1], attributes[ids].sum())
        with self.assertRaisesRegex(ValueError, "identical attributes"):
            model(x, torch.cat((edges, edges.flip(0)), 1), torch.cat((attributes, attributes + 1)))

    def test_pool_only_core_but_receive_context_messages(self):
        model = CoarseGraphPredictor(
            NetworkConfig(input_dim=3, hidden_dim=8, region_layers=1, coarse_layers=0, dropout=0, use_size_feature=False),
            CoarseningConfig(radius=2, center_fraction=0.4, canonicalize=False),
        )
        x = torch.randn(14, 3)
        output = model(x, chain_edges(14))
        topo = output.topology
        candidates = [i for i, (core, ctx) in enumerate(zip(topo.cores, topo.contexts)) if len(ctx) > len(core)]
        self.assertTrue(candidates)
        k = candidates[0]
        projected = model.input_projection(x)
        local = model.region_gnn(projected[topo.contexts[k]], topo.context_edges[k], x.new_empty((len(topo.context_edge_ids[k]), 0)))
        torch.testing.assert_close(output.region_embeddings[k], local[topo.core_positions[k]].mean(0))
        # Perturb a context-only node adjacent to the core: it must affect that core.
        core_set = set(topo.cores[k].tolist())
        context_set = set(topo.contexts[k].tolist())
        outside = next(v for u, v in torch.cat((topo.edges, topo.edges.flip(0)), 1).T.tolist() if u in core_set and v in context_set - core_set)
        perturbed = x.clone()
        perturbed[outside] += torch.tensor([8., -3., 5.])
        changed = model(perturbed, chain_edges(14))
        self.assertFalse(torch.allclose(output.region_embeddings[k], changed.region_embeddings[k]))

    def test_legacy_permutation_with_persistent_ids(self):
        n = 14
        x = torch.randn(n, 3)
        edges = chain_edges(n)
        model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, dropout=0), CoarseningConfig(radius=2, canonicalize=False))
        ids = torch.arange(n) + 100
        first = model(x, edges, node_ids=ids)
        permutation = torch.randperm(n)
        inverse = torch.argsort(permutation)
        second = model(x[permutation], inverse[edges], node_ids=ids[permutation])
        torch.testing.assert_close(second.topology.owner, first.topology.owner[permutation])
        torch.testing.assert_close(first.prediction, second.prediction, atol=1e-6, rtol=1e-5)

    def test_edge_features_affect_predictions_and_receive_gradients(self):
        model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, edge_dim=1, dropout=0), CoarseningConfig(radius=1, canonicalize=False))
        x = torch.randn(16, 3, requires_grad=True)
        attributes = torch.ones(15, 1, requires_grad=True)
        first = model(x, chain_edges(16), attributes)
        second = model(x, chain_edges(16), attributes * 4)
        self.assertFalse(torch.allclose(first.prediction, second.prediction))
        first.prediction.square().sum().backward()
        for value in (x, attributes):
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(value.grad.abs().sum().item(), 0)
        for module in (model.region_gnn, model.coarse_gnn, model.head):
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in module.parameters()))

    def make_adapter(self, frozen):
        encoder = BondAwareDiffusionModel(ModelConfig(hidden_dim=16, num_layers=1, dropout=0.2, time_dim=8))
        predictor = CoarseGraphPredictor(NetworkConfig(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0, region_layers=1, coarse_layers=1))
        return DiffusionCoarseModel(encoder, predictor, frozen)

    def test_frozen_encoder_and_batch_padding(self):
        model = self.make_adapter(True)
        model.train()
        self.assertFalse(model.encoder.training)
        graphs = [chain_graph(1), chain_graph(15)]
        batched = model(collate_graphs(graphs))
        self.assertEqual(batched.predictions.shape, (2, 1))
        for i, graph in enumerate(graphs):
            alone = model(collate_graphs([graph]))
            torch.testing.assert_close(batched.predictions[i], alone.predictions[0], atol=1e-6, rtol=1e-5)
        batched.predictions.square().sum().backward()
        self.assertTrue(all(p.grad is None for p in model.encoder.parameters()))
        self.assertTrue(any(p.grad is not None for p in model.predictor.parameters()))

    def test_finetuning_reaches_encoder_and_optimizer_updates_head(self):
        model = self.make_adapter(False).train()
        optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.001)
        before = model.predictor.head[-1].weight.detach().clone()
        result = model(collate_graphs([chain_graph(1), chain_graph(20)]))
        (result.predictions - 1).square().mean().backward()
        for layer in model.encoder.layers:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
        optimizer.step()
        self.assertFalse(torch.equal(before, model.predictor.head[-1].weight))
        self.assertTrue(all(p.grad is None for p in model.encoder.edge_head.parameters()))

    def test_pooling_modes_zero_layers_and_invalid_inputs(self):
        for pool in ("mean", "sum", "size_weighted_mean"):
            model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, region_layers=0, coarse_layers=0, graph_pool=pool), CoarseningConfig(canonicalize=False))
            self.assertTrue(torch.isfinite(model(torch.randn(1, 3), chain_edges(1)).prediction).all())
        with self.assertRaisesRegex(ValueError, "Empty"):
            build_topology(0, chain_edges(1))
        with self.assertRaisesRegex(ValueError, "self-loops"):
            build_topology(1, torch.tensor([[0], [0]]))
        with self.assertRaisesRegex(ValueError, "out-of-range"):
            build_topology(2, torch.tensor([[0], [2]]))
        with self.assertRaises(ValueError):
            CoarseningConfig(center_fraction=0)
        model = self.make_adapter(True)
        batch = collate_graphs([chain_graph(2)])
        batch.bonds[0, 0, 1] = batch.bonds[0, 1, 0] = 5
        with self.assertRaisesRegex(ValueError, "clean bond"):
            model(batch)


if __name__ == "__main__":
    unittest.main()
