"""Method-property tests: no persistent IDs and full coarsening per permutation."""
import itertools
import unittest
import torch

from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig, build_topology
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.config import ModelConfig
from diffusion.bond_diffusion.data import MoleculeGraph, collate_graphs
from diffusion.bond_diffusion.model import BondAwareDiffusionModel
from examples.run_coarse_demo import chain_graph


class CanonicalCoarseningTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(87)

    def adapter(self, config=None):
        encoder = BondAwareDiffusionModel(ModelConfig(hidden_dim=16, num_layers=2, time_dim=8, dropout=0))
        predictor = CoarseGraphPredictor(config or NetworkConfig(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0))
        return DiffusionCoarseModel(encoder, predictor).eval()

    def test_100_full_permutations_have_identical_coarsening_and_prediction(self):
        model = self.adapter()
        cycle = chain_graph(14)
        cycle.bonds[0, -1] = cycle.bonds[-1, 0] = 1
        cycle.node_features[:, 4] = 2
        disconnected = chain_graph(12)
        disconnected.bonds[5, 6] = disconnected.bonds[6, 5] = 0
        disconnected.node_features[:, 4] = (disconnected.bonds > 0).sum(1)
        colored = chain_graph(20)
        colored.node_features[4, 0] = 8
        colored.node_features[11, 1] = 6
        colored.bonds[8, 9] = colored.bonds[9, 8] = 2
        colored.bonds[14, 15] = colored.bonds[15, 14] = 3
        samples = [chain_graph(1), chain_graph(24), chain_graph(100), cycle, disconnected, colored]
        generator = torch.Generator().manual_seed(91)
        with torch.no_grad():
            for graph in samples:
                with self.subTest(nodes=len(graph.node_features)):
                    reference = model(collate_graphs([graph])).graphs[0]
                    for _ in range(100):
                        perm = torch.randperm(len(graph.node_features), generator=generator)
                        moved = MoleculeGraph(graph.node_features[perm], graph.bonds[perm][:, perm], graph.substructure_mask[perm][:, perm])
                        result = model(collate_graphs([moved])).graphs[0]
                        self.assertEqual(reference.topology.canonical_signature(), result.topology.canonical_signature())
                        torch.testing.assert_close(reference.topology.coarse_edges, result.topology.coarse_edges, rtol=0, atol=0)
                        torch.testing.assert_close(reference.coarse_edge_attr, result.coarse_edge_attr, rtol=0, atol=0)
                        torch.testing.assert_close(reference.prediction, result.prediction, rtol=0, atol=1e-6)

    def test_atom_and_bond_labels_and_bidirectional_storage_are_preserved(self):
        # The original node rows deliberately share values with some edge labels:
        # the two color namespaces must still remain disjoint.
        nodes = torch.tensor([[1, 0], [2, 0], [1, 0], [2, 1], [1, 0], [1, 0]])
        edges = torch.tensor([[0, 1, 1, 3, 4], [1, 2, 3, 4, 5]])
        labels = torch.tensor([1, 2, 1, 3, 2])
        reference = build_topology(6, edges, node_labels=nodes, edge_labels=labels)
        for _ in range(20):
            p = torch.randperm(6)
            inverse = torch.argsort(p)
            shuffled = torch.randperm(10)
            both_edges = torch.cat((inverse[edges], inverse[edges.flip(0)]), 1)[:, shuffled]
            both_labels = labels.repeat(2)[shuffled]
            result = build_topology(6, both_edges, node_labels=nodes[p], edge_labels=both_labels)
            self.assertEqual(reference.canonical_signature(), result.canonical_signature())
            torch.testing.assert_close(result.node_labels, nodes[p][result.atom_order])
            torch.testing.assert_close(result.owner_input[result.atom_order], result.owner)
            torch.testing.assert_close(result.input_to_canonical[result.atom_order], torch.arange(6))
        changed_nodes = nodes.clone()
        changed_nodes[0, 0] = 8
        changed_edges = labels.clone()
        changed_edges[0] = 4
        self.assertNotEqual(reference.canonical_signature(), build_topology(6, edges, node_labels=changed_nodes, edge_labels=labels).canonical_signature())
        self.assertNotEqual(reference.canonical_signature(), build_topology(6, edges, node_labels=nodes, edge_labels=changed_edges).canonical_signature())

    def test_canonicalization_never_uses_embeddings_or_persistent_ids(self):
        predictor = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, dropout=0))
        edges = torch.tensor([[0, 1], [1, 2]])
        labels = torch.tensor([[6], [8], [6]])
        x = torch.randn(3, 3)
        first = predictor(x, edges, node_labels=labels)
        second = predictor(x * 100, edges, node_labels=labels)
        self.assertEqual(first.topology.canonical_signature(), second.topology.canonical_signature())
        with self.assertRaisesRegex(ValueError, "raw discrete node_labels"):
            predictor(x, edges)
        with self.assertRaisesRegex(ValueError, "long tensor"):
            predictor(x, edges, node_labels=x)
        with self.assertRaisesRegex(ValueError, "cannot override"):
            predictor(x, edges, node_labels=labels, node_ids=torch.arange(3))

    def test_every_enhancement_switch_is_independent(self):
        graph = chain_graph(30)
        encoder_config = ModelConfig(hidden_dim=16, num_layers=1, time_dim=8, dropout=0)
        # Exercise all combinations, including no region edge features with
        # coarse bond features on, and the reverse. Canonical chemistry is fixed.
        signature = None
        for region_edges, count, coarse_edges, size, weighted in itertools.product((False, True), repeat=5):
            config = NetworkConfig(
                input_dim=16, hidden_dim=8, edge_dim=4, region_layers=1, coarse_layers=1, dropout=0,
                use_region_edge_features=region_edges, use_coarse_edge_count=count,
                use_coarse_edge_features=coarse_edges, use_size_feature=size,
                graph_pool="size_weighted_mean" if weighted else "mean",
            )
            model = DiffusionCoarseModel(BondAwareDiffusionModel(encoder_config), CoarseGraphPredictor(config))
            output = model(collate_graphs([graph])).graphs[0]
            self.assertEqual(output.coarse_edge_attr.size(1), int(count) + 4 * int(coarse_edges))
            self.assertEqual(model.predictor.region_gnn.layers[0].edge_projection is not None, region_edges)
            if signature is None:
                signature = output.topology.canonical_signature()
            self.assertEqual(signature, output.topology.canonical_signature())
            output.prediction.square().sum().backward()
            self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_base_adapter_edge_dim_zero_still_canonicalizes_bond_types(self):
        config = NetworkConfig.base(input_dim=16, hidden_dim=16, edge_dim=0, dropout=0)
        model = self.adapter(config)
        graph = chain_graph(20)
        graph.bonds[6, 7] = graph.bonds[7, 6] = 2
        output = model(collate_graphs([graph])).graphs[0]
        self.assertEqual(output.coarse_edge_attr.size(1), 0)
        self.assertIn(2, output.topology.edge_labels.flatten().tolist())
        self.assertIsNone(model.predictor.size_projection)
        self.assertTrue(all(layer.edge_projection is None for layer in model.predictor.region_gnn.layers))
        self.assertTrue(all(layer.edge_projection is None for layer in model.predictor.coarse_gnn.layers))
        for _ in range(10):
            perm = torch.randperm(20)
            moved = MoleculeGraph(graph.node_features[perm], graph.bonds[perm][:, perm], graph.substructure_mask[perm][:, perm])
            result = model(collate_graphs([moved])).graphs[0]
            torch.testing.assert_close(output.prediction, result.prediction, rtol=0, atol=1e-6)

    def test_canonical_gathers_preserve_feature_gradients(self):
        config = NetworkConfig(input_dim=3, hidden_dim=8, edge_dim=1, dropout=0)
        predictor = CoarseGraphPredictor(config)
        n = 20
        edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
        nodes = torch.randn(n, 3, requires_grad=True)
        attributes = torch.ones(n - 1, 1, requires_grad=True)
        labels = torch.arange(n).unsqueeze(1)
        output = predictor(nodes, edges, attributes, node_labels=labels, edge_labels=torch.ones(n - 1, dtype=torch.long))
        output.prediction.square().sum().backward()
        for values in (nodes, attributes):
            self.assertIsNotNone(values.grad)
            self.assertTrue(torch.isfinite(values.grad).all())
            self.assertGreater(values.grad.abs().sum().item(), 0)


if __name__ == "__main__":
    unittest.main()
