"""Cache equivalence, invalidation and no-recomputation contract tests."""
from dataclasses import asdict, replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig, TopologyCache, build_topology
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel, precompute_batch_topologies
from diffusion.bond_diffusion.config import ModelConfig
from diffusion.bond_diffusion.data import MoleculeGraph, collate_graphs
from diffusion.bond_diffusion.model import BondAwareDiffusionModel
from run_coarse_demo import chain_graph


def chain_inputs(n=25):
    edges = torch.stack((torch.arange(n - 1), torch.arange(1, n)))
    labels = dict(node_labels=torch.arange(n).remainder(3).unsqueeze(1),
                  edge_labels=torch.ones(n - 1, dtype=torch.long))
    return edges, labels


class TopologyCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        torch.manual_seed(34)

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            self.assertEqual(a.dtype, b.dtype)
            self.assertEqual(b.device.type, "cpu")
            self.assertTrue(torch.equal(a, b))
            self.assertFalse(b.requires_grad)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, list):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_full_disk_roundtrip_memory_eviction_and_no_rebuild(self):
        with tempfile.TemporaryDirectory() as directory:
            cache = TopologyCache(directory, max_memory_entries=1)
            saved = []
            for n in (1, 25, 100):
                edges, labels = chain_inputs(n)
                topology = cache.get_or_build(n, edges, **labels)
                saved.append(topology)
                self.assert_nested_equal(asdict(topology), asdict(build_topology(n, edges, **labels)))
            self.assertEqual(cache.stats["memory_entries"], 1)
            self.assertEqual(len(list(Path(directory).glob("*.pt"))), 3)
            # A new process/object must reuse every stored field without Bliss/BFS.
            fresh = TopologyCache(directory)
            with patch("coarse_gnn.cache.build_topology", side_effect=AssertionError("Rebuilt topology")), \
                 patch("coarse_gnn.topology.canonical_atom_order", side_effect=AssertionError("Ran Bliss")), \
                 patch("coarse_gnn.topology._distances", side_effect=AssertionError("Ran BFS")):
                for n, reference in zip((1, 25, 100), saved):
                    edges, labels = chain_inputs(n)
                    loaded = fresh.get_or_build(n, edges, **labels)
                    self.assert_nested_equal(asdict(reference), asdict(loaded))
                    self.assertIs(loaded, fresh.get_or_build(n, edges, **labels))
            self.assertEqual(fresh.stats, dict(hits=3, disk_hits=3, misses=0, memory_entries=3))
            fresh.clear_memory()
            self.assertEqual(fresh.stats["memory_entries"], 0)

    def test_input_config_and_version_changes_invalidate(self):
        cache = TopologyCache()
        edges, labels = chain_inputs()
        config = CoarseningConfig()
        original = cache.get_or_build(25, edges, config, **labels)
        alternatives = []
        for field, value in (("radius", 2), ("center_fraction", 0.3),
                             ("max_residual_size", 1), ("canonicalize", False)):
            alternatives.append(cache.get_or_build(25, edges, replace(config, **{field: value}), **labels))
        changed_nodes = labels["node_labels"].clone()
        changed_nodes[0] = 7
        alternatives.append(cache.get_or_build(25, edges, node_labels=changed_nodes, edge_labels=labels["edge_labels"]))
        changed_bonds = labels["edge_labels"].clone()
        changed_bonds[0] = 2
        alternatives.append(cache.get_or_build(25, edges, node_labels=labels["node_labels"], edge_labels=changed_bonds))
        alternatives.append(cache.get_or_build(25, edges.flip(1), **labels))
        changed_edges = edges.clone()
        changed_edges[:, -1] = torch.tensor([0, 24])
        alternatives.append(cache.get_or_build(25, changed_edges, **labels))
        with patch("coarse_gnn.topology.TOPOLOGY_VERSION", 999):
            alternatives.append(cache.get_or_build(25, edges, **labels))
        self.assertEqual(len({t.input_fingerprint for t in [original] + alternatives}), 10)
        self.assertEqual(cache.misses, 10)
        legacy = replace(config, canonicalize=False)
        first = cache.get_or_build(25, edges, legacy, node_ids=torch.arange(25), **labels)
        second = cache.get_or_build(25, edges, legacy, node_ids=torch.arange(25).flip(0), **labels)
        self.assertNotEqual(first.input_fingerprint, second.input_fingerprint)

    def test_cached_and_explicit_predictions_and_gradients_match_uncached(self):
        edges, labels = chain_inputs()
        # Shuffled bidirectional storage exercises input_edge_ids/edge_positions.
        shuffle = torch.randperm(48)
        edges = torch.cat((edges, edges.flip(0)), 1)[:, shuffle]
        labels["edge_labels"] = labels["edge_labels"].repeat(2)[shuffle]
        x0 = torch.randn(25, 3)
        for edge_dim in (0, 1):
            with self.subTest(edge_dim=edge_dim), tempfile.TemporaryDirectory() as directory:
                model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, edge_dim=edge_dim, dropout=0))
                attr0 = torch.randn(24, edge_dim).repeat(2, 1)[shuffle]
                def run(topology=None):
                    model.zero_grad(set_to_none=True)
                    x = x0.clone().requires_grad_()
                    attr = attr0.clone().requires_grad_()
                    out = model(x, edges, attr, **labels, topology=topology)
                    out.prediction.square().sum().backward()
                    gradients = {name: p.grad.clone() for name, p in model.named_parameters() if p.grad is not None}
                    return out, x.grad, attr.grad, gradients
                expected = run()
                TopologyCache(directory).get_or_build(25, edges, **labels)
                model.topology_cache = TopologyCache(directory)
                with patch("coarse_gnn.cache.build_topology", side_effect=AssertionError("Unexpected build")), \
                     patch("coarse_gnn.topology.canonical_atom_order", side_effect=AssertionError("Unexpected Bliss")):
                    cached = run()
                    explicit = run(cached[0].topology)
                for actual in (cached, explicit):
                    torch.testing.assert_close(expected[0].prediction, actual[0].prediction, rtol=0, atol=0)
                    torch.testing.assert_close(expected[0].coarse_edge_attr, actual[0].coarse_edge_attr, rtol=0, atol=0)
                    torch.testing.assert_close(expected[1], actual[1], rtol=0, atol=0)
                    if edge_dim:
                        torch.testing.assert_close(expected[2], actual[2], rtol=0, atol=0)
                        self.assertGreater(actual[2].abs().sum().item(), 0)
                    self.assertEqual(expected[3].keys(), actual[3].keys())
                    for name in expected[3]:
                        torch.testing.assert_close(expected[3][name], actual[3][name], rtol=0, atol=0)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    model(x0, edges.flip(1), attr0, **labels, topology=cached[0].topology)
                model.coarsening = CoarseningConfig(radius=2)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    model(x0, edges, attr0, **labels, topology=cached[0].topology)

    def test_molecular_precomputation_batch_shuffle_and_encoder_finetuning(self):
        graphs = [chain_graph(1), chain_graph(25), chain_graph(12)]
        graphs[2].bonds[5, 6] = graphs[2].bonds[6, 5] = 0
        graphs[2].node_features[:, 4] = (graphs[2].bonds > 0).sum(1)
        with tempfile.TemporaryDirectory() as directory:
            cache = TopologyCache(directory)
            precomputed = precompute_batch_topologies(collate_graphs(graphs), cache)
            encoder = BondAwareDiffusionModel(ModelConfig(hidden_dim=16, num_layers=1, dropout=0, time_dim=8))
            predictor = CoarseGraphPredictor(NetworkConfig(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0),
                                             topology_cache=TopologyCache(directory))
            model = DiffusionCoarseModel(encoder, predictor, freeze_encoder=False).eval()
            batch = collate_graphs(graphs)
            with patch.object(encoder, "encode_nodes", side_effect=AssertionError("Precomputation encoded features")):
                prepared = model.precompute_topologies(batch)
            for first, second in zip(precomputed, prepared):
                self.assert_nested_equal(asdict(first), asdict(second))
            with patch("coarse_gnn.cache.build_topology", side_effect=AssertionError("Recomputed cached topology")):
                output = model(batch)
                explicit = model(batch, topologies=prepared)
                torch.testing.assert_close(output.predictions, explicit.predictions, rtol=0, atol=0)
                moved = model(collate_graphs([graphs[2], graphs[0]]))
                torch.testing.assert_close(moved.predictions, output.predictions[[2, 0]], atol=1e-6, rtol=0)
                with self.assertRaisesRegex(ValueError, "does not match"):
                    model(collate_graphs([graphs[2], graphs[0]]), topologies=prepared[:2])
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
                before = encoder.layers[0].state_dict()
                before = {key: value.clone() for key, value in before.items()}
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    model(batch).predictions.square().mean().backward()
                    self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in encoder.layers[0].parameters()))
                    self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))
                    optimizer.step()
                self.assertTrue(any(not torch.equal(value, encoder.layers[0].state_dict()[key]) for key, value in before.items()))

    def test_permutations_build_separate_safe_entries_and_keep_invariance(self):
        cache = TopologyCache()
        model = CoarseGraphPredictor(NetworkConfig(input_dim=3, hidden_dim=8, dropout=0), topology_cache=cache).eval()
        edges, labels = chain_inputs()
        x = labels["node_labels"].float().repeat(1, 3)
        reference = model(x, edges, **labels)
        for _ in range(20):
            permutation = torch.randperm(25)
            inverse = torch.argsort(permutation)
            moved_labels = dict(node_labels=labels["node_labels"][permutation], edge_labels=labels["edge_labels"])
            moved = model(x[permutation], inverse[edges], **moved_labels)
            self.assertEqual(reference.topology.canonical_signature(), moved.topology.canonical_signature())
            torch.testing.assert_close(reference.prediction, moved.prediction, rtol=0, atol=1e-6)
            with patch("coarse_gnn.cache.build_topology", side_effect=AssertionError("Cache hit rebuilt")):
                reused = model(x[permutation], inverse[edges], **moved_labels)
            torch.testing.assert_close(moved.prediction, reused.prediction, rtol=0, atol=0)
            with self.assertRaisesRegex(ValueError, "does not match"):
                model(x[permutation], inverse[edges], **moved_labels, topology=reference.topology)
        self.assertEqual(cache.misses, 21)
        self.assertEqual(cache.hits, 20)

    def test_corrupt_disk_cache_reports_actionable_error(self):
        with tempfile.TemporaryDirectory() as directory:
            edges, labels = chain_inputs()
            TopologyCache(directory).get_or_build(25, edges, **labels)
            path = next(Path(directory).glob("*.pt"))
            path.write_bytes(b"incomplete cache")
            with self.assertRaisesRegex(RuntimeError, "remove this file and precompute again"):
                TopologyCache(directory).get_or_build(25, edges, **labels)


if __name__ == "__main__":
    unittest.main()
