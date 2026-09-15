"""Disjoint packed graph, RNG and gradient equivalence against serial execution."""
import unittest
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import torch

from coarse_gnn import CoarseGraphPredictor, CoarseningConfig, NetworkConfig, TopologyCache
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from coarse_gnn.packed import packed_predict, prepare_graph_batch
from diffusion.bond_diffusion.config import ModelConfig
from diffusion.bond_diffusion.data import collate_graphs
from diffusion.bond_diffusion.model import BondAwareDiffusionModel
from examples.run_coarse_demo import chain_graph


class PackedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def compare(self, cfg, device="cpu", train=False):
        torch.manual_seed(81)
        encoder = BondAwareDiffusionModel(ModelConfig(hidden_dim=16, num_layers=1, time_dim=8, dropout=0.1))
        model = DiffusionCoarseModel(encoder, CoarseGraphPredictor(cfg, CoarseningConfig(radius=2), topology_cache=TopologyCache()),
                                    freeze_encoder=False).to(device)
        model.train(train)
        graphs = [chain_graph(1), chain_graph(24), chain_graph(11), chain_graph(6)]
        graphs[2].bonds[4, 5] = graphs[2].bonds[5, 4] = 0
        graphs[2].node_features[:, 4] = (graphs[2].bonds > 0).sum(1)
        graphs[3].bonds[0, 5] = graphs[3].bonds[5, 0] = 2
        graphs[3].node_features[:, 4] = (graphs[3].bonds > 0).sum(1)
        cpu = collate_graphs(graphs)
        prepared = prepare_graph_batch(cpu, model.predictor).to(device)
        raw = cpu.to(device)
        torch.manual_seed(31)
        serial = model(raw)
        rng_serial = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        serial.predictions.square().sum().backward()
        gradients = {key: p.grad.clone() for key, p in model.named_parameters() if p.grad is not None}
        model.zero_grad(set_to_none=True)
        torch.manual_seed(31)
        nodes = model.encoder.encode_nodes(prepared.node_features, prepared.bonds, prepared.node_mask)
        predicted, regions, coarse, graph = packed_predict(model.predictor, nodes, prepared, return_intermediates=True)
        rng_packed = torch.cuda.get_rng_state() if device == "cuda" else torch.get_rng_state()
        torch.testing.assert_close(predicted, serial.predictions, atol=2e-6, rtol=1e-5)
        torch.testing.assert_close(regions, torch.cat([out.region_embeddings for out in serial.graphs]), atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(coarse, torch.cat([out.coarse_embeddings for out in serial.graphs]), atol=3e-6, rtol=1e-5)
        torch.testing.assert_close(graph, serial.graph_embeddings, atol=3e-6, rtol=1e-5)
        self.assertTrue(torch.equal(rng_serial, rng_packed))
        predicted.square().sum().backward()
        actual = {key: p.grad for key, p in model.named_parameters() if p.grad is not None}
        self.assertEqual(gradients.keys(), actual.keys())
        for key in gradients:
            torch.testing.assert_close(actual[key], gradients[key], atol=1e-5, rtol=1e-4, msg=key)

    def test_forward_backward_pooling_edges_and_disconnected_graphs(self):
        for pool in ("mean", "sum", "size_weighted_mean"):
            for enhanced in (False, True):
                with self.subTest(pool=pool, enhanced=enhanced):
                    options = dict(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0,
                                   graph_pool=pool, region_pool="sum" if enhanced else "mean",
                                   edge_reduce="mean", coarse_layers=2 if enhanced else 0)
                    cfg = NetworkConfig(**options) if enhanced else NetworkConfig.base(**options)
                    self.compare(cfg)

    def test_training_dropout_preserves_cpu_rng_and_gradients(self):
        self.compare(NetworkConfig(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0.2), train=True)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_training_dropout_preserves_cuda_rng_and_gradients(self):
        torch.use_deterministic_algorithms(True)
        self.compare(NetworkConfig(input_dim=16, hidden_dim=16, edge_dim=4, dropout=0.2), device="cuda", train=True)


if __name__ == "__main__":
    unittest.main()
