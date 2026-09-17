"""V2 method-definition tests (topology invariants and relation-only correction)."""
import copy
import unittest
from dataclasses import replace
from collections import deque

import torch

from coarse_gnn import (
    V2CoarseGraphPredictor,
    V2CoarseningConfig,
    V2NetworkConfig,
    build_v2_topology,
)
from coarse_gnn.diffusion_adapter import DiffusionCoarseModel
from diffusion.bond_diffusion.config import ModelConfig
from diffusion.bond_diffusion.data import MoleculeGraph, collate_graphs
from diffusion.bond_diffusion.model import BondAwareDiffusionModel


def chain_edges(n):
    return torch.stack((torch.arange(n - 1), torch.arange(1, n)))


def labels(n, edges):
    return dict(
        node_labels=torch.tensor([[6, 5, 0, 3, 2]]).repeat(n, 1),
        edge_labels=torch.ones(edges.size(1), dtype=torch.long),
    )


def induced_radius(edges, members, center):
    allowed = set(members)
    adjacency = {node: [] for node in allowed}
    for u, v in edges.T.tolist():
        if u in allowed and v in allowed:
            adjacency[u].append(v)
            adjacency[v].append(u)
    distance = {center: 0}
    queue = deque([center])
    while queue:
        node = queue.popleft()
        for neighbor in adjacency[node]:
            if neighbor not in distance:
                distance[neighbor] = distance[node] + 1
                queue.append(neighbor)
    if len(distance) != len(allowed):
        raise AssertionError("region is disconnected")
    return max(distance.values(), default=0)


def molecule(n, edges):
    bonds = torch.zeros(n, n, dtype=torch.long)
    bonds[edges[0], edges[1]] = 1
    bonds[edges[1], edges[0]] = 1
    features = torch.zeros(n, 5, dtype=torch.long)
    features[:, 0] = 6
    features[:, 1] = 5
    features[:, 3] = 3
    features[:, 4] = (bonds > 0).sum(1)
    return MoleculeGraph(features, bonds, torch.zeros_like(bonds, dtype=torch.bool))


class V2Tests(unittest.TestCase):
    def test_exclusive_connected_radius_bounded_partition(self):
        n = 100
        edges = chain_edges(n)
        topology = build_v2_topology(n, edges, V2CoarseningConfig(), **labels(n, edges))
        self.assertEqual(sorted(torch.cat(topology.cores).tolist()), list(range(n)))
        self.assertTrue(all(torch.equal(core, context) for core, context in zip(topology.cores, topology.contexts)))
        self.assertFalse(topology.is_residual.any())
        for center, core in zip(topology.centers.tolist(), topology.cores):
            self.assertLessEqual(induced_radius(topology.edges, core.tolist(), center), 4)
        self.assertEqual(topology.stats["max_region_radius"], 4)

    def test_each_disconnected_component_receives_a_center(self):
        edges = torch.tensor([[0, 1, 4, 5], [1, 2, 5, 6]])
        topology = build_v2_topology(8, edges, V2CoarseningConfig(), **labels(8, edges))
        self.assertEqual(topology.stats["num_components"], 4)
        self.assertEqual(topology.stats["initial_center_budget"], 4)
        self.assertEqual(len(topology.cores), 4)
        self.assertEqual(topology.coarse_edges.size(1), 0)

    def test_no_edge_full_model_is_exactly_region_only(self):
        torch.manual_seed(7)
        n = 20
        edges = chain_edges(n)
        raw = labels(n, edges)
        attributes = torch.nn.functional.one_hot(raw["edge_labels"] - 1, num_classes=4).float()
        full = V2CoarseGraphPredictor(V2NetworkConfig(input_dim=5, hidden_dim=8, dropout=0))
        region = copy.deepcopy(full)
        region.config = replace(region.config, coarse_layers=0)
        x = torch.randn(n, 5)
        full_output = full(x, edges, attributes, **raw)
        region_output = region(x, edges, attributes, **raw)
        self.assertGreater(full_output.topology.coarse_edges.size(1), 0)
        # Remove every inter-region relation while preserving identical Region weights.
        disconnected = copy.deepcopy(full_output.topology)
        disconnected.coarse_edges = torch.empty((2, 0), dtype=torch.long)
        disconnected.boundary_edge_ids = torch.empty(0, dtype=torch.long)
        disconnected.boundary_groups = torch.empty(0, dtype=torch.long)
        disconnected.edge_counts = torch.empty(0, dtype=torch.long)
        # A topology fingerprint is an input/topology identity; explicit mutation is
        # only used here to exercise the mathematical no-edge contract.
        no_edge_output = full(x, edges, attributes, topology=disconnected, **raw)
        self.assertTrue(torch.equal(
            no_edge_output.coarse_embeddings,
            no_edge_output.region_embeddings,
        ))
        torch.testing.assert_close(no_edge_output.prediction, region_output.prediction, rtol=0, atol=0)

    def test_permutation_invariance_for_chain_branch_and_symmetric_cycle(self):
        chain = chain_edges(25)
        branch = torch.tensor([
            [0, 1, 2, 3, 2, 5, 6, 2, 8, 9, 8, 11, 12, 8],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
        ])
        cycle = torch.cat((chain_edges(16), torch.tensor([[15], [0]])), dim=1)
        for name, n, edges in (("chain", 25, chain), ("branch", 15, branch), ("cycle", 16, cycle)):
            with self.subTest(graph=name):
                torch.manual_seed(101)
                encoder = BondAwareDiffusionModel(
                    ModelConfig(hidden_dim=16, num_layers=2, dropout=0, time_dim=8),
                )
                predictor = V2CoarseGraphPredictor(
                    V2NetworkConfig(input_dim=16, hidden_dim=16, dropout=0),
                )
                model = DiffusionCoarseModel(encoder, predictor, freeze_encoder=True).eval()
                graph = molecule(n, edges)
                reference = model(collate_graphs([graph]))
                for _ in range(30):
                    permutation = torch.randperm(n)
                    moved_graph = MoleculeGraph(
                        graph.node_features[permutation],
                        graph.bonds[permutation][:, permutation],
                        graph.substructure_mask[permutation][:, permutation],
                    )
                    moved = model(collate_graphs([moved_graph]))
                    self.assertEqual(
                        reference.graphs[0].topology.canonical_signature(),
                        moved.graphs[0].topology.canonical_signature(),
                    )
                    torch.testing.assert_close(
                        reference.predictions, moved.predictions, rtol=1e-6, atol=1e-6,
                    )

    def test_coarse_attributes_are_log_count_and_bond_proportions(self):
        torch.manual_seed(9)
        n = 20
        edges = chain_edges(n)
        raw = labels(n, edges)
        raw["edge_labels"][::3] = 2
        attributes = torch.nn.functional.one_hot(raw["edge_labels"] - 1, num_classes=4).float()
        model = V2CoarseGraphPredictor(V2NetworkConfig(input_dim=5, hidden_dim=8, dropout=0))
        output = model(torch.randn(n, 5), edges, attributes, **raw)
        self.assertEqual(output.coarse_edge_attr.size(1), 5)
        if output.coarse_edge_attr.numel():
            torch.testing.assert_close(output.coarse_edge_attr[:, 1:].sum(1), torch.ones(output.coarse_edge_attr.size(0)))
            torch.testing.assert_close(output.coarse_edge_attr[:, 0], output.topology.edge_counts.float().log1p())


if __name__ == "__main__":
    unittest.main()
