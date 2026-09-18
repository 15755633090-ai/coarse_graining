"""Contracts for packed hierarchical execution and efficient batching."""
import unittest

import torch

from coarse_gnn import (
    AtomCountBucketBatchSampler,
    HierarchicalPredictor,
    HierarchyNetworkConfig,
    build_hierarchical_topology,
    pack_hierarchy,
)


def chain_topology(count):
    edges = (
        torch.stack((torch.arange(count - 1), torch.arange(1, count)))
        if count > 1 else torch.empty((2, 0), dtype=torch.long)
    )
    nodes = torch.tensor([[6, 0, 0]], dtype=torch.long).repeat(count, 1)
    bonds = torch.ones(edges.size(1), dtype=torch.long)
    return build_hierarchical_topology(count, edges, node_labels=nodes, edge_labels=bonds)


class PackedHierarchyTests(unittest.TestCase):
    def test_plan_covers_atoms_and_aligns_recursive_levels(self):
        topologies = [chain_topology(1), chain_topology(30), chain_topology(80)]
        plan = pack_hierarchy(topologies, padded_nodes=80)
        self.assertEqual(plan.atom_gather.numel(), 111)
        self.assertEqual(int(plan.l1_atom_lengths.sum()), 111)
        self.assertEqual(plan.graph_l1_lengths.tolist(), plan.contexts.graph_query_lengths.tolist())
        self.assertEqual(plan.l2.child_gather.numel(), int(plan.l2.parent_lengths.sum()))
        self.assertEqual(plan.l3.child_gather.numel(), int(plan.l3.parent_lengths.sum()))
        self.assertEqual(plan.l2.edge_attributes.shape[1], 6)
        self.assertEqual(plan.l3.structure_features.shape[1], 6)

    def test_full_l1_is_same_execution_with_different_context_plan(self):
        plan = pack_hierarchy([chain_topology(40)], padded_nodes=40, full_l1=True)
        self.assertTrue(bool((plan.contexts.context_levels == 1).all()))
        count = int(plan.graph_l1_lengths[0])
        self.assertEqual(plan.contexts.context_indices.numel(), count * (count - 1))

    def test_vectorized_model_handles_mixed_sizes_and_gradients(self):
        topologies = [chain_topology(1), chain_topology(30), chain_topology(80)]
        plan = pack_hierarchy(topologies, padded_nodes=80)
        model = HierarchicalPredictor(HierarchyNetworkConfig(
            input_dim=8, base_dim=5, hidden_dim=16, dropout=0,
        ))
        nodes = torch.randn(3, 80, 8, requires_grad=True)
        base = torch.randn(3, 5, requires_grad=True)
        prediction = torch.randn(3, 1)
        output = model(nodes, base, prediction, plan)
        self.assertEqual(output.prediction.shape, (3, 1))
        self.assertEqual(output.graph_embedding.shape, (3, 16))
        self.assertTrue(torch.isfinite(output.prediction).all())
        output.prediction.square().sum().backward()
        self.assertGreater(nodes.grad.abs().sum().item(), 0)
        self.assertGreater(base.grad.abs().sum().item(), 0)

    def test_atom_count_bucket_sampler_is_complete_and_epoch_deterministic(self):
        counts = [1, 80, 4, 60, 7, 40, 10, 20, 30]
        sampler = AtomCountBucketBatchSampler(counts, 3, seed=17, bucket_size_multiplier=2)
        first = list(sampler)
        self.assertEqual(sorted(index for batch in first for index in batch), list(range(len(counts))))
        self.assertEqual(first, list(sampler))
        sampler.set_epoch(1)
        self.assertNotEqual(first, list(sampler))


if __name__ == "__main__":
    unittest.main()
