"""Topology-only tests for the near-fine/far-coarse hierarchy."""
import unittest
import tempfile

import torch

from coarse_gnn import HierarchyCache, build_hierarchical_topology, pack_hierarchy_contexts


def chain_edges(count):
    if count == 1:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((torch.arange(count - 1), torch.arange(1, count)))


def attributed(count, edges):
    nodes = torch.tensor([[6, 0, 0]]).repeat(count, 1)
    labels = torch.ones(edges.size(1), dtype=torch.long)
    return nodes, labels


class HierarchyTests(unittest.TestCase):
    def build_chain(self, count):
        edges = chain_edges(count)
        nodes, labels = attributed(count, edges)
        return build_hierarchical_topology(
            count, edges, node_labels=nodes, edge_labels=labels,
        )

    def test_chain_builds_compact_recursive_levels(self):
        topology = self.build_chain(120)
        component = topology.components[0]
        self.assertEqual(len(component.levels), 3)
        counts = [level.num_tokens for level in component.levels]
        self.assertGreater(counts[0], counts[1])
        self.assertGreater(counts[1], counts[2])
        self.assertTrue(all(level.child_diameters.max().item() <= 4 for level in component.levels[:1]))
        for level in component.levels:
            self.assertEqual(int(level.atom_counts.sum()), 120)
            self.assertEqual(
                int(level.internal_edge_counts.sum() + level.edge_counts.sum()), 119,
            )

    def test_adaptive_context_is_exact_and_near_neighbors_are_l1(self):
        component = self.build_chain(80).components[0]
        for query in range(component.levels[0].num_tokens):
            context = component.adaptive_context(query)
            leaves = []
            for token in context:
                token_leaves = component.levels[token.level - 1].leaf_sets[token.index]
                leaves.extend(token_leaves.tolist())
                if token.d_min <= 1:
                    self.assertEqual(token.level, 1)
            expected = [i for i in range(component.levels[0].num_tokens) if i != query]
            self.assertEqual(sorted(leaves), expected)
            self.assertEqual(len(leaves), len(set(leaves)))
        diagnostics = self.build_chain(80).diagnostics()
        self.assertTrue(diagnostics["built_l2"])
        self.assertTrue(diagnostics["used_l2"] or diagnostics["used_l3"])

    def test_small_and_empty_context_contracts(self):
        one = self.build_chain(1).components[0]
        self.assertEqual(one.adaptive_context(0), [])
        small = self.build_chain(6).components[0]
        self.assertGreaterEqual(small.levels[0].num_tokens, 2)
        self.assertEqual(
            sum(small.levels[token.level - 1].leaf_sets[token.index].numel()
                for token in small.adaptive_context(0)),
            small.levels[0].num_tokens - 1,
        )
        self.assertFalse(self.build_chain(6).diagnostics()["built_l2"])

    def test_disconnected_components_have_independent_hierarchies(self):
        left = chain_edges(25)
        right = chain_edges(20) + 25
        edges = torch.cat((left, right), dim=1)
        nodes, labels = attributed(45, edges)
        topology = build_hierarchical_topology(45, edges, node_labels=nodes, edge_labels=labels)
        self.assertEqual(len(topology.components), 2)
        self.assertEqual(sorted(component.atom_indices.numel() for component in topology.components), [20, 25])
        for component in topology.components:
            for query in range(component.levels[0].num_tokens):
                covered = sum(
                    component.levels[token.level - 1].leaf_sets[token.index].numel()
                    for token in component.adaptive_context(query)
                )
                self.assertEqual(covered, component.levels[0].num_tokens - 1)

    def test_recursive_raw_edge_statistics_are_conserved(self):
        count = 70
        edges = chain_edges(count)
        nodes, labels = attributed(count, edges)
        labels[::3] = 2
        topology = build_hierarchical_topology(
            count, edges, node_labels=nodes, edge_labels=labels,
        )
        component = topology.components[0]
        expected_types = torch.bincount(labels - 1, minlength=2)
        for level in component.levels:
            self.assertEqual(int(level.edge_counts.sum() + level.internal_edge_counts.sum()), edges.size(1))
            actual_types = level.edge_type_counts.sum(0) + level.internal_edge_type_counts.sum(0)
            torch.testing.assert_close(actual_types, expected_types)

    def test_random_reindexing_preserves_complete_hierarchy_signature(self):
        count = 36
        chain = chain_edges(count)
        branches = torch.tensor([[8, 18, 28], [35, 34, 33]])
        edges = torch.cat((chain, branches), dim=1)
        nodes, labels = attributed(count, edges)
        labels[-3:] = torch.tensor([2, 3, 2])
        reference = build_hierarchical_topology(
            count, edges, node_labels=nodes, edge_labels=labels,
        ).canonical_signature()
        generator = torch.Generator().manual_seed(411)
        for _ in range(20):
            permutation = torch.randperm(count, generator=generator)
            inverse = torch.argsort(permutation)
            moved = build_hierarchical_topology(
                count, inverse[edges], node_labels=nodes[permutation], edge_labels=labels,
            )
            self.assertEqual(reference, moved.canonical_signature())

    def test_highly_symmetric_graphs_are_reindexing_stable(self):
        cycle_nodes = 18
        cycle = torch.cat((chain_edges(cycle_nodes), torch.tensor([[cycle_nodes - 1], [0]])), dim=1)
        leaves = 16
        star = torch.stack((torch.zeros(leaves, dtype=torch.long), torch.arange(1, leaves + 1)))
        first = torch.cat((chain_edges(10), torch.tensor([[9], [0]])), dim=1)
        twins = torch.cat((first, first + 10), dim=1)
        for count, edges in ((cycle_nodes, cycle), (leaves + 1, star), (20, twins)):
            with self.subTest(nodes=count):
                nodes, labels = attributed(count, edges)
                reference = build_hierarchical_topology(
                    count, edges, node_labels=nodes, edge_labels=labels,
                ).canonical_signature()
                generator = torch.Generator().manual_seed(617 + count)
                for _ in range(20):
                    permutation = torch.randperm(count, generator=generator)
                    inverse = torch.argsort(permutation)
                    actual = build_hierarchical_topology(
                        count, inverse[edges], node_labels=nodes[permutation], edge_labels=labels,
                    ).canonical_signature()
                    self.assertEqual(reference, actual)

    def test_bidirectional_storage_does_not_double_count_edges(self):
        count = 30
        edges = chain_edges(count)
        nodes, labels = attributed(count, edges)
        both = torch.cat((edges, edges.flip(0)), dim=1)
        both_labels = labels.repeat(2)
        topology = build_hierarchical_topology(
            count, both, node_labels=nodes, edge_labels=both_labels,
        )
        for level in topology.components[0].levels:
            self.assertEqual(int(level.edge_counts.sum() + level.internal_edge_counts.sum()), count - 1)

    def test_contexts_pack_without_per_graph_runtime_traversal(self):
        topologies = [self.build_chain(1), self.build_chain(30), self.build_chain(80)]
        plan = pack_hierarchy_contexts(topologies)
        self.assertEqual(plan.query_offsets.numel(), plan.query_l1_indices.numel() + 1)
        self.assertEqual(int(plan.query_offsets[-1]), plan.context_indices.numel())
        self.assertEqual(plan.context_levels.shape, plan.context_indices.shape)
        self.assertEqual(plan.context_query_indices.shape, plan.context_indices.shape)
        self.assertEqual(plan.d_min.shape, plan.context_indices.shape)
        self.assertEqual(plan.d_max.shape, plan.context_indices.shape)
        self.assertEqual(plan.d_mean.shape, plan.context_indices.shape)
        self.assertEqual(plan.rho.shape, plan.context_indices.shape)
        self.assertEqual(plan.leaf_counts.shape, plan.context_indices.shape)
        self.assertEqual(plan.atom_counts.shape, plan.context_indices.shape)
        self.assertEqual(plan.query_diameters.shape, plan.query_l1_indices.shape)
        moved = plan.to("cpu")
        for name in plan.__dataclass_fields__:
            torch.testing.assert_close(getattr(plan, name), getattr(moved, name))
        self.assertEqual(plan.graph_query_lengths.numel(), len(topologies))
        self.assertTrue(bool((plan.context_indices >= 0).all()))
        for query in range(plan.query_l1_indices.numel()):
            start, stop = plan.query_offsets[query:query + 2]
            self.assertTrue(bool((plan.context_query_indices[start:stop] == query).all()))
        for level in (1, 2, 3):
            selected = plan.context_indices[plan.context_levels == level]
            if selected.numel():
                self.assertLess(int(selected.max()), int(plan.level_token_counts[level - 1]))

    def test_hierarchy_disk_cache_is_batch_order_independent(self):
        inputs = []
        for count in (20, 35):
            edges = chain_edges(count)
            nodes, labels = attributed(count, edges)
            inputs.append((count, edges, None, nodes, labels))
        with tempfile.TemporaryDirectory() as directory:
            first = HierarchyCache(directory, max_memory_entries=0)
            expected = first.get_or_build_many(inputs)
            second = HierarchyCache(directory)
            actual = second.get_or_build_many(reversed(inputs))
            self.assertEqual(
                [value.canonical_signature() for value in expected],
                [value.canonical_signature() for value in reversed(actual)],
            )
            self.assertEqual(second.disk_hits, 2)
            self.assertTrue(all(
                topology.components[0]._context_cache for topology in actual
            ))


if __name__ == "__main__":
    unittest.main()
