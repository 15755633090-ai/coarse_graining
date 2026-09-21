"""Topology-only contracts for LocalBoundary-GINE edge schedules."""
import unittest

import torch

from coarse_gnn import (
    build_hierarchical_topology,
    local_boundary_reachability,
    split_local_boundary_component,
)


def chain_edges(count):
    if count < 2:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.stack((torch.arange(count - 1), torch.arange(1, count)))


def build_chain(count):
    edges = chain_edges(count)
    nodes = torch.tensor([[6, 0, 0]]).repeat(count, 1)
    labels = torch.ones(edges.size(1), dtype=torch.long)
    return build_hierarchical_topology(
        count, edges, node_labels=nodes, edge_labels=labels,
    )


class LocalBoundaryTests(unittest.TestCase):
    def test_cached_parent_membership_partitions_every_l1_edge(self):
        component = build_chain(120).components[0]
        self.assertGreaterEqual(len(component.levels), 2)
        plan = split_local_boundary_component(component)
        edge_count = component.levels[0].edges.size(1)
        self.assertEqual(
            plan.local_edge_ids.numel() + plan.boundary_edge_ids.numel(), edge_count,
        )
        self.assertFalse(bool(torch.isin(
            plan.local_edge_ids, plan.boundary_edge_ids,
        ).any()))
        flattened = torch.cat(plan.window_members)
        self.assertEqual(
            torch.sort(flattened).values.tolist(),
            list(range(component.levels[0].num_tokens)),
        )
        local, local_boundary, added = local_boundary_reachability(component, plan)
        self.assertTrue(bool((local_boundary | ~local).all()))
        self.assertTrue(bool(added.any()))

    def test_component_without_l2_has_one_window_and_no_boundary_edges(self):
        component = build_chain(6).components[0]
        self.assertEqual(len(component.levels), 1)
        plan = split_local_boundary_component(component)
        self.assertEqual(len(plan.window_members), 1)
        self.assertEqual(plan.boundary_edge_ids.numel(), 0)
        self.assertEqual(
            plan.local_edge_ids.numel(), component.levels[0].edges.size(1),
        )

    def test_misaligned_cached_parent_membership_is_rejected(self):
        component = build_chain(120).components[0]
        component.levels[1].owner = component.levels[1].owner[:-1]
        with self.assertRaisesRegex(ValueError, "one-to-one"):
            split_local_boundary_component(component)


if __name__ == "__main__":
    unittest.main()
