"""Packed LocalBoundary execution must match an intentionally serial oracle."""
import copy
import unittest

import torch

from coarse_gnn import (
    LOCAL_BOUNDARY_VARIANTS,
    LocalBoundaryNetworkConfig,
    LocalBoundaryPredictor,
    build_hierarchical_topology,
    l1_edge_attributes,
    pack_local_boundary,
    split_local_boundary_component,
)


def chain_topology(count):
    edges = (
        torch.stack((torch.arange(count - 1), torch.arange(1, count)))
        if count > 1 else torch.empty((2, 0), dtype=torch.long)
    )
    nodes = torch.tensor([[6, 0, 0]], dtype=torch.long).repeat(count, 1)
    labels = torch.ones(edges.size(1), dtype=torch.long)
    return build_hierarchical_topology(count, edges, node_labels=nodes, edge_labels=labels)


def slow_reference(model, nodes, base_embeddings, base_predictions, topologies):
    graph_embeddings, corrections, all_l1 = [], [], []
    for graph_index, topology in enumerate(topologies):
        atoms = model.atom_adapter(nodes[graph_index, topology.atom_order])
        graph_rows, graph_counts = [], []
        for component_index, component in enumerate(topology.components):
            rows = []
            l1 = component.levels[0]
            for members in l1.members:
                selected = atoms[members]
                size = selected.new_tensor([members.numel()]).log1p()
                rows.append(model.l1_mlp(torch.cat((
                    selected.mean(0), selected.max(0).values, size,
                )).unsqueeze(0))[0])
            values = torch.stack(rows)
            split = split_local_boundary_component(component)
            full_attributes = l1_edge_attributes(topology, component_index).to(values)
            empty_edges, empty_attributes = l1.edges[:, :0], full_attributes[:0]
            if model.variant == "self_only":
                schedule = ((empty_edges, empty_attributes),) * 2
            elif model.variant == "local_only":
                schedule = (
                    (split.local_edges, full_attributes[split.local_edge_ids]),
                    (empty_edges, empty_attributes),
                )
            elif model.variant == "local_boundary":
                schedule = (
                    (split.local_edges, full_attributes[split.local_edge_ids]),
                    (split.boundary_edges, full_attributes[split.boundary_edge_ids]),
                )
            else:
                schedule = ((l1.edges, full_attributes),) * 2
            for gine, (edges, attributes) in zip(model.gines, schedule):
                values = gine(values, edges.to(values.device), attributes)
            graph_rows.append(values)
            graph_counts.append(l1.atom_counts.to(values.device))
            all_l1.append(values)
        values = torch.cat(graph_rows)
        counts = torch.cat(graph_counts).to(values.dtype).unsqueeze(1)
        mean = (values * counts).sum(0) / counts.sum()
        maximum = values.max(0).values
        size = values.new_tensor([topology.atom_order.numel()]).log1p()
        graph = model.graph_mlp(torch.cat((mean, maximum, size)).unsqueeze(0))[0]
        correction = model.correction_head(torch.cat((
            model.base_projection(base_embeddings[graph_index]), graph,
        )).unsqueeze(0))[0]
        graph_embeddings.append(graph)
        corrections.append(correction)
    correction = torch.stack(corrections)
    return (
        torch.cat(all_l1), torch.stack(graph_embeddings), correction,
        base_predictions + correction,
    )


class PackedLocalBoundaryTests(unittest.TestCase):
    def test_attributes_are_computed_once_then_sliced_by_edge_id(self):
        topology = chain_topology(120)
        plan = pack_local_boundary([topology], 120)
        component = topology.components[0]
        split = split_local_boundary_component(component)
        full = l1_edge_attributes(topology, 0)
        torch.testing.assert_close(plan.full_edge_attributes, full)
        torch.testing.assert_close(plan.local_edge_attributes, full[split.local_edge_ids])
        torch.testing.assert_close(plan.boundary_edge_attributes, full[split.boundary_edge_ids])
        self.assertEqual(
            plan.full_edges.size(1),
            plan.local_edges.size(1) + plan.boundary_edges.size(1),
        )

    def test_all_variants_match_slow_reference(self):
        torch.manual_seed(913)
        topologies = [chain_topology(6), chain_topology(40), chain_topology(120)]
        plan = pack_local_boundary(topologies, 120)
        nodes = torch.randn(3, 120, 8)
        base = torch.randn(3, 5)
        prediction = torch.randn(3, 1)
        initial = None
        for variant in LOCAL_BOUNDARY_VARIANTS:
            torch.manual_seed(77)
            model = LocalBoundaryPredictor(LocalBoundaryNetworkConfig(
                input_dim=8, base_dim=5, hidden_dim=16, dropout=0,
            ), variant).eval()
            state = {key: value.clone() for key, value in model.state_dict().items()}
            if initial is None:
                initial = state
            else:
                self.assertEqual(initial.keys(), state.keys())
                for key in initial:
                    torch.testing.assert_close(initial[key], state[key], rtol=0, atol=0)
            packed = model(nodes, base, prediction, plan)
            l1, graphs, corrections, predictions = slow_reference(
                model, nodes, base, prediction, topologies,
            )
            torch.testing.assert_close(packed.l1_embedding, l1, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(packed.graph_embedding, graphs, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(packed.correction, corrections, atol=1e-6, rtol=1e-6)
            torch.testing.assert_close(packed.prediction, predictions, atol=1e-6, rtol=1e-6)

    def test_packed_and_slow_gradients_match(self):
        torch.manual_seed(402)
        topologies = [chain_topology(40), chain_topology(100)]
        plan = pack_local_boundary(topologies, 100)
        config = LocalBoundaryNetworkConfig(input_dim=7, base_dim=6, hidden_dim=12, dropout=0)
        packed_model = LocalBoundaryPredictor(config, "local_boundary").double()
        slow_model = copy.deepcopy(packed_model)
        packed_nodes = torch.randn(2, 100, 7, dtype=torch.double, requires_grad=True)
        slow_nodes = packed_nodes.detach().clone().requires_grad_(True)
        packed_base = torch.randn(2, 6, dtype=torch.double, requires_grad=True)
        slow_base = packed_base.detach().clone().requires_grad_(True)
        prediction = torch.randn(2, 1, dtype=torch.double)
        packed_output = packed_model(packed_nodes, packed_base, prediction, plan).prediction
        slow_output = slow_reference(
            slow_model, slow_nodes, slow_base, prediction, topologies,
        )[-1]
        packed_output.square().sum().backward()
        slow_output.square().sum().backward()
        # Packed segment reductions intentionally accumulate in float32, so
        # double inputs still differ from the serial reductions at roundoff.
        torch.testing.assert_close(packed_output, slow_output, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(packed_nodes.grad, slow_nodes.grad, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(packed_base.grad, slow_base.grad, atol=1e-6, rtol=1e-6)
        for packed_parameter, slow_parameter in zip(packed_model.parameters(), slow_model.parameters()):
            torch.testing.assert_close(
                packed_parameter.grad, slow_parameter.grad, atol=1e-6, rtol=1e-6,
            )


if __name__ == "__main__":
    unittest.main()
