"""Contracts for packed hierarchical execution and efficient batching."""
import math
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


def _slow_edge_features(topology, previous, edge_ids):
    columns = [key[0] - 1 for key in topology.edge_type_keys]
    rows = []
    for edge_id in edge_ids:
        left, right = previous.edges[:, edge_id].tolist()
        count = int(previous.edge_counts[edge_id])
        proportions = [0.0] * 4
        for source, target in enumerate(columns):
            proportions[target] = float(previous.edge_type_counts[edge_id, source]) / count
        density = count / min(int(previous.atom_counts[left]), int(previous.atom_counts[right]))
        rows.append([math.log1p(count), *proportions, density])
    return rows


def _slow_parent_level(model, topology, component, children, target_level):
    level = component.levels[target_level - 1]
    previous = component.levels[target_level - 2]
    outputs = []
    boundary = torch.zeros(level.num_tokens, dtype=torch.long)
    if level.edges.numel():
        boundary.index_add_(0, level.edges[0], level.edge_counts)
        boundary.index_add_(0, level.edges[1], level.edge_counts)
    for parent, members in enumerate(level.members):
        lookup = {int(child): local for local, child in enumerate(members.tolist())}
        edge_ids, local_edges = [], []
        for edge_id, (left, right) in enumerate(previous.edges.T.tolist()):
            if int(level.owner[left]) == parent and int(level.owner[right]) == parent:
                edge_ids.append(edge_id)
                local_edges.append((lookup[left], lookup[right]))
        edges = (
            torch.tensor(local_edges, dtype=torch.long, device=children.device).T.contiguous()
            if local_edges else torch.empty((2, 0), dtype=torch.long, device=children.device)
        )
        attributes = children.new_tensor(_slow_edge_features(topology, previous, edge_ids)).reshape(-1, 6)
        encoded = model.gines[target_level - 2](children[members], edges, attributes)
        weights = previous.atom_counts[members].to(encoded.device, encoded.dtype).unsqueeze(1)
        mean = (encoded * weights).sum(0) / weights.sum()
        maximum = encoded.max(0).values
        structure = encoded.new_tensor([
            math.log1p(int(level.atom_counts[parent])),
            math.log1p(level.leaf_sets[parent].numel()),
            int(level.child_radii[parent]),
            int(level.child_diameters[parent]),
            math.log1p(int(level.internal_edge_counts[parent])),
            math.log1p(int(boundary[parent])),
        ])
        outputs.append(model.parent_mlps[target_level - 2](
            torch.cat((mean, maximum, structure)).unsqueeze(0),
        )[0])
    return torch.stack(outputs)


def slow_reference(model, nodes, base_embeddings, base_predictions, topologies):
    """Intentionally unbatched oracle: graph -> component -> parent -> query."""
    graph_records = []
    all_levels = [[], [], []]
    for graph_index, topology in enumerate(topologies):
        canonical_atoms = model.atom_adapter(nodes[graph_index, topology.atom_order])
        component_records = []
        for component in topology.components:
            l1_rows = []
            for members in component.levels[0].members:
                atoms = canonical_atoms[members]
                size = atoms.new_tensor([members.numel()]).log1p()
                l1_rows.append(model.l1_mlp(torch.cat((
                    atoms.mean(0), atoms.max(0).values, size,
                )).unsqueeze(0))[0])
            raw_levels = [torch.stack(l1_rows)]
            if len(component.levels) >= 2:
                raw_levels.append(_slow_parent_level(model, topology, component, raw_levels[-1], 2))
            if len(component.levels) >= 3:
                raw_levels.append(_slow_parent_level(model, topology, component, raw_levels[-1], 3))
            projected = [
                model.level_projections[index](value)
                for index, value in enumerate(raw_levels)
            ]
            component_records.append((component, projected))

        updated_components = []
        for component, projected in component_records:
            updated_queries = []
            diameter = max(1, int(component.l1_distances.max()))
            for query_index, query in enumerate(projected[0]):
                context = component.adaptive_context(
                    query_index, topology.config.expansion_threshold,
                )
                if not context:
                    updated_queries.append(query)
                    continue
                scores, values = [], []
                for token in context:
                    selected = projected[token.level - 1][token.index]
                    score = (
                        model.query_projection(query) * model.key_projection(selected)
                    ).sum() / math.sqrt(model.config.hidden_dim)
                    distance = query.new_tensor([
                        token.d_min, token.d_max, token.d_mean,
                        token.d_min / diameter, token.d_max / diameter, token.d_mean / diameter,
                    ])
                    score = score + model.distance_mlp(distance).squeeze(0)
                    score = score + model.level_bias.weight[token.level - 1, 0]
                    leaves = component.levels[token.level - 1].leaf_sets[token.index].numel()
                    atoms = int(component.levels[token.level - 1].atom_counts[token.index])
                    score = score + model.beta_leaf * query.new_tensor(float(leaves)).log()
                    score = score + model.beta_atom * query.new_tensor(float(atoms)).log1p()
                    scores.append(score)
                    values.append(model.value_projection(selected))
                weights = torch.softmax(torch.stack(scores), dim=0)
                summary = (weights.unsqueeze(1) * torch.stack(values)).sum(0)
                gate = torch.sigmoid(model.gate(torch.cat((query, summary))))
                updated_queries.append(query + gate * model.context_projection(summary))
            updated = torch.stack(updated_queries)
            updated_components.append(updated)
            all_levels[0].append(updated)
            for level_index in range(1, len(projected)):
                all_levels[level_index].append(projected[level_index])

        graph_l1 = torch.cat(updated_components)
        graph_weights = torch.cat([
            component.levels[0].atom_counts for component, _ in component_records
        ]).to(graph_l1.device, graph_l1.dtype).unsqueeze(1)
        graph_mean = (graph_l1 * graph_weights).sum(0) / graph_weights.sum()
        graph_max = graph_l1.max(0).values
        graph_size = graph_l1.new_tensor([topology.atom_order.numel()]).log1p()
        graph = model.graph_mlp(torch.cat((graph_mean, graph_max, graph_size)).unsqueeze(0))[0]
        correction_inputs = [model.base_projection(base_embeddings[graph_index]), graph]
        if model.config.include_base_prediction:
            correction_inputs.append(base_predictions[graph_index])
        correction = model.correction_head(torch.cat(correction_inputs).unsqueeze(0))[0]
        graph_records.append((graph, correction))

    levels = tuple(
        torch.cat(rows) if rows else nodes.new_empty((0, model.config.hidden_dim))
        for rows in all_levels
    )
    graphs = torch.stack([row[0] for row in graph_records])
    corrections = torch.stack([row[1] for row in graph_records])
    return levels, graphs, corrections


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
        moved = plan.to("cpu")
        torch.testing.assert_close(moved.atom_gather, plan.atom_gather)
        torch.testing.assert_close(moved.l2.child_gather, plan.l2.child_gather)
        torch.testing.assert_close(
            moved.contexts.context_query_indices, plan.contexts.context_query_indices,
        )

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

    def test_packed_matches_slow_graph_parent_and_query_reference(self):
        torch.manual_seed(812)
        topologies = [chain_topology(1), chain_topology(30), chain_topology(120)]
        plan = pack_hierarchy(topologies, padded_nodes=120)
        model = HierarchicalPredictor(HierarchyNetworkConfig(
            input_dim=8, base_dim=5, hidden_dim=16, dropout=0,
            include_base_prediction=True,
        )).eval()
        nodes = torch.randn(3, 120, 8)
        base = torch.randn(3, 5)
        prediction = torch.randn(3, 1)
        packed = model(nodes, base, prediction, plan)
        levels, graphs, corrections = slow_reference(
            model, nodes, base, prediction, topologies,
        )
        for packed_level, reference_level in zip(packed.level_embeddings, levels):
            torch.testing.assert_close(packed_level, reference_level, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(packed.graph_embedding, graphs, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(packed.correction, corrections, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(
            packed.prediction, prediction + corrections, atol=1e-6, rtol=1e-6,
        )

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available")
    def test_cuda_bfloat16_plan_transfer_and_backward(self):
        topologies = [chain_topology(1), chain_topology(30)]
        plan = pack_hierarchy(topologies, padded_nodes=30).to("cuda")
        model = HierarchicalPredictor(HierarchyNetworkConfig(
            input_dim=8, base_dim=5, hidden_dim=16, dropout=0,
        )).cuda()
        nodes = torch.randn(2, 30, 8, device="cuda", requires_grad=True)
        base = torch.randn(2, 5, device="cuda", requires_grad=True)
        prediction = torch.randn(2, 1, device="cuda")
        with torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(nodes, base, prediction, plan)
            loss = output.prediction.float().square().mean()
        loss.backward()
        self.assertTrue(torch.isfinite(output.prediction).all())
        self.assertTrue(torch.isfinite(nodes.grad).all())
        self.assertTrue(torch.isfinite(base.grad).all())

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
