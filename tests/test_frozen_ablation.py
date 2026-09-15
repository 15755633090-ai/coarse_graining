"""Synthetic switch checks; no real encoder, Lipo data or training run is used."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from coarse_gnn import TopologyCache
from coarse_gnn.diffusion_adapter import precompute_batch_topologies
from diffusion.bond_diffusion.data import collate_graphs
from examples.run_coarse_demo import chain_graph
from frozen_features import FrozenGraphBatch, frozen_factory
from scripts.experiments.ablation_models import (
    TRAIN_VARIANTS, CHECK_VARIANTS, core_only_topology, make_factory,
)
from scripts.experiments.run_frozen_ablation import baseline_pending, summarize


class FakeEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_dim=16)
        self.weight = nn.Parameter(torch.ones(16))

    def encode_nodes(self, *args):
        raise AssertionError("An ablation forward must use cached features")


def fake_original_factory(*args):
    return SimpleNamespace(encoder=FakeEncoder())


class FrozenAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)

    def setUp(self):
        self.cache = TopologyCache()
        self.factory = make_factory(fake_original_factory, self.cache)
        self.graphs = [chain_graph(n) for n in (24, 31, 1)]
        graph = collate_graphs(self.graphs)
        topologies = precompute_batch_topologies(graph, self.cache)
        generator = torch.Generator().manual_seed(123)
        nodes = torch.randn(*graph.node_features.shape[:2], 16, generator=generator)
        self.batch = FrozenGraphBatch(graph, topologies, nodes)

    def model(self, mode, dropout=0):
        torch.manual_seed(7)
        return self.factory(mode, 1, Path('unused.pt'), 'cpu', dropout)

    def test_shared_initialization_rng_and_exact_parameter_matching(self):
        mother = self.model('base_coarse')
        mother_rng = torch.get_rng_state().clone()
        state = mother.predictor.state_dict()
        base_count = sum(p.numel() for p in mother.parameters() if p.requires_grad)
        for mode in TRAIN_VARIANTS + CHECK_VARIANTS:
            model = self.model(mode)
            self.assertTrue(torch.equal(torch.get_rng_state(), mother_rng))
            for key, value in model.predictor.state_dict().items():
                if key.startswith(('head.', 'input_projection.')):
                    torch.testing.assert_close(value, state[key], rtol=0, atol=0)
            if mode in ('no_edge', 'deeper_region', 'core_only'):
                self.assertEqual(base_count, sum(p.numel() for p in model.parameters() if p.requires_grad))
        deeper = self.model('deeper_region')
        self.assertEqual(len(deeper.predictor.region_gnn.layers), 5)
        self.assertEqual(len(deeper.predictor.coarse_gnn.layers), 0)
        for i in range(3):
            for key, value in mother.predictor.coarse_gnn.layers[i].state_dict().items():
                torch.testing.assert_close(deeper.predictor.region_gnn.layers[i + 2].state_dict()[key], value, rtol=0, atol=0)

    def test_c_and_d_reference_switches_reproduce_existing_implementation(self):
        old_factory = frozen_factory(fake_original_factory, self.cache)
        for mode in ('region_only', 'base_coarse'):
            current = self.model(mode, dropout=0.1).train()
            torch.manual_seed(7)
            original = old_factory(mode, 1, Path('unused.pt'), 'cpu', 0.1).train()
            for key, value in current.state_dict().items():
                torch.testing.assert_close(value, original.state_dict()[key], rtol=0, atol=0)
            torch.manual_seed(456)
            expected = original(self.batch)
            expected_rng = torch.get_rng_state().clone()
            torch.manual_seed(456)
            actual = current(self.batch)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertTrue(torch.equal(torch.get_rng_state(), expected_rng))

    def test_atom_and_weighted_pooling_predictions_and_gradients_agree(self):
        atom, weighted = self.model('atom'), self.model('direct_weighted')
        a, b = atom(self.batch), weighted(self.batch)
        torch.testing.assert_close(a, b, rtol=2e-5, atol=2e-6)
        a.square().sum().backward()
        b.square().sum().backward()
        for (name_a, p_a), (name_b, p_b) in zip(atom.predictor.named_parameters(), weighted.predictor.named_parameters()):
            self.assertEqual(name_a, name_b)
            torch.testing.assert_close(p_a.grad, p_b.grad, rtol=2e-4, atol=2e-6)
        self.assertNotIn('direct_weighted', TRAIN_VARIANTS)

    def test_double_precision_identity_and_mean_reweighting(self):
        values = torch.arange(20, dtype=torch.float64).reshape(5, 4)
        cores = (values[:1], values[1:])
        means = torch.stack([c.mean(0) for c in cores])
        weighted = (means * torch.tensor([1., 4.]).unsqueeze(1)).sum(0) / 5
        torch.testing.assert_close(values.mean(0), weighted, rtol=0, atol=0)
        self.assertFalse(torch.allclose(values.mean(0), means.mean(0)))

    def test_b_mean_is_mean_of_core_means(self):
        model = self.model('direct_mean').eval()
        expected = []
        for index, topology in enumerate(self.batch.topologies):
            nodes = self.batch.node_states[index, topology.atom_order]
            projected = model.predictor.input_projection(nodes)
            cores = torch.stack([projected[core].mean(0) for core in topology.cores])
            expected.append(model.predictor.head(cores.mean(0)))
        torch.testing.assert_close(model(self.batch), torch.stack(expected), rtol=2e-5, atol=2e-6)

    def test_no_edge_matches_serial_self_updates_and_preserves_input_plan(self):
        model = self.model('no_edge').eval()
        before_source = self.batch.plan.coarse_source.clone()
        actual = model(self.batch)
        expected = []
        for index, topology in enumerate(self.batch.topologies):
            nodes = self.batch.node_states[index, topology.atom_order]
            predictor = model.predictor
            projected = predictor.input_projection(nodes)
            regions = []
            for context, edges, positions in zip(topology.contexts, topology.context_edges, topology.core_positions):
                local = predictor.region_gnn(projected[context], edges, nodes.new_empty((edges.size(1), 0)))
                regions.append(local[positions].mean(0))
            regions = torch.stack(regions)
            coarse = predictor.coarse_gnn(regions, torch.empty(2, 0, dtype=torch.long), nodes.new_empty((0, 0)))
            expected.append(predictor.head(coarse.mean(0)))
        torch.testing.assert_close(actual, torch.stack(expected), rtol=2e-5, atol=2e-6)
        self.assertTrue(torch.equal(self.batch.plan.coarse_source, before_source))
        self.assertGreater(len(before_source), 0)
        self.assertEqual(len(model.predictor.coarse_gnn.layers), 3)

    def test_core_only_changes_only_contexts_and_region_edges(self):
        originals = self.batch.topologies
        transformed = [core_only_topology(t) for t in originals]
        for old, new in zip(originals, transformed):
            for name in ('owner', 'centers', 'coarse_edges', 'edge_counts', 'atom_order', 'is_residual'):
                self.assertIs(getattr(new, name), getattr(old, name))
            self.assertEqual(old.input_fingerprint, new.input_fingerprint)
            for core, context, edges, ids in zip(new.cores, new.contexts, new.context_edges, new.context_edge_ids):
                self.assertTrue(torch.equal(core, context))
                self.assertTrue(torch.equal(context[edges], old.edges[:, ids]))
            self.assertIsNot(old.contexts, new.contexts)
        self.assertTrue(any(len(context) > len(core) for t in originals for core, context in zip(t.cores, t.contexts)))
        model = self.model('core_only')
        with self.assertRaisesRegex(ValueError, 'CoreOnlyDataset'):
            model(self.batch)
        batch = FrozenGraphBatch(self.batch.graph, transformed, self.batch.node_states)
        self.assertTrue(torch.isfinite(model(batch)).all())

    def test_all_training_switches_keep_encoder_frozen_and_have_gradients(self):
        for mode in TRAIN_VARIANTS:
            model = self.model(mode, dropout=0.1).train()
            batch = self.batch
            if mode == 'core_only':
                batch = FrozenGraphBatch(batch.graph, [core_only_topology(t) for t in batch.topologies], batch.node_states)
            model(batch).square().mean().backward()
            self.assertFalse(model.encoder.training)
            self.assertTrue(all(not p.requires_grad and p.grad is None for p in model.encoder.parameters()))
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.head.parameters()))

    def test_gate_and_paired_summary_use_all_five_seeds(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            mother, output = root / 'mother', root / 'ablation'
            self.assertEqual(len(baseline_pending(mother)), 10)
            for mode, folder, offset in (('region_only', mother, 0.1), ('base_coarse', mother, 0.), ('no_edge', output, 0.05)):
                for seed in range(5):
                    path = folder / 'lipo' / mode / f'seed_{seed}' / 'result.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(dict(mode=mode, seed=seed, test_metrics={'rmse': .8 + seed * .01 + offset})))
            self.assertEqual(baseline_pending(mother), [])
            report = summarize(output, mother)
            self.assertFalse(report['complete'])
            paired = {p['question']: p for p in report['paired']}
            self.assertEqual(paired['C_to_D']['n'], 5)
            self.assertAlmostEqual(paired['C_to_D']['mean_delta_rmse'], -.1)
            self.assertAlmostEqual(paired['inter_region_messages']['mean_delta_rmse'], -.05)


if __name__ == '__main__':
    unittest.main()
