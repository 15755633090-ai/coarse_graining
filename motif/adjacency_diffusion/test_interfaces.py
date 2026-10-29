"""Adjacency-specific leakage, equation, alignment and objective checks."""
import ast
from dataclasses import replace
from pathlib import Path
import typing
import unittest
from unittest.mock import patch

import torch
from rdkit import Chem
from torch_geometric.data import Batch

from .data import Condition, atom_features, graph_from_smiles, condition_from_batch, adjacency_target
from .diffusion import Schedule
from .metrics import recovery_statistics, summarize
from .model import AdjacencyModel, Config


class AdjacencyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        torch.manual_seed(29)
        self.graph = graph_from_smiles("CCOC(=O)c1ccccc1")
        self.batch = Batch.from_data_list([self.graph])
        self.model = AdjacencyModel(Config(hidden_dim=12, layers=2, dropout=0)).eval()
        self.condition = condition_from_batch(self.batch)
        self.target = adjacency_target(self.batch, self.condition)
        self.t = torch.tensor([1000])
        self.noisy, self.noise = self.model.schedule.sample(self.target, self.t, self.condition.pairs)

    def test_feature_definitions_match_pinned_motil_featurizer(self):
        source = Path(__file__).resolve().parents[2] / 'coarse_graining1/MotiL_local/scientific_audit_20261012/official_source/chemprop/features/featurization.py'
        if not source.exists():
            self.skipTest('Pinned official source not provided in this checkout')
        tree = ast.parse(source.read_text(encoding='utf-8-sig'))
        nodes = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in ('onek_encoding_unk','atom_features')) or
                 (isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id in ('MAX_ATOMIC_NUM','ATOM_FEATURES') for t in n.targets))]
        namespace = {'Chem': Chem, 'List': typing.List, 'Union': typing.Union}
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), namespace)
        for smiles in ('CCO', 'N[C@@H](C)C(=O)O', '[Na+].[Cl-]', 'c1ccncc1', '[Fe+3]'):
            for atom in Chem.MolFromSmiles(smiles).GetAtoms():
                self.assertEqual(atom_features(atom), namespace['atom_features'](atom))
                self.assertEqual(len(atom_features(atom)), 133)

    def test_all_pairs_include_nonbonds_and_noise_is_symmetric(self):
        n = self.graph.num_nodes
        self.assertEqual(int(self.condition.pairs.sum()), n*(n-1))
        self.assertGreater(int(self.condition.pairs.sum()), int(self.target.sum()))
        torch.testing.assert_close(self.noisy, self.noisy.transpose(1,2), rtol=0, atol=0)
        self.assertTrue((self.noise[~self.condition.pairs] == 0).all())
        nonbond = self.condition.pairs & (self.target == 0)
        self.assertGreater(float(self.noisy[nonbond].abs().sum()), 0)

    def test_clean_target_and_bond_data_do_not_enter_condition(self):
        other = self.batch.clone()
        other.edge_index = torch.empty(2, 0, dtype=torch.long)
        other.edge_attr = torch.randn(17, 8)
        other.boundary = torch.ones(self.batch.num_nodes, dtype=torch.bool)
        other.fragment_id = torch.randint(0, 100, other.fragment_id.shape)
        changed = condition_from_batch(other, motif_prior=False)
        self.assertEqual(set(changed.__dataclass_fields__), {'x', 'nodes', 'pairs', 'same_fragment'})
        expected = self.model.predict_noise(self.condition, self.noisy, self.t)
        actual = self.model.predict_noise(changed, self.noisy, self.t)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        with self.assertRaises(TypeError):
            self.model.encoder(self.batch, self.noisy, self.t)

    def test_noisy_adjacency_and_time_have_output_and_gradient_paths(self):
        noisy = self.noisy.clone().requires_grad_()
        predicted = self.model.predict_noise(self.condition, noisy, self.t)
        gradient = torch.autograd.grad(predicted.square().sum(), noisy)[0]
        self.assertGreater(float(gradient[self.condition.pairs].abs().sum()), 0)
        changed = self.model.predict_noise(self.condition, -noisy.detach(), self.t)
        self.assertFalse(torch.allclose(predicted, changed))
        changed_time = self.model.predict_noise(self.condition, noisy.detach(), torch.tensor([1]))
        self.assertFalse(torch.allclose(predicted, changed_time))

    def test_reconstruction_formula_and_frobenius_noise_weight(self):
        schedule = Schedule()
        target = self.target.double()
        steps = torch.tensor([1000])
        noisy, noise = schedule.sample(target, steps, self.condition.pairs, torch.Generator().manual_seed(2))
        clean = schedule.reconstruct(noisy, noise, steps, self.condition.pairs)
        torch.testing.assert_close(clean, target, atol=1e-12, rtol=1e-12)
        perturbation = .1 * self.condition.pairs.to(target.dtype)
        wrong = schedule.reconstruct(noisy, noise+perturbation, steps, self.condition.pairs)
        a = schedule.coefficient(target, steps)
        expected = (((1-a)/a) * perturbation.square()).sum()
        torch.testing.assert_close((wrong-target).square().sum(), expected, rtol=1e-12, atol=1e-10)
        self.assertGreater(float(expected), float(perturbation.square().sum()) * 1000)

    def test_loss_is_clean_adjacency_frobenius_not_uniform_noise_mse(self):
        prediction = self.model.predict_noise(self.condition, self.noisy, self.t)
        restored = self.model.schedule.reconstruct(self.noisy, prediction, self.t, self.condition.pairs)
        expected = (restored-self.target).square().sum((1,2)).mean()
        with patch.object(self.model.schedule, 'sample', return_value=(self.noisy, self.noise)):
            loss, _ = self.model.reconstruction_loss(self.batch)
        torch.testing.assert_close(loss, expected)

    def test_node_permutation_and_fragment_id_relabeling(self):
        model = AdjacencyModel(Config(hidden_dim=12,layers=2,dropout=0,motif_prior=True,conditional_gate=True)).eval()
        condition = condition_from_batch(self.batch, True)
        p = torch.randperm(self.graph.num_nodes)
        moved = Condition(condition.x[:,p], condition.nodes[:,p], condition.pairs[:,p][:,:,p],
                          condition.same_fragment[:,p][:,:,p])
        expected = model.predict_noise(condition, self.noisy, self.t)
        actual = model.predict_noise(moved, self.noisy[:,p][:,:,p], self.t)
        torch.testing.assert_close(actual, expected[:,p][:,:,p], atol=1e-6, rtol=1e-5)
        relabeled = self.batch.clone()
        relabeled.fragment_id = self.batch.fragment_id * 17 + 301
        new = condition_from_batch(relabeled, True)
        torch.testing.assert_close(new.same_fragment, condition.same_fragment)

    def test_internal_isolation_and_candidate_gate_semantics(self):
        model = AdjacencyModel(Config(hidden_dim=12,layers=2,dropout=0,motif_prior=True,conditional_gate=True)).eval()
        condition = condition_from_batch(self.batch, True)
        _, first = model.predict_noise(condition, self.noisy, self.t, True)
        chosen = self.batch.fragment_id == 0
        x = condition.x.clone()
        x[:,~chosen] += torch.randn_like(x[:,~chosen])*5
        _, second = model.predict_noise(replace(condition, x=x), self.noisy, self.t, True)
        torch.testing.assert_close(first['internal'][:,chosen], second['internal'][:,chosen], atol=0, rtol=0)
        self.assertFalse(torch.allclose(first['context'][:,chosen], second['context'][:,chosen]))
        cross = condition.pairs & ~condition.same_fragment
        self.assertTrue((cross & (self.target == 0)).any())
        for gate in first['gates']:
            self.assertTrue(((gate[cross] > 0) & (gate[cross] < 1)).all())
            torch.testing.assert_close(gate[condition.same_fragment], torch.ones_like(gate[condition.same_fragment]))

    def test_batch_padding_singletons_and_clean_prediction_no_sampling(self):
        singleton = graph_from_smiles('C')
        combined = Batch.from_data_list([self.graph, singleton])
        condition = condition_from_batch(combined)
        noisy = torch.zeros_like(condition.pairs, dtype=torch.float32)
        changed = noisy.clone().masked_fill(~condition.pairs, 1e4)
        t = torch.tensor([1000,1000])
        torch.testing.assert_close(self.model.predict_noise(condition,noisy,t), self.model.predict_noise(condition,changed,t))
        with patch('torch.randn', side_effect=AssertionError('sampling')), patch('torch.randint', side_effect=AssertionError('sampling')):
            actual = self.model(combined)
            separate = torch.cat([self.model(Batch.from_data_list([g])) for g in (self.graph,singleton)])
        torch.testing.assert_close(actual,separate,atol=1e-6,rtol=1e-5)
        loss, statistics = self.model.reconstruction_loss(Batch.from_data_list([singleton]))
        self.assertEqual(float(loss.detach()),0)
        self.assertIsNone(summarize(statistics)['all']['pair_mse'])

    def test_zero_adjacency_baseline_does_not_hide_missing_bonds(self):
        stats = recovery_statistics(self.target, torch.zeros_like(self.target), self.noise, self.noise,
                                    self.condition,self.t,1000)
        result = summarize(stats)['all']
        self.assertEqual(result['positive_recall'],0)
        self.assertEqual(result['balanced_accuracy'],.5)
        self.assertGreater(result['reconstruction_frobenius'],0)
        self.assertAlmostEqual(result['reconstruction_frobenius'],result['zero_adjacency_frobenius'])

    def test_gate_gradients_in_pretrain_and_finetune(self):
        model = AdjacencyModel(Config(hidden_dim=12,layers=2,dropout=0,motif_prior=True,conditional_gate=True))
        model.set_stage('pretrain')
        loss,_=model.reconstruction_loss(self.batch,torch.Generator().manual_seed(8))
        loss.backward()
        self.assertGreater(float(model.encoder.layers[0].gate[0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.property_head[0].weight.grad)
        model.zero_grad(set_to_none=True)
        model.set_stage('finetune')
        model(self.batch).square().sum().backward()
        self.assertGreater(float(model.encoder.layers[0].gate[0].weight.grad.abs().sum()),0)
        self.assertIsNone(model.noise_head[0].weight.grad)
        uniform=AdjacencyModel(Config(hidden_dim=12,layers=2,dropout=0,motif_prior=True)).eval()
        _, states = uniform(self.batch,True)
        condition=condition_from_batch(self.batch,True)
        for gate in states['gates']:
            torch.testing.assert_close(gate[condition.pairs],torch.ones_like(gate[condition.pairs]))


if __name__ == '__main__':
    unittest.main()
