import unittest
from pathlib import Path

import torch

from bond_diffusion.config import DiffusionConfig
from bond_diffusion.data import MoleculeGraph, collate_graphs, graph_from_smiles
from bond_diffusion.diffusion import DiscreteGraphDiffusion
from bond_diffusion.losses import ChemicalDiffusionLoss
from bond_diffusion.trainer import load_encoder


class EncoderAudit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(__file__).resolve().parents[1]
        cls.checkpoint = cls.root / 'outputs/ogb_clean/best.pt'
        cls.model = load_encoder(cls.checkpoint)

    def setUp(self):
        torch.manual_seed(73)
        self.model.eval()
        self.model.zero_grad(set_to_none=True)
        self.graphs = [graph_from_smiles(s) for s in ['CCO', 'CC(=O)O', 'c1ccccc1', 'CCN']]
        self.batch = collate_graphs(self.graphs)

    def encode(self, batch):
        return self.model.encode_nodes(batch.node_features, batch.bonds, batch.node_mask)

    def test_checkpoint_finite_and_export_exact(self):
        self.assertTrue(all(torch.isfinite(p).all() for p in self.model.parameters()))
        exported = load_encoder(self.root / 'outputs/ogb_clean/encoder.pt')
        for key, value in self.model.state_dict().items():
            torch.testing.assert_close(value, exported.state_dict()[key], rtol=0, atol=0)

    def test_batch_padding_does_not_change_real_atoms(self):
        with torch.no_grad():
            combined = self.encode(self.batch)
            self.assertTrue((combined[~self.batch.node_mask] == 0).all())
            for i, graph in enumerate(self.graphs):
                alone = self.encode(collate_graphs([graph]))[0]
                torch.testing.assert_close(combined[i, :alone.size(0)], alone, rtol=1e-5, atol=1e-5)

    def test_node_permutation_equivariance(self):
        graph = self.graphs[1]
        perm = torch.tensor([2, 0, 3, 1])
        reordered = MoleculeGraph(graph.node_features[perm], graph.bonds[perm][:, perm], graph.substructure_mask[perm][:, perm])
        with torch.no_grad():
            expected = self.encode(collate_graphs([graph]))[:, perm]
            actual = self.encode(collate_graphs([reordered]))
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-5)

    def test_noise_symmetry_padding_and_vocabulary(self):
        config = DiffusionConfig(min_noise=1.0, max_noise=1.0)
        diffusion = DiscreteGraphDiffusion(config, self.model.config)
        noisy = diffusion.corrupt(self.batch)
        torch.testing.assert_close(noisy.bonds, noisy.bonds.transpose(1, 2), rtol=0, atol=0)
        self.assertTrue((noisy.bonds.diagonal(dim1=1, dim2=2) == 0).all())
        self.assertTrue((noisy.node_features[~self.batch.node_mask] == 0).all())
        pair_valid = self.batch.node_mask[:, :, None] & self.batch.node_mask[:, None, :]
        self.assertTrue((noisy.bonds[~pair_valid] == 0).all())
        self.assertTrue(noisy.node_corruption_mask[self.batch.node_mask].all())
        for field, size in enumerate(diffusion.node_vocab_sizes):
            self.assertTrue(((noisy.node_features[:, :, field] >= 0) & (noisy.node_features[:, :, field] < size)).all())

    def test_diffusion_gradients_reach_every_backbone_component(self):
        noisy = DiscreteGraphDiffusion(DiffusionConfig(), self.model.config).corrupt(self.batch, torch.full((4,), 40))
        output = self.model(noisy.node_features, noisy.bonds, self.batch.node_mask, noisy.timesteps)
        torch.testing.assert_close(output.bond_logits, output.bond_logits.transpose(1, 2), rtol=0, atol=0)
        loss = ChemicalDiffusionLoss()(output, self.batch, noisy)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        for name, param in self.model.named_parameters():
            if name.startswith('graph_projection.'):
                self.assertIsNone(param.grad, name)
            else:
                self.assertIsNotNone(param.grad, name)
                self.assertTrue(torch.isfinite(param.grad).all(), name)
        for name, module in [('node_embeddings', self.model.node_embeddings), ('bond_embedding', self.model.bond_embedding), ('time_embedding', self.model.time_embedding), *[(f'layer_{i}', layer) for i, layer in enumerate(self.model.layers)]]:
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0, name)

    def test_node_encoder_supports_finetuning(self):
        nodes = self.encode(self.batch)
        (nodes * torch.randn_like(nodes)).sum().backward()
        for layer in self.model.layers:
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0 for p in layer.parameters()))

    def test_single_atom_batch_has_finite_loss_and_gradients(self):
        batch = collate_graphs([graph_from_smiles('[Na+]'), graph_from_smiles('[Cl-]')])
        noisy = DiscreteGraphDiffusion(DiffusionConfig(), self.model.config).corrupt(batch)
        output = self.model(noisy.node_features, noisy.bonds, batch.node_mask, noisy.timesteps)
        loss = ChemicalDiffusionLoss()(output, batch, noisy)
        self.assertEqual(loss.bond.item(), 0)
        self.assertTrue(torch.isfinite(loss.total))
        loss.total.backward()
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in self.model.parameters() if p.grad is not None))

    def test_saved_optimizer_covers_backbone_but_not_graph_projection(self):
        checkpoint = torch.load(self.checkpoint, map_location='cpu', weights_only=False)
        ids = [i for group in checkpoint['optimizer_state']['param_groups'] for i in group['params']]
        params = list(self.model.named_parameters())
        self.assertEqual(len(ids), len(params))
        missing = [name for (name, _), i in zip(params, ids) if i not in checkpoint['optimizer_state']['state']]
        self.assertEqual(missing, [name for name, _ in params if name.startswith('graph_projection.')])


if __name__ == '__main__':
    unittest.main(verbosity=2)
