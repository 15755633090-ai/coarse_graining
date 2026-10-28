"""Method invariants, not performance claims. Run with unittest."""
import unittest
import torch
from torch_geometric.data import Batch

from .data import graph_from_smiles
from .model import Config, MotifContextModel, VARIANTS


class MethodTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.graphs = [graph_from_smiles(s) for s in ("CCOC(=O)c1ccccc1", "C", "[Na+].[Cl-]")]
        self.batch = Batch.from_data_list(self.graphs)
        self.model = MotifContextModel(2, Config(hidden_dim=16, atom_layers=2, dropout=0)).eval()

    def test_partition_batch_and_attachment_alignment(self):
        offset = 0
        for graph in self.graphs:
            self.assertEqual(len(graph.atom_to_motif), graph.num_nodes)
            self.assertEqual(set(graph.atom_to_motif.tolist()), set(range(int(graph.motif_count))))
            offset += int(graph.motif_count)
        self.assertEqual(int(self.batch.atom_to_motif.max()) + 1, offset)
        u, v = self.batch.attachment_index
        self.assertTrue(torch.equal(self.batch.atom_to_motif[u], self.batch.motif_edge_index[0]))
        self.assertTrue(torch.equal(self.batch.atom_to_motif[v], self.batch.motif_edge_index[1]))
        self.assertTrue(torch.equal(self.batch.batch[u], self.batch.batch[v]))
        # Every cut bond is represented in both directions.
        edges = {tuple(e) for e in self.batch.attachment_index.t().tolist()}
        self.assertTrue(all((v, u) in edges for u, v in edges))

    def test_single_atom_no_edges_all_modes(self):
        batch = Batch.from_data_list([self.graphs[1]])
        for mode in VARIANTS:
            prediction, loss, _ = self.model(batch, mode, auxiliary=True)
            self.assertEqual(prediction.shape, (1, 2))
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(prediction).all())

    def test_clean_branch_is_identical_across_auxiliary_modes(self):
        reference = self.model(self.batch, "context")
        for mode in ("fixed", "random", "no_communication", "mask", "shuffle"):
            output, _, _ = self.model(self.batch, mode, auxiliary=True)
            torch.testing.assert_close(output, reference, rtol=0, atol=0)

    def test_corruption_untouched_and_per_molecule_time(self):
        _, q, gid = self.model.encode(self.batch)
        z, steps, selected = self.model.corrupt(q, gid, "random")
        torch.testing.assert_close(z[~selected], q[~selected], rtol=0, atol=0)
        self.assertTrue((steps[~selected] == 0).all())
        for g in gid.unique():
            mask = (gid == g) & selected
            self.assertTrue(mask.any())
            self.assertEqual(steps[mask].unique().numel(), 1)
        masked, _, selected = self.model.corrupt(q, gid, "mask")
        self.assertTrue((masked[selected] == 0).all())

    def test_auxiliary_gradients_and_communication_switch(self):
        _, loss, _ = self.model(self.batch, "random", auxiliary=True)
        loss.backward()
        self.assertGreater(float(self.model.context_layers[0].message[0].weight.grad.abs().sum()), 0)
        self.assertIsNone(self.model.clean_relation[0].weight.grad)
        self.assertIsNone(self.model.property_head[0].weight.grad)
        self.model.zero_grad(set_to_none=True)
        _, loss, _ = self.model(self.batch, "no_communication", auxiliary=True)
        loss.backward()
        self.assertIsNone(self.model.context_layers[0].message[0].weight.grad)
        self.assertGreater(float(self.model.denoise_head[0].weight.grad.abs().sum()), 0)

    def test_graph_batching_preserves_predictions(self):
        together = self.model(self.batch, "context")
        separate = torch.cat([self.model(Batch.from_data_list([g]), "context") for g in self.graphs])
        torch.testing.assert_close(together, separate, atol=1e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
