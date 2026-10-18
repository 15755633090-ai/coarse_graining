"""Executable invariance and end-to-end gradient checks for the new branch."""
import torch
from .fixed_topology import Encoder, graph, corrupt, Config, records
from .adapters import _BoundProjection


def main():
    torch.set_num_threads(2); torch.manual_seed(13)
    model = Encoder({'hidden': 24, 'layers': 2, 'steps': 10})
    g = graph('CC(=O)Nc1ccccc1')
    before = {k: v.clone() for k, v in g.items()}
    for seed in range(30):
        x, e, nm, em, t = corrupt(g, model.cfg, torch.Generator().manual_seed(seed))
        assert all(torch.equal(g[k], v) for k, v in before.items())
        assert torch.equal(x[nm], torch.tensor([119, 17]).expand(int(nm.sum()), -1))
        assert (em | ~(nm[g['uv'][0]] | nm[g['uv'][1]])).all()
        assert (e[em] == 5).all()
    model.eval()
    h, _, b = model(g)
    reversed_g = {**g, 'uv': g['uv'].flip(0)}
    h2, _, b2 = model(reversed_g)
    torch.testing.assert_close(h, h2); torch.testing.assert_close(b, b2)
    perm = torch.randperm(len(h)); inverse = torch.argsort(perm)
    gp = {'x': g['x'][perm], 'uv': inverse[g['uv']], 'e': g['e']}
    torch.testing.assert_close(model(gp)[0][inverse], h)
    items = [{'graph': g, 'members': [[0, 1, 2], [1, 3], [4, 5, 6, 7, 8, 9]]}]
    s = model.encode_records(items)
    torch.testing.assert_close(s[0], h[[0, 1, 2]].mean(0))
    torch.testing.assert_close(model.encode_records(items*2), torch.cat((s, s)))
    projection = _BoundProjection(torch.nn.Linear(4, 8), 24, 'residual', .1)
    original = torch.randn(1, 5, 4)
    token = projection.override.set((s, [[0, 1], [0, 2], [0, 3]]))
    try: out = projection(original)
    finally: projection.override.reset(token)
    torch.testing.assert_close(out[:, [0, 4]], projection.base(original)[:, [0, 4]])
    out.square().mean().backward()
    assert model.atoms[0].weight.grad.abs().sum() > 0
    assert projection.motif_gate.grad.abs().sum() > 0
    model.zero_grad(set_to_none=True)
    loss, detail = model.objective({'graph': g}, torch.Generator().manual_seed(5))
    assert torch.isfinite(loss); loss.backward()
    for smiles in ('[Na+]', 'C.C', 'CC'):
        one = graph(smiles)
        loss, _ = model.objective({'graph': one}, torch.Generator().manual_seed(5))
        assert torch.isfinite(loss)
        assert model(one)[0].shape[0] == len(one['x'])
    print('PASS: immutable topology, linked masks, undirected symmetry, permutation equivariance, overlapping motif readout, residual-only binding, property gradients, edgeless graphs')


if __name__ == '__main__': main()
