"""Executable invariance and end-to-end gradient checks for the new branch."""
import torch
import math
import tempfile
from pathlib import Path
from dataclasses import asdict
from .fixed_topology import Encoder, graph, corrupt, Config, noise_probability, load_encoder, SCHEMA
from .supervision import SupervisionCounter
from .adapters import _BoundProjection


def main():
    torch.set_num_threads(2); torch.manual_seed(13)
    cfg = Config()
    assert noise_probability(cfg, 0) == 0
    assert .05 < noise_probability(cfg, 1) < .051
    assert noise_probability(cfg, cfg.steps) == cfg.max_noise
    values = [noise_probability(cfg,t) for t in range(cfg.steps+1)]
    assert all(a <= b for a,b in zip(values,values[1:]))
    old = Config(min_noise=0)
    assert math.isclose(noise_probability(old,1),.85*(1-math.cos(math.pi/100))/2)
    for bad in (-.1,.85,1.):
        try: Config(min_noise=bad)
        except ValueError: pass
        else: raise AssertionError('Invalid floor accepted')
    # Exact aggregation, including edge-only targets when bond_weight is zero.
    c = SupervisionCounter(6)
    examples = [dict(t=1,atoms=4,bonds=3,masked_atoms=0,masked_bonds=0),
                dict(t=2,atoms=4,bonds=3,masked_atoms=1,masked_bonds=2),
                dict(t=3,atoms=2,bonds=1,masked_atoms=0,masked_bonds=1),
                dict(t=6,atoms=1,bonds=0,masked_atoms=1,masked_bonds=0)]
    for d in examples: c.add(d)
    report = c.report()
    assert report['all']['sampled_graphs']==4 and report['all']['effective_graphs']==3
    assert report['bins']['low']['sampled_graphs']==2 and report['t1']['zero_target_graphs']==1
    assert sum(d['masked_bonds'] for d in report['by_time'].values())==3
    merged = SupervisionCounter(6); merged.merge(c); assert merged.report()==report
    report['t1']['sampled_graphs']=999; assert c.report()['t1']['sampled_graphs']==1
    no_bonds = SupervisionCounter(6,bond_weight=0); no_bonds.add(examples[2])
    assert no_bonds.report()['all']['zero_target_graphs']==1
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
    # Legacy checkpoint configs without min_noise must load with floor=0.
    legacy_config = asdict(model.cfg); legacy_config.pop('min_noise')
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'old.pt'
        torch.save({'schema':SCHEMA,'model_config':legacy_config,'model':model.state_dict()},path)
        loaded,_ = load_encoder(path)
        assert loaded.cfg.min_noise==0
        torch.testing.assert_close(loaded(g)[0],model(g)[0])
    print('PASS: noise floor/endpoints, legacy checkpoint semantics, exact per-time supervision aggregation')
    print('PASS: immutable topology, linked masks, undirected symmetry, permutation equivariance, overlapping motif readout, residual-only binding, property gradients, edgeless graphs')


if __name__ == '__main__': main()
