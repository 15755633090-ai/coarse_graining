"""Real HimNet encoder checks; synthetic inputs, no performance conclusions."""
import argparse
from pathlib import Path

import torch

from .him_scale_readout import ARMS, HimScaleReadout, ScaleProbe, matched_width
from .paper_data import HierarchicalBatch, load_paper, source_inventory
from .data import save_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--paper-root', type=Path, default=Path(
        'coarse_graining1/HimNet_reproduction_20261013/windows'))
    parser.add_argument('--output', type=Path, default=Path('motif/runs/him_scale_readout_check.json'))
    args = parser.parse_args()
    torch.manual_seed(25); torch.set_num_threads(2)
    cls, _, _ = load_paper(args.paper_root, 'HimNet', 'cpu')
    source_before = source_inventory(args.paper_root)
    model = cls('synthetic', 89, 98, 6338, hidden_size=32,
                depth=3, device='cpu', out_dim=1, num_heads=4).eval()
    # Exercise padding, variable node counts, and an absent motif without RDKit caches.
    graphs = []
    for na, ns in ((3, 2), (2, 0), (1, 1)):
        size = na + ns + 1
        graphs.append(dict(n_atoms=size, n_bonds=2,
            f_atoms=torch.randn(size, 89).tolist(),
            f_bonds=torch.randn(2, 98).tolist(),
            a2b=[[1], [0]] + [[] for _ in range(size-2)],
            b2a=[0, 1], b2revb=[1, 0],
            num_part=torch.tensor([[na, ns, 1]]), fp_x=torch.randn(1, 6338)))
    batch = HierarchicalBatch(graphs)
    reader = HimScaleReadout(model.encoder, frozen=True)
    reader.train()  # Frozen encoder must stay in eval, including dropout.
    assert not reader.encoder.training
    original_state = {k: v.clone() for k, v in reader.state_dict().items()}
    with torch.no_grad():
        expected = model.encoder(batch)
        result = reader(batch)
        torch.testing.assert_close(result['full'], expected)
        weights = result['counts'].unsqueeze(-1)
        reconstructed = ((result['p'] + result['q']) * weights).sum(1) / weights.sum(1)
        reconstructed = reader.encoder.act_func(reader.encoder.output_layer(reconstructed))
        torch.testing.assert_close(reconstructed, expected)
        assert result['z'].shape == (3, 3, 32)
        assert torch.equal(result['z'][1, 1], torch.zeros(32))
        assert torch.isfinite(result['z']).all()
        torch.testing.assert_close(reader(batch)['z'], result['z'])
    parameters = {}
    for arm in ARMS:
        probe = ScaleProbe(32, arm, width=16, dropout=0)
        opt = torch.optim.Adam(probe.parameters(), lr=.001)
        prediction = probe(result['z'].detach())
        loss = (prediction-torch.tensor([.1, .2, .3])).square().mean()
        loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in probe.parameters())
        opt.step(); probe.eval()
        copy_probe = ScaleProbe(32, arm, width=16, dropout=0).eval()
        copy_probe.load_state_dict(probe.state_dict(), strict=True)
        torch.testing.assert_close(copy_probe(result['z']), probe(result['z']))
        parameters[arm] = sum(p.numel() for p in probe.parameters())
    assert all(p.grad is None for p in reader.parameters())
    assert all(torch.equal(v, original_state[k]) for k, v in reader.state_dict().items())
    # Optional end-to-end mode must retain both paths' gradients.
    active = HimScaleReadout(model.encoder, frozen=False).eval()
    active(batch)['z'].square().mean().backward()
    for parameter in (active.encoder.mpnn_encoder.W_o.weight,
                      active.encoder.atom_proj.weight, active.encoder.output_layer.weight):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0
    assert source_inventory(args.paper_root) == source_before
    report = dict(passed=True, synthetic_only=True, trained_checkpoint=False,
                  source=str(args.paper_root.resolve()), source_sha256=source_before,
                  forward_sha256=reader.forward_sha256,
                  full_reconstruction_max_error=float((result['full']-expected).abs().max()),
                  parameters=parameters,
                  matched_A_to_ASG=matched_width(32, 'A', parameters['ASG']),
                  upstream_unchanged=True, formal_training=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    save_json(args.output, report)
    print({k: v for k, v in report.items() if k != 'source_sha256'})


if __name__ == '__main__':
    main()
