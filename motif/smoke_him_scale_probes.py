"""Real-data smoke of frozen HimNet probes; deliberately no test evaluation."""
import argparse
import importlib.util
import json
from pathlib import Path

import torch

from .data import load_dataset, save_json, sha
from .him_scale_readout import ARMS, HimScaleReadout, ScaleProbe, matched_width
from .paper_data import HierarchicalBatch, load_paper, source_inventory


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, default=Path('motif/runs/him_repaired_pairs_3seeds_20261019'))
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.manual_seed(25); torch.set_num_threads(2)
    protocol = read_json(a.run/'protocol.json')
    data_path = Path(protocol['data']); preflight = Path(protocol['preflight'])
    metadata = read_json(preflight/'preflight.json')
    manifest = read_json(a.run/'him_seed0/manifest.json')
    checkpoint_path = a.run/'him_seed0/full_best.pt'
    cache_path = preflight/'him_graphs.pt'
    paper_root = a.run/'upstream'
    assert metadata['passed'] and manifest['args']['him_repair']
    assert sha(data_path) == metadata['data_sha256'] == manifest['data_sha256']
    assert sha(cache_path) == metadata['cache_sha256']
    assert source_inventory(paper_root) == metadata['paper_source_sha256']
    data = load_dataset(data_path)
    rows = {s: [r for r in data['molecules'] if r['split'] == s] for s in ('train', 'valid')}
    for s, rs in rows.items():
        assert [r['row'] for r in rs] == metadata['rows'][s] == manifest['rows'][s]
    # The fixed smoke subset is chosen by original row order, never by scores.
    selected = {'train': rows['train'][:32], 'valid': rows['valid'][:16]}
    labels = torch.tensor([r['label'] for r in rows['train']])
    mean, std = labels.mean(), labels.std(unbiased=False)
    assert std > 0
    cache = torch.load(cache_path, map_location='cpu', weights_only=True)
    assert cache['repair_schema'] == metadata['repair_schema']
    cls, _, _ = load_paper(paper_root, 'HimNet', 'cpu')
    graph = cache['graphs'][selected['train'][0]['row']]
    model = cls('memory', len(graph['f_atoms'][0]), len(graph['f_bonds'][0]),
                graph['fp_x'].shape[-1], device='cpu', out_dim=1)
    repair_path = a.run/'source/motif/him_repair_20261019/HGNNEncoder.py'
    source_manifest = read_json(a.run/'source_manifest.json')
    assert sha(repair_path) == source_manifest[str(repair_path.relative_to(a.run))]
    spec = importlib.util.spec_from_file_location('frozen_him_smoke_repair', repair_path)
    repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)
    old = model.encoder.mpnn_encoder
    model.encoder.mpnn_encoder = repair.HGNNEncoder(
        old.atom_fdim, old.bond_fdim, old.hidden_size, old.depth, 'cpu')
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model'], strict=True); model.eval()
    reader = HimScaleReadout(model.encoder, frozen=True)
    before = {k: v.clone() for k, v in reader.state_dict().items()}
    features, targets, errors = {}, {}, []
    for split, rs in selected.items():
        chunks = []
        for start in range(0, len(rs), 8):
            batch = HierarchicalBatch([cache['graphs'][r['row']] for r in rs[start:start+8]])
            with torch.no_grad():
                expected = model.encoder(batch)
                output = reader(batch)
                torch.testing.assert_close(output['full'], expected)
                n = output['counts'].unsqueeze(-1)
                combined = ((output['p']+output['q'])*n).sum(1)/n.sum(1)
                recovered = reader.encoder.act_func(reader.encoder.output_layer(combined))
                torch.testing.assert_close(recovered, expected)
                errors.append(float((recovered-expected).abs().max()))
                # Repaired model must tolerate a different batch composition.
                single = reader(HierarchicalBatch([cache['graphs'][rs[start]['row']]]))
                torch.testing.assert_close(single['z'][0], output['z'][0], atol=1e-6, rtol=1e-5)
            chunks.append(output['z'])
        features[split] = torch.cat(chunks)
        targets[split] = torch.tensor([r['label'] for r in rs])
    assert all(torch.isfinite(v).all() for v in features.values())
    a.output.mkdir(parents=True, exist_ok=False)
    torch.save({'features': features, 'rows': {s: [r['row'] for r in rs] for s, rs in selected.items()}},
               a.output/'representations.pt')
    configs = [(arm, 32, 'main') for arm in ARMS]
    for combo in ('AS', 'AG', 'SG', 'ASG'):
        target = sum(v.numel() for v in ScaleProbe(256, combo, width=32).parameters())
        for arm in combo:
            match = matched_width(256, arm, target)
            assert match['relative_error'] < .01
            configs.append((arm, match['width'], f'matched_to_{combo}'))
    results = []
    for arm, width, control in configs:
        torch.manual_seed(25)
        probe = ScaleProbe(256, arm, width=width, dropout=.1)
        optimizer = torch.optim.AdamW(probe.parameters(), lr=1e-3, weight_decay=1e-5)
        path = a.output/f'{control}_{arm}.pt'
        best, history = float('inf'), []
        for epoch in range(1, 4):
            probe.train()
            order = torch.randperm(32, generator=torch.Generator().manual_seed(25+epoch))
            for ids in order.split(8):
                torch.manual_seed(25000+epoch*100+int(ids[0]))
                prediction = probe(features['train'][ids])
                loss = (prediction-(targets['train'][ids]-mean)/std).square().mean()
                assert torch.isfinite(loss)
                optimizer.zero_grad(); loss.backward()
                assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in probe.parameters())
                torch.nn.utils.clip_grad_norm_(probe.parameters(), 5, error_if_nonfinite=True)
                optimizer.step()
            probe.eval()
            with torch.no_grad():
                pred = probe(features['valid'])*std+mean
                rmse = float((pred-targets['valid']).square().mean().sqrt())
            history.append({'epoch': epoch, 'valid_rmse': rmse})
            if rmse < best:
                best = rmse
                torch.save({'model': probe.state_dict(), 'epoch': epoch, 'mean': mean, 'std': std}, path)
        saved = torch.load(path, weights_only=True)
        probe.load_state_dict(saved['model'], strict=True); probe.eval()
        with torch.no_grad(): reference = probe(features['valid']).clone()
        with torch.no_grad():
            for v in probe.parameters(): v.add_(1)
        probe.load_state_dict(saved['model'], strict=True)
        with torch.no_grad(): torch.testing.assert_close(probe(features['valid']), reference, rtol=0, atol=0)
        save_json(a.output/f'{control}_{arm}_history.json', history)
        results.append({'arm': arm, 'control': control, 'width': width,
                        'parameters': sum(v.numel() for v in probe.parameters()),
                        'best_epoch': saved['epoch'], 'valid_rmse_smoke_only': best})
    assert all(torch.equal(v, before[k]) for k, v in reader.state_dict().items())
    assert all(v.grad is None for v in reader.parameters())
    report = dict(passed=True, smoke_only=True, test_forward=False, formal_comparison=False,
                  checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=sha(checkpoint_path),
                  checkpoint_epoch=checkpoint['epoch'], checkpoint_source_used_fingerprint=True,
                  data_sha256=sha(data_path), graph_cache_sha256=sha(cache_path),
                  repair_source_sha256=sha(repair_path), paper_source_sha256=source_inventory(paper_root),
                  implementation_sha256={name: sha(Path(__file__).parent/name) for name in
                    ('him_scale_readout.py', 'smoke_him_scale_probes.py')},
                  rows={s: [r['row'] for r in rs] for s, rs in selected.items()},
                  normalization='all original training rows only', mean=float(mean), std=float(std),
                  max_reconstruction_error=max(errors), frozen_unchanged=True, results=results)
    save_json(a.output/'acceptance.json', report)
    print({'passed': True, 'runs': len(results), 'train': 32, 'valid': 16,
           'test_forward': False, 'max_reconstruction_error': max(errors)}, flush=True)


if __name__ == '__main__':
    main()
