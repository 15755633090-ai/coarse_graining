"""Paired frozen-encoder HimNet readout experiment on the locked Lipo split."""
import argparse
import csv
import importlib.util
import json
import statistics
import time
from pathlib import Path

import torch

from .data import load_dataset, save_json, sha
from .him_scale_readout import ARMS, HimScaleReadout, ScaleProbe, matched_width
from .paper_data import HierarchicalBatch, load_paper, source_inventory


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def table(path, rows):
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('motif/runs/him_repaired_pairs_3seeds_20261019'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[0, 1, 2, 3, 4])
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--width', type=int, default=128)
    parser.add_argument('--device', default='cuda:0')
    a = parser.parse_args()
    assert len(set(a.seeds)) == len(a.seeds) and a.epochs > 0 and a.patience > 0
    assert a.batch_size > 0 and a.width > 0
    torch.set_num_threads(2); torch.manual_seed(25)
    torch.backends.cudnn.benchmark = False
    old_protocol = read(a.run/'protocol.json')
    data_path, preflight = Path(old_protocol['data']), Path(old_protocol['preflight'])
    metadata = read(preflight/'preflight.json')
    manifest = read(a.run/'him_seed0/manifest.json')
    cache_path = preflight/'him_graphs.pt'
    checkpoint_path = a.run/'him_seed0/full_best.pt'
    paper = a.run/'upstream'
    repair_path = a.run/'source/motif/him_repair_20261019/HGNNEncoder.py'
    assert metadata['passed'] and manifest['args']['him_repair']
    assert sha(data_path) == metadata['data_sha256'] == manifest['data_sha256']
    assert sha(cache_path) == metadata['cache_sha256']
    assert source_inventory(paper) == metadata['paper_source_sha256']
    assert sha(repair_path) == read(a.run/'source_manifest.json')[str(repair_path.relative_to(a.run))]
    smoke = read(Path('motif/runs/him_scale_probes_real_smoke_20261025/acceptance.json'))
    checkpoint_hash = sha(checkpoint_path)
    assert smoke['passed'] and not smoke['test_forward'] and smoke['checkpoint_sha256'] == checkpoint_hash
    data = load_dataset(data_path)
    rows = {s: [r for r in data['molecules'] if r['split'] == s] for s in ('train', 'valid', 'test')}
    for s, rs in rows.items():
        assert [r['row'] for r in rs] == metadata['rows'][s] == manifest['rows'][s]
    ids = [r['row'] for rs in rows.values() for r in rs]
    assert len(ids) == len(set(ids)) == len(data['molecules'])
    labels = torch.tensor([r['label'] for r in rows['train']], device=a.device)
    mean, std = labels.mean(), labels.std(unbiased=False)
    assert std > 0
    hidden = 256
    configs = [dict(name=arm, arm=arm, width=a.width, control='main') for arm in ARMS]
    matches = {}
    for combo in ('AS', 'AG', 'SG', 'ASG'):
        budget = sum(p.numel() for p in ScaleProbe(hidden, combo, a.width).parameters())
        for arm in combo:
            match = matched_width(hidden, arm, budget)
            assert match['relative_error'] < .01
            name = f'{arm}_matched_{len(combo)}'
            matches[f'{combo}:{arm}'] = dict(name=name, target=budget, **match)
            if not any(c['name'] == name for c in configs):
                configs.append(dict(name=name, arm=arm, width=match['width'], control=f'matched_{len(combo)}'))
    a.output.mkdir(parents=True, exist_ok=False)
    implementation = {name: sha(Path(__file__).parent/name) for name in
                      ('him_scale_readout.py', 'train_him_scale_probes.py')}
    protocol = dict(scope='frozen task-adapted repaired HimNet; readout contributions, not node deletion',
        encoder_seed=0, head_seeds=a.seeds, epochs=a.epochs, patience=a.patience,
        batch_size=a.batch_size, hidden=hidden, depth=5, dropout=.1, optimizer='AdamW',
        lr=.001, weight_decay=1e-5, gradient_clip=5, width=a.width,
        checkpoint=str(checkpoint_path.resolve()), checkpoint_sha256=checkpoint_hash,
        data_sha256=sha(data_path), graph_cache_sha256=sha(cache_path), repair_sha256=sha(repair_path),
        paper_source_sha256=source_inventory(paper), implementation_sha256=implementation,
        configs=configs, capacity_matches=matches, device=a.device,
        rows={s: [r['row'] for r in rs] for s, rs in rows.items()},
        normalization='training labels only', mean=float(mean), std=float(std),
        fingerprint_in_probe=False, encoder_training_used_fingerprint_and_alignment=True,
        historical_test_seen=True, selection='strict minimum validation RMSE; test once after reload',
        no_hyperparameter_search=True, seeds_measure='head initialization/training, not encoder or split variability')
    save_json(a.output/'protocol.json', protocol)
    for name in implementation:
        (a.output/name).write_bytes((Path(__file__).parent/name).read_bytes())
    def status(**kwargs):
        save_json(a.output/'status.json', kwargs)
    status(phase='loading_verified_graph_cache', completed=0, total=len(configs)*len(a.seeds))
    print({'phase': 'loading_verified_graph_cache', 'planned_runs': len(configs)*len(a.seeds)}, flush=True)
    # Locally generated cache, verified against its recorded SHA-256 above.
    # Normal pickle loading avoids the costly scalar-list weights-only interpreter.
    cache = torch.load(cache_path, map_location='cpu', weights_only=False)
    assert cache['repair_schema'] == metadata['repair_schema']
    cache.pop('atomgraphs', None)
    cls, _, _ = load_paper(paper, 'HimNet', a.device)
    graph = cache['graphs'][rows['train'][0]['row']]
    model = cls('memory', len(graph['f_atoms'][0]), len(graph['f_bonds'][0]),
                graph['fp_x'].shape[-1], device=a.device, out_dim=1)
    spec = importlib.util.spec_from_file_location('formal_frozen_him_repair', repair_path)
    repair = importlib.util.module_from_spec(spec); spec.loader.exec_module(repair)
    old = model.encoder.mpnn_encoder
    model.encoder.mpnn_encoder = repair.HGNNEncoder(old.atom_fdim, old.bond_fdim, hidden, old.depth, a.device)
    state = torch.load(checkpoint_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state['model'], strict=True); model.to(a.device).eval()
    reader = HimScaleReadout(model.encoder, frozen=True).to(a.device)
    before = {k: v.detach().cpu().clone() for k, v in reader.state_dict().items()}
    features, missing, reconstruction = {}, {}, []
    for split, rs in rows.items():
        values, absent = [], 0
        for start in range(0, len(rs), 16):
            batch = HierarchicalBatch([cache['graphs'][r['row']] for r in rs[start:start+16]])
            with torch.no_grad():
                output = reader(batch)
                if start == 0:
                    expected = model.encoder(batch)
                    torch.testing.assert_close(output['full'], expected, atol=1e-6, rtol=1e-5)
                    count = output['counts'].unsqueeze(-1)
                    combined = ((output['p']+output['q'])*count).sum(1)/count.sum(1)
                    recovered = reader.encoder.act_func(reader.encoder.output_layer(combined))
                    torch.testing.assert_close(recovered, expected, atol=1e-6, rtol=1e-5)
                    reconstruction.append(float((recovered-expected).abs().max()))
                assert torch.isfinite(output['z']).all()
                absent += int((~output['present'][:, 1]).sum())
                values.append(output['z'].cpu())
            if start % 256 == 0:
                status(phase='encoding', split=split, encoded=min(start+16, len(rs)), split_rows=len(rs), completed=0)
                print({'phase': 'encoding', 'split': split, 'rows': min(start+16, len(rs))}, flush=True)
        features[split] = torch.cat(values); missing[split] = absent
    assert all(torch.equal(v.cpu(), before[k]) for k, v in reader.state_dict().items())
    torch.save(dict(features=features, rows=protocol['rows'], checkpoint_sha256=checkpoint_hash,
                    no_motif=missing), a.output/'representations.pt')
    save_json(a.output/'encoding_acceptance.json', dict(passed=True, frozen_unchanged=True,
              max_reconstruction_error=max(reconstruction), no_motif=missing))
    del cache, model, reader, before, state
    torch.cuda.empty_cache()
    features = {s: v.to(a.device) for s, v in features.items()}
    targets = {s: torch.tensor([r['label'] for r in rs], device=a.device) for s, rs in rows.items()}
    results = []
    for seed in a.seeds:
        for config in configs:
            started = time.perf_counter()
            torch.manual_seed(seed)
            probe = ScaleProbe(hidden, config['arm'], config['width']).to(a.device)
            optimizer = torch.optim.AdamW(probe.parameters(), lr=.001, weight_decay=1e-5)
            folder = a.output/f'seed_{seed}'/config['name']; folder.mkdir(parents=True)
            best, best_epoch, history = float('inf'), 0, []
            for epoch in range(1, a.epochs+1):
                probe.train()
                order = torch.randperm(len(rows['train']), generator=torch.Generator().manual_seed(seed*10000+epoch))
                for step, cpu_ids in enumerate(order.split(a.batch_size)):
                    # Identical sample order and dropout RNG per paired step.
                    torch.manual_seed(seed*1000000+epoch*1000+step)
                    batch_ids = cpu_ids.to(a.device)
                    prediction = probe(features['train'][batch_ids])
                    loss = (prediction-(targets['train'][batch_ids]-mean)/std).square().mean()
                    if not torch.isfinite(loss): raise FloatingPointError((seed, config, epoch, step))
                    optimizer.zero_grad(set_to_none=True); loss.backward()
                    torch.nn.utils.clip_grad_norm_(probe.parameters(), 5, error_if_nonfinite=True)
                    optimizer.step()
                probe.eval()
                with torch.no_grad():
                    pred = probe(features['valid'])*std+mean
                    valid = float((pred-targets['valid']).square().mean().sqrt())
                history.append(dict(epoch=epoch, valid_rmse=valid))
                if valid < best:
                    best, best_epoch = valid, epoch
                    torch.save(dict(model={k:v.detach().cpu().clone() for k,v in probe.state_dict().items()},
                        epoch=epoch, mean=mean.cpu(), std=std.cpu(), config=config, seed=seed), folder/'best.pt')
                status(phase='training', seed=seed, config=config['name'], epoch=epoch,
                       best_epoch=best_epoch, completed=len(results), total=len(configs)*len(a.seeds))
                if epoch-best_epoch >= a.patience: break
            saved = torch.load(folder/'best.pt', map_location='cpu', weights_only=True)
            probe.load_state_dict(saved['model'], strict=True); probe.eval()
            with torch.no_grad():
                valid_pred = probe(features['valid'])*std+mean
                checked_valid = float((valid_pred-targets['valid']).square().mean().sqrt())
                assert abs(checked_valid-best) < 1e-6
                test_pred = probe(features['test'])*std+mean
                test = float((test_pred-targets['test']).square().mean().sqrt())
            save_json(folder/'history.json', history)
            for split, prediction in (('valid', valid_pred), ('test', test_pred)):
                table(folder/f'{split}_predictions.csv', [dict(row=r['row'], target=r['label'], prediction=float(y))
                    for r, y in zip(rows[split], prediction.cpu())])
            result = dict(seed=seed, **config, parameters=sum(v.numel() for v in probe.parameters()),
                valid_rmse=best, test_rmse=test, best_epoch=best_epoch, epochs_run=epoch,
                reached_budget=epoch == a.epochs, seconds=time.perf_counter()-started)
            results.append(result); save_json(a.output/'results_partial.json', results)
            print({'completed': len(results), 'total': len(configs)*len(a.seeds),
                   'seed': seed, 'config': config['name'], 'epochs': epoch}, flush=True)
    table(a.output/'runs.csv', results)
    summary = []
    for config in configs:
        rr = [r for r in results if r['name'] == config['name']]
        values = [r['test_rmse'] for r in rr]
        summary.append(dict(**config, parameters=rr[0]['parameters'], n=len(values),
                            test_rmse_mean=statistics.mean(values),
                            test_rmse_sd=statistics.stdev(values) if len(values)>1 else 0))
    table(a.output/'summary.csv', summary)
    lookup = {(r['seed'], r['name']): r['test_rmse'] for r in results}
    comparisons = []
    for combo in ('AS', 'AG', 'SG', 'ASG'):
        controls = list(combo) + [matches[f'{combo}:{arm}']['name'] for arm in combo]
        if combo == 'ASG': controls += ['AS', 'AG', 'SG']
        for control in controls:
            delta = [lookup[s, combo]-lookup[s, control] for s in a.seeds]
            for seed, difference in zip(a.seeds, delta):
                comparisons.append(dict(combo=combo, control=control, seed=seed, delta_rmse=difference))
    table(a.output/'paired_deltas.csv', comparisons)
    save_json(a.output/'completion.json', dict(passed=True, runs=len(results), no_motif=missing,
        test_evaluations=len(results), historical_test_seen=True, encoder_seed=0,
        interpretation='fixed task-adapted encoder; head-seed variation only', summary=summary))
    status(phase='complete', completed=len(results), total=len(results))
    print({'phase': 'complete', 'runs': len(results)}, flush=True)


if __name__ == '__main__':
    main()
