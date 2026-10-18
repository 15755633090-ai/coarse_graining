"""Train motif-independent fixed-topology denoising; validation selects weights."""
import argparse
from dataclasses import asdict
from pathlib import Path
import torch
from rdkit import Chem
from .data import load_dataset, save_json, sha, code_identity
from .fixed_topology import Encoder, Config, SCHEMA, records, corrupt, noise_probability
from .supervision import SupervisionCounter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    for name, default in [('epochs', 50), ('batch-size', 16), ('hidden', 128), ('layers', 3), ('steps', 100), ('seed', 0), ('smoke-per-split', 0)]:
        p.add_argument('--'+name, type=int, default=default)
    p.add_argument('--region-probability', type=float, default=.2)
    p.add_argument('--region-fraction', type=float, default=.2)
    p.add_argument('--max-noise', type=float, default=.85)
    p.add_argument('--min-noise', type=float, default=.05, help='Explicit floor; 0 reproduces the old schedule')
    p.add_argument('--bond-weight', type=float, default=1.)
    p.add_argument('--lr', type=float, default=2e-4)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    a = p.parse_args()
    if min(a.epochs, a.batch_size) < 1 or a.lr <= 0 or a.smoke_per_split < 0: raise ValueError('Invalid training configuration')
    torch.manual_seed(a.seed); torch.set_num_threads(4)
    data = load_dataset(a.data)
    # Fail on cross-split molecule identity overlap, including alternate SMILES.
    identities = {}
    for row in data['molecules']:
        key = Chem.MolToSmiles(Chem.MolFromSmiles(row['smiles']), isomericSmiles=True)
        split = row['split']
        if key in identities and identities[key] != split:
            raise ValueError('Canonical molecule overlaps splits; resolve the split before pretraining')
        identities[key] = split
    rows = {s: [r for r in data['molecules'] if r['split'] == s] for s in ('train', 'valid')}
    if a.smoke_per_split: rows = {s: rs[:a.smoke_per_split] for s, rs in rows.items()}
    if any(not rs for rs in rows.values()): raise ValueError('Empty split')
    sets = {s: records(data, rs) for s, rs in rows.items()}
    cfg = Config(hidden=a.hidden, layers=a.layers, steps=a.steps, min_noise=a.min_noise, max_noise=a.max_noise,
                 region_probability=a.region_probability, region_fraction=a.region_fraction, bond_weight=a.bond_weight)
    model = Encoder(asdict(cfg)).to(a.device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=1e-5)
    source = code_identity()
    a.output.mkdir(parents=True, exist_ok=False)
    manifest = {'schema': SCHEMA, 'data_sha256': sha(a.data), 'source': source,
                'rows': {s: [r['row'] for r in rs] for s, rs in rows.items()},
                'test_used_for_optimization_or_selection': False, 'test_identity_audit_only': True,
                'motif_condition': False, 'property_labels_used': False,
                'smoke': bool(a.smoke_per_split or data['smoke']),
                'objective': 'uniform-t masked-attribute CE surrogate, graph mean; not exact ELBO',
                'config': asdict(cfg), 'seed': a.seed, 'lr': a.lr, 'batch_size': a.batch_size,
                't1_base_mask_probability': noise_probability(cfg,1),
                'noise_schedule': 'q(0)=0; q(t>=1)=min+(max-min)*(1-cos(pi*t/T))/2',
                'floor_rationale': 'engineering coverage setting; not selected using property labels or test performance'}
    save_json(a.output/'manifest.json', manifest)
    # Prospective t=1 mask diagnostic. Does not train or alter the training RNG.
    audit = {}
    for split, items in sets.items():
        counter = SupervisionCounter(cfg.steps, cfg.bond_weight)
        for item in items:
            g = item['graph']
            gen = torch.Generator().manual_seed(a.seed+200000+int(item['row']))
            _, _, nm, em, t = corrupt(g,cfg,gen,timestep=1)
            counter.add({'t':t,'atoms':len(g['x']),'bonds':len(g['e']),
                         'masked_atoms':int(nm.sum()),'masked_bonds':int(em.sum())})
        audit[split] = counter.report()['t1']
    save_json(a.output/'t1_mask_audit.json', {'diagnostic_only_not_training':True,
              'base_probability':noise_probability(cfg,1),'splits':audit})
    rng = torch.Generator().manual_seed(a.seed)
    history = []; best = float('inf')
    cumulative = SupervisionCounter(cfg.steps,cfg.bond_weight)
    supervision = {'epochs':[], 'note':'cumulative_train counts actual draws; validation is repeated fixed noise, not independent draws'}
    for epoch in range(1, a.epochs+1):
        scores = {}; counts = {}; epoch_reports = {}
        for split, items in sets.items():
            training = split == 'train'; model.train(training)
            order = torch.randperm(len(items), generator=rng).tolist() if training else list(range(len(items)))
            total = 0.; atoms = bonds = 0
            counter = SupervisionCounter(cfg.steps,cfg.bond_weight)
            for start in range(0, len(items), a.batch_size):
                selected = order[start:start+a.batch_size]
                if training: opt.zero_grad(set_to_none=True)
                for i in selected:
                    # Per-row fixed validation corruption, independent of batching/epoch.
                    gen = rng if training else torch.Generator().manual_seed(a.seed+100000+int(items[i]['row']))
                    with torch.set_grad_enabled(training):
                        loss, detail = model.objective(items[i], gen)
                        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite pretraining loss')
                        if training: (loss/len(selected)).backward()
                    total += float(loss.detach()); atoms += detail['masked_atoms']; bonds += detail['masked_bonds']
                    counter.add(detail)
                if training:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5., error_if_nonfinite=True); opt.step()
            if not atoms: raise RuntimeError('No masked validation/train atoms; increase smoke size or steps/noise')
            scores[split] = total/len(items); counts[split] = {'masked_atoms': atoms, 'masked_bonds': bonds}
            epoch_reports[split] = counter.report()
            counts[split].update({k:epoch_reports[split][k] for k in ('bins','t1')})
            if training: cumulative.merge(counter)
        history.append({'epoch': epoch, **scores, 'counts': counts})
        if scores['valid'] < best:
            best = scores['valid']
            torch.save({**manifest, 'model': model.state_dict(), 'model_config': asdict(cfg),
                        'method': 'both', 'epoch': epoch}, a.output/'best.pt')
        torch.save({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'rng': rng.get_state(),
                    'torch_rng': torch.get_rng_state(), 'epoch': epoch, 'manifest': manifest}, a.output/'last.pt')
        supervision['epochs'].append({'epoch':epoch,**epoch_reports})
        supervision['cumulative_train'] = cumulative.report()
        save_json(a.output/'supervision.json', supervision)
        save_json(a.output/'history.json', history)
        print({'epoch':epoch,**scores,'t1_train':epoch_reports['train']['t1']}, flush=True)
    if code_identity() != source: raise RuntimeError('Source changed during training')


if __name__ == '__main__': main()
