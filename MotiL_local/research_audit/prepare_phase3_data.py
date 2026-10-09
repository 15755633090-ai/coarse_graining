"""Canonical, overlap-filtered ZINC pilot subset and real >10-type batch audit."""
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from rdkit import Chem, RDLogger
from motif_instances import load_patterns
from motif_ablation import choose_motifs

ROOT = Path(__file__).resolve().parent
MICRO = ROOT.parent/'MotiL_micromolecule'
OUT = ROOT.parent/'phase3_runs_20261009'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_smiles(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        for row in csv.DictReader(f):
            yield row.get('smiles') or row.get('SMILES')


def main():
    RDLogger.DisableLog('rdApp.*')
    protocol = json.loads((ROOT/'pilot_protocol.json').read_text(encoding='utf-8'))
    path = MICRO/'data/zinc15_250K.csv'
    assert sha(path) == protocol['pretraining_data']['sha256']
    OUT.mkdir(exist_ok=True)
    if (OUT/'data_manifest.json').exists():
        raise SystemExit('Manifest exists; do not overwrite the prepared subset')
    excluded = set()
    downstream = {}
    for task in ['esol', 'bbbp', 'clintox', 'bace']:
        task_path = MICRO/'data'/f'{task}.csv'
        for s in read_smiles(task_path):
            mol = Chem.MolFromSmiles(s) if s else None
            if mol is not None:
                excluded.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        downstream[task] = sha(task_path)
    canonical, invalid = set(), 0
    count = 0
    for s in read_smiles(path):
        count += 1
        mol = Chem.MolFromSmiles(s) if s else None
        if mol is None:
            invalid += 1
            continue
        canonical.add(Chem.MolToSmiles(mol, isomericSmiles=True))
        if count % 50000 == 0:
            print('Canonicalized', count, flush=True)
    eligible = sorted(canonical-excluded)
    rng = np.random.default_rng(2024)
    subset = [eligible[i] for i in rng.permutation(len(eligible))[:4096]]
    assert len(subset) == 4096 and not (set(subset) & excluded)
    subset_path = OUT/'zinc_subset.csv'
    with subset_path.open('w', encoding='utf-8', newline='') as f:
        w = csv.writer(f); w.writerow(['smiles']); w.writerows([s] for s in subset)
    patterns = load_patterns(MICRO/'chemprop/data/funcgroup.txt')
    counts, example = [], None
    for start in range(0, len(subset), 16):
        molecules = [Chem.MolFromSmiles(s) for s in subset[start:start+16]]
        offset, scopes = 1, []
        for mol in molecules:
            scopes.append((offset, mol.GetNumAtoms())); offset += mol.GetNumAtoms()
        full = choose_motifs(molecules, scopes, patterns, 'B2b')
        counts.append(len(full))
        if len(full) > 10 and example is None:
            capped = choose_motifs(molecules, scopes, patterns, 'B2a-2')
            assert len(capped) == 10
            example = {'start_index': start, 'smiles': subset[start:start+16],
                       'uncapped_types': list(full), 'capped_types': list(capped),
                       'full_instances': sum(map(len, full.values())),
                       'capped_instances': sum(map(len, capped.values()))}
    manifest = {'zinc_sha256': sha(path), 'source_rows': count, 'invalid': invalid,
        'unique_canonical': len(canonical), 'canonical_exact_overlap_removed': len(canonical & excluded),
        'downstream_sha256': downstream, 'subset_seed': 2024, 'subset_size': len(subset),
        'subset_sha256': sha(subset_path), 'subset_exact_downstream_overlap': 0,
        'batch_type_counts': counts, 'batches_over_ten': sum(n>10 for n in counts),
        'batches_total': len(counts), 'real_over_ten_example': example,
        'limitations': 'Exact canonical isomeric-SMILES exclusion only; original checkpoint exposure unknown.'}
    (OUT/'data_manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    print('Prepared', len(subset), 'molecules;', manifest['batches_over_ten'], '/', len(counts), 'batches over ten types', flush=True)
    assert example is not None, 'No real >10-type batch found; review before proceeding'


if __name__ == '__main__':
    main()
