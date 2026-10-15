"""Reuse audited membership, without re-extracting or modifying motifs."""
import argparse
import csv
import gzip
import json
from pathlib import Path
import torch
from rdkit import Chem
from .data import graph_features, save_json, sha, SCHEMA

ROOT=Path(__file__).resolve().parents[1]

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,default=ROOT/'coarse_graining1/motif_structure_audit_20261014/outputs/verified_motifs.json')
    p.add_argument('--output',type=Path,default=ROOT/'motif/data/lipo.pt')
    p.add_argument('--labels',type=Path,default=ROOT/'coarse_graining1/datasets/lipo/raw/graph-label.csv.gz')
    p.add_argument('--limit-molecules',type=int,default=0,help='Smoke only; 0 = all. Never used silently.')
    args=p.parse_args()
    manifest=args.audit.parent/'manifest.json'
    if manifest.exists():
        checked=json.loads(manifest.read_text(encoding='utf-8'))
        if checked['unresolved_structural_errors'] or sha(args.audit)!=checked['verified_data_sha256']:
            raise ValueError('Audited source verification failed')
    records=json.loads(args.audit.read_text(encoding='utf-8'))
    with gzip.open(args.labels,'rt',encoding='utf-8') as f: labels=[float(r[0]) for r in csv.reader(f)]
    rows={}; motifs=[]
    allowed=sorted({r['source_row'] for r in records})
    if args.limit_molecules: allowed=allowed[:args.limit_molecules]
    allowed=set(allowed)
    for r in records:
        row=r['source_row']
        if row not in allowed: continue
        if row not in rows:
            mol=Chem.MolFromSmiles(r['smiles'])
            rows[row]={'row':row,'smiles':r['smiles'],'split':r['split'],'label':labels[row],
                       'graph':graph_features(mol,range(mol.GetNumAtoms())),'motifs':{}}
        else: mol=Chem.MolFromSmiles(r['smiles'])
        groups=[list(m['atoms']) for m in r['motifs']]
        rows[row]['motifs'][r['method']]=[]
        for k,members in enumerate(groups):
            item=graph_features(mol,members)
            item.update(row=row,split=r['split'],method=r['method'],index=k,members=members)
            rows[row]['motifs'][r['method']].append(len(motifs)); motifs.append(item)
    payload={'schema':SCHEMA,'molecules':list(rows.values()),'motifs':motifs,
             'source_sha256':sha(args.audit),'labels_sha256':sha(args.labels),'smoke':bool(args.limit_molecules)}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    torch.save(payload,args.output)
    report={'molecules':len(rows),'motifs':len(motifs),'cache_sha256':sha(args.output),
            'source_sha256':payload['source_sha256'],'labels_sha256':payload['labels_sha256'],
            'schema':SCHEMA,'smoke':payload['smoke'],'modified_extraction_rules':False,
            'synthetic_singletons_added':False,
            'aromatic_annotation':'preserved from source; no whole-molecule degree/H/ring/chirality features',
            'by_method_split':{f'{m}:{s}':sum(g['method']==m and g['split']==s for g in motifs)
                               for m in ('HimNet','HMG-MRL') for s in ('train','valid','test')}}
    save_json(args.output.with_suffix('.json'),report)
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
