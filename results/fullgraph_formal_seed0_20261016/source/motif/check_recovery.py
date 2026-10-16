"""Metric bookkeeping tests with known predictions, not a quality experiment."""
from pathlib import Path
import torch
from rdkit import Chem
from torch.nn import functional as F
from .data import graph_features,collate_graphs,ATOM_DIMS,BOND_CLASSES,save_json
from .recovery import RecoveryMeter

def perfect(batch):
    return {'atom_logits':[F.one_hot(batch['x'][...,j],d).float()*100 for j,d in enumerate(ATOM_DIMS)],
            'edge_logits':F.one_hot(batch['edges'],BOND_CLASSES).float()*100}

def main():
    graphs=[]
    for s in ('CCO','[Na+]'):
        mol=Chem.MolFromSmiles(s);graphs.append(graph_features(mol,range(mol.GetNumAtoms())))
    b=collate_graphs(graphs);t=torch.tensor([1,0]);meter=RecoveryMeter(12)
    meter.update(perfect(b),b,t);r=meter.report()
    assert r['low']['bond_recall']==1 and r['low']['nonbond_false_positive_rate']==0
    assert r['low']['motif_exact_all_attributes_and_bonds_rate']==1
    assert r['clean_t0']['bond_recall'] is None and r['high']['motif_exact_element_and_bonds_rate'] is None
    # Miss one real bond, invent one absent bond, keep element predictions exact.
    bad=perfect(b);bad['edge_logits'][0,0,1]=F.one_hot(torch.tensor(0),BOND_CLASSES)*100
    bad['edge_logits'][0,0,2]=F.one_hot(torch.tensor(1),BOND_CLASSES)*100
    changed=RecoveryMeter(12);changed.update(bad,b,t);q=changed.report()['low']
    assert q['bond_recall']==.5 and q['nonbond_false_positive_rate']==1
    assert q['bond_type_accuracy_on_true_bonds']==.5 and q['motif_exact_element_and_bonds_rate']==0
    split=RecoveryMeter(12)
    for k,g in enumerate(graphs):
        one=collate_graphs([g]);split.update(perfect(one),one,t[k:k+1])
    assert split.report()==r
    assert [meter.bucket(i) for i in (0,1,4,5,8,9,12)]==['clean_t0','low','low','medium','medium','high','high']
    excluded=RecoveryMeter(12);excluded.update(perfect(b),b,torch.zeros(2,dtype=torch.long),include_clean=False)
    assert excluded.report()['clean_t0']['counts']['motifs']==0
    # A correct element is not sufficient for all-attribute reconstruction.
    other=perfect(b);other['atom_logits'][1][0,0]=F.one_hot(torch.tensor(0),ATOM_DIMS[1])*100
    strict=RecoveryMeter(12);strict.update(other,b,t)
    assert strict.report()['low']['motif_exact_element_and_bonds_rate']==1
    assert strict.report()['low']['motif_exact_all_attributes_and_bonds_rate']==0
    result={'passed':True,'checks':['known perfect predictions','missed and spurious bonds',
            'singleton and empty denominators','batch partition invariance','bin boundaries',
            'clean double-count prevention','element vs all-attribute exactness'],
            'denoising_quality_claim':False}
    out=Path(__file__).parent/'runs';out.mkdir(exist_ok=True);save_json(out/'recovery_checks.json',result);print(result)

if __name__=='__main__':main()
