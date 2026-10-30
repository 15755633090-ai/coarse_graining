"""Fixed computable chemical targets, inspired by knowledge-guided pretraining.

These are RDKit pseudo-labels, not experimental property labels or a KPGT
reproduction. Descriptor inputs are never supplied to the graph encoder.
"""
import numpy as np
import torch
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors
from .data import condition_from_batch
from .typed_bonds import edge_target

DESCRIPTORS=(('MolWt',Descriptors.MolWt),('MolLogP',Descriptors.MolLogP),
    ('TPSA',rdMolDescriptors.CalcTPSA),('HDonors',Lipinski.NumHDonors),
    ('HAcceptors',Lipinski.NumHAcceptors),('RotatableBonds',Lipinski.NumRotatableBonds),
    ('AromaticRings',rdMolDescriptors.CalcNumAromaticRings),
    ('AliphaticRings',rdMolDescriptors.CalcNumAliphaticRings),
    ('FractionCSP3',rdMolDescriptors.CalcFractionCSP3),('RingCount',rdMolDescriptors.CalcNumRings),
    ('HeavyAtomCount',Lipinski.HeavyAtomCount),('ValenceElectrons',Descriptors.NumValenceElectrons))

def chemical_targets(smiles):
    desc=[];fps=[]
    for s in smiles:
        mol=Chem.MolFromSmiles(s)
        if mol is None:raise ValueError('Invalid target molecule')
        desc.append([fn(mol) for _,fn in DESCRIPTORS])
        bits=np.zeros(512,dtype=np.float32)
        DataStructs.ConvertToNumpyArray(Chem.RDKFingerprint(mol,fpSize=512),bits)
        fps.append(bits)
    d=torch.tensor(desc,dtype=torch.float32);f=torch.tensor(np.stack(fps))
    if not torch.isfinite(d).all():raise ValueError('Nonfinite descriptor')
    return d,f

def target_statistics(descriptors,fingerprints):
    center=descriptors.mean(0);scale=descriptors.std(0,unbiased=False)
    descriptor_active=scale>1e-6
    freq=fingerprints.mean(0);fingerprint_active=(freq>0)&(freq<1)
    if not descriptor_active.any() or not fingerprint_active.any():raise ValueError('No variable targets')
    return {'center':center,'scale':scale.clamp_min(1e-6),'descriptor_active':descriptor_active,
            'fingerprint_active':fingerprint_active,'positive_weight':((1-freq)/freq.clamp_min(1e-6)).clamp(max=20)}

def graph_readout(model,batch):
    c=condition_from_batch(batch,model.config.motif_prior);x=edge_target(batch,c)
    clean=torch.cat((x,torch.zeros_like(x[...,:1])),-1)
    t=torch.zeros(len(x),device=x.device,dtype=torch.long)
    atoms=model.encoder(c,clean,t)['atoms']
    return atoms.sum(1)/c.nodes.sum(1).clamp_min(1)[:,None]

def knowledge_loss(prediction,descriptors,fingerprints,stats):
    standardized=(descriptors-stats['center'])/stats['scale']
    d=torch.nn.functional.mse_loss(prediction[:,:len(DESCRIPTORS)][:,stats['descriptor_active']],standardized[:,stats['descriptor_active']])
    active=stats['fingerprint_active']
    f=torch.nn.functional.binary_cross_entropy_with_logits(prediction[:,len(DESCRIPTORS):][:,active],fingerprints[:,active],pos_weight=stats['positive_weight'][active])
    return d+f,{'descriptor_mse':d,'fingerprint_bce':f}
