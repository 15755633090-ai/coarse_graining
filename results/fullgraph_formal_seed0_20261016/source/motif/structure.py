"""Internal graph-pattern labels; no implicit H, boundary valence or task labels.

Patterns describe connectivity, not electronic effects or functional-group identity.
Only the same x/edges available to the encoder are used. No sanitization/capping.
"""
import torch
from torch.nn import functional as F
from rdkit import Chem

PATTERNS = (
    ('carbonyl','[#6]=[#8]'),
    ('carbon_oxygen_single','[#6]-[#8]'),
    ('carbon_nitrogen_single','[#6]-[#7]'),
    ('amide_connectivity','[#7]-[#6]=[#8]'),
    ('carboxyl_connectivity','[#8]-[#6]=[#8]'),
    ('nitrile','[#6]#[#7]'),
    ('carbon_double_bond','[#6]=[#6]'),
    ('carbon_triple_bond','[#6]#[#6]'),
    ('sulfur_oxygen_double','[#16]=[#8]'),
    ('phosphorus_oxygen','[#15]~[#8]'),
    ('carbon_halogen','[#6]~[#9,#17,#35,#53]'),
    ('aromatic_bond','[a]:[a]'),
)
QUERIES=tuple(Chem.MolFromSmarts(smarts) for _,smarts in PATTERNS)
DEFINITION={'version':'internal-connectivity-patterns-v1','patterns':list(PATTERNS),
            'input':'cached internal x/edges only; no outside atoms, H counts or downstream labels',
            'meaning':'multi-label structural patterns, not electronic properties',
            'unmatched':'all-zero vector is valid; no forced functional-group class'}

def structure_target(graph):
    rw=Chem.RWMol()
    for row in graph['x'].tolist():
        atom=Chem.Atom(row[0]);atom.SetFormalCharge(row[1]-8)
        atom.SetIsAromatic(bool(row[2]));atom.SetNoImplicit(True);rw.AddAtom(atom)
    types={1:Chem.BondType.SINGLE,2:Chem.BondType.DOUBLE,3:Chem.BondType.TRIPLE,
           4:Chem.BondType.AROMATIC,5:Chem.BondType.UNSPECIFIED}
    for i,j in torch.triu(graph['edges']!=0,diagonal=1).nonzero().tolist():
        rw.AddBond(i,j,types[int(graph['edges'][i,j])])
    mol=rw.GetMol()
    return torch.tensor([mol.HasSubstructMatch(q) for q in QUERIES],dtype=torch.float32)

def attach_targets(graphs):
    # Cache repeated structures without consulting molecule/split/type identities.
    cache={};result=[]
    for graph in graphs:
        key=(graph['x'].numpy().tobytes(),graph['edges'].numpy().tobytes(),len(graph['x']))
        if key not in cache:cache[key]=structure_target(graph)
        result.append({**graph,'structure_target':cache[key]})
    return result

def structure_loss(model,clean_output,targets):
    # This head has no access to node/edge states: every gradient passes via s_M.
    logits=model.structure_head(clean_output['motif'])
    return F.binary_cross_entropy_with_logits(logits,targets,reduction='none').mean(1).mean()
