"""Aligned denoising recovery metrics, never used for checkpoint selection."""
import torch
from .data import ATOM_DIMS,pair_mask

ATTRIBUTES=('element','formal_charge','aromatic_annotation','induced_degree')
BINS=('clean_t0','low','medium','high')

class RecoveryMeter:
    def __init__(self,steps):
        self.steps=steps
        self.counts={name:{'motifs':0,'atoms':0,'attribute_correct':[0]*len(ATOM_DIMS),
                          'all_attributes_correct_atoms':0,'true_bonds':0,'true_nonbonds':0,
                          'recalled_bonds':0,'false_positive_bonds':0,'correct_bond_types':0,
                          'exact_connectivity_motifs':0,'exact_element_and_bonds_motifs':0,
                          'exact_all_attributes_and_bonds_motifs':0,'method_counts':{}}
                     for name in BINS}

    def bucket(self,t):
        if not 0<=t<=self.steps:raise ValueError('Time outside schedule')
        if t==0:return 'clean_t0'
        if 3*t<=self.steps:return 'low'
        if 3*t<=2*self.steps:return 'medium'
        return 'high'

    @torch.no_grad()
    def update(self,output,batch,t,graphs=None,include_clean=True):
        atom=torch.stack([v.argmax(-1) for v in output['atom_logits']],-1).cpu()
        edge=output['edge_logits'].argmax(-1).cpu()
        x=batch['x'].cpu();truth=batch['edges'].cpu();mask=batch['mask'].cpu()
        upper=torch.triu(pair_mask(mask),diagonal=1)
        for k,time in enumerate(t.cpu().tolist()):
            bucket=self.bucket(time)
            if bucket=='clean_t0' and not include_clean:continue
            c=self.counts[bucket];selected=mask[k]
            good=atom[k,selected]==x[k,selected]
            a=edge[k][upper[k]];b=truth[k][upper[k]];bond=b!=0;nonbond=~bond
            c['motifs']+=1;c['atoms']+=int(selected.sum())
            for j in range(len(ATOM_DIMS)):c['attribute_correct'][j]+=int(good[:,j].sum())
            c['all_attributes_correct_atoms']+=int(good.all(-1).sum())
            c['true_bonds']+=int(bond.sum());c['true_nonbonds']+=int(nonbond.sum())
            c['recalled_bonds']+=int(((a!=0)&bond).sum())
            c['false_positive_bonds']+=int(((a!=0)&nonbond).sum())
            c['correct_bond_types']+=int(((a==b)&bond).sum())
            connectivity=bool(((a!=0)==bond).all());edges=bool((a==b).all())
            c['exact_connectivity_motifs']+=int(connectivity)
            c['exact_element_and_bonds_motifs']+=int(bool(good[:,0].all()) and edges)
            c['exact_all_attributes_and_bonds_motifs']+=int(bool(good.all()) and edges)
            if graphs is not None:
                name=graphs[k]['method'];c['method_counts'][name]=c['method_counts'].get(name,0)+1

    def report(self):
        def ratio(a,b):return a/b if b else None
        out={}
        for name,c in self.counts.items():
            out[name]={'counts':c,
                'atom_attribute_accuracy':{a:ratio(c['attribute_correct'][j],c['atoms']) for j,a in enumerate(ATTRIBUTES)},
                'atom_all_attributes_accuracy':ratio(c['all_attributes_correct_atoms'],c['atoms']),
                'bond_recall':ratio(c['recalled_bonds'],c['true_bonds']),
                'nonbond_false_positive_rate':ratio(c['false_positive_bonds'],c['true_nonbonds']),
                'bond_type_accuracy_on_true_bonds':ratio(c['correct_bond_types'],c['true_bonds']),
                'motif_exact_connectivity_rate':ratio(c['exact_connectivity_motifs'],c['motifs']),
                'motif_exact_element_and_bonds_rate':ratio(c['exact_element_and_bonds_motifs'],c['motifs']),
                'motif_exact_all_attributes_and_bonds_rate':ratio(c['exact_all_attributes_and_bonds_motifs'],c['motifs'])}
        return out

DEFINITIONS={
    'bins':'t=0 separately; low: 0<t/T<=1/3; medium: 1/3<t/T<=2/3; high: 2/3<t/T<=1',
    'noise_scale_note':'Bins describe normalized time, not linearly increasing Gaussian variance; see cosine alpha_bar schedule.',
    'population':'Clean pass covers all evaluated validation motifs. Noisy bins use fixed item-seeded validation draws; bins are not matched motif populations.',
    'denominators':'Attribute rates over atoms; bond rates over true bonds/nonbonds; exact rates over motifs. All raw counts retained; undefined ratios are null.',
    'bond_type_accuracy':'Correct predicted type / all true bonds; a missed bond counts as incorrect, not excluded.',
    'exact':'Aligned input atom indices, element or all encoded attributes, plus every unordered pair including no-bonds. Not graph-isomorphism matching.',
    'singletons':'No edge pairs; edge rates undefined when denominator is zero. Exact recovery then depends on atom attributes.',
    'chemical_validity_measured':False,'motif_vector_only_decoding_measured':False,
    'checkpoint_selection':'Unchanged: minimum validation denoising loss; recovery metrics are descriptive only.'}
