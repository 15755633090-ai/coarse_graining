"""Implementation checks, not scientific motif-effectiveness experiments."""
from pathlib import Path
from types import SimpleNamespace
import copy
import torch
from torch import nn
from rdkit import Chem
from .data import graph_features,collate_graphs,pair_mask,save_json
from .diffusion import MotifDenoiser,DiffusionSchedule,denoising_loss
from .context import MotifPropertyModel,attach_members
from .adapters import HMGMRLAdapter,HimNetAdapter

def main():
    torch.manual_seed(42);torch.set_num_threads(2)
    g=graph_features(Chem.MolFromSmiles('CC(=O)O'),[1,2,3])
    acid=graph_features(Chem.MolFromSmiles('CCC(=O)O'),[2,3,4])
    assert torch.equal(g['x'],acid['x']) and torch.equal(g['edges'],acid['edges'])
    singleton=graph_features(Chem.MolFromSmiles('[Na+]'),[0])
    b=collate_graphs([g,singleton]);model=MotifDenoiser(32,2,20);schedule=DiffusionSchedule(20)
    xt,et=schedule(b,torch.tensor([9,19]));assert torch.equal(et,et.transpose(1,2))
    assert not bool(xt[~b['mask']].any())
    assert not bool(et[~pair_mask(b['mask'])].any())
    model.eval();clean=model.encode(b)
    alone=model.encode(collate_graphs([g]));torch.testing.assert_close(alone['motif'][0],clean['motif'][0],atol=2e-6,rtol=2e-5)
    p=torch.tensor([2,0,1]);permuted={'x':g['x'][p],'edges':g['edges'][p][:,p]}
    perm=model.encode(collate_graphs([permuted]))
    torch.testing.assert_close(alone['motif'],perm['motif'],atol=2e-6,rtol=2e-5)
    torch.testing.assert_close(alone['nodes'][:,p],perm['nodes'],atol=2e-6,rtol=2e-5)
    out=model(xt,et,torch.tensor([9,19]),b['mask']);loss,_=denoising_loss(out,b);loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters())
    assert float(model.readout[0].weight.grad.abs().sum())>0
    one=model.encode(collate_graphs([singleton]));one_loss,_=denoising_loss(one,collate_graphs([singleton]))
    assert torch.isfinite(one_loss)
    whole=graph_features(Chem.MolFromSmiles('CC(=O)O'),range(4))
    raw=[dict(g,row=0,method='HMG-MRL',index=0,members=[1,2,3]),
         dict(graph_features(Chem.MolFromSmiles('CC(=O)O'),[0,1]),row=0,method='HMG-MRL',index=1,members=[0,1])]
    mol={'row':0,'graph':whole,'motifs':{'HMG-MRL':[0,1]}}
    empty={'row':1,'graph':graph_features(Chem.MolFromSmiles('CC'),[0,1]),'motifs':{'HMG-MRL':[]}}
    data=attach_members({'molecules':[mol,empty],'motifs':raw})
    property_model=MotifPropertyModel(model,32,2).eval()
    pred=property_model([mol,empty],raw,'HMG-MRL')
    single=property_model([mol],raw,'HMG-MRL')
    torch.testing.assert_close(pred[0],single[0],atol=2e-6,rtol=2e-5)
    property_model.train();property_model([mol],raw,'HMG-MRL').sum().backward()
    assert float(property_model.denoiser.readout[0].weight.grad.abs().sum())>0

    # Interface fixture checks: upstream implementations are separately audited;
    # these small fixtures exercise adapter binding and gradient contracts only.
    class HMGFixture(nn.Module):
        def __init__(self): super().__init__();self.frag_input_projection=nn.Linear(7,8)
        def forward(self,frag_src_features): return self.frag_input_projection(frag_src_features)
    base=HMGFixture();adapter=HMGMRLAdapter(base,copy.deepcopy(model))
    original=torch.randn(1,4,7);reference=adapter.backbone.frag_input_projection.base(original)
    result=adapter(collate_graphs([g]),[[0,2]],frag_src_features=original)
    torch.testing.assert_close(result[:,[0,1,3]],reference[:,[0,1,3]])
    assert adapter.backbone.frag_input_projection.override.get() is None
    result.sum().backward();assert adapter.backbone.frag_input_projection.motif_projection.weight.grad is not None
    try: adapter(collate_graphs([g]),[[0,99]],frag_src_features=original)
    except ValueError: pass
    else: raise AssertionError('Bad HMG binding accepted')
    assert adapter.backbone.frag_input_projection.override.get() is None
    class GraphEncoder(nn.Module):
        def forward(self,graph):return graph.f_atoms[1:].mean(0)[None]
    class FP(nn.Module):
        def forward(self,x,training=False):
            if training:return x,x,{'view':x}
            return x
    class Fuse(nn.Module):
        def forward(self,x):return x.sum(0)
    class HimFixture(nn.Module):
        def __init__(self):
            super().__init__();self.atom_fdim=8;self.encoder=GraphEncoder();self.fp_encoder=FP()
            self.feature_fusion=Fuse();self.mlp=nn.Linear(8,1);self.graph_proj_head=nn.Linear(8,8)
    him=HimNetAdapter(HimFixture(),copy.deepcopy(model))
    graph=SimpleNamespace(f_atoms=torch.randn(6,8),fp_x=torch.randn(1,8),a_scope=[(1,5)],mol_atom_num=[3])
    before=graph.f_atoms.clone();prediction=him(graph,collate_graphs([g]),[4],training=True)
    assert torch.equal(before,graph.f_atoms)
    prediction.sum().backward();assert him.projection.weight.grad is not None
    try:him(graph,collate_graphs([g]),[1])
    except ValueError:pass
    else:raise AssertionError('Atom mistaken for motif node')
    result={'all_passed':True,'implementation_checks_only':True,'scientific_performance_claim':False,
            'checks':['induced-only degree and no external H features','symmetric pair noise',
                      'padding and diagonal masking','clean permutation equivariance',
                      'padding/batch independence','singleton finite loss','readout receives denoising gradients',
                      'context batch independence and uncovered atoms','end-to-end gradients',
                      'HMG adapter preserves unbound tokens and resets exception context',
                      'Him adapter preserves input graph and rejects non-motif positions']}
    out=Path(__file__).parent/'runs';out.mkdir(exist_ok=True)
    save_json(out/'implementation_checks.json',result);print(result)

if __name__=='__main__':main()
