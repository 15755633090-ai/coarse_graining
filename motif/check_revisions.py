"""Regression tests for per-motif reduction and bounded encoders."""
import copy
from pathlib import Path
import torch
from rdkit import Chem
from .data import graph_features,collate_graphs,save_json
from .diffusion import MotifDenoiser,DiffusionSchedule,denoising_loss
from .batching import SizeBucketSampler,encode_motifs,validation_noise

def main():
    torch.manual_seed(55);torch.set_num_threads(2)
    graphs=[]
    for i,s in enumerate(['[Na+]','CC','CC(=O)O','c1ccccc1']):
        m=Chem.MolFromSmiles(s);g=graph_features(m,range(m.GetNumAtoms()))
        g.update(row=i,method='test',index=0);graphs.append(g)
    model=MotifDenoiser(16,2,20).eval();schedule=DiffusionSchedule(20)
    batch,xt,et,t=validation_noise(schedule,graphs,51,'cpu')
    full,stats=denoising_loss(model(xt,et,t,batch['mask']),batch)
    parts=[]
    for group in ([graphs[0]],[graphs[1],graphs[2]],[graphs[3]]):
        b,x,e,ts=validation_noise(schedule,group,51,'cpu')
        loss,info=denoising_loss(model(x,e,ts,b['mask']),b);parts.extend(info['per_motif'])
    torch.testing.assert_close(full,torch.stack(parts).mean(),atol=1e-6,rtol=1e-6)
    for i,g in enumerate(graphs):
        b,x,e,ts=validation_noise(schedule,[g],51,'cpu');n=len(g['x'])
        assert torch.equal(xt[i,:n],x[0]) and torch.equal(et[i,:n,:n],e[0]) and t[i]==ts[0]
    sampler=SizeBucketSampler([len(g['x']) for g in graphs],3,True,19,72)
    indices=list(sampler);assert sorted(i for b in indices for i in b)==list(range(4))
    assert all(len(b)*max(len(graphs[i]['x']) for i in b)**2<=72 for b in indices)
    assert indices==list(sampler)
    try:SizeBucketSampler([20],32,pair_budget=399)
    except ValueError:pass
    else:raise AssertionError('Oversized single motif silently admitted')
    reference=copy.deepcopy(model)
    pooled=model.encode(collate_graphs(graphs))['motif'];pooled.square().sum().backward()
    chunked=encode_motifs(reference,graphs,'cpu',2,pair_budget=72,checkpoint_grad=True)
    torch.testing.assert_close(pooled,chunked,atol=2e-6,rtol=2e-5)
    chunked.square().sum().backward()
    for (name,a),(other,b) in zip(model.named_parameters(),reference.named_parameters()):
        assert name==other
        if a.grad is None:assert b.grad is None
        else:torch.testing.assert_close(a.grad,b.grad,atol=3e-5,rtol=3e-4)
    result={'passed':True,'per_motif_loss_batch_invariant':True,'validation_noise_batch_invariant':True,
            'bucket_coverage_and_budget':True,'checkpoint_chunk_outputs_and_gradients':True,
            'performance_experiment':False}
    out=Path(__file__).parent/'runs';out.mkdir(exist_ok=True);save_json(out/'revision_checks.json',result);print(result)

if __name__=='__main__':main()
