"""Compare optimized computations to the frozen running pretraining source."""
import argparse
import ast
import copy
import subprocess
import sys
from pathlib import Path
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from rdkit import Chem
from .data import ATOM_DIMS,SCHEMA,graph_features,collate_graphs,pair_mask,save_json,sha
from .diffusion import MotifDenoiser,DiffusionSchedule,denoising_loss
from .recovery import RecoveryMeter,BINS,ATTRIBUTES
from .batching import validation_noise,CachedValidation

def extract(path,name,env):
    node=next(n for n in ast.parse(path.read_text(encoding='utf-8')).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name==name)
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),env);return env[name]

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2);torch.manual_seed(71)
    old_loss=extract(args.reference_root/'diffusion.py','denoising_loss',{'torch':torch,'F':F,'pair_mask':pair_mask,'ATOM_DIMS':ATOM_DIMS})
    old_meter=extract(args.reference_root/'recovery.py','RecoveryMeter',{'torch':torch,'pair_mask':pair_mask,'ATOM_DIMS':ATOM_DIMS,'BINS':BINS,'ATTRIBUTES':ATTRIBUTES})
    graphs=[]
    for i,s in enumerate(['[Na+]','CC','CCO','c1ccccc1','C.C']):
        mol=Chem.MolFromSmiles(s);g=graph_features(mol,range(mol.GetNumAtoms()));g.update(row=i,method='test',index=0);graphs.append(g)
    batch=collate_graphs(graphs);schedule=DiffusionSchedule(20);t=torch.tensor([0,2,9,19,13]);xt,et=schedule(batch,t)
    a=MotifDenoiser(16,2,20);b=copy.deepcopy(a)
    ao=a(xt,et,t,batch['mask']);bo=b(xt,et,t,batch['mask']);la,_=old_loss(ao,batch);lb,_=denoising_loss(bo,batch)
    torch.testing.assert_close(la,lb,atol=1e-6,rtol=1e-6);la.backward();lb.backward()
    for (name,x),(other,y) in zip(a.named_parameters(),b.named_parameters()):
        assert name==other;torch.testing.assert_close(x.grad,y.grad,atol=3e-6,rtol=2e-4)
    before=old_meter(20);after=RecoveryMeter(20)
    for include in (True,False):
        before.update(ao,batch,t,graphs,include_clean=include);after.update(bo,batch,t,graphs,include_clean=include)
    assert before.report()==after.report()
    for budget in (0,16):
        loader=DataLoader(graphs,batch_size=2,collate_fn=list)
        cached=CachedValidation(loader,schedule,77,'cpu',max_mb=budget)
        for gs,bs,x,e,ts in cached:
            ref,rx,re,rt=validation_noise(schedule,gs,77,'cpu')
            assert torch.equal(x,rx) and torch.equal(e,re) and torch.equal(ts,rt)
            assert all(torch.equal(bs[k],ref[k]) for k in ref)
    # Exercise actual full-checkpoint restore with a tiny synthetic cache.
    motifs=[]
    for split in ('train','valid'):
        for i,g in enumerate(graphs):motifs.append({**g,'split':split,'row':i+(0 if split=='train' else 10)})
    dataset=args.output/'toy.pt';torch.save({'schema':SCHEMA,'motifs':motifs,'smoke':True},dataset)
    base=[sys.executable,'-X','utf8','-m','motif.pretrain','--data',str(dataset),'--batch-size','3',
          '--hidden','16','--layers','1','--steps','20','--seed','7','--device','cpu']
    def run(name,epochs,resume=None):
        cmd=base+['--output',str(args.output/name),'--epochs',str(epochs)]
        if resume:cmd+=['--resume-from',str(resume)]
        completed=subprocess.run(cmd,capture_output=True,text=True,encoding='utf-8')
        (args.output/(name+'.log')).write_text(completed.stdout+completed.stderr,encoding='utf-8')
        if completed.returncode:raise RuntimeError(completed.stderr)
    run('uninterrupted',2);run('first',1);run('resumed',2,args.output/'first/last.pt')
    full=torch.load(args.output/'uninterrupted/last.pt',weights_only=True)
    resumed=torch.load(args.output/'resumed/last.pt',weights_only=True)
    assert all(torch.equal(v,resumed['model'][k]) for k,v in full['model'].items())
    assert torch.equal(full['rng_cpu'],resumed['rng_cpu'])
    for x,y in zip(full['history'],resumed['history']):
        assert x['train_loss']==y['train_loss'] and x['valid_loss']==y['valid_loss']
    result={'passed':True,'vectorized_loss_and_gradients_match':True,'recovery_integer_counts_exact_match':True,
            'cached_and_uncached_validation_noise_bitwise_equal':True,'resume_cpu_model_and_rng_bitwise_equal':True,
            'reference_sha256':{n:sha(args.reference_root/n) for n in ('diffusion.py','recovery.py')},
            'loss_absolute_difference':float(abs(la.detach()-lb.detach())),'scientific_performance_experiment':False}
    save_json(args.output/'checks.json',result);print(result)

if __name__=='__main__':main()
