"""Pretrain the original discrete diffusion on full graphs with clean motif types."""
import argparse
import time
from pathlib import Path
import torch
from .data import load_dataset,save_json,sha,code_identity
from .fullgraph import (FullGraphEncoder,ModelConfig,DiffusionConfig,
    DiscreteGraphDiffusion,ChemicalDiffusionLoss,vocabulary,records,SCHEMA)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['HimNet','HMG-MRL'],required=True)
    p.add_argument('--epochs',type=int,default=50);p.add_argument('--batch-size',type=int,default=4)
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--layers',type=int,default=3)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--max-batches',type=int,default=0)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args()
    if min(args.epochs,args.batch_size,args.hidden,args.layers)<1:raise ValueError('Positive sizes required')
    args.output.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(4);torch.manual_seed(args.seed)
    data=load_dataset(args.data);vocab,counts=vocabulary(data,args.method)
    sets={s:records(data,[r for r in data['molecules'] if r['split']==s],args.method,vocab) for s in ('train','valid')}
    cfg=ModelConfig(hidden_dim=args.hidden,num_layers=args.layers).to_dict()
    dc=DiffusionConfig();model=FullGraphEncoder(cfg,vocab).to(args.device)
    diffusion=DiscreteGraphDiffusion(dc,model.legacy.config)
    objective=ChemicalDiffusionLoss().to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=2e-4)
    save_json(args.output/'types.json',{'vocabulary':vocab,'train_counts':counts,'rule':'canonical induced fragment SMILES; train only',
        'coverage':{s:{'motifs':sum(len(r['types']) for r in rs),'unknown':sum(t==1 for r in rs for t in r['types'])} for s,rs in sets.items()}})
    from .fullgraph import LEGACY
    source={'module':code_identity(),'legacy':{p.name:sha(p) for p in (LEGACY/'bond_diffusion').glob('*.py')}}
    save_json(args.output/'source.json',source)
    history=[];best=float('inf')
    for epoch in range(args.epochs):
        started=time.perf_counter()
        scores={}
        for split,items in sets.items():
            training=split=='train';model.train(training)
            order=torch.randperm(len(items)).tolist() if training else list(range(len(items)))
            total=count=0
            # Reset a local RNG context for reproducible validation corruptions.
            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if str(args.device).startswith('cuda') else []):
                if not training:torch.manual_seed(args.seed+100000)
                for k,start in enumerate(range(0,len(items),args.batch_size)):
                    if args.max_batches and k>=args.max_batches:break
                    chunk=[items[i] for i in order[start:start+args.batch_size]]
                    with torch.set_grad_enabled(training):
                        batch,_=model.batch(chunk,args.device);noisy=diffusion.corrupt(batch)
                        out=model(chunk,noisy);loss=objective(out,batch,noisy).total
                        if not torch.isfinite(loss):raise FloatingPointError('Nonfinite diffusion loss')
                        if training:
                            optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
                    total+=float(loss.detach())*len(chunk);count+=len(chunk)
                    if (k+1)%25==0:
                        save_json(args.output/'progress.json',{'epoch':epoch+1,'phase':split,'batch':k+1,
                            'batches':(len(items)+args.batch_size-1)//args.batch_size,'seconds':time.perf_counter()-started})
            if not count:raise ValueError('Empty split')
            scores[split]=total/count
        history.append({'epoch':epoch+1,**scores,'seconds':time.perf_counter()-started});print(history[-1],flush=True)
        if scores['valid']<best:
            best=scores['valid'];torch.save({'schema':SCHEMA,'model_config':cfg,'vocab':vocab,'model':model.state_dict(),
                'method':args.method,'epoch':epoch+1,'data_sha256':sha(args.data),'diffusion_config':dc.to_dict(),
                'smoke':bool(args.max_batches or data['smoke']),'recovery_label':'motif-type-conditioned recovery',
                'pool':'member mean followed by trainable projection (projection trained downstream)'},args.output/'best.pt')
        save_json(args.output/'history.json',history)
        save_json(args.output/'progress.json',{'phase':'epoch_complete',**history[-1]})
    save_json(args.output/'manifest.json',{'condition':'clean motif type embeddings; never corrupted','test_used':False,
        'noise':'original DiscreteGraphDiffusion','loss':'original ChemicalDiffusionLoss','data_sha256':sha(args.data),
        'smoke':bool(args.max_batches or data['smoke']),'performance_gain_established':False})

if __name__=='__main__':main()
