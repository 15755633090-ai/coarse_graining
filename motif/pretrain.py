import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import load_dataset, collate_graphs, to_device, save_json, sha, SCHEMA, code_identity
from .diffusion import MotifDenoiser, DiffusionSchedule, denoising_loss
from .batching import SizeBucketSampler,validation_noise
from .recovery import RecoveryMeter,DEFINITIONS

def load_encoder(path,device='cpu'):
    ck=torch.load(path,map_location='cpu',weights_only=True)
    if ck['schema']!=SCHEMA: raise ValueError('Checkpoint schema mismatch')
    model=MotifDenoiser(**ck['config']); model.load_state_dict(ck['model'],strict=True)
    return model.to(device),ck

def main():
    p=argparse.ArgumentParser(description='Train shared motif denoiser; never trains on test motifs')
    p.add_argument('--data',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--method',choices=['both','HimNet','HMG-MRL'],default='both')
    p.add_argument('--epochs',type=int,default=50); p.add_argument('--batch-size',type=int,default=32)
    p.add_argument('--hidden',type=int,default=128); p.add_argument('--layers',type=int,default=3)
    p.add_argument('--steps',type=int,default=100); p.add_argument('--seed',type=int,default=0)
    p.add_argument('--lr',type=float,default=1e-3); p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-batches',type=int,default=0,help='Smoke only; cap batches per epoch')
    p.add_argument('--pair-budget',type=int,default=0,help='Max padded N*N pairs per batch; 0 disables this extra cap')
    args=p.parse_args(); torch.manual_seed(args.seed); torch.set_num_threads(4)
    source_identity=code_identity()
    if args.epochs<1: raise ValueError('epochs must be positive')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists(): raise FileExistsError('Choose a new output folder; checkpoints are not overwritten')
    data=load_dataset(args.data)
    groups=[g for g in data['motifs'] if args.method=='both' or g['method']==args.method]
    train=[g for g in groups if g['split']=='train']; valid=[g for g in groups if g['split']=='valid']
    if not train or not valid: raise ValueError('Need both train and valid motifs')
    model=MotifDenoiser(args.hidden,args.layers,args.steps).to(args.device)
    schedule=DiffusionSchedule(args.steps).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    sampler=SizeBucketSampler([len(g['x']) for g in train],args.batch_size,True,args.seed,args.pair_budget)
    loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate_graphs,num_workers=0)
    val=DataLoader(valid,batch_sampler=SizeBucketSampler([len(g['x']) for g in valid],args.batch_size,pair_budget=args.pair_budget),
                   collate_fn=list,num_workers=0)
    best=float('inf'); history=[];recovery_history=[]
    for epoch in range(args.epochs):
        model.train(); values=[];sampler.set_epoch(epoch)
        for i,b in enumerate(loader):
            if args.max_batches and i>=args.max_batches: break
            b=to_device(b,args.device); t=schedule.sample_t(len(b['x']),args.device)
            xt,et=schedule(b,t); out=model(xt,et,t,b['mask']); loss,_=denoising_loss(out,b)
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite denoising loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step(); values.append((float(loss.detach()),len(b['x'])))
        model.eval(); scores=[];meter=RecoveryMeter(args.steps)
        # Same noise/t for every validation pass, independent of training RNG.
        with torch.no_grad():
            for i,graphs in enumerate(val):
                if args.max_batches and i>=args.max_batches: break
                b,xt,et,t=validation_noise(schedule,graphs,args.seed+100000,args.device)
                output=model(xt,et,t,b['mask']);loss,_=denoising_loss(output,b)
                scores.append((float(loss),len(b['x'])))
                # t=0 gets its own complete validation pass, never double-count
                # the clean items already sampled by the noisy loss evaluator.
                meter.update(output,b,t,graphs,include_clean=False)
                clean=model.encode(b)
                meter.update(clean,b,torch.zeros(len(graphs),dtype=torch.long,device=args.device),graphs)
        score=sum(v*n for v,n in scores)/sum(n for _,n in scores)
        history.append({'epoch':epoch+1,'train_loss':sum(v*n for v,n in values)/sum(n for _,n in values),'valid_loss':score})
        recovery=meter.report();recovery_history.append({'epoch':epoch+1,'bins':recovery})
        save_json(args.output/'recovery_metrics.json',{'definitions':DEFINITIONS,'epochs':recovery_history,
                  'smoke':bool(args.max_batches or data['smoke'])})
        if score<best:
            best=score
            torch.save({'schema':SCHEMA,'config':model.config,'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'epoch':epoch+1,'valid_loss':score,'data_sha256':sha(args.data),
                        'method':args.method,'seed':args.seed,'clean_probability':schedule.clean_probability,
                        'recovery_metrics':recovery,
                        'train_molecules':sorted({g['row'] for g in train}),
                        'smoke':bool(args.max_batches or data['smoke'])},args.output/'best.pt')
        save_json(args.output/'history.json',history)
        print(history[-1],flush=True)
    if code_identity()!=source_identity: raise RuntimeError('Source changed during this run')
    save_json(args.output/'manifest.json',{'data_sha256':sha(args.data),'checkpoint_sha256':sha(args.output/'best.pt'),
               'source_sha256':source_identity,'torch_version':str(torch.__version__),
               'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'test_used':False,'train_motifs':len(train),'valid_motifs':len(valid),
               'loss_reduction':'mean_per_motif','validation_noise':'item_identity_seeded',
               'recovery_metrics':'recovery_metrics.json','clean_metrics_pass':True,
               'smoke':bool(args.max_batches or data['smoke'])})

if __name__=='__main__': main()
