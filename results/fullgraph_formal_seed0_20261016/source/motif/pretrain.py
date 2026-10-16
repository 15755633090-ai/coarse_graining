import argparse
import shutil
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import load_dataset, collate_graphs, to_device, save_json, sha, SCHEMA, code_identity
from .diffusion import MotifDenoiser, DiffusionSchedule, denoising_loss
from .batching import SizeBucketSampler,CachedValidation
from .recovery import RecoveryMeter,DEFINITIONS
from .structure import PATTERNS,DEFINITION,attach_targets,structure_loss

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
    p.add_argument('--validation-cache-mb',type=int,default=512)
    p.add_argument('--resume-from',type=Path,help='Full last.pt only; weight-only best.pt is rejected')
    p.add_argument('--structure-weight',type=float,default=0.,help='Clean s_M-only internal pattern BCE weight; 0 preserves original objective')
    args=p.parse_args(); torch.manual_seed(args.seed); torch.set_num_threads(4)
    source_identity=code_identity()
    if args.epochs<1: raise ValueError('epochs must be positive')
    if not 0<=args.structure_weight<float('inf'):raise ValueError('Invalid structure weight')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists() and not args.resume_from: raise FileExistsError('Choose a new output folder; checkpoints are not overwritten')
    data_hash=sha(args.data)
    data=load_dataset(args.data)
    groups=[g for g in data['motifs'] if args.method=='both' or g['method']==args.method]
    train=[g for g in groups if g['split']=='train']; valid=[g for g in groups if g['split']=='valid']
    if not train or not valid: raise ValueError('Need both train and valid motifs')
    if args.structure_weight:
        train=attach_targets(train);valid=attach_targets(valid)
        save_json(args.output/'structure_labels.json',{'definition':DEFINITION,
                  'train_positive_counts':torch.stack([g['structure_target'] for g in train]).sum(0).tolist(),
                  'valid_positive_counts':torch.stack([g['structure_target'] for g in valid]).sum(0).tolist(),
                  'train_count':len(train),'valid_count':len(valid),'test_used':False})
    model=MotifDenoiser(args.hidden,args.layers,args.steps,structure_dim=len(PATTERNS) if args.structure_weight else 0).to(args.device)
    schedule=DiffusionSchedule(args.steps).to(args.device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-5)
    sampler=SizeBucketSampler([len(g['x']) for g in train],args.batch_size,True,args.seed,args.pair_budget)
    loader=DataLoader(train,batch_sampler=sampler,collate_fn=collate_graphs,num_workers=0)
    val=DataLoader(valid,batch_sampler=SizeBucketSampler([len(g['x']) for g in valid],args.batch_size,pair_budget=args.pair_budget),
                   collate_fn=list,num_workers=0)
    protocol={'data_sha256':data_hash,'config':model.config,'method':args.method,'seed':args.seed,
              'batch_size':args.batch_size,'pair_budget':args.pair_budget,'lr':args.lr,
              'max_batches':args.max_batches,'clean_probability':schedule.clean_probability}
    if args.structure_weight:protocol.update(structure_weight=args.structure_weight,structure_definition=DEFINITION)
    started=time.perf_counter()
    validation=CachedValidation(val,schedule,args.seed+100000,args.device,args.validation_cache_mb,args.max_batches)
    cache_info={**validation.metadata(),'seconds':time.perf_counter()-started}
    save_json(args.output/'validation_cache.json',cache_info);print({'validation_cache':cache_info},flush=True)
    best=float('inf'); history=[];recovery_history=[];start_epoch=0;resume_info=None
    if args.resume_from:
        resumed=torch.load(args.resume_from,map_location='cpu',weights_only=True)
        if 'optimizer' not in resumed or 'rng_cpu' not in resumed:
            raise ValueError('Weight-only checkpoint cannot resume optimizer/RNG; restart explicitly instead of silently resetting Adam')
        if resumed['protocol']!=protocol:raise ValueError('Resume protocol/data/config mismatch')
        model.load_state_dict(resumed['model'],strict=True);optimizer.load_state_dict(resumed['optimizer'])
        best=resumed['best_valid_loss'];history=resumed['history'];recovery_history=resumed['recovery_history'];start_epoch=resumed['epoch']
        if args.epochs<=start_epoch:raise ValueError('Requested epoch limit already reached')
        old_best=args.resume_from.parent/'best.pt'
        if not old_best.exists() or sha(old_best)!=resumed['best_checkpoint_sha256']:raise ValueError('Missing/mismatched best checkpoint for resume')
        if old_best.resolve()!=(args.output/'best.pt').resolve():shutil.copy2(old_best,args.output/'best.pt')
        torch.set_rng_state(resumed['rng_cpu'])
        if str(args.device).startswith('cuda'):torch.cuda.set_rng_state_all(resumed['rng_cuda'])
        resume_info={'checkpoint':str(args.resume_from),'sha256':sha(args.resume_from),'completed_epochs':start_epoch,
                     'optimizer_and_rng_restored':True,'previous_source_sha256':resumed['source_sha256']}
    def atomic_save(value,path):
        tmp=path.with_suffix('.tmp');torch.save(value,tmp);tmp.replace(path)
    def cpu_tree(value):
        if isinstance(value,torch.Tensor):return value.detach().cpu().clone()
        if isinstance(value,dict):return {k:cpu_tree(v) for k,v in value.items()}
        if isinstance(value,list):return [cpu_tree(v) for v in value]
        if isinstance(value,tuple):return tuple(cpu_tree(v) for v in value)
        return value
    for epoch in range(start_epoch,args.epochs):
        phase=time.perf_counter();model.train();total=torch.zeros((),device=args.device);count=0;sampler.set_epoch(epoch)
        train_structure=torch.zeros((),device=args.device)
        for i,b in enumerate(loader):
            if args.max_batches and i>=args.max_batches: break
            b=to_device(b,args.device); t=schedule.sample_t(len(b['x']),args.device)
            xt,et=schedule(b,t); out=model(xt,et,t,b['mask']); loss,_=denoising_loss(out,b)
            if args.structure_weight:
                auxiliary=structure_loss(model,model.encode(b),b['structure_target'])
                loss=loss+args.structure_weight*auxiliary
                train_structure+=auxiliary.detach()*len(b['x'])
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite denoising loss')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5); optimizer.step();total+=loss.detach()*len(b['x']);count+=len(b['x'])
            if (i+1)%100==0:
                save_json(args.output/'progress.json',{'phase':'train','epoch':epoch+1,'batch':i+1,'batches_total':len(sampler),
                          'elapsed_seconds':time.perf_counter()-phase})
        train_loss=float(total/count);train_count=count;train_seconds=time.perf_counter()-phase
        phase=time.perf_counter();model.eval();total=torch.zeros((),device=args.device);count=0;meter=RecoveryMeter(args.steps)
        valid_structure=torch.zeros((),device=args.device);valid_denoise=torch.zeros((),device=args.device)
        # Same noise/t for every validation pass, independent of training RNG.
        with torch.no_grad():
            for i,(graphs,b,xt,et,t) in enumerate(validation):
                output=model(xt,et,t,b['mask']);loss,_=denoising_loss(output,b)
                valid_denoise+=loss*len(b['x'])
                # t=0 gets its own complete validation pass, never double-count
                # the clean items already sampled by the noisy loss evaluator.
                meter.update(output,b,t,graphs,include_clean=False)
                clean=model.encode(b)
                if args.structure_weight:
                    targets=torch.stack([g['structure_target'] for g in graphs]).to(args.device)
                    auxiliary=structure_loss(model,clean,targets)
                    loss=loss+args.structure_weight*auxiliary
                    valid_structure+=auxiliary*len(graphs)
                total+=loss*len(b['x']);count+=len(b['x'])
                meter.update(clean,b,torch.zeros(len(graphs),dtype=torch.long,device=args.device),graphs)
        score=float(total/count);valid_seconds=time.perf_counter()-phase
        history.append({'epoch':epoch+1,'train_loss':train_loss,'valid_loss':score,
                        'train_seconds':train_seconds,'valid_seconds':valid_seconds,'epoch_seconds':train_seconds+valid_seconds})
        if args.structure_weight:
            history[-1].update(train_structure_bce=float(train_structure/train_count),
                               valid_structure_bce=float(valid_structure/count),valid_denoising_loss=float(valid_denoise/count))
        recovery=meter.report();recovery_history.append({'epoch':epoch+1,'bins':recovery})
        save_json(args.output/'recovery_metrics.json',{'definitions':DEFINITIONS,'epochs':recovery_history,
                  'smoke':bool(args.max_batches or data['smoke'])})
        if score<best:
            best=score
            atomic_save({'schema':SCHEMA,'config':model.config,'model':cpu_tree(model.state_dict()),
                        'epoch':epoch+1,'valid_loss':score,'data_sha256':data_hash,
                        'method':args.method,'seed':args.seed,'clean_probability':schedule.clean_probability,
                        'recovery_metrics':recovery,
                        'structure_supervision':{'weight':args.structure_weight,'definition':DEFINITION if args.structure_weight else None,
                                                'selection':'validation denoising + weight * clean structure BCE'},
                        'train_molecules':sorted({g['row'] for g in train}),
                        'smoke':bool(args.max_batches or data['smoke'])},args.output/'best.pt')
        atomic_save({'schema':SCHEMA,'config':model.config,'model':cpu_tree(model.state_dict()),
                     'optimizer':cpu_tree(optimizer.state_dict()),'epoch':epoch+1,'protocol':protocol,
                     'best_valid_loss':best,'best_checkpoint_sha256':sha(args.output/'best.pt'),
                     'rng_cpu':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if str(args.device).startswith('cuda') else [],
                     'history':history,'recovery_history':recovery_history,'source_sha256':source_identity},args.output/'last.pt')
        save_json(args.output/'history.json',history)
        print(history[-1],flush=True)
        save_json(args.output/'progress.json',{'phase':'epoch_complete',**history[-1]})
    if code_identity()!=source_identity: raise RuntimeError('Source changed during this run')
    save_json(args.output/'manifest.json',{'data_sha256':data_hash,'checkpoint_sha256':sha(args.output/'best.pt'),
               'source_sha256':source_identity,'torch_version':str(torch.__version__),
               'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
               'test_used':False,'train_motifs':len(train),'valid_motifs':len(valid),
               'loss_reduction':'mean_per_motif','validation_noise':'item_identity_seeded',
               'recovery_metrics':'recovery_metrics.json','clean_metrics_pass':True,
               'validation_cache':cache_info,'resume':resume_info,'full_resume_checkpoint':'last.pt',
               'smoke':bool(args.max_batches or data['smoke'])})

if __name__=='__main__': main()
