import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from .data import load_dataset,save_json,sha,SCHEMA,code_identity
from .pretrain import load_encoder
from .diffusion import MotifDenoiser
from .context import MotifPropertyModel,attach_members

def main():
    p=argparse.ArgumentParser(description='Standalone atom+motif context model; NOT a full paper reproduction')
    p.add_argument('--data',type=Path,required=True);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--method',choices=['HimNet','HMG-MRL'],required=True)
    p.add_argument('--arm',choices=['pretrained','random','atom-only'],default='pretrained')
    p.add_argument('--freeze-pretrained',action='store_true')
    p.add_argument('--epochs',type=int,default=200);p.add_argument('--patience',type=int,default=30)
    p.add_argument('--batch-size',type=int,default=16);p.add_argument('--seed',type=int,default=0)
    p.add_argument('--motif-batch-size',type=int,default=32)
    p.add_argument('--hidden',type=int,default=128);p.add_argument('--context-layers',type=int,default=3)
    p.add_argument('--device',default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--max-batches',type=int,default=0,help='Smoke only')
    args=p.parse_args();torch.manual_seed(args.seed);torch.set_num_threads(4)
    source_identity=code_identity()
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists(): raise FileExistsError('Choose a new output folder')
    if args.epochs<1: raise ValueError('epochs must be positive')
    data=attach_members(load_dataset(args.data)); checkpoint=None
    if args.arm=='pretrained':
        if not args.checkpoint: raise ValueError('Pretrained arm requires --checkpoint')
        denoiser,checkpoint=load_encoder(args.checkpoint)
        if checkpoint['data_sha256']!=sha(args.data): raise ValueError('Pretraining/downstream data hash mismatch')
        if checkpoint['method'] not in ('both',args.method): raise ValueError('Pretraining motif method mismatch')
    elif args.checkpoint:
        _,ref=load_encoder(args.checkpoint)
        torch.manual_seed(args.seed+1000)
        denoiser=MotifDenoiser(**ref['config'])  # same shape, explicitly fresh weights
    else: denoiser=MotifDenoiser(hidden=args.hidden)
    split={s:[m for m in data['molecules'] if m['split']==s] for s in ('train','valid','test')}
    if any(not x for x in split.values()): raise ValueError('Need train/valid/test molecule splits')
    torch.manual_seed(args.seed+2000)  # shared downstream initialization stream
    model=MotifPropertyModel(denoiser,args.hidden,args.context_layers,args.freeze_pretrained,args.arm!='atom-only',args.motif_batch_size).to(args.device)
    y=torch.tensor([m['label'] for m in split['train']]);mean=float(y.mean());std=float(y.std(unbiased=False).clamp_min(1e-6))
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.001,weight_decay=1e-5)
    loader=DataLoader(split['train'],batch_size=args.batch_size,shuffle=True,collate_fn=list,
                      generator=torch.Generator().manual_seed(args.seed+1000))
    def evaluate(rows):
        model.eval();pred=[];target=[];ids=[]
        with torch.no_grad():
            for i,batch in enumerate(DataLoader(rows,batch_size=args.batch_size,collate_fn=list)):
                if args.max_batches and i>=args.max_batches: break
                pred.extend((model(batch,data['motifs'],args.method)*std+mean).cpu().tolist())
                target.extend(m['label'] for m in batch);ids.extend(m['row'] for m in batch)
        rmse=float(((torch.tensor(pred)-torch.tensor(target)).square().mean()).sqrt())
        return rmse,{'prediction':pred,'target':target,'row_ids':ids}
    best=float('inf');best_epoch=0;history=[]
    for epoch in range(1,args.epochs+1):
        model.train()
        for i,batch in enumerate(loader):
            if args.max_batches and i>=args.max_batches: break
            target=torch.tensor([m['label'] for m in batch],device=args.device)
            pred=model(batch,data['motifs'],args.method)
            loss=(pred-(target-mean)/std).square().mean()
            if not torch.isfinite(loss): raise FloatingPointError('Non-finite property loss')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),5);optimizer.step()
        score,_=evaluate(split['valid']);history.append({'epoch':epoch,'valid_rmse':score})
        if score<best:
            best=score;best_epoch=epoch
            torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},
                        'denoiser_config':denoiser.config,'hidden':args.hidden,'context_layers':args.context_layers,
                        'motif_batch_size':args.motif_batch_size,'arm':args.arm,'method':args.method,
                        'target_mean':mean,'target_std':std,'epoch':epoch,'schema':SCHEMA},args.output/'best.pt')
        save_json(args.output/'history.json',history);print(history[-1],flush=True)
        if epoch-best_epoch>=args.patience: break
    ck=torch.load(args.output/'best.pt',map_location='cpu',weights_only=True);model.load_state_dict(ck['model'])
    # Test only once after validation-based model selection.
    test,pred=evaluate(split['test']);save_json(args.output/'test_predictions.json',pred)
    if code_identity()!=source_identity: raise RuntimeError('Source changed during this run')
    report={'valid_rmse':best,'test_rmse':test,'best_epoch':best_epoch,
            'arm':args.arm,'method':args.method,'seed':args.seed,
            'data_sha256':sha(args.data),'pretrained_sha256':sha(args.checkpoint) if args.checkpoint else None,
            'source_sha256':source_identity,'torch_version':str(torch.__version__),
            'test_molecules_evaluated':len(pred['row_ids']),
            'parameters':sum(p.numel() for p in model.parameters()),
            'trainable_parameters':sum(p.numel() for p in model.parameters() if p.requires_grad),
            'standalone_context_model':True,'full_paper_reproduction':False,
            'smoke':bool(args.max_batches or data['smoke'] or (checkpoint and checkpoint['smoke']))}
    save_json(args.output/'results.json',report);print(report)

if __name__=='__main__': main()
