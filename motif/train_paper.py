"""Real-data regression training for original HMG-MRL/HimNet + motif module.

Uses the original model classes and paper feature pipelines, not context.py.
No claim to reproduce a paper's reported split/hyperparameter statistics.
"""
import argparse
import time
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from rdkit import Chem
from .data import load_dataset,save_json,sha,code_identity
from .pretrain import load_encoder
from .adapters import HMGMRLAdapter,HimNetAdapter,hmg_motif_inputs,himnet_motif_inputs
from .paper_data import load_paper,source_inventory,prepare_him_graphs,HierarchicalBatch,load_hmg_features,hmg_inputs,align_motif_graphs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--paper',required=True,choices=['HMG-MRL','HimNet']);p.add_argument('--paper-root',required=True,type=Path)
    p.add_argument('--fullgraph',action='store_true',help='Clean fullgraph conditional encoder then motif pooling')
    p.add_argument('--fixed-topology',action='store_true',help='Motif-independent attribute diffusion; motif-only residual injection')
    p.add_argument('--encoder-lr',type=float,default=None)
    p.add_argument('--fast-fullgraph',action='store_true',help='Vectorized condition and readout; same parameters and batching')
    p.add_argument('--fusion',choices=['replace','residual'],default='replace')
    p.add_argument('--gate-init',type=float,default=.1)
    p.add_argument('--data',required=True,type=Path);p.add_argument('--checkpoint',type=Path)
    p.add_argument('--output',required=True,type=Path);p.add_argument('--hmg-feature-cache',type=Path)
    p.add_argument('--him-graph-cache',type=Path);p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--arm',choices=['pretrained','random','original'],default='pretrained')
    p.add_argument('--freeze-pretrained',action='store_true');p.add_argument('--epochs',type=int,default=100)
    p.add_argument('--patience',type=int,default=20);p.add_argument('--batch-size',type=int,default=16)
    p.add_argument('--motif-batch-size',type=int,default=32);p.add_argument('--pair-budget',type=int,default=0)
    p.add_argument('--seed',type=int,default=0);p.add_argument('--lr',type=float,default=1e-4)
    p.add_argument('--aux-weight',type=float,default=None);p.add_argument('--hidden',type=int,default=None)
    p.add_argument('--smoke-per-split',type=int,default=0,help='Real-data implementation check only; first N per split')
    p.add_argument('--device',default='cuda:0' if torch.cuda.is_available() else 'cpu')
    args=p.parse_args();torch.manual_seed(args.seed);torch.set_num_threads(4)
    if args.lr<=0 or (args.encoder_lr is not None and args.encoder_lr<=0):raise ValueError('Learning rates must be positive')
    if args.fixed_topology:
        if args.fullgraph or args.fast_fullgraph:raise ValueError('Choose exactly one encoder family')
        if args.fusion!='residual':raise ValueError('Fixed topology first version requires --fusion residual')
        if args.aux_weight not in (None,0):raise ValueError('First fixed-topology version has no auxiliary objective')
        args.aux_weight=0.
    if args.arm=='random' and not args.fixed_topology:raise ValueError('Random control is implemented for --fixed-topology')
    if args.fast_fullgraph and not args.fullgraph:raise ValueError('--fast-fullgraph requires --fullgraph')
    if args.epochs<1 or args.batch_size<2:raise ValueError('Need epochs>=1, batch-size>=2 (paper BatchNorm)')
    if args.paper=='HMG-MRL' and not args.hmg_feature_cache:raise ValueError('Require --hmg-feature-cache from original HMG preprocessing; no substitute features')
    if args.arm in ('pretrained','random') and not args.checkpoint and not args.prepare_only:raise ValueError('Require checkpoint for weights/config provenance')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'best.pt').exists():raise FileExistsError('Choose a new output directory')
    code=code_identity();upstream=source_inventory(args.paper_root);data=load_dataset(args.data)
    splits={s:[m for m in data['molecules'] if m['split']==s] for s in ('train','valid','test')}
    if args.smoke_per_split:
        splits={s:rows[:args.smoke_per_split] for s,rows in splits.items()}
    if any(len(x)<2 for x in splits.values()):raise ValueError('Need at least two real molecules per split')
    selected=[m for rows in splits.values() for m in rows]
    if not bool(torch.isfinite(torch.tensor([m['label'] for m in selected])).all()):
        raise FloatingPointError('Non-finite molecule target labels')
    cls,loader,cfg=load_paper(args.paper_root,args.paper,args.device,args.hidden)
    denoiser=ck=None
    if not args.prepare_only and args.arm in ('pretrained','random'):
        if args.fixed_topology:
            from .fixed_topology import load_encoder as load_fixed
            torch.manual_seed(args.seed+1000)
            denoiser,ck=load_fixed(args.checkpoint,args.device,random=args.arm=='random')
        elif args.fullgraph:
            if args.fast_fullgraph:
                from .fullgraph_fast import load_encoder as load_fullgraph
            else:
                from .fullgraph import load_encoder as load_fullgraph
            denoiser,ck=load_fullgraph(args.checkpoint,args.device)
        else:
            denoiser,ck=load_encoder(args.checkpoint,args.device)
        if ck['data_sha256']!=sha(args.data):raise ValueError('Pretraining data provenance mismatch')
        if ck['method'] not in ('both',args.paper):raise ValueError('Pretrained extraction family mismatch')
        if ck['smoke'] and not args.smoke_per_split:raise ValueError('Smoke pretrained weights cannot be used for a formal comparison')
    provenance={'data_sha256':sha(args.data),'paper':args.paper,'paper_root':str(args.paper_root.resolve()),
                'module_source_sha256':code,'paper_source_sha256':upstream,
                'rows':{s:[m['row'] for m in rows] for s,rows in splits.items()},
                'arm':args.arm,'task':'Lipo-style single-target regression',
                'original_model_class':True,'standalone_context_model':False,
                'smoke':bool(args.smoke_per_split or data['smoke'] or (ck and ck['smoke'])),
                'args':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}}
    if args.paper=='HimNet':
        if args.him_graph_cache:
            cache=torch.load(args.him_graph_cache,map_location='cpu',weights_only=True)
            if cache['data_sha256']!=sha(args.data) or cache['paper_source_sha256']!=upstream:raise ValueError('HimNet graph cache provenance mismatch')
            graphs=cache['graphs']
            if any(m['row'] not in graphs for m in selected):raise ValueError('HimNet graph cache missing selected rows')
        else:
            graphs=prepare_him_graphs(selected,loader)
            torch.save({'graphs':graphs,'data_sha256':sha(args.data),'paper_source_sha256':upstream},args.output/'him_graphs.pt')
        internal,token_positions,alignment=align_motif_graphs(data,selected,args.paper,him_graphs=graphs)
        if args.fixed_topology and denoiser:
            from .fixed_topology import records
            full_records={r['row']:r for r in records(data,selected,args.paper,internal)}
        elif args.fullgraph and denoiser:
            from .fullgraph import records
            full_records={r['row']:r for r in records(data,selected,args.paper,denoiser.vocab,internal)}
        sample=graphs[selected[0]['row']]
        ad=len(sample['f_atoms'][0]);bd=len(sample['f_bonds'][0]);fd=sample['fp_x'].shape[-1]
        torch.manual_seed(args.seed+2000)
        backbone=cls('in_memory',ad,bd,fd,hidden_size=args.hidden or 256,device=args.device,out_dim=1).to(args.device)
        model=HimNetAdapter(backbone,denoiser,args.freeze_pretrained,args.motif_batch_size,args.pair_budget,args.fusion,args.gate_init).to(args.device) if denoiser else backbone
        def forward(rows,training):
            items=[graphs[m['row']] for m in rows];batch=HierarchicalBatch(items)
            if denoiser:
                mb=[g for m in rows for g in internal[m['row']]]
                if args.fullgraph or args.fixed_topology: mb=[full_records[m['row']] for m in rows]
                positions=[start+atoms+j for m,(start,_),atoms in zip(rows,batch.a_scope,batch.mol_atom_num) for j in token_positions[m['row']]]
                if not mb:mb=None
                pred=model(batch,mb,positions,training=training)
            else:
                # Same original forward branches, bypassing only disk lookup.
                ligand=backbone.encoder(batch)
                if training:
                    fp,proj,views=backbone.fp_encoder(batch.fp_x.to(args.device).float(),training=True)
                    backbone.contrastive_data={'graph_proj':torch.nn.functional.normalize(backbone.graph_proj_head(ligand),dim=1),
                                               'fp_proj':proj,'fp_view_projs':views}
                else:fp=backbone.fp_encoder(batch.fp_x.to(args.device).float())
                pred=backbone.mlp(backbone.feature_fusion(torch.stack([ligand,fp])))
            aux=model.compute_contrastive_loss() if training else pred.sum()*0
            return pred.reshape(-1),aux
        aux_weight=.2 if args.aux_weight is None else args.aux_weight
    else:
        features,assemble=load_hmg_features(args.hmg_feature_cache,args.paper_root)
        internal,token_positions,alignment=align_motif_graphs(data,selected,args.paper,hmg_features=features)
        if args.fixed_topology and denoiser:
            from .fixed_topology import records
            full_records={r['row']:r for r in records(data,selected,args.paper,internal)}
        elif args.fullgraph and denoiser:
            from .fullgraph import records
            full_records={r['row']:r for r in records(data,selected,args.paper,denoiser.vocab,internal)}
        # Check all selected real records up front; no silent preprocessing drop.
        for start in range(0,len(selected),args.batch_size):hmg_inputs(selected[start:start+args.batch_size],features,assemble,args.device)
        sample,_,_=hmg_inputs(selected[:2],features,assemble,args.device)
        cfg.atom_dim_in=sample['x_atoms'].shape[-1];cfg.bond_dim_in=sample['x_bonds'].shape[-1]
        torch.manual_seed(args.seed+2000)
        backbone=cls(cfg.atom_dim_in,cfg.bond_dim_in,1).to(args.device)
        model=HMGMRLAdapter(backbone,denoiser,args.freeze_pretrained,args.motif_batch_size,args.pair_budget,args.fusion,args.gate_init).to(args.device) if denoiser else backbone
        def forward(rows,training):
            inputs,mols,maps=hmg_inputs(rows,features,assemble,args.device)
            if denoiser:
                mb=[g for m in rows for g in internal[m['row']]]
                if args.fullgraph or args.fixed_topology: mb=[full_records[m['row']] for m in rows]
                positions=[[i,j] for i,m in enumerate(rows) for j in token_positions[m['row']]]
                if not mb:mb=None
                out=model(mb,positions,**inputs)
            else:out=backbone(**inputs)
            return out[1].reshape(-1),out[2]
        aux_weight=cfg.ct_loss_coff if args.aux_weight is None else args.aux_weight
        provenance['original_feature_cache_sha256']=sha(args.hmg_feature_cache)
        provenance['original_3d_features_retained']=True
    if args.fullgraph:
        alignment['input']='clean full molecular graph + unnoised motif type conditions; mean member pooling'
    if args.fixed_topology:
        alignment['input']='clean fixed topology; no motif conditions; member mean; lowest trained time t=1'
        provenance['auxiliary_objectives_enabled']=False
        provenance['readout_note']='member mean, preserving original membership/order; not claimed identical to original feature construction'
    save_json(args.output/'motif_alignment.json',alignment);provenance['motif_alignment']=alignment
    initial={k.replace('frag_input_projection.base.','frag_input_projection.'):v.detach().cpu().clone()
             for k,v in backbone.state_dict().items() if 'motif_projection' not in k and 'motif_gate' not in k}
    torch.save(initial,args.output/'initial_backbone.pt')
    if ck:
        provenance['encoder_config_checkpoint_sha256']=sha(args.checkpoint)
        provenance['encoder_weights_loaded']=args.arm=='pretrained'
        if args.arm=='pretrained':
            provenance['pretrained_epoch']=ck['epoch'];provenance['pretrained_sha256']=sha(args.checkpoint)
    save_json(args.output/'manifest.json',provenance)
    if args.prepare_only:
        print('Original paper real-data inputs checked; prepare-only, no training');return
    y=torch.tensor([m['label'] for m in splits['train']]);mean=float(y.mean());std=float(y.std(unbiased=False).clamp_min(1e-6))
    encoder_ids={id(p) for p in denoiser.parameters()} if denoiser else set()
    groups=[{'params':[p for p in model.parameters() if p.requires_grad and id(p) not in encoder_ids],'lr':args.lr}]
    if denoiser and not args.freeze_pretrained:
        groups.append({'params':[p for p in denoiser.parameters() if p.requires_grad],
                       'lr':args.encoder_lr if args.encoder_lr is not None else (args.lr*.1 if args.fixed_topology else args.lr)})
    optimizer=torch.optim.AdamW(groups,weight_decay=1e-5)
    torch.manual_seed(args.seed+4000)
    n=len(splits['train']);train_batch=args.batch_size
    # Do not discard a singleton remainder: merge it into the preceding batch.
    def train_batches(epoch):
        order=torch.randperm(n,generator=torch.Generator().manual_seed(args.seed+epoch)).tolist()
        groups=[order[i:i+train_batch] for i in range(0,n,train_batch)]
        if len(groups)>1 and len(groups[-1])==1:
            last=groups.pop();groups[-1].extend(last)
        return [[splits['train'][i] for i in group] for group in groups]
    def fail_nonfinite(value,rows,phase,field):
        if not bool(torch.isfinite(value).all()):
            report={'phase':phase,'field':field,'epoch':epoch,'rows':[m['row'] for m in rows]}
            save_json(args.output/'nonfinite_error.json',report)
            raise FloatingPointError(str(report))
    epoch=0
    def evaluate(rows,phase):
        model.eval();pred=[];ys=[];ids=[]
        with torch.no_grad():
            for i in range(0,len(rows),args.batch_size):
                batch=rows[i:i+args.batch_size];out,_=forward(batch,False)
                fail_nonfinite(out,batch,phase,'prediction')
                pred.extend((out*std+mean).cpu().tolist());ys.extend(m['label'] for m in batch);ids.extend(m['row'] for m in batch)
        score=float((torch.tensor(pred)-torch.tensor(ys)).square().mean().sqrt())
        fail_nonfinite(torch.tensor(score),rows,phase,'RMSE')
        return score,{'prediction':pred,'target':ys,'row_ids':ids}
    history=[];best=float('inf');best_epoch=0
    for epoch in range(1,args.epochs+1):
        started=time.perf_counter();model.train()
        batches=train_batches(epoch)
        for batch_index,rows in enumerate(batches):
            target=torch.tensor([m['label'] for m in rows],device=args.device)
            pred,aux=forward(rows,True);loss=(pred-(target-mean)/std).square().mean()+aux_weight*aux
            fail_nonfinite(pred,rows,'train','prediction');fail_nonfinite(aux,rows,'train','auxiliary_loss')
            fail_nonfinite(loss,rows,'train','objective')
            optimizer.zero_grad(set_to_none=True);loss.backward()
            try:torch.nn.utils.clip_grad_norm_(model.parameters(),5,error_if_nonfinite=True)
            except RuntimeError:
                save_json(args.output/'nonfinite_error.json',{'phase':'train','field':'gradient','epoch':epoch,'rows':[m['row'] for m in rows]})
                raise
            optimizer.step()
            if (batch_index+1)%25==0:save_json(args.output/'progress.json',{'epoch':epoch,'batch':batch_index+1,'batches':len(batches),'phase':'train'})
        score,_=evaluate(splits['valid'],'valid');history.append({'epoch':epoch,'valid_rmse':score,'seconds':time.perf_counter()-started})
        if score<best:
            best=score;best_epoch=epoch
            torch.save({'model':{k:v.detach().cpu().clone() for k,v in model.state_dict().items()},'target_mean':mean,'target_std':std,
                        'epoch':epoch,'paper':args.paper,'denoiser_config':denoiser.config if denoiser else None,
                        'manifest':provenance},args.output/'best.pt')
        save_json(args.output/'history.json',history);print(history[-1],flush=True)
        save_json(args.output/'progress.json',{'phase':'epoch_complete',**history[-1]})
        if epoch-best_epoch>=args.patience:break
    if not best_epoch:raise RuntimeError('No finite validation checkpoint was selected; test evaluation cancelled')
    model.load_state_dict(torch.load(args.output/'best.pt',map_location='cpu',weights_only=True)['model'],strict=True)
    test,pred=evaluate(splits['test'],'test');save_json(args.output/'test_predictions.json',pred)
    if code_identity()!=code or source_inventory(args.paper_root)!=upstream:raise RuntimeError('Source changed during run')
    result={'paper':args.paper,'arm':args.arm,'original_model_class':True,'standalone_context_model':False,
            'valid_rmse':best,'test_rmse':test,'best_epoch':best_epoch,'smoke':provenance['smoke'],
            'formal_performance_claim':False,'test_evaluated_after_selection':True,
            'pretrained_sha256':sha(args.checkpoint) if denoiser and args.arm=='pretrained' else None}
    save_json(args.output/'results.json',result);print(result)

if __name__=='__main__':main()
