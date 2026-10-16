"""Run the fixed Lipo paired experiment from an immutable source snapshot."""
import json
import argparse
import hashlib
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]

def checkpoint_info(checkpoint):
    import torch
    ck=torch.load(checkpoint,map_location='cpu',weights_only=True)
    if ck.get('smoke',True):raise ValueError('Formal pairing requires non-smoke pretrained weights')
    history_path=checkpoint.parent/'history.json'
    if not history_path.exists():raise ValueError('Require pretraining history.json beside checkpoint')
    history=json.loads(history_path.read_text(encoding='utf-8-sig'))
    best=min(history,key=lambda item:item['valid_loss'])
    if ck['epoch']!=best['epoch'] or abs(ck['valid_loss']-best['valid_loss'])>1e-7:
        raise ValueError('Checkpoint is not the validation-best model in its pretraining history')
    digest=hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    supervision=ck.get('structure_supervision',{'weight':0.,'definition':None})
    dimension=ck['config'].get('structure_dim',0)
    if supervision['weight']>0 and (not dimension or not supervision.get('definition')):
        raise ValueError('Positive structure supervision requires head config and label definitions')
    if dimension and supervision['weight']<=0:raise ValueError('Structure head has no positive supervision provenance')
    return json.loads(json.dumps(dict(checkpoint=str(checkpoint),checkpoint_sha256=digest,checkpoint_epoch=ck['epoch'],
                checkpoint_valid_loss=ck['valid_loss'],structure_supervision=supervision)))

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def worker(options):
    import torch
    RUN=options.output;ROOT=options.workspace_root
    source = RUN / 'source'
    checkpoint=options.checkpoint;info=checkpoint_info(checkpoint)
    expected=json.loads((RUN/'launch.json').read_text(encoding='utf-8'))['checkpoint_info']
    if info!=expected:raise ValueError('Checkpoint changed after snapshot preparation')
    save(RUN/'protocol.json', dict(seed=0, epochs=100, patience=20, batch_size=16,
         lr=1e-4, joint_finetuning=True, **info, split='existing Lipo scaffold 3360/420/420',
         test='once after validation selection', started=datetime.now(timezone.utc).isoformat()))
    active = {}
    handles = []
    def start(name, module, args):
        out=(RUN/(name+'.stdout.log')).open('w',encoding='utf-8')
        err=(RUN/(name+'.stderr.log')).open('w',encoding='utf-8');handles.extend([out,err])
        cmd=[sys.executable,'-X','utf8','-u','-m',module,*map(str,args)]
        process=subprocess.Popen(cmd,cwd=source,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        active[name]=process
        save(RUN/(name+'.process.json'),dict(pid=process.pid,command=cmd,started=time.time()))
        return process
    def finish(name, process):
        code=process.wait();save(RUN/(name+'.exit.json'),dict(exit_code=code,ended=time.time()))
        if code:raise RuntimeError(f'{name} failed, exit {code}; inspect stderr log')
    data=ROOT/'motif/data/lipo.pt'
    hmgroot=ROOT/'research_code_audit/HMG-MRL'
    himroot=ROOT/'coarse_graining1/HimNet_reproduction_20261013/windows'
    prep=start('hmg_features','motif.prepare_hmg',[
        '--data',data,'--paper-root',hmgroot,'--output',RUN/'hmg_features',
        '--dependency-path',ROOT/'motif/vendor/deepchem_runtime']) if 'HMG-MRL' in options.papers and not options.hmg_feature_cache else None
    common=['--data',data,'--epochs','100','--patience','20','--batch-size','16',
            '--motif-batch-size','32','--pair-budget','262144','--seed','0','--lr','0.0001']
    results=[]
    try:
        for paper, paperroot in [('HimNet',himroot),('HMG-MRL',hmgroot)]:
            if paper not in options.papers:continue
            if paper=='HMG-MRL' and prep is not None:finish('hmg_features',prep)
            for arm in ['original','pretrained']:
                name=('him' if paper=='HimNet' else 'hmg')+'_'+arm
                args=['--paper',paper,'--paper-root',paperroot,'--arm',arm,'--output',RUN/name,*common]
                if arm=='pretrained':args+=['--checkpoint',checkpoint]
                if paper=='HimNet' and arm=='pretrained':args+=['--him-graph-cache',RUN/'him_original/him_graphs.pt']
                if paper=='HMG-MRL':args+=['--hmg-feature-cache',options.hmg_feature_cache or RUN/'hmg_features/features.pickle']
                finish(name,start(name,'motif.train_paper',args))
                results.append(json.loads((RUN/name/'results.json').read_text(encoding='utf-8')))
                save(RUN/'completed_results.json',results)
        comparisons=[]
        for prefix in ['him','hmg']:
            if ('HimNet' if prefix=='him' else 'HMG-MRL') not in options.papers:continue
            a=RUN/(prefix+'_original');b=RUN/(prefix+'_pretrained')
            sa=torch.load(a/'initial_backbone.pt',weights_only=True)
            sb=torch.load(b/'initial_backbone.pt',weights_only=True)
            assert sa.keys()==sb.keys() and all(torch.equal(sa[k],sb[k]) for k in sa), 'Unpaired backbone initialization'
            ra=json.loads((a/'results.json').read_text());rb=json.loads((b/'results.json').read_text())
            comparisons.append(dict(paper=ra['paper'],original=ra,pretrained=rb,
                test_rmse_difference=rb['test_rmse']-ra['test_rmse'],initial_backbone_equal=True))
        save(RUN/'comparison.json',dict(comparisons=comparisons,single_seed=True))
        save(RUN/'exit.json',dict(success=True,ended=time.time()))
    except Exception as exc:
        save(RUN/'exit.json',dict(success=False,error=str(exc),ended=time.time()))
        raise
    finally:
        for h in handles:h.close()

def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True,help='New experiment directory; never overwrite existing runs')
    parser.add_argument('--workspace-root',type=Path,default=ROOT,help='Workspace containing motif data and original paper sources')
    parser.add_argument('--prepare-only',action='store_true',help='Create and validate snapshot without starting training')
    parser.add_argument('--papers',nargs='+',choices=['HimNet','HMG-MRL'],default=['HimNet','HMG-MRL'])
    parser.add_argument('--hmg-feature-cache',type=Path,help='Reuse an existing original feature cache; completeness is still checked')
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    options=parser.parse_args(argv)
    for name in ('checkpoint','output','workspace_root'):setattr(options,name,getattr(options,name).resolve())
    if options.hmg_feature_cache:options.hmg_feature_cache=options.hmg_feature_cache.resolve()
    if options.worker:
        worker(options);return
    info=checkpoint_info(options.checkpoint)
    options.output.mkdir(parents=True,exist_ok=False)
    target=options.output/'source/motif';target.mkdir(parents=True)
    for path in Path(__file__).parent.glob('*.py'):shutil.copy2(path,target/path.name)
    command=[sys.executable,'-X','utf8','-u','-m','motif.launch_paper_pair','--worker',
             '--checkpoint',str(options.checkpoint),'--output',str(options.output),
             '--workspace-root',str(options.workspace_root)]
    command+=['--papers',*options.papers]
    if options.hmg_feature_cache:command+=['--hmg-feature-cache',str(options.hmg_feature_cache)]
    save(options.output/'launch.json',dict(checkpoint_info=info,worker_command=command,
         worker_cwd=str(target.parent),prepare_only=options.prepare_only))
    print(json.dumps(info,ensure_ascii=False),flush=True)
    if options.prepare_only:
        print('Snapshot prepared; worker command and cwd are recorded in launch.json',flush=True);return
    # Default invocation really runs the immutable snapshot; no hidden second invocation.
    subprocess.run(command,cwd=target.parent,check=True)

if __name__=='__main__':main()
