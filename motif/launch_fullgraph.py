"""Immutable, sequential seed-0 fullgraph experiment with failure records."""
import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]

def save(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding='utf-8')

def worker(run,root):
    import torch
    launch=json.loads((run/'launch.json').read_text(encoding='utf-8'))
    # psutil tests identity as well as PID, avoiding waits on a reused PID.
    import psutil
    for pid in launch['wait_pids']:
        try:
            process=psutil.Process(pid)
            if 'python' not in process.name().lower():continue
            while process.is_running():
                save(run/'status.json',{'phase':'waiting_for_existing_training','pid':pid,'updated':time.time()})
                try:process.wait(timeout=30);break
                except psutil.TimeoutExpired:pass
        except psutil.NoSuchProcess:pass
    completed=[]
    def execute(name,module,args):
        cmd=[sys.executable,'-X','utf8','-u','-m',module,*map(str,args)]
        with (run/(name+'.stdout.log')).open('w',encoding='utf-8') as out,(run/(name+'.stderr.log')).open('w',encoding='utf-8') as err:
            process=subprocess.Popen(cmd,cwd=run/'source',stdout=out,stderr=err,
                creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
            save(run/(name+'.process.json'),{'pid':process.pid,'command':cmd,'started':time.time()})
            save(run/'status.json',{'phase':'running','stage':name,'pid':process.pid,'updated':time.time()})
            code=process.wait()
        save(run/(name+'.exit.json'),{'exit_code':code,'ended':time.time()})
        if code:raise RuntimeError(f'{name} failed ({code}); see stderr log')
        completed.append(name)
        save(run/'completed.json',completed)
    try:
        data=root/'motif/data/lipo.pt'
        for paper,prefix,paperroot in [('HimNet','him',root/'coarse_graining1/HimNet_reproduction_20261013/windows'),
                                      ('HMG-MRL','hmg',root/'research_code_audit/HMG-MRL')]:
            pretrain=run/(prefix+'_pretrain')
            execute(prefix+'_pretrain','motif.pretrain_fullgraph',['--data',data,'--output',pretrain,'--method',paper,
                '--epochs',50,'--batch-size',4,'--hidden',128,'--layers',3,'--seed',0])
            ck=torch.load(pretrain/'best.pt',map_location='cpu',weights_only=True)
            if ck['smoke']:raise ValueError('Formal experiment received smoke checkpoint')
            for arm in ('original','pretrained'):
                args=['--paper',paper,'--paper-root',paperroot,'--data',data,'--output',run/(prefix+'_'+arm),
                    '--arm',arm,'--epochs',100,'--patience',20,'--batch-size',16,'--motif-batch-size',4,
                    '--pair-budget',262144,'--seed',0,'--lr',0.0001]
                if arm=='pretrained':args+=['--fullgraph','--checkpoint',pretrain/'best.pt']
                if paper=='HimNet' and arm=='pretrained':args+=['--him-graph-cache',run/'him_original/him_graphs.pt']
                if paper=='HMG-MRL':args+=['--hmg-feature-cache',root/'motif/runs/hmg_features_finite_20261016/features.pickle']
                execute(prefix+'_'+arm,'motif.train_paper',args)
        comparisons=[]
        for prefix in ('him','hmg'):
            paths=[run/(prefix+'_'+a) for a in ('original','pretrained')]
            states=[torch.load(p/'initial_backbone.pt',map_location='cpu',weights_only=True) for p in paths]
            if states[0].keys()!=states[1].keys() or not all(torch.equal(states[0][k],states[1][k]) for k in states[0]):
                raise ValueError('Unpaired backbone initialization')
            results=[json.loads((p/'results.json').read_text(encoding='utf-8')) for p in paths]
            comparisons.append({'paper':results[0]['paper'],'original':results[0],'fullgraph_pretrained':results[1],
                'test_rmse_difference':results[1]['test_rmse']-results[0]['test_rmse'],'initial_backbone_equal':True})
        save(run/'comparison.json',{'comparisons':comparisons,'single_seed':True})
        save(run/'status.json',{'phase':'complete','updated':time.time()})
    except Exception as exc:
        save(run/'status.json',{'phase':'failed','error':str(exc),'updated':time.time()});raise

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--root',type=Path,default=ROOT)
    p.add_argument('--wait-pids',nargs='*',type=int,default=[]);p.add_argument('--worker',action='store_true')
    args=p.parse_args();run=args.output.resolve();root=args.root.resolve()
    if args.worker:worker(run,root);return
    run.mkdir(parents=True,exist_ok=False)
    source=run/'source';(source/'motif').mkdir(parents=True)
    for f in (root/'motif').glob('*.py'):shutil.copy2(f,source/'motif'/f.name)
    shutil.copytree(root/'扩散GNN/01_扩散预训练/bond_diffusion',source/'扩散GNN/01_扩散预训练/bond_diffusion',ignore=shutil.ignore_patterns('__pycache__'))
    save(run/'launch.json',{'seed':0,'pretrain_epochs':50,'property_epochs':100,'patience':20,
        'wait_pids':args.wait_pids,'root':str(root),'source':str(source),'single_seed':True,'started':time.time()})
    command=[sys.executable,'-X','utf8','-u','-m','motif.launch_fullgraph','--worker','--root',str(root),'--output',str(run)]
    with (run/'worker.stdout.log').open('w',encoding='utf-8') as out,(run/'worker.stderr.log').open('w',encoding='utf-8') as err:
        process=subprocess.Popen(command,cwd=source,stdout=out,stderr=err,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    save(run/'worker.json',{'pid':process.pid,'command':command})
    print(json.dumps({'pid':process.pid,'output':str(run)},ensure_ascii=False),flush=True)

if __name__=='__main__':main()
