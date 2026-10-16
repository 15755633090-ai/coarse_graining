"""Run the fixed Lipo paired experiment from an immutable source snapshot."""
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'motif/runs/paper_lipo_pair_seed0_20261015'

def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')

def worker():
    import torch
    source = RUN / 'source'
    checkpoint = ROOT / 'motif/runs/pretrain_lipo_both_seed0_fast_20261015/best.pt'
    ck = torch.load(checkpoint, map_location='cpu', weights_only=True)
    history = json.loads((checkpoint.parent / 'history.json').read_text(encoding='utf-8-sig'))
    best = min(history, key=lambda x: x['valid_loss'])
    assert ck['epoch'] == best['epoch'] == 49 and not ck['smoke']
    save(RUN/'protocol.json', dict(seed=0, epochs=100, patience=20, batch_size=16,
         lr=1e-4, joint_finetuning=True, checkpoint_epoch=ck['epoch'],
         checkpoint_valid_loss=ck['valid_loss'], split='existing Lipo scaffold 3360/420/420',
         test='once after validation selection', started=datetime.now(timezone.utc).isoformat()))
    active = {}
    handles = []
    def start(name, module, args):
        out=(RUN/(name+'.stdout.log')).open('w',encoding='utf-8')
        err=(RUN/(name+'.stderr.log')).open('w',encoding='utf-8');handles.extend([out,err])
        cmd=[sys.executable,'-X','utf8','-u','-m',module,*map(str,args)]
        process=subprocess.Popen(cmd,cwd=source,stdout=out,stderr=err,creationflags=subprocess.CREATE_NO_WINDOW)
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
        '--dependency-path',ROOT/'motif/vendor/deepchem_runtime'])
    common=['--data',data,'--epochs','100','--patience','20','--batch-size','16',
            '--motif-batch-size','32','--pair-budget','262144','--seed','0','--lr','0.0001']
    results=[]
    try:
        for paper, paperroot in [('HimNet',himroot),('HMG-MRL',hmgroot)]:
            if paper=='HMG-MRL':finish('hmg_features',prep)
            for arm in ['original','pretrained']:
                name=('him' if paper=='HimNet' else 'hmg')+'_'+arm
                args=['--paper',paper,'--paper-root',paperroot,'--arm',arm,'--output',RUN/name,*common]
                if arm=='pretrained':args+=['--checkpoint',checkpoint]
                if paper=='HimNet' and arm=='pretrained':args+=['--him-graph-cache',RUN/'him_original/him_graphs.pt']
                if paper=='HMG-MRL':args+=['--hmg-feature-cache',RUN/'hmg_features/features.pickle']
                finish(name,start(name,'motif.train_paper',args))
                results.append(json.loads((RUN/name/'results.json').read_text(encoding='utf-8')))
                save(RUN/'completed_results.json',results)
        comparisons=[]
        for prefix in ['him','hmg']:
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

if __name__=='__main__':
    if '--worker' in sys.argv:
        worker()
    else:
        RUN.mkdir(parents=True,exist_ok=False)
        target=RUN/'source/motif';target.mkdir(parents=True)
        for path in (ROOT/'motif').glob('*.py'):shutil.copy2(path,target/path.name)
        print(RUN)
