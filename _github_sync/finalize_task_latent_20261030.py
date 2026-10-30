"""Audit completed bounded pilot; preserve exact source snapshots separately from Git text."""
import hashlib,json,zipfile
from pathlib import Path
import torch

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'motif/adjacency_diffusion'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')

def main():
    audit={};weights=[]
    folders=['task_latent_smoke_20261030','task_latent_smoke_v2_20261030','task_latent_pilot_20261030']
    archive=BASE/'TASK_LATENT_SOURCE_SNAPSHOTS_20261030.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for name in folders:
            folder=BASE/'runs'/name;plan=read(folder/'plan.json')
            for rel,expected in plan['source_sha256'].items():
                p=folder/'frozen_source'/rel
                assert digest(p)==expected,(name,rel)
                z.write(p,f'{name}/frozen_source/{rel}')
            z.write(folder/'plan.json',f'{name}/plan.json')
            state=read(folder/'status.json')
            if name==folders[0]:
                assert state['state']=='failed_before_training'
                audit[name]={'source_files_verified':len(plan['source_sha256']),'state':state['state']};continue
            assert state['state']=='complete'
            results=read(folder/'results.json');rng={}
            for row in results['runs']:
                arm=folder/f"{row['arm']}_seed_{row['seed']}"
                history=read(arm/'history.json')
                assert len(history)==plan['epochs']
                assert min(h['validation']['rmse'] for h in history)==row['validation']['rmse']
                assert row['replay_passed']
                previous=rng.setdefault(row['seed'],row['noise_rng_sha256']);assert previous==row['noise_rng_sha256']
                best=torch.load(arm/'best.pt',map_location='cpu',weights_only=True)
                latest=torch.load(arm/'latest.pt',map_location='cpu',weights_only=True)
                nested=latest['best_checkpoint']
                assert best['best_score']==nested['best_score']==row['validation']['rmse']
                assert latest['epoch']==plan['epochs']
                assert all(torch.equal(v,nested['model'][k]) for k,v in best['model'].items())
                assert best['run_metadata']['same_original_initial_encoder_and_head']
                for filename in ['best.pt','latest.pt']:
                    p=arm/filename;weights.append({'path':p.relative_to(ROOT).as_posix(),'bytes':p.stat().st_size,'sha256':digest(p),'published':False})
            audit[name]={'source_files_verified':len(plan['source_sha256']),'runs_verified':len(results['runs']),
                'history_minimum_checkpoint_and_replay_consistent':True,'paired_noise_rng':True,'same_original_initialization':True}
    write(BASE/'TASK_LATENT_FINAL_AUDIT_20261030.json',{'checks':audit,'source_archive':{'path':archive.relative_to(ROOT).as_posix(),'sha256':digest(archive)},'local_checkpoints':weights,'test_evaluated':False})
    print(json.dumps(audit))

if __name__=='__main__':main()
