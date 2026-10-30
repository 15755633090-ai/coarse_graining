"""Prepare reviewable source/results; no model-weight upload."""
import json,hashlib,argparse
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--batch',type=int);a=p.parse_args()
base=ROOT/'motif/adjacency_diffusion'
files=[base/n for n in ['task_latent_denoising.py','test_task_latent_denoising.py','TASK_LATENT_DENOISING_20261030.md','TASK_LATENT_TESTS_20261030.json','TASK_LATENT_FINAL_AUDIT_20261030.json']]
files += [ROOT/'_github_sync'/n for n in ['run_task_latent_pilot_20261030.py','finalize_task_latent_20261030.py','package_task_latent_20261030.py']]
for name in ['task_latent_smoke_20261030','task_latent_smoke_v2_20261030','task_latent_pilot_20261030']:
    folder=base/'runs'/name
    files += list(folder.glob('*.json'))
    if 'pilot' in name:
        for arm in folder.glob('*_seed_*'):files += list(arm.glob('*.json'))
entries=[]
for f in sorted(set(files)):
    content=f.read_text(encoding='utf-8-sig');entries.append({'path':f.relative_to(ROOT).as_posix(),'mode':'100644','type':'blob','content':content})
batches=[];current=[];size=0
for e in entries:
    length=len(json.dumps(e,ensure_ascii=False))
    if current and size+length>29000:batches.append(current);current=[];size=0
    current.append(e);size+=length
if current:batches.append(current)
if a.batch is None:print(json.dumps({'batches':len(batches),'files':len(files),'source_archive_local_only':True}))
else:print(json.dumps(batches[a.batch],ensure_ascii=True))
