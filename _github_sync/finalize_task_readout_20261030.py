"""Verify matched task-side readout results and keep architecture/SSL gains separate."""
import hashlib,json,statistics,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from motif.adjacency_diffusion.task_readout import load_model
from motif.adjacency_diffusion.masked_bonds import load_model as load_original
out=ROOT/'motif/adjacency_diffusion/runs/task_readout_20261030'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
plan=read(out/'plan.json');result=read(out/'results.json');assert read(out/'status.json')['state']=='complete'
assert len(result['runs'])==6
for p,h in plan['source_sha256'].items():assert hashlib.sha256((out/'frozen_source'/p).read_bytes()).hexdigest()==h,p
old=Path(plan['old_run']);clip=[]
for r in result['runs']:
    folder=out/f"{r['arm']}_seed_{r['seed']}";h=read(folder/'history.json');cfg=read(folder/'config.json')
    assert len(h)==60 and r['checkpoint_replay_passed']
    assert min(x['validation']['rmse'] for x in h)==r['validation']['rmse']
    assert cfg['splits']==read(old/f"scratch_task_seed_{r['seed']}"/'config.json')['splits']
    clip.append({'arm':r['arm'],'seed':r['seed'],'mean_clip_fraction':statistics.mean(x['clip_fraction'] for x in h),
        'mean_pre_clip_norm':statistics.mean(x['pre_clip_norm'] for x in h)})
for seed in [0,1,2]:
    pair=[r for r in result['runs'] if r['seed']==seed];assert pair[0]['initial']==pair[1]['initial']
new,_=load_model(out/'scratch_seed_0/best.pt');new.set_stage('finetune')
original,_=load_original(old/'scratch_task_seed_0/best.pt');original.set_stage('finetune')
new_n=sum(p.numel() for p in new.parameters() if p.requires_grad);old_n=sum(p.numel() for p in original.parameters() if p.requires_grad)
assert new_n-old_n==192
write(out/'acceptance.json',{'state':'complete','all_60_epochs':True,'all_source_hashes_match':True,'all_replays_passed':True,
    'paired_initialization_verified':True,'original_active_parameters':old_n,'new_active_parameters':new_n,'test_evaluated':False})
write(out/'gradient_diagnostics.json',clip)
lines=['# Normalized size-aware readout validation','','| seed | new scratch | new pretrained | pretrained minus scratch |','|---:|---:|---:|---:|']
for seed in [0,1,2]:
    p={r['arm']:r['validation']['rmse'] for r in result['runs'] if r['seed']==seed}
    lines.append(f"| {seed} | {p['scratch']:.5f} | {p['masked']:.5f} | {p['masked']-p['scratch']:+.5f} |")
lines+=['','| arm | RMSE mean ± sample SD |','|---|---:|']
for name,m in result['aggregate'].items():lines.append(f"| {name} | {m['mean_rmse']:.5f} ± {m['sample_sd']:.5f} |")
lines+=['','Old mean-readout references: scratch0.78711 ±0.02804; masked0.81446 ±0.05114.',
    'The readout bundle adds192 parameters to both new arms. Do not credit shared architecture gains to pretraining. All results are exploratory on reused ESOL validation; no test metrics.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'aggregate':result['aggregate'],'paired':result['paired'],'active_parameter_difference':new_n-old_n}))
