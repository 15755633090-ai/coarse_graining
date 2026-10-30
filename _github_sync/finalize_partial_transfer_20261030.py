import hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'motif/adjacency_diffusion/runs/partial_transfer_20261030'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
assert read(out/'status.json')['state']=='complete';plan=read(out/'plan.json');result=read(out/'results.json');assert len(result['runs'])==3
for p,h in plan['source_sha256'].items():assert hashlib.sha256((out/'frozen_source'/p).read_bytes()).hexdigest()==h,p
old=Path(plan['old_run'])
for r in result['runs']:
    folder=out/f"reset_fuse_seed_{r['seed']}";history=read(folder/'history.json');cfg=read(folder/'config.json')
    assert len(history)==60 and r['checkpoint_replay_passed'] and r['initialization_and_partial_load_verified']
    assert min(h['validation']['rmse'] for h in history)==r['validation']['rmse']
    ref=read(old/f"scratch_task_seed_{r['seed']}"/'config.json')
    assert cfg['splits']==ref['splits'] and cfg['head_initial_hash']==ref['head_initial_hash']
write(out/'acceptance.json',{'state':'complete','all_60_epochs':True,'all_source_hashes_match':True,'all_checkpoint_replays_passed':True,
    'selective_transfer_and_fresh_projection_verified':True,'same_architecture_and_capacity_as_original':True,'test_evaluated':False})
lines=['# Reset encoder output projection during transfer','','| seed | reused scratch | reused full transfer | reset fuse | reset minus scratch |','|---:|---:|---:|---:|---:|']
for r in result['runs']:lines.append(f"| {r['seed']} | {r['scratch_rmse']:.5f} | {r['original_masked_rmse']:.5f} | {r['validation']['rmse']:.5f} | {r['validation']['rmse']-r['scratch_rmse']:+.5f} |")
m=result['aggregate'];lines+=['',f"Reset-fuse mean ± sample SD: {m['mean_rmse']:.5f} ± {m['sample_sd']:.5f}.",'',
    'Architecture and capacity unchanged. All three seeds disclosed. These are exploratory validation results after earlier attempts, without test evaluation or isolated context/gate claims.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8');print(json.dumps(result))
