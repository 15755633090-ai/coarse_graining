"""Validate and report every candidate, including unsuccessful configurations."""
import hashlib,json,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
out=ROOT/'motif/adjacency_diffusion/runs/transfer_revision_20261030'
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
status=read(out/'status.json');assert status['state']=='complete'
plan=read(out/'plan.json');result=read(out/'results.json');assert len(result['runs'])==10
for name,h in plan['source_sha256'].items():
    assert hashlib.sha256((out/'frozen_source'/name).read_bytes()).hexdigest()==h,name
# Execution uses the frozen tree; later optional revisions may exist in ROOT.
for row in result['runs']:
    folder=out/f"{row['phase']}_{row['candidate']}_{row['arm']}_seed_{row['seed']}"
    h=read(folder/'history.json');assert len(h)==60 and h[-1]['epoch']==60
    assert row['checkpoint_replay_passed']
    write(folder/'status.json',{'state':'complete',**row})
    assert min(r['validation']['rmse'] for r in h)==row['validation']['rmse']
    config=read(folder/'config.json')
    assert config['initial']==row['initial']
    assert config['settings']==plan['candidates'][row['candidate']]
for name in plan['candidates']:
    pair=[r for r in result['runs'] if r['phase']=='screen' and r['candidate']==name]
    assert len(pair)==2 and pair[0]['initial']==pair[1]['initial']
old=Path(plan['old_run']);old_sources=read(old/'pretrain_sources.json');old_plan=read(old/'plan.json')
for item in read(out/'pretraining_sources.json'):
    p=Path(item['path']);assert hashlib.sha256(p.read_bytes()).hexdigest()==item['sha256']
    preseed=int(p.parent.name.rsplit('_',1)[1])
    assert item['initial_state_hash']==old_sources[str(preseed)]['masked']['initial_state_hash']
    assert item['checkpoint_replay_passed']
selected=result['selected'];rep=[r for r in result['runs'] if r['phase']=='replication']
rep_aggregate={}
for arm in ('scratch','masked'):
    vals=[r['validation']['rmse'] for r in rep if r['arm']==arm]
    rep_aggregate[arm]={'n':len(vals),'mean_rmse':statistics.mean(vals),'sample_sd':statistics.stdev(vals)}
result['replication_only']=rep_aggregate;write(out/'results.json',result)
accept={'state':'complete','property_jobs':10,'all_60_epochs':True,'all_best_checkpoints_replayed':True,
        'paired_initialization_verified':True,'mass_pretraining_initialization_matches_original_masked':True,
        'frozen_source_hashes_match':True,'test_evaluated':False,'selected':selected,
        'interpretation':'screen seed0 selected; seeds1,2 are seed replication on reused ESOL validation, not untouched tests'}
write(out/'acceptance.json',accept)
lines=['# Bounded transfer revision results','','All validation RMSE values below use60 full epochs; lower is better.','',
       '| phase | candidate | seed | scratch | masked | masked minus scratch |','|---|---|---:|---:|---:|---:|']
keys=sorted({(r['phase'],r['candidate'],r['seed']) for r in result['runs']})
for phase,name,seed in keys:
    p={r['arm']:r['validation']['rmse'] for r in result['runs'] if (r['phase'],r['candidate'],r['seed'])==(phase,name,seed)}
    lines.append(f"| {phase} | {name} | {seed} | {p['scratch']:.5f} | {p['masked']:.5f} | {p['masked']-p['scratch']:+.5f} |")
lines+=['',f'Selected using masked seed0: {selected}.','',
        'Selected aggregate (includes the seed used for selection):',json.dumps(result['aggregate_selected']),
        '', 'Replication seeds1,2 only:',json.dumps(rep_aggregate),'',
        'All candidates are disclosed. This is a small exploratory search on a reused validation split. No test evaluation, no isolated gate or context-causality claim.']
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'selected':selected,'aggregate':result['aggregate_selected'],'replication':rep_aggregate,'paired':result['paired']}))
