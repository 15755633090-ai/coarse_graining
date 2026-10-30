"""Report posterior correction independently of the earlier bounded search."""
import argparse,hashlib,json
import torch
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
p=argparse.ArgumentParser();p.add_argument('--run-name',choices=('posterior_weighting_20261030','posterior_bounded_20261030'),default='posterior_weighting_20261030');args=p.parse_args()
out=ROOT/'motif/adjacency_diffusion/runs'/args.run_name
def read(p):return json.loads(p.read_text(encoding='utf-8'))
def write(p,x):p.write_text(json.dumps(x,indent=2,ensure_ascii=False)+'\n',encoding='utf-8')
status=read(out/'status.json');assert status['state']=='complete'
plan=read(out/'plan.json');results=read(out/'results.json');assert len(results['runs'])==3
for p,h in plan['source_sha256'].items():assert hashlib.sha256((out/'frozen_source'/p).read_bytes()).hexdigest()==h,p
old=Path(plan['old_run']);oldplan=read(old/'plan.json');recovery=[]
for r in results['runs']:
    seed=r['seed'];preseed=oldplan['task_to_pretrain_seed'][str(seed)]
    folder=out/f'posterior_task_seed_{seed}';history=read(folder/'history.json')
    old_config=read(old/f'scratch_task_seed_{seed}'/'config.json')
    new_config=read(folder/'config.json')
    assert new_config['splits']==old_config['splits']
    assert new_config['head_initial_hash']==old_config['head_initial_hash']
    old_cp=torch.load(old/f'scratch_task_seed_{seed}'/'best.pt',map_location='cpu',weights_only=True)
    new_cp=torch.load(folder/'best.pt',map_location='cpu',weights_only=True)
    assert new_cp['training_max_t']==plan.get('training_max_t',1000)
    for key in ('center','scale'):
        torch.testing.assert_close(old_cp['target_scaler'][key],new_cp['target_scaler'][key],atol=0,rtol=0)
    assert len(history)==60 and r['epochs_run']==60
    assert min(h['validation']['rmse'] for h in history)==r['validation']['rmse']
    assert r['checkpoint_replay_passed'] and r['clean_scratch_equivalence_passed']
    write(folder/'status.json',{'state':'complete',**r})
    r['posterior_minus_scratch']=r['validation']['rmse']-r['scratch_rmse']
    r['posterior_minus_original_masked']=r['validation']['rmse']-r['original_masked_rmse']
    pre=out/'pretraining'/f'posterior_seed_{preseed}';c=read(pre/'completion.json')
    assert c['target_counts_match_original'] and c['checkpoint_replay_passed'] and c['trainable_initialization_matches_original']
    assert hashlib.sha256((pre/'encoder.pt').read_bytes()).hexdigest()==c['checkpoint_sha256']
    before=read(old/'pretraining'/f'masked_seed_{preseed}'/'completion.json')
    for t in ['50','200','500','1000']:
        a=before['recovery']['confirmation']['by_timestep'][t]['masked_recovery']
        b=c['recovery_confirmation']['by_timestep'][t]['masked_recovery']
        assert a['counts']['pairs']==b['counts']['pairs'] and a['counts']['bonds']==b['counts']['bonds']
        recovery.append({'seed':seed,'t':int(t),'original':a,'posterior':b})
write(out/'results.json',results);write(out/'recovery_comparison.json',recovery)
write(out/'acceptance.json',{'state':'complete','property_runs':3,'pretraining_runs':3,'all_60_epochs':True,
    'all_checkpoints_replayed':True,'all_source_hashes_match':True,'equal_supervision_counts_verified':True,
    'clean_scratch_prediction_gradient_equivalence_verified':True,'test_evaluated':False})
lines=[f"# Posterior observation-weighting results (training max t={plan.get('training_max_t',1000)})",'','| seed | reused scratch | original masked | posterior masked | posterior minus scratch |',
       '|---:|---:|---:|---:|---:|']
for r in results['runs']:
    lines.append(f"| {r['seed']} | {r['scratch_rmse']:.5f} | {r['original_masked_rmse']:.5f} | {r['validation']['rmse']:.5f} | {r['posterior_minus_scratch']:+.5f} |")
lines+=['','| arm | validation RMSE mean ± sample SD |','|---|---:|']
for arm,m in results['aggregate'].items():lines.append(f"| {arm} | {m['mean_rmse']:.5f} ± {m['sd']:.5f} |")
lines+=['','Three exploratory seeds on reused ESOL validation. Scratch references are reused with exact clean forward/gradient equivalence checks. No test scores or isolated gate/context claims.',
        '', '| seed | t | original masked AP | posterior masked AP | original true-bond type | posterior true-bond type |',
        '|---:|---:|---:|---:|---:|---:|']
for r in recovery:
    a,b=r['original'],r['posterior']
    lines.append(f"| {r['seed']} | {r['t']} | {a['presence_AP']:.5f} | {b['presence_AP']:.5f} | {a['type_accuracy_true_bonds']:.5f} | {b['type_accuracy_true_bonds']:.5f} |")
(out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
print(json.dumps({'aggregate':results['aggregate'],'paired_deltas':[{'seed':r['seed'],'posterior_minus_scratch':r['posterior_minus_scratch']} for r in results['runs']]}))
