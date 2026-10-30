"""Audit fixed architecture, source states, supervision counts and completed runs."""
import hashlib
import json
from pathlib import Path
import subprocess
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'motif/adjacency_diffusion/runs/masked_property_comparison_20261029'
status=json.loads((run/'status.json').read_text(encoding='utf-8'))
assert status.get('state')=='complete' and status['completed_property_jobs']==9 and status['completed_pretrain_jobs']==6
plan=json.loads((run/'plan.json').read_text(encoding='utf-8'))
assert plan['epochs']==60 and plan['patience']==60
assert plan['input_channels']==18 and plan['target_channels']==17
plan_hash=hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()
for name in ('masked_bonds.py','typed_bonds.py','typed_train.py','typed_refined.py','model.py','data.py','diffusion.py'):
    path='motif/adjacency_diffusion/'+name
    original=subprocess.check_output(['git','-C',str(ROOT/'coarse_graining1'),'show','7c1cc1ed0d4ee678adca992fff2fdf9a35e550b8:'+path]).decode('utf-8-sig')
    actual=(run/'frozen_source'/path).read_text(encoding='utf-8-sig')
    assert original==actual,'Model changed relative to audited 7c1cc1e: '+name
sources=json.loads((run/'pretrain_sources.json').read_text(encoding='utf-8'))
recoveries=[]
for seed,arms in sources.items():
    assert arms['unmasked']['initial_state_hash']==arms['masked']['initial_state_hash']
    counts={}
    for arm,item in arms.items():
        path=Path(item['path'])
        assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        completion=json.loads((path.parent/'completion.json').read_text(encoding='utf-8'))
        assert completion['checkpoint_replay_passed']
        counts[arm]=completion['supervised_candidate_pairs_accumulated']
        for timestep,metrics in completion['recovery']['confirmation']['by_timestep'].items():
            recoveries.append({'pretrain_seed':int(seed),'arm':arm,'timestep':int(timestep),
                'all_pair_presence_AP':metrics['presence_AP'],
                'all_pair_true_bond_type_accuracy':metrics['type_accuracy_true_bonds'],
                'masked_recovery':metrics['masked_recovery'],
                'supervised_pairs_accumulated':completion['supervised_candidate_pairs_accumulated'],
                'supervised_true_bonds_accumulated':completion['supervised_true_bonds_accumulated'],
                'source':str((path.parent/'completion.json').relative_to(ROOT))})
    assert counts['masked']<counts['unmasked']
results=json.loads((run/'results.json').read_text(encoding='utf-8'))
assert len(results['runs'])==9
for seed in (0,1,2):
    rows=[r for r in results['runs'] if r['task_seed']==seed]
    assert {r['arm'] for r in rows}=={'scratch','unmasked','masked'}
    assert len({r['head_initial_hash'] for r in rows})==1
    assert len({r['encoder_initial_hash'] for r in rows})==1
    configs=[]
    for r in rows:
        assert r['epochs_run']==60 and r['checkpoint_replay_passed']
        configs.append(json.loads((run/f'{r["arm"]}_task_seed_{seed}'/'config.json').read_text(encoding='utf-8')))
    assert len({json.dumps(c['config'],sort_keys=True) for c in configs})==1
    assert len({json.dumps(c['splits'],sort_keys=True) for c in configs})==1
    assert all(c['run_plan_sha256']==plan_hash for c in configs)
(run/'recovery_disclosure.json').write_text(json.dumps({'rows':recoveries,
    'warning':'Different objectives and target counts; do not rank arms by total recovery loss.',
    'context_dependence_proven':False,'test_evaluated':False},indent=2),encoding='utf-8')
acceptance={'state':'complete','model_code_matches_7c1cc1e':True,'same_18_channel_architecture':True,
    'pretrain_initialization_paired':True,'property_initialization_paired':True,'all_property_runs_60_epochs':True,
    'all_checkpoint_replays_passed':True,'masked_target_counts_disclosed':True,
    'context_dependence_proven':False,'test_evaluated':False}
(run/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
print(json.dumps({'aggregate':results['aggregate'],'paired_deltas':results['paired_deltas'],'acceptance':acceptance}))
