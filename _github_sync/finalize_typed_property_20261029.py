"""Audit completion and report all locked three-arm results without retuning."""
import hashlib
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'motif/adjacency_diffusion/runs/typed_property_comparison_20261029'
status=json.loads((run/'status.json').read_text(encoding='utf-8'))
if status.get('state')!='complete' or status.get('completed_property_jobs')!=9:
    raise RuntimeError('Nine completed runs required')
results=json.loads((run/'results.json').read_text(encoding='utf-8'))
plan_hash=hashlib.sha256((run/'plan.json').read_bytes()).hexdigest()
rows=results['runs']
expected={(seed,arm) for seed in (0,1,2) for arm in ('scratch','full','bounded')}
assert {(r['task_seed'],r['arm']) for r in rows}==expected
assert len(rows)==9
for row in rows:
    folder=run/f'{row["arm"]}_task_seed_{row["task_seed"]}'
    config=json.loads((folder/'config.json').read_text(encoding='utf-8'))
    assert config['run_plan_sha256']==plan_hash
    assert config['splits'].keys()=={'train','valid'}
    assert row['checkpoint_replay_passed'] and not row['test_evaluated']
for seed in (0,1,2):
    group=[r for r in rows if r['task_seed']==seed]
    assert len({r['encoder_initial_hash'] for r in group})==1
    assert len({r['head_initial_hash'] for r in group})==1
sources=json.loads((run/'resolved_pretrain_sources.json').read_text(encoding='utf-8'))
for by_arm in sources.values():
    for source in by_arm.values():
        assert hashlib.sha256(Path(source['path']).read_bytes()).hexdigest()==source['sha256']
summary={}
for arm in ('scratch','full','bounded'):
    values=[r['validation']['rmse'] for r in rows if r['arm']==arm]
    assert abs(statistics.mean(values)-results['aggregate'][arm]['mean_rmse'])<1e-12
    summary[arm]={'mean_rmse':statistics.mean(values),'sample_sd':statistics.stdev(values)}
pairs=results['paired_deltas']
acceptance={'completed_jobs':9,'three_independent_pretrain_seeds':[7,1,2],
    'paired_initialization_verified':True,'all_checkpoint_replays_passed':True,
    'source_checkpoint_hashes_verified':True,'run_plan_hash_verified':True,
    'validation_only':True,'test_evaluated':False,'summary':summary,
    'full_beats_scratch_seeds':sum(r['full_minus_scratch']<0 for r in pairs),
    'bounded_beats_scratch_seeds':sum(r['bounded_minus_scratch']<0 for r in pairs),
    'bounded_beats_full_seeds':sum(r['bounded_minus_full']<0 for r in pairs),
    'conclusion':'No downstream benefit demonstrated for these ESOL settings; retain recovery claim separately.'}
(run/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
report=run/'REPORT.md'
report.write_text(report.read_text(encoding='utf-8').replace(
    'Aggregates are provisional until all nine runs complete.',
    'All nine runs completed; paired initialization, checkpoint replay and source hashes passed.'),encoding='utf-8')
print(json.dumps(acceptance))
