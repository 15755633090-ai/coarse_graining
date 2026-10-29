"""Aggregate diagnostics and controlled follow-ups; do not select a new method."""
import json
from pathlib import Path
import statistics
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'motif/adjacency_diffusion/runs'
base=json.loads((RUNS/'typed_property_comparison_20261029/results.json').read_text(encoding='utf-8'))
diagnostic=json.loads((RUNS/'typed_transfer_diagnostics_20261029/results.json').read_text(encoding='utf-8'))
heads=json.loads((RUNS/'typed_frozen_head_diagnostics_20261029/results.json').read_text(encoding='utf-8'))
for folder in ('typed_low_encoder_lr_20261029','typed_full_budget_diagnostics_20261029'):
    if json.loads((RUNS/folder/'status.json').read_text(encoding='utf-8')).get('state')!='complete':
        raise RuntimeError('Controlled follow-ups are incomplete: '+folder)
low=json.loads((RUNS/'typed_low_encoder_lr_20261029/results.json').read_text(encoding='utf-8'))
full=json.loads((RUNS/'typed_full_budget_diagnostics_20261029/results.json').read_text(encoding='utf-8'))


def aggregate(rows):
    output={}
    for arm in ('scratch','full','bounded'):
        values=[r['validation']['rmse'] for r in rows if r['arm']==arm]
        if values:
            output[arm]={'n':len(values),'mean_rmse':statistics.mean(values),
                         'sample_sd':statistics.stdev(values) if len(values)>1 else None}
    return output


probe_summary={}
for arm in ('scratch','full','bounded'):
    selected=[r for r in diagnostic['runs'] if r['arm']==arm]
    probe_summary[arm]={
        'initial_ridge_validation_rmse':statistics.mean(r['initial_frozen_ridge']['validation_rmse'] for r in selected),
        'frozen_mlp_raw_validation_rmse':statistics.mean(r['validation_rmse'] for r in heads['runs'] if r['arm']==arm and r['input']=='raw'),
        'frozen_mlp_standardized_validation_rmse':statistics.mean(r['validation_rmse'] for r in heads['runs'] if r['arm']==arm and r['input']=='standardized'),
        'best_eval_training_rmse':statistics.mean(r['best']['training_rmse_eval'] for r in selected),
        'initial_feature_variance_trace':statistics.mean(r['initial_geometry']['variance_trace'] for r in selected),
        'initial_effective_rank':statistics.mean(r['initial_geometry']['effective_rank'] for r in selected)}
continuation=[]
for r in full['runs']:
    original=next(x for x in base['runs'] if x['task_seed']==r['task_seed'] and x['arm']==r['arm'])
    continuation.append({'seed':r['task_seed'],'arm':r['arm'],'original_epochs':original['epochs_run'],
        'original_best_rmse':original['validation']['rmse'],'full60_best_rmse':r['validation']['rmse'],
        'change':r['validation']['rmse']-original['validation']['rmse']})
lr_effect=[]
for seed in (0,1,2):
    initial={r['arm']:r['validation']['rmse'] for r in base['runs'] if r['task_seed']==seed}
    changed={r['arm']:r['validation']['rmse'] for r in low['runs'] if r['task_seed']==seed}
    lr_effect.append({'seed':seed,'scratch_low_minus_base':changed['scratch']-initial['scratch'],
                     'bounded_low_minus_base':changed['bounded']-initial['bounded'],
                     'transfer_gap_base':initial['bounded']-initial['scratch'],
                     'transfer_gap_low_encoder_lr':changed['bounded']-changed['scratch']})
result={'original_early_stopped':aggregate(base['runs']),'full60_no_early_stop':aggregate(full['runs']),
        'lower_encoder_lr_early_stopped':aggregate(low['runs']),'probes':probe_summary,
        'continuation_effects':continuation,'encoder_lr_effects':lr_effect,
        'lower_lr_early_stop_is_a_limitation':True,'test_evaluated':False,
        'no_unique_cause_claim_without_support':True}
out=RUNS/'typed_transfer_diagnostics_20261029'
(out/'summary.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps(result))
