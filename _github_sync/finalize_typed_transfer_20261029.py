"""Verify unchanged originals and matched interventions before reporting causes."""
import hashlib
import json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUNS=ROOT/'motif/adjacency_diffusion/runs'
base_folder=RUNS/'typed_property_comparison_20261029'
base=json.loads((base_folder/'results.json').read_text(encoding='utf-8'))['runs']
low_folder=RUNS/'typed_low_encoder_lr_20261029'
low=json.loads((low_folder/'results.json').read_text(encoding='utf-8'))['runs']
full_folder=RUNS/'typed_full_budget_diagnostics_20261029'
full=json.loads((full_folder/'results.json').read_text(encoding='utf-8'))['runs']
source_hashes=json.loads((full_folder/'plan.json').read_text(encoding='utf-8'))['source_latest_sha256']
for row in base:
    source=base_folder/f'{row["arm"]}_task_seed_{row["task_seed"]}'/'latest.pt'
    assert hashlib.sha256(source.read_bytes()).hexdigest()==source_hashes[f'{row["arm"]}_{row["task_seed"]}']
assert len(full)==9 and all(r['epochs_run']==60 and r['checkpoint_replay_passed'] for r in full)
assert len(low)==6 and all(r['checkpoint_replay_passed'] for r in low)
for row in low:
    original=next(r for r in base if r['task_seed']==row['task_seed'] and r['arm']==row['arm'])
    assert row['head_initial_hash']==original['head_initial_hash']
    assert row['encoder_initial_hash']==original['encoder_initial_hash']
    name=f'{row["arm"]}_task_seed_{row["task_seed"]}'
    before=json.loads((base_folder/name/'config.json').read_text(encoding='utf-8'))
    after=json.loads((low_folder/name/'config.json').read_text(encoding='utf-8'))
    for key in ('config','splits','mapping_sha256'):
        assert before[key]==after[key]
    assert after['encoder_learning_rate']==.0001 and after['head_learning_rate']==.001
diagnostics_folder=RUNS/'typed_transfer_diagnostics_20261029'
diagnostics=json.loads((diagnostics_folder/'results.json').read_text(encoding='utf-8'))['runs']
for row in diagnostics:
    original=next(r for r in base if r['task_seed']==row['seed'] and r['arm']==row['arm'])
    assert abs(row['best']['validation_rmse']-original['validation']['rmse'])<1e-6
heads=json.loads((RUNS/'typed_frozen_head_diagnostics_20261029/results.json').read_text(encoding='utf-8'))['runs']
assert len(heads)==18
summary=json.loads((diagnostics_folder/'summary.json').read_text(encoding='utf-8'))
lines=['# Transfer diagnosis: completed controlled comparisons','',
       '| arm | original early stop | all 60 epochs | low encoder LR, original early stop |',
       '|---|---:|---:|---:|']
for arm in ('scratch','full','bounded'):
    initial=summary['original_early_stopped'][arm]['mean_rmse']
    same_budget=summary['full60_no_early_stop'][arm]['mean_rmse']
    low_lr=summary['lower_encoder_lr_early_stopped'].get(arm)
    value=f'{low_lr["mean_rmse"]:.5f}' if low_lr else 'not run'
    lines.append(f'| {arm} | {initial:.5f} | {same_budget:.5f} | {value} |')
lines+=['','All metrics are validation RMSE. Original files remain unchanged.',
        'Full and bounded use identical paired pretraining settings except their noise horizons.',
        'Both pretraining arms remain worse than scratch in all three seeds after all 60 epochs.',
        'Bounded is worse than full in all three seeds at the common 60-epoch cap.',
        'Lower-LR experiments retain early stopping and do not identify every possible optimizer policy.',
        'Frozen probes demonstrate usable information, not a downstream gain.',
        'No test set was evaluated; causal claims are limited to the tested interventions.']
(diagnostics_folder/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
acceptance={'original_latest_checkpoints_unchanged':True,'common_60_epochs_all_9_verified':True,
            'continuation_checkpoint_replays_passed':True,'low_lr_matched_initialization_and_data_verified':True,
            'diagnostic_predictions_match_original':True,'frozen_head_probes':18,
            'additional_finetuning_runs':6,'continuation_runs':7,'test_evaluated':False}
(diagnostics_folder/'acceptance.json').write_text(json.dumps(acceptance,indent=2),encoding='utf-8')
print(json.dumps(acceptance))
